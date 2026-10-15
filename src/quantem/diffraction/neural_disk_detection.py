"""Neural diffraction spot detection with :class:`~quantem.core.ml.spot_unet.SpotUNet`.

The pipeline needs no template and almost no parameters:

1. Estimate the disk diameter from the autocorrelation of one pattern (a disk
   correlated with itself gives a central peak about one diameter wide) and resample
   every pattern so its disks land near the size the network was trained on.
2. Normalise: ``log1p``, subtract a wide Gaussian background, then standardise by
   the median and MAD.
3. Run the network and keep heatmap maxima above a confidence threshold, moved by the
   predicted subpixel offset; map positions back to detector pixels.

The preprocessing must match how the checkpoints were trained, so its maths is not to
be tuned here. The scale, the suppression window and the background width are
resolved once per stack from its first pattern, because every pattern in a scan has
the same physical disk size.

The trained weights are not part of quantem. They are downloaded on first use from
em-database (entry ``SpotUNetDiffraction``) and carry their own licence, which is
printed the first time they are loaded; pass ``model_path`` to use a local checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from quantem.core.ml.spot_unet import SpotUNet
from quantem.diffraction.spot_detection import (
    drop_masked_peaks,
    gaussian_blur,
    pack_with_disk_intensity,
)

#: Disk diameter in pixels the network was trained on.
CANONICAL_DIAMETER = 9.0
#: Bounds of the resample factor, so a pathological diameter estimate (beam stop,
#: empty or saturated pattern) cannot trigger an extreme resample. Very large disks
#: run at the lower bound, where the size-scaled window and background take over.
SCALE_CLIP = (0.3, 3.0)
DEFAULT_BACKGROUND_SIGMA = 12.0
DEFAULT_THRESHOLD = 0.3
EMDATABASE_ENTRY = "SpotUNetDiffraction"

_MODELS: dict[tuple[str, str], SpotUNet] = {}
_LICENSE_SHOWN = False


def load_spot_unet(
    model_path: str | Path | None = None,
    *,
    version: str | None = None,
    device: str | torch.device = "cpu",
) -> SpotUNet:
    """Load a trained SpotUNet, downloading the weights from em-database if needed.

    Parameters
    ----------
    model_path : str or Path, optional
        Local checkpoint. ``None`` (default) downloads the ``SpotUNetDiffraction``
        weights from em-database.
    version : str, optional
        Dated em-database version to pin, e.g. ``"260716"``. ``None`` fetches the
        latest weights.
    device : str or torch.device, default="cpu"
        Device the model is moved to.

    Returns
    -------
    SpotUNet
        The model in evaluation mode. Loaded models are cached per path and device.
    """
    path = Path(model_path) if model_path is not None else _download_weights(version)
    key = (str(path), str(device))
    if key not in _MODELS:
        # weights_only: a downloaded checkpoint must never unpickle arbitrary objects.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        architecture = checkpoint.get("config", checkpoint)
        model = SpotUNet(
            in_ch=int(architecture.get("in_ch", 1)),
            base=int(architecture["base"]),
            levels=int(architecture.get("levels", 2)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        _MODELS[key] = model.to(device).eval()
    return _MODELS[key]


def _download_weights(version: str | None) -> Path:
    global _LICENSE_SHOWN
    try:
        from emdatabase import data
    except ImportError as error:
        raise ImportError(
            "The neural detector downloads its weights with em-database "
            "(pip install em-database); pass model_path= to use a local checkpoint."
        ) from error
    entry_class = getattr(data, EMDATABASE_ENTRY, None)
    if entry_class is None:
        raise RuntimeError(
            f"This em-database release has no {EMDATABASE_ENTRY!r} entry yet; upgrade "
            "em-database, or pass model_path= to use a local checkpoint."
        )
    entry = entry_class()
    if not _LICENSE_SHOWN:
        print(
            f"SpotUNet weights from em-database ({EMDATABASE_ENTRY}) are licensed "
            f"{entry.metadata.license}: free for research and other non-commercial use; "
            "they may not be redistributed in, or used to provide, a product or service "
            "that is sold. See the em-database entry for the full terms."
        )
        _LICENSE_SHOWN = True
    return Path(entry.download(version=version, background=False, progressbar=False))


def estimate_disk_diameter(pattern: np.ndarray, highpass_sigma: float = 20.0) -> float:
    """Disk diameter in pixels from the width of the autocorrelation's central peak.

    The pattern is high-passed first to remove the smooth background and tame the
    bright central beam.

    Parameters
    ----------
    pattern : np.ndarray
        ``(H, W)`` diffraction pattern.
    highpass_sigma : float, default=20.0
        Width in pixels of the background removed before the autocorrelation.

    Returns
    -------
    float
        Estimated disk diameter in pixels.
    """
    f = np.clip(np.asarray(pattern, dtype=np.float64), 0, None)
    f = np.clip(f - gaussian_filter(f, highpass_sigma), 0, None)
    if f.max() <= 0:
        return CANONICAL_DIAMETER
    spectrum = np.fft.fft2(f)
    autocorrelation = np.fft.fftshift(np.real(np.fft.ifft2(spectrum * np.conj(spectrum))))
    height, width = autocorrelation.shape
    centre_row, centre_col = height // 2, width // 2
    peak = autocorrelation[centre_row, centre_col]
    if peak <= 0:
        return CANONICAL_DIAMETER

    def half_maximum_width(profile: np.ndarray, centre: int) -> int:
        left = right = centre
        while left > 0 and profile[left] > 0.5:
            left -= 1
        while right < profile.size - 1 and profile[right] > 0.5:
            right += 1
        return max(right - left, 1)

    # Average the two widths rather than the two profiles, which only have the same
    # length on a square detector.
    return float(
        0.5
        * (
            half_maximum_width(autocorrelation[centre_row] / peak, centre_col)
            + half_maximum_width(autocorrelation[:, centre_col] / peak, centre_row)
        )
    )


def canonical_scale_factor(diameter: float) -> float:
    """Resample factor that brings disks of ``diameter`` pixels to the trained size."""
    factor = float(np.clip(CANONICAL_DIAMETER / diameter, *SCALE_CLIP))
    return 1.0 if abs(factor - 1.0) < 0.05 else factor


def _working_diameter(pattern: np.ndarray, factor: float, spot_diameter: float | None) -> float:
    """Disk diameter after resampling.

    The autocorrelation estimate saturates on big disks, so it is iterated with a
    size-matched high-pass and corrected for its measured ~0.83 undershoot.
    """
    if spot_diameter:
        return float(spot_diameter) * factor
    diameter = estimate_disk_diameter(pattern)
    if np.isfinite(diameter) and diameter > 20:
        for _ in range(3):
            refined = estimate_disk_diameter(pattern, highpass_sigma=max(20.0, diameter))
            if abs(refined - diameter) < 1:
                diameter = refined
                break
            diameter = max(diameter, refined)
        diameter = diameter / 0.83
    return diameter * factor


def _resolve_window_and_background(
    min_spacing: int, background_sigma: float | None, working_diameter: float
) -> tuple[int, float]:
    """Suppression window and background width for the working disk size.

    Disks at the trained size keep the defaults. Big disks run near native size need
    both scaled: a small window fires a cloud of detections across one broad disk,
    and a 12 pixel background carves its flat interior into a ring.
    """
    big = bool(np.isfinite(working_diameter) and working_diameter > 20)
    if big:
        min_spacing = max(min_spacing, int(round(working_diameter * 0.55)))
    if background_sigma is None:
        background_sigma = (
            max(DEFAULT_BACKGROUND_SIGMA, 0.6 * working_diameter)
            if big
            else DEFAULT_BACKGROUND_SIGMA
        )
    return min_spacing, float(background_sigma)


def _median(flat: torch.Tensor) -> torch.Tensor:
    """Per-row median of ``(B, N)``, averaging the two middle values like ``np.median``."""
    n = flat.shape[1]
    ordered, _ = flat.sort(dim=1)
    if n % 2:
        return ordered[:, n // 2 : n // 2 + 1]
    return 0.5 * (ordered[:, n // 2 - 1 : n // 2] + ordered[:, n // 2 : n // 2 + 1])


def _network_input(
    dps: torch.Tensor, factor: float, background_sigma: float, levels: int
) -> torch.Tensor:
    """Resample, normalise and reflect-pad a ``(B, H, W)`` stack to ``(B, 1, H', W')``."""
    x = dps.to(torch.float32)
    if factor != 1.0:
        height, width = x.shape[-2:]
        size = (int(round(height * factor)), int(round(width * factor)))
        x = F.interpolate(x[:, None], size=size, mode="bilinear", align_corners=True)[:, 0]
    x = torch.log1p(x.clamp(min=0))
    radius = int(4.0 * background_sigma + 0.5)
    x = x - gaussian_blur(x, background_sigma, radius)
    flat = x.reshape(x.shape[0], -1)
    median = _median(flat)
    deviation = _median((flat - median).abs()) + 1e-6
    x = ((flat - median) / (1.4826 * deviation)).reshape_as(x)
    multiple = 2**levels
    pad_rows = (multiple - x.shape[-2] % multiple) % multiple
    pad_cols = (multiple - x.shape[-1] % multiple) % multiple
    x = x[:, None]
    if pad_rows or pad_cols:
        x = F.pad(x, (0, pad_cols, 0, pad_rows), mode="reflect")
    return x


def _decode(
    heatmap_logits: torch.Tensor, offset: torch.Tensor, threshold: float, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heatmap maxima above ``threshold`` moved by the predicted subpixel offset.

    Returns ``(T, 3)`` ``[row, col, confidence]`` peaks and their ``(T,)`` pattern index.
    """
    heatmap = torch.sigmoid(heatmap_logits)
    pooled = F.max_pool2d(heatmap, 2 * window + 1, stride=1, padding=window)
    index = ((heatmap == pooled) & (heatmap >= threshold))[:, 0].nonzero()
    batch_index, rows, cols = index[:, 0], index[:, 1], index[:, 2]
    peaks = torch.stack(
        (
            rows.to(torch.float32) + offset[batch_index, 0, rows, cols],
            cols.to(torch.float32) + offset[batch_index, 1, rows, cols],
            heatmap[batch_index, 0, rows, cols],
        ),
        dim=1,
    )
    return peaks, batch_index


@torch.no_grad()
def detect_disks_neural_batch(
    dps: torch.Tensor,
    model: SpotUNet,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_spacing: int = 4,
    spot_diameter: float | None = None,
    background_sigma: float | None = None,
    max_num_peaks: int = 1000,
    mask: np.ndarray | None = None,
    forward_batch_size: int = 64,
) -> list[np.ndarray]:
    """Detect spots across a stack of patterns with a trained SpotUNet.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` diffraction patterns on the model's device.
    model : SpotUNet
        Trained model, e.g. from :func:`load_spot_unet`.
    threshold : float, default=0.3
        Minimum heatmap confidence.
    min_spacing : int, default=4
        Half-width in pixels of the suppression window, at the original scale.
    spot_diameter : float, optional
        Disk diameter in pixels, overriding the autocorrelation estimate when it
        gets the size wrong.
    background_sigma : float, optional
        Width in pixels of the background removed before the network (see
        :func:`calibrate_neural`). ``None`` picks it from the disk size.
    max_num_peaks : int, default=1000
        Maximum number of peaks kept per pattern.
    mask : np.ndarray, optional
        ``(H, W)`` boolean beam-stop mask; peaks inside it are dropped.
    forward_batch_size : int, default=64
        Patterns per forward pass; bounds the activation memory, not the result.

    Returns
    -------
    list of np.ndarray
        Length-``B`` list of ``(M, 3)`` arrays of ``[q_row, q_col, intensity]``.
    """
    first = dps[0].detach().cpu().numpy()
    diameter = spot_diameter or estimate_disk_diameter(first)
    factor = canonical_scale_factor(diameter)
    window = max(2, int(round(min_spacing * factor)))
    window, background_sigma = _resolve_window_and_background(
        window, background_sigma, _working_diameter(first, factor, spot_diameter)
    )
    device = next(model.parameters()).device

    all_peaks, all_index = [], []
    for start in range(0, dps.shape[0], forward_batch_size):
        chunk = dps[start : start + forward_batch_size].to(device)
        heatmap, offset = model(_network_input(chunk, factor, background_sigma, model.levels))
        peaks, batch_index = _decode(heatmap, offset, threshold, window)
        all_peaks.append(peaks)
        all_index.append(batch_index + start)
    peaks = torch.cat(all_peaks)
    batch_index = torch.cat(all_index)
    peaks[:, :2] = peaks[:, :2] / factor
    # The reflect padding mirrors real spots into the margin; drop those copies.
    height, width = dps.shape[-2:]
    inside = (peaks[:, 0] <= height - 0.5) & (peaks[:, 1] <= width - 0.5)
    peaks, batch_index = peaks[inside], batch_index[inside]
    peaks, batch_index = drop_masked_peaks(peaks, batch_index, mask)
    return pack_with_disk_intensity(
        dps.to(device, torch.float32), peaks, batch_index, max(1.0, diameter / 2), max_num_peaks
    )


@torch.no_grad()
def calibrate_neural(
    dps: torch.Tensor,
    model: SpotUNet,
    *,
    spot_diameter: float | None = None,
    tune_threshold: bool = True,
) -> dict[str, float]:
    """Pick the background width (and, for faint data, the threshold) on a few patterns.

    The network's own peak confidence has a clean maximum against the background
    width, so the width that maximises it is chosen per pattern and the median taken.
    When even the best confidence is weak (faint spots, amorphous or in-SEM data) the
    threshold is lowered so real faint peaks survive.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` representative patterns spread across the scan.
    model : SpotUNet
        Trained model.
    spot_diameter : float, optional
        Disk diameter in pixels, overriding the autocorrelation estimate.
    tune_threshold : bool, default=True
        Lower the threshold for faint data.

    Returns
    -------
    dict
        ``background_sigma``, ``threshold``, ``scale_factor`` and ``confidence``, to
        pass on to :func:`detect_disks_neural_batch`.
    """
    first = dps[0].detach().cpu().numpy()
    factor = canonical_scale_factor(spot_diameter or estimate_disk_diameter(first))
    candidates = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    working_diameter = _working_diameter(first, factor, spot_diameter)
    if np.isfinite(working_diameter) and working_diameter > 20:
        candidates += [round(ratio * working_diameter, 1) for ratio in (0.5, 0.6, 0.7)]

    device = next(model.parameters()).device
    best_sigmas, best_confidences = [], []
    for pattern in dps.to(device):
        confidences = []
        for sigma in candidates:
            heatmap, _ = model(_network_input(pattern[None], factor, sigma, model.levels))
            confidences.append(float(torch.sigmoid(heatmap).max()))
        best = int(np.argmax(confidences))
        best_sigmas.append(candidates[best])
        best_confidences.append(confidences[best])

    confidence = float(np.median(best_confidences))
    threshold = DEFAULT_THRESHOLD
    if tune_threshold and confidence < 0.45:
        threshold = float(np.clip(0.5 * confidence, 0.1, 0.3))
    return {
        "background_sigma": float(np.median(best_sigmas)),
        "threshold": threshold,
        "scale_factor": factor,
        "confidence": confidence,
    }
