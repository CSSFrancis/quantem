"""Neural diffraction spot detection with :class:`~quantem.core.ml.spot_unet.SpotUNet`.

The pipeline needs no template and almost no parameters:

1. Estimate the disk diameter from the autocorrelation of one pattern (a disk
   correlated with itself gives a central peak about one diameter wide) and resample
   every pattern so its disks land near the size the network was trained on.
2. Normalise: ``log1p``, subtract a wide Gaussian background, then standardise by
   the median and MAD.
3. Run the network and keep heatmap maxima above a confidence threshold, moved by the
   predicted subpixel offset. A network with a disk-coverage head then moves each
   centre to the centroid of its predicted aperture.
4. Map positions back to detector pixels and, by default, re-place each disk on the
   raw pattern at native resolution with the centre networks
   (:mod:`~quantem.diffraction.centre_refinement`).

Every detection is ``[q_row, q_col, intensity, confidence, sigma]``: the mean of the
raw pattern over a disk around it, the heatmap peak (how sure the network is there
is a disk), and the one-standard-deviation uncertainty of its centre in pixels. Sigma
is the centre networks' own estimate when they ran, and otherwise the
counting-noise limit for a spot as wide as the network's response
(:func:`positional_sigma`). The two are never mixed in one pattern: the
counting-noise limit is tens of times smaller, so a disk the networks declined would
otherwise outweigh every disk they placed.

The preprocessing must match how the checkpoints were trained, so its maths is not to
be tuned here. The scale, the suppression window and the background width are
resolved once per stack from its first pattern, because every pattern in a scan has
the same physical disk size.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quantem.core.ml.spot_unet import SpotUNet
from quantem.diffraction import neural_weights
from quantem.diffraction.centre_refinement import (
    CentreRefiner,
    coverage_centroid,
    load_centre_refiner,
    refine_centres,
)
from quantem.diffraction.spot_detection import disk_mean_intensity, drop_masked_peaks

#: Disk diameter in pixels the network was trained on.
CANONICAL_DIAMETER = 9.0
#: Bounds of the resample factor, so a pathological diameter estimate (beam stop,
#: empty or saturated pattern) cannot trigger an extreme resample. Very large disks
#: run at the lower bound, where the size-scaled window and background take over.
SCALE_CLIP = (0.3, 3.0)
DEFAULT_BACKGROUND_SIGMA = 12.0
DEFAULT_THRESHOLD = 0.3
#: Radius (px) of the disk the intensity column averages the raw pattern over.
INTENSITY_RADIUS = 5.0
#: Columns of a neural detection.
NEURAL_PEAK_FIELDS = ("q_row", "q_col", "intensity", "confidence", "sigma")

_MODELS: dict[tuple[str, str], SpotUNet] = {}
_REFINERS: dict[tuple[str, str], CentreRefiner] = {}


def load_spot_unet(
    model: str | Path | None = None,
    *,
    version: str | None = None,
    device: str | torch.device = "cpu",
) -> SpotUNet:
    """Load a trained SpotUNet detector, downloading the weights if needed.

    Parameters
    ----------
    model : str or Path, optional
        A detector name from :data:`~quantem.diffraction.neural_weights.DETECTOR_FILES`
        or a local checkpoint. ``None`` is the default detector.
    version : str, optional
        Dated em-database version to pin.
    device : str or torch.device, default="cpu"
        Device the model is moved to.

    Returns
    -------
    SpotUNet
        The model in evaluation mode, with ``mad_floor`` (floor on the normalisation
        scale, or ``None``) and ``default_threshold`` read from the checkpoint. Loaded
        models are cached per path and device.
    """
    path = neural_weights.detector_path(model, version)
    key = (str(path), str(device))
    if key not in _MODELS:
        # weights_only: a downloaded checkpoint must never unpickle arbitrary objects.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        architecture = {**checkpoint, **checkpoint.get("config", {})}
        network = SpotUNet(
            in_ch=int(architecture.get("in_ch", 1)),
            base=int(architecture["base"]),
            levels=int(architecture.get("levels", 2)),
            mask_head=bool(architecture.get("mask_head", False)),
        )
        network.load_state_dict(checkpoint["state_dict"])
        mad_floor = float(architecture.get("mad_floor", 0.0) or 0.0)
        network.mad_floor = mad_floor if mad_floor > 0 else None
        network.default_threshold = float(architecture.get("threshold", 0.3) or 0.3)
        _MODELS[key] = network.to(device).eval()
    return _MODELS[key]


def load_centre_step(
    step: bool | str | Path | None, entry_name: str, device: str | torch.device = "cpu"
) -> CentreRefiner | None:
    """The network for one centre step, or ``None`` when the step is off.

    Parameters
    ----------
    step : bool, str or Path
        ``False`` for off, ``True`` to download the em-database entry, or a path to a
        local checkpoint.
    entry_name : str
        The em-database entry for ``step=True``.
    device : str or torch.device, default="cpu"
        Device the network is moved to.

    Returns
    -------
    CentreRefiner or None
        The step's refiner. A download that fails warns and returns ``None``, so
        detection still completes with the centres it has.
    """
    if step is False or step is None:
        return None
    try:
        path = neural_weights.entry_path(entry_name) if step is True else Path(step)
    except (ImportError, RuntimeError) as error:
        warnings.warn(
            f"centre network {entry_name} unavailable ({error}); keeping the centres it "
            "would have refined.",
            stacklevel=3,
        )
        return None
    key = (str(path), str(device))
    if key not in _REFINERS:
        _REFINERS[key] = load_centre_refiner(path, device)
    return _REFINERS[key]


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
    from scipy.ndimage import gaussian_filter

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


def _reflect_index_map(n: int, pad: int, device: torch.device) -> torch.Tensor:
    """Half-sample symmetric reflect indices (``scipy.ndimage`` ``'reflect'``)."""
    p = torch.remainder(torch.arange(-pad, n + pad, device=device), 2 * n)
    return torch.where(p >= n, 2 * n - 1 - p, p)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """``scipy.ndimage.gaussian_filter`` of a ``(B, H, W)`` stack (truncate 4, reflect)."""
    radius = int(4.0 * float(sigma) + 0.5)
    if radius < 1:
        return x
    offsets = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    height, width = x.shape[-2:]
    y = x.unsqueeze(1).index_select(2, _reflect_index_map(height, radius, x.device))
    y = F.conv2d(y, kernel.view(1, 1, -1, 1))
    y = y.index_select(3, _reflect_index_map(width, radius, x.device))
    return F.conv2d(y, kernel.view(1, 1, 1, -1)).squeeze(1)


def _network_input(
    dps: torch.Tensor,
    factor: float,
    background_sigma: float,
    levels: int,
    mad_floor: float | None = None,
) -> torch.Tensor:
    """Resample, normalise and reflect-pad a ``(B, H, W)`` stack to ``(B, 1, H', W')``.

    ``mad_floor`` bounds the robust scale from below: on sparse counted patterns the
    MAD is about zero and single counts would otherwise become huge values. A model
    trained with a floor declares it in its checkpoint.
    """
    x = dps.to(torch.float32)
    if factor != 1.0:
        height, width = x.shape[-2:]
        size = (int(round(height * factor)), int(round(width * factor)))
        x = F.interpolate(x[:, None], size=size, mode="bilinear", align_corners=True)[:, 0]
    x = torch.log1p(x.clamp(min=0))
    x = x - _gaussian_blur(x, background_sigma)
    flat = x.reshape(x.shape[0], -1)
    median = _median(flat)
    scale = 1.4826 * (_median((flat - median).abs()) + 1e-6)
    if mad_floor is not None:
        scale = scale.clamp_min(mad_floor)
    x = ((flat - median) / scale).reshape_as(x)
    multiple = 2**levels
    pad_rows = (multiple - x.shape[-2] % multiple) % multiple
    pad_cols = (multiple - x.shape[-1] % multiple) % multiple
    x = x[:, None]
    if pad_rows or pad_cols:
        x = F.pad(x, (0, pad_cols, 0, pad_rows), mode="reflect")
    return x


def _mask_centroid(mask_logits, batch_index, rows, cols, radius):
    """:func:`coverage_centroid` of a mask head's output around each seed."""
    coverage = torch.sigmoid(mask_logits)
    _, _, height, width = coverage.shape
    half = int(np.ceil(1.6 * radius + 3))
    steps = torch.arange(-half, half + 1, device=coverage.device, dtype=torch.float32)
    window_rows = rows.round()[:, None] + steps[None]
    window_cols = cols.round()[:, None] + steps[None]
    inside = ((window_rows >= 0) & (window_rows < height))[:, :, None] & (
        (window_cols >= 0) & (window_cols < width)
    )[:, None, :]
    patch = (
        coverage[
            batch_index[:, None, None],
            0,
            window_rows.long().clamp(0, height - 1)[:, :, None],
            window_cols.long().clamp(0, width - 1)[:, None, :],
        ]
        * inside
    )
    return coverage_centroid(
        patch,
        window_rows[:, :, None],
        window_cols[:, None, :],
        rows.clone(),
        cols.clone(),
        radius,
    )


def _decode(
    outputs: tuple[torch.Tensor, ...],
    threshold: float,
    window: int,
    width_radius: float = 3.0,
    mask_radius: float = 4.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heatmap maxima above ``threshold``, at the pixel plus the predicted offset.

    With a mask head each position moves to its mask centroid, unless the mask there is
    under 15 % of the expected disk area or its centroid lands more than 1.5 px away (a
    neighbour's mask, or none).

    Returns ``(T, 4)`` ``[row, col, confidence, width]`` in working pixels, where width
    is the spread (one standard deviation) of the softmax of the heatmap logits within
    ``width_radius``, and the ``(T,)`` pattern index.
    """
    logits, offset = outputs[0], outputs[1]
    heatmap = torch.sigmoid(logits)
    _, _, height, width = heatmap.shape
    pooled = F.max_pool2d(heatmap, 2 * window + 1, stride=1, padding=window)
    index = ((heatmap == pooled) & (heatmap >= threshold))[:, 0].nonzero()
    if index.numel() == 0:
        return torch.zeros((0, 4), device=heatmap.device), torch.zeros(
            0, dtype=torch.long, device=heatmap.device
        )
    batch_index, rows, cols = index[:, 0], index[:, 1], index[:, 2]
    confidence = heatmap[batch_index, 0, rows, cols]

    half = max(1, int(math.ceil(width_radius)))
    steps = torch.arange(-half, half + 1, device=heatmap.device)
    step_rows, step_cols = torch.meshgrid(steps, steps, indexing="ij")
    inside = (step_rows**2 + step_cols**2) <= width_radius**2 + 1e-6
    grid_rows = (rows[:, None, None] + step_rows).clamp(0, height - 1)
    grid_cols = (cols[:, None, None] + step_cols).clamp(0, width - 1)
    window_logits = logits[batch_index[:, None, None], 0, grid_rows, grid_cols]
    in_frame = (grid_rows == rows[:, None, None] + step_rows) & (
        grid_cols == cols[:, None, None] + step_cols
    )
    window_logits = torch.where(
        inside & in_frame, window_logits, torch.full_like(window_logits, -float("inf"))
    )
    weights = torch.softmax(window_logits.flatten(1), 1).reshape(window_logits.shape)
    mean_row = (weights * grid_rows).sum((1, 2))
    mean_col = (weights * grid_cols).sum((1, 2))
    variance = 0.5 * (
        (weights * (grid_rows - mean_row[:, None, None]) ** 2).sum((1, 2))
        + (weights * (grid_cols - mean_col[:, None, None]) ** 2).sum((1, 2))
    )
    spread = torch.sqrt(variance.clamp_min(0))

    peak_rows = rows.float() + offset[batch_index, 0, rows, cols]
    peak_cols = cols.float() + offset[batch_index, 1, rows, cols]
    if len(outputs) > 2:
        centre_rows, centre_cols, area = _mask_centroid(
            outputs[2], batch_index, peak_rows, peak_cols, mask_radius
        )
        trusted = (torch.hypot(centre_rows - peak_rows, centre_cols - peak_cols) < 1.5) & (
            area > 0.15 * np.pi * mask_radius**2
        )
        peak_rows = torch.where(trusted, centre_rows, peak_rows)
        peak_cols = torch.where(trusted, centre_cols, peak_cols)
    return torch.stack([peak_rows, peak_cols, confidence, spread], 1), batch_index


def _to_pattern_coordinates(
    peaks: torch.Tensor, shape: tuple[int, int], factor: float
) -> torch.Tensor:
    """Map ``[row, col, confidence, width]`` from the resampled working image back.

    The resample aligns the first and last pixel centres, so working pixel ``i`` sits
    at ``i * (n - 1) / (out - 1)`` with ``out = round(n * factor)``, not at
    ``i / factor``; the difference is a uniform scale error of up to ~0.6 %.
    """
    if factor == 1.0 or len(peaks) == 0:
        return peaks
    peaks = peaks.clone()
    for axis, n in enumerate(shape):
        out = int(round(n * factor))
        peaks[:, axis] = peaks[:, axis] * ((n - 1) / (out - 1))
    peaks[:, 3] = peaks[:, 3] / factor
    return peaks


def positional_sigma(width_px, signal, background_per_px, pixel: float = 1.0) -> np.ndarray:
    """One standard deviation of a centre, in pixels, from counting statistics.

    The localisation-precision formula for a spot of standard deviation ``s`` measured
    with ``N`` signal counts over a background of ``b²`` counts per pixel, pixel size
    ``a`` (Thompson, Larson and Webb, Biophys. J. 82, 2775 (2002))::

        sigma² = (s² + a²/12) / N  +  8 π s⁴ b² / (a² N²)

    ``s`` is the width of the network's response to the disk, ``N`` the disk's
    background-subtracted counts and ``b²`` its local background level (Poisson, so the
    variance equals the mean). NaN where the disk has no net signal.
    """
    s = np.maximum(np.asarray(width_px, np.float64), 0.5)
    n = np.asarray(signal, np.float64)
    b2 = np.clip(np.asarray(background_per_px, np.float64), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (s * s + pixel * pixel / 12.0) / n + 8 * np.pi * s**4 * b2 / (
            pixel * pixel * n * n
        )
        sigma = np.sqrt(variance)
    return np.where(n > 0, sigma, np.nan).astype(np.float32)


def _masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``np.median(values[i][mask[i]])`` for every row in one sort (0 for an empty row)."""
    count = mask.sum(1)
    ordered = np.sort(np.where(mask, values, np.inf), axis=1)
    rows = np.arange(len(values))
    low = ordered[rows, np.maximum(count - 1, 0) // 2]
    high = ordered[rows, count // 2 - (count == 0)]
    return np.where(count > 0, 0.5 * (low.astype(np.float64) + high), 0.0)


def disk_counts(
    pattern: np.ndarray, positions: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Background-subtracted counts inside ``radius`` of each ``(row, col)``, and the
    background per pixel (the median of the ring 1-4 px outside the disk)."""
    outer = radius + 4.0
    half = int(np.ceil(outer)) + 1
    padded = np.pad(np.asarray(pattern, np.float32), half, mode="edge")
    rows = np.rint(positions[:, 0]).astype(np.intp)
    cols = np.rint(positions[:, 1]).astype(np.intp)
    steps = np.arange(-half, half + 1)
    grid_rows = np.clip(
        (rows[:, None] + half + steps[None, :])[:, :, None], 0, padded.shape[0] - 1
    )
    grid_cols = np.clip(
        (cols[:, None] + half + steps[None, :])[:, None, :], 0, padded.shape[1] - 1
    )
    crops = padded[grid_rows, grid_cols]
    distance_rows = (rows[:, None] + steps[None, :])[:, :, None] - positions[:, 0, None, None]
    distance_cols = (cols[:, None] + steps[None, :])[:, None, :] - positions[:, 1, None, None]
    distance = np.sqrt(distance_rows**2 + distance_cols**2)
    ring = (distance > radius + 1.0) & (distance <= outer)
    background = _masked_median(
        crops.reshape(len(positions), -1), ring.reshape(len(positions), -1)
    )
    signal = ((crops - background[:, None, None]) * (distance <= radius)).sum((1, 2))
    return signal, background


def counts_radius(pattern: np.ndarray, spot_radius: float | None) -> float:
    """Disk radius (px) for the counts and the centre steps: ``spot_radius`` when given,
    else half the autocorrelation diameter estimate."""
    if spot_radius:
        return float(spot_radius)
    diameter = estimate_disk_diameter(np.asarray(pattern, np.float32))
    if not np.isfinite(diameter):
        return INTENSITY_RADIUS
    return float(np.clip(0.5 * diameter, 2.0, 64.0))


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
    centre_refiner: CentreRefiner | None = None,
    friedel_refiner: CentreRefiner | None = None,
    forward_batch_size: int = 64,
) -> list[np.ndarray]:
    """Detect spots across a stack of patterns with a trained SpotUNet.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` diffraction patterns.
    model : SpotUNet
        Trained detector, e.g. from :func:`load_spot_unet`.
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
    centre_refiner : CentreRefiner, optional
        The refine step (F5), run on every detection.
    friedel_refiner : CentreRefiner, optional
        The Friedel-partner step (P4), run after the refine step.
    forward_batch_size : int, default=64
        Patterns per forward pass; bounds the activation memory, not the result.

    Returns
    -------
    list of np.ndarray
        Length-``B`` list of ``(M, 5)`` arrays of
        ``[q_row, q_col, intensity, confidence, sigma]``, sorted by intensity.
    """
    height, width = dps.shape[-2:]
    patterns = dps.detach().cpu().numpy().astype(np.float32)
    first = patterns[0]
    factor = canonical_scale_factor(spot_diameter or estimate_disk_diameter(first))
    working_diameter = _working_diameter(first, factor, spot_diameter)
    window = max(2, int(round(min_spacing * factor)))
    window, background_sigma = _resolve_window_and_background(
        window, background_sigma, working_diameter
    )
    has_diameter = bool(np.isfinite(working_diameter) and working_diameter > 0)
    width_radius = max(1.5, min(0.5 * working_diameter if has_diameter else window, window))
    mask_radius = 0.5 * (working_diameter if has_diameter else CANONICAL_DIAMETER)
    device = next(model.parameters()).device
    mad_floor = getattr(model, "mad_floor", None)

    per_pattern: list[np.ndarray] = []
    for start in range(0, len(patterns), forward_batch_size):
        chunk = torch.as_tensor(patterns[start : start + forward_batch_size], device=device)
        outputs = model(_network_input(chunk, factor, background_sigma, model.levels, mad_floor))
        found, batch_index = _decode(outputs, threshold, window, width_radius, mask_radius)
        found = _to_pattern_coordinates(found, (height, width), factor)
        # The reflect padding mirrors real spots into the margin; drop those copies.
        inside = (found[:, 0] <= height - 0.5) & (found[:, 1] <= width - 0.5)
        found, batch_index = drop_masked_peaks(found[inside], batch_index[inside], mask)
        found, batch_index = found.cpu().numpy(), batch_index.cpu().numpy()
        per_pattern.extend(found[batch_index == i] for i in range(len(chunk)))

    radius = counts_radius(first, 0.5 * spot_diameter if spot_diameter else None)
    positions = [p[:, :2] for p in per_pattern]
    step_sigmas = None
    for refiner in (centre_refiner, friedel_refiner):
        if refiner is None:
            continue
        positions, sigmas = refine_centres(patterns, positions, radius, refiner)
        if sigmas is not None:
            # A disk this step kept keeps the earlier step's uncertainty.
            step_sigmas = (
                sigmas
                if step_sigmas is None
                else [np.where(np.isnan(new), old, new) for old, new in zip(step_sigmas, sigmas)]
            )

    results = []
    for i, (found, position) in enumerate(zip(per_pattern, positions)):
        if len(found) == 0:
            results.append(np.empty((0, len(NEURAL_PEAK_FIELDS)), dtype=np.float64))
            continue
        position = np.asarray(position, np.float64)
        intensity = disk_mean_intensity(
            torch.as_tensor(patterns[i][None]),
            torch.as_tensor(position),
            torch.zeros(len(position), dtype=torch.long),
            INTENSITY_RADIUS,
        ).numpy()
        if step_sigmas is not None:
            sigma = step_sigmas[i]
        else:
            sigma = positional_sigma(found[:, 3], *disk_counts(patterns[i], position, radius))
        rows = np.column_stack([position, intensity, found[:, 2], sigma]).astype(np.float64)
        results.append(rows[np.argsort(-rows[:, 2], kind="stable")][:max_num_peaks])
    return results


@torch.no_grad()
def neural_heatmap(
    dps: torch.Tensor,
    model: SpotUNet,
    *,
    spot_diameter: float | None = None,
    background_sigma: float | None = None,
) -> torch.Tensor:
    """The network's spot confidence for each pattern, on the detector pixel grid.

    For display: the map :func:`detect_disks_neural_batch` thresholds, resampled back
    from the network's working scale.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` diffraction patterns.
    model : SpotUNet
        Trained model.
    spot_diameter : float, optional
        Disk diameter in pixels, overriding the autocorrelation estimate.
    background_sigma : float, optional
        Width in pixels of the background removed before the network.

    Returns
    -------
    torch.Tensor
        ``(B, H, W)`` confidences in ``[0, 1]``.
    """
    first = dps[0].detach().cpu().numpy()
    factor = canonical_scale_factor(spot_diameter or estimate_disk_diameter(first))
    _, background_sigma = _resolve_window_and_background(
        0, background_sigma, _working_diameter(first, factor, spot_diameter)
    )
    device = next(model.parameters()).device
    network_input = _network_input(
        dps.to(device), factor, background_sigma, model.levels, getattr(model, "mad_floor", None)
    )
    heatmap = model(network_input)[0]
    height, width = dps.shape[-2:]
    working = (int(round(height * factor)), int(round(width * factor)))
    heatmap = torch.sigmoid(heatmap[:, :, : working[0], : working[1]])
    return F.interpolate(heatmap, size=(height, width), mode="bilinear", align_corners=True)[:, 0]


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
    threshold is lowered below the model's default so real faint peaks survive.

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
    mad_floor = getattr(model, "mad_floor", None)
    best_sigmas, best_confidences = [], []
    for pattern in dps.to(device):
        confidences = []
        for sigma in candidates:
            network_input = _network_input(pattern[None], factor, sigma, model.levels, mad_floor)
            confidences.append(float(torch.sigmoid(model(network_input)[0]).max()))
        best = int(np.argmax(confidences))
        best_sigmas.append(candidates[best])
        best_confidences.append(confidences[best])

    confidence = float(np.median(best_confidences))
    default = float(getattr(model, "default_threshold", DEFAULT_THRESHOLD))
    threshold = default
    if tune_threshold and confidence < 0.45:
        threshold = float(np.clip(0.5 * confidence, 0.1, default))
    return {
        "background_sigma": float(np.median(best_sigmas)),
        "threshold": threshold,
        "scale_factor": factor,
        "confidence": confidence,
    }
