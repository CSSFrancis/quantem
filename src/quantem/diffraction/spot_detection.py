"""Template-free diffraction spot detection: a Difference-of-Gaussians band-pass, plus
the pieces it shares with the neural detector (beam-stop masking, disk-mean
intensities, per-pattern packing).

Unlike template matching, the Difference-of-Gaussians detector needs no probe: the
band-pass ``G(sigma1) - G(sigma2)`` suppresses pixel noise below ``sigma1`` and the
smooth diffuse background above ``sigma2`` in one separable real-space pass, so a
faint small spot stands out without a disk-matched template. The threshold is an
absolute signal-to-noise ratio (the response over ``1.4826 * MAD`` of the response),
so it does not depend on the pattern's intensity scale.

A beam stop is a peak-rejection region, never an image edit. Filling or zeroing the
masked pixels puts an intensity step on the mask edge that the band-pass fires on,
so detection runs on the unmodified pattern, the mask is left out of the noise
estimate, and peaks inside it are dropped.

All detectors return per-pattern ``(M, 3)`` arrays of ``[q_row, q_col, intensity]``
in detector pixels, sorted by descending intensity, where the intensity is the mean
of the raw pattern over a small disk around each peak. A single-pixel sample is
dominated by shot noise and by where the subpixel centre landed, which shows up as
banding in an intensity-weighted virtual image.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

from quantem.diffraction.disk_detection import (
    _detect_peaks_batched,
    _refine_parabolic_batched,
    _split_by_counts,
)


def detect_beamstop(
    mean_pattern: np.ndarray, *, fraction: float = 0.15, dilate: int = 5
) -> np.ndarray | None:
    """Find a physical beam stop in a scan-averaged diffraction pattern.

    A beam stop blocks electrons, so in the mean pattern it is a stable, connected,
    low-intensity region. The mask is dilated because the brightest feature in a
    beam-stopped pattern is the halo hugging the stop edge, not the occluded core;
    about 5 pixels keeps peak finders off that rim without losing real spots.

    Parameters
    ----------
    mean_pattern : np.ndarray
        ``(H, W)`` mean diffraction pattern.
    fraction : float, default=0.15
        Pixels below this fraction of the pattern mean count as blocked.
    dilate : int, default=5
        Dilation of the mask in pixels.

    Returns
    -------
    np.ndarray or None
        ``(H, W)`` boolean mask, or ``None`` when no plausible stop is found (the
        threshold catches almost nothing, or almost the whole pattern).
    """
    pattern = np.asarray(mean_pattern, dtype=np.float64)
    if pattern.ndim != 2 or pattern.size == 0:
        return None
    mean_value = float(pattern.mean())
    if mean_value <= 0:
        return None
    mask = pattern < mean_value * fraction
    covered = mask.mean()
    if covered < 0.001 or covered > 0.45:
        return None
    return dilate_mask(mask, dilate)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of a 2D mask by a square of half-width ``radius`` pixels."""
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    return maximum_filter(np.asarray(mask, dtype=np.uint8), size=2 * int(radius) + 1) > 0


def _reflect_index_map(n: int, pad: int, device: torch.device) -> torch.Tensor:
    """Half-sample symmetric reflect indices (``scipy.ndimage`` ``'reflect'``) for an
    axis of length ``n`` padded by ``pad`` on each side; valid for ``pad >= n``."""
    p = torch.arange(-pad, n + pad, device=device)
    period = 2 * n
    p = torch.remainder(p, period)
    return torch.where(p >= n, period - 1 - p, p)


def gaussian_blur(x: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    """Separable Gaussian blur of a ``(B, H, W)`` stack with half-sample reflection.

    Parameters
    ----------
    x : torch.Tensor
        ``(B, H, W)`` stack.
    sigma : float
        Gaussian width in pixels.
    radius : int
        Kernel half-width in pixels; ``0`` returns ``x`` unchanged.

    Returns
    -------
    torch.Tensor
        ``(B, H, W)`` blurred stack.
    """
    if radius < 1 or sigma <= 0:
        return x
    offsets = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    height, width = x.shape[-2:]
    y = x.unsqueeze(1).index_select(2, _reflect_index_map(height, radius, x.device))
    y = F.conv2d(y, kernel.view(1, 1, -1, 1))
    y = y.index_select(3, _reflect_index_map(width, radius, x.device))
    return F.conv2d(y, kernel.view(1, 1, 1, -1)).squeeze(1)


def difference_of_gaussians_snr(
    dps: torch.Tensor, sigma1: float, sigma2: float, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Band-pass response of each pattern, as a signal-to-noise ratio.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` diffraction patterns.
    sigma1, sigma2 : float
        Inner and outer Gaussian widths in pixels (``sigma1 < sigma2``).
    mask : torch.Tensor, optional
        ``(H, W)`` boolean beam-stop mask, excluded from the noise estimate and set to
        zero in the output.

    Returns
    -------
    torch.Tensor
        ``(B, H, W)`` response over ``1.4826 * MAD``.
    """
    batch = dps.shape[0]
    response = gaussian_blur(dps, sigma1, max(1, round(3 * sigma1))) - gaussian_blur(
        dps, sigma2, max(1, round(3 * sigma2))
    )
    if mask is not None:
        statistics = response.masked_fill(mask[None], float("nan")).reshape(batch, -1)
    else:
        statistics = response.reshape(batch, -1)
    median = statistics.nanmedian(dim=1).values.view(batch, 1, 1)
    deviation = (statistics - median.view(batch, 1)).abs().nanmedian(dim=1).values
    scale = 1.4826 * deviation.view(batch, 1, 1)
    # A near-flat band-pass (low-texture pattern) has MAD ~ 0; the std keeps a lone
    # spot on a flat field detectable instead of dividing by zero.
    std = response.reshape(batch, -1).std(dim=1, unbiased=False).view(batch, 1, 1)
    scale = torch.where(scale > 1e-6, scale, std).clamp_min(1e-6)
    snr = (response - median) / scale
    if mask is not None:
        snr = snr.masked_fill(mask[None], 0.0)
    return snr


def detect_disks_dog_batch(
    dps: torch.Tensor,
    *,
    sigma1: float = 0.8,
    sigma2: float = 2.0,
    threshold: float = 10.0,
    min_spacing: float = 5.0,
    edge_boundary: int = 1,
    subpixel: bool = True,
    max_num_peaks: int = 1000,
    mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Detect spots across a stack of patterns with a Difference-of-Gaussians band-pass.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` diffraction patterns.
    sigma1, sigma2 : float, default=0.8, 2.0
        Inner and outer Gaussian widths in pixels. ``sigma1`` should be about the spot
        width; ``sigma2`` a few times larger.
    threshold : float, default=10.0
        Minimum band-pass signal-to-noise ratio. Around 10 balances recall against
        false positives; lower it toward 6 for faint spots.
    min_spacing : float, default=5.0
        Minimum spacing in pixels between kept peaks; fainter close peaks are dropped.
    edge_boundary : int, default=1
        Width in pixels of the border in which peaks are ignored.
    subpixel : bool, default=True
        Refine each peak with a 3-point parabola on the band-pass response.
    max_num_peaks : int, default=1000
        Maximum number of peaks kept per pattern.
    mask : np.ndarray, optional
        ``(H, W)`` boolean beam-stop mask; peaks inside it are dropped.

    Returns
    -------
    list of np.ndarray
        Length-``B`` list of ``(M, 3)`` arrays of ``[q_row, q_col, intensity]``.
    """
    dps = dps.to(torch.float32)
    mask_t = None
    if mask is not None and np.any(mask):
        mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=dps.device)
    snr = difference_of_gaussians_snr(dps, sigma1, sigma2, mask_t)
    # A flat-topped disk's band-pass response is a ring with several local maxima;
    # only the maximum of its whole min_spacing window counts as a candidate.
    window = int(min_spacing)
    if window > 0:
        window_max = F.max_pool2d(snr[:, None], 2 * window + 1, stride=1, padding=window)[:, 0]
        candidates = snr.masked_fill(snr < window_max, float("-inf"))
    else:
        candidates = snr
    peaks, batch_index, _ = _detect_peaks_batched(
        candidates, edge_boundary, float(threshold), float(min_spacing), max_num_peaks
    )
    finite = torch.isfinite(peaks[:, 2])
    peaks, batch_index = peaks[finite], batch_index[finite]
    if subpixel and peaks.shape[0]:
        peaks = _refine_parabolic_batched(snr, peaks, batch_index)
    return pack_with_disk_intensity(dps, peaks, batch_index, max(1.0, float(np.ceil(sigma2))))


def disk_mean_intensity(
    dps: torch.Tensor, peaks: torch.Tensor, batch_index: torch.Tensor, radius: float
) -> torch.Tensor:
    """Mean of each raw pattern over a disk of ``radius`` pixels around each peak.

    Pixels of the disk that fall outside the pattern are left out of the mean.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` raw diffraction patterns.
    peaks : torch.Tensor
        ``(T, >=2)`` peaks with ``[row, col]`` in the first two columns.
    batch_index : torch.Tensor
        ``(T,)`` pattern index of each peak.
    radius : float
        Disk radius in pixels.

    Returns
    -------
    torch.Tensor
        ``(T,)`` disk-mean intensities.
    """
    if peaks.shape[0] == 0:
        return peaks.new_zeros((0,))
    height, width = dps.shape[-2:]
    r = max(1, int(round(float(radius))))
    offset_row, offset_col = torch.meshgrid(
        torch.arange(-r, r + 1, device=dps.device),
        torch.arange(-r, r + 1, device=dps.device),
        indexing="ij",
    )
    inside = offset_row**2 + offset_col**2 <= r * r
    offset_row, offset_col = offset_row[inside], offset_col[inside]
    rows = peaks[:, 0].round().long()[:, None] + offset_row[None]
    cols = peaks[:, 1].round().long()[:, None] + offset_col[None]
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    values = dps[batch_index[:, None], rows.clamp(0, height - 1), cols.clamp(0, width - 1)]
    values = torch.where(valid, values, torch.zeros_like(values))
    return values.sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def drop_masked_peaks(
    peaks: torch.Tensor, batch_index: torch.Tensor, mask: np.ndarray | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop peaks whose nearest pixel lies inside a ``(H, W)`` beam-stop mask."""
    if mask is None or peaks.shape[0] == 0 or not np.any(mask):
        return peaks, batch_index
    mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=peaks.device)
    height, width = mask_t.shape
    rows = peaks[:, 0].round().long().clamp(0, height - 1)
    cols = peaks[:, 1].round().long().clamp(0, width - 1)
    keep = ~mask_t[rows, cols]
    return peaks[keep], batch_index[keep]


def pack_with_disk_intensity(
    dps: torch.Tensor,
    peaks: torch.Tensor,
    batch_index: torch.Tensor,
    radius: float,
    max_num_peaks: int | None = None,
) -> list[np.ndarray]:
    """Replace the score column by the disk-mean intensity and split per pattern.

    Each pattern's peaks are sorted by descending intensity and capped at
    ``max_num_peaks``.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` raw diffraction patterns.
    peaks : torch.Tensor
        ``(T, 3)`` ``[row, col, score]`` peaks pooled over the stack.
    batch_index : torch.Tensor
        ``(T,)`` pattern index of each peak.
    radius : float
        Disk radius in pixels for the intensity.
    max_num_peaks : int, optional
        Maximum number of peaks kept per pattern.

    Returns
    -------
    list of np.ndarray
        Length-``B`` list of ``(M, 3)`` arrays of ``[q_row, q_col, intensity]``.
    """
    batch = dps.shape[0]
    if peaks.shape[0] == 0:
        return [np.empty((0, 3), dtype=np.float64) for _ in range(batch)]
    peaks = peaks.clone()
    peaks[:, 2] = disk_mean_intensity(dps, peaks, batch_index, radius)
    by_intensity = torch.argsort(peaks[:, 2], descending=True)
    order = by_intensity[torch.argsort(batch_index[by_intensity], stable=True)]
    peaks, batch_index = peaks[order], batch_index[order]
    if max_num_peaks is not None:
        counts = torch.bincount(batch_index, minlength=batch)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(peaks.shape[0], device=peaks.device) - starts[batch_index]
        keep = rank < max_num_peaks
        peaks, batch_index = peaks[keep], batch_index[keep]
    return _split_by_counts(peaks, torch.bincount(batch_index, minlength=batch))
