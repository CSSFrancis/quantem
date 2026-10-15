"""The centre steps: re-place each detected disk on the raw pattern at native resolution.

The neural detector finds disks on a rescaled, locally normalised working image, so
its centres are only as good as that image. Two steps look again at the raw counts:

1. **Refine** (the centre network F5): each disk's own window, from the detector's
   centre.
2. **Friedel partner** (P4), on top of the refined centres: each disk's window
   together with the window at its Friedel mirror point ``2 x beam - p``, which shows
   the other dynamical fill of the same reflection. It works on spots of 6-16 px
   radius and leaves others where the refine step put them.

Each step resamples a window around every disk so the disk spans ``crop_radius``
window pixels, normalises it (subtract the mean of an outer annulus, divide by the
disk's contrast over it), segments the disk with a
:class:`~quantem.core.ml.spot_centre_net.SpotCentreNet` and takes the coverage
centroid. Disks the first pass moved by more than ``recrop_over x R`` are re-cut at
the new centre and segmented again.

A step declines a disk, keeping its incoming centre, when it would move it by half a
spot radius or more, when the coverage is under a fifth of the expected disk, or when
the predicted sigma is ``max_sigma_fraction`` of the spot radius or more: a centre
that wandered onto a neighbour is worse than the one it replaced. Outside the
network's ``min_spot_radius``-``max_spot_radius`` the step is skipped altogether. The
predicted sigma ranks centres well but its absolute scale is uncertain to about 2x,
so it is for weighting or ranking vectors, not a calibrated error bar.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quantem.core.ml.spot_centre_net import SpotCentreNet, input_channels

#: The largest move a step accepts, as a fraction of the spot radius.
MAX_SHIFT_FRACTION = 0.5
#: Below this spot radius (px) the window holds too few pixels to beat the detector.
DEFAULT_MIN_SPOT_RADIUS = 5.0
#: Windows refined per batch, bounding the window stack whatever the scan size.
DEFAULT_BATCH = 4096
#: Windows per forward, bounding the network's activations.
FORWARD_BATCH = 1024


def coverage_centroid(
    coverage: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    centre_row: torch.Tensor,
    centre_col: torch.Tensor,
    radius: float,
    iterations: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centroid of a disk-coverage map (0..1) around each starting centre.

    The window is a soft disk with a one-pixel ramp, so the centroid moves continuously
    with the start: the first pass reaches ``1.3 * radius + 1.5``, the next the radius
    the coverage itself implies (area / pi) plus 2, centred on the previous centroid.
    A window with no coverage keeps its centre.

    Parameters
    ----------
    coverage : torch.Tensor
        ``(M, S, S)`` coverage.
    rows, columns : torch.Tensor
        Pixel coordinates, broadcastable to ``(M, S, 1)`` and ``(M, 1, S)``.
    centre_row, centre_col : torch.Tensor
        ``(M,)`` starting centres in the same coordinates.
    radius : float
        Expected disk radius.
    iterations : int, default=2
        Number of passes.

    Returns
    -------
    tuple of torch.Tensor
        The centroid rows, columns and covered areas, each ``(M,)``.
    """
    window = torch.full_like(centre_row, 1.3 * radius + 1.5)
    area = torch.zeros_like(centre_row)
    for _ in range(iterations):
        distance = torch.sqrt(
            (rows - centre_row[:, None, None]) ** 2 + (columns - centre_col[:, None, None]) ** 2
        )
        weight = (window[:, None, None] - distance + 0.5).clamp(0, 1) * coverage
        area = weight.sum((1, 2))
        found = area > 1e-3
        safe_area = area.clamp_min(1e-6)
        centre_row = torch.where(found, (weight * rows).sum((1, 2)) / safe_area, centre_row)
        centre_col = torch.where(found, (weight * columns).sum((1, 2)) / safe_area, centre_col)
        window = torch.sqrt(area / math.pi) + 2.0
    return centre_row, centre_col, area


class CentreRefiner:
    """A trained :class:`SpotCentreNet` and the window contract it was trained with.

    Parameters
    ----------
    network : SpotCentreNet
        The network, in evaluation mode.
    crop_radius : float, default=10.0
        Window pixels a spot radius is resampled to.
    crop_half : int, default=16
        Half-width of the resampled window.
    annulus : tuple of float, default=(0.78, 1.0)
        Background annulus, as fractions of ``crop_half``.
    passes : int, default=2
        Maximum number of passes.
    recrop_over : float, default=0.15
        Re-cut disks the first pass moved by more than this fraction of the radius.
    max_sigma_fraction : float, default=0.25
        Decline disks whose sigma is at least this fraction of the radius.
    sigma_scale : float, default=1.0
        Calibration of the sigma head.
    min_spot_radius, max_spot_radius : float
        Spot radii (px) outside which the step keeps the incoming centres.
    """

    def __init__(
        self,
        network: SpotCentreNet,
        *,
        crop_radius: float = 10.0,
        crop_half: int = 16,
        annulus: tuple[float, float] = (0.78, 1.0),
        passes: int = 2,
        recrop_over: float = 0.15,
        max_sigma_fraction: float = 0.25,
        sigma_scale: float = 1.0,
        min_spot_radius: float = DEFAULT_MIN_SPOT_RADIUS,
        max_spot_radius: float = math.inf,
    ):
        self.network = network
        self.crop_radius = float(crop_radius)
        self.crop_half = int(crop_half)
        self.annulus = (float(annulus[0]), float(annulus[1]))
        self.passes = max(1, int(passes))
        self.recrop_over = float(recrop_over)
        self.max_sigma_fraction = float(max_sigma_fraction)
        self.sigma_scale = float(sigma_scale)
        self.min_spot_radius = float(min_spot_radius)
        self.max_spot_radius = float(max_spot_radius)

    @property
    def needs_partner(self) -> bool:
        """The network also reads each disk's Friedel-mirror window."""
        return int(self.network.encoders[0][0].weight.shape[1]) == 2

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    def crop_half_width(self, spot_radius: float) -> int:
        """Half-width (raw px) of the window cut around each disk."""
        # The resampled grid reaches crop_half / crop_radius radii from the centre,
        # which sits up to half a pixel off the window centre and, on a re-crop pass,
        # may have moved by up to the largest accepted shift. One more pixel keeps the
        # bilinear samples inside the window.
        reach = self.crop_half / self.crop_radius * float(spot_radius)
        if self.passes > 1:
            reach += MAX_SHIFT_FRACTION * float(spot_radius)
        return int(math.ceil(reach + 0.5)) + 1

    def _offsets(self) -> torch.Tensor:
        return torch.arange(
            -self.crop_half, self.crop_half + 1, device=self.device, dtype=torch.float32
        )

    def _resample(self, crops: torch.Tensor, centres: torch.Tensor, step: float) -> torch.Tensor:
        size = crops.shape[-1]
        offsets = self._offsets()
        rows = centres[:, 0, None] + offsets[None] * step
        columns = centres[:, 1, None] + offsets[None] * step
        samples = rows.shape[1]
        grid = torch.stack(
            [
                (columns[:, None, :].expand(-1, samples, -1) / (size - 1)) * 2 - 1,
                (rows[:, :, None].expand(-1, -1, samples) / (size - 1)) * 2 - 1,
            ],
            -1,
        )
        return F.grid_sample(
            crops[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

    def _normalise(self, sampled: torch.Tensor) -> torch.Tensor:
        count = sampled.shape[0]
        offsets = self._offsets()
        distance = torch.sqrt(offsets[:, None] ** 2 + offsets[None] ** 2).reshape(-1)
        inner, outer = self.annulus
        ring = (distance > inner * self.crop_half) & (distance <= outer * self.crop_half)
        disk = distance <= self.crop_radius
        flat = sampled.reshape(count, -1)
        background = flat[:, ring].mean(1)
        level = (flat[:, disk].mean(1) - background).abs().clamp_min(1e-3)
        return ((flat - background[:, None]) / level[:, None]).reshape_as(sampled).clamp(-4, 8)

    def _pass(self, crops, centre, step, partner=None):
        """One pass at ``centre``: the new centre, coverage area and sigma (raw px)."""
        sampled = self._normalise(self._resample(crops, centre, step))
        if partner is not None:
            partner_crops, partner_centre = partner
            mirror = self._normalise(self._resample(partner_crops, partner_centre, step))
            sampled = torch.cat([sampled, mirror.flip(-1, -2)], 1)
        logits, log_sigmas = [], []
        for start in range(0, len(sampled), FORWARD_BATCH):
            output, log_sigma = self.network(sampled[start : start + FORWARD_BATCH])
            logits.append(output)
            log_sigmas.append(log_sigma)
        logits = torch.cat(logits)
        offsets = self._offsets()
        start = torch.zeros(logits.shape[0], device=logits.device)
        offset_row, offset_col, area = coverage_centroid(
            torch.sigmoid(logits[:, 0]),
            offsets[None, :, None],
            offsets[None, None, :],
            start,
            start.clone(),
            self.crop_radius,
        )
        offset = torch.stack([offset_row, offset_col], 1)
        sigma = torch.exp(torch.cat(log_sigmas)) * step * self.sigma_scale
        return centre + offset * step, area, sigma

    @torch.no_grad()
    def refine(
        self,
        crops: np.ndarray,
        centres: np.ndarray,
        spot_radius: float,
        partner_crops: np.ndarray | None = None,
        partner_centres: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine disks given their raw windows.

        Parameters
        ----------
        crops : np.ndarray
            ``(M, S, S)`` raw windows.
        centres : np.ndarray
            ``(M, 2)`` incoming centres in window pixels.
        spot_radius : float
            Disk radius in raw pixels.
        partner_crops, partner_centres : np.ndarray, optional
            The Friedel-mirror windows and mirror points, for a two-channel network.

        Returns
        -------
        tuple of np.ndarray
            ``(M, 2)`` refined centres in window pixels and ``(M,)`` sigmas, both NaN
            for a declined disk.
        """
        if len(crops) == 0:
            return np.zeros((0, 2)), np.zeros(0, np.float32)
        if self.needs_partner and partner_crops is None:
            raise ValueError("this centre network needs each disk's Friedel-mirror window")
        radius = float(spot_radius)
        step = radius / self.crop_radius
        device = self.device
        crop_tensor = torch.as_tensor(np.asarray(crops, np.float32), device=device)
        seeds = torch.as_tensor(np.asarray(centres, np.float32), device=device)
        partner = None
        if self.needs_partner:
            partner_tensor = torch.as_tensor(np.asarray(partner_crops, np.float32), device=device)
            mirror_seeds = torch.as_tensor(np.asarray(partner_centres, np.float32), device=device)
            partner = (partner_tensor, mirror_seeds)
        centre, area, sigma = self._pass(crop_tensor, seeds, step, partner)
        for _ in range(1, self.passes):
            again = (centre - seeds).norm(dim=1) > self.recrop_over * radius
            if not again.any():
                break
            # a partner window follows the mirror of the moved centre
            moved_partner = (
                None
                if partner is None
                else (
                    partner_tensor[again],
                    mirror_seeds[again] - (centre[again] - seeds[again]),
                )
            )
            moved, moved_area, moved_sigma = self._pass(
                crop_tensor[again], centre[again], step, moved_partner
            )
            centre[again], area[again], sigma[again] = moved, moved_area, moved_sigma
        declined = (
            ((centre - seeds).norm(dim=1) >= MAX_SHIFT_FRACTION * radius)
            | (area <= 0.2 * math.pi * self.crop_radius**2)
            | (sigma >= self.max_sigma_fraction * radius)
        )
        centre[declined] = float("nan")
        sigma[declined] = float("nan")
        return centre.double().cpu().numpy(), sigma.float().cpu().numpy()


def load_centre_refiner(path: str | Path, device: str | torch.device = "cpu") -> CentreRefiner:
    """A :class:`CentreRefiner` from a checkpoint.

    The checkpoint holds ``state_dict`` plus the scalars ``base``, ``levels``,
    ``depthwise``, ``sigma_head``, ``crop_radius``, ``crop_half`` and optionally
    ``sigma_scale``, ``min_spot_radius`` and ``max_spot_radius``.
    """
    # weights_only: a downloaded checkpoint must never unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network = SpotCentreNet(
        int(checkpoint.get("base", 8)),
        int(checkpoint.get("levels", 2)),
        bool(checkpoint.get("depthwise", 0)),
        bool(checkpoint.get("sigma_head", 1)),
        channels=input_channels(checkpoint["state_dict"]),
    )
    if network.sigma_head is None:
        raise ValueError("a centre network needs a sigma head")
    network.load_state_dict(checkpoint["state_dict"])
    return CentreRefiner(
        network.to(device).eval(),
        crop_radius=float(checkpoint.get("crop_radius", 10.0)),
        crop_half=int(checkpoint.get("crop_half", 16)),
        sigma_scale=float(checkpoint.get("sigma_scale", 1.0)),
        min_spot_radius=float(checkpoint.get("min_spot_radius", DEFAULT_MIN_SPOT_RADIUS)),
        max_spot_radius=float(checkpoint.get("max_spot_radius", math.inf)),
    )


def extract_crops(
    pattern: np.ndarray, centres: np.ndarray, half_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windows of ``2 * half_width + 1`` px around each centre's nearest pixel.

    Pixels outside the pattern are zero. Returns the ``(M, S, S)`` float32 windows
    and the centres in window pixels.
    """
    pattern = np.asarray(pattern, np.float32)
    centres = np.asarray(centres, np.float64).reshape(-1, 2)
    size = 2 * half_width + 1
    if len(centres) == 0:
        return np.zeros((0, size, size), np.float32), np.zeros((0, 2))
    height, width = pattern.shape
    nearest = np.rint(centres).astype(np.intp)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, height - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, width - 1)
    offsets = np.arange(-half_width, half_width + 1)
    rows = nearest[:, 0, None] + offsets
    columns = nearest[:, 1, None] + offsets
    inside = ((rows >= 0) & (rows < height))[:, :, None] & ((columns >= 0) & (columns < width))[
        :, None, :
    ]
    crops = pattern[
        np.clip(rows, 0, height - 1)[:, :, None], np.clip(columns, 0, width - 1)[:, None, :]
    ]
    return np.where(inside, crops, np.float32(0)), centres - (nearest - half_width)


def frame_beams(positions: list[np.ndarray], frame_shape, spot_radius: float) -> np.ndarray:
    """Each pattern's centre of Friedel symmetry, from its ``(n, 2)`` disk centres.

    Start at the disk nearest the mean of all disks, pair each disk with the one
    nearest its mirror (within ``max(0.4 R, 2 px)``) and re-centre on the mean pair
    midpoint, three times. A pattern with fewer than 4 disks or 2 pairs takes the
    median of the other patterns' beams, or the pattern centre when none has one.
    """
    tolerance = max(0.4 * float(spot_radius), 2.0)
    beams = np.full((len(positions), 2), np.nan)
    for number, points in enumerate(positions):
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if len(points) < 4:
            continue
        centre = points[np.argmin(np.hypot(*(points - points.mean(0)).T))]
        found = False
        for _ in range(3):
            mirror = 2 * centre - points
            distance = np.hypot(
                mirror[:, None, 0] - points[None, :, 0], mirror[:, None, 1] - points[None, :, 1]
            )
            np.fill_diagonal(distance, np.inf)
            partner = distance.argmin(1)
            index = np.arange(len(points))
            paired = (distance[index, partner] < tolerance) & (index < partner)
            if paired.sum() < 2:
                found = False
                break
            centre = 0.5 * (points[paired] + points[partner[paired]]).mean(0)
            found = True
        if found:
            beams[number] = centre
    if np.isfinite(beams).any():
        fallback = np.nanmedian(beams, 0)
    else:
        fallback = np.asarray(frame_shape, np.float64) / 2
    return np.where(np.isfinite(beams), beams, fallback)


def refine_centres(
    patterns,
    positions: list[np.ndarray],
    spot_radius: float,
    refiner: CentreRefiner,
    batch_size: int = DEFAULT_BATCH,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Refine every pattern's ``(n, 2)`` ``[row, col]`` disk centres.

    Windows are cut and refined a batch at a time across patterns. Each disk's result
    depends only on its own window (and, for the Friedel-partner step, its pattern's
    beam), so refining one pattern alone gives the same answer as refining it among
    others.

    Parameters
    ----------
    patterns : sequence of np.ndarray
        The raw ``(H, W)`` patterns.
    positions : list of np.ndarray
        Per pattern, the ``(n, 2)`` incoming centres.
    spot_radius : float
        Disk radius in pixels.
    refiner : CentreRefiner
        The step's network.
    batch_size : int, default=4096
        Disks refined per batch.

    Returns
    -------
    tuple
        Per pattern, the refined ``(n, 2)`` float32 centres and the ``(n,)`` sigma
        (NaN where the incoming centre was kept); the sigmas are ``None`` when the
        step was skipped for this spot radius.
    """
    radius = float(spot_radius)
    refined = [np.asarray(p, np.float32).reshape(-1, 2).copy() for p in positions]
    if not refiner.min_spot_radius <= radius <= refiner.max_spot_radius:
        return refined, None
    half_width = refiner.crop_half_width(radius)
    sigmas = [np.full(len(p), np.nan, np.float32) for p in refined]
    beams = (
        frame_beams(refined, np.asarray(patterns[0]).shape, radius)
        if refiner.needs_partner
        else None
    )

    def cut(pending, points_of):
        windows = [extract_crops(patterns[i], points_of(i), half_width) for i, _ in pending]
        return (
            np.concatenate([crops for crops, _ in windows]),
            np.concatenate([local for _, local in windows]),
        )

    def flush(pending):
        crops, local = cut(pending, lambda i: refined[i].astype(np.float64))
        if refiner.needs_partner:
            mirror_crops, mirror_local = cut(
                pending, lambda i: 2 * beams[i] - refined[i].astype(np.float64)
            )
            new_local, sigma = refiner.refine(crops, local, radius, mirror_crops, mirror_local)
        else:
            new_local, sigma = refiner.refine(crops, local, radius)
        new_local = np.asarray(new_local, np.float64).reshape(-1, 2)
        shift = np.hypot(*(new_local - local).T)
        accepted = np.isfinite(new_local).all(1) & (shift <= MAX_SHIFT_FRACTION * radius)
        start = 0
        for index, points in pending:
            stop = start + len(points)
            keep = accepted[start:stop]
            moved = refined[index] + (new_local[start:stop] - local[start:stop])
            refined[index][keep] = moved[keep]
            sigmas[index][keep] = np.asarray(sigma[start:stop])[keep]
            start = stop

    pending, pending_count = [], 0
    for index, points in enumerate(refined):
        if len(points) == 0:
            continue
        pending.append((index, points.copy()))
        pending_count += len(points)
        if pending_count >= batch_size:
            flush(pending)
            pending, pending_count = [], 0
    if pending:
        flush(pending)
    return refined, sigmas
