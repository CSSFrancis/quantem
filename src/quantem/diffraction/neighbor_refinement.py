"""Reject false detections using the neighboring scan positions.

Run after detection at a permissive threshold: the detector over-detects for recall,
and this step encodes physical constraints to drop the false positives while keeping
faint real disks.

- **Overlap merge**: candidates closer than ``min_separation`` are one disk detected
  twice; they merge to an intensity-weighted position.
- **Persistence**: a real disk recurs at about the same position in the adjacent scan
  positions; a false positive does not. This is the strongest signal.
- **Friedel (soft)**: a disk with a partner near ``-g`` about the direct beam scores
  higher. It is a soft prior, not a filter, because amorphous and disordered regions
  legitimately break Friedel symmetry.

Each candidate's score combines its relative brightness with these signals, and
candidates scoring at least ``keep_score`` survive.
"""

from __future__ import annotations

import numpy as np


def merge_overlaps(peaks: np.ndarray, min_separation: float = 3.0) -> np.ndarray:
    """Merge candidates closer than ``min_separation``, keeping the brightest one's score.

    Parameters
    ----------
    peaks : np.ndarray
        ``(N, >=3)`` ``[row, col, score]`` candidates.
    min_separation : float, default=3.0
        Distance in pixels below which two candidates are the same disk.

    Returns
    -------
    np.ndarray
        ``(M, 3)`` merged ``[row, col, score]``; positions are score-weighted.
    """
    peaks = np.asarray(peaks, np.float64)
    if len(peaks) <= 1:
        return peaks[:, :3].astype(np.float32)
    kept = []
    used = np.zeros(len(peaks), bool)
    for index in np.argsort(-peaks[:, 2]):
        if used[index]:
            continue
        distance = np.hypot(peaks[:, 0] - peaks[index, 0], peaks[:, 1] - peaks[index, 1])
        group = (distance <= min_separation) & ~used
        used[group] = True
        # A score that is a disk-mean intensity can be zero (or negative on
        # background-subtracted data) for the whole group; average unweighted then.
        weights = np.clip(peaks[group, 2], 0, None)
        weights = weights if weights.sum() > 0 else None
        kept.append(
            [
                np.average(peaks[group, 0], weights=weights),
                np.average(peaks[group, 1], weights=weights),
                peaks[group, 2].max(),
            ]
        )
    return np.asarray(kept, np.float32)


def persistence_score(
    peaks: np.ndarray, neighbor_peaks: list[np.ndarray], tolerance: float = 3.0
) -> np.ndarray:
    """Fraction of the neighboring positions with a candidate within ``tolerance``."""
    peaks = np.asarray(peaks, np.float64)
    if len(peaks) == 0 or not neighbor_peaks:
        return np.zeros(len(peaks))
    score = np.zeros(len(peaks))
    for i, peak in enumerate(peaks):
        hits = sum(
            1
            for neighbor in neighbor_peaks
            if len(neighbor)
            and np.min(
                np.hypot(
                    np.asarray(neighbor)[:, 0] - peak[0], np.asarray(neighbor)[:, 1] - peak[1]
                )
            )
            <= tolerance
        )
        score[i] = hits / len(neighbor_peaks)
    return score


def friedel_score(
    peaks: np.ndarray, center: np.ndarray, tolerance: float = 3.0, min_radius: float = 8.0
) -> np.ndarray:
    """1 where a candidate has a partner near its mirror about ``center``, else 0.

    Candidates within ``min_radius`` of the center always score 1.
    """
    peaks = np.asarray(peaks, np.float64)
    center = np.asarray(center, np.float64)[:2]
    if len(peaks) == 0:
        return np.zeros(0)
    score = np.zeros(len(peaks))
    relative = peaks[:, :2] - center
    radius = np.linalg.norm(relative, axis=1)
    for i in range(len(peaks)):
        if radius[i] < min_radius:
            score[i] = 1.0
            continue
        mirror = center - relative[i]
        if np.min(np.hypot(peaks[:, 0] - mirror[0], peaks[:, 1] - mirror[1])) <= tolerance:
            score[i] = 1.0
    return score


def refine_with_neighbors(
    peaks: np.ndarray,
    neighbor_peaks: list[np.ndarray],
    frame_shape: tuple[int, int],
    *,
    min_separation: float = 3.0,
    persistence_tolerance: float = 3.0,
    friedel_tolerance: float = 3.0,
    confidence_weight: float = 0.2,
    persistence_weight: float = 1.0,
    friedel_weight: float = 0.2,
    keep_score: float = 0.55,
    min_persistence: float | None = 0.5,
) -> np.ndarray:
    """Keep the candidates of one position that the evidence supports.

    The score is ``confidence_weight * brightness + persistence_weight * persistence +
    friedel_weight * friedel``, with brightness the candidate's score over the
    position's brightest. The Friedel center is the brightest candidate within a fifth
    of the pattern of its middle (the direct beam).

    Parameters
    ----------
    peaks : np.ndarray
        ``(N, >=3)`` candidates of this position; column 2 is their score or intensity.
        Further columns are carried over from the nearest original candidate.
    neighbor_peaks : list of np.ndarray
        The candidates of the neighboring positions.
    frame_shape : tuple of int
        ``(H, W)`` of the patterns.

    Returns
    -------
    np.ndarray
        The kept candidates, with the same columns as ``peaks``.
    """
    peaks = np.asarray(peaks, np.float32)
    if len(peaks) == 0:
        return peaks
    merged = merge_overlaps(peaks, min_separation)
    persistence = persistence_score(merged, neighbor_peaks, persistence_tolerance)
    middle = np.array(frame_shape[:2]) / 2.0
    near = np.hypot(merged[:, 0] - middle[0], merged[:, 1] - middle[1]) < 0.2 * frame_shape[0]
    candidates = merged[near] if near.any() else merged
    center = candidates[np.argmax(candidates[:, 2]), :2]
    friedel = friedel_score(merged, center, friedel_tolerance)
    brightness = np.clip(merged[:, 2] / (merged[:, 2].max() + 1e-9), 0, 1)
    score = (
        confidence_weight * brightness
        + persistence_weight * persistence
        + friedel_weight * friedel
    )
    keep = score >= keep_score
    if min_persistence is not None:
        keep &= persistence >= min_persistence
    kept = merged[keep]
    if peaks.shape[1] <= 3 or len(kept) == 0:
        return kept.reshape(-1, 3) if peaks.shape[1] <= 3 else np.zeros((0, peaks.shape[1]))
    distance = np.hypot(kept[:, None, 0] - peaks[None, :, 0], kept[:, None, 1] - peaks[None, :, 1])
    return np.column_stack([kept, peaks[distance.argmin(1), 3:]]).astype(np.float32)


def refine_scan_with_neighbors(
    peaks_grid: list[list[np.ndarray]], frame_shape: tuple[int, int], **kwargs
) -> list[list[np.ndarray]]:
    """:func:`refine_with_neighbors` at every position of a scan.

    Uses the four adjacent positions. A position with fewer than two neighbors, or
    with no candidates, is left as it is rather than risk dropping real disks for lack
    of evidence.

    Parameters
    ----------
    peaks_grid : list of list of np.ndarray
        ``peaks_grid[row][col]`` is that position's ``(N, >=3)`` candidates.
    frame_shape : tuple of int
        ``(H, W)`` of the patterns.
    **kwargs
        Passed to :func:`refine_with_neighbors`.

    Returns
    -------
    list of list of np.ndarray
        The kept candidates, in the same layout.
    """
    rows, cols = len(peaks_grid), len(peaks_grid[0]) if peaks_grid else 0
    refined = []
    for r in range(rows):
        refined_row = []
        for c in range(cols):
            peaks = peaks_grid[r][c]
            neighbors = [
                peaks_grid[r + dr][c + dc][:, :3]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < rows and 0 <= c + dc < cols
            ]
            if len(peaks) == 0 or len(neighbors) < 2:
                refined_row.append(peaks)
                continue
            refined_row.append(
                refine_with_neighbors(peaks, neighbors, frame_shape, **kwargs).astype(peaks.dtype)
            )
        refined.append(refined_row)
    return refined
