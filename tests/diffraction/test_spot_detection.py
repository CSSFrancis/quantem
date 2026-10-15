"""The Difference-of-Gaussians detector of BraggVectors.detect_disks, and neighbour refine."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.spot_detection import detect_beamstop, detect_disks_dog_batch

H = W = 64
SCAN = (3, 4)
_YY, _XX = np.mgrid[0:H, 0:W]
# Subpixel spot centres shared by every pattern, away from the beam stop corner.
SPOTS = [(20.3, 18.6), (20.1, 44.4), (44.7, 19.2), (43.6, 45.5), (32.2, 31.8)]


def _pattern(rng, spot_sigma=1.2, amplitude=200.0):
    dp = sum(
        amplitude * np.exp(-((_YY - r) ** 2 + (_XX - c) ** 2) / (2 * spot_sigma**2))
        for r, c in SPOTS
    )
    return dp + 5.0 + rng.normal(0, 1.0, (H, W))


def _dataset(spot_sigma=1.2):
    rng = np.random.default_rng(0)
    data = np.stack([_pattern(rng, spot_sigma) for _ in range(SCAN[0] * SCAN[1])])
    return Dataset4dstem.from_array(data.reshape(*SCAN, H, W).astype(np.float32))


def _recovered(peaks, tolerance):
    """Number of planted spots with a detected peak within ``tolerance`` pixels."""
    if peaks.shape[0] == 0:
        return 0
    return sum(
        bool((np.hypot(peaks[:, 0] - r, peaks[:, 1] - c) < tolerance).any()) for r, c in SPOTS
    )


def test_dog_recovers_planted_spots_with_subpixel_accuracy():
    bv = BraggVectors.from_dataset(_dataset())
    peaks = bv.detect_disks(method="dog", threshold=8.0, progressbar=False)
    assert peaks.shape == SCAN
    for r in range(SCAN[0]):
        for c in range(SCAN[1]):
            cell = peaks[r, c].numpy()
            assert cell.shape[0] == len(SPOTS)
            assert _recovered(cell, tolerance=0.25) == len(SPOTS)
    assert bv.metadata["detect"]["method"] == "dog"
    assert bv.bvm is not None


def test_dog_needs_no_template_and_positions_leave_state_untouched():
    bv = BraggVectors.from_dataset(_dataset())
    test = bv.detect_disks(method="dog", positions=[(0, 0), (2, 3)], threshold=8.0)
    assert test.shape == (2,)
    assert _recovered(test[1].numpy(), tolerance=0.25) == len(SPOTS)
    assert bv.peaks is None


def test_dog_intensity_is_the_raw_disk_mean_and_sorted():
    dps = torch.as_tensor(np.asarray(_dataset().array[0, :2]))
    for cell in detect_disks_dog_batch(dps, threshold=8.0):
        assert np.all(np.diff(cell[:, 2]) <= 0)
        # The disk mean of a 200-count spot on a 5-count floor sits well between them.
        assert np.all((cell[:, 2] > 20) & (cell[:, 2] < 205))


def test_beamstop_mask_drops_peaks_inside_it():
    dataset = _dataset()
    mask = np.zeros((H, W), dtype=bool)
    mask[38:52, 38:52] = True  # covers the spot at (43.6, 45.5)
    bv = BraggVectors.from_dataset(dataset)
    cell = bv.detect_disks(method="dog", positions=[(0, 0)], threshold=8.0, mask=mask)[0].numpy()
    assert cell.shape[0] == len(SPOTS) - 1
    assert not mask[np.rint(cell[:, 0]).astype(int), np.rint(cell[:, 1]).astype(int)].any()


def test_detect_beamstop_finds_a_dark_region_and_ignores_none():
    pattern = np.full((H, W), 100.0)
    pattern[:10, :10] = 1.0
    mask = detect_beamstop(pattern, dilate=2)
    assert mask is not None and mask[:12, :12].all() and not mask[20:, 20:].any()
    assert detect_beamstop(np.full((H, W), 100.0)) is None


def test_default_method_is_still_template_correlation():
    bv = BraggVectors.from_dataset(_dataset())
    with pytest.raises(ValueError, match="make_template"):
        bv.detect_disks(positions=[(0, 0)])
    with pytest.raises(ValueError, match="method"):
        bv.detect_disks(method="unknown", positions=[(0, 0)])


def test_neighbor_refine_drops_a_spot_seen_at_one_position_only():
    dataset = _dataset()
    data = np.array(dataset.array)
    spurious = (52.0, 30.0)
    data[1, 1] += 200.0 * np.exp(
        -((_YY - spurious[0]) ** 2 + (_XX - spurious[1]) ** 2) / (2 * 1.2**2)
    )
    dataset = Dataset4dstem.from_array(data)

    def spurious_found(neighbor_refine):
        bv = BraggVectors.from_dataset(dataset)
        cell = bv.detect_disks(
            method="dog", threshold=8.0, neighbor_refine=neighbor_refine, progressbar=False
        )[1, 1].numpy()
        return bool((np.hypot(cell[:, 0] - spurious[0], cell[:, 1] - spurious[1]) < 1).any())

    assert spurious_found(neighbor_refine=False)
    assert not spurious_found(neighbor_refine=True)
    bv = BraggVectors.from_dataset(dataset)
    kept = bv.detect_disks(method="dog", threshold=8.0, neighbor_refine=True, progressbar=False)
    assert _recovered(kept[1, 1].numpy(), tolerance=0.25) == len(SPOTS)
    assert bv.metadata["detect"]["neighbor_refine"] is True
