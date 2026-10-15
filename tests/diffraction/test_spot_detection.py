"""Template-free detectors of BraggVectors.detect_disks: Difference-of-Gaussians and neural."""

import os

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.core.ml.spot_unet import SpotUNet
from quantem.diffraction import neural_disk_detection
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.neural_disk_detection import (
    _decode,
    detect_disks_neural_batch,
    load_spot_unet,
)
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


def _random_checkpoint(path, nested_config=False, levels=2):
    torch.manual_seed(0)
    model = SpotUNet(base=4, levels=levels)
    config = {"base": 4, "in_ch": 1, "levels": levels}
    if nested_config:
        torch.save({"state_dict": model.state_dict(), "config": config}, path)
    else:
        torch.save({"state_dict": model.state_dict(), **config}, path)
    return model


@pytest.mark.parametrize("nested_config", [False, True])
def test_load_spot_unet_reads_both_checkpoint_layouts(tmp_path, nested_config):
    path = tmp_path / "spot_unet.pt"
    reference = _random_checkpoint(path, nested_config=nested_config).eval()
    model = load_spot_unet(path)
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x)[0], reference(x)[0])


def test_neural_method_runs_end_to_end(tmp_path):
    path = tmp_path / "spot_unet.pt"
    _random_checkpoint(path)
    bv = BraggVectors.from_dataset(_dataset())
    # A random network carries no meaning; a zero threshold exercises every stage.
    peaks = bv.detect_disks(method="neural", model_path=path, threshold=0.0, progressbar=False)
    assert peaks.shape == SCAN
    for r in range(SCAN[0]):
        for c in range(SCAN[1]):
            cell = peaks[r, c].numpy()
            assert cell.shape[1] == 3 and cell.shape[0] > 0
            assert cell[:, 0].max() < H and cell[:, 1].max() < W
            assert np.all(np.diff(cell[:, 2]) <= 0)
    assert bv.metadata["detect"]["method"] == "neural"


def test_decode_moves_peaks_by_the_offset_head():
    logits = torch.full((2, 1, 16, 16), -10.0)
    logits[1, 0, 5, 9] = 10.0
    offset = torch.zeros(2, 2, 16, 16)
    offset[1, :, 5, 9] = torch.tensor([0.25, -0.4])
    peaks, batch_index = _decode(logits, offset, threshold=0.3, window=2)
    assert batch_index.tolist() == [1]
    assert torch.allclose(peaks[0, :2], torch.tensor([5.25, 8.6]))


def test_missing_emdatabase_entry_points_to_model_path(monkeypatch):
    from emdatabase import data

    monkeypatch.delattr(data, neural_disk_detection.EMDATABASE_ENTRY, raising=False)
    with pytest.raises(RuntimeError, match="model_path"):
        load_spot_unet()


@pytest.mark.skipif(
    not os.environ.get("SPOT_UNET_WEIGHTS"),
    reason="set SPOT_UNET_WEIGHTS to a trained SpotUNet checkpoint",
)
def test_trained_weights_localise_planted_disks():
    model = load_spot_unet(os.environ["SPOT_UNET_WEIGHTS"])
    # Flat-topped disks of radius 4, the regime the network was trained on.
    rng = np.random.default_rng(1)
    dp = sum(
        200.0 / (1 + np.exp((np.hypot(_YY - r, _XX - c) - 4.0) / 0.5)) for r, c in SPOTS
    )
    dp = rng.poisson(dp + 5.0).astype(np.float32)
    cell = detect_disks_neural_batch(torch.as_tensor(dp[None]), model)[0]
    assert _recovered(cell, tolerance=1.0) == len(SPOTS)
