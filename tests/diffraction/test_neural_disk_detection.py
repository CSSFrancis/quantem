"""The neural detector of BraggVectors.detect_disks and its center steps.

Most tests use randomly initialized networks, which carry no meaning but exercise
every stage without a download. ``SPOT_UNET_WEIGHTS`` pointing at a directory of the
trained checkpoints enables the tests that need real weights.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.core.ml.spot_center_net import SpotCenterNet
from quantem.core.ml.spot_unet import SpotUNet
from quantem.diffraction import neural_weights
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.center_refinement import (
    extract_crops,
    frame_beams,
    load_center_refiner,
    refine_centers,
)
from quantem.diffraction.neural_disk_detection import (
    NEURAL_PEAK_FIELDS,
    _decode,
    _to_pattern_coordinates,
    detect_disks_neural_batch,
    load_spot_unet,
    neural_heatmap,
    positional_sigma,
)

H = W = 64
SCAN = (3, 4)
_YY, _XX = np.mgrid[0:H, 0:W]
SPOTS = [(20.3, 18.6), (20.1, 44.4), (44.7, 19.2), (43.6, 45.5), (32.2, 31.8)]
WEIGHTS = os.environ.get("SPOT_UNET_WEIGHTS")


def _dataset():
    rng = np.random.default_rng(0)
    frames = [
        sum(200.0 * np.exp(-((_YY - r) ** 2 + (_XX - c) ** 2) / (2 * 1.2**2)) for r, c in SPOTS)
        + 5.0
        + rng.normal(0, 1.0, (H, W))
        for _ in range(SCAN[0] * SCAN[1])
    ]
    return Dataset4dstem.from_array(np.stack(frames).reshape(*SCAN, H, W).astype(np.float32))


def _detector_checkpoint(path, mask_head=False, nested_config=False):
    torch.manual_seed(0)
    model = SpotUNet(base=4, levels=2, mask_head=mask_head)
    config = {"base": 4, "in_ch": 1, "levels": 2, "mask_head": int(mask_head)}
    if nested_config:
        torch.save({"state_dict": model.state_dict(), "config": config}, path)
    else:
        torch.save({"state_dict": model.state_dict(), **config, "mad_floor": 0.05}, path)
    return model


def _center_checkpoint(path, channels=1, **scalars):
    torch.manual_seed(1)
    network = SpotCenterNet(base=4, levels=2, channels=channels)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "base": 4,
            "levels": 2,
            "crop_radius": 10.0,
            "crop_half": 16,
            **scalars,
        },
        path,
    )


@pytest.mark.parametrize("mask_head", [False, True])
@pytest.mark.parametrize("nested_config", [False, True])
def test_load_spot_unet_reads_both_checkpoint_layouts(tmp_path, mask_head, nested_config):
    path = tmp_path / "spot_unet.pt"
    reference = _detector_checkpoint(path, mask_head, nested_config).eval()
    model = load_spot_unet(path)
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        ours, theirs = model(x), reference(x)
    assert len(ours) == (3 if mask_head else 2)
    for a, b in zip(ours, theirs):
        assert torch.allclose(a, b)
    assert model.mad_floor == (None if nested_config else 0.05)


def test_neural_method_returns_confidence_and_sigma(tmp_path):
    path = tmp_path / "spot_unet.pt"
    _detector_checkpoint(path)
    bv = BraggVectors.from_dataset(_dataset())
    peaks = bv.detect_disks(
        method="neural", model=path, threshold=0.0, refine_centers=False, progressbar=False
    )
    assert peaks.shape == SCAN
    assert tuple(peaks.fields) == NEURAL_PEAK_FIELDS
    cell = peaks[1, 2].numpy()
    assert cell.shape[1] == 5 and cell.shape[0] > 0
    assert cell[:, 0].max() < H and cell[:, 1].max() < W
    assert np.all(np.diff(cell[:, 2]) <= 0)
    assert np.all((cell[:, 3] >= 0) & (cell[:, 3] <= 1))
    assert bv.metadata["detect"]["refine_centers"] is False
    assert bv.bvm is not None  # the extra fields do not disturb the Bragg vector map


def test_center_steps_run_and_their_sigma_replaces_the_counting_limit(tmp_path):
    detector, refiner, partner = (tmp_path / n for n in ("d.pt", "f5.pt", "p4.pt"))
    _detector_checkpoint(detector)
    _center_checkpoint(refiner, min_spot_radius=0.0)
    _center_checkpoint(partner, channels=2, min_spot_radius=0.0)
    bv = BraggVectors.from_dataset(_dataset())
    common = dict(method="neural", model=detector, threshold=0.0, positions=[(0, 0), (2, 3)])
    plain = bv.detect_disks(refine_centers=False, **common)
    stepped = bv.detect_disks(refine_centers=refiner, friedel_partner=partner, **common)
    assert plain[0].numpy().shape == stepped[0].numpy().shape
    # Once a step runs, sigma is its own estimate (NaN where it declined a disk)
    # rather than the counting-noise limit.
    plain_sigma, stepped_sigma = plain[0].numpy()[:, 4], stepped[0].numpy()[:, 4]
    assert np.isfinite(plain_sigma).any()
    assert not np.allclose(plain_sigma, stepped_sigma, equal_nan=True)


def test_default_center_steps_warn_and_continue_without_weights(tmp_path, monkeypatch):
    from emdatabase import data

    monkeypatch.delattr(data, neural_weights.CENTER_REFINER_ENTRY, raising=False)
    path = tmp_path / "spot_unet.pt"
    _detector_checkpoint(path)
    bv = BraggVectors.from_dataset(_dataset())
    with pytest.warns(UserWarning, match=neural_weights.CENTER_REFINER_ENTRY):
        found = bv.detect_disks(method="neural", model=path, threshold=0.0, positions=[(0, 0)])
    assert found[0].numpy().shape[1] == 5


def test_decode_moves_peaks_by_the_offset_head():
    logits = torch.full((2, 1, 16, 16), -10.0)
    logits[1, 0, 5, 9] = 10.0
    offset = torch.zeros(2, 2, 16, 16)
    offset[1, :, 5, 9] = torch.tensor([0.25, -0.4])
    peaks, batch_index = _decode((logits, offset), threshold=0.3, window=2)
    assert batch_index.tolist() == [1]
    assert torch.allclose(peaks[0, :2], torch.tensor([5.25, 8.6]))
    assert 0.99 < float(peaks[0, 2]) <= 1.0


def test_decode_centers_on_the_mask_when_it_is_a_disk():
    logits = torch.full((1, 1, 32, 32), -10.0)
    logits[0, 0, 15, 15] = 10.0
    offset = torch.zeros(1, 2, 32, 32)
    rows, cols = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    disk = torch.hypot(rows - 15.6, cols - 15.3) <= 4.5
    mask = torch.where(disk, 10.0, -10.0)[None, None]
    peaks, _ = _decode((logits, offset, mask), threshold=0.3, window=2, mask_radius=4.5)
    assert abs(float(peaks[0, 0]) - 15.6) < 0.1 and abs(float(peaks[0, 1]) - 15.3) < 0.1


def test_positions_map_back_with_the_end_aligned_inverse():
    peaks = torch.tensor([[0.0, 0.0, 1.0, 2.0], [20.0, 20.0, 1.0, 2.0]])
    factor = 0.5
    back = _to_pattern_coordinates(peaks, (41, 41), factor)
    # 41 px resampled to 20: working pixel 19 is the last native pixel, 40.
    assert torch.allclose(back[1, :2], torch.tensor([20 * 40 / 19, 20 * 40 / 19]))
    assert float(back[0, 3]) == 4.0


def test_positional_sigma_shrinks_with_counts():
    sigma = positional_sigma([1.0, 1.0, 1.0], [100.0, 10000.0, 0.0], [1.0, 1.0, 1.0])
    assert sigma[1] < sigma[0] and np.isnan(sigma[2])


def test_frame_beams_finds_the_center_of_symmetry():
    rng = np.random.default_rng(2)
    beam = np.array([30.4, 33.1])
    g = rng.uniform(-20, 20, (6, 2))
    points = np.concatenate([beam[None], beam + g, beam - g])
    assert np.allclose(frame_beams([points], (64, 64), 3.0)[0], beam, atol=1e-6)
    # too few disks: the pattern center
    assert np.allclose(frame_beams([points[:2]], (64, 64), 3.0)[0], [32, 32])


def test_extract_crops_zero_outside_the_pattern():
    pattern = np.arange(100, dtype=np.float32).reshape(10, 10)
    crops, local = extract_crops(pattern, np.array([[0.2, 9.4]]), 2)
    assert crops.shape == (1, 5, 5)
    assert crops[0, 0, 0] == 0 and crops[0, 2, 2] == pattern[0, 9]
    assert np.allclose(local, [[2.2, 2.4]])


def test_refine_centers_skips_outside_its_spot_radius_range(tmp_path):
    path = tmp_path / "p4.pt"
    _center_checkpoint(path, channels=2, min_spot_radius=6.0, max_spot_radius=16.0)
    refiner = load_center_refiner(path)
    assert refiner.needs_partner
    positions = [np.array([[10.0, 10.0]])]
    refined, sigmas = refine_centers([np.zeros((32, 32))], positions, 3.0, refiner)
    assert sigmas is None and np.array_equal(refined[0], positions[0])


def test_detector_names_live_in_one_place(tmp_path):
    assert neural_weights.DEFAULT_DETECTOR in neural_weights.DETECTOR_ENTRIES
    with pytest.raises(ValueError, match="model must be"):
        neural_weights.detector_path("not-a-detector")
    path = tmp_path / "local.pt"
    path.write_bytes(b"")
    assert neural_weights.detector_path(path) == path


def test_neural_heatmap_is_on_the_detector_grid(tmp_path):
    path = tmp_path / "spot_unet.pt"
    _detector_checkpoint(path, mask_head=True)
    dps = torch.as_tensor(np.asarray(_dataset().array[0, :2]))
    heatmap = neural_heatmap(dps, load_spot_unet(path))
    assert heatmap.shape == (2, H, W)
    assert float(heatmap.min()) >= 0 and float(heatmap.max()) <= 1


@pytest.mark.skipif(not WEIGHTS, reason="set SPOT_UNET_WEIGHTS to the trained checkpoints")
@pytest.mark.parametrize(
    "detector", ["spotunet_anyscale8_peaked_sharpened.pt", "spotunet_dynamical_mask8.pt"]
)
def test_trained_weights_localize_disks_and_the_steps_tighten_them(detector):
    weights = Path(WEIGHTS)
    model = load_spot_unet(weights / detector)
    refiner = load_center_refiner(weights / "center_fast_f5.pt")
    rng = np.random.default_rng(1)
    rows, cols = np.mgrid[0:128, 0:128]
    centers = [
        (64 + 24 * i + 0.3 * j, 64 + 24 * j - 0.2 * i)
        for i in (-2, -1, 0, 1, 2)
        for j in (-2, -1, 0, 1, 2)
    ]
    pattern = sum(
        150.0 / (1 + np.exp((np.hypot(rows - r, cols - c) - 7.0) / 0.5)) for r, c in centers
    )
    pattern = torch.as_tensor(rng.poisson(pattern + 4.0).astype(np.float32)[None])

    def error(found):
        distance = np.hypot(
            found[:, None, 0] - np.array(centers)[None, :, 0],
            found[:, None, 1] - np.array(centers)[None, :, 1],
        )
        assert (distance.min(0) < 1.0).all()
        return float(np.median(distance.min(0)))

    plain = detect_disks_neural_batch(pattern, model)[0]
    refined = detect_disks_neural_batch(pattern, model, center_refiner=refiner)[0]
    assert error(refined) <= error(plain) + 0.02
    assert np.isfinite(refined[:, 4]).mean() > 0.5
