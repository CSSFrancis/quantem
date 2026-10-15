"""Teaching DE SpotNet from marks, saving the result, and where the weights come from."""

import numpy as np
import pytest
import torch

from quantem.core.ml.spot_net import SpotNet
from quantem.diffraction import neural_weights
from quantem.diffraction.neural_adapt import (
    DISK,
    NOT_DISK,
    Working,
    load_replay,
    mark_targets,
    marks_learned,
    original_f1,
    teach_spot_net,
)
from quantem.diffraction.neural_disk_detection import load_spot_net, save_spot_net

SIZE = 64


def tiny_model(mask_head=False):
    torch.manual_seed(0)
    return SpotNet(base=4, levels=2, mask_head=mask_head).eval()


def pattern(seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    image = sum(
        100.0 / (1 + np.exp((np.hypot(rows - r, cols - c) - 4.0) / 0.5))
        for r, c in [(16, 16), (16, 48), (48, 16), (48, 48), (32, 32)]
    )
    return rng.poisson(image + 3.0).astype(np.float32)


def tiny_replay(path):
    rng = np.random.default_rng(1)
    heatmap = np.zeros((32, 32, 32), np.uint8)
    heatmap[:, 16, 16] = 255
    mask = heatmap > 0
    np.savez(
        path,
        x=rng.normal(size=(32, 32, 32)).astype(np.float16),
        heatmap=heatmap,
        offset=np.zeros((32, 2, 32, 32), np.float16),
        mask=mask,
    )
    return path


def test_mark_targets_hold_marks_and_weigh_each_mark_once():
    model = tiny_model()
    working = Working([pattern()], model, spot_radius=4.0, threshold=0.0)
    marks = [np.array([[16.0, 16.0, DISK], [5.0, 40.0, NOT_DISK]])]
    labels = mark_targets(model, working, marks, disk_radius=4.0)
    assert labels["target"].shape == labels["heatmap_base"].shape
    assert float(labels["target"].max()) == 1.0
    # two marks, each normalized to a total weight of one
    assert float(labels["weight"].sum()) == pytest.approx(2.0, rel=1e-4)


@pytest.mark.parametrize("mask_head", [False, True])
def test_teach_fine_tunes_a_copy_and_reports(tmp_path, mask_head):
    base = tiny_model(mask_head)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    marks = [np.array([[16.0, 16.0, DISK], [5.0, 40.0, NOT_DISK]])]
    taught, report = teach_spot_net(
        base,
        [pattern()],
        marks,
        spot_radius=4.0,
        replay=tiny_replay(tmp_path / "replay.npz"),
        until_learned=False,
        steps=3,
    )
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in taught.state_dict().items())
    assert report["steps"] == 3 and report["marks_total"] == 2
    assert report["disk_marks"] == 1 and report["not_disk_marks"] == 1
    assert report["replay"] is True and 0.0 <= report["original_f1"] <= 1.0
    assert not any(p.requires_grad for p in taught.parameters())


def test_teach_without_replay_reports_no_general_score():
    taught, report = teach_spot_net(
        tiny_model(),
        [pattern()],
        [np.array([[5.0, 40.0, NOT_DISK]])],
        until_learned=False,
        steps=2,
    )
    assert report["original_f1"] is None and report["replay"] is False


def test_marks_learned_checks_each_mark_with_the_detection_decode():
    model = tiny_model()
    working = Working([pattern()], model, spot_radius=4.0, threshold=1.0)
    # at threshold 1 nothing is detected: "not a disk" is learned, "disk here" is not
    learned = marks_learned(model, working, [np.array([[5, 5, NOT_DISK], [16, 16, DISK]])], 4.0)
    assert learned.tolist() == [True, False]


def test_replay_bank_round_trip_and_original_f1(tmp_path):
    bank = load_replay(tiny_replay(tmp_path / "replay.npz"))
    assert bank["x"].shape == (32, 1, 32, 32) and bank["offset"].shape == (32, 2, 32, 32)
    assert 0.0 <= original_f1(tiny_model(), bank) <= 1.0


def test_saved_taught_model_loads_with_model_keyword(tmp_path):
    model = tiny_model(mask_head=True)
    model.mad_floor, model.default_threshold = 0.05, 0.2
    path = save_spot_net(model, tmp_path / "taught.pt")
    loaded = load_spot_net(path)
    assert loaded.head_mask is not None and loaded.mad_floor == 0.05
    assert loaded.default_threshold == 0.2
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        assert all(torch.allclose(a, b) for a, b in zip(model(x), loaded(x)))
    assert load_spot_net(model) is model  # a model in memory is used as it is


def test_entries_are_the_torch_ones_pinned_to_one_version(monkeypatch):
    calls = []

    class FakeEntry:
        class metadata:
            license = "CC-BY-NC-4.0"

        def download(self, version, background, progressbar):
            calls.append(version)
            return "fake.pt"

    from emdatabase import data

    for entry in neural_weights.DETECTOR_ENTRIES.values():
        assert entry.endswith("_Torch")
        monkeypatch.setattr(data, entry, FakeEntry, raising=False)
    assert neural_weights.CENTER_REFINER_ENTRY.endswith("_Torch")
    assert neural_weights.FRIEDEL_PARTNER_ENTRY.endswith("_Torch")
    assert str(neural_weights.detector_path("dynamical")) == "fake.pt"
    neural_weights.detector_path("spotnet", version="999999")
    assert calls == [neural_weights.WEIGHTS_VERSION, "999999"]
