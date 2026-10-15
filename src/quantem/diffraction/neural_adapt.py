"""Teach DE SpotNet one dataset from a user's marks.

The user marks wrong detections ("not a disk") and places where they see a disk the
detector missed ("disk here") on a few patterns. A copy of the model is fine-tuned on
those marks for a few seconds; the original is never touched.

What was measured to work, on synthetic data with a deliberate domain shift (five
datasets, two draws of marks each):

* All weights, Adam at 1e-3. Training runs until every mark is learned (no detection
  on each "not a disk" mark and one on each "disk here" mark, checked with the same
  decode detection uses) or a few seconds pass; the report says which. Training only
  the last heatmap layer made detection worse: eight feature channels cannot separate
  a faint disk from a hot pixel linearly.
* Almost all of the gain is fewer false positives (20 to 1-2 per pattern, F1 0.75 to
  0.83). "Not a disk" marks carry it; "disk here" marks barely move recall, and on
  their own they raise the response everywhere.
* Unmarked pixels are held to the base model's own output (weighted toward its peaks),
  so a mark changes the neighborhood of what was marked, not the whole pattern. The
  heads that place a disk (the offset head and, on a model that has one, the mask
  head) are held to the base model too: a click is not subpixel accurate, so it names
  the disk, never its center.
* Replaying 20 % of the synthetic frames the model was trained on keeps it a disk
  detector: without them its F1 on that distribution falls from 0.86 to about 0.60,
  with them to about 0.78. :func:`original_f1` reports that number, so the user can
  see how far teaching pulled the model toward their data.

A taught model is therefore a model for this dataset, not a better default.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quantem.core.ml.spot_net import SpotNet
from quantem.diffraction.neural_disk_detection import (
    CANONICAL_DIAMETER,
    DEFAULT_THRESHOLD,
    _decode,
    _network_input,
    _resolve_window_and_background,
    _to_pattern_coordinates,
    _working_diameter,
    canonical_scale_factor,
    counts_radius,
    estimate_disk_diameter,
)

#: Mark labels: a disk is here / this detection is wrong.
DISK, NOT_DISK = 1, 0

STEPS = 20
#: Never stop before this: stopping as soon as the marks pass (often ~8 steps)
#: generalizes less to unmarked patterns.
MIN_STEPS = 20
MAX_STEPS = 300
MAX_SECONDS = 3.0
LEARNING_RATE = 1e-3
REPLAY_WEIGHT = 0.2
DISTILL_WEIGHT = 1.0
OFFSET_WEIGHT = 1.0
#: Unmarked pixels are held to the base output with this extra weight on its peaks:
#: a plain mean over ~50k background pixels lets every unmarked detection drift.
PEAK_HOLD = 20.0
HEATMAP_SIGMA = 1.2
#: Replay frames used during teaching, and the held-out ones :func:`original_f1` scores.
REPLAY, GAUGE = slice(0, 24), slice(24, 32)


class Working:
    """A stack of patterns at the model's working resolution, with the decode
    parameters detection would use for them."""

    def __init__(
        self,
        patterns: np.ndarray,
        model: SpotNet,
        *,
        spot_radius: float | None = None,
        min_spacing: int = 4,
        background_sigma: float | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        patterns = np.stack([np.asarray(p, np.float32) for p in patterns])
        spot_diameter = 2.0 * spot_radius if spot_radius else None
        first = patterns[0]
        self.factor = canonical_scale_factor(spot_diameter or estimate_disk_diameter(first))
        working_diameter = _working_diameter(first, self.factor, spot_diameter)
        window = max(2, int(round(min_spacing * self.factor)))
        self.min_spacing, self.background_sigma = _resolve_window_and_background(
            window, background_sigma, working_diameter
        )
        has_diameter = bool(np.isfinite(working_diameter) and working_diameter > 0)
        self.mask_radius = 0.5 * (working_diameter if has_diameter else CANONICAL_DIAMETER)
        self.threshold = float(threshold)
        self.shape = patterns.shape[1:]
        self.x = _network_input(
            torch.as_tensor(patterns),
            self.factor,
            self.background_sigma,
            model.levels,
            getattr(model, "mad_floor", None),
        )

    def peaks(self, outputs) -> list[np.ndarray]:
        """``(N, 3)`` ``[row, col, confidence]`` per pattern, in detector pixels."""
        found, batch_index = _decode(
            outputs, self.threshold, self.min_spacing, mask_radius=self.mask_radius
        )
        found = _to_pattern_coordinates(found, self.shape, self.factor).cpu().numpy()
        batch_index = batch_index.cpu().numpy()
        return [found[batch_index == b, :3] for b in range(outputs[0].shape[0])]


@torch.no_grad()
def mark_targets(base, working: Working, marks: list[np.ndarray], disk_radius: float):
    """Heatmap target and weight from per-pattern marks ``[row, col, label]``.

    A "disk here" mark within ~2 px of an existing detection confirms it. Otherwise it
    is snapped to the base heatmap's maximum within ~1.5 px of the click, never onto
    an existing detection, since a missed disk often sits beside a found one. A "not a
    disk" mark sets a small disk of the heatmap to zero. Every mark weighs the same
    however many pixels it covers.
    """
    device = next(base.parameters()).device
    x = working.x.to(device)
    outputs = base(x)
    heatmap_base, offset_base = outputs[0], outputs[1]
    _, _, height, width = x.shape
    f = working.factor
    target = torch.zeros_like(heatmap_base)
    weight = torch.zeros_like(heatmap_base)
    rows = torch.arange(height, device=device).view(height, 1).float()
    cols = torch.arange(width, device=device).view(1, width).float()
    reject_radius = max(2.0, 0.4 * disk_radius * f)
    probability = torch.sigmoid(heatmap_base)
    window = working.min_spacing
    pooled = F.max_pool2d(probability, 2 * window + 1, stride=1, padding=window)
    found = ((probability == pooled) & (probability >= working.threshold)).squeeze(1).nonzero()
    for b, pattern_marks in enumerate(marks):
        detections = found[found[:, 0] == b][:, 1:].float()
        for mark_row, mark_col, label in np.asarray(pattern_marks, np.float64).reshape(-1, 3):
            center_row, center_col = mark_row * f, mark_col * f
            if int(label) == NOT_DISK:
                region = ((rows - center_row) ** 2 + (cols - center_col) ** 2) <= reject_radius**2
                target[b, 0][region] = 0.0
                weight[b, 0][region] += 1.0 / region.sum().clamp_min(1)
                continue
            peak_row, peak_col = int(round(center_row)), int(round(center_col))
            distance = (
                torch.hypot(detections[:, 0] - center_row, detections[:, 1] - center_col)
                if len(detections)
                else None
            )
            if distance is not None and float(distance.min()) <= 2.0 * f:
                peak_row, peak_col = (int(v) for v in detections[int(torch.argmin(distance))])
            else:
                reach = max(1, int(round(1.5 * f)))
                row0, row1 = max(0, peak_row - reach), min(height, peak_row + reach + 1)
                col0, col1 = max(0, peak_col - reach), min(width, peak_col + reach + 1)
                patch = heatmap_base[b, 0, row0:row1, col0:col1].clone()
                patch_rows = torch.arange(row0, row1, device=device).view(-1, 1).float()
                patch_cols = torch.arange(col0, col1, device=device).view(1, -1).float()
                for detection_row, detection_col in detections.tolist():
                    near = (patch_rows - detection_row) ** 2 + (patch_cols - detection_col) ** 2
                    patch[near <= (2.5 * f) ** 2] = -1e9
                k = int(torch.argmax(patch))
                if float(patch.flatten()[k]) > -1e8:
                    peak_row, peak_col = row0 + k // patch.shape[1], col0 + k % patch.shape[1]
            peak_row = min(max(peak_row, 0), height - 1)
            peak_col = min(max(peak_col, 0), width - 1)
            blob = torch.exp(
                -((rows - peak_row) ** 2 + (cols - peak_col) ** 2) / (2 * HEATMAP_SIGMA**2)
            )
            target[b, 0] = torch.maximum(target[b, 0], blob)
            target[b, 0, peak_row, peak_col] = 1.0
            region = ((rows - peak_row) ** 2 + (cols - peak_col) ** 2) <= 9.0
            weight[b, 0][region] += 1.0 / region.sum()
    return {
        "x": x,
        "target": target,
        "weight": weight,
        "heatmap_base": heatmap_base,
        "offset_base": offset_base,
        "mask_base": outputs[2] if len(outputs) > 2 else None,
    }


def _mark_loss(logit, target, weight):
    if float(weight.sum()) <= 0:
        return logit.sum() * 0
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (loss * weight).sum() / weight.sum()


def _hold_loss(logit, heatmap_base, weight):
    """KL divergence to the base model on unmarked pixels, weighted toward its peaks."""
    base = torch.sigmoid(heatmap_base)
    free = (weight <= 0).float() * (1.0 + PEAK_HOLD * base)
    cross = F.binary_cross_entropy_with_logits(logit, base, reduction="none")
    entropy = F.binary_cross_entropy(base.clamp(1e-6, 1 - 1e-6), base, reduction="none")
    return ((cross - entropy) * free).sum() / free.sum().clamp_min(1)


def _offset_hold(offset, offset_base, heatmap_base):
    near = (torch.sigmoid(heatmap_base) > 0.1).float()
    return ((offset - offset_base).abs() * near).sum() / (2 * near.sum()).clamp_min(1)


def _mask_hold(mask, mask_base, heatmap_base):
    """The mask head held to the base model near its detections: it places each disk."""
    near = (torch.sigmoid(heatmap_base) > 0.1).float()
    difference = (torch.sigmoid(mask) - torch.sigmoid(mask_base)).abs()
    return (difference * near).sum() / near.sum().clamp_min(1)


def _focal(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    positive = (target >= 0.999).float()
    negative = 1.0 - positive
    loss = (
        -((1 - p) ** 2) * torch.log(p) * positive
        - (p**2) * torch.log(1 - p) * (1.0 - target) ** 4 * negative
    )
    return loss.sum() / positive.sum().clamp_min(1.0)


def _dihedral(t, k, offset=False):
    """One of the 8 symmetries of the square; ``offset`` also turns the ``(d_row,
    d_col)`` channels with the image."""
    if k & 1:
        t = t.transpose(-1, -2)
        if offset:
            t = t.flip(1)
    if k & 2:
        t = t.flip(-2)
        if offset:
            t = torch.cat([-t[:, :1], t[:, 1:]], 1)
    if k & 4:
        t = t.flip(-1)
        if offset:
            t = torch.cat([t[:, :1], -t[:, 1:]], 1)
    return t


def load_replay(path: str | Path) -> dict[str, torch.Tensor]:
    """The replay buffer: 32 crops of the synthetic frames DE SpotNet was trained on
    (disk radius 3-20 px, 128 px crops). The first 24 are replayed while teaching,
    the last 8 are held out for :func:`original_f1`.

    Parameters
    ----------
    path : str or Path
        An ``.npz`` with ``x`` (float16), ``heatmap`` (uint8, 0-255), ``offset``
        (float16, two channels) and ``mask`` (bool), each 32 x 128 x 128.

    Returns
    -------
    dict of torch.Tensor
        ``x``, ``heatmap``, ``offset`` and ``mask``.
    """
    with np.load(path) as bank:
        return {
            "x": torch.from_numpy(bank["x"].astype(np.float32))[:, None],
            "heatmap": torch.from_numpy(bank["heatmap"].astype(np.float32) / 255.0)[:, None],
            "offset": torch.from_numpy(bank["offset"].astype(np.float32)),
            "mask": torch.from_numpy(bank["mask"].astype(np.float32)),
        }


def _replay_loss(model, bank, rng, device, count=2):
    index = rng.integers(REPLAY.start, REPLAY.stop, count)
    x = bank["x"][index].to(device)
    heatmap = bank["heatmap"][index].to(device)
    offset = bank["offset"][index].to(device)
    mask = bank["mask"][index].to(device)[:, None]
    logit, predicted = model(x)[:2]
    offset_loss = F.l1_loss(predicted * mask, offset * mask, reduction="sum")
    return _focal(logit, heatmap) + offset_loss / mask.sum().clamp_min(1)


@torch.no_grad()
def original_f1(model, bank, threshold: float = DEFAULT_THRESHOLD) -> float:
    """F1 on the held-out replay frames (matched within 3 px).

    The built-in detector scores about 0.86 here; a fall measures how far teaching
    pulled the model away from being a general disk detector.
    """
    device = next(model.parameters()).device
    model.eval()
    x = bank["x"][GAUGE].to(device)
    outputs = model(x)[:2]
    found, batch_index = _decode(outputs, threshold, 2)
    found, batch_index = found.cpu().numpy(), batch_index.cpu().numpy()
    true_positives = false_positives = false_negatives = 0
    for k, i in enumerate(range(GAUGE.start, GAUGE.stop)):
        mask = bank["mask"][i].numpy()
        offset = bank["offset"][i].numpy()
        rows, cols = np.nonzero(mask)
        centers = np.stack([rows + offset[0][rows, cols], cols + offset[1][rows, cols]], 1)
        truth = []
        for center in centers:
            if not truth or min(np.hypot(*(center - t)) for t in truth) > 2:
                truth.append(center)
        truth = np.asarray(truth).reshape(-1, 2)
        predicted = found[batch_index == k, :2]
        matched = _greedy_matches(truth, predicted, 3.0)
        true_positives += matched
        false_positives += len(predicted) - matched
        false_negatives += len(truth) - matched
    return float(
        2 * true_positives / max(2 * true_positives + false_positives + false_negatives, 1)
    )


def _greedy_matches(a, b, tolerance) -> int:
    if not len(a) or not len(b):
        return 0
    distance = np.linalg.norm(a[:, None] - b[None], axis=2)
    used_a, used_b, count = set(), set(), 0
    for i, j in zip(*np.unravel_index(np.argsort(distance, axis=None), distance.shape)):
        if distance[i, j] > tolerance:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        count += 1
    return count


@torch.no_grad()
def marks_learned(model, working: Working, marks: list[np.ndarray], disk_radius: float):
    """Per mark, whether ``model`` now agrees with it: no detection within ~0.4 disk
    radii of a "not a disk" mark, a detection within ~0.5 radii of a "disk here" mark.
    """
    found = working.peaks(model(working.x.to(next(model.parameters()).device)))
    agreed = []
    for pattern_marks, peaks in zip(marks, found):
        for mark_row, mark_col, label in np.asarray(pattern_marks, np.float64).reshape(-1, 3):
            distance = (
                np.hypot(peaks[:, 0] - mark_row, peaks[:, 1] - mark_col).min()
                if len(peaks)
                else np.inf
            )
            if int(label) == NOT_DISK:
                agreed.append(distance > max(2.0, 0.4 * disk_radius))
            else:
                agreed.append(distance <= max(2.5, 0.5 * disk_radius))
    return np.asarray(agreed, bool)


def teach_spot_net(
    base: SpotNet,
    patterns,
    marks: list[np.ndarray],
    *,
    spot_radius: float | None = None,
    min_spacing: int = 4,
    background_sigma: float | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    replay: str | Path | dict | None = None,
    until_learned: bool = True,
    steps: int = STEPS,
    max_steps: int = MAX_STEPS,
    max_seconds: float = MAX_SECONDS,
    learning_rate: float = LEARNING_RATE,
    replay_weight: float = REPLAY_WEIGHT,
    seed: int = 0,
) -> tuple[SpotNet, dict]:
    """Fine-tune a copy of ``base`` on marks made on ``patterns``.

    Parameters
    ----------
    base : SpotNet
        The detector the marks were made against; it is not modified.
    patterns : sequence of np.ndarray
        The ``(H, W)`` marked patterns.
    marks : list of np.ndarray
        Per pattern, ``(n, 3)`` ``[row, col, label]`` in detector pixels, with label
        :data:`DISK` or :data:`NOT_DISK`.
    spot_radius, min_spacing, background_sigma, threshold
        The detection parameters in use, so the copy is trained on exactly the input
        detection will give it. ``spot_radius`` ``None`` estimates it.
    replay : str, Path or dict, optional
        The replay buffer (see :func:`load_replay`). ``None`` teaches without it, and
        the report's ``original_f1`` values are then ``None``.
    until_learned : bool, default=True
        Step until every mark is learned (checked every other step, never before
        :data:`MIN_STEPS`) or ``max_seconds`` / ``max_steps`` run out; otherwise
        exactly ``steps``.

    Returns
    -------
    tuple
        The taught model (on the base model's device, in evaluation mode) and a
        report: seconds, steps, why it stopped, marks learned, and ``original_f1``
        before and after.
    """
    started = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = next(base.parameters()).device
    patterns = [np.asarray(p, np.float32) for p in patterns]
    disk_radius = float(spot_radius or counts_radius(patterns[0], None))
    bank = load_replay(replay) if isinstance(replay, (str, Path)) else replay
    base = base.eval()
    model = copy.deepcopy(base).eval()  # BatchNorm statistics stay frozen
    working = Working(
        patterns,
        base,
        spot_radius=spot_radius,
        min_spacing=min_spacing,
        background_sigma=background_sigma,
        threshold=threshold,
    )
    labels = mark_targets(base, working, marks, disk_radius)
    before = original_f1(base, bank) if bank is not None else None
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    limit = int(max_steps) if until_learned else int(steps)
    stopped, taken, learned = "steps", 0, None
    budget_started = time.perf_counter()
    with torch.enable_grad():
        for step in range(limit):
            if until_learned and step >= MIN_STEPS and step % 2 == 0:
                model.eval()
                learned = marks_learned(model, working, marks, disk_radius)
                if learned.all():
                    stopped = "learned"
                    break
                if time.perf_counter() - budget_started > max_seconds:
                    stopped = "time"
                    break
            taken = step + 1
            k = int(rng.integers(0, 8))
            outputs = model(_dihedral(labels["x"], k))
            heatmap_base = _dihedral(labels["heatmap_base"], k)
            weight = _dihedral(labels["weight"], k)
            loss = (
                _mark_loss(outputs[0], _dihedral(labels["target"], k), weight)
                + DISTILL_WEIGHT * _hold_loss(outputs[0], heatmap_base, weight)
                + OFFSET_WEIGHT
                * _offset_hold(
                    outputs[1], _dihedral(labels["offset_base"], k, offset=True), heatmap_base
                )
            )
            if labels["mask_base"] is not None:
                loss = loss + OFFSET_WEIGHT * _mask_hold(
                    outputs[2], _dihedral(labels["mask_base"], k), heatmap_base
                )
            if bank is not None and replay_weight:
                loss = loss + replay_weight * _replay_loss(model, bank, rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0:
                # The first optimizer step imports parts of torch on first use (seconds
                # in a fresh process); the time budget is for training, so it starts here.
                budget_started = time.perf_counter()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if stopped != "learned":
        learned = marks_learned(model, working, marks, disk_radius)
        stopped = "learned" if learned.all() else stopped
    after = original_f1(model, bank) if bank is not None else None
    all_marks = np.concatenate([np.asarray(m, np.float64).reshape(-1, 3) for m in marks])
    kinds = all_marks[:, 2].astype(int)
    report = {
        "seconds": time.perf_counter() - started,
        "steps": int(taken),
        "stopped": stopped,
        "patterns": len(patterns),
        "disk_marks": int((kinds == DISK).sum()),
        "not_disk_marks": int((kinds == NOT_DISK).sum()),
        "marks_learned": int(learned.sum()),
        "marks_total": int(len(learned)),
        "original_f1_base": before,
        "original_f1": after,
        "replay": bank is not None,
        "learning_rate": float(learning_rate),
        "replay_weight": float(replay_weight) if bank is not None else 0.0,
    }
    return model, report


def preferred_input(model: SpotNet, *, size: int = 48, steps: int = 200, seed: int = 0):
    """What the model treats as an ideal disk: the input maximizing its center heatmap
    logit, by gradient ascent from near zero with an L2 and a total-variation penalty
    so the answer is a smooth picture rather than noise. At the working resolution,
    where every disk is about 9 px across; the icons of the models are this image."""
    device = next(model.parameters()).device
    model = model.eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.enable_grad():
        x = (0.01 * torch.randn(1, 1, size, size, generator=generator)).to(device)
        x.requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.05)
        for _ in range(int(steps)):
            logit = model(x)[0]
            variation = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (
                x[..., :, 1:] - x[..., :, :-1]
            ).abs().mean()
            loss = -logit[0, 0, size // 2, size // 2] + 0.02 * (x**2).mean() + 0.3 * variation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return x.detach().cpu().numpy()[0, 0]
