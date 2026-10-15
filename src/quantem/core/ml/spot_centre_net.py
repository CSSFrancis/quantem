"""U-Net that segments one diffraction disk in a small window, for the centre steps.

The network turns a normalised window around a disk into a disk-coverage map; the
disk's centre is that map's centroid (see
:mod:`quantem.diffraction.centre_refinement`). A convolution moves with the disk, so
the centre does too, and a centroid cannot leave the disk the way a regressed offset
can. A sigma head, a linear layer on the mean-pooled bottleneck, predicts the log of
the centre's standard deviation in window pixels.

With ``channels=2`` the second input channel is the window at the disk's Friedel
mirror point, rotated 180 degrees, so the network sees the dynamical fills of both
``g`` and ``-g``.

Layer names are the checkpoints' state-dict keys; do not rename them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    """Replicate-pad a square window to a side divisible by ``multiple`` (the poolings
    must line up); returns the padded window and the leading pad."""
    size = x.shape[-1]
    pad_total = (-size) % multiple
    before, after = pad_total // 2, pad_total - pad_total // 2
    return F.pad(x, (before, after, before, after), mode="replicate"), before


def _convolution(channels_in: int, channels_out: int, depthwise: bool) -> nn.Module:
    if depthwise and channels_in > 1:
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in, bias=False),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
        )
    return nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False)


def _block(channels_in: int, channels_out: int, depthwise: bool) -> nn.Sequential:
    return nn.Sequential(
        _convolution(channels_in, channels_out, depthwise),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
        _convolution(channels_out, channels_out, depthwise),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class SpotCentreNet(nn.Module):
    """U-Net with ``levels`` poolings returning coverage logits and the log sigma.

    Parameters
    ----------
    base : int, default=8
        Channel count of the first encoder level.
    levels : int, default=2
        Number of poolings.
    depthwise : bool, default=False
        Use depthwise-separable convolutions.
    sigma_head : bool, default=True
        Predict the log of the centre's standard deviation.
    channels : int, default=1
        Input channels: 1 for the disk's window, 2 to add its Friedel-mirror window.
    """

    def __init__(
        self,
        base: int = 8,
        levels: int = 2,
        depthwise: bool = False,
        sigma_head: bool = True,
        channels: int = 1,
    ):
        super().__init__()
        self.levels = levels
        widths = [base * 2**level for level in range(levels + 1)]
        self.encoders = nn.ModuleList(
            [
                _block(channels if level == 0 else widths[level - 1], widths[level], depthwise)
                for level in range(levels + 1)
            ]
        )
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[level], widths[level - 1], 2, stride=2)
                for level in range(levels, 0, -1)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                _block(2 * widths[level - 1], widths[level - 1], depthwise)
                for level in range(levels, 0, -1)
            ]
        )
        self.head = nn.Conv2d(base, 1, 1)
        self.sigma_head = nn.Linear(widths[-1], 1) if sigma_head else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Coverage logits ``(B, 1, S, S)`` and log sigma ``(B,)`` (or ``None``)."""
        size = x.shape[-1]
        x, before = _pad_to_multiple(x, 2**self.levels)
        features = []
        hidden = x
        for level, encoder in enumerate(self.encoders):
            hidden = encoder(hidden if level == 0 else F.max_pool2d(hidden, 2))
            features.append(hidden)
        decoded = features[-1]
        for step, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            decoded = decoder(torch.cat([up(decoded), features[self.levels - 1 - step]], 1))
        logits = self.head(decoded)[:, :, before : before + size, before : before + size]
        log_sigma = (
            self.sigma_head(features[-1].mean((2, 3)))[:, 0]
            if self.sigma_head is not None
            else None
        )
        return logits, log_sigma


def input_channels(state_dict: dict) -> int:
    """Input channels of a :class:`SpotCentreNet` checkpoint, from its first convolution."""
    for key in ("encoders.0.0.weight", "encoders.0.0.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise ValueError("no first convolution in this centre network's state dict")
