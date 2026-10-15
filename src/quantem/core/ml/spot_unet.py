"""Small U-Net that detects diffraction spots, with a heatmap head, a subpixel-offset head
and, optionally, a disk-coverage head.

The architecture must stay bit-faithful to the trained checkpoints, so the layer
layout and names here are not to be edited. The trained weights are distributed
separately through em-database (see :mod:`quantem.diffraction.neural_weights`) under
their own licence; this file only defines the network.

Output heads, all at input resolution:

- heatmap: one channel of spot-presence logits.
- offset: two channels, the subpixel ``(d_row, d_col)`` of the spot centre from the
  pixel that carries the heatmap peak.
- mask (``mask_head=True``): one channel of logits for how much of each pixel the
  disk's aperture covers. Its centroid is the aperture's centre, however the
  intensity is distributed inside the disk.

The input is ``(B, C, H, W)`` with ``C`` stacked frames (``C = 1`` for one pattern).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SpotUNet(nn.Module):
    """Compact U-Net with a deliberately small receptive field.

    ``levels`` is the number of downsampling steps. The receptive field must cover
    roughly one disk and its immediate surround, not the whole periodic lattice, so
    the network learns a local spot detector rather than the lattice symmetry and
    keeps working on defects, grain boundaries and isolated spots.

    Parameters
    ----------
    in_ch : int, default=1
        Number of stacked input frames.
    base : int, default=16
        Channel count of the first encoder level; each level doubles it.
    levels : int, default=2
        Number of downsampling steps.
    mask_head : bool, default=False
        Add the disk-coverage head; :meth:`forward` then returns three outputs.
    """

    def __init__(self, in_ch: int = 1, base: int = 16, levels: int = 2, mask_head: bool = False):
        super().__init__()
        self.levels = levels
        channels = [base * (2**i) for i in range(levels + 1)]
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList()
        previous = in_ch
        for count in channels:
            self.enc.append(_conv_block(previous, count))
            previous = count
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2))
            self.dec.append(_conv_block(channels[i - 1] * 2, channels[i - 1]))
        self.head_hm = nn.Conv2d(base, 1, 1)
        self.head_off = nn.Conv2d(base, 2, 1)
        nn.init.constant_(self.head_hm.bias, -4.0)
        self.head_mask = nn.Conv2d(base, 1, 1) if mask_head else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Heatmap logits, offsets and, with a mask head, coverage logits."""
        features = []
        h = x
        for i, encoder in enumerate(self.enc):
            h = encoder(h if i == 0 else self.pool(h))
            features.append(h)
        d = features[-1]
        for j, (up, decoder) in enumerate(zip(self.up, self.dec)):
            skip = features[self.levels - 1 - j]
            d = decoder(torch.cat([up(d), skip], 1))
        heatmap = self.head_hm(d)
        offset = torch.tanh(self.head_off(d)) * 1.6
        if self.head_mask is not None:
            return heatmap, offset, self.head_mask(d)
        return heatmap, offset
