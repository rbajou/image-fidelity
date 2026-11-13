"""Compact 2D U-Net for image-to-image prediction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda inplace=True: nn.LeakyReLU(0.1, inplace=inplace),
    "silu": nn.SiLU,
    "gelu": lambda inplace=True: nn.GELU(),
}


def _norm(kind: str | None, channels: int) -> nn.Module:
    if kind in (None, "none"):
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"Unsupported normalization {kind!r}")


def _activation(kind: str) -> nn.Module:
    try:
        return _ACTIVATIONS[kind]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {kind!r}; choose from {sorted(_ACTIVATIONS)}") from exc


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, activation: str, normalization: str | None):
        bias = normalization in (None, "none")
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            _norm(normalization, out_channels),
            _activation(activation),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            _norm(normalization, out_channels),
            _activation(activation),
        )


class UNet(nn.Module):
    """U-Net with ``depth`` downsamplings; channels double per level.

    With ``base_channels=32`` and ``depth=3`` the widths are 32 -> 64 -> 128
    (encoder) and 256 (bottleneck). With ``residual=True`` the network predicts
    the degradation and returns ``x - net(x)`` (requires
    ``in_channels == out_channels``), i.e. residual learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        activation: str = "relu",
        normalization: str | None = "batch",
        residual: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if residual and in_channels != out_channels:
            raise ValueError("residual learning requires in_channels == out_channels")
        self.depth = depth
        self.residual = residual
        widths = [base_channels * 2**level for level in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.encoders.append(DoubleConv(previous, width, activation, normalization))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(widths[-2], widths[-1], activation, normalization)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(width * 2, width, 2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width, activation, normalization))
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        if residual:
            # Start as the identity mapping.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @property
    def divisor(self) -> int:
        """Spatial sizes must be multiples of this value."""
        return 2**self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.divisor
        pad_w = (-width) % self.divisor
        inputs = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x

        skips = []
        h = inputs
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            h = decoder(torch.cat([upsample(h), skip], dim=1))
        out = self.head(h)[..., :height, :width]
        return x - out if self.residual else out
