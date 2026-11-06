"""Sampling/acquisition degradations: resolution loss, pixel binning, and dead/missing pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.ndimage import zoom

from .base import Degradation, as_rng


def _resize(image: np.ndarray, target_shape: tuple[int, int], order: int = 3) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    tgt_h, tgt_w = target_shape
    zoom_factors = (tgt_h / src_h, tgt_w / src_w) + (1.0,) * (image.ndim - 2)
    return zoom(image, zoom_factors, order=order)


@dataclass
class Downsample(Degradation):
    """Reduces spatial resolution by ``factor``, then optionally resizes back (spline interpolation).

    Simulates a lower-resolution acquisition while keeping the output the
    same shape as the input, so it can be paired with the original for
    supervised restoration.
    """

    factor: int = 2
    restore_size: bool = True
    order: int = 3

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small_shape = (max(1, h // self.factor), max(1, w // self.factor))
        small = _resize(image, small_shape, order=self.order)
        if not self.restore_size:
            return small.astype(image.dtype, copy=False)
        return _resize(small, (h, w), order=self.order).astype(image.dtype, copy=False)


@dataclass
class PixelBinning(Degradation):
    """Averages non-overlapping ``factor`` x ``factor`` blocks (hardware pixel binning).

    Unlike `Downsample`, the (optional) upsampling back to the original size
    is nearest-neighbor, reproducing the blocky look of binned sensor readout.
    """

    factor: int = 2
    restore_size: bool = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        factor = self.factor
        crop_h, crop_w = h - h % factor, w - w % factor
        cropped = image[:crop_h, :crop_w, ...]
        new_h, new_w = crop_h // factor, crop_w // factor

        if image.ndim == 2:
            reshaped = cropped.reshape(new_h, factor, new_w, factor)
        else:
            reshaped = cropped.reshape(new_h, factor, new_w, factor, *cropped.shape[2:])
        binned = reshaped.mean(axis=(1, 3))

        if not self.restore_size:
            return binned.astype(image.dtype, copy=False)

        upsampled = np.repeat(np.repeat(binned, factor, axis=0), factor, axis=1)
        pad_h, pad_w = h - upsampled.shape[0], w - upsampled.shape[1]
        if pad_h or pad_w:
            pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
            upsampled = np.pad(upsampled, pad_width, mode="edge")
        return upsampled.astype(image.dtype, copy=False)


@dataclass
class MissingPixels(Degradation):
    """Randomly drops pixels (all channels at once) to ``fill_value``, e.g. dead sensor pixels."""

    probability: float = 0.01
    fill_value: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rng = as_rng(self.seed)
        mask = rng.random(image.shape[:2]) < self.probability
        out = image.copy()
        out[mask] = self.fill_value
        return out
