"""Patch extraction utilities, e.g. splitting a 1024x1024 image into 256x256 patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .base import Sample

_IntOrPair = Union[int, Tuple[int, int]]


def _as_pair(value: _IntOrPair) -> Tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)  # type: ignore[return-value]


def _padded_size(size: int, patch: int, stride: int) -> int:
    """Smallest size >= size (and >= patch) evenly tileable by `patch` at the given `stride`."""
    size = max(size, patch)
    remainder = (size - patch) % stride
    return size if remainder == 0 else size + (stride - remainder)


@dataclass
class PatchExtractor:
    """Extracts fixed-size patches on a regular grid, e.g. 1024x1024 -> N x 256x256.

    ``stride`` defaults to ``patch_size`` (non-overlapping tiling). When an
    array isn't evenly tileable it is padded (``pad_mode``, a `numpy.pad`
    mode) up to the next valid size; set ``pad_mode=None`` to raise instead.
    """

    patch_size: _IntOrPair = 256
    stride: Optional[_IntOrPair] = None
    pad_mode: Optional[str] = "reflect"

    def __post_init__(self) -> None:
        self.patch_size = _as_pair(self.patch_size)
        self.stride = _as_pair(self.stride) if self.stride is not None else self.patch_size

    def grid_shape(self, height: int, width: int) -> Tuple[int, int]:
        """Number of (rows, cols) of patches produced for a given image size."""
        ph, pw = self.patch_size
        sh, sw = self.stride
        padded_h = _padded_size(height, ph, sh)
        padded_w = _padded_size(width, pw, sw)
        return (padded_h - ph) // sh + 1, (padded_w - pw) // sw + 1

    def _extract(self, array: np.ndarray, pad_mode: Optional[str]) -> np.ndarray:
        ph, pw = self.patch_size
        sh, sw = self.stride
        h, w = array.shape[:2]
        padded_h = _padded_size(h, ph, sh)
        padded_w = _padded_size(w, pw, sw)
        pad_h, pad_w = padded_h - h, padded_w - w

        if pad_h or pad_w:
            if pad_mode is None:
                raise ValueError(
                    f"Array of shape {(h, w)} is not evenly tileable with patch_size={self.patch_size} "
                    f"and stride={self.stride}; set pad_mode to auto-pad."
                )
            pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (array.ndim - 2)
            array = np.pad(array, pad_width, mode=pad_mode)

        patches = [
            array[y : y + ph, x : x + pw, ...]
            for y in range(0, padded_h - ph + 1, sh)
            for x in range(0, padded_w - pw + 1, sw)
        ]
        return np.stack(patches, axis=0)

    def extract(self, array: np.ndarray) -> np.ndarray:
        """Split a single (H, W[, C]) array into an (N, patch_h, patch_w[, C]) array of patches."""
        return self._extract(array, self.pad_mode)

    def extract_pair(
        self, image: np.ndarray, target: np.ndarray, target_pad_mode: Optional[str] = "constant"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract aligned patches from an image and its target (e.g. a mask), on the same grid."""
        return self._extract(image, self.pad_mode), self._extract(target, target_pad_mode)

    def extract_sample(self, sample: Sample, target_pad_mode: Optional[str] = "constant") -> List[Sample]:
        """Split a `Sample` into a list of patch-level `Sample`s, keeping image/target aligned."""
        image_patches = self.extract(sample.image)
        target_patches = (
            self._extract(sample.target, target_pad_mode) if sample.target is not None else None
        )
        return [
            Sample(
                image=image_patch,
                target=target_patches[i] if target_patches is not None else None,
                metadata={**sample.metadata, "patch_index": i},
            )
            for i, image_patch in enumerate(image_patches)
        ]
