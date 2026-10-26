"""Common interface and shared utilities for classical (model-based) restoration methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Restorer(ABC):
    """A single restoration method, called on a degraded observation."""

    @abstractmethod
    def __call__(self, observed: np.ndarray) -> np.ndarray:
        """Restore ``observed`` and return the estimated clean image."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


def psf_to_otf(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Zero-pads ``kernel`` to ``shape``, wraps it so its center sits at the origin, and FFTs it (psf2otf).

    This is the standard trick to build the frequency response ("optical
    transfer function") of a circulant convolution operator from a small,
    centered point-spread function, for FFT-based deconvolution.
    """
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(padded)
