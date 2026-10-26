"""Median filter: a nonlinear, edge-preserving denoiser especially effective against impulsive noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from ..base import Restorer


@dataclass
class MedianFilter(Restorer):
    """Replaces each pixel with the median over a ``size`` x ``size`` spatial neighborhood (per channel)."""

    size: int = 3

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        footprint_size = (self.size, self.size) + (1,) * (observed.ndim - 2)
        return median_filter(observed, size=footprint_size, mode="reflect")
