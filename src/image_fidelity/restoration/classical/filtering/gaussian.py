"""Gaussian smoothing filter: a simple linear low-pass denoiser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ..base import Restorer


@dataclass
class GaussianFilter(Restorer):
    """Isotropic Gaussian blur used as a (crude) denoiser: trades noise for loss of detail."""

    sigma: float = 1.0

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        if observed.ndim == 2:
            return gaussian_filter(observed, sigma=self.sigma, mode="reflect")
        sigmas = (self.sigma, self.sigma) + (0,) * (observed.ndim - 2)
        return gaussian_filter(observed, sigma=sigmas, mode="reflect")
