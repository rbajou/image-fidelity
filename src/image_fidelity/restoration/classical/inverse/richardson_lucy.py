"""Richardson-Lucy deconvolution: iterative, physically-motivated method for Poisson (photon-limited) noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve

from ..base import Restorer


@dataclass
class RichardsonLucyDeconvolution(Restorer):
    """Iteratively deconvolves a known PSF ``kernel`` assuming Poisson noise (multiplicative update rule)."""

    kernel: np.ndarray
    n_iter: int = 30
    eps: float = 1e-12

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        if observed.ndim == 2:
            return self._deconvolve_channel(observed)
        return np.stack(
            [self._deconvolve_channel(observed[..., c]) for c in range(observed.shape[-1])],
            axis=-1,
        )

    def _deconvolve_channel(self, channel: np.ndarray) -> np.ndarray:
        channel = np.clip(channel.astype(np.float64), self.eps, None)
        flipped_kernel = self.kernel[::-1, ::-1]
        estimate = channel.copy()

        for _ in range(self.n_iter):
            reblurred = convolve(estimate, self.kernel, mode="reflect") + self.eps
            ratio = channel / reblurred
            estimate = estimate * convolve(ratio, flipped_kernel, mode="reflect")

        return estimate.astype(channel.dtype, copy=False)
