"""Tikhonov-regularized deconvolution: quadratic penalty on the solution, solved in the Fourier domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Restorer, psf_to_otf


@dataclass
class TikhonovDeconvolution(Restorer):
    """Deconvolves a known PSF ``kernel`` by minimizing ``||k*x - y||^2 + reg_weight * ||x||^2``.

    Solved in closed form via FFT (circulant convolution assumption), which
    is fast but can introduce boundary (wrap-around) artifacts.
    """

    kernel: np.ndarray
    reg_weight: float = 0.01

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        if observed.ndim == 2:
            return self._deconvolve_channel(observed)
        return np.stack(
            [self._deconvolve_channel(observed[..., c]) for c in range(observed.shape[-1])],
            axis=-1,
        )

    def _deconvolve_channel(self, channel: np.ndarray) -> np.ndarray:
        otf = psf_to_otf(self.kernel, channel.shape)
        spectrum = np.fft.fft2(channel)
        deconvolved = (np.conj(otf) * spectrum) / (np.abs(otf) ** 2 + self.reg_weight)
        restored = np.fft.ifft2(deconvolved).real
        return restored.astype(channel.dtype, copy=False)
