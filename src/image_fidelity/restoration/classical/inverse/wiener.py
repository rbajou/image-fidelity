"""Wiener deconvolution: the classical linear MMSE deblurring filter for a known PSF and noise level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Restorer, psf_to_otf


@dataclass
class WienerDeconvolution(Restorer):
    """Deconvolves a known PSF ``kernel`` assuming white noise, balancing sharpness against noise (``balance``).

    ``balance`` approximates the noise-to-signal power ratio: near 0 favors
    perfect inversion (amplifies noise), large values favor a smoother, more
    stable (but blurrier) estimate.
    """

    kernel: np.ndarray
    balance: float = 0.01

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
        wiener_filter = np.conj(otf) / (np.abs(otf) ** 2 + self.balance)
        restored = np.fft.ifft2(wiener_filter * spectrum).real
        return restored.astype(channel.dtype, copy=False)
