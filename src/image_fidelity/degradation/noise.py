"""Noise degradations: photon shot noise, additive gaussian, impulsive, and sensor readout noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .base import Degradation, as_rng


@dataclass
class PoissonNoise(Degradation):
    """Photon shot noise: signal-dependent noise from counting a finite number of photons.

    ``image`` is assumed to be a float array in ``[0, 1]``, interpreted as a
    normalized photon flux. It is scaled to ``photon_count`` photons, a
    Poisson draw is taken, and the result is rescaled back to ``[0, 1]``.
    """

    photon_count: float = 100.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rng = as_rng(self.seed)
        photons = np.clip(image, 0.0, None) * self.photon_count
        noisy = rng.poisson(photons).astype(np.float64) / self.photon_count
        return noisy.astype(image.dtype, copy=False)


@dataclass
class GaussianNoise(Degradation):
    """Additive white Gaussian noise: ``image + N(mean, sigma^2)``."""

    sigma: float = 0.02
    mean: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rng = as_rng(self.seed)
        noise = rng.normal(self.mean, self.sigma, size=image.shape)
        return (image + noise).astype(image.dtype, copy=False)


@dataclass
class SaltPepperNoise(Degradation):
    """Impulsive noise: a random fraction of pixels are forced to an extreme value.

    ``amount`` is the fraction of pixels replaced overall; ``salt_vs_pepper``
    is the fraction of those replaced pixels set to ``salt_value`` (the rest
    are set to ``pepper_value``).
    """

    amount: float = 0.05
    salt_vs_pepper: float = 0.5
    salt_value: float = 1.0
    pepper_value: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rng = as_rng(self.seed)
        out = image.copy()
        mask = rng.random(image.shape[:2]) < self.amount
        salt_mask = mask & (rng.random(image.shape[:2]) < self.salt_vs_pepper)
        pepper_mask = mask & ~salt_mask
        out[salt_mask] = self.salt_value
        out[pepper_mask] = self.pepper_value
        return out


@dataclass
class ReadoutNoise(Degradation):
    """Additive sensor readout noise: signal-independent Gaussian noise plus a fixed bias.

    Unlike `GaussianNoise`, this models electronic read noise from the sensor
    itself (roughly constant regardless of exposure/photon count).
    """

    sigma: float = 0.01
    bias: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rng = as_rng(self.seed)
        noise = rng.normal(0.0, self.sigma, size=image.shape)
        return (image + self.bias + noise).astype(image.dtype, copy=False)
