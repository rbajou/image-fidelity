"""Blur degradations: convolution with a gaussian kernel, a motion PSF, or a disk (defocus) PSF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from .base import Degradation


def convolve_per_channel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``kernel`` over the spatial dimensions, independently per channel if any."""
    if image.ndim == 2:
        return convolve(image, kernel, mode="reflect")
    return np.stack(
        [convolve(image[..., c], kernel, mode="reflect") for c in range(image.shape[-1])],
        axis=-1,
    )


def gaussian_kernel(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    """Isotropic Gaussian PSF as an explicit 2D kernel, for solvers that need the operator explicitly."""
    r = radius if radius is not None else max(1, int(np.ceil(3 * sigma)))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def motion_kernel(theta: float, length: float) -> np.ndarray:
    """Line PSF of amplitude ``length`` (pixels) oriented at angle ``theta`` (degrees)."""
    size = max(3, int(np.ceil(length)) | 1)  # smallest odd size >= length, >= 3
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    dx, dy = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    for t in np.linspace(-length / 2, length / 2, num=max(int(length * 4), 2)):
        x, y = center + t * dx, center + t * dy
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < size and 0 <= iy < size:
            kernel[iy, ix] += 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    return kernel / kernel.sum()


def disk_kernel(radius: float) -> np.ndarray:
    """Uniform disk PSF: a point spreads into a circular blob of the given radius (pixels)."""
    r = max(1, int(np.ceil(radius)))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x**2 + y**2) <= radius**2
    kernel = mask.astype(np.float64)
    return kernel / kernel.sum()


@dataclass
class GaussianBlur(Degradation):
    """Convolution with an isotropic Gaussian kernel of standard deviation ``sigma``."""

    sigma: float = 1.2

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return gaussian_filter(image, sigma=self.sigma, mode="reflect")
        sigmas = (self.sigma, self.sigma) + (0,) * (image.ndim - 2)
        return gaussian_filter(image, sigma=sigmas, mode="reflect")


@dataclass
class MotionBlur(Degradation):
    """Convolution with a linear motion PSF: direction ``theta`` (degrees), amplitude ``length`` (pixels)."""

    theta: float = 0.0
    length: float = 5.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        kernel = motion_kernel(self.theta, self.length)
        return convolve_per_channel(image, kernel)


@dataclass
class DefocusBlur(Degradation):
    """Convolution with a uniform disk PSF of the given ``radius`` (pixels): a point becomes a circular blob."""

    radius: float = 2.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        kernel = disk_kernel(self.radius)
        return convolve_per_channel(image, kernel)
