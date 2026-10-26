"""Level 1 - image quality: pixel-wise fidelity between a restored image and its reference (MSE, PSNR, SSIM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from .base import Metric


def mse(restored: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared error between `restored` and `reference`."""
    restored = np.asarray(restored, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.mean((restored - reference) ** 2))


def mae(restored: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute error between `restored` and `reference`."""
    restored = np.asarray(restored, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.mean(np.abs(restored - reference)))


def _image_pair(restored: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    restored = np.asarray(restored, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if restored.shape != reference.shape:
        raise ValueError("`restored` and `reference` must have the same shape")
    if restored.size == 0:
        raise ValueError("images must not be empty")
    return restored, reference


def nrmse(restored: np.ndarray, reference: np.ndarray) -> float:
    """Root mean squared error normalized by the reference intensity range."""
    restored, reference = _image_pair(restored, reference)
    error = float(np.sqrt(np.mean((restored - reference) ** 2)))
    data_range = float(np.ptp(reference))
    if data_range == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / data_range


def gradient_error(restored: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 error between image gradients; lower is better."""
    restored, reference = _image_pair(restored, reference)
    axes = (0, 1) if restored.ndim >= 2 else (0,)
    restored_gradient = np.gradient(restored, axis=axes)
    reference_gradient = np.gradient(reference, axis=axes)
    error = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(restored_gradient, reference_gradient))))
    reference_norm = float(np.sqrt(sum(np.sum(g**2) for g in reference_gradient)))
    if reference_norm == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / reference_norm


def bias(restored: np.ndarray, reference: np.ndarray) -> float:
    """Mean signed intensity error (restored minus reference)."""
    restored, reference = _image_pair(restored, reference)
    return float(np.mean(restored - reference))


def flux_ratio(restored: np.ndarray, reference: np.ndarray) -> float:
    """Ratio of total restored intensity to reference intensity."""
    restored, reference = _image_pair(restored, reference)
    reference_flux = float(np.sum(reference))
    restored_flux = float(np.sum(restored))
    if reference_flux == 0.0:
        return 1.0 if restored_flux == 0.0 else float("inf")
    return restored_flux / reference_flux


def psnr(restored: np.ndarray, reference: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB); infinite when `restored` exactly matches `reference`."""
    error = mse(restored, reference)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / error))


def ssim(
    restored: np.ndarray,
    reference: np.ndarray,
    data_range: float = 1.0,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Mean structural similarity index over a sliding window (Wang et al., 2004).

    Multi-channel images (``H, W, C``) are scored per channel and averaged.
    """
    restored = np.asarray(restored, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if restored.shape != reference.shape:
        raise ValueError("`restored` and `reference` must have the same shape")
    if restored.ndim == 3:
        channel_scores = [
            _ssim_2d(restored[..., c], reference[..., c], data_range, window_size, k1, k2)
            for c in range(restored.shape[-1])
        ]
        return float(np.mean(channel_scores))
    return _ssim_2d(restored, reference, data_range, window_size, k1, k2)


def _ssim_2d(x: np.ndarray, y: np.ndarray, data_range: float, window_size: int, k1: float, k2: float) -> float:
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = uniform_filter(x, size=window_size)
    mu_y = uniform_filter(y, size=window_size)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    sigma_x_sq = uniform_filter(x * x, size=window_size) - mu_x_sq
    sigma_y_sq = uniform_filter(y * y, size=window_size) - mu_y_sq
    sigma_xy = uniform_filter(x * y, size=window_size) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(numerator / denominator))


@dataclass
class MSE(Metric):
    """Mean squared error, as a reusable `Metric`."""

    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> float:
        return mse(restored, reference)


@dataclass
class PSNR(Metric):
    """Peak signal-to-noise ratio, as a reusable `Metric`."""

    data_range: float = 1.0

    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> float:
        return psnr(restored, reference, data_range=self.data_range)


@dataclass
class SSIM(Metric):
    """Structural similarity index, as a reusable `Metric`."""

    data_range: float = 1.0
    window_size: int = 7
    k1: float = 0.01
    k2: float = 0.03

    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> float:
        return ssim(restored, reference, self.data_range, self.window_size, self.k1, self.k2)
