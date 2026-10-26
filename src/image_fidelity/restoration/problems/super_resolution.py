"""Super-resolution problem: recover a high-resolution image from a low-resolution observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from image_fidelity.degradations.blur import convolve_per_channel, gaussian_kernel
from image_fidelity.degradations.noise import GaussianNoise
from image_fidelity.degradations.sampling import Downsample

from .base import InverseProblem


@dataclass
class SuperResolutionProblem(InverseProblem):
    """Observation is a low-resolution image: optional anti-aliasing blur, then downsampling by ``factor``.

    The output keeps the original spatial size (upsampled back with spline
    interpolation) so it can be paired directly with the high-resolution
    target for supervised restoration.
    """

    factor: int = 2
    blur_sigma: Optional[float] = 1.0
    noise_sigma: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def degrade(self, image: np.ndarray) -> np.ndarray:
        blurred = image
        if self.blur_sigma:
            blurred = convolve_per_channel(image, gaussian_kernel(self.blur_sigma))
        downsampled = Downsample(factor=self.factor, restore_size=True)(blurred)
        if self.noise_sigma:
            downsampled = GaussianNoise(sigma=self.noise_sigma, seed=self.seed)(downsampled)
        return downsampled

    def operator_params(self) -> Dict[str, Any]:
        return {"factor": self.factor, "blur_sigma": self.blur_sigma, "noise_sigma": self.noise_sigma}
