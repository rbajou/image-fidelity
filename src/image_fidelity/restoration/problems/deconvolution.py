"""Deconvolution problem: recover a sharp image from an observation blurred by a known PSF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

from image_fidelity.degradations.blur import convolve_per_channel, gaussian_kernel
from image_fidelity.degradations.noise import GaussianNoise

from .base import InverseProblem


@dataclass
class DeconvolutionProblem(InverseProblem):
    """Observation is the clean image convolved with a known PSF, plus optional noise: ``y = k * x + n``."""

    kernel: np.ndarray = field(default_factory=lambda: gaussian_kernel(sigma=1.2))
    noise_sigma: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def degrade(self, image: np.ndarray) -> np.ndarray:
        blurred = convolve_per_channel(image, self.kernel)
        if self.noise_sigma:
            blurred = GaussianNoise(sigma=self.noise_sigma, seed=self.seed)(blurred)
        return blurred

    def operator_params(self) -> Dict[str, Any]:
        return {"kernel": self.kernel, "noise_sigma": self.noise_sigma}
