"""Denoising problem: recover a clean image from a noisy observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

from image_fidelity.degradations.base import Degradation, DegradationPipeline
from image_fidelity.degradations.noise import GaussianNoise

from .base import InverseProblem


@dataclass
class DenoisingProblem(InverseProblem):
    """The observation is a noisy version of the clean image: ``y = x + n`` (or another noise model).

    By default applies additive white Gaussian noise; pass any `Degradation`
    (or list of them) via ``noise`` to use a different/combined noise model
    (Poisson, salt & pepper, readout, ...).
    """

    noise: Union[Degradation, List[Degradation]] = field(default_factory=GaussianNoise)

    def __post_init__(self) -> None:
        steps = self.noise if isinstance(self.noise, list) else [self.noise]
        self._pipeline = DegradationPipeline(steps)

    def degrade(self, image: np.ndarray) -> np.ndarray:
        return self._pipeline(image)

    def operator_params(self) -> Dict[str, Any]:
        first = self.noise[0] if isinstance(self.noise, list) else self.noise
        return {"noise_sigma": getattr(first, "sigma", None)}
