"""Common interface shared by all inverse imaging problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class InverseProblem(ABC):
    """A forward degradation model together with the parameters classical solvers need to invert it.

    Subclasses wrap one (or a pipeline of) `Degradation` steps from
    `image_fidelity.degradations` and expose the operator's parameters (PSF, mask,
    downsampling factor, ...) so that classical model-based methods
    (Wiener, Richardson-Lucy, Tikhonov, ...) can use them explicitly, unlike
    learned methods which only see (degraded, clean) pairs.
    """

    @abstractmethod
    def degrade(self, image: np.ndarray) -> np.ndarray:
        """Apply the forward degradation to a clean image."""

    @abstractmethod
    def operator_params(self) -> Dict[str, Any]:
        """Parameters describing the forward operator (PSF, mask, factor, ...) for classical solvers."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.degrade(image)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"
