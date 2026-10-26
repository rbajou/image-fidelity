"""Common interface shared by all degradations and the pipeline that chains them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union

import numpy as np


def as_rng(seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    """Normalize a seed (or an existing `Generator`) into a `numpy.random.Generator`."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class Degradation(ABC):
    """A single, callable image degradation (e.g. a noise or blur model).

    Implementations receive and return a float image (any range/dtype
    conventions are up to the caller; degradations should be composable and
    not assume a fixed input range unless documented otherwise).
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the degradation to ``image`` and return the degraded array."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class IdentityDegradation(Degradation):
    """Return an unchanged copy of an image."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.copy()


class DegradationPipeline:
    """Applies a sequence of `Degradation` steps in order.

    Example::

        degradation = DegradationPipeline([
            PoissonNoise(photon_count=100),
            GaussianNoise(sigma=0.02),
            GaussianBlur(sigma=1.2),
        ])
        degraded = degradation(image)
    """

    def __init__(self, steps: List[Degradation]):
        self.steps = list(steps)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for step in self.steps:
            image = step(image)
        return image

    def __iter__(self) -> Iterator[Degradation]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        steps = ",\n    ".join(repr(step) for step in self.steps)
        return f"DegradationPipeline([\n    {steps}\n])"
