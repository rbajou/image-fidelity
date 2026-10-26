"""Level 3 - relevance to the final application: structure detection, density/measurement, anomaly, ...

Wraps a concrete application-level analysis (`DownstreamTask`) so restoration
quality can be scored by how well it preserves the measurement the end
application actually cares about (e.g. nucleus count, object density, an
anomaly score, a volumetric reconstruction quantity), rather than generic
image or segmentation metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import ndimage

from .base import Metric
from .segmentation import measurement_relative_error


class DownstreamTask(ABC):
    """A concrete application-level analysis extracted from an image (count, density, anomaly score, ...)."""

    @abstractmethod
    def analyze(self, image: np.ndarray) -> float:
        """Compute the application-relevant scalar measurement for `image`."""

    def __call__(self, image: np.ndarray) -> float:
        return self.analyze(image)


@dataclass
class NucleusCountTask(DownstreamTask):
    """Counts connected foreground components above `threshold` (e.g. nuclei in a BBBC038-like image)."""

    threshold: float = 0.5

    def analyze(self, image: np.ndarray) -> float:
        mask = np.asarray(image) > self.threshold
        _, n_components = ndimage.label(mask)
        return float(n_components)


@dataclass
class ObjectDensityTask(DownstreamTask):
    """Fraction of foreground pixels above `threshold`, a simple proxy for structure density."""

    threshold: float = 0.5

    def analyze(self, image: np.ndarray) -> float:
        mask = np.asarray(image) > self.threshold
        return float(mask.mean())


@dataclass
class AnomalyScoreTask(DownstreamTask):
    """Fraction of pixels deviating from the image mean by more than `n_sigma` standard deviations.

    A minimal stand-in for an anomaly-detection pipeline: restoration is
    scored on whether it preserves this anomaly signal, not the raw pixels.
    """

    n_sigma: float = 3.0

    def analyze(self, image: np.ndarray) -> float:
        image = np.asarray(image, dtype=np.float64)
        deviation = np.abs(image - image.mean())
        threshold = self.n_sigma * image.std()
        return float(np.mean(deviation > threshold))


@dataclass
class DownstreamRelevance(Metric):
    """Scores how well a restoration preserves a `DownstreamTask` measurement, relative to the reference image.

    Compares `task(restored)` against `task(reference)` (the reference stands
    in for ground truth when no independent annotation is available).
    """

    task: DownstreamTask

    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        restored_value = self.task(restored)
        reference_value = self.task(reference)
        return {
            "restored_value": restored_value,
            "reference_value": reference_value,
            "relative_error": measurement_relative_error(restored_value, reference_value),
        }
