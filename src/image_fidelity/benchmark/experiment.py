"""Restoration and segmentation method configurations used by benchmark runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from image_fidelity.restoration.evaluation import RestorationMethod


@dataclass
class RestorationExperiment:
    """One configured restoration method in a benchmark."""

    experiment_id: str | int
    method: str
    restorer: RestorationMethod
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RestorationSample:
    """An observed/reference image pair and its acquisition conditions."""

    image_id: str | int
    observed: np.ndarray
    reference: np.ndarray
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


RestorationFactory = Callable[..., RestorationMethod]


@dataclass
class SegmentationExperiment:
    """One configured instance-segmentation method in a benchmark."""

    experiment_id: str | int
    method: str
    segmenter: Callable[[np.ndarray], np.ndarray]
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationSample:
    """An original image, its degraded observation, and its annotated instance mask."""

    image_id: str | int
    original: np.ndarray
    observed: np.ndarray
    target: np.ndarray
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

