"""Method configurations used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from image_fidelity.evaluation import RestorationMethod


@dataclass
class Experiment:
    """One configured restoration method in a benchmark."""

    experiment_id: str | int
    method: str
    restorer: RestorationMethod
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    """An observed/reference image pair and its acquisition conditions."""

    image_id: str | int
    observed: np.ndarray
    reference: np.ndarray
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


RestorationFactory = Callable[..., RestorationMethod]