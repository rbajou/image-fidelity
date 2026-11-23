"""Serializable per-image results from a restoration evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class EvaluationResult:
    """One row of benchmark metrics plus the reconstruction used to compute it."""

    experiment_id: str | int
    image_id: str | int
    seed: int | None
    method: str
    metrics: Mapping[str, float]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    reconstruction: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def to_record(self) -> dict[str, Any]:
        """Return flat tabular data, excluding image arrays."""
        record = {
            "experiment_id": self.experiment_id,
            "image_id": self.image_id,
            "seed": self.seed,
            "method": self.method,
            **self.metrics,
        }
        overlap = record.keys() & self.parameters.keys()
        if overlap:
            raise ValueError(f"parameters overwrite result columns: {sorted(overlap)}")
        record.update(self.parameters)
        return record