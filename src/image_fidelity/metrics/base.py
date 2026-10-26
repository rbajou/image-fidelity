"""Common interface shared by all restoration-evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Metric(ABC):
    """A single evaluation metric comparing a restored image against a reference.

    Returns either a scalar score or a dict of related scores (e.g.
    precision/recall/F1), depending on the metric.
    """

    @abstractmethod
    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> Any:
        """Compute the metric value(s) between `restored` and `reference`."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"
