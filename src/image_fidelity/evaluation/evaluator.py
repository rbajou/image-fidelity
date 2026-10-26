"""Framework-independent evaluation of a single image restoration."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from image_fidelity.metrics import bias, flux_ratio, gradient_error, nrmse, psnr, ssim

from .results import EvaluationResult

RestorationMethod = Callable[[np.ndarray], np.ndarray]


class Evaluator:
    """Apply a restoration method once, then score its output against a reference."""

    def __init__(self, data_range: float = 1.0, ssim_window_size: int = 7):
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        self.data_range = data_range
        self.ssim_window_size = ssim_window_size

    def evaluate(
        self,
        method: RestorationMethod,
        observed: np.ndarray,
        reference: np.ndarray,
        *,
        experiment_id: str | int = "",
        image_id: str | int = "",
        seed: int | None = None,
        method_name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Restore one observation and return common quality metrics and runtime."""
        observed = np.asarray(observed)
        reference = np.asarray(reference)
        if observed.ndim == 0 or reference.ndim == 0:
            raise ValueError("observed and reference must be image arrays")
        if observed.shape != reference.shape:
            raise ValueError("observed and reference must have the same shape")

        start = perf_counter()
        reconstruction = np.asarray(method(observed))
        runtime = perf_counter() - start
        if reconstruction.shape != reference.shape:
            raise ValueError("restoration output and reference must have the same shape")

        scores = {
            "PSNR": psnr(reconstruction, reference, data_range=self.data_range),
            "SSIM": ssim(
                reconstruction,
                reference,
                data_range=self.data_range,
                window_size=self.ssim_window_size,
            ),
            "NRMSE": nrmse(reconstruction, reference),
            "grad_error": gradient_error(reconstruction, reference),
            "bias": bias(reconstruction, reference),
            "flux_ratio": flux_ratio(reconstruction, reference),
            "runtime": runtime,
        }
        return EvaluationResult(
            experiment_id=experiment_id,
            image_id=image_id,
            seed=seed,
            method=method_name or type(method).__name__,
            metrics=scores,
            parameters=dict(parameters or {}),
            reconstruction=reconstruction,
        )