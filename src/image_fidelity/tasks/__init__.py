"""Hydra task handlers for image-fidelity workflows."""

from .benchmark_restore import run_restoration_benchmark
from .benchmark_segment import run_segmentation_benchmark
from .dispatch import TASK_HANDLERS, dispatch
from .plot import run_plot
from .restore import run_restore
from .train import run_training_task

__all__ = [
    "TASK_HANDLERS",
    "dispatch",
    "run_restore",
    "run_restoration_benchmark",
    "run_segmentation_benchmark",
    "run_plot",
    "run_training_task",
]
