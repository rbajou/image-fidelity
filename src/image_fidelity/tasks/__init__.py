"""Hydra task handlers for restoration workflows."""

from .benchmark import run_benchmark
from .dispatch import TASK_HANDLERS, dispatch
from .plot import run_plot
from .restore import run_restore
from .train import run_training_task

__all__ = ["TASK_HANDLERS", "dispatch", "run_restore", "run_benchmark", "run_plot", "run_training_task"]
