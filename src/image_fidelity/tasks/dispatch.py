"""Task handler registry and dispatch."""

from pathlib import Path
from typing import Callable

from omegaconf import DictConfig

from .benchmark import run_benchmark
from .degrade import run_degrade
from .download import run_download
from .plot import run_plot
from .restore import run_restore
from .segment import run_segment
from .train import run_training_task

TASK_HANDLERS: dict[str, Callable[[DictConfig], Path]] = {
    "restore": run_restore,
    "segment": run_segment,
    "benchmark": run_benchmark,
    "train": run_training_task,
    "degrade": run_degrade,
    "plot": run_plot,
    "download": run_download,
}


def dispatch(config: DictConfig) -> Path:
    """Run the configured task or report that its handler is not implemented."""
    task_name = str(config.task.name)
    try:
        handler = TASK_HANDLERS[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_HANDLERS))
        raise ValueError(f"Unsupported task {task_name!r}; implemented tasks: {available}") from exc
    return handler(config)
