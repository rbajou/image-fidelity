"""Hydra callbacks for task orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from image_fidelity.benchmark.aggregation import aggregate_benchmark_runs
from image_fidelity.visualization import save_sweep_figures


class AggregateBenchmarkResults(Callback):
    """Combine benchmark results and plot metrics after a multirun sweep."""

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        sweep_dir = Path(config.hydra.sweep.dir)
        if not sweep_dir.is_dir() or aggregate_benchmark_runs(sweep_dir) is None:
            return
        summary_path = sweep_dir / "results" / "summary.parquet"
        if summary_path.is_file():
            save_sweep_figures(summary_path, sweep_dir / "figures")
