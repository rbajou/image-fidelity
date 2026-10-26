"""Combine per-job result tables produced by a Hydra multirun sweep."""

from __future__ import annotations

from pathlib import Path


def aggregate_benchmark_runs(sweep_root: str | Path) -> Path | None:
    """Concatenate every job's ``results/runs.parquet`` and ``results/summary.parquet``.

    The combined tables are written to ``<sweep_root>/results/``. Summaries are
    concatenated rather than recomputed, so each job keeps its own rows (and the
    swept method/degradation parameter columns it recorded).

    Returns the aggregated ``runs.parquet`` path, or ``None`` if no job wrote one.
    """
    sweep_root = Path(sweep_root)
    results_dir = sweep_root / "results"

    def job_tables(name: str) -> list[Path]:
        return sorted(path for path in sweep_root.glob(f"**/results/{name}") if path.parent != results_dir)

    run_paths = job_tables("runs.parquet")
    if not run_paths:
        return None

    import pandas as pd

    results_dir.mkdir(parents=True, exist_ok=True)
    runs_path = results_dir / "runs.parquet"
    pd.concat([pd.read_parquet(path) for path in run_paths], ignore_index=True, sort=False).to_parquet(
        runs_path, index=False
    )

    summary_paths = job_tables("summary.parquet")
    if summary_paths:
        pd.concat([pd.read_parquet(path) for path in summary_paths], ignore_index=True, sort=False).to_parquet(
            results_dir / "summary.parquet", index=False
        )
    return runs_path
