"""Tabular export shared by restoration and segmentation benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from image_fidelity.metrics.statistics import summarize_results


def write_results_tables(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    group_by: Sequence[str],
) -> tuple[Path, Path]:
    """Write per-run ``runs.parquet`` and aggregated ``summary.parquet`` tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Writing result tables requires the optional dependencies; install with "
            "`pip install '.[benchmark]'`."
        ) from exc

    runs_path = output_dir / "runs.parquet"
    try:
        pd.DataFrame(records).to_parquet(runs_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "A Parquet engine is required; install the benchmark dependencies with `pip install '.[benchmark]'`."
        ) from exc

    summaries = summarize_results(records, group_by=group_by)
    summary_path = output_dir / "summary.parquet"
    pd.DataFrame(summaries).to_parquet(summary_path, index=False)
    return runs_path, summary_path

