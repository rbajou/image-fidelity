"""Tabular export and plotting helpers shared by restoration and segmentation benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from image_fidelity.evaluation.statistics import summarize_results


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


def plot_metric_vs_parameter(
    records: Sequence[Mapping[str, Any]],
    parameter: str,
    metric: str,
    output_path: str | Path,
) -> Path:
    """Plot a metric against one numeric parameter, with a line per method."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib; install the 'visualization' extra.") from exc

    grouped: dict[str, dict[float, list[float]]] = {}
    for record in records:
        if parameter not in record or metric not in record:
            raise ValueError(f"records must contain {parameter!r} and {metric!r}")
        grouped.setdefault(str(record["method"]), {}).setdefault(float(record[parameter]), []).append(
            float(record[metric])
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    for method, points in grouped.items():
        x_values = sorted(points)
        y_values = [sum(points[x]) / len(points[x]) for x in x_values]
        axis.plot(x_values, y_values, marker="o", label=method)
    axis.set_xlabel(parameter)
    axis.set_ylabel(metric)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path