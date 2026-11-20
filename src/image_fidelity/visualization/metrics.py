"""Metric figures for restoration and segmentation benchmarks (distributions, boxplots, parameter sweeps)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from .params import RCPARAMS

RESTORATION_METRIC_COLUMNS = ("PSNR", "SSIM", "NRMSE", "grad_error", "bias", "flux_ratio", "runtime")
SEGMENTATION_METRIC_COLUMNS = ("dice", "iou", "precision", "recall", "f1", "ap", "count_error", "runtime")
_MAX_COLUMNS = 4
_SUMMARY_COLUMNS = {"method", "n"}


def _figure_grid(count: int):
    try:
        from matplotlib.figure import Figure
        from matplotlib.pyplot import rcParams
        rcParams.update(RCPARAMS)
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib; install the 'visualization' extra.") from exc

    columns = min(count, _MAX_COLUMNS)
    rows = -(-count // columns)
    figure = Figure(figsize=(4 * columns, 3.2 * rows), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    return figure, axes.ravel()[:count]


def default_metrics(columns: Sequence[str], suffix: str = "") -> tuple[str, ...]:
    """Restoration metrics if the table has any, else segmentation metrics (``suffix``: e.g. ``_mean``)."""
    for candidates in (RESTORATION_METRIC_COLUMNS, SEGMENTATION_METRIC_COLUMNS):
        selected = tuple(metric for metric in candidates if f"{metric}{suffix}" in columns)
        if selected and selected != ("runtime",):
            return selected
    return RESTORATION_METRIC_COLUMNS


def _available_metrics(columns: Sequence[str], metrics: Sequence[str]) -> list[str]:
    selected = [metric for metric in metrics if metric in columns]
    if not selected:
        raise ValueError(f"none of the metrics {list(metrics)} are present in the results")
    return selected


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _save(figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


def plot_metric_distributions(runs, path: str | Path, metrics: Sequence[str] | None = None) -> Path:
    """Histogram of every per-image metric value, one subplot per metric."""
    metrics = metrics or default_metrics(runs.columns)
    selected = _available_metrics(runs.columns, metrics)
    figure, axes = _figure_grid(len(selected))
    for axis, metric in zip(axes, selected):
        values = _finite(runs[metric])
        if values.size:
            axis.hist(
                values,
                bins="auto",
                color="tab:blue",
                alpha=0.8,
                label=f"mean = {values.mean():.4g}\nstddev = {values.std():.4g}",
            )
            axis.legend(fontsize="small")
        axis.set_title(metric)
        axis.set_xlabel(metric)
        axis.set_ylabel("count")
    return _save(figure, path)


def plot_metric_boxplots(
    runs,
    path: str | Path,
    metrics: Sequence[str] | None = None,
    group_by: str | None = None,
) -> Path:
    """Boxplot (median and interquartile range) per metric, with one box per ``group_by`` value."""
    metrics = metrics or default_metrics(runs.columns)
    selected = _available_metrics(runs.columns, metrics)
    if group_by is None:
        groups = [("all", runs)]
    else:
        groups = [(str(name), frame) for name, frame in runs.groupby(group_by, sort=False)]
    figure, axes = _figure_grid(len(selected))
    for axis, metric in zip(axes, selected):
        data = [_finite(frame[metric]) for _, frame in groups]
        positions = [index for index, values in enumerate(data, start=1) if values.size]
        if positions:
            axis.boxplot([data[position - 1] for position in positions], positions=positions)
        axis.set_xticks(range(1, len(groups) + 1), [name for name, _ in groups], rotation=30, ha="right")
        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.grid(True, which='both',linestyle='dotted', linewidth="0.5", color='grey')

    return _save(figure, path)


def swept_parameters(summary) -> list[str]:
    """Numeric summary columns taking several distinct values, i.e. the swept parameters."""
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    swept = []
    for column in summary.columns:
        if column in _SUMMARY_COLUMNS or column.endswith(("_mean", "_std")):
            continue
        values = summary[column]
        if is_bool_dtype(values) or not is_numeric_dtype(values):
            continue
        if values.dropna().nunique() > 1:
            swept.append(column)
    return swept


def plot_metric_sweep(
    summary,
    parameter: str,
    path: str | Path,
    metrics: Sequence[str] | None = None,
) -> Path:
    """Mean metric (error bar: standard deviation) against ``parameter``, one line per method."""
    metrics = metrics or default_metrics(summary.columns, "_mean")
    available = [metric for metric in metrics if f"{metric}_mean" in summary.columns]
    selected = _available_metrics(available, metrics)
    others = [column for column in swept_parameters(summary) if column != parameter]
    line_keys = ["method", *others]
    frame = summary.dropna(subset=[parameter])

    figure, axes = _figure_grid(len(selected))
    for axis, metric in zip(axes, selected):
        for key, line in frame.groupby(line_keys, sort=False, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            line = line.sort_values(parameter)
            label = ", ".join(f"{name}={value}" if name != "method" else str(value) for name, value in zip(line_keys, key))
            axis.errorbar(
                line[parameter],
                line[f"{metric}_mean"],
                yerr=line.get(f"{metric}_std"),
                marker="o",
                capsize=3,
                label=label,
            )
        axis.set_title(metric)
        axis.set_xlabel(parameter)
        axis.set_ylabel(f"{metric} (mean ± std)")
        axis.grid(True, which='both',linestyle='dotted', linewidth="0.5", color='grey')
        positive = frame[parameter][frame[parameter] > 0]
        if len(positive) == len(frame) and positive.max() / positive.min() >= 50:
            axis.set_xscale("log")
    axes[0].legend(fontsize="small")
    return _save(figure, path)


def save_run_figures(
    records,
    figures_dir: str | Path,
    *,
    boxplot_by_method: bool = False,
    distribution_by_method: bool = False,
) -> list[Path]:
    """Write the per-run distribution figure and its boxplots from per-image result records.

    With ``boxplot_by_method`` the boxplots compare methods (and are skipped for a single
    method); otherwise one box per metric summarises the whole run. With
    ``distribution_by_method``, write one distribution per method instead of pooling scores.
    """
    import pandas as pd

    runs = pd.DataFrame(records)
    figures_dir = Path(figures_dir)
    if distribution_by_method:
        groups = list(runs.groupby("method", sort=False))
        names = [re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._") or "sample" for name, _ in groups]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Method names produce duplicate distribution filenames")
        paths = [
            plot_metric_distributions(frame, figures_dir / f"metric_distributions_{name}.png")
            for name, (_, frame) in zip(names, groups)
        ]
    else:
        paths = [plot_metric_distributions(runs, figures_dir / "metric_distributions.png")]
    if not boxplot_by_method:
        paths.append(plot_metric_boxplots(runs, figures_dir / "metric_boxplots.png"))
    elif runs["method"].nunique() > 1:
        paths.append(plot_metric_boxplots(runs, figures_dir / "metric_boxplots_by_method.png", group_by="method"))
    return paths


def save_sweep_figures(summary_path: str | Path, figures_dir: str | Path) -> list[Path]:
    """Plot metrics against each swept parameter found in an aggregated ``summary.parquet``."""
    import pandas as pd

    summary = pd.read_parquet(summary_path)
    figures_dir = Path(figures_dir)
    return [
        plot_metric_sweep(summary, parameter, figures_dir / f"metrics_vs_{parameter}.png")
        for parameter in swept_parameters(summary)
    ]
