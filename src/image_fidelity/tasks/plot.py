"""Plot task handler: regenerate figures from saved result tables without recomputation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from image_fidelity.visualization import (
    plot_metric_boxplots,
    plot_metric_distributions,
    plot_metric_sweep,
    swept_parameters,
)
from .common import safe_name
from .common import safe_name

LOGGER = logging.getLogger(__name__)


def _results_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    return path / "results" if (path / "results").is_dir() else path


def _run_dir(path: Path) -> Path:
    results = _results_dir(path)
    return results.parent if results.name == "results" else results


def _figures_dir(inputs, fallback: Path) -> Path:
    """Figures go to the input run's ``figures/``; several inputs share the plot task's own directory."""
    if len(inputs) != 1:
        return fallback / "figures"
    results = _results_dir(Path(str(inputs[0])))
    run_dir = results.parent if results.name == "results" else results
    return run_dir / "figures"


def load_table(inputs, source: str) -> pd.DataFrame:
    """Concatenate ``<source>.parquet`` from every input run directory or parquet file."""
    tables = []
    for item in inputs:
        table_path = _results_dir(Path(str(item))) / f"{source}.parquet"
        if table_path.is_file():
            tables.append(pd.read_parquet(table_path))
        else:
            LOGGER.warning("Missing %s", table_path)
    if not tables:
        raise FileNotFoundError(f"No {source}.parquet found in task.inputs")
    return pd.concat(tables, ignore_index=True, sort=False)


def _metrics(spec: DictConfig) -> tuple[str, ...] | None:
    metrics = spec.get("metrics")
    return tuple(metrics) if metrics else None


def _distributions(table, spec, figures_dir):
    split_by = spec.get("split_by")
    if not split_by:
        split_by = spec.get("split_by")
    if not split_by or split_by not in table.columns:
        return [plot_metric_distributions(table, figures_dir / spec.filename, _metrics(spec))]
    stem, suffix = Path(str(spec.filename)).stem, Path(str(spec.filename)).suffix
    return [
        plot_metric_distributions(frame, figures_dir / f"{stem}_{safe_name(str(name))}{suffix}", _metrics(spec))
        for name, frame in table.groupby(split_by, sort=False)
    ]
    if split_by not in table.columns:
        LOGGER.info("Skipping distributions: no %r column", split_by)
        return []
    base = Path(str(spec.filename))
    return [
        plot_metric_distributions(
            frame, figures_dir / f"{base.stem}_{safe_name(str(value))}{base.suffix}", _metrics(spec)
        )
        for value, frame in table.groupby(split_by, sort=False)
    ]


def _boxplots(table, spec, figures_dir):
    group_by = spec.get("group_by")
    if group_by and (group_by not in table.columns or table[group_by].nunique() < 2):
        LOGGER.info("Skipping boxplots: fewer than two %r values", group_by)
        return []
    return [plot_metric_boxplots(table, figures_dir / spec.filename, _metrics(spec), group_by=group_by)]


def _sweep(table, spec, figures_dir):
    parameters = list(spec.get("parameters") or swept_parameters(table))
    if not parameters:
        LOGGER.info("Skipping sweep plots: no swept parameter in summary")
    return [
        plot_metric_sweep(table, parameter, figures_dir / str(spec.filename).format(parameter=parameter), _metrics(spec))
        for parameter in parameters
    ]


PLOT_TYPES = {
    "metric_distributions": _distributions,
    "metric_boxplots": _boxplots,
    "metric_sweep": _sweep,
}


def run_plot(config: DictConfig) -> Path:
    """Render every plot configured in ``task.plots`` from the parquet tables in ``task.inputs``."""
    inputs = OmegaConf.to_container(config.task.get("inputs") or [], resolve=True)
    if isinstance(inputs, str):
        inputs = [inputs]
    if not inputs:
        raise ValueError("Plot task has no inputs; set task.inputs=[<run_dir>,...]")

    # A single input keeps its figures next to its results; several inputs go to this task's own output dir.
    base = _run_dir(Path(str(inputs[0]))) if len(inputs) == 1 else Path(config.paths.output_dir)
    figures_dir = base / "figures"
    tables: dict[str, pd.DataFrame | None] = {}
    for name, spec in config.task.get("plots", {}).items():
        plot_type = str(spec.type)
        try:
            handler = PLOT_TYPES[plot_type]
        except KeyError as exc:
            raise ValueError(f"Unknown plot type {plot_type!r} for {name!r}; available: {', '.join(PLOT_TYPES)}") from exc
        source = str(spec.source)
        if source not in tables:
            try:
                tables[source] = load_table(inputs, source)
            except FileNotFoundError as exc:
                LOGGER.warning("Skipping plot %r: %s", name, exc)
                tables[source] = None
        if tables[source] is None:
            continue
        for path in handler(tables[source], spec, figures_dir):
            LOGGER.info("Wrote %s", path)
    return figures_dir
