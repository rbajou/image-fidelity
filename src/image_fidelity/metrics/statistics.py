"""Aggregation helpers for per-image evaluation records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _group_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _group_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_group_key(item) for item in value)
    if isinstance(value, np.ndarray):
        return _group_key(value.tolist())
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def summarize_results(
    records: Iterable[Mapping[str, Any]],
    group_by: Sequence[str] = ("method",),
) -> list[dict[str, Any]]:
    """Compute mean and population standard deviation per group and metric."""
    rows = list(records)
    if not rows:
        return []
    metric_names = [
        key
        for key, value in rows[0].items()
        if key not in {"experiment_id", "image_id", "seed", "method", *group_by}
        and isinstance(value, (int, float, np.number))
    ]

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(_group_key(row.get(key)) for key in group_by)].append(row)

    summaries = []
    for group in groups.values():
        summary = {key: group[0].get(key) for key in group_by}
        summary["n"] = len(group)
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group if row.get(metric) is not None], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
        summaries.append(summary)
    return summaries


def select_group_columns(records: Iterable[Mapping[str, Any]], exclude: Sequence[str] = ()) -> list[str]:
    """Return record keys usable as group-by columns, skipping array/dict values (e.g. a kernel)."""
    excluded = set(exclude)
    columns: dict[str, bool] = {}
    for row in records:
        for key, value in row.items():
            if key in excluded or key in columns:
                continue
            columns[key] = not isinstance(value, (Mapping, list, tuple, np.ndarray))
    return [key for key, is_scalar in columns.items() if is_scalar]