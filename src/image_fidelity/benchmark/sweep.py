"""Cartesian hyperparameter sweeps for restoration methods."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Mapping

from .experiment import Experiment, RestorationFactory


def parameter_sweep(
    method: str,
    factory: RestorationFactory,
    parameter_grid: Mapping[str, Iterable[Any]],
    *,
    first_experiment_id: int = 1,
) -> list[Experiment]:
    """Instantiate one experiment per Cartesian combination of parameters."""
    names = list(parameter_grid)
    values = [list(parameter_grid[name]) for name in names]
    if any(not options for options in values):
        return []
    combinations = product(*values) if names else [()]
    experiments = []
    for index, combination in enumerate(combinations):
        parameters = dict(zip(names, combination))
        experiments.append(
            Experiment(
                experiment_id=first_experiment_id + index,
                method=method,
                restorer=factory(**parameters),
                parameters=parameters,
            )
        )
    return experiments