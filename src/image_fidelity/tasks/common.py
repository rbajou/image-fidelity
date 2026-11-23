"""Utilities shared by task handlers."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

NON_GROUP_COLUMNS = ("experiment_id", "image_id", "seed", "method", "runtime")


def degradation_columns(config: DictConfig) -> dict[str, Any]:
    """Name degradation and scalar parameters as result columns."""
    parameters = OmegaConf.to_container(config.degradation, resolve=True)
    target = str(parameters.pop("_target_", "none"))
    parameters.pop("seed", None)
    columns: dict[str, Any] = {"degradation": target.rsplit(".", 1)[-1]}
    columns.update({f"degradation_{key}": value for key, value in parameters.items()})
    return columns


def image_sample_count(config: DictConfig) -> int:
    if not bool(config.task.save.get("images", True)):
        return 0
    return max(int(config.task.get("image_samples", 4)), 0)
