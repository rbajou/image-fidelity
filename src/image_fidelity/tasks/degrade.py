"""Degrade task handler: degrade a subset of images and save original/degraded figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from image_fidelity.datasets import load_samples, sample_id, seed_component
from image_fidelity.visualization import save_image_pair

from .common import safe_name


def run_degrade(config: DictConfig) -> Path:
    """Degrade each selected image and save one original/degraded figure per image."""
    _, samples = load_samples(config)
    degradation_config = config.degradation
    if not degradation_config.get("_target_") or str(degradation_config._target_).endswith("IdentityDegradation"):
        degradation_config = config.task.default_degradation
    degradation = instantiate(degradation_config)
    output_dir = Path(config.paths.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    save_arrays = bool(config.task.get("save", {}).get("arrays", False))
    for index, sample in enumerate(samples):
        seed_component(degradation, int(config.seed) + index)
        degraded = degradation(sample.image)
        name = safe_name(sample_id(sample, index))
        save_image_pair(images_dir / f"{name}.png", sample.image, degraded, title=name)
        if save_arrays:
            np.save(images_dir / f"{name}.npy", degraded)
    return output_dir
