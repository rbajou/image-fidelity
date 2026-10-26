"""Training task handler."""

from pathlib import Path

from omegaconf import DictConfig


def run_training_task(config: DictConfig) -> Path:
    from image_fidelity.training import run_train

    return run_train(config)
