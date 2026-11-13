"""Training task handler."""

from pathlib import Path

from omegaconf import DictConfig


def run_training_task(config: DictConfig) -> Path:
    objective = str(config.task.get("objective", "restoration"))
    if objective == "restoration":
        from image_fidelity.training import run_train
    elif objective == "segmentation":
        from image_fidelity.segmentation.training import run_train
    else:
        raise ValueError(f"Unsupported training objective {objective!r}; choose restoration or segmentation")
    return run_train(config)
