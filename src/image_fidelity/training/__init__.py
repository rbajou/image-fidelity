"""Shared training components and legacy restoration entry points."""

from .engine import SeededSampler

__all__ = ["PatchDataset", "SeededSampler", "run_train", "test_model"]


def __getattr__(name: str):
    if name in {"PatchDataset", "run_train", "test_model"}:
        from image_fidelity.restoration import training

        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
