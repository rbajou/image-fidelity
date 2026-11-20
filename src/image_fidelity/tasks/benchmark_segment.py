"""Segmentation benchmark task handler."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from image_fidelity.benchmark import (
    SegmentationBenchmarkRunner,
    SegmentationExperiment,
    SegmentationSample,
)
from image_fidelity.datasets import load_samples, sample_id, seed_component

from .common import degradation_columns, image_sample_count


def run_segmentation_benchmark(config: DictConfig) -> Path:
    """Build shared degraded samples and compare segmenters against annotated targets."""
    _, samples = load_samples(config)
    missing_targets = [
        sample_id(sample, index) for index, sample in enumerate(samples) if sample.target is None
    ]
    if missing_targets:
        raise ValueError(
            "Segmentation benchmarks require an annotated target mask for every sample; "
            f"missing targets: {missing_targets}. Use an annotated split (for BBBC038, "
            "validation is drawn from stage1_train) or provide an annotated test set."
        )

    method_configs = config.task.get("methods", {})
    if not method_configs:
        raise ValueError(
            "Segmentation benchmark has no methods; select an experiment or configure task.methods"
        )

    experiments = []
    for index, (method_name, method_config) in enumerate(method_configs.items(), start=1):
        segmenter = instantiate(method_config)
        parameters = OmegaConf.to_container(method_config, resolve=True)
        parameters.pop("_target_", None)
        experiments.append(
            SegmentationExperiment(
                experiment_id=index,
                method=str(method_name),
                segmenter=segmenter,
                parameters=parameters,
            )
        )

    degradation = instantiate(config.degradation)
    degradation_info = degradation_columns(config)
    benchmark_samples = []
    for index, sample in enumerate(samples):
        target = sample.target
        if target is None:
            raise ValueError(f"Segmentation benchmark sample {sample_id(sample, index)!r} has no annotated target")
        seed = int(config.seed) + index
        seed_component(degradation, seed)
        benchmark_samples.append(
            SegmentationSample(
                image_id=sample_id(sample, index),
                original=sample.image,
                observed=degradation(sample.image),
                target=target,
                seed=seed,
                parameters=degradation_info,
                metadata=sample.metadata,
            )
        )

    runner = SegmentationBenchmarkRunner(output_root=config.paths.output_dir, nest_run_dir=False)
    return runner.run(
        benchmark_samples,
        experiments,
        config=OmegaConf.to_container(config, resolve=True),
        metric_names=[str(name).lower() for name in config.metrics.get("names", [])],
        thresholds=config.metrics.get("thresholds"),
        compute_runtime=bool(config.metrics.get("compute_runtime", True)),
        save_predictions=bool(config.task.save.get("predictions", True)),
        save_results=bool(config.task.save.get("results", True)),
        image_samples=image_sample_count(config),
        save_figures=bool(config.task.save.get("figures", True)),
    )
