"""Restoration benchmark task handler."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from image_fidelity.benchmark import RestorationBenchmarkRunner, RestorationExperiment, RestorationSample
from image_fidelity.datasets import load_samples, sample_id, seed_component

from .common import degradation_columns, image_sample_count


def run_restoration_benchmark(config: DictConfig) -> Path:
    """Build shared degraded samples and execute all configured restorers."""
    _, samples = load_samples(config)
    method_configs = config.task.get("methods", {})
    if not method_configs:
        raise ValueError("Benchmark task has no methods; select an experiment or configure task.methods")

    experiments = []
    for index, (method_name, method_config) in enumerate(method_configs.items(), start=1):
        restorer = instantiate(method_config)
        parameters = OmegaConf.to_container(method_config, resolve=True)
        parameters.pop("_target_", None)
        experiments.append(
            RestorationExperiment(
                experiment_id=index,
                method=str(method_name),
                restorer=restorer,
                parameters=parameters,
            )
        )

    degradation = instantiate(config.degradation)
    degradation_info = degradation_columns(config)
    benchmark_samples = []
    for index, sample in enumerate(samples):
        seed = int(config.seed) + index
        seed_component(degradation, seed)
        benchmark_samples.append(
            RestorationSample(
                image_id=sample_id(sample, index),
                observed=degradation(sample.image),
                reference=sample.image,
                seed=seed,
                parameters=degradation_info,
                metadata=sample.metadata,
            )
        )

    runner = RestorationBenchmarkRunner(output_root=config.paths.output_dir, nest_run_dir=False)
    return runner.run(
        benchmark_samples,
        experiments,
        config=OmegaConf.to_container(config, resolve=True),
        save_predictions=bool(config.task.save.get("predictions", True)),
        save_results=bool(config.task.save.get("results", True)),
        image_samples=image_sample_count(config),
        save_figures=bool(config.task.save.get("figures", True)),
    )
