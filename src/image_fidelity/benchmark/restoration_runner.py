"""Run restoration comparisons and persist reproducible benchmark artifacts."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from image_fidelity.metrics import select_group_columns
from image_fidelity.restoration.evaluation import RestorationEvaluator

from image_fidelity.visualization import save_before_after, save_run_figures

from image_fidelity.io.naming import safe_name

from .experiment import RestorationExperiment, RestorationSample
from .reporting import write_results_tables


NON_GROUP_COLUMNS = (
    "experiment_id", "image_id", "seed", "method", "PSNR", "SSIM", "NRMSE",
    "grad_error", "bias", "flux_ratio", "runtime",
)


class RestorationBenchmarkRunner:
    """Evaluate configured restorers over image pairs and save restoration outputs."""

    def __init__(
        self,
        output_root: str | Path = "results",
        evaluator: RestorationEvaluator | None = None,
        nest_run_dir: bool = True,
    ):
        self.output_root = Path(output_root)
        self.evaluator = evaluator or RestorationEvaluator()
        self.nest_run_dir = nest_run_dir

    def _create_run_dir(self, benchmark_id: str | None = None) -> Path:
        if not self.nest_run_dir:
            self.output_root.mkdir(parents=True, exist_ok=True)
            return self.output_root

        self.output_root.mkdir(parents=True, exist_ok=True)
        if benchmark_id is not None:
            output_dir = self.output_root / safe_name(benchmark_id)
            output_dir.mkdir()
            return output_dir

        prefix = f"benchmark_{datetime.now():%Y-%m-%d}_"
        index = 1
        while (self.output_root / f"{prefix}{index:03d}").exists():
            index += 1
        output_dir = self.output_root / f"{prefix}{index:03d}"
        output_dir.mkdir()
        return output_dir

    def run(
        self,
        samples: Iterable[RestorationSample],
        experiments: Iterable[RestorationExperiment],
        *,
        config: Mapping[str, Any] | None = None,
        benchmark_id: str | None = None,
        save_predictions: bool = True,
        save_results: bool = True,
        image_samples: int = 0,
        save_figures: bool = False,
    ) -> Path:
        """Run all image/method combinations and return the benchmark directory."""
        samples = list(samples)
        experiments = list(experiments)
        if not samples:
            raise ValueError("benchmark requires at least one sample")
        if not experiments:
            raise ValueError("benchmark requires at least one experiment")

        output_dir = self._create_run_dir(benchmark_id)
        results_dir = output_dir / "results"
        predictions_dir = output_dir / "predictions"
        images_dir = output_dir / "images"
        figures_dir = output_dir / "figures"
        logs_dir = output_dir / "logs"
        directories = [figures_dir, logs_dir]
        if save_results:
            directories.append(results_dir)
        if save_predictions:
            directories.append(predictions_dir)
        if save_predictions or image_samples > 0:
            directories.append(images_dir)
        for directory in directories:
            directory.mkdir()

        log_path = logs_dir / "benchmark.log"
        logger = logging.getLogger(f"image_fidelity.benchmark.{output_dir.name}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        logger.addHandler(handler)
        records: list[dict[str, Any]] = []
        try:
            logger.info("Starting benchmark %s", output_dir.name)
            for experiment_index, experiment in enumerate(experiments):
                method_dirname = safe_name(experiment.method).lower()
                preview_restored: list[np.ndarray] = []
                for sample_index, sample in enumerate(samples):
                    overlap = sample.parameters.keys() & experiment.parameters.keys()
                    if overlap:
                        raise ValueError(f"sample and method parameters overlap: {sorted(overlap)}")
                    parameters = {**sample.parameters, **experiment.parameters}
                    result = self.evaluator.evaluate(
                        experiment.restorer,
                        sample.observed,
                        sample.reference,
                        experiment_id=experiment.experiment_id,
                        image_id=sample.image_id,
                        seed=sample.seed,
                        method_name=experiment.method,
                        parameters=parameters,
                    )
                    record = result.to_record()
                    records.append(record)

                    if len(preview_restored) < image_samples:
                        preview_restored.append(np.asarray(result.reconstruction))
                    if save_predictions:
                        artifact_name = (
                            f"{experiment_index:04d}_{sample_index:04d}_"
                            f"{safe_name(experiment.experiment_id)}_{safe_name(sample.image_id)}.npy"
                        )
                        method_dir = predictions_dir / method_dirname
                        method_dir.mkdir(exist_ok=True)
                        np.save(method_dir / artifact_name, result.reconstruction)
                        residual = np.asarray(sample.reference, dtype=np.float64) - np.asarray(
                            result.reconstruction, dtype=np.float64
                        )
                        np.save(images_dir / artifact_name, residual)
                    logger.info(
                        "experiment=%s image=%s method=%s runtime=%.6f",
                        experiment.experiment_id,
                        sample.image_id,
                        experiment.method,
                        result.metrics["runtime"],
                    )

                if preview_restored:
                    count = len(preview_restored)
                    save_before_after(
                        images_dir / f"{experiment_index:04d}_{safe_name(experiment.experiment_id)}_"
                        f"{safe_name(experiment.method)}.png",
                        [np.asarray(sample.reference) for sample in samples[:count]],
                        [np.asarray(sample.observed) for sample in samples[:count]],
                        preview_restored,
                        [str(sample.image_id) for sample in samples[:count]],
                    )

            if save_results:
                group_columns = select_group_columns(records, exclude=NON_GROUP_COLUMNS)
                write_results_tables(records, results_dir, group_by=("method", *group_columns))
            if save_figures:
                save_run_figures(records, figures_dir, boxplot_by_method=True, distribution_by_method=True)

            run_config = {
                **dict(config or {}),
                "experiments": [
                    {"experiment_id": item.experiment_id, "method": item.method, "parameters": item.parameters}
                    for item in experiments
                ],
            }
            with (output_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
                json.dump(run_config, config_file, indent=2, default=str)
                config_file.write("\n")

            metadata = {
                "benchmark_id": output_dir.name,
                "created_at": datetime.now().astimezone().isoformat(),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "sample_count": len(samples),
                "experiment_count": len(experiments),
                "result_count": len(records),
            }
            with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
                metadata_file.write("\n")
            logger.info("Completed benchmark with %d result rows", len(records))
        except Exception:
            logger.exception("Benchmark failed")
            raise
        finally:
            handler.close()
            logger.removeHandler(handler)
        return output_dir
