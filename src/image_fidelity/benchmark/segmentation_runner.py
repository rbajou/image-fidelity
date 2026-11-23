"""Run instance-segmentation comparisons and persist their predictions and scores."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from image_fidelity.metrics import select_group_columns
from image_fidelity.segmentation.evaluation import evaluate_instances
from image_fidelity.visualization import save_run_figures, save_segmentation_preview

from image_fidelity.io.naming import safe_name

from .experiment import SegmentationExperiment, SegmentationSample
from .reporting import write_results_tables


class SegmentationBenchmarkRunner:
    """Evaluate segmenters against annotated instance masks and save segmentation outputs."""

    def __init__(self, output_root: str | Path = "results", nest_run_dir: bool = True):
        self.output_root = Path(output_root)
        self.nest_run_dir = nest_run_dir

    def _create_run_dir(self, benchmark_id: str | None = None) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not self.nest_run_dir:
            return self.output_root
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
        samples: Iterable[SegmentationSample],
        experiments: Iterable[SegmentationExperiment],
        *,
        config: Mapping[str, Any] | None = None,
        benchmark_id: str | None = None,
        metric_names: Iterable[str] = (),
        thresholds: Iterable[float] | None = None,
        compute_runtime: bool = True,
        save_predictions: bool = True,
        save_results: bool = True,
        image_samples: int = 0,
        save_figures: bool = False,
    ) -> Path:
        """Run every segmenter on every annotated sample and save the results."""
        samples = list(samples)
        experiments = list(experiments)
        if not samples:
            raise ValueError("segmentation benchmark requires at least one sample")
        if not experiments:
            raise ValueError("segmentation benchmark requires at least one experiment")

        selected_metrics = tuple(metric_names)
        output_dir = self._create_run_dir(benchmark_id)
        results_dir = output_dir / "results"
        predictions_dir = output_dir / "predictions"
        images_dir = output_dir / "images"
        figures_dir = output_dir / "figures"
        logs_dir = output_dir / "logs"
        directories = [logs_dir]
        if save_results:
            directories.append(results_dir)
        if save_predictions:
            directories.append(predictions_dir)
        if save_predictions or image_samples > 0:
            directories.append(images_dir)
        for directory in directories:
            directory.mkdir()

        logger = logging.getLogger(f"image_fidelity.segmentation_benchmark.{output_dir.name}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logs_dir / "benchmark.log", encoding="utf-8")
        logger.addHandler(handler)
        records: list[dict[str, Any]] = []
        try:
            logger.info("Starting segmentation benchmark %s", output_dir.name)
            for experiment_index, experiment in enumerate(experiments):
                method_dirname = safe_name(experiment.method).lower()
                for sample_index, sample in enumerate(samples):
                    start = perf_counter()
                    prediction = np.asarray(experiment.segmenter(sample.observed))
                    runtime = perf_counter() - start
                    scores = evaluate_instances(
                        prediction,
                        sample.target,
                        **({"thresholds": thresholds} if thresholds is not None else {}),
                    )
                    unknown = set(selected_metrics) - scores.keys()
                    if unknown:
                        raise ValueError(
                            f"Unsupported segmentation metric(s) {sorted(unknown)}; "
                            f"available: {sorted(scores)}"
                        )
                    score_columns = selected_metrics or tuple(scores)
                    method_parameters = {
                        f"method_{method_dirname}_{key}": value
                        for key, value in experiment.parameters.items()
                    }
                    parameters = {**sample.parameters, **method_parameters}
                    reserved = {
                        "experiment_id",
                        "image_id",
                        "seed",
                        "method",
                        *score_columns,
                    }
                    if compute_runtime:
                        reserved.add("runtime")
                    overlap = parameters.keys() & reserved
                    if overlap:
                        raise ValueError(f"parameters overwrite result columns: {sorted(overlap)}")
                    record = {
                        "experiment_id": experiment.experiment_id,
                        "image_id": sample.image_id,
                        "seed": sample.seed,
                        "method": experiment.method,
                        **{name: scores[name] for name in score_columns},
                        **parameters,
                    }
                    if compute_runtime:
                        record["runtime"] = runtime
                    records.append(record)

                    artifact_name = (
                        f"{experiment_index:04d}_{sample_index:04d}_"
                        f"{safe_name(experiment.experiment_id)}_{safe_name(sample.image_id)}.npy"
                    )
                    if save_predictions:
                        method_dir = predictions_dir / method_dirname
                        method_dir.mkdir(exist_ok=True)
                        np.save(method_dir / artifact_name, prediction)

                    if sample_index < image_samples:
                        save_segmentation_preview(
                            images_dir / artifact_name.replace(".npy", ".png"),
                            sample.original,
                            sample.observed,
                            prediction,
                            sample.target,
                            image_id=str(sample.image_id),
                            method=experiment.method,
                            degradation=sample.parameters,
                            scores=scores,
                        )
                    logger.info(
                        "experiment=%s image=%s method=%s runtime=%.6f",
                        experiment.experiment_id,
                        sample.image_id,
                        experiment.method,
                        runtime,
                    )

            if save_results:
                score_columns = selected_metrics or tuple(scores)
                excluded = {"experiment_id", "image_id", "seed", "method", *score_columns}
                if compute_runtime:
                    excluded.add("runtime")
                group_columns = select_group_columns(records, exclude=excluded)
                write_results_tables(records, results_dir, group_by=("method", *group_columns))
            if save_figures:
                save_run_figures(records, figures_dir, boxplot_by_method=True, distribution_by_method=True)

            run_config = {
                **dict(config or {}),
                "experiments": [
                    {
                        "experiment_id": item.experiment_id,
                        "method": item.method,
                        "parameters": item.parameters,
                    }
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
            logger.info("Completed segmentation benchmark with %d result rows", len(records))
        except Exception:
            logger.exception("Segmentation benchmark failed")
            raise
        finally:
            handler.close()
            logger.removeHandler(handler)
        return output_dir
