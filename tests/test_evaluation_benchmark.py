import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.benchmark import BenchmarkRunner, BenchmarkSample, Experiment, parameter_sweep
from image_fidelity.benchmark import runner as benchmark_runner
from image_fidelity.evaluation import Evaluator, summarize_results


def test_evaluator_returns_common_scores_and_flat_result_record():
    """Evaluation scores and metadata are available in a flat result record."""
    reference = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    result = Evaluator().evaluate(
        lambda observed: observed,
        reference,
        reference,
        experiment_id=3,
        image_id="image-a",
        seed=7,
        method_name="identity",
        parameters={"iterations": 5},
    )

    record = result.to_record()
    assert record["experiment_id"] == 3
    assert record["image_id"] == "image-a"
    assert record["iterations"] == 5
    assert record["PSNR"] == float("inf")
    assert record["SSIM"] == 1.0
    assert record["NRMSE"] == 0.0
    assert record["runtime"] >= 0.0
    assert result.reconstruction.shape == reference.shape


def test_parameter_sweep_instantiates_cartesian_combinations():
    """A parameter sweep creates one experiment for each Cartesian combination."""
    experiments = parameter_sweep(
        "test-method",
        lambda scale, iterations: lambda image: image * scale + iterations * 0,
        {"scale": [1, 2], "iterations": [3, 4]},
    )

    assert len(experiments) == 4
    assert experiments[0].parameters == {"scale": 1, "iterations": 3}
    assert experiments[-1].experiment_id == 4


def test_summaries_preserve_method_specific_parameter_columns():
    """Summary rows retain method-specific parameters and fill absent values."""
    summaries = summarize_results(
        [
            {"method": "rl", "iterations": 5, "PSNR": 20.0},
            {"method": "rl", "iterations": 10, "PSNR": 22.0},
            {"method": "bm3d", "sigma": 0.1, "window": [3, 3], "PSNR": 18.0},
        ],
        group_by=("method", "iterations", "sigma", "window"),
    )

    assert len(summaries) == 3
    assert summaries[0]["PSNR_mean"] == 20.0
    assert summaries[0]["sigma"] is None
    assert summaries[2]["window"] == [3, 3]


def test_runner_writes_benchmark_artifact_tree(tmp_path, monkeypatch):
    """A benchmark run writes result tables, predictions, residuals, and metadata."""
    def fake_write_results_tables(records, output_dir, group_by):
        assert records[0]["photons"] == 100
        assert records[0]["iterations"] == 0
        assert group_by == ("method", "photons", "iterations")
        runs_path = output_dir / "runs.parquet"
        summary_path = output_dir / "summary.parquet"
        runs_path.write_bytes(b"mock parquet")
        summary_path.write_bytes(b"mock parquet")
        return runs_path, summary_path

    monkeypatch.setattr(benchmark_runner, "write_results_tables", fake_write_results_tables)
    reference = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    samples = [
        BenchmarkSample(
            image_id="cell-1",
            observed=reference + 0.01,
            reference=reference,
            seed=2,
            parameters={"photons": 100},
        )
    ]
    experiments = [Experiment(1, "identity", lambda image: image, {"iterations": 0})]

    output_dir = BenchmarkRunner(tmp_path).run(samples, experiments, benchmark_id="benchmark-test")

    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "results" / "runs.parquet").is_file()
    assert (output_dir / "results" / "summary.parquet").is_file()
    assert (output_dir / "predictions" / "identity").is_dir()
    residual_path = next((output_dir / "images").glob("*.npy"))
    assert np.allclose(np.load(residual_path), -0.01)
    assert (output_dir / "logs" / "benchmark.log").is_file()

def test_benchmark_runner_saves_one_preview_per_experiment(tmp_path):
    """Preview images are saved per experiment when prediction saving is disabled."""
    reference = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    samples = [BenchmarkSample(image_id=i, observed=reference * 0.5, reference=reference) for i in range(3)]
    experiments = [
        Experiment(1, "identity", lambda image: image),
        Experiment(2, "clip", lambda image: np.clip(image, 0, 0.3)),
    ]

    output_dir = BenchmarkRunner(tmp_path).run(
        samples, experiments, benchmark_id="preview", save_predictions=False, image_samples=2
    )

    assert len(list((output_dir / "images").glob("*.png"))) == 2
