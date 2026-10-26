from pathlib import Path

import pandas as pd

from image_fidelity.benchmark.aggregation import aggregate_benchmark_runs


def _write_job_results(job_dir: Path, sigma: float, psnr: float) -> None:
    results_dir = job_dir / "results"
    results_dir.mkdir(parents=True)
    row = {
        "image_id": "sample-1",
        "method": "gaussian",
        "degradation": "GaussianNoise",
        "degradation_sigma": sigma,
        "PSNR": psnr,
    }
    pd.DataFrame([row]).to_parquet(results_dir / "runs.parquet", index=False)
    summary = {key: row[key] for key in ("method", "degradation", "degradation_sigma")}
    pd.DataFrame([{**summary, "n": 1, "PSNR_mean": psnr}]).to_parquet(results_dir / "summary.parquet", index=False)


def test_aggregate_benchmark_runs_concatenates_runs_and_summaries_per_job(tmp_path):
    for job_num, (sigma, psnr) in enumerate([(0.1, 24.9), (0.2, 21.3), (0.5, 14.8)]):
        _write_job_results(tmp_path / str(job_num), sigma, psnr)

    aggregated_path = aggregate_benchmark_runs(tmp_path)

    assert aggregated_path == tmp_path / "results" / "runs.parquet"
    assert sorted(pd.read_parquet(aggregated_path)["degradation_sigma"]) == [0.1, 0.2, 0.5]

    summary = pd.read_parquet(tmp_path / "results" / "summary.parquet")
    assert sorted(summary["degradation_sigma"]) == [0.1, 0.2, 0.5]
    assert set(summary["degradation"]) == {"GaussianNoise"}


def test_aggregate_benchmark_runs_returns_none_without_results(tmp_path):
    assert aggregate_benchmark_runs(tmp_path) is None
