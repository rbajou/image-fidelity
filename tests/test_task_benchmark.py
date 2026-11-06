from image_fidelity.benchmark import runner as benchmark_runner
from image_fidelity.tasks import dispatch


def test_benchmark_dispatch_runs_configured_methods(compose_task_config, monkeypatch):
    """The benchmark task runs configured methods and writes its result tables."""

    def fake_write_results_tables(records, output_dir, group_by):
        assert len(records) == 4
        runs_path = output_dir / "runs.parquet"
        summary_path = output_dir / "summary.parquet"
        runs_path.write_bytes(b"test parquet")
        summary_path.write_bytes(b"test parquet")
        return runs_path, summary_path

    monkeypatch.setattr(benchmark_runner, "write_results_tables", fake_write_results_tables)
    config = compose_task_config("experiment=benchmark_restore_classical")

    output_dir = dispatch(config)

    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "results" / "runs.parquet").is_file()
    assert not (output_dir / "predictions").exists()
