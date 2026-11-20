import numpy as np
import pandas as pd
import pytest

from image_fidelity.visualization import save_run_figures, save_sweep_figures, swept_parameters
from image_fidelity.visualization import metrics as metrics_module
from image_fidelity.visualization.metrics import RESTORATION_METRIC_COLUMNS


def _runs(methods, rows=6):
    rng = np.random.default_rng(0)
    return [
        {"method": method, **{metric: float(rng.normal()) for metric in RESTORATION_METRIC_COLUMNS}}
        for method in methods
        for _ in range(rows)
    ]


def test_run_figures_without_method_comparison(tmp_path):
    """Run-level plots are generated when no method comparison is requested."""
    paths = save_run_figures(_runs(["gaussian"]), tmp_path)

    assert [path.name for path in paths] == ["metric_distributions.png", "metric_boxplots.png"]
    assert all(path.is_file() for path in paths)


def test_run_figures_compare_methods_only_with_several_methods(tmp_path):
    """Method comparison plots appear only when multiple methods are present."""
    single = save_run_figures(_runs(["gaussian"]), tmp_path / "single", boxplot_by_method=True)
    several = save_run_figures(_runs(["gaussian", "median"]), tmp_path / "several", boxplot_by_method=True)

    assert [path.name for path in single] == ["metric_distributions.png"]
    assert [path.name for path in several] == ["metric_distributions.png", "metric_boxplots_by_method.png"]


def test_run_figures_split_distributions_by_method(tmp_path, monkeypatch):
    """Each method's histogram receives only its own per-image scores."""
    records = [
        {"method": "unet", "dice": 0.9},
        {"method": "watershed", "dice": 0.2},
        {"method": "unet", "dice": 0.8},
    ]
    plotted = {}
    original = metrics_module.plot_metric_distributions

    def capture(frame, path, metrics=None):
        plotted[path.name] = (frame["method"].tolist(), frame["dice"].tolist())
        return original(frame, path, metrics)

    monkeypatch.setattr(metrics_module, "plot_metric_distributions", capture)
    paths = save_run_figures(records, tmp_path, distribution_by_method=True, boxplot_by_method=True)

    assert plotted == {
        "metric_distributions_unet.png": (["unet", "unet"], [0.9, 0.8]),
        "metric_distributions_watershed.png": (["watershed"], [0.2]),
    }
    assert [path.name for path in paths] == [
        "metric_distributions_unet.png",
        "metric_distributions_watershed.png",
        "metric_boxplots_by_method.png",
    ]
    assert all(path.is_file() for path in paths)


def test_split_distributions_reject_filename_collisions(tmp_path):
    records = [{"method": "a/b", "dice": 1.0}, {"method": "a b", "dice": 0.5}]
    with pytest.raises(ValueError, match="duplicate distribution filenames"):
        save_run_figures(records, tmp_path, distribution_by_method=True)


def test_split_distributions_with_one_method(tmp_path):
    paths = save_run_figures(
        [{"method": "watershed", "dice": 0.8}], tmp_path,
        distribution_by_method=True, boxplot_by_method=True,
    )
    assert [path.name for path in paths] == ["metric_distributions_watershed.png"]
    assert paths[0].is_file()


def test_sweep_figures_plot_each_swept_parameter(tmp_path):
    """Sweep plotting creates a metric chart for each varying parameter."""
    summary = pd.DataFrame(
        [
            {
                "method": "gaussian",
                "degradation": "GaussianNoise",
                "degradation_sigma": sigma,
                "sigma": 1.0,
                "n": 3,
                **{f"{metric}_{stat}": value for metric in RESTORATION_METRIC_COLUMNS for stat, value in (("mean", sigma), ("std", 0.1))},
            }
            for sigma in (0.1, 0.2, 0.5)
        ]
    )
    summary_path = tmp_path / "summary.parquet"
    summary.to_parquet(summary_path, index=False)

    assert swept_parameters(summary) == ["degradation_sigma"]
    paths = save_sweep_figures(summary_path, tmp_path / "figures")

    assert [path.name for path in paths] == ["metrics_vs_degradation_sigma.png"]
    assert paths[0].is_file()
