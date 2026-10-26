import numpy as np
import pandas as pd

from image_fidelity.visualization import save_run_figures, save_sweep_figures, swept_parameters
from image_fidelity.visualization.metrics import RESTORATION_METRIC_COLUMNS


def _runs(methods, rows=6):
    rng = np.random.default_rng(0)
    return [
        {"method": method, **{metric: float(rng.normal()) for metric in RESTORATION_METRIC_COLUMNS}}
        for method in methods
        for _ in range(rows)
    ]


def test_run_figures_without_method_comparison(tmp_path):
    paths = save_run_figures(_runs(["gaussian"]), tmp_path)

    assert [path.name for path in paths] == ["metric_distributions.png", "metric_boxplots.png"]
    assert all(path.is_file() for path in paths)


def test_run_figures_compare_methods_only_with_several_methods(tmp_path):
    single = save_run_figures(_runs(["gaussian"]), tmp_path / "single", boxplot_by_method=True)
    several = save_run_figures(_runs(["gaussian", "median"]), tmp_path / "several", boxplot_by_method=True)

    assert [path.name for path in single] == ["metric_distributions.png"]
    assert [path.name for path in several] == ["metric_distributions.png", "metric_boxplots_by_method.png"]


def test_sweep_figures_plot_each_swept_parameter(tmp_path):
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
