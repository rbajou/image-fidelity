import numpy as np
import pandas as pd

from image_fidelity.tasks import dispatch


def test_restore_dispatch_writes_prediction_and_metrics(compose_task_config):
    """The restore task saves a prediction, metric records, and figures."""
    config = compose_task_config(
        "task=restore",
        "method=gaussian",
        "degradation=gaussian_noise",
        "degradation.sigma=0.05",
    )

    output_dir = dispatch(config)

    prediction = np.load(output_dir / "predictions" / "sample-1.npy")
    runs = pd.read_parquet(output_dir / "results" / "runs.parquet").to_dict("records")
    assert prediction.shape == (24, 24)
    assert runs[0]["image_id"] == "sample-1"
    assert runs[0]["PSNR"] > 0
    assert set(runs[0]) >= {"SSIM", "NRMSE", "grad_error", "bias", "flux_ratio", "runtime"}
    assert (output_dir / "figures" / "metric_distributions.png").is_file()
    assert (output_dir / "figures" / "metric_boxplots.png").is_file()
