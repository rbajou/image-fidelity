from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from image_fidelity.restoration.classical import RichardsonLucyDeconvolution
from image_fidelity.restoration.nn import TorchRestorer, UNet
from image_fidelity.tasks.callbacks import AggregateBenchmarkResults

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_hydra_configs_compose_classical_benchmark():
    """The classical benchmark experiment composes its task and methods."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="config", overrides=["experiment=benchmark_restore_classical"])

    assert config.task.name == "benchmark"
    assert set(config.task.methods) == {"gaussian", "median", "nlm", "tv"}
    assert config.degradation.sigma == pytest.approx(0.1)


def test_hydra_callback_is_registered_from_tasks_package():
    """Hydra registers the benchmark aggregation callback from the task package."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="config", return_hydra_config=True)

    callback_config = config.hydra.callbacks.aggregate_benchmark_results
    assert callback_config._target_ == "image_fidelity.tasks.callbacks.AggregateBenchmarkResults"
    assert isinstance(instantiate(callback_config), AggregateBenchmarkResults)


@pytest.mark.parametrize(
    ("experiment_name", "expected_task", "expected_run_name"),
    [
        ("benchmark_restore_classical", "benchmark", "classical_bbbc038"),
        ("benchmark_restore_sweep", "benchmark", "sweep_bbbc010"),
        ("benchmark_restore_unet_static_vs_variable_nl", "benchmark", "unet_static_vs_variable_nl"),
        ("benchmark_restore_unet_vs_classical", "benchmark", "unet_vs_classical_bbbc038"),
        ("benchmark_segment_classical", "benchmark", "segment_classical"),
        ("finetune_unet", "train", "unet_bbbc038"),
        ("restore_nlm", "restore", "nlm_bbbc038"),
        ("restore_unet", "restore", "unet_bbbc038"),
        ("segment_unet", "segment", "unet_bbbc038"),
        ("segment_watershed", "segment", "watershed_bbbc038"),
        ("train_unet", "train", "unet_bbbc038"),
    ],
)
def test_experiment_run_names_are_included_in_output_dirs(experiment_name, expected_task, expected_run_name):
    """Named experiments include the task and experiment name in output paths."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=[f"experiment={experiment_name}"],
            return_hydra_config=True,
        )

    assert config.task.name == expected_task
    assert config.run_name == expected_run_name
    assert config.hydra.run.dir.startswith(f"outputs/{expected_task}/")
    assert config.hydra.run.dir.endswith(f"_{expected_run_name}")
    assert config.hydra.sweep.dir == config.hydra.run.dir


@pytest.mark.parametrize("task_name", ["benchmark", "restore", "segment", "train"])
def test_task_name_is_output_suffix_without_named_experiment(task_name):
    """Task-only runs include the task name in their output directory."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=[f"task={task_name}"],
            return_hydra_config=True,
        )

    assert config.hydra.run.dir.startswith(f"outputs/{task_name}/")
    assert config.hydra.run.dir.endswith(f"_{task_name}")


@pytest.mark.parametrize("method_name", ["gaussian", "nlm", "tv"])
def test_hydra_configs_compose_classical_method_overrides(method_name):
    """Classical restoration method overrides instantiate successfully."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=["task=restore", f"method={method_name}", "degradation=gaussian_noise"],
        )

    assert config.task.name == "restore"
    assert "_target_" in config.method
    assert instantiate(config.method) is not None


def test_hydra_config_instantiates_richardson_lucy():
    """The Richardson-Lucy config creates a normalized kernel and iteration count."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="config", overrides=["method=richardson_lucy"])

    method = instantiate(config.method)

    assert isinstance(method, RichardsonLucyDeconvolution)
    assert method.kernel.shape == (3, 3)
    assert method.kernel.sum() == pytest.approx(1.0)
    assert method.n_iter == 30


def test_hydra_config_instantiates_neural_restorer_and_model():
    """The U-Net restoration config creates a TorchRestorer with a U-Net model."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=["task=restore", "method=unet_restorer", "method.checkpoint=null"],
        )

    restorer = instantiate(config.method)

    assert isinstance(restorer, TorchRestorer)
    assert isinstance(restorer.model, UNet)
