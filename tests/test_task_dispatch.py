import pytest

from image_fidelity.tasks import dispatch


def test_dispatch_rejects_unimplemented_task(compose_task_config):
    """Dispatch reports the supported task names for an unknown task."""
    config = compose_task_config()
    config.task.name = "unknown"

    with pytest.raises(
        ValueError,
        match="implemented tasks: benchmark, degrade, download, plot, restore, segment, train",
    ):
        dispatch(config)
