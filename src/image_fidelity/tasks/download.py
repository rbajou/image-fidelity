"""Download task handler: fetch public datasets into ``paths.data_dir``."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..datasets.download import download_datasets


def _as_list(value) -> list:
    """Normalize a config value (None, scalar, ListConfig or empty list) to a plain list."""
    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return list(value) if isinstance(value, (list, tuple)) else [value]


def run_download(config: DictConfig) -> Path:
    """Download ``task.datasets`` of ``task.collection`` into ``paths.data_dir``."""
    task = config.task
    dataset_ids = _as_list(task.get("datasets"))
    if not dataset_ids:
        raise ValueError("Download task has no datasets; run e.g. `imfid download bbbc 038`")
    data_dir = Path(str(config.paths.data_dir))
    files = _as_list(task.get("files"))
    directories = download_datasets(
        str(task.collection),
        dataset_ids,
        data_dir,
        files=files or None,
        extract=bool(task.get("extract", True)),
        keep_archives=bool(task.get("keep_archives", True)),
        force=bool(task.get("force", False)),
    )
    return directories[0] if len(directories) == 1 else data_dir
