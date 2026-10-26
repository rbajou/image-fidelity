"""Generic multi-format readers for images and array data.

Supported out of the box: PNG, JPEG, BMP (Pillow), TIFF (tifffile), NumPy
(.npy/.npz), Zarr and HDF5 (optional dependencies, imported lazily).
New formats can be added at runtime via :func:`register_reader`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


class UnsupportedFormatError(ValueError):
    """Raised when no reader is registered for a given file extension."""


_READERS: Dict[str, Callable[..., np.ndarray]] = {}


def register_reader(*extensions: str) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register a reader function for one or more (dot-prefixed) file extensions."""

    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        for ext in extensions:
            _READERS[ext.lower()] = func
        return func

    return decorator


@register_reader(".png", ".jpg", ".jpeg", ".bmp", ".gif")
def _read_pil(path: Path, mode: Optional[str] = None, **_: Any) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        if mode is not None:
            img = img.convert(mode)
        return np.array(img)


@register_reader(".tif", ".tiff")
def _read_tiff(path: Path, **kwargs: Any) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Reading TIFF files requires the 'tifffile' package. Install it with `pip install tifffile`."
        ) from exc
    return tifffile.imread(path, **kwargs)


@register_reader(".npy")
def _read_npy(path: Path, **kwargs: Any) -> np.ndarray:
    return np.load(path, **kwargs)


@register_reader(".npz")
def _read_npz(path: Path, key: Optional[str] = None, **kwargs: Any) -> np.ndarray:
    with np.load(path, **kwargs) as data:
        key = key or data.files[0]
        return data[key]


@register_reader(".zarr")
def _read_zarr(path: Path, key: Optional[str] = None, **kwargs: Any) -> np.ndarray:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Reading Zarr stores requires the 'zarr' package. Install it with `pip install zarr`."
        ) from exc
    store = zarr.open(str(path), mode="r", **kwargs)
    array = store[key] if key is not None else store
    return np.asarray(array)


@register_reader(".h5", ".hdf5")
def _read_hdf5(path: Path, key: str = "data", **kwargs: Any) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5 files requires the 'h5py' package. Install it with `pip install h5py`."
        ) from exc
    with h5py.File(path, "r", **kwargs) as handle:
        if key not in handle:
            raise KeyError(f"Dataset key '{key}' not found in {path}. Available keys: {list(handle.keys())}")
        return handle[key][()]


def read_image(path: str | Path, **kwargs: Any) -> np.ndarray:
    """Read an image or array from disk, dispatching on the file extension.

    Extra keyword arguments are forwarded to the format-specific reader
    (e.g. ``key=`` to select a dataset/array inside an .npz/.zarr/.h5 file).
    """
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise UnsupportedFormatError(
            f"No reader registered for extension '{path.suffix}'. Supported: {sorted(_READERS)}"
        )
    return reader(path, **kwargs)