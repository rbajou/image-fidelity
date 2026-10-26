"""Generic multi-format writers for images and array data.

Mirrors :mod:`image_fidelity.io.readers`: PNG/JPEG/BMP (Pillow), TIFF (tifffile), NumPy
(.npy/.npz), Zarr and HDF5 (optional dependencies, imported lazily).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .readers import UnsupportedFormatError

_WRITERS: Dict[str, Callable[..., None]] = {}


def register_writer(*extensions: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Register a writer function for one or more (dot-prefixed) file extensions."""

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        for ext in extensions:
            _WRITERS[ext.lower()] = func
        return func

    return decorator


@register_writer(".png", ".jpg", ".jpeg", ".bmp")
def _write_pil(path: Path, array: np.ndarray, **kwargs: Any) -> None:
    from PIL import Image

    Image.fromarray(array).save(path, **kwargs)


@register_writer(".tif", ".tiff")
def _write_tiff(path: Path, array: np.ndarray, **kwargs: Any) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Writing TIFF files requires the 'tifffile' package. Install it with `pip install tifffile`."
        ) from exc
    tifffile.imwrite(path, array, **kwargs)


@register_writer(".npy")
def _write_npy(path: Path, array: np.ndarray, **kwargs: Any) -> None:
    np.save(path, array, **kwargs)


@register_writer(".npz")
def _write_npz(path: Path, array: np.ndarray, key: str = "data", **kwargs: Any) -> None:
    np.savez(path, **{key: array}, **kwargs)


@register_writer(".zarr")
def _write_zarr(path: Path, array: np.ndarray, key: Optional[str] = None, **kwargs: Any) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Writing Zarr stores requires the 'zarr' package. Install it with `pip install zarr`."
        ) from exc
    if key is None:
        zarr.save(str(path), array, **kwargs)
    else:
        zarr.save(str(path), **{key: array}, **kwargs)


@register_writer(".h5", ".hdf5")
def _write_hdf5(path: Path, array: np.ndarray, key: str = "data", **kwargs: Any) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Writing HDF5 files requires the 'h5py' package. Install it with `pip install h5py`."
        ) from exc
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=array, **kwargs)


def write_image(path: str | Path, array: np.ndarray, **kwargs: Any) -> None:
    """Write an image or array to disk, dispatching on the file extension.

    Extra keyword arguments are forwarded to the format-specific writer
    (e.g. ``key=`` to name the array inside an .npz/.zarr/.h5 file).
    """
    path = Path(path)
    writer = _WRITERS.get(path.suffix.lower())
    if writer is None:
        raise UnsupportedFormatError(
            f"No writer registered for extension '{path.suffix}'. Supported: {sorted(_WRITERS)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(path, array, **kwargs)
