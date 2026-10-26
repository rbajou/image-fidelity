"""Lightweight metadata extraction for image/array files.

Avoids loading full pixel data when the underlying format allows it
(Pillow header parsing, TIFF/Zarr/HDF5 lazy handles, memory-mapped NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ImageMetadata:
    """Format-agnostic description of an image/array file."""

    path: Path
    format: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    extra: Dict[str, Any] = field(default_factory=dict)


def _pil_metadata(path: Path) -> ImageMetadata:
    from PIL import Image

    _MODE_TO_DTYPE = {"1": "bool", "L": "uint8", "P": "uint8", "I": "int32", "F": "float32"}
    with Image.open(path) as img:
        n_bands = len(img.getbands())
        shape = (img.height, img.width) if n_bands == 1 else (img.height, img.width, n_bands)
        return ImageMetadata(
            path=path,
            format=img.format or path.suffix.lstrip(".").upper(),
            shape=shape,
            dtype=_MODE_TO_DTYPE.get(img.mode, "uint8"),
            size_bytes=path.stat().st_size,
            extra={"mode": img.mode},
        )


def _tiff_metadata(path: Path) -> ImageMetadata:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Reading TIFF metadata requires the 'tifffile' package. Install it with `pip install tifffile`."
        ) from exc
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return ImageMetadata(
            path=path,
            format="TIFF",
            shape=tuple(series.shape),
            dtype=str(series.dtype),
            size_bytes=path.stat().st_size,
            extra={"n_pages": len(tif.pages)},
        )


def _npy_metadata(path: Path) -> ImageMetadata:
    array = np.load(path, mmap_mode="r")
    return ImageMetadata(
        path=path, format="NPY", shape=array.shape, dtype=str(array.dtype), size_bytes=path.stat().st_size,
    )


def _npz_metadata(path: Path, key: Optional[str] = None) -> ImageMetadata:
    with np.load(path) as data:
        chosen_key = key or data.files[0]
        arr = data[chosen_key]
        return ImageMetadata(
            path=path,
            format="NPZ",
            shape=arr.shape,
            dtype=str(arr.dtype),
            size_bytes=path.stat().st_size,
            extra={"keys": data.files},
        )


def _zarr_metadata(path: Path, key: Optional[str] = None) -> ImageMetadata:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Reading Zarr metadata requires the 'zarr' package. Install it with `pip install zarr`."
        ) from exc
    store = zarr.open(str(path), mode="r")
    array = store[key] if key is not None else store
    return ImageMetadata(
        path=path,
        format="Zarr",
        shape=array.shape,
        dtype=str(array.dtype),
        size_bytes=sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
        extra={"chunks": getattr(array, "chunks", None)},
    )


def _hdf5_metadata(path: Path, key: str = "data") -> ImageMetadata:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5 metadata requires the 'h5py' package. Install it with `pip install h5py`."
        ) from exc
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"Dataset key '{key}' not found in {path}. Available keys: {list(handle.keys())}")
        dataset = handle[key]
        return ImageMetadata(
            path=path,
            format="HDF5",
            shape=dataset.shape,
            dtype=str(dataset.dtype),
            size_bytes=path.stat().st_size,
            extra={"keys": list(handle.keys())},
        )


_METADATA_READERS = {
    ".png": _pil_metadata,
    ".jpg": _pil_metadata,
    ".jpeg": _pil_metadata,
    ".bmp": _pil_metadata,
    ".tif": _tiff_metadata,
    ".tiff": _tiff_metadata,
    ".npy": _npy_metadata,
    ".npz": _npz_metadata,
    ".zarr": _zarr_metadata,
    ".h5": _hdf5_metadata,
    ".hdf5": _hdf5_metadata,
}


def extract_metadata(path: str | Path, **kwargs: Any) -> ImageMetadata:
    """Extract shape/dtype/format metadata for a file, dispatching on extension."""
    path = Path(path)
    reader = _METADATA_READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"No metadata reader registered for extension '{path.suffix}'.")
    return reader(path, **kwargs)
