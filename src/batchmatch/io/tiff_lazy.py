"""Lazy TIFF reader backed by tifffile's zarr interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import tifffile

from batchmatch.io.convert import to_bchw_float
from batchmatch.io.downsample import crop_region
from batchmatch.io.tiff_meta import TiffMeta
from batchmatch.io.tiff_read import resolve_channel_index

if TYPE_CHECKING:
    import zarr as _zarr_mod

Tensor = torch.Tensor

__all__ = ["LazyTiffReader"]


class LazyTiffReader:
    """Lazy reader backed by tifffile's zarr interface.

    Requires the ``zarr`` package.  Provides ``__getitem__`` for spatial
    slicing and a ``.read()`` / ``.to_tensor()`` method to materialise.
    """

    def __init__(
        self,
        path: str | Path,
        meta: TiffMeta,
        *,
        channel: int | str | None = None,
        level: int = 0,
    ) -> None:
        self.path = Path(path)
        self.meta = meta
        self._channel = channel
        self._level = level
        self._store: _zarr_mod.Array | None = None

    def _open(self) -> "_zarr_mod.Array":
        if self._store is not None:
            return self._store
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "Lazy mode requires the 'zarr' package. "
                "Install it with: pip install zarr"
            ) from exc
        store = tifffile.imread(self.path, aszarr=True, level=self._level)
        self._store = zarr.open(store, mode="r")
        return self._store

    @property
    def shape(self) -> tuple[int, ...]:
        return self._open().shape

    @property
    def dtype(self) -> np.dtype:
        return self._open().dtype

    def __getitem__(self, key: tuple) -> np.ndarray:
        """Spatial slicing — reads only the requested region."""
        arr = self._open()
        data = np.asarray(arr[key])
        return data

    def read(
        self,
        region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Materialise the (optionally cropped) array."""
        arr = self._open()
        axes = self.meta.axes_raw

        if self._channel is not None:
            idx = resolve_channel_index(
                self._channel, self.meta.channel_names, axes
            )
            c_ax = axes.find("C")
            slices = [slice(None)] * len(arr.shape)
            slices[c_ax] = idx
            data = np.asarray(arr[tuple(slices)])
            axes = axes.replace("C", "")
        else:
            data = np.asarray(arr)

        if region is not None:
            data = crop_region(data, axes, region)

        return data

    def to_tensor(
        self,
        region: tuple[int, int, int, int] | None = None,
        normalize: bool = True,
    ) -> Tensor:
        """Read and convert to a BCHW float32 tensor."""
        arr = self.read(region=region)
        if normalize:
            return to_bchw_float(arr)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
        while t.ndim < 4:
            t = t.unsqueeze(0)
        return t

    def close(self) -> None:
        self._store = None

    def __enter__(self) -> "LazyTiffReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"LazyTiffReader({self.path.name!r}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )
