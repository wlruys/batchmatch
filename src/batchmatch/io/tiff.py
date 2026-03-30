"""Load multi-channel TIFF files (ImageJ and OME-TIFF) as PyTorch tensors.

Produces BCHW float32 tensors normalised to [0, 1], matching the format
returned by ``batchmatch.io.ImageIO.load``.

Supports memory-efficient loading for large images via:
- Single-channel selection (``channel=``)
- Spatial region cropping (``region=``)
- Area-average downsampling on load (``downsample=``)
- Pyramid-aware downsampling (``downsample_mode="pyramid"``)
- Lazy/streaming reads (``return_mode="lazy"``, requires zarr)
- Metadata-only reads (``read_meta()``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch

from batchmatch.io.convert import to_bchw_float
from batchmatch.io.downsample import area_downsample, crop_region, validate_region
from batchmatch.io.tiff_lazy import LazyTiffReader
from batchmatch.io.tiff_meta import TiffMeta, _parse_meta, read_meta
from batchmatch.io.tiff_read import (
    find_best_pyramid_level,
    read_all_channels,
    read_pyramid_level,
    read_single_channel,
    resolve_channel_index,
)

Tensor = torch.Tensor

DownsampleMode = Literal["area", "pyramid"]
ReturnMode = Literal["tensor", "lazy"]

__all__ = [
    "load_tiff",
    "read_meta",
    "TiffMeta",
    "LazyTiffReader",
    "DownsampleMode",
    "ReturnMode",
]


def load_tiff(
    path: str | Path,
    channel: int | str | None = None,
    region: tuple[int, int, int, int] | None = None,
    downsample: int = 1,
    *,
    return_mode: ReturnMode = "tensor",
    level: int | Literal["auto"] = "auto",
    downsample_mode: DownsampleMode = "area",
) -> "tuple[Tensor, TiffMeta] | tuple[LazyTiffReader, TiffMeta]":
    """Load a TIFF file and return a (tensor, metadata) tuple.

    Supports ImageJ multi-page TIFFs and OME-TIFF.

    When *channel* is specified and the file stores one channel per page
    (the common case for both formats), only that page is read — avoiding
    allocation and decompression of the other channels entirely.

    Args:
        path:            Path to the TIFF file.
        channel:         Select a single channel by index or name.  ``None``
                         keeps all.
        region:          Spatial crop as ``(y_start, x_start, height, width)``
                         in pixels.  Applied *before* downsampling.
        downsample:      Integer factor for Y and X axes (e.g. 4 = quarter res).
        return_mode:     ``"tensor"`` (default) returns a torch Tensor;
                         ``"lazy"`` returns a :class:`LazyTiffReader` (needs zarr).
        level:           Pyramid level — ``"auto"`` picks the best level for the
                         requested *downsample*, or pass an int to select explicitly.
        downsample_mode: ``"area"`` (anti-aliased block average, default) or
                         ``"pyramid"`` (use pyramid level first, then area for
                         any remainder).

    Returns:
        tensor: BCHW ``float32`` tensor normalised to [0, 1] (or ``LazyTiffReader``).
        meta:   :class:`TiffMeta` with raw and output axes/shape/pixel size.
    """
    path = Path(path)

    with tifffile.TiffFile(path) as tif:
        meta = _parse_meta(tif)

        # --- pyramid level selection ----------------------------------------
        pyramid_level = 0
        remaining_ds = downsample
        if downsample > 1 and downsample_mode == "pyramid":
            level = "auto"
        if level == "auto" and downsample > 1:
            pyramid_level, remaining_ds = find_best_pyramid_level(tif, downsample)
        elif isinstance(level, int) and level > 0:
            pyramid_level = level
            remaining_ds = downsample

        meta.pyramid_level_used = pyramid_level

        # --- lazy mode: return early ----------------------------------------
        if return_mode == "lazy":
            reader = LazyTiffReader(
                path, meta, channel=channel, level=pyramid_level,
            )
            return reader, meta

        # --- eager read -----------------------------------------------------
        if channel is not None:
            idx = resolve_channel_index(
                channel, meta.channel_names, meta.axes_raw,
            )
            if pyramid_level > 0:
                arr = read_pyramid_level(tif, pyramid_level)
                c_axis = meta.axes_raw.find("C")
                arr = np.take(arr, idx, axis=c_axis)
            else:
                arr = read_single_channel(tif, meta, idx)
            axes = meta.axes_raw.replace("C", "")
            channel_names = [meta.channel_names[idx]] if meta.channel_names else []
        else:
            if pyramid_level > 0:
                arr = read_pyramid_level(tif, pyramid_level)
            else:
                arr = read_all_channels(tif)
            axes = meta.axes_raw
            channel_names = list(meta.channel_names)

    # --- validate and apply region ------------------------------------------
    if region is not None:
        y_ax = axes.find("Y")
        x_ax = axes.find("X")
        region = validate_region(
            region, arr.shape[y_ax], arr.shape[x_ax],
        )
        arr = crop_region(arr, axes, region)

    # --- downsample ---------------------------------------------------------
    if remaining_ds > 1:
        arr = area_downsample(arr, axes, remaining_ds)

    # --- convert to tensor --------------------------------------------------
    tensor = to_bchw_float(arr)

    # --- update output metadata ---------------------------------------------
    meta.axes = axes
    meta.shape = tuple(tensor.shape)
    meta.channel_names = channel_names
    effective_ds = downsample
    if meta.pixel_size_um_raw is not None:
        meta.pixel_size_um = meta.pixel_size_um_raw * effective_ds
    elif meta.pixel_size_um is not None and effective_ds > 1:
        meta.pixel_size_um = meta.pixel_size_um * effective_ds

    return tensor, meta
