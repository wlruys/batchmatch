"""Internal low-level TIFF array reading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from batchmatch.io._tiff_meta import TiffMeta

__all__ = [
    "resolve_channel_index",
    "read_single_channel",
    "read_all_channels",
    "find_best_pyramid_level",
    "read_pyramid_level",
]


def resolve_channel_index(
    channel: int | str,
    channel_names: list[str],
    axes: str,
) -> int:
    if "C" not in axes:
        raise ValueError(
            f"channel={channel!r} requested but file has no C axis "
            f"(axes={axes!r}).  Omit the channel argument for this file."
        )
    if isinstance(channel, int):
        return channel
    if channel in channel_names:
        return channel_names.index(channel)
    lower = channel.lower()
    for i, name in enumerate(channel_names):
        if name.lower() == lower:
            return i
    raise ValueError(
        f"Channel name {channel!r} not found. "
        f"Available channels: {channel_names}"
    )


def read_single_channel(
    tif: tifffile.TiffFile,
    meta: TiffMeta,
    channel_idx: int,
) -> np.ndarray:
    series = tif.series[0]
    c_axis = meta.axes.find("C")
    if c_axis < 0 or len(series.pages) <= 1:
        arr = series.asarray()
        return np.take(arr, channel_idx, axis=c_axis)

    page = series.pages[channel_idx]

    if page.parent is not tif:
        parent_path = Path(tif.filehandle.dirname) / page.parent.filename
        with tifffile.TiffFile(parent_path) as t2:
            return t2.pages[0].asarray()

    return page.asarray()


def read_all_channels(tif: tifffile.TiffFile) -> np.ndarray:
    return tif.series[0].asarray()


def find_best_pyramid_level(
    tif: tifffile.TiffFile,
    downsample: int,
) -> tuple[int, int]:
    series0 = tif.series[0]
    levels: list[tifffile.TiffPageSeries] = []
    if hasattr(series0, "levels"):
        levels = list(series0.levels)
    else:
        levels = list(tif.series)

    if len(levels) <= 1:
        return 0, downsample

    base_shape = levels[0].shape
    base_y = base_shape[-2] if len(base_shape) >= 2 else base_shape[-1]

    best_level = 0
    best_remaining = downsample
    for i, lvl in enumerate(levels):
        lvl_y = lvl.shape[-2] if len(lvl.shape) >= 2 else lvl.shape[-1]
        if lvl_y == 0:
            continue
        effective_ds = base_y / lvl_y
        if effective_ds <= downsample:
            remaining = max(1, round(downsample / effective_ds))
            if remaining < best_remaining or (
                remaining == best_remaining and effective_ds > 1
            ):
                best_level = i
                best_remaining = remaining

    return best_level, best_remaining


def read_pyramid_level(
    tif: tifffile.TiffFile,
    level: int,
) -> np.ndarray:
    series0 = tif.series[0]
    if hasattr(series0, "levels"):
        return series0.levels[level].asarray()
    return tif.series[level].asarray()
