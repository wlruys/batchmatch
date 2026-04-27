"""Final-resolution export of a registered moving image.

Consumes :class:`RegistrationTransform` (or a manifest round-trip) and
applies :attr:`RegistrationTransform.matrix_ref_full_from_mov_full` to
every moving channel in one pass. Registration may use a single channel;
export reopens the moving source with ``channels="all"`` by default.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, Sequence, Union

import numpy as np
import torch
import tifffile
import zarr

from batchmatch.io.downsample import area_downsample
from batchmatch.io.convert import to_bchw_float
from batchmatch.io.images import (
    ImageIO,
    TiffExportConfig,
    _build_ome_metadata,
    _tiff_resolution_from_source,
    load_image,
)
from batchmatch.io.schema import REGISTRATION_SCHEMA
from batchmatch.io.space import SourceInfo
from batchmatch.io.utils import PathLike, assert_overwrite, ensure_parent_dir, pathify

if TYPE_CHECKING:
    from batchmatch.search.transform import RegistrationTransform
from batchmatch.warp.resample import warp_to_reference

__all__ = [
    "RegisteredExport",
    "export_moving_mask_tiff",
    "export_registered",
    "export_stacked_registered_tiff",
]


Channels = Union[Literal["all"], int, str, Sequence[Union[int, str]]]
Canvas = Literal["reference", "union"]

_TIFF_EXTENSIONS = frozenset({".tif", ".tiff"})
_DEFAULT_STREAM_TILE_SIZE = 512
_DEFAULT_PYRAMID_LEVELS = 4
_TILE_MARGIN = 2


@dataclass(frozen=True)
class RegisteredExport:
    """Metadata returned by ``export_registered(..., return_metadata=True)``."""

    path: pathlib.Path
    canvas: Canvas
    output_source: SourceInfo
    bbox_ref_full_xyxy: tuple[int, int, int, int]
    matrix_canvas_from_mov_full: np.ndarray
    mask_path: pathlib.Path | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "canvas": self.canvas,
            "mask": self.mask_path.name if self.mask_path is not None else None,
            "bbox_ref_full_xyxy": list(self.bbox_ref_full_xyxy),
            "matrix_canvas_from_mov_full": self.matrix_canvas_from_mov_full.tolist(),
            "source": self.output_source.to_dict(),
        }


def _load_transform(
    source: Union["RegistrationTransform", dict[str, Any], PathLike],
) -> tuple["RegistrationTransform", Optional[dict[str, Any]]]:
    from batchmatch.search.transform import RegistrationTransform

    if isinstance(source, RegistrationTransform):
        return source, None
    if isinstance(source, dict):
        manifest = source
    else:
        manifest = json.loads(pathify(source).read_text(encoding="utf-8"))
    if manifest.get("schema") != REGISTRATION_SCHEMA.name:
        raise ValueError("export_registered expects a registration manifest.")
    transform = RegistrationTransform.from_manifest(manifest)
    return transform, manifest


def _ref_canvas_hw(ref: SourceInfo) -> tuple[int, int]:
    if not ref.level_shapes:
        raise ValueError("Reference SourceInfo missing level_shapes.")
    h, w = ref.level_shapes[0]
    return int(h), int(w)


def _shift_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _apply_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([points_xy.astype(np.float64), ones], axis=1)
    out = hom @ np.asarray(matrix, dtype=np.float64).T
    return out[:, :2] / out[:, 2:3]


def _image_corners(hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float64,
    )


def _union_bbox_for_sources(
    ref_hw: tuple[int, int],
    mov_hw: tuple[int, int],
    matrix_ref_from_mov: np.ndarray,
) -> tuple[int, int, int, int]:
    ref_h, ref_w = ref_hw
    mov_h, mov_w = mov_hw
    mov_ref = _apply_points(
        _image_corners((mov_h, mov_w)),
        matrix_ref_from_mov,
    )

    x0 = int(np.floor(min(0.0, float(mov_ref[:, 0].min()))))
    y0 = int(np.floor(min(0.0, float(mov_ref[:, 1].min()))))
    x1 = int(np.ceil(max(float(ref_w), float(mov_ref[:, 0].max()))))
    y1 = int(np.ceil(max(float(ref_h), float(mov_ref[:, 1].max()))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid union export bbox {(x0, y0, x1, y1)}.")
    return x0, y0, x1, y1


def _union_bbox_ref_full(transform: "RegistrationTransform") -> tuple[int, int, int, int]:
    return _union_bbox_for_sources(
        _ref_canvas_hw(transform.reference.source),
        transform.moving.source.base_shape_hw,
        transform.matrix_ref_full_from_mov_full,
    )


def _selected_channel_names(
    source: SourceInfo,
    channels: Channels,
) -> tuple[str, ...] | None:
    names = tuple(source.channel_names)
    if not names:
        return None
    if channels == "all":
        return names

    selected: list[str] = []
    lowered = {name.lower(): i for i, name in enumerate(names)}
    requested: Sequence[int | str]
    if isinstance(channels, (int, str)):
        requested = (channels,)
    else:
        requested = tuple(channels)
    for ch in requested:
        if isinstance(ch, int):
            if 0 <= ch < len(names):
                selected.append(names[ch])
            else:
                selected.append(f"channel_{ch}")
        else:
            idx = lowered.get(str(ch).lower())
            selected.append(names[idx] if idx is not None else str(ch))
    return tuple(selected)


def _output_source_for_canvas(
    *,
    output_path: pathlib.Path,
    reference: SourceInfo,
    moving: SourceInfo,
    channels: Channels,
    canvas: Canvas,
    out_hw: tuple[int, int],
    bbox_ref_full_xyxy: tuple[int, int, int, int],
) -> SourceInfo:
    out_h, out_w = out_hw
    x0, y0, _, _ = bbox_ref_full_xyxy

    origin_xy = None
    extent_xy = None
    if reference.pixel_size_xy is not None:
        sx, sy = reference.pixel_size_xy
        rox, roy = reference.origin_xy or (0.0, 0.0)
        origin_xy = (float(rox) + float(x0) * float(sx), float(roy) + float(y0) * float(sy))
        extent_xy = (float(out_w) * float(sx), float(out_h) * float(sy))

    return SourceInfo(
        source_path=str(output_path),
        series_index=0,
        level_count=1,
        level_shapes=((int(out_h), int(out_w)),),
        axes="CYX",
        dtype=moving.dtype,
        channel_names=_selected_channel_names(moving, channels) or (),
        pixel_size_xy=reference.pixel_size_xy,
        unit=reference.unit,
        origin_xy=origin_xy,
        physical_extent_xy=extent_xy,
        format="ome-tiff" if canvas == "union" else reference.format,
    )


def _normalize_tiff_tile_size(tile_size: Optional[int]) -> int:
    value = _DEFAULT_STREAM_TILE_SIZE if tile_size is None else int(tile_size)
    if value <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size!r}.")
    return max(16, int(math.ceil(value / 16.0)) * 16)


def _pyramid_shapes(
    base_hw: tuple[int, int],
    pyramid_levels: int,
) -> tuple[tuple[int, int], ...]:
    h, w = int(base_hw[0]), int(base_hw[1])
    shapes: list[tuple[int, int]] = [(h, w)]
    for _ in range(max(0, int(pyramid_levels))):
        h //= 2
        w //= 2
        if h < 1 or w < 1:
            break
        shapes.append((h, w))
    return tuple(shapes)


def _level_tile_size(base_tile: int, factor: int) -> int:
    return max(16, int(math.ceil(max(16, base_tile // max(1, factor)) / 16.0)) * 16)


def _np_dtype_from_torch(dtype: Optional[torch.dtype]) -> np.dtype:
    if dtype is None:
        return np.dtype("float32")
    try:
        return np.dtype(torch.empty((), dtype=dtype).numpy().dtype)
    except TypeError as exc:  # pragma: no cover - defensive for unsupported torch dtypes
        raise ValueError(f"Unsupported TIFF export dtype: {dtype}.") from exc


class _PlaneReader:
    source: SourceInfo
    shape_hw: tuple[int, int]
    channel_count: int
    channel_names: tuple[str, ...]

    def read(self, channel: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _ArrayPlaneReader(_PlaneReader):
    """Fallback reader for raster files or TIFFs without sliceable storage."""

    def __init__(self, path: pathlib.Path) -> None:
        loaded = load_image(path, channels=None, downsample=1, grayscale=False).to("cpu")
        self._chw = loaded.detail.image[0].detach().cpu().contiguous()
        self.source = loaded.space.source
        self.shape_hw = (int(self._chw.shape[-2]), int(self._chw.shape[-1]))
        self.channel_count = int(self._chw.shape[0])
        self.channel_names = tuple(self.source.channel_names)

    def read(self, channel: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        arr = self._chw[int(channel), int(y0) : int(y1), int(x0) : int(x1)]
        return np.ascontiguousarray(arr.numpy())


class _TiffPlaneReader(_PlaneReader):
    """Slice TIFF planes through tifffile's Zarr adapter."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        with ImageIO(grayscale=False).open(path) as src:
            self.source = src.source
            meta = getattr(src, "tiff_meta", None)
            axes = str(meta.axes if meta is not None else src.source.axes)

        self._store = tifffile.imread(str(path), aszarr=True, series=0, level=0)
        self._zarray = zarr.open(self._store, mode="r")
        self.axes = axes
        self.shape = tuple(int(v) for v in self._zarray.shape)
        if len(self.axes) != len(self.shape):
            raise ValueError(
                f"TIFF axes/shape mismatch for streaming export: "
                f"axes={self.axes!r}, shape={self.shape!r}."
            )
        if "Y" not in self.axes or "X" not in self.axes:
            raise ValueError(f"Streaming TIFF export requires X/Y axes, got {self.axes!r}.")
        self.shape_hw = (
            int(self.shape[self.axes.index("Y")]),
            int(self.shape[self.axes.index("X")]),
        )
        sample_axis = "C" if "C" in self.axes else ("S" if "S" in self.axes else None)
        self._sample_axis = sample_axis
        self.channel_count = int(self.shape[self.axes.index(sample_axis)]) if sample_axis else 1
        self.channel_names = tuple(self.source.channel_names)

    def read(self, channel: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        key: list[int | slice] = []
        for ax, size in zip(self.axes, self.shape):
            if ax == "Y":
                key.append(slice(int(y0), int(y1)))
            elif ax == "X":
                key.append(slice(int(x0), int(x1)))
            elif ax == self._sample_axis:
                key.append(int(channel))
            else:
                if int(size) != 1:
                    raise ValueError(
                        f"Streaming export only supports singleton non-XYC axes; "
                        f"got axis {ax!r} with size {size} in {self.path}."
                    )
                key.append(0)
        arr = np.asarray(self._zarray[tuple(key)])
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2-D plane after channel selection from {self.path}, "
                f"got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr)

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            close()


class _ConstantPlaneReader(_PlaneReader):
    """Virtual single-channel reader used for moving-footprint masks."""

    def __init__(self, shape_hw: tuple[int, int], source: SourceInfo) -> None:
        self.source = source
        self.shape_hw = (int(shape_hw[0]), int(shape_hw[1]))
        self.channel_count = 1
        self.channel_names = ()

    def read(self, channel: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        del channel
        return np.ones((int(y1) - int(y0), int(x1) - int(x0)), dtype=np.float32)


@dataclass
class _StreamChannel:
    reader: _PlaneReader
    channel: int
    matrix_canvas_from_source: np.ndarray
    mode: str = "bilinear"
    fill_value: float = 0.0


def _open_plane_reader(path: pathlib.Path) -> _PlaneReader:
    if path.suffix.lower() in _TIFF_EXTENSIONS:
        try:
            return _TiffPlaneReader(path)
        except Exception:
            # Some TIFF variants cannot be sliced through tifffile's Zarr
            # adapter. Fall back to the existing eager reader so export still
            # succeeds, while tiled output writing remains memory bounded.
            return _ArrayPlaneReader(path)
    return _ArrayPlaneReader(path)


def _resolve_reader_channels(
    reader: _PlaneReader,
    channels: Channels | None,
) -> tuple[int, ...]:
    if channels is None or channels == "all":
        return tuple(range(reader.channel_count))

    requested: Sequence[int | str]
    if isinstance(channels, (int, str)):
        requested = (channels,)
    else:
        requested = tuple(channels)

    names = tuple(reader.channel_names)
    lowered = {name.lower(): i for i, name in enumerate(names)}
    resolved: list[int] = []
    for ch in requested:
        if isinstance(ch, int):
            idx = int(ch)
        else:
            text = str(ch)
            if text.isdigit():
                idx = int(text)
            elif text.lower() in lowered:
                idx = lowered[text.lower()]
            else:
                raise ValueError(
                    f"Channel {ch!r} not found. Available channels: {list(names)}"
                )
        if idx < 0 or idx >= reader.channel_count:
            raise IndexError(
                f"Channel index {idx} out of range for {reader.channel_count} channels."
            )
        resolved.append(idx)
    return tuple(resolved)


def _channel_names_for_indices(
    reader: _PlaneReader,
    indices: Sequence[int],
    *,
    prefix: str | None = None,
) -> tuple[str, ...]:
    names = list(reader.channel_names)
    if len(names) < reader.channel_count:
        names.extend(f"channel_{i}" for i in range(len(names), reader.channel_count))
    out: list[str] = []
    for idx in indices:
        name = names[int(idx)] if int(idx) < len(names) else f"channel_{idx}"
        # Use '_' separator: ':' has split-on-colon behavior in some
        # Bio-Formats / QuPath versions and also collides with the OME
        # Channel:N:M ID convention.
        out.append(f"{prefix}_{name}" if prefix else name)
    return tuple(out)


def _source_bounds_for_tile(
    matrix_canvas_from_source: np.ndarray,
    *,
    source_hw: tuple[int, int],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[int, int, int, int] | None:
    h, w = source_hw
    matrix_source_from_canvas = np.linalg.inv(np.asarray(matrix_canvas_from_source, dtype=np.float64))
    pts = np.array(
        [
            [float(x0), float(y0)],
            [float(max(x0, x1 - 1)), float(y0)],
            [float(x0), float(max(y0, y1 - 1))],
            [float(max(x0, x1 - 1)), float(max(y0, y1 - 1))],
        ],
        dtype=np.float64,
    )
    src = _apply_points(pts, matrix_source_from_canvas)
    sx0 = max(0, int(math.floor(float(src[:, 0].min()))) - _TILE_MARGIN)
    sy0 = max(0, int(math.floor(float(src[:, 1].min()))) - _TILE_MARGIN)
    sx1 = min(int(w), int(math.ceil(float(src[:, 0].max()))) + _TILE_MARGIN + 1)
    sy1 = min(int(h), int(math.ceil(float(src[:, 1].max()))) + _TILE_MARGIN + 1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    return sy0, sy1, sx0, sx1


def _render_base_tile(
    spec: _StreamChannel,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    out_h = int(y1) - int(y0)
    out_w = int(x1) - int(x0)
    if out_h <= 0 or out_w <= 0:
        return np.empty((0, 0), dtype=np.float32)

    bounds = _source_bounds_for_tile(
        spec.matrix_canvas_from_source,
        source_hw=spec.reader.shape_hw,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
    )
    if bounds is None:
        return np.full((out_h, out_w), float(spec.fill_value), dtype=np.float32)

    sy0, sy1, sx0, sx1 = bounds
    crop = spec.reader.read(spec.channel, sy0, sy1, sx0, sx1)
    crop_tensor = to_bchw_float(crop)

    matrix_tile_from_canvas = _shift_matrix(-x0, -y0)
    matrix_source_from_crop = _shift_matrix(sx0, sy0)
    matrix_tile_from_crop = (
        matrix_tile_from_canvas
        @ np.asarray(spec.matrix_canvas_from_source, dtype=np.float64)
        @ matrix_source_from_crop
    )
    warped = warp_to_reference(
        crop_tensor,
        matrix_tile_from_crop,
        out_hw=(out_h, out_w),
        tile_size=None,
        fill_value=spec.fill_value,
        mode=spec.mode,
    )
    return np.ascontiguousarray(warped.image[0, 0].detach().cpu().numpy())


def _render_level_tile(
    spec: _StreamChannel,
    *,
    level: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    factor = 1 << int(level)
    if factor == 1:
        return _render_base_tile(spec, y0=y0, y1=y1, x0=x0, x1=x1)
    high = _render_base_tile(
        spec,
        y0=int(y0) * factor,
        y1=int(y1) * factor,
        x0=int(x0) * factor,
        x1=int(x1) * factor,
    )
    return np.ascontiguousarray(area_downsample(high, "YX", factor))


def _cast_tile(arr: np.ndarray, dtype: np.dtype, *, mask_threshold: bool) -> np.ndarray:
    if mask_threshold:
        arr = (arr > 0.5).astype(dtype, copy=False)
    elif arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def _write_streamed_warped_ome_tiff(
    *,
    output_path: pathlib.Path,
    channels: Sequence[_StreamChannel],
    output_source: SourceInfo,
    out_hw: tuple[int, int],
    tile_size: Optional[int],
    pyramid_levels: int,
    overwrite: bool,
    dtype: np.dtype,
    compression: Optional[str] = None,
    mask_threshold: bool = False,
) -> SourceInfo:
    ensure_parent_dir(output_path)
    assert_overwrite(output_path, overwrite)

    base_tile = _normalize_tiff_tile_size(tile_size)
    level_shapes = _pyramid_shapes(out_hw, pyramid_levels)
    output_source = replace(
        output_source,
        level_count=len(level_shapes),
        level_shapes=tuple(level_shapes),
        dtype=str(np.dtype(dtype)),
        format="ome-tiff",
    )
    channel_count = len(channels)
    if channel_count < 1:
        raise ValueError("At least one channel is required for TIFF export.")

    # Use natural CYX (multi-channel) or YX (single-channel) axes.  Adding a
    # synthetic singleton-T dimension is a common cause of viewers (older
    # Bio-Formats / QuPath / naive TIFF readers) interpreting the channel
    # IFDs as a time series — keep dims faithful to the data.
    ome_axes = "CYX" if channel_count > 1 else "YX"
    metadata = _build_ome_metadata(
        output_source,
        output_source.channel_names or None,
        channel_count,
        axes=ome_axes,
    )
    resolution, resolutionunit = _tiff_resolution_from_source(output_source)
    base_kwargs: dict[str, object] = {
        "photometric": "minisblack",
    }
    if compression is not None:
        base_kwargs["compression"] = compression

    def iter_tiles(level: int, shape_hw: tuple[int, int], level_tile: int) -> Iterator[np.ndarray]:
        h, w = shape_hw
        for spec in channels:
            for yy in range(0, h, level_tile):
                for xx in range(0, w, level_tile):
                    y_end = min(h, yy + level_tile)
                    x_end = min(w, xx + level_tile)
                    tile_arr = _render_level_tile(
                        spec,
                        level=level,
                        y0=yy,
                        y1=y_end,
                        x0=xx,
                        x1=x_end,
                    )
                    yield _cast_tile(tile_arr, dtype, mask_threshold=mask_threshold)

    with tifffile.TiffWriter(str(output_path), ome=True, bigtiff=True) as tw:
        for level, shape_hw in enumerate(level_shapes):
            factor = 1 << level
            level_tile = _level_tile_size(base_tile, factor)
            shape: tuple[int, ...]
            if channel_count > 1:
                shape = (channel_count, int(shape_hw[0]), int(shape_hw[1]))
            else:
                shape = (int(shape_hw[0]), int(shape_hw[1]))
            kwargs = dict(base_kwargs)
            kwargs["tile"] = (level_tile, level_tile)
            if resolution is not None:
                kwargs["resolution"] = (
                    float(resolution[0]) / float(factor),
                    float(resolution[1]) / float(factor),
                )
                kwargs["resolutionunit"] = resolutionunit
            if level == 0:
                kwargs["metadata"] = metadata
                if len(level_shapes) > 1:
                    kwargs["subifds"] = len(level_shapes) - 1
            else:
                kwargs["subfiletype"] = 1
            tw.write(
                iter_tiles(level, shape_hw, level_tile),
                shape=shape,
                dtype=dtype,
                **kwargs,
            )
    return output_source


def _prefixed_channel_names(
    reader: _PlaneReader,
    indices: Sequence[int],
    prefix: str,
) -> tuple[str, ...]:
    return _channel_names_for_indices(reader, indices, prefix=prefix)


def _stacked_output_source(
    *,
    path: pathlib.Path,
    reference: SourceInfo,
    moving: SourceInfo,
    channel_names: tuple[str, ...],
    out_hw: tuple[int, int],
    bbox_ref_full_xyxy: tuple[int, int, int, int],
) -> SourceInfo:
    out_h, out_w = out_hw
    x0, y0, _, _ = bbox_ref_full_xyxy
    origin_xy = None
    extent_xy = None
    if reference.pixel_size_xy is not None:
        sx, sy = reference.pixel_size_xy
        ox, oy = reference.origin_xy or (0.0, 0.0)
        origin_xy = (float(ox) + float(x0) * float(sx), float(oy) + float(y0) * float(sy))
        extent_xy = (float(out_w) * float(sx), float(out_h) * float(sy))

    return SourceInfo(
        source_path=str(path),
        series_index=0,
        level_count=1,
        level_shapes=(out_hw,),
        axes="CYX",
        dtype="float32",
        channel_names=channel_names,
        pixel_size_xy=reference.pixel_size_xy,
        unit=reference.unit,
        origin_xy=origin_xy,
        physical_extent_xy=extent_xy,
        format="ome-tiff",
    )


def _mask_output_source(
    *,
    path: pathlib.Path,
    reference: SourceInfo,
    out_hw: tuple[int, int],
    bbox_ref_full_xyxy: tuple[int, int, int, int],
) -> SourceInfo:
    out_h, out_w = out_hw
    x0, y0, _, _ = bbox_ref_full_xyxy
    origin_xy = None
    extent_xy = None
    if reference.pixel_size_xy is not None:
        sx, sy = reference.pixel_size_xy
        ox, oy = reference.origin_xy or (0.0, 0.0)
        origin_xy = (float(ox) + float(x0) * float(sx), float(oy) + float(y0) * float(sy))
        extent_xy = (float(out_w) * float(sx), float(out_h) * float(sy))
    return SourceInfo(
        source_path=str(path),
        series_index=0,
        level_count=1,
        level_shapes=(out_hw,),
        axes="YX",
        dtype="uint8",
        channel_names=(),
        pixel_size_xy=reference.pixel_size_xy,
        unit=reference.unit,
        origin_xy=origin_xy,
        physical_extent_xy=extent_xy,
        format="ome-tiff",
    )


def _eager_stacked_export(
    *,
    reference_path: pathlib.Path,
    moving_path: pathlib.Path,
    reference_channels: Channels,
    moving_channels: Channels,
    matrix_canvas_from_ref: np.ndarray,
    matrix_canvas_from_mov: np.ndarray,
    source: SourceInfo,
    output_path: pathlib.Path,
    out_hw: tuple[int, int],
    tile_size: Optional[int],
    pyramid_levels: int,
    overwrite: bool,
) -> pathlib.Path:
    ref_arg = None if reference_channels == "all" else reference_channels
    mov_arg = None if moving_channels == "all" else moving_channels
    reference = load_image(reference_path, channels=ref_arg, downsample=1, grayscale=False)
    moving = load_image(moving_path, channels=mov_arg, downsample=1, grayscale=False)
    ref_canvas = warp_to_reference(
        reference.detail,
        matrix_canvas_from_ref,
        out_hw=out_hw,
        tile_size=tile_size,
    )
    mov_canvas = warp_to_reference(
        moving.detail,
        matrix_canvas_from_mov,
        out_hw=out_hw,
        tile_size=tile_size,
    )
    stacked = torch.cat([ref_canvas.image, mov_canvas.image], dim=1)
    return ImageIO().save_tiff(
        stacked,
        output_path,
        config=TiffExportConfig(
            format="ome-tiff",
            pyramid_levels=max(0, int(pyramid_levels)),
            overwrite=overwrite,
            channel_names=source.channel_names,
            metadata_from_source=True,
        ),
        source=source,
    )


def export_stacked_registered_tiff(
    transform: "RegistrationTransform",
    *,
    output_path: PathLike,
    reference_path: Optional[PathLike] = None,
    moving_path: Optional[PathLike] = None,
    matrix_ref_from_mov: Optional[np.ndarray] = None,
    canvas: Canvas = "union",
    reference_channels: Channels = "all",
    moving_channels: Channels = "all",
    tile_size: Optional[int] = None,
    pyramid_levels: int = _DEFAULT_PYRAMID_LEVELS,
    streaming: bool = True,
    overwrite: bool = False,
    compression: Optional[str] = None,
) -> dict[str, Any]:
    """Write reference and registered moving channels into one OME-TIFF.

    The default path streams one output tile at a time directly into a
    tiled BigTIFF and writes SubIFD pyramid levels for responsive viewing.
    Set ``streaming=False`` to use the older eager full-canvas path (requires a lot of RAM for large TIFF files).
    """
    out_path = pathify(output_path)
    ref_path = pathify(reference_path) if reference_path is not None else pathlib.Path(transform.reference.source.source_path)
    mov_path = pathify(moving_path) if moving_path is not None else pathlib.Path(transform.moving.source.source_path)
    matrix = (
        np.asarray(matrix_ref_from_mov, dtype=np.float64)
        if matrix_ref_from_mov is not None
        else np.asarray(transform.matrix_ref_full_from_mov_full, dtype=np.float64)
    )

    ref_reader = _open_plane_reader(ref_path)
    mov_reader = _open_plane_reader(mov_path)
    try:
        if canvas == "reference":
            ref_h, ref_w = ref_reader.shape_hw
            bbox = (0, 0, ref_w, ref_h)
        elif canvas == "union":
            bbox = _union_bbox_for_sources(ref_reader.shape_hw, mov_reader.shape_hw, matrix)
        else:
            raise ValueError(f"Unknown export canvas {canvas!r}.")

        x0, y0, x1, y1 = bbox
        out_hw = (y1 - y0, x1 - x0)
        matrix_canvas_from_ref = _shift_matrix(-x0, -y0)
        matrix_canvas_from_mov = matrix_canvas_from_ref @ matrix

        ref_indices = _resolve_reader_channels(ref_reader, reference_channels)
        mov_indices = _resolve_reader_channels(mov_reader, moving_channels)
        channel_names = (
            _prefixed_channel_names(ref_reader, ref_indices, "reference")
            + _prefixed_channel_names(mov_reader, mov_indices, "moving")
        )
        source = _stacked_output_source(
            path=out_path,
            reference=ref_reader.source,
            moving=mov_reader.source,
            channel_names=channel_names,
            out_hw=out_hw,
            bbox_ref_full_xyxy=bbox,
        )

        if streaming:
            specs = [
                _StreamChannel(ref_reader, idx, matrix_canvas_from_ref)
                for idx in ref_indices
            ] + [
                _StreamChannel(mov_reader, idx, matrix_canvas_from_mov)
                for idx in mov_indices
            ]
            source = _write_streamed_warped_ome_tiff(
                output_path=out_path,
                channels=specs,
                output_source=source,
                out_hw=out_hw,
                tile_size=tile_size,
                pyramid_levels=pyramid_levels,
                overwrite=overwrite,
                dtype=np.dtype("float32"),
                compression=compression,
            )
        else:
            _eager_stacked_export(
                reference_path=ref_path,
                moving_path=mov_path,
                reference_channels=reference_channels,
                moving_channels=moving_channels,
                matrix_canvas_from_ref=matrix_canvas_from_ref,
                matrix_canvas_from_mov=matrix_canvas_from_mov,
                source=source,
                output_path=out_path,
                out_hw=out_hw,
                tile_size=tile_size,
                pyramid_levels=pyramid_levels,
                overwrite=overwrite,
            )
            source = replace(
                source,
                level_count=len(_pyramid_shapes(out_hw, pyramid_levels)),
                level_shapes=_pyramid_shapes(out_hw, pyramid_levels),
            )

        return {
            "path": out_path.name,
            "canvas": canvas,
            "bbox_ref_full_xyxy": list(bbox),
            "matrix_canvas_from_ref_full": matrix_canvas_from_ref.tolist(),
            "matrix_canvas_from_mov_full": matrix_canvas_from_mov.tolist(),
            "source": source.to_dict(),
            "streaming": bool(streaming),
            "tile_size": _normalize_tiff_tile_size(tile_size) if streaming else tile_size,
            "pyramid_levels": max(0, source.level_count - 1),
        }
    finally:
        ref_reader.close()
        mov_reader.close()


def export_moving_mask_tiff(
    transform: "RegistrationTransform",
    *,
    output_path: PathLike,
    moving_path: Optional[PathLike] = None,
    reference_path: Optional[PathLike] = None,
    matrix_ref_from_mov: Optional[np.ndarray] = None,
    bbox_ref_full_xyxy: Optional[tuple[int, int, int, int]] = None,
    tile_size: Optional[int] = None,
    pyramid_levels: int = _DEFAULT_PYRAMID_LEVELS,
    streaming: bool = True,
    overwrite: bool = False,
    compression: Optional[str] = None,
) -> dict[str, Any]:
    """Write a valid-moving-footprint mask on the union canvas."""
    out_path = pathify(output_path)
    ref_path = pathify(reference_path) if reference_path is not None else pathlib.Path(transform.reference.source.source_path)
    mov_path = pathify(moving_path) if moving_path is not None else pathlib.Path(transform.moving.source.source_path)
    matrix = (
        np.asarray(matrix_ref_from_mov, dtype=np.float64)
        if matrix_ref_from_mov is not None
        else np.asarray(transform.matrix_ref_full_from_mov_full, dtype=np.float64)
    )

    mov_reader = _open_plane_reader(mov_path)
    ref_reader: _PlaneReader | None = None
    try:
        reference_source = transform.reference.source
        reference_hw = _ref_canvas_hw(reference_source)
        if ref_path.exists():
            ref_reader = _open_plane_reader(ref_path)
            reference_source = ref_reader.source
            reference_hw = ref_reader.shape_hw
        bbox = (
            tuple(int(v) for v in bbox_ref_full_xyxy)
            if bbox_ref_full_xyxy is not None
            else _union_bbox_for_sources(reference_hw, mov_reader.shape_hw, matrix)
        )
        x0, y0, x1, y1 = bbox
        out_hw = (y1 - y0, x1 - x0)
        matrix_canvas_from_mov = _shift_matrix(-x0, -y0) @ matrix
        source = _mask_output_source(
            path=out_path,
            reference=reference_source,
            out_hw=out_hw,
            bbox_ref_full_xyxy=bbox,
        )

        if streaming:
            mask_reader = _ConstantPlaneReader(mov_reader.shape_hw, mov_reader.source)
            source = _write_streamed_warped_ome_tiff(
                output_path=out_path,
                channels=[
                    _StreamChannel(
                        mask_reader,
                        0,
                        matrix_canvas_from_mov,
                        mode="nearest",
                        fill_value=0.0,
                    )
                ],
                output_source=source,
                out_hw=out_hw,
                tile_size=tile_size,
                pyramid_levels=pyramid_levels,
                overwrite=overwrite,
                dtype=np.dtype("uint8"),
                compression=compression,
                mask_threshold=True,
            )
        else:
            mask = torch.ones(
                (1, 1, mov_reader.shape_hw[0], mov_reader.shape_hw[1]),
                dtype=torch.float32,
            )
            mask_canvas = warp_to_reference(
                mask,
                matrix_canvas_from_mov,
                out_hw=out_hw,
                tile_size=tile_size,
                mode="nearest",
            )
            ImageIO().save_tiff(
                (mask_canvas.image > 0.5).to(torch.uint8),
                out_path,
                config=TiffExportConfig(
                    format="ome-tiff",
                    pyramid_levels=max(0, int(pyramid_levels)),
                    overwrite=overwrite,
                    metadata_from_source=True,
                ),
                source=source,
            )
            source = replace(
                source,
                level_count=len(_pyramid_shapes(out_hw, pyramid_levels)),
                level_shapes=_pyramid_shapes(out_hw, pyramid_levels),
            )

        return {
            "path": out_path.name,
            "source": source.to_dict(),
            "streaming": bool(streaming),
            "tile_size": _normalize_tiff_tile_size(tile_size) if streaming else tile_size,
            "pyramid_levels": max(0, source.level_count - 1),
        }
    finally:
        if ref_reader is not None:
            ref_reader.close()
        mov_reader.close()


def export_registered(
    source: Union["RegistrationTransform", dict[str, Any], PathLike],
    *,
    output_path: PathLike,
    moving_path: Optional[PathLike] = None,
    channels: Channels = "all",
    canvas: Canvas = "reference",
    mask_output_path: Optional[PathLike] = None,
    return_metadata: bool = False,
    dtype: Optional[torch.dtype] = None,
    tile_size: Optional[int] = None,
    pyramid_levels: int = _DEFAULT_PYRAMID_LEVELS,
    streaming: bool = True,
    compression: Optional[str] = None,
    fill_value: float = 0.0,
    overwrite: bool = False,
) -> pathlib.Path | RegisteredExport:
    """Warp the moving image to the reference canvas and save.

    Args:
        source: A :class:`RegistrationTransform` or a manifest dict / path.
        output_path: Destination file path.
        moving_path: Override for the moving source file. Defaults to
            the path recorded in ``transform.moving.source``.
        channels: ``"all"`` (default) or an explicit list of channel
            indices / names to load from the moving source.
        canvas: ``"reference"`` writes the legacy reference-sized output.
            ``"union"`` writes a padded canvas containing both the reference
            and transformed moving image extents.
        mask_output_path: Optional path for a valid-moving-footprint mask.
            Supported with ``canvas="union"``.
        return_metadata: Return a :class:`RegisteredExport` instead of only
            the output path.
        dtype: Optional output dtype override.
        tile_size: Tile size for the grid_sample loop.
        pyramid_levels: Number of 2x SubIFD pyramid levels for TIFF output.
        streaming: For TIFF output, stream one transformed tile at a time
            directly to the output file. Enabled by default.
        compression: Optional tifffile compression codec. Defaults to
            uncompressed/lossless storage.
        fill_value: Fill value outside the moving footprint.
        overwrite: Allow overwriting ``output_path``.
    """
    transform, _ = _load_transform(source)

    mov_src = transform.moving.source
    ref_src = transform.reference.source
    mov_file = pathify(moving_path) if moving_path is not None else pathlib.Path(mov_src.source_path)

    if canvas == "reference":
        ref_h, ref_w = _ref_canvas_hw(ref_src)
        bbox = (0, 0, ref_w, ref_h)
        matrix_canvas_from_mov = transform.matrix_ref_full_from_mov_full
        out_hw = (ref_h, ref_w)
    elif canvas == "union":
        bbox = _union_bbox_ref_full(transform)
        x0, y0, x1, y1 = bbox
        out_hw = (y1 - y0, x1 - x0)
        matrix_canvas_from_mov = _shift_matrix(-x0, -y0) @ transform.matrix_ref_full_from_mov_full
    else:
        raise ValueError(f"Unknown export canvas {canvas!r}.")

    out_path = pathify(output_path)
    if streaming and out_path.suffix.lower() in _TIFF_EXTENSIONS:
        reader = _open_plane_reader(mov_file)
        try:
            selected = _resolve_reader_channels(reader, channels)
            channel_names = _channel_names_for_indices(reader, selected)
            out_source = _output_source_for_canvas(
                output_path=out_path,
                reference=ref_src,
                moving=reader.source,
                channels=channels,
                canvas=canvas,
                out_hw=out_hw,
                bbox_ref_full_xyxy=bbox,
            )
            out_source = replace(
                out_source,
                channel_names=channel_names,
                dtype=str(_np_dtype_from_torch(dtype)),
                format="ome-tiff",
            )
            saved_source = _write_streamed_warped_ome_tiff(
                output_path=out_path,
                channels=[
                    _StreamChannel(
                        reader,
                        idx,
                        matrix_canvas_from_mov,
                        fill_value=fill_value,
                    )
                    for idx in selected
                ],
                output_source=out_source,
                out_hw=out_hw,
                tile_size=tile_size,
                pyramid_levels=pyramid_levels,
                overwrite=overwrite,
                dtype=_np_dtype_from_torch(dtype),
                compression=compression,
            )

            saved_mask_path: pathlib.Path | None = None
            if mask_output_path is not None:
                export_moving_mask_tiff(
                    transform,
                    output_path=mask_output_path,
                    moving_path=mov_file,
                    matrix_ref_from_mov=matrix_canvas_from_mov
                    if canvas == "reference"
                    else transform.matrix_ref_full_from_mov_full,
                    bbox_ref_full_xyxy=bbox,
                    tile_size=tile_size,
                    pyramid_levels=pyramid_levels,
                    streaming=True,
                    overwrite=overwrite,
                    compression=compression,
                )
                saved_mask_path = pathify(mask_output_path)

            if return_metadata:
                return RegisteredExport(
                    path=out_path,
                    canvas=canvas,
                    output_source=saved_source,
                    bbox_ref_full_xyxy=bbox,
                    matrix_canvas_from_mov_full=matrix_canvas_from_mov,
                    mask_path=saved_mask_path,
                )
            return out_path
        finally:
            reader.close()

    channels_arg = None if channels == "all" else channels
    moving = load_image(mov_file, downsample=1, channels=channels_arg, grayscale=False)

    warped = warp_to_reference(
        moving.detail,
        matrix_canvas_from_mov,
        out_hw=out_hw,
        tile_size=tile_size,
        fill_value=fill_value,
    )

    out = warped.image
    if dtype is not None:
        out = out.to(dtype=dtype)

    out_source = _output_source_for_canvas(
        output_path=out_path,
        reference=ref_src,
        moving=mov_src,
        channels=channels,
        canvas=canvas,
        out_hw=out_hw,
        bbox_ref_full_xyxy=bbox,
    )
    saved_path = ImageIO().save(
        out,
        out_path,
        overwrite=overwrite,
        source=out_source,
        channel_names=out_source.channel_names or None,
    )

    saved_mask_path: pathlib.Path | None = None
    if mask_output_path is not None:
        mask = torch.ones(
            (1, 1, moving.detail.H, moving.detail.W),
            dtype=torch.float32,
            device=moving.detail.image.device,
        )
        mask_warped = warp_to_reference(
            mask,
            matrix_canvas_from_mov,
            out_hw=out_hw,
            tile_size=tile_size,
            fill_value=0.0,
            mode="nearest",
        )
        mask_source = SourceInfo(
            source_path=str(pathify(mask_output_path)),
            series_index=0,
            level_count=1,
            level_shapes=out_source.level_shapes,
            axes="YX",
            dtype="uint8",
            channel_names=(),
            pixel_size_xy=out_source.pixel_size_xy,
            unit=out_source.unit,
            origin_xy=out_source.origin_xy,
            physical_extent_xy=out_source.physical_extent_xy,
            format="ome-tiff",
        )
        saved_mask_path = ImageIO().save(
            (mask_warped.image > 0.5).to(torch.uint8),
            mask_output_path,
            overwrite=overwrite,
            source=mask_source,
        )

    if return_metadata:
        return RegisteredExport(
            path=saved_path,
            canvas=canvas,
            output_source=out_source,
            bbox_ref_full_xyxy=bbox,
            matrix_canvas_from_mov_full=matrix_canvas_from_mov,
            mask_path=saved_mask_path,
        )
    return saved_path
