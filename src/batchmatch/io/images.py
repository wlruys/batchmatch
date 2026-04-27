"""Image I/O: format-agnostic loading and saving.

Design:

- :class:`ImageSource` is the backend-agnostic file handle. Every format
  (TIFF, PNG/JPEG/BMP, ...) implements it. Use it as a context manager.
- :class:`ImagePolicy` captures how raw bytes should become a tensor
  (grayscale, dtype, device, channel reduction). It is a frozen dataclass.
- :class:`SaveOptions` captures format-specific save knobs.
- :class:`ImageIO` is a thin stateless facade with ``open`` / ``load`` /
  ``save`` / ``list``.

Every ``read`` returns a :class:`SpatialImage` whose :class:`ImageSpace`
records the exact ``(pyramid_level, region, downsample, channel_selection)``
used at load time.

Adding a new format = implement a new ``ImageSource`` subclass and register
it in ``_SOURCE_BACKENDS``.
"""

from __future__ import annotations

import math
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image
from torchvision.io import read_image

from batchmatch.base.detail import build_image_td
from batchmatch.helpers.image import image_to_float, image_to_uint8
from batchmatch.helpers.tensor import to_bchw
from batchmatch.io._tiff_meta import TiffMeta, _parse_meta
from batchmatch.io._tiff_read import (
    find_best_pyramid_level,
    read_all_channels,
    read_pyramid_level,
    read_single_channel,
    resolve_channel_index,
)
from batchmatch.io.convert import to_bchw_float
from batchmatch.io.downsample import area_downsample, crop_region, validate_region
from batchmatch.io.space import (
    ImageSpace,
    RegionYXHW,
    SourceFormat,
    SourceInfo,
    SpatialImage,
)
from batchmatch.io.utils import (
    PathLike,
    assert_overwrite,
    coerce_extensions,
    ensure_parent_dir,
    pathify,
)
from batchmatch.view.render import (
    apply_colormap,
    downsample_for_display,
    prepare_for_display,
    select_channels,
    to_rgb,
    to_uint8 as _render_to_uint8,
)

Tensor = torch.Tensor
Channels = Optional[Union[int, str, Sequence[Union[int, str]]]]
RegionLike = Optional[Union[RegionYXHW, tuple[int, int, int, int], Sequence[int]]]
Level = Union[int, Literal["auto"]]

__all__ = [
    "ImageIO",
    "ImagePolicy",
    "ImageSource",
    "PreviewConfig",
    "RasterSource",
    "SaveOptions",
    "TiffExportConfig",
    "TiffSource",
    "load_image",
    "open_image",
    "save_image",
    "save_preview",
    "save_tiff",
]

_TIFF_EXTENSIONS = frozenset({".tif", ".tiff"})
_JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})
_PNG_BMP_EXTENSIONS = frozenset({".png", ".bmp"})
_RASTER_EXTENSIONS = _JPEG_EXTENSIONS | _PNG_BMP_EXTENSIONS
_SUPPORTED_EXTENSIONS = _TIFF_EXTENSIONS | _RASTER_EXTENSIONS


# ---------------------------------------------------------------------------
# Policy / options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImagePolicy:
    """How raw image data should be presented as a tensor."""

    grayscale: bool = True
    dtype: Optional[torch.dtype] = torch.float32
    device: Optional[torch.device] = None
    channel_reduction: Literal["luma", "mean", "first"] = "luma"

    def apply(self, tensor: Tensor) -> Tensor:
        if self.grayscale and tensor.shape[1] > 1:
            if self.channel_reduction == "first":
                tensor = tensor[:, :1]
            elif self.channel_reduction == "luma":
                if tensor.shape[1] >= 3:
                    weights = torch.tensor(
                        [0.2989, 0.5870, 0.1140],
                        dtype=torch.float32,
                        device=tensor.device,
                    ).view(1, 3, 1, 1)
                    tensor = (
                        tensor[:, :3].to(dtype=torch.float32) * weights
                    ).sum(dim=1, keepdim=True)
                    if not tensor.dtype.is_floating_point:
                        tensor = tensor.round()
                else:
                    tensor = tensor.mean(dim=1, keepdim=True)
            else:
                tensor = tensor.mean(dim=1, keepdim=True)

        if self.dtype is not None and tensor.dtype != self.dtype:
            if self.dtype.is_floating_point:
                tensor = image_to_float(tensor, dtype=self.dtype)
            else:
                if tensor.dtype.is_floating_point:
                    tensor = image_to_uint8(tensor)
                if tensor.dtype != self.dtype:
                    tensor = tensor.to(dtype=self.dtype)

        if self.device is not None:
            tensor = tensor.to(device=self.device)

        return tensor


@dataclass(frozen=True)
class SaveOptions:
    """Format-agnostic save knobs. Format-specific fields are ignored by
    formats that don't use them."""

    overwrite: bool = False
    quality: int = 95  # JPEG
    convert_uint8: Optional[bool] = None  # None = auto: raster yes, TIFF no
    tiff_compression: Optional[str] = None
    tiff_photometric: Optional[str] = None


@dataclass(frozen=True)
class TiffExportConfig:
    """Configuration for TIFF export conventions.

    Controls the output format, channel interpretation, metadata
    embedding, and dtype handling when writing TIFF files.

    Parameters
    ----------
    format : ``"auto"`` | ``"ome-tiff"`` | ``"imagej-tiff"`` | ``"tiff"``
        Output TIFF variant.  ``"auto"`` (default) preserves the source
        format when a :class:`SourceInfo` is available, otherwise falls
        back to ``"ome-tiff"``.
    photometric : ``"auto"`` | ``"minisblack"`` | ``"rgb"``
        How channels are interpreted.  ``"auto"`` selects ``"minisblack"``
        for single-channel or ImageJ multi-channel images and ``"rgb"``
        for 3-channel non-ImageJ images.
    compression : str or None
        tifffile compression codec (e.g. ``"zlib"``, ``"lzw"``,
        ``"zstd"``).  ``None`` means uncompressed.
    pyramid_levels : int
        Number of sub-resolution pyramid levels to generate (each is a
        2× downsample of the previous).  ``0`` (default) writes a single
        resolution.  Pyramids are only supported with ``"ome-tiff"``
        format; the setting is silently ignored for other formats.
    convert_uint8 : bool
        If ``True``, floating-point data is quantised to ``uint8`` before
        writing.
    overwrite : bool
        Allow overwriting an existing file.
    channel_names : tuple of str or None
        Explicit channel names embedded in the file metadata.  ``None``
        preserves names from the source or omits them.
    metadata_from_source : bool
        When ``True`` (default), pixel-size, unit, and calibration are
        copied from the :class:`SourceInfo` passed at save time.
    """

    format: Literal["auto", "ome-tiff", "imagej-tiff", "tiff"] = "auto"
    photometric: Literal["auto", "minisblack", "rgb"] = "auto"
    compression: Optional[str] = None
    pyramid_levels: int = 0
    convert_uint8: bool = False
    overwrite: bool = False
    channel_names: Optional[tuple[str, ...]] = None
    metadata_from_source: bool = True

    def _resolve_format(self, source: Optional[SourceInfo]) -> str:
        """Return the concrete TIFF format string."""
        if self.format != "auto":
            return self.format
        if source is not None and source.format in ("ome-tiff", "imagej-tiff", "tiff"):
            return source.format
        return "ome-tiff"

    def _resolve_photometric(self, fmt: str, ndim: int, channels: int) -> str:
        """Return the concrete photometric string."""
        if self.photometric != "auto":
            return self.photometric
        if ndim == 2:
            return "minisblack"
        # ImageJ hyperstack never uses RGB photometric
        if fmt == "imagej-tiff":
            return "minisblack"
        if channels == 3:
            return "rgb"
        return "minisblack"

    def _resolve_channel_names(
        self, source: Optional[SourceInfo]
    ) -> Optional[tuple[str, ...]]:
        """Return channel names: explicit config → source → None."""
        if self.channel_names is not None:
            return self.channel_names
        if self.metadata_from_source and source is not None and source.channel_names:
            return source.channel_names
        return None


@dataclass(frozen=True)
class PreviewConfig:
    """Controls how image data is presented when saving raster previews.

    Applies normalization, gamma correction, channel selection, and
    optional colormap before converting to uint8 for PNG/JPEG output.

    Parameters
    ----------
    normalize : str
        Normalization mode applied before gamma.  One of ``"minmax"``
        (default), ``"none"``, ``"percentile"``, or ``"abs"``.
    percentile_low, percentile_high : float
        Percentile bounds when ``normalize="percentile"``.
    vmin, vmax : float or None
        Explicit intensity bounds.  When both are set they override
        the *normalize* mode.
    gamma : float
        Display gamma (applied as ``x ** (1/gamma)``).  Values > 1
        brighten dark regions; < 1 increases contrast.
    colormap : str or None
        Matplotlib colormap name for single-channel images.  ``None``
        (default) uses identity mapping / grayscale expansion.
    channel : int, tuple of int, or None
        Channel selection before rendering.  ``None`` keeps auto
        behaviour (≤3 channels pass through, >3 takes first).
    quality : int
        JPEG quality (1–100).  Ignored for PNG.
    overwrite : bool
        Allow overwriting an existing file.
    max_size : int or None
        Cap ``max(H, W)`` to this value using area interpolation.
        ``None`` keeps the original resolution.
    """

    normalize: Literal["minmax", "none", "percentile", "abs"] = "minmax"
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    gamma: float = 1.0
    colormap: Optional[str] = None
    channel: Optional[Union[int, tuple[int, ...]]] = None
    quality: int = 95
    overwrite: bool = False
    max_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_region(region: RegionLike) -> Optional[RegionYXHW]:
    if region is None:
        return None
    if isinstance(region, RegionYXHW):
        return region
    return RegionYXHW.from_list(region)


def _axes_hw(axes: str, shape: Sequence[int]) -> tuple[int, int]:
    y_idx = axes.find("Y")
    x_idx = axes.find("X")
    if y_idx < 0 or x_idx < 0:
        raise ValueError(f"Axes {axes!r} missing X/Y.")
    return int(shape[y_idx]), int(shape[x_idx])


def _tiff_level_shapes(tif: tifffile.TiffFile, axes: str) -> tuple[tuple[int, int], ...]:
    if not tif.series:
        return ()
    series0 = tif.series[0]
    levels = list(series0.levels) if hasattr(series0, "levels") else list(tif.series)
    shapes: list[tuple[int, int]] = []
    for lvl in levels:
        shapes.append(_axes_hw(axes, lvl.shape))
    return tuple(shapes)


def _tiff_format(tif: tifffile.TiffFile) -> SourceFormat:
    if tif.is_ome:
        return "ome-tiff"
    if tif.is_imagej:
        return "imagej-tiff"
    return "tiff"


def _source_info_from_tiff(
    tif: tifffile.TiffFile,
    meta: TiffMeta,
    path: pathlib.Path,
) -> SourceInfo:
    level_shapes = _tiff_level_shapes(tif, meta.axes)
    base_h, base_w = level_shapes[0] if level_shapes else _axes_hw(meta.axes, meta.shape)
    origin_xy = meta.origin_xy or ((0.0, 0.0) if meta.pixel_size_xy is not None else None)
    phys_extent = None
    if meta.pixel_size_xy is not None:
        phys_extent = (base_w * meta.pixel_size_xy[0], base_h * meta.pixel_size_xy[1])
    return SourceInfo(
        source_path=str(path),
        series_index=0,
        level_count=int(meta.num_pyramid_levels),
        level_shapes=level_shapes or ((base_h, base_w),),
        axes=meta.axes,
        dtype=meta.original_dtype or "unknown",
        channel_names=tuple(meta.channel_names),
        pixel_size_xy=meta.pixel_size_xy,
        unit=meta.unit,
        origin_xy=origin_xy,
        physical_extent_xy=phys_extent,
        format=_tiff_format(tif),
    )


# ---------------------------------------------------------------------------
# Source backends
# ---------------------------------------------------------------------------


class ImageSource(ABC):
    """Backend-agnostic image file handle.

    Lifecycle: ``open`` → zero or more ``read`` calls → ``close``. Use as a
    context manager. ``source`` is populated on open and stable for the
    lifetime of the handle.

    Per-read parameters live on ``read`` / ``read_array`` (region,
    downsample, channels, level). Presentation policy (grayscale, dtype,
    device) is passed as an :class:`ImagePolicy`.
    """

    path: pathlib.Path
    source: SourceInfo
    default_policy: Optional[ImagePolicy] = None

    @abstractmethod
    def read_array(
        self,
        *,
        region: Optional[RegionYXHW] = None,
        downsample: int = 1,
        channels: Channels = None,
        level: Level = "auto",
    ) -> tuple[np.ndarray, str, Optional[tuple[int, ...]], int]:
        """Low-level read.

        Returns ``(array, axes, resolved_channel_indices, pyramid_level)``.
        The returned ``pyramid_level`` is the level that was physically
        read from; any remaining integer downsample is already applied to
        the array.
        """

    def read(
        self,
        *,
        region: RegionLike = None,
        downsample: int = 1,
        channels: Channels = None,
        level: Level = "auto",
        policy: Optional[ImagePolicy] = None,
    ) -> SpatialImage:
        """High-level read: return a :class:`SpatialImage` with geometry."""
        pol = policy if policy is not None else (self.default_policy or ImagePolicy())
        region_yxhw = _coerce_region(region)
        if region_yxhw is None:
            base_h, base_w = self.source.base_shape_hw
            region_yxhw = RegionYXHW(y=0, x=0, h=base_h, w=base_w)

        arr, _axes, resolved_channels, pyramid_level = self.read_array(
            region=region_yxhw,
            downsample=int(downsample),
            channels=channels,
            level=level,
        )
        tensor = to_bchw_float(arr)
        tensor = pol.apply(tensor)

        detail = build_image_td(tensor)
        space = ImageSpace.from_load(
            source=self.source,
            pyramid_level=int(pyramid_level),
            region=region_yxhw,
            downsample=int(downsample),
            shape_hw=(int(tensor.shape[-2]), int(tensor.shape[-1])),
            channel_selection=resolved_channels,
        )
        return SpatialImage(detail=detail, space=space)

    def close(self) -> None:
        pass

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TiffSource(ImageSource):
    """TIFF / OME-TIFF source with pyramid level auto-selection.

    The constructor opens the file once to parse metadata, then closes it.
    The file handle is re-opened lazily on the first ``read``/``read_array``
    call via ``_ensure_open()``.  This keeps construction cheap for
    listing/metadata-only workflows.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._tif: tifffile.TiffFile | None = None
        with tifffile.TiffFile(path) as tif:
            self._tiff_meta = _parse_meta(tif)
            self.source = _source_info_from_tiff(tif, self._tiff_meta, path)

    @property
    def tiff_meta(self) -> TiffMeta:
        return self._tiff_meta

    def _ensure_open(self) -> tifffile.TiffFile:
        if self._tif is None:
            self._tif = tifffile.TiffFile(self.path)
        return self._tif

    def _resolve_channels(self, channels: Channels) -> Optional[list[int]]:
        if channels is None:
            return None
        names = list(self._tiff_meta.channel_names)
        axes = self._tiff_meta.axes
        if isinstance(channels, (int, str)):
            return [resolve_channel_index(channels, names, axes)]
        if isinstance(channels, Sequence) and not isinstance(channels, (bytes,)):
            return [resolve_channel_index(c, names, axes) for c in channels]
        raise TypeError(f"channels must be int, str, or sequence thereof, got {type(channels)!r}.")

    def _pick_level(self, level: Level, downsample: int, tif: tifffile.TiffFile) -> tuple[int, int]:
        if level == "auto":
            if downsample > 1:
                return find_best_pyramid_level(tif, downsample)
            return 0, downsample
        return int(level), downsample

    def read_array(
        self,
        *,
        region: Optional[RegionYXHW] = None,
        downsample: int = 1,
        channels: Channels = None,
        level: Level = "auto",
    ) -> tuple[np.ndarray, str, Optional[tuple[int, ...]], int]:
        tif = self._ensure_open()
        meta = self._tiff_meta
        ch_indices = self._resolve_channels(channels)
        pyramid_level, remaining_ds = self._pick_level(level, downsample, tif)

        if ch_indices is not None and len(ch_indices) == 1:
            idx = ch_indices[0]
            if pyramid_level > 0:
                arr = read_pyramid_level(tif, pyramid_level)
                c_ax = meta.axes.find("C")
                arr = np.take(arr, idx, axis=c_ax)
            else:
                arr = read_single_channel(tif, meta, idx)
            axes = meta.axes.replace("C", "") if "C" in meta.axes else meta.axes
        else:
            if pyramid_level > 0:
                arr = read_pyramid_level(tif, pyramid_level)
            else:
                arr = read_all_channels(tif)
            axes = meta.axes
            if ch_indices is not None:
                c_ax = meta.axes.find("C")
                arr = np.take(arr, ch_indices, axis=c_ax)

        arr = np.ascontiguousarray(arr)

        if region is not None:
            # Region is in source full-res coords; translate to current pyramid-level coords.
            # Use floor for origin and ceil for extent to guarantee the returned
            # region fully covers the requested source-space area.
            base_h, base_w = self.source.base_shape_hw
            level_h, level_w = self.source.level_shape_hw(pyramid_level)
            sx = level_w / base_w
            sy = level_h / base_h
            y0 = math.floor(region.y * sy)
            x0 = math.floor(region.x * sx)
            y1 = math.ceil((region.y + region.h) * sy)
            x1 = math.ceil((region.x + region.w) * sx)
            region_level = (
                y0,
                x0,
                max(1, y1 - y0),
                max(1, x1 - x0),
            )
            region_level = validate_region(
                region_level,
                arr.shape[axes.find("Y")],
                arr.shape[axes.find("X")],
            )
            arr = crop_region(arr, axes, region_level)

        if remaining_ds > 1:
            arr = area_downsample(arr, axes, remaining_ds)

        arr = np.ascontiguousarray(arr)
        resolved = tuple(ch_indices) if ch_indices is not None else None
        return arr, axes, resolved, int(pyramid_level)

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
            self._tif = None


class RasterSource(ImageSource):
    """PNG / JPEG / BMP via torchvision.

    Unlike :class:`TiffSource`, the full image is decoded eagerly in the
    constructor.  This is simple and fast for typical raster sizes but
    means the entire image is held in memory even if only a small region
    or channel subset is needed.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        raw = read_image(str(path))  # CHW
        self._raw_bchw = to_bchw(raw)
        h = int(self._raw_bchw.shape[2])
        w = int(self._raw_bchw.shape[3])
        self.source = SourceInfo(
            source_path=str(path),
            series_index=0,
            level_count=1,
            level_shapes=((h, w),),
            axes="CYX",
            dtype=str(raw.dtype),
            channel_names=(),
            pixel_size_xy=None,
            unit=None,
            origin_xy=None,
            physical_extent_xy=None,
            format="raster",
        )

    def read_array(
        self,
        *,
        region: Optional[RegionYXHW] = None,
        downsample: int = 1,
        channels: Channels = None,
        level: Level = "auto",
    ) -> tuple[np.ndarray, str, Optional[tuple[int, ...]], int]:
        if level not in (0, "auto"):
            raise ValueError(f"Raster formats have no pyramid levels (got level={level!r}).")

        img = self._raw_bchw
        selection: Optional[tuple[int, ...]] = None

        if channels is not None:
            if isinstance(channels, int):
                img = img[:, channels : channels + 1]
                selection = (int(channels),)
            elif isinstance(channels, str):
                raise ValueError(f"Raster formats do not support named channels (got {channels!r}).")
            elif isinstance(channels, Sequence) and not isinstance(channels, (bytes,)):
                idx = torch.tensor([int(c) for c in channels], dtype=torch.long)
                img = img.index_select(1, idx)
                selection = tuple(int(c) for c in channels)
            else:
                raise TypeError(f"channels must be int or sequence for raster, got {type(channels)!r}.")

        if region is not None:
            img = img[..., region.y : region.y2, region.x : region.x2]

        if downsample > 1:
            work = img if img.dtype.is_floating_point else image_to_float(img, dtype=torch.float32)
            img = F.avg_pool2d(work, kernel_size=downsample, stride=downsample)

        arr = img[0].detach().cpu().numpy()
        return np.ascontiguousarray(arr), "CYX", selection, 0


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


_SOURCE_BACKENDS: dict[frozenset[str], type[ImageSource]] = {
    _TIFF_EXTENSIONS: TiffSource,
    _RASTER_EXTENSIONS: RasterSource,
}


def _source_for(path: pathlib.Path) -> ImageSource:
    suffix = path.suffix.lower()
    for suffixes, cls in _SOURCE_BACKENDS.items():
        if suffix in suffixes:
            return cls(path)
    raise ValueError(
        f"Unsupported image extension: {suffix!r} "
        f"(supported: {sorted(_SUPPORTED_EXTENSIONS)})."
    )


@dataclass(frozen=True, init=False)
class ImageIO:
    """Stateless facade holding a default policy and default save options."""

    policy: ImagePolicy
    save_options: SaveOptions

    def __init__(
        self,
        policy: Optional[ImagePolicy] = None,
        save_options: Optional[SaveOptions] = None,
        *,
        grayscale: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        channel_reduction: Optional[Literal["luma", "mean", "first"]] = None,
        overwrite: Optional[bool] = None,
        jpeg_quality: Optional[int] = None,
        convert_uint8: Optional[bool] = None,
    ) -> None:
        pol = policy if policy is not None else ImagePolicy()
        pol_patch: dict = {}
        if grayscale is not None:
            pol_patch["grayscale"] = bool(grayscale)
        if dtype is not None:
            pol_patch["dtype"] = dtype
        if device is not None:
            pol_patch["device"] = device
        if channel_reduction is not None:
            pol_patch["channel_reduction"] = channel_reduction
        if pol_patch:
            pol = replace(pol, **pol_patch)

        opts = save_options if save_options is not None else SaveOptions()
        opts_patch: dict = {}
        if overwrite is not None:
            opts_patch["overwrite"] = bool(overwrite)
        if jpeg_quality is not None:
            opts_patch["quality"] = int(jpeg_quality)
        if convert_uint8 is not None:
            opts_patch["convert_uint8"] = bool(convert_uint8)
        if opts_patch:
            opts = replace(opts, **opts_patch)

        object.__setattr__(self, "policy", pol)
        object.__setattr__(self, "save_options", opts)

    def open(self, path: PathLike) -> ImageSource:
        src = _source_for(pathify(path))
        src.default_policy = self.policy
        return src

    def load(
        self,
        path: PathLike,
        *,
        region: RegionLike = None,
        downsample: int = 1,
        channels: Channels = None,
        level: Level = "auto",
        policy: Optional[ImagePolicy] = None,
    ) -> SpatialImage:
        with self.open(path) as src:
            return src.read(
                region=region,
                downsample=downsample,
                channels=channels,
                level=level,
                policy=policy if policy is not None else self.policy,
            )

    def save(
        self,
        image: Union[Tensor, SpatialImage, object],
        path: PathLike,
        *,
        options: Optional[SaveOptions] = None,
        overwrite: Optional[bool] = None,
        source: Optional[SourceInfo] = None,
        channel_names: Optional[Sequence[str]] = None,
        metadata_mode: Literal["preserve", "drop"] = "preserve",
    ) -> pathlib.Path:
        opts = options if options is not None else self.save_options
        if overwrite is not None:
            opts = replace(opts, overwrite=bool(overwrite))

        p = pathify(path)
        suffix = p.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported save extension: {suffix!r} "
                f"(supported: {sorted(_SUPPORTED_EXTENSIONS)})."
            )
        ensure_parent_dir(p)
        assert_overwrite(p, opts.overwrite)

        if source is None and isinstance(image, SpatialImage):
            source = image.space.source
            if channel_names is None and source.channel_names:
                channel_names = source.channel_names

        img = to_bchw(_coerce_tensor(image)).detach()
        if img.shape[0] != 1:
            raise ValueError(f"save() requires batch size 1, got shape {tuple(img.shape)}.")
        chw = img[0].contiguous().cpu()

        if suffix in _TIFF_EXTENSIONS:
            _write_tiff(
                chw, p, opts,
                source=source if metadata_mode == "preserve" else None,
                channel_names=tuple(channel_names) if channel_names else None,
            )
        else:
            _write_raster(chw, p, suffix, opts)
        return p

    def save_tiff(
        self,
        image: Union[Tensor, SpatialImage, object],
        path: PathLike,
        *,
        config: Optional[TiffExportConfig] = None,
        source: Optional[SourceInfo] = None,
    ) -> pathlib.Path:
        """Write a TIFF file with explicit export configuration.

        Parameters
        ----------
        image
            Tensor, SpatialImage, or any image container.
        path
            Output file path (must end in ``.tif`` / ``.tiff``).
        config
            TIFF export settings.  Defaults to :class:`TiffExportConfig`
            with all-auto resolution.
        source
            Optional source metadata for calibration / format context.
            When *image* is a :class:`SpatialImage` and *source* is
            ``None``, source metadata is taken from the image.
        """
        cfg = config if config is not None else TiffExportConfig()

        p = pathify(path)
        suffix = p.suffix.lower()
        if suffix not in _TIFF_EXTENSIONS:
            raise ValueError(
                f"save_tiff() requires a .tif/.tiff path, got {suffix!r}.")
        ensure_parent_dir(p)
        assert_overwrite(p, cfg.overwrite)

        if source is None and isinstance(image, SpatialImage):
            source = image.space.source

        img = to_bchw(_coerce_tensor(image)).detach()
        if img.shape[0] != 1:
            raise ValueError(f"save_tiff() requires batch size 1, got shape {tuple(img.shape)}.")
        chw = img[0].contiguous().cpu()

        _write_tiff_configured(chw, p, cfg, source=source)
        return p

    def save_preview(
        self,
        image: Union[Tensor, SpatialImage, object],
        path: PathLike,
        *,
        config: Optional[PreviewConfig] = None,
    ) -> pathlib.Path:
        """Save a raster preview with presentation adjustments.

        Applies normalization, gamma, optional colormap, and channel
        selection before writing a PNG/JPEG/BMP file.

        Parameters
        ----------
        image
            Tensor, SpatialImage, or any image container.
        path
            Output file path (PNG, JPEG, or BMP extension).
        config
            Preview settings.  Defaults to :class:`PreviewConfig` with
            min-max normalisation and gamma 1.0.
        """
        cfg = config if config is not None else PreviewConfig()

        p = pathify(path)
        suffix = p.suffix.lower()
        if suffix not in _RASTER_EXTENSIONS:
            raise ValueError(
                f"save_preview() requires a raster path (.png/.jpg/.bmp), got {suffix!r}."
            )
        ensure_parent_dir(p)
        assert_overwrite(p, cfg.overwrite)

        img = to_bchw(_coerce_tensor(image)).detach()
        if img.shape[0] != 1:
            raise ValueError(
                f"save_preview() requires batch size 1, got shape {tuple(img.shape)}."
            )
        chw = img[0].contiguous().cpu()

        _write_preview(chw, p, suffix, cfg)
        return p

    def list(
        self,
        directory: PathLike,
        *,
        extensions: Optional[Iterable[str]] = None,
    ) -> list[pathlib.Path]:
        exts = coerce_extensions(extensions)
        d = pathify(directory)
        return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)

    def exists(self, path: PathLike) -> bool:
        return pathify(path).exists()


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------


def load_image(
    path: PathLike,
    *,
    region: RegionLike = None,
    downsample: int = 1,
    channels: Channels = None,
    level: Level = "auto",
    grayscale: bool = False,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = None,
) -> SpatialImage:
    policy = ImagePolicy(grayscale=grayscale, dtype=dtype, device=device)
    return ImageIO(policy=policy).load(
        path,
        region=region,
        downsample=downsample,
        channels=channels,
        level=level,
    )


def open_image(path: PathLike) -> ImageSource:
    return ImageIO().open(path)


def save_image(
    image: Union[Tensor, SpatialImage, object],
    path: PathLike,
    *,
    overwrite: bool = False,
    quality: int = 95,
    convert_uint8: Optional[bool] = None,
    tiff_compression: Optional[str] = None,
    tiff_photometric: Optional[str] = None,
) -> pathlib.Path:
    opts = SaveOptions(
        overwrite=overwrite,
        quality=quality,
        convert_uint8=convert_uint8,
        tiff_compression=tiff_compression,
        tiff_photometric=tiff_photometric,
    )
    return ImageIO(save_options=opts).save(image, path)


def save_tiff(
    image: Union[Tensor, SpatialImage, object],
    path: PathLike,
    *,
    config: Optional[TiffExportConfig] = None,
    source: Optional[SourceInfo] = None,
) -> pathlib.Path:
    """Write a TIFF with explicit export configuration.

    Convenience wrapper around ``ImageIO().save_tiff(...)``.
    """
    return ImageIO().save_tiff(image, path, config=config, source=source)


def save_preview(
    image: Union[Tensor, SpatialImage, object],
    path: PathLike,
    *,
    config: Optional[PreviewConfig] = None,
) -> pathlib.Path:
    """Save a raster preview with presentation adjustments.

    Convenience wrapper around ``ImageIO().save_preview(...)``.
    """
    return ImageIO().save_preview(image, path, config=config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_tensor(image: Union[Tensor, SpatialImage, object]) -> Tensor:
    """Extract a raw tensor from various image containers.

    Supported inputs (checked in order):
    1. ``torch.Tensor`` — returned as-is.
    2. ``SpatialImage`` — ``image.detail.image``.
    3. Any object with an ``.image`` attribute that is a Tensor.
    4. Any mapping-like object where ``"image" in obj`` and
       ``obj.get("image")`` returns a Tensor (e.g. TensorDict).
    """
    if torch.is_tensor(image):
        return image
    if isinstance(image, SpatialImage):
        return image.detail.image
    if hasattr(image, "image"):
        attr = getattr(image, "image")
        if torch.is_tensor(attr):
            return attr
    if hasattr(image, "get") and "image" in image:  # type: ignore[operator]
        got = image.get("image")  # type: ignore[call-arg]
        if torch.is_tensor(got):
            return got
    raise TypeError(f"Expected a Tensor or image container, got {type(image)!r}.")


def _write_tiff(
    chw: Tensor,
    path: pathlib.Path,
    opts: SaveOptions,
    *,
    source: Optional[SourceInfo] = None,
    channel_names: Optional[tuple[str, ...]] = None,
) -> None:
    """Legacy entry point — bridges SaveOptions into TiffExportConfig."""
    fmt: Literal["auto", "ome-tiff", "imagej-tiff", "tiff"] = "auto"
    if source is not None and source.format in ("ome-tiff", "imagej-tiff", "tiff"):
        fmt = source.format  # type: ignore[assignment]
    photo: Literal["auto", "minisblack", "rgb"] = "auto"
    if opts.tiff_photometric is not None:
        photo = opts.tiff_photometric  # type: ignore[assignment]
    cfg = TiffExportConfig(
        format=fmt,
        photometric=photo,
        compression=opts.tiff_compression,
        convert_uint8=bool(opts.convert_uint8) if opts.convert_uint8 is not None else False,
        channel_names=channel_names,
        metadata_from_source=source is not None,
    )
    _write_tiff_configured(chw, path, cfg, source=source)


def _build_pyramid_levels(
    arr: np.ndarray,
    n_levels: int,
) -> list[np.ndarray]:
    """Generate *n_levels* successive 2× area-downsampled copies of *arr*.

    Works on both 2-D ``(H, W)`` and 3-D ``(C, H, W)`` arrays.
    """
    axes = "CYX" if arr.ndim == 3 else "YX"
    levels: list[np.ndarray] = []
    current = arr
    for _ in range(n_levels):
        y_ax = axes.find("Y")
        x_ax = axes.find("X")
        if current.shape[y_ax] < 2 or current.shape[x_ax] < 2:
            break
        down = area_downsample(current, axes, 2)
        # area_downsample returns float32; cast back to source dtype
        if down.dtype != arr.dtype:
            if np.issubdtype(arr.dtype, np.integer):
                down = np.round(down).astype(arr.dtype)
            else:
                down = down.astype(arr.dtype)
        levels.append(down)
        current = down
    return levels


def _write_tiff_configured(
    chw: Tensor,
    path: pathlib.Path,
    cfg: TiffExportConfig,
    *,
    source: Optional[SourceInfo] = None,
) -> None:
    """Write a TIFF file using :class:`TiffExportConfig`."""
    img = chw
    if img.dtype == torch.bool:
        img = img.to(dtype=torch.uint8) * 255
    if cfg.convert_uint8 and img.dtype != torch.uint8:
        img = _to_uint8_chw(img)

    arr = img.numpy()
    c = int(arr.shape[0])
    if c == 1:
        arr2d_or_cyx = arr[0]
    else:
        arr2d_or_cyx = arr

    effective_source = source if cfg.metadata_from_source else None
    fmt = cfg._resolve_format(effective_source)
    photometric = cfg._resolve_photometric(fmt, arr2d_or_cyx.ndim, c)
    channel_names = cfg._resolve_channel_names(effective_source)
    if fmt == "ome-tiff" and c > 1 and channel_names and cfg.photometric == "auto":
        photometric = "minisblack"
    resolution, resolutionunit = _tiff_resolution_from_source(effective_source)

    kwargs: dict = {}
    if cfg.compression is not None:
        kwargs["compression"] = cfg.compression
    kwargs["photometric"] = photometric

    if fmt == "ome-tiff":
        # Use natural CYX (multi-channel) or YX (single-channel) axes.  Some
        # readers (older Bio-Formats, naive TIFF readers) treat synthetic
        # singleton-T planes as a 6-frame time series, so we never insert a
        # phantom T dimension.
        ome_axes = "CYX" if arr2d_or_cyx.ndim == 3 else "YX"
        ome_metadata = _build_ome_metadata(effective_source, channel_names, c, axes=ome_axes)
        with tifffile.TiffWriter(str(path), ome=True) as tw:
            write_kwargs = dict(kwargs)
            if resolution is not None:
                write_kwargs["resolution"] = resolution
                write_kwargs["resolutionunit"] = resolutionunit
            n_sub = cfg.pyramid_levels
            if n_sub > 0:
                write_kwargs["subifds"] = int(n_sub)
            tw.write(arr2d_or_cyx, metadata=ome_metadata, **write_kwargs)
            if n_sub > 0:
                for sub_arr in _build_pyramid_levels(arr2d_or_cyx, n_sub):
                    tw.write(sub_arr, subfiletype=1, **kwargs)
        return

    if fmt == "imagej-tiff":
        # ImageJ hyperstack stores multi-channel as separate grayscale planes.
        if arr2d_or_cyx.ndim == 3:
            kwargs["photometric"] = "minisblack"
        ij_metadata: dict = {"axes": "CYX" if arr2d_or_cyx.ndim == 3 else "YX"}
        if channel_names:
            ij_metadata["Labels"] = list(channel_names)
        if effective_source is not None and effective_source.unit:
            ij_metadata["unit"] = effective_source.unit
        kwargs["metadata"] = ij_metadata
        kwargs["imagej"] = True
        if resolution is not None:
            kwargs["resolution"] = resolution
        tifffile.imwrite(str(path), arr2d_or_cyx, **kwargs)
        return

    # Plain TIFF fallback
    if arr2d_or_cyx.ndim == 3:
        kwargs.setdefault("metadata", {"axes": "CYX"})
    if resolution is not None:
        kwargs["resolution"] = resolution
        kwargs["resolutionunit"] = resolutionunit
    tifffile.imwrite(str(path), arr2d_or_cyx, **kwargs)


def _tiff_resolution_from_source(
    source: Optional[SourceInfo],
) -> tuple[Optional[tuple[float, float]], Optional[str]]:
    """Convert SourceInfo pixel calibration to baseline TIFF Resolution tags.

    TIFF baseline only allows ``ResolutionUnit`` ∈ {NONE, INCH, CENTIMETER}.
    Sub-cm units (μm, mm) are converted to pixels-per-cm so legacy tools
    that ignore OME-XML still recover the correct physical scale.
    """
    if source is None or source.pixel_size_xy is None:
        return None, None
    sx, sy = source.pixel_size_xy
    if sx <= 0 or sy <= 0:
        return None, None
    unit = (source.unit or "").lower()
    if unit in ("um", "micron", "µm", "micrometer", "micrometre"):
        # 1 cm = 10000 µm; pixels-per-cm = 10000 / pixel_size_µm
        return (10000.0 / float(sx), 10000.0 / float(sy)), "CENTIMETER"
    if unit in ("mm", "millimeter", "millimetre"):
        return (10.0 / float(sx), 10.0 / float(sy)), "CENTIMETER"
    if unit in ("cm", "centimeter", "centimetre"):
        return (1.0 / float(sx), 1.0 / float(sy)), "CENTIMETER"
    if unit in ("inch", "in"):
        return (1.0 / float(sx), 1.0 / float(sy)), "INCH"
    return (1.0 / float(sx), 1.0 / float(sy)), "NONE"


def _normalize_ome_unit(unit: Optional[str]) -> Optional[str]:
    """Map free-form unit strings to OME's preferred symbols (µm, mm, ...)."""
    if not unit:
        return None
    low = unit.lower()
    if low in ("um", "micron", "µm", "micrometer", "micrometre"):
        return "µm"
    if low in ("mm", "millimeter", "millimetre"):
        return "mm"
    if low in ("cm", "centimeter", "centimetre"):
        return "cm"
    if low in ("nm", "nanometer", "nanometre"):
        return "nm"
    if low in ("inch", "in"):
        return "in"
    if low in ("m", "meter", "metre"):
        return "m"
    return unit


def _sanitize_channel_names(
    names: Optional[Sequence[str]],
    count: int,
) -> Optional[list[str]]:
    """Make channel names safe for tools that mis-parse colons / control chars."""
    if not names:
        return None
    out: list[str] = []
    for raw in list(names)[:count]:
        text = str(raw)
        # Some Bio-Formats versions split on ':' when constructing channel
        # IDs, which corrupts compound names like ``reference:DAPI``.
        cleaned = text.replace(":", "_").strip()
        out.append(cleaned or f"channel_{len(out)}")
    while len(out) < count:
        out.append(f"channel_{len(out)}")
    return out


def _build_ome_metadata(
    source: Optional[SourceInfo],
    channel_names: Optional[tuple[str, ...]],
    channel_count: int,
    *,
    axes: Optional[str] = None,
) -> dict:
    md: dict = {"axes": axes or ("CYX" if channel_count > 1 else "YX")}
    if source is not None and source.pixel_size_xy is not None:
        sx, sy = source.pixel_size_xy
        ome_unit = _normalize_ome_unit(source.unit)
        md["PhysicalSizeX"] = float(sx)
        md["PhysicalSizeY"] = float(sy)
        if ome_unit:
            md["PhysicalSizeXUnit"] = ome_unit
            md["PhysicalSizeYUnit"] = ome_unit
        if source.origin_xy is not None:
            ox, oy = source.origin_xy
            if channel_count > 1:
                plane: dict[str, object] = {
                    "PositionX": [float(ox)] * int(channel_count),
                    "PositionY": [float(oy)] * int(channel_count),
                }
            else:
                plane = {
                    "PositionX": float(ox),
                    "PositionY": float(oy),
                }
            if ome_unit:
                if channel_count > 1:
                    plane["PositionXUnit"] = [ome_unit] * int(channel_count)
                    plane["PositionYUnit"] = [ome_unit] * int(channel_count)
                else:
                    plane["PositionXUnit"] = ome_unit
                    plane["PositionYUnit"] = ome_unit
            md["Plane"] = plane
    cleaned_names = _sanitize_channel_names(channel_names, channel_count)
    if cleaned_names:
        md["Channel"] = {"Name": cleaned_names}
    return md


def _write_raster(chw: Tensor, path: pathlib.Path, suffix: str, opts: SaveOptions) -> None:
    img = chw
    convert = opts.convert_uint8 if opts.convert_uint8 is not None else True

    if img.dtype.is_floating_point or img.dtype == torch.bool:
        if not convert:
            raise ValueError(
                f"Saving {suffix!r} from dtype {img.dtype} requires convert_uint8=True."
            )
        img = _to_uint8_chw(img)

    c = int(img.shape[0])
    if suffix in _JPEG_EXTENSIONS and c not in {1, 3}:
        raise ValueError(f"JPEG save requires 1 or 3 channels, got {c}.")
    if suffix in _PNG_BMP_EXTENSIONS and c not in {1, 3, 4}:
        raise ValueError(f"{suffix!r} save requires 1, 3, or 4 channels, got {c}.")

    if img.dtype == torch.uint16 and suffix == ".png" and c == 1:
        Image.fromarray(img[0].numpy()).save(path)
        return

    if img.dtype != torch.uint8:
        if convert:
            img = _to_uint8_chw(img)
        else:
            raise ValueError(f"Saving {suffix!r} from dtype {img.dtype} is unsupported.")

    arr = img[0].numpy() if c == 1 else img.permute(1, 2, 0).numpy()
    pil = Image.fromarray(arr)
    pil.save(path, **({"quality": int(opts.quality)} if suffix in _JPEG_EXTENSIONS else {}))


def _write_preview(
    chw: Tensor,
    path: pathlib.Path,
    suffix: str,
    cfg: PreviewConfig,
) -> None:
    """Apply a presentation pipeline and save as a raster preview.

    Pipeline: channel selection → downsample → normalize + gamma →
    colormap / RGB conversion → uint8 → PIL save.
    """
    img = chw.float() if not chw.dtype.is_floating_point else chw

    # 1. Channel selection
    img = select_channels(img, cfg.channel)

    # 2. Downsample
    img = downsample_for_display(img, cfg.max_size)

    # 3. Normalize + gamma
    img = prepare_for_display(
        img,
        mode=cfg.normalize,
        percentile_low=cfg.percentile_low,
        percentile_high=cfg.percentile_high,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        gamma=cfg.gamma,
    )

    # 4. Colormap or RGB expansion
    c = int(img.shape[0])
    if cfg.colormap is not None and c == 1:
        img = apply_colormap(img, cfg.colormap)
    else:
        img = to_rgb(img)

    # 5. Convert to uint8
    img = _render_to_uint8(img)

    # 6. PIL save
    arr = img.permute(1, 2, 0).numpy()
    pil = Image.fromarray(arr)
    save_kwargs: dict = {}
    if suffix in _JPEG_EXTENSIONS:
        save_kwargs["quality"] = int(cfg.quality)
    pil.save(path, **save_kwargs)


def _to_uint8_chw(image: Tensor) -> Tensor:
    if image.dtype == torch.uint8:
        return image
    if image.dtype == torch.bool:
        return image.to(dtype=torch.uint8) * 255
    if image.dtype.is_floating_point:
        return image_to_uint8(image.unsqueeze(0))[0]
    if image.dtype in {
        torch.int8,
        torch.uint16,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint32,
        torch.uint64,
    }:
        # image_to_uint8 normalizes per-batch via min/max (with +1e-8 guard
        # against constant-value images), so arbitrary integer ranges are
        # mapped correctly after the float32 cast.
        work = image.to(dtype=torch.float32)
        return image_to_uint8(work.unsqueeze(0))[0]
    raise ValueError(f"Unsupported image dtype for uint8 conversion: {image.dtype}.")
