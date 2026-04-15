"""Internal TIFF metadata extraction helpers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
import tifffile

__all__ = ["TiffMeta", "read_meta", "_parse_meta"]

_UM_ALIASES = {"\u00b5m", "\\u00B5m", "um", "µm"}


@dataclass
class TiffMeta:
    """File-level metadata extracted from a TIFF file."""

    axes: str
    shape: tuple[int, ...]
    channel_names: list[str]
    pixel_size_xy: tuple[float, float] | None = None
    unit: str | None = None
    original_dtype: str = ""
    num_pyramid_levels: int = 1
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel_names = list(self.channel_names)

    @property
    def height(self) -> int:
        return self.shape[-2]

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def num_channels(self) -> int:
        return self.shape[-3] if len(self.shape) >= 3 else 1

    @property
    def physical_extent_um(self) -> tuple[float, float] | None:
        if self.pixel_size_xy is None:
            return None
        return (self.width * self.pixel_size_xy[0], self.height * self.pixel_size_xy[1])

    def memory_estimate_mb(self, dtype: np.dtype | str = "float32") -> float:
        itemsize = np.dtype(dtype).itemsize
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        return n_elements * itemsize / (1024 * 1024)

    @property
    def axes_raw(self) -> str:
        warnings.warn(
            "TiffMeta.axes_raw is deprecated. Use TiffMeta.axes.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.axes

    @property
    def shape_raw(self) -> tuple[int, ...]:
        warnings.warn(
            "TiffMeta.shape_raw is deprecated. Use TiffMeta.shape.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.shape

    @property
    def channel_names_raw(self) -> list[str]:
        warnings.warn(
            "TiffMeta.channel_names_raw is deprecated. Use TiffMeta.channel_names.",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self.channel_names)

    @property
    def pixel_size_xy_raw(self) -> tuple[float, float] | None:
        warnings.warn(
            "TiffMeta.pixel_size_xy_raw is deprecated. Use TiffMeta.pixel_size_xy.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.pixel_size_xy

    @property
    def pyramid_level_used(self) -> int:
        warnings.warn(
            "TiffMeta.pyramid_level_used is deprecated. "
            "Inspect the loaded tensor/view state instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return 0


def _count_pyramid_levels(tif: tifffile.TiffFile) -> int:
    if not tif.series:
        return 1
    series0 = tif.series[0]
    if hasattr(series0, "levels"):
        return len(series0.levels)
    return max(1, len(tif.series))


def _pixel_size_from_tags(page: tifffile.TiffPage) -> tuple[tuple[float, float], str | None] | None:
    xres = page.tags.get("XResolution")
    yres = page.tags.get("YResolution")
    if xres is None or yres is None:
        return None
    xnum, xden = xres.value
    ynum, yden = yres.value
    if xden == 0 or xnum == 0 or yden == 0 or ynum == 0:
        return None
    px_per_unit_x = xnum / xden
    px_per_unit_y = ynum / yden

    desc = page.tags.get("ImageDescription")
    if desc is not None:
        m = re.search(r"unit=([^\n\r]+)", desc.value)
        if m and m.group(1).strip() in _UM_ALIASES:
            return (1.0 / px_per_unit_x, 1.0 / px_per_unit_y), "um"

    res_unit = page.tags.get("ResolutionUnit")
    if res_unit is not None:
        if res_unit.value == 2:
            return (25400.0 / px_per_unit_x, 25400.0 / px_per_unit_y), "um"
        if res_unit.value == 3:
            return (10000.0 / px_per_unit_x, 10000.0 / px_per_unit_y), "um"
    return None


def _parse_imagej_meta(tif: tifffile.TiffFile) -> TiffMeta:
    series = tif.series[0]
    page = tif.pages[0]

    ij = page.tags.get("IJMetadata")
    channel_names = ij.value.get("Labels", []) if ij is not None else []
    px = _pixel_size_from_tags(page)
    pixel_size_xy = px[0] if px is not None else None
    unit = px[1] if px is not None else None

    return TiffMeta(
        axes=series.axes,
        shape=series.shape,
        channel_names=channel_names,
        pixel_size_xy=pixel_size_xy,
        unit=unit,
        original_dtype=str(series.dtype),
        num_pyramid_levels=_count_pyramid_levels(tif),
    )


def _parse_ome_meta(tif: tifffile.TiffFile) -> TiffMeta:
    series = tif.series[0]
    ome_xml = tif.ome_metadata
    root = ET.fromstring(ome_xml)

    ns_match = re.match(r"\{(.+)\}", root.tag)
    ns = {"ome": ns_match.group(1)} if ns_match else {}

    pixels = root.find(".//ome:Pixels", ns)

    channel_names = []
    if pixels is not None:
        for ch in pixels.findall("ome:Channel", ns):
            channel_names.append(ch.get("Name", ""))

    pixel_size_xy = None
    unit = None
    if pixels is not None:
        px = pixels.get("PhysicalSizeX")
        py = pixels.get("PhysicalSizeY")
        ux = pixels.get("PhysicalSizeXUnit", "")
        uy = pixels.get("PhysicalSizeYUnit", "")
        if px is not None and py is not None:
            px_val = float(px)
            py_val = float(py)
            if px_val > 0 and py_val > 0 and ux in _UM_ALIASES and uy in _UM_ALIASES:
                pixel_size_xy = (px_val, py_val)
                unit = "um"

    return TiffMeta(
        axes=series.axes,
        shape=series.shape,
        channel_names=channel_names,
        pixel_size_xy=pixel_size_xy,
        unit=unit,
        original_dtype=str(series.dtype),
        num_pyramid_levels=_count_pyramid_levels(tif),
    )


def _parse_meta(tif: tifffile.TiffFile) -> TiffMeta:
    return _parse_ome_meta(tif) if tif.is_ome else _parse_imagej_meta(tif)


def read_meta(path: str | Path) -> TiffMeta:
    with tifffile.TiffFile(Path(path)) as tif:
        return _parse_meta(tif)
