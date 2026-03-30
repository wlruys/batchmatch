"""TIFF metadata extraction for ImageJ and OME-TIFF files."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

__all__ = ["TiffMeta", "read_meta"]

_UM_ALIASES = {"\u00b5m", "\\u00B5m", "um", "µm"}


@dataclass
class TiffMeta:
    """Metadata extracted from a TIFF file.

    Attributes that reflect the *file* are stored in ``axes_raw``,
    ``shape_raw``, and ``pixel_size_um_raw``.  After ``load_tiff``
    applies channel selection, region cropping, and downsampling the
    *output* attributes (``axes``, ``shape``, ``pixel_size_um``) are
    updated to match the returned tensor.

    For metadata-only reads (``read_meta``) raw and output values are
    identical.
    """

    # --- raw (file-level) ---------------------------------------------------
    axes_raw: str
    shape_raw: tuple[int, ...]
    channel_names: list[str]
    pixel_size_um_raw: float | None = None
    original_dtype: str = ""
    num_pyramid_levels: int = 1
    extra: dict = field(default_factory=dict)

    # --- output (after transforms) ------------------------------------------
    axes: str = ""
    shape: tuple[int, ...] = ()
    pixel_size_um: float | None = None
    pyramid_level_used: int = 0

    def __post_init__(self) -> None:
        if not self.axes:
            self.axes = self.axes_raw
        if not self.shape:
            self.shape = self.shape_raw
        if self.pixel_size_um is None and self.pixel_size_um_raw is not None:
            self.pixel_size_um = self.pixel_size_um_raw

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
        """(width_um, height_um) or None if pixel size is unknown."""
        if self.pixel_size_um is None:
            return None
        return (self.width * self.pixel_size_um, self.height * self.pixel_size_um)

    def memory_estimate_mb(self, dtype: np.dtype | str = "float32") -> float:
        """Estimated memory (MB) for the full image as the given dtype."""
        itemsize = np.dtype(dtype).itemsize
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        return n_elements * itemsize / (1024 * 1024)


# ---------------------------------------------------------------------------
# Metadata parsers
# ---------------------------------------------------------------------------

def _count_pyramid_levels(tif: tifffile.TiffFile) -> int:
    """Count the number of resolution levels in the first series."""
    if not tif.series:
        return 1
    series0 = tif.series[0]
    if hasattr(series0, "levels"):
        return len(series0.levels)
    return max(1, len(tif.series))


def _pixel_size_from_tags(page: tifffile.TiffPage) -> float | None:
    """Derive µm/pixel from TIFF resolution tags + unit heuristics."""
    xres = page.tags.get("XResolution")
    if xres is None:
        return None
    num, den = xres.value
    if den == 0 or num == 0:
        return None
    px_per_unit = num / den

    desc = page.tags.get("ImageDescription")
    if desc is not None:
        m = re.search(r"unit=([^\n\r]+)", desc.value)
        if m and m.group(1).strip() in _UM_ALIASES:
            return 1.0 / px_per_unit

    res_unit = page.tags.get("ResolutionUnit")
    if res_unit is not None:
        if res_unit.value == 2:  # inch
            return 25400.0 / px_per_unit
        if res_unit.value == 3:  # cm
            return 10000.0 / px_per_unit
    return None


def _parse_imagej_meta(tif: tifffile.TiffFile) -> TiffMeta:
    """Extract metadata from an ImageJ-style TIFF."""
    series = tif.series[0]
    page = tif.pages[0]

    ij = page.tags.get("IJMetadata")
    channel_names = ij.value.get("Labels", []) if ij is not None else []
    pixel_size_um = _pixel_size_from_tags(page)

    return TiffMeta(
        axes_raw=series.axes,
        shape_raw=series.shape,
        channel_names=channel_names,
        pixel_size_um_raw=pixel_size_um,
        original_dtype=str(series.dtype),
        num_pyramid_levels=_count_pyramid_levels(tif),
    )


def _parse_ome_meta(tif: tifffile.TiffFile) -> TiffMeta:
    """Extract metadata from an OME-TIFF (XML in ImageDescription)."""
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

    pixel_size_um = None
    if pixels is not None:
        px = pixels.get("PhysicalSizeX")
        unit = pixels.get("PhysicalSizeXUnit", "")
        if px is not None:
            px_val = float(px)
            if px_val > 0 and unit in _UM_ALIASES:
                pixel_size_um = px_val

    return TiffMeta(
        axes_raw=series.axes,
        shape_raw=series.shape,
        channel_names=channel_names,
        pixel_size_um_raw=pixel_size_um,
        original_dtype=str(series.dtype),
        num_pyramid_levels=_count_pyramid_levels(tif),
    )


def _parse_meta(tif: tifffile.TiffFile) -> TiffMeta:
    return _parse_ome_meta(tif) if tif.is_ome else _parse_imagej_meta(tif)


def read_meta(path: str | Path) -> TiffMeta:
    """Read only the metadata from a TIFF file (no pixel data loaded)."""
    with tifffile.TiffFile(Path(path)) as tif:
        return _parse_meta(tif)
