from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import torch

from batchmatch.helpers.affine import (
    mat_compose,
    mat_crop,
    mat_downsample_int,
    mat_identity,
    mat_inv,
    mat_pad,
    mat_resize,
)

if TYPE_CHECKING:  # pragma: no cover
    from batchmatch.base.tensordicts import ImageDetail

__all__ = [
    "RegionYXHW",
    "PaddingLTRB",
    "SourceFormat",
    "SourceInfo",
    "ImageSpace",
    "SpatialImage",
]


SourceFormat = Literal["ome-tiff", "imagej-tiff", "tiff", "raster"]



@dataclass(frozen=True)
class RegionYXHW:
    """Rectangular region in pixels.

    ``(y, x)`` is the top-left corner, ``(h, w)`` the extent. Coordinates
    are in whatever space the containing object declares — at load time
    they are in source full-resolution pixels.
    """

    y: int
    x: int
    h: int
    w: int

    def __post_init__(self) -> None:
        if self.y < 0 or self.x < 0:
            raise ValueError(f"RegionYXHW offsets must be non-negative, got y={self.y}, x={self.x}.")
        if self.h <= 0 or self.w <= 0:
            raise ValueError(f"RegionYXHW extents must be positive, got h={self.h}, w={self.w}.")

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def x2(self) -> int:
        return self.x + self.w

    def to_list(self) -> list[int]:
        return [int(self.y), int(self.x), int(self.h), int(self.w)]

    @classmethod
    def from_list(cls, data: Sequence[int]) -> "RegionYXHW":
        if len(data) != 4:
            raise ValueError(f"RegionYXHW.from_list expected 4 ints, got {list(data)!r}.")
        y, x, h, w = (int(v) for v in data)
        return cls(y=y, x=x, h=h, w=w)

    def scaled(self, sx: float, sy: float) -> "RegionYXHW":
        """Scale the region by the given ``(sx, sy)`` factor.

        Used by loaders to translate a full-resolution region into
        pyramid-level coordinates. Rounds to nearest integer.
        """
        return RegionYXHW(
            y=int(round(self.y * sy)),
            x=int(round(self.x * sx)),
            h=int(round(self.h * sy)),
            w=int(round(self.w * sx)),
        )


@dataclass(frozen=True)
class PaddingLTRB:
    """Padding on each side of an image, in pixels."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        for name, value in (
            ("left", self.left),
            ("top", self.top),
            ("right", self.right),
            ("bottom", self.bottom),
        ):
            if value < 0:
                raise ValueError(f"PaddingLTRB.{name} must be non-negative, got {value}.")

    @property
    def horizontal(self) -> int:
        return self.left + self.right

    @property
    def vertical(self) -> int:
        return self.top + self.bottom

    def to_list(self) -> list[int]:
        return [int(self.left), int(self.top), int(self.right), int(self.bottom)]

    @classmethod
    def from_list(cls, data: Sequence[int]) -> "PaddingLTRB":
        if len(data) != 4:
            raise ValueError(f"PaddingLTRB.from_list expected 4 ints, got {list(data)!r}.")
        left, top, right, bottom = (int(v) for v in data)
        return cls(left=left, top=top, right=right, bottom=bottom)

    def as_torch_pad(self) -> tuple[int, int, int, int]:
        """Order expected by ``torch.nn.functional.pad``: (left, right, top, bottom)."""
        return (int(self.left), int(self.right), int(self.top), int(self.bottom))


@dataclass(frozen=True)
class SourceInfo:
    """Immutable description of a source image file."""

    source_path: str
    series_index: int
    level_count: int
    level_shapes: tuple[tuple[int, int], ...]
    axes: str
    dtype: str
    channel_names: tuple[str, ...] = ()
    pixel_size_xy: Optional[tuple[float, float]] = None
    unit: Optional[str] = None
    origin_xy: Optional[tuple[float, float]] = None
    physical_extent_xy: Optional[tuple[float, float]] = None
    format: SourceFormat = "raster"

    @property
    def path(self) -> Path:
        return Path(self.source_path)

    @property
    def base_shape_hw(self) -> tuple[int, int]:
        if not self.level_shapes:
            raise ValueError("SourceInfo.level_shapes is empty.")
        return self.level_shapes[0]

    def level_shape_hw(self, level: int) -> tuple[int, int]:
        if level < 0 or level >= len(self.level_shapes):
            raise IndexError(
                f"pyramid level {level} out of range for {len(self.level_shapes)} levels."
            )
        return self.level_shapes[level]

    def has_calibration(self) -> bool:
        return self.pixel_size_xy is not None and self.origin_xy is not None

    def to_dict(self) -> dict:
        cal = None
        if self.pixel_size_xy is not None or self.origin_xy is not None:
            cal = {
                "pixel_size_xy": (
                    list(self.pixel_size_xy) if self.pixel_size_xy is not None else None
                ),
                "unit": self.unit,
                "origin_xy": (
                    list(self.origin_xy) if self.origin_xy is not None else None
                ),
                "extent_xy": (
                    list(self.physical_extent_xy)
                    if self.physical_extent_xy is not None
                    else None
                ),
            }
        return {
            "path": self.source_path,
            "format": self.format,
            "series": int(self.series_index),
            "axes": self.axes,
            "dtype": self.dtype,
            "channels": list(self.channel_names),
            "pyramid": {
                "levels": int(self.level_count),
                "shapes_hw": [list(s) for s in self.level_shapes],
            },
            "calibration": cal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceInfo":
        pyr = data.get("pyramid", {})
        cal = data.get("calibration") or {}
        return cls(
            source_path=str(data["path"]),
            series_index=int(data.get("series", 0)),
            level_count=int(pyr.get("levels", 1)),
            level_shapes=tuple(
                tuple(int(v) for v in s) for s in pyr.get("shapes_hw", [])
            ),
            axes=str(data.get("axes", "")),
            dtype=str(data.get("dtype", "")),
            channel_names=tuple(data.get("channels", []) or ()),
            pixel_size_xy=(
                tuple(float(v) for v in cal["pixel_size_xy"])
                if cal.get("pixel_size_xy") is not None
                else None
            ),
            unit=cal.get("unit"),
            origin_xy=(
                tuple(float(v) for v in cal["origin_xy"])
                if cal.get("origin_xy") is not None
                else None
            ),
            physical_extent_xy=(
                tuple(float(v) for v in cal["extent_xy"])
                if cal.get("extent_xy") is not None
                else None
            ),
            format=str(data.get("format", "raster")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ImageSpace:
    """Exact mapping from source pixels to the current tensor's pixels.

    ``matrix_image_from_source`` is a 3x3 matrix taking a source
    full-resolution pixel ``(x, y, 1)`` to the current image pixel.
    Loaders construct the initial matrix from ``(pyramid_level, region,
    downsample)``; subsequent spatial stages compose additional matrices
    via :meth:`compose`.

    ``channel_selection`` records which source channels were selected at
    load time (``None`` means all channels).
    """

    source: SourceInfo
    pyramid_level: int
    region: RegionYXHW
    downsample: int
    shape_hw: tuple[int, int]
    matrix_image_from_source: np.ndarray
    channel_selection: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix_image_from_source, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"matrix_image_from_source must be 3x3, got {m.shape}.")
        object.__setattr__(self, "matrix_image_from_source", m)
        if self.pyramid_level < 0:
            raise ValueError(f"pyramid_level must be >= 0, got {self.pyramid_level}.")
        if self.downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {self.downsample}.")
        if len(self.shape_hw) != 2:
            raise ValueError(f"shape_hw must be (h, w), got {self.shape_hw!r}.")

    @property
    def matrix_source_from_image(self) -> np.ndarray:
        return mat_inv(self.matrix_image_from_source)

    @classmethod
    def from_load(
        cls,
        *,
        source: SourceInfo,
        pyramid_level: int,
        region: RegionYXHW,
        downsample: int,
        shape_hw: tuple[int, int],
        channel_selection: Optional[Sequence[int]] = None,
    ) -> "ImageSpace":
        """Build an :class:`ImageSpace` for a freshly loaded tensor.

        Composes ``matrix_image_from_source`` from the loader-known
        parameters:

        1. scale from source full-res to the chosen pyramid level
        2. translate by the region offset (in pyramid-level coords)
        3. integer downsample on top

        ``region`` is given in **source full-resolution coordinates**; the
        loader may have physically cropped at the pyramid level, but the
        region stored here is what the caller asked for.
        """
        base_h, base_w = source.base_shape_hw
        level_h, level_w = source.level_shape_hw(pyramid_level)

        mats: list[np.ndarray] = []
        if (base_h, base_w) != (level_h, level_w):
            mats.append(mat_resize(base_w, base_h, level_w, level_h))

        sx = float(level_w) / float(base_w)
        sy = float(level_h) / float(base_h)
        x_l = float(region.x) * sx
        y_l = float(region.y) * sy
        mats.append(mat_crop(x_l, y_l, float(region.w) * sx, float(region.h) * sy))

        if downsample > 1:
            mats.append(mat_downsample_int(int(downsample)))

        matrix = mat_compose(mats) if mats else mat_identity()
        return cls(
            source=source,
            pyramid_level=int(pyramid_level),
            region=region,
            downsample=int(downsample),
            shape_hw=(int(shape_hw[0]), int(shape_hw[1])),
            matrix_image_from_source=matrix,
            channel_selection=tuple(int(c) for c in channel_selection) if channel_selection is not None else None,
        )

    def compose(
        self,
        matrix_next_from_current: np.ndarray,
        *,
        shape_hw: tuple[int, int],
    ) -> "ImageSpace":
        """Return a new :class:`ImageSpace` with one more spatial op applied."""
        m_next = np.asarray(matrix_next_from_current, dtype=np.float64) @ self.matrix_image_from_source
        return replace(
            self,
            matrix_image_from_source=m_next,
            shape_hw=(int(shape_hw[0]), int(shape_hw[1])),
        )

    def with_channel_selection(self, channel_selection: Optional[Sequence[int]]) -> "ImageSpace":
        return replace(
            self,
            channel_selection=(
                tuple(int(c) for c in channel_selection) if channel_selection is not None else None
            ),
        )

    def to_dict(self) -> dict:
        """Serialize the geometric parameters (no source info).

        Source metadata is stored once at the manifest top level; see
        :meth:`ProductIO.save`.
        """
        return {
            "pyramid_level": int(self.pyramid_level),
            "region_yxhw": self.region.to_list(),
            "downsample": int(self.downsample),
            "channels": (
                list(self.channel_selection) if self.channel_selection is not None else None
            ),
            "shape_hw": [int(self.shape_hw[0]), int(self.shape_hw[1])],
            "matrix_image_from_source": self.matrix_image_from_source.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict, source: "SourceInfo") -> "ImageSpace":
        """Reconstruct from a dict plus the owning :class:`SourceInfo`."""
        return cls(
            source=source,
            pyramid_level=int(data["pyramid_level"]),
            region=RegionYXHW.from_list(data["region_yxhw"]),
            downsample=int(data["downsample"]),
            shape_hw=(int(data["shape_hw"][0]), int(data["shape_hw"][1])),
            matrix_image_from_source=np.asarray(data["matrix_image_from_source"], dtype=np.float64),
            channel_selection=(
                tuple(int(c) for c in data["channels"])
                if data.get("channels") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SpatialImage:
    """An :class:`ImageDetail` tensor paired with its :class:`ImageSpace`.

    Flows through the registration pipeline: loaders return it, spatial
    stages consume and return it, registration consumes two of them.

    The pair is immutable — any geometry-changing operation should return
    a new :class:`SpatialImage` via :meth:`with_detail_and_compose`.
    """

    detail: "ImageDetail"
    space: ImageSpace

    @property
    def image(self) -> torch.Tensor:
        return self.detail.image

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.space.shape_hw

    def to(self, device: torch.device | str) -> "SpatialImage":
        return SpatialImage(detail=self.detail.to(device), space=self.space)

    def clone(self) -> "SpatialImage":
        return SpatialImage(detail=self.detail.clone(), space=self.space)

    def with_detail(self, detail: "ImageDetail") -> "SpatialImage":
        """Replace the detail while keeping the same space (geometry unchanged)."""
        return SpatialImage(detail=detail, space=self.space)

    def with_detail_and_compose(
        self,
        detail: "ImageDetail",
        matrix_next_from_current: np.ndarray,
        *,
        shape_hw: tuple[int, int],
    ) -> "SpatialImage":
        """Replace the detail and compose one more spatial op into the space."""
        return SpatialImage(
            detail=detail,
            space=self.space.compose(matrix_next_from_current, shape_hw=shape_hw),
        )
