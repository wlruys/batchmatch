"""Final-resolution export of a registered moving image.

Consumes :class:`RegistrationTransform` (or a manifest round-trip) and
applies :attr:`RegistrationTransform.matrix_ref_full_from_mov_full` to
every moving channel in one pass. Registration may use a single channel;
export reopens the moving source with ``channels="all"`` by default.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import numpy as np
import torch

from batchmatch.io.images import ImageIO, load_image
from batchmatch.io.schema import REGISTRATION_SCHEMA
from batchmatch.io.space import SourceInfo
from batchmatch.io.utils import PathLike, pathify

if TYPE_CHECKING:
    from batchmatch.search.transform import RegistrationTransform
from batchmatch.warp.resample import warp_to_reference

__all__ = ["RegisteredExport", "export_registered"]


Channels = Union[Literal["all"], Sequence[Union[int, str]]]
Canvas = Literal["reference", "union"]


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


def _union_bbox_ref_full(transform: "RegistrationTransform") -> tuple[int, int, int, int]:
    ref_h, ref_w = _ref_canvas_hw(transform.reference.source)
    mov_h, mov_w = transform.moving.source.base_shape_hw
    mov_ref = _apply_points(
        _image_corners((mov_h, mov_w)),
        transform.matrix_ref_full_from_mov_full,
    )

    x0 = int(np.floor(min(0.0, float(mov_ref[:, 0].min()))))
    y0 = int(np.floor(min(0.0, float(mov_ref[:, 1].min()))))
    x1 = int(np.ceil(max(float(ref_w), float(mov_ref[:, 0].max()))))
    y1 = int(np.ceil(max(float(ref_h), float(mov_ref[:, 1].max()))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid union export bbox {(x0, y0, x1, y1)}.")
    return x0, y0, x1, y1


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
    for ch in channels:
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
        fill_value: Fill value outside the moving footprint.
        overwrite: Allow overwriting ``output_path``.
    """
    transform, _ = _load_transform(source)

    mov_src = transform.moving.source
    ref_src = transform.reference.source
    mov_file = pathify(moving_path) if moving_path is not None else pathlib.Path(mov_src.source_path)

    channels_arg = None if channels == "all" else list(channels)
    moving = load_image(mov_file, downsample=1, channels=channels_arg, grayscale=False)

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

    out_path = pathify(output_path)
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
