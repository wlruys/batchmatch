"""Final-resolution export of a registered moving image.

Consumes :class:`RegistrationTransform` (or a manifest round-trip) and
applies :attr:`RegistrationTransform.matrix_ref_full_from_mov_full` to
every moving channel in one pass. Registration may use a single channel;
export reopens the moving source with ``channels="all"`` by default.
"""

from __future__ import annotations

import json
import pathlib
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

__all__ = ["export_registered"]


Channels = Union[Literal["all"], Sequence[Union[int, str]]]


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


def export_registered(
    source: Union["RegistrationTransform", dict[str, Any], PathLike],
    *,
    output_path: PathLike,
    moving_path: Optional[PathLike] = None,
    channels: Channels = "all",
    dtype: Optional[torch.dtype] = None,
    tile_size: Optional[int] = None,
    fill_value: float = 0.0,
    overwrite: bool = False,
) -> pathlib.Path:
    """Warp the moving image to the reference canvas and save.

    Args:
        source: A :class:`RegistrationTransform` or a manifest dict / path.
        output_path: Destination file path.
        moving_path: Override for the moving source file. Defaults to
            the path recorded in ``transform.moving.source``.
        channels: ``"all"`` (default) or an explicit list of channel
            indices / names to load from the moving source.
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

    ref_h, ref_w = _ref_canvas_hw(ref_src)
    warped = warp_to_reference(
        moving.detail,
        transform.matrix_ref_full_from_mov_full,
        out_hw=(ref_h, ref_w),
        tile_size=tile_size,
        fill_value=fill_value,
    )

    out = warped.image
    if dtype is not None:
        out = out.to(dtype=dtype)

    out_path = pathify(output_path)
    return ImageIO().save(
        out,
        out_path,
        overwrite=overwrite,
        source=ref_src,
        channel_names=moving.space.source.channel_names or None,
    )
