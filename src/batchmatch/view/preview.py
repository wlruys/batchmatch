"""Registration preview rendering.

Start from :attr:`RegistrationTransform.matrix_ref_full_from_mov_full`,
warp the moving image into the reference full-res canvas, then optionally
crop (union/intersection of valid regions) and resize for display.
Previews never replay the search pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from batchmatch.base.tensordicts import ImageDetail
from batchmatch.helpers.affine import apply_to_points
from batchmatch.io.space import SpatialImage
from batchmatch.search.transform import RegistrationTransform
from batchmatch.warp.resample import warp_to_reference
from .config import ChannelSelection

__all__ = ["RegistrationPreview", "render_registration_preview"]


CropMode = Literal["union", "intersection", None]


@dataclass(frozen=True)
class RegistrationPreview:
    reference: ImageDetail
    moving_warped: ImageDetail
    crop_box_xyxy: Optional[tuple[int, int, int, int]]
    output_hw: tuple[int, int]
    # Which channel(s) were selected when building this preview.  None means
    # all channels are present (or auto-selected at render time).
    channel: Optional[ChannelSelection] = None


def _moving_corners_in_ref(
    matrix_ref_from_mov: np.ndarray,
    mov_hw: tuple[int, int],
) -> np.ndarray:
    h, w = mov_hw
    corners = np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64
    )
    return apply_to_points(corners, matrix_ref_from_mov)


def _compute_crop(
    mode: CropMode,
    matrix_ref_from_mov: np.ndarray,
    mov_hw: tuple[int, int],
    ref_hw: tuple[int, int],
) -> Optional[tuple[int, int, int, int]]:
    if mode is None:
        return None
    ref_h, ref_w = ref_hw
    mov_in_ref = _moving_corners_in_ref(matrix_ref_from_mov, mov_hw)
    mx0 = float(mov_in_ref[:, 0].min())
    my0 = float(mov_in_ref[:, 1].min())
    mx1 = float(mov_in_ref[:, 0].max())
    my1 = float(mov_in_ref[:, 1].max())

    if mode == "intersection":
        x0 = max(0.0, mx0)
        y0 = max(0.0, my0)
        x1 = min(float(ref_w), mx1)
        y1 = min(float(ref_h), my1)
    elif mode == "union":
        x0 = min(0.0, mx0)
        y0 = min(0.0, my0)
        x1 = max(float(ref_w), mx1)
        y1 = max(float(ref_h), my1)
    else:
        raise ValueError(f"Unknown crop mode {mode!r}.")

    ix0 = int(np.floor(x0))
    iy0 = int(np.floor(y0))
    ix1 = int(np.ceil(x1))
    iy1 = int(np.ceil(y1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0, iy0, ix1, iy1)


def _crop_or_pad_to_ref(
    tensor: torch.Tensor,
    box: tuple[int, int, int, int],
    ref_hw: tuple[int, int],
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    ref_h, ref_w = ref_hw
    out_h = y1 - y0
    out_w = x1 - x0

    sx0 = max(0, min(ref_w, x0))
    sy0 = max(0, min(ref_h, y0))
    sx1 = max(sx0, min(ref_w, x1))
    sy1 = max(sy0, min(ref_h, y1))

    pad_left = max(0, sx0 - x0)
    pad_top = max(0, sy0 - y0)
    pad_right = max(0, x1 - sx1)
    pad_bottom = max(0, y1 - sy1)

    cropped = tensor[..., sy0:sy1, sx0:sx1]
    if pad_left or pad_right or pad_top or pad_bottom:
        cropped = F.pad(
            cropped,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
    assert cropped.shape[-2:] == (out_h, out_w), (
        f"preview crop shape {tuple(cropped.shape[-2:])} != target {(out_h, out_w)}"
    )
    return cropped


def render_registration_preview(
    transform: RegistrationTransform,
    moving: SpatialImage,
    reference: SpatialImage,
    *,
    crop_mode: CropMode = None,
    preview_size: Optional[tuple[int, int]] = None,
    fill_value: float = 0.0,
    tile_size: Optional[int] = None,
    channel: Optional[ChannelSelection] = None,
) -> RegistrationPreview:
    """Warp ``moving`` onto ``reference`` using the transform and return
    a ``RegistrationPreview``.

    ``crop_mode`` and ``preview_size`` are purely display-side and never
    feed back into transform estimation.

    For OME-TIFF or other multi-channel images, *channel* selects which
    channel(s) to include in the preview output (``int`` for a single
    channel, ``tuple[int, ...]`` for an explicit RGB mapping, ``None`` for
    auto-selection at render time).  Selecting a subset early reduces peak
    memory usage when working with images larger than 20 k × 20 k pixels.

    For very large images the warp is performed at native resolution.  Use
    *preview_size* to downsample the output before returning, or leave it
    as ``None`` when native resolution is required.
    """
    ref_h, ref_w = reference.shape_hw
    mov_h, mov_w = moving.shape_hw

    warped = warp_to_reference(
        moving.detail,
        transform.matrix_ref_full_from_mov_full,
        out_hw=(ref_h, ref_w),
        tile_size=tile_size,
        fill_value=fill_value,
    )

    ref_img = reference.detail.image
    mov_img = warped.image

    crop_box = _compute_crop(
        crop_mode,
        transform.matrix_ref_full_from_mov_full,
        (mov_h, mov_w),
        (ref_h, ref_w),
    )
    if crop_box is not None:
        ref_img = _crop_or_pad_to_ref(ref_img, crop_box, (ref_h, ref_w))
        mov_img = _crop_or_pad_to_ref(mov_img, crop_box, (ref_h, ref_w))

    # Channel selection: narrow to the requested channel(s) to reduce memory
    # usage before any further downsampling, especially important for
    # OME-TIFF images with many modality channels.
    if channel is not None:
        if isinstance(channel, int):
            ref_img = ref_img[:, channel : channel + 1]
            mov_img = mov_img[:, channel : channel + 1]
        else:
            indices = list(channel)
            ref_img = ref_img[:, indices]
            mov_img = mov_img[:, indices]

    if preview_size is not None:
        ph, pw = int(preview_size[0]), int(preview_size[1])
        ref_img = F.interpolate(ref_img, size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)
        mov_img = F.interpolate(mov_img, size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)

    from batchmatch.base.detail import build_image_td

    out_hw = (int(mov_img.shape[-2]), int(mov_img.shape[-1]))
    return RegistrationPreview(
        reference=build_image_td(ref_img),
        moving_warped=build_image_td(mov_img),
        crop_box_xyxy=crop_box,
        output_hw=out_hw,
        channel=channel,
    )
