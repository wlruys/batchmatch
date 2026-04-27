"""Tiled multi-channel resampling helper.

:func:`warp_to_reference` warps a moving image into a reference canvas
given ``matrix_ref_from_mov`` (3x3, ``M_ref_from_mov`` in full-res pixel
coordinates). One ``grid_sample`` call per tile handles all channels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from batchmatch.base.detail import build_image_td
from batchmatch.base.tensordicts import ImageDetail
from batchmatch.helpers.affine import mat_inv

__all__ = ["warp_to_reference"]


def warp_to_reference(
    moving: ImageDetail | torch.Tensor,
    matrix_ref_from_mov: np.ndarray,
    *,
    out_hw: tuple[int, int],
    tile_size: Optional[int] = None,
    fill_value: float = 0.0,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> ImageDetail:
    """Warp ``moving`` onto the reference canvas.

    Args:
        moving: ``ImageDetail`` or ``[B, C, H, W]`` tensor in the moving
            full-res frame.
        matrix_ref_from_mov: 3x3 mapping from moving pixels to reference
            pixels. The inverse is used to sample.
        out_hw: ``(H, W)`` of the output canvas.
        tile_size: Tile height/width in output pixels. ``None`` = full
            canvas in one pass.
        fill_value: Constant fill value for out-of-bounds samples (used
            when ``padding_mode='zeros'`` but offset into ``out``).
    """
    if isinstance(moving, ImageDetail):
        mov_tensor = moving.image
    else:
        mov_tensor = moving
    mov_tensor = mov_tensor.to(dtype=torch.float32)

    M_mov_from_ref = mat_inv(np.asarray(matrix_ref_from_mov, dtype=np.float64))

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    device = mov_tensor.device
    dtype = mov_tensor.dtype

    B, C = int(mov_tensor.shape[0]), int(mov_tensor.shape[1])
    H_in, W_in = int(mov_tensor.shape[-2]), int(mov_tensor.shape[-1])

    out = torch.full(
        (B, C, out_h, out_w), float(fill_value), dtype=dtype, device=device
    )
    M = torch.as_tensor(M_mov_from_ref, dtype=dtype, device=device)

    tile = int(tile_size) if tile_size is not None else max(out_h, out_w)

    for y0 in range(0, out_h, tile):
        for x0 in range(0, out_w, tile):
            y1 = min(out_h, y0 + tile)
            x1 = min(out_w, x0 + tile)

            ys = torch.arange(y0, y1, device=device, dtype=dtype)
            xs = torch.arange(x0, x1, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            ones = torch.ones_like(grid_x)
            pts = torch.stack([grid_x, grid_y, ones], dim=-1)
            mov_pts = pts @ M.T
            x_mov = mov_pts[..., 0]
            y_mov = mov_pts[..., 1]

            # Normalize to [-1, 1] for grid_sample with align_corners=False.
            # align_corners=False maps the [-1, 1] range to [-0.5, W-0.5],
            # so pixel x maps to: norm_x = (2*x + 1) / W - 1.
            x_norm = (2.0 * x_mov + 1.0) / float(W_in) - 1.0
            y_norm = (2.0 * y_mov + 1.0) / float(H_in) - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            tile_out = F.grid_sample(
                mov_tensor,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
            out[..., y0:y1, x0:x1] = tile_out

    return build_image_td(out)
