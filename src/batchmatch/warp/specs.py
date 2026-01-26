from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from batchmatch.base.tensordicts import ImageDetail, WarpParams

Tensor = torch.Tensor

_CACHE_MAX_SIZE = 16


@lru_cache(maxsize=_CACHE_MAX_SIZE)
def _compute_ones_cached(
    H: int, W: int, device_type: str, device_index: Optional[int], dtype: torch.dtype
) -> Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    return torch.ones((1, 1, H, W), device=device, dtype=dtype)


@lru_cache(maxsize=_CACHE_MAX_SIZE)
def _compute_fill_cached(
    fill_key: tuple,
    channels: int,
    device_type: str,
    device_index: Optional[int],
    dtype: torch.dtype,
) -> Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)

    if fill_key[0] == "seq":
        fill_value = fill_key[1]
        base = torch.tensor(fill_value, device=device, dtype=dtype).view(1, channels, 1, 1)
    else:
        fill_value = fill_key[1]
        base = torch.tensor(float(fill_value), device=device, dtype=dtype).view(1, 1, 1, 1)
    return base


def _get_ones_base(H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return _compute_ones_cached(H, W, device.type, device.index, dtype)


def _fill_value_key(fill_value: Optional[float | Sequence[float]]) -> tuple:
    if fill_value is None:
        return ("none",)
    if isinstance(fill_value, (tuple, list)):
        return ("seq", tuple(float(v) for v in fill_value))
    return ("scalar", float(fill_value))


def _fill_is_zero(fill_value: Optional[float | Sequence[float]]) -> bool:
    if fill_value is None:
        return True
    if isinstance(fill_value, (tuple, list)):
        return all(float(v) == 0.0 for v in fill_value)
    return float(fill_value) == 0.0


def _prepare_fill_base(
    fill_value: Optional[float | Sequence[float]],
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if fill_value is None:
        raise ValueError("fill_value must be set when requesting fill base.")
    if isinstance(fill_value, (tuple, list)):
        if len(fill_value) != channels:
            raise ValueError(
                f"fill_value length {len(fill_value)} does not match channels {channels}."
            )

    fill_key = _fill_value_key(fill_value)
    return _compute_fill_cached(fill_key, channels, device.type, device.index, dtype)


def warp_params_to_image_detail(params: WarpParams, td: "ImageDetail") -> None:
    td.set(ImageDetail.Keys.WARP.ROOT, params)


def warp_params_from_image_detail(td: "ImageDetail") -> WarpParams:
    if ImageDetail.Keys.WARP.ROOT in td.keys():
        warp_td = td.get(ImageDetail.Keys.WARP.ROOT)
        if isinstance(warp_td, WarpParams):
            return warp_td
        if isinstance(warp_td, TensorDict):
            return WarpParams(dict(warp_td.items()), batch_size=warp_td.batch_size)

    params = WarpParams.from_components(
        angle=td.get(ImageDetail.Keys.WARP.ANGLE, default=None),
        scale_x=td.get(ImageDetail.Keys.WARP.SCALE_X, default=None),
        scale_y=td.get(ImageDetail.Keys.WARP.SCALE_Y, default=None),
        shear_x=td.get(ImageDetail.Keys.WARP.SHEAR_X, default=None),
        shear_y=td.get(ImageDetail.Keys.WARP.SHEAR_Y, default=None),
        tx=td.get(ImageDetail.Keys.WARP.TX, default=None),
        ty=td.get(ImageDetail.Keys.WARP.TY, default=None),
    )
    
    cx = td.get(ImageDetail.Keys.WARP.CX, default=None)
    if cx is not None:
        params.set(WarpParams.Keys.CENTER_X, cx)
    
    cy = td.get(ImageDetail.Keys.WARP.CY, default=None)
    if cy is not None:
        params.set(WarpParams.Keys.CENTER_Y, cy)
        
    return params


def compute_warp_matrices(
    params: WarpParams,
    H: int,
    W: int,
    *,
    align_corners: bool = False,
    out_fwd: Optional[Tensor] = None,
    out_inv: Optional[Tensor] = None,
    inplace: bool = True,
) -> Tuple[Tensor, Tensor]:
    angle = params[WarpParams.Keys.ANGLE]
    scale_x = params[WarpParams.Keys.SCALE_X]
    scale_y = params[WarpParams.Keys.SCALE_Y]
    shear_x = params[WarpParams.Keys.SHEAR_X]
    shear_y = params[WarpParams.Keys.SHEAR_Y]
    tx = params[WarpParams.Keys.TX]
    ty = params[WarpParams.Keys.TY]

    B = angle.shape[0]
    device = angle.device
    dtype = angle.dtype

    angle_rad = torch.deg2rad(angle)
    tan_shx = torch.tan(torch.deg2rad(shear_x))
    tan_shy = torch.tan(torch.deg2rad(shear_y))
    cos_th = torch.cos(angle_rad)
    sin_th = torch.sin(angle_rad)

    cx = (W - 1.0) * 0.5
    cy = (H - 1.0) * 0.5

    cx_vals = params.get(WarpParams.Keys.CENTER_X, default=None)
    if cx_vals is None or cx_vals.shape != (B,) or cx_vals.device != device or cx_vals.dtype != dtype:
        cx_vals = torch.full((B,), cx, device=device, dtype=dtype)

    cy_vals = params.get(WarpParams.Keys.CENTER_Y, default=None)
    if cy_vals is None or cy_vals.shape != (B,) or cy_vals.device != device or cy_vals.dtype != dtype:
        cy_vals = torch.full((B,), cy, device=device, dtype=dtype)

    tan_xy = tan_shy * tan_shx
    tmp = tan_xy + 1.0

    a00 = scale_x * (cos_th - sin_th * tan_shy)
    a10 = scale_x * (sin_th + cos_th * tan_shy)
    a01 = scale_y * (cos_th * tan_shx - sin_th * tmp)
    a11 = scale_y * (sin_th * tan_shx + cos_th * tmp)

    t0 = cx_vals - (a00 * cx_vals + a01 * cy_vals) + tx
    t1 = cy_vals - (a10 * cx_vals + a11 * cy_vals) + ty

    if not inplace:
        zeros = torch.zeros_like(a00)
        ones = torch.ones_like(a00)
        M_fwd = torch.stack(
            [
                torch.stack([a00, a01, t0], dim=-1),
                torch.stack([a10, a11, t1], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=1,
        )
    else:
        if out_fwd is None:
            out_fwd = torch.zeros((B, 3, 3), device=device, dtype=dtype)
            out_fwd[:, 2, 2] = 1.0

        M_fwd = out_fwd
        M_fwd[:, 0, 0] = a00
        M_fwd[:, 0, 1] = a01
        M_fwd[:, 0, 2] = t0
        M_fwd[:, 1, 0] = a10
        M_fwd[:, 1, 1] = a11
        M_fwd[:, 1, 2] = t1
        M_fwd[:, 2, 0] = 0
        M_fwd[:, 2, 1] = 0

    det = a00 * a11 - a01 * a10

    det_threshold = 1e-6
    if device.type == "cpu" and not torch.compiler.is_compiling():
        near_singular = det.abs() < det_threshold
        near_singular_count = int(near_singular.sum().item())
        if near_singular_count > 0:
            warnings.warn(
                "Near-singular transform detected "
                f"({near_singular_count} of {B} samples have |det| < {det_threshold:g}).",
                RuntimeWarning,
                stacklevel=2,
            )

    det = det.clamp_min(torch.finfo(det.dtype).tiny)

    inv00, inv01 = a11 / det, -a01 / det
    inv10, inv11 = -a10 / det, a00 / det
    inv_t0 = -(inv00 * t0 + inv01 * t1)
    inv_t1 = -(inv10 * t0 + inv11 * t1)

    if not inplace:
        zeros = torch.zeros_like(a00)
        ones = torch.ones_like(a00)
        M_inv = torch.stack(
            [
                torch.stack([inv00, inv01, inv_t0], dim=-1),
                torch.stack([inv10, inv11, inv_t1], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=1,
        )
    else:
        if out_inv is None:
            out_inv = torch.zeros((B, 3, 3), device=device, dtype=dtype)
            out_inv[:, 2, 2] = 1.0

        M_inv = out_inv
        M_inv[:, 0, 0] = inv00
        M_inv[:, 0, 1] = inv01
        M_inv[:, 0, 2] = inv_t0
        M_inv[:, 1, 0] = inv10
        M_inv[:, 1, 1] = inv11
        M_inv[:, 1, 2] = inv_t1
        M_inv[:, 2, 0] = 0
        M_inv[:, 2, 1] = 0

    return M_fwd, M_inv


def compute_warp_grid(
    matrix: Tensor,
    H: int,
    W: int,
    *,
    align_corners: bool = False,
    pixel_coords: Optional[Tuple[Tensor, Tensor]] = None,
    out: Optional[Tensor] = None,
    inplace: bool = True,
) -> Tensor:
    B = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    if align_corners:
        denom_x = float(W - 1) if W > 1 else 1.0
        denom_y = float(H - 1) if H > 1 else 1.0
        sx = 2.0 / denom_x
        sy = 2.0 / denom_y
        bx = -1.0
        by = -1.0
    else:
        denom_x = float(W)
        denom_y = float(H)
        sx = 2.0 / denom_x
        sy = 2.0 / denom_y
        bx = (1.0 / denom_x) - 1.0
        by = (1.0 / denom_y) - 1.0

    if pixel_coords is not None:
        xs, ys = pixel_coords
    else:
        xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
        ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)

    m00 = matrix[:, 0, 0].view(B, 1, 1)
    m01 = matrix[:, 0, 1].view(B, 1, 1)
    m02 = matrix[:, 0, 2].view(B, 1, 1)
    m10 = matrix[:, 1, 0].view(B, 1, 1)
    m11 = matrix[:, 1, 1].view(B, 1, 1)
    m12 = matrix[:, 1, 2].view(B, 1, 1)

    grid_x = xs * (m00 * sx) + ys * (m01 * sx) + (m02 * sx + bx)
    grid_y = xs * (m10 * sy) + ys * (m11 * sy) + (m12 * sy + by)

    if out is not None:
        if out.shape != (B, H, W, 2):
            out = out.view(B, H, W, 2)
        torch.stack([grid_x, grid_y], dim=-1, out=out)
        return out
    else:
        return torch.stack([grid_x, grid_y], dim=-1)



def apply_warp_grid(
    images: Tensor,
    grid: Tensor,
    *,
    mode: str = "bilinear",
    fill_value: Optional[float | Sequence[float]] = 0.0,
    align_corners: bool = False,
    inplace: bool = False,
    sample_out: Optional[Tensor] = None,
) -> Tensor:
    if isinstance(fill_value, (tuple, list)) and len(fill_value) != images.shape[1]:
        raise ValueError(
            f"fill_value length {len(fill_value)} does not match channels {images.shape[1]}."
        )
    use_fill = not _fill_is_zero(fill_value)
    if not use_fill:
        return F.grid_sample(
            images,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=align_corners,
        )

    sample: Tensor
    if inplace and sample_out is not None:
        expected_shape = (images.shape[0], images.shape[1] + 1, images.shape[2], images.shape[3])
        if (
            sample_out.shape == expected_shape
            and sample_out.device == images.device
            and sample_out.dtype == images.dtype
            and sample_out.is_contiguous()
        ):
            sample_out[:, :-1].copy_(images)
            sample_out[:, -1:].fill_(1.0)
            sample = sample_out
        else:
            sample = torch.cat([images, _get_ones_base(images.shape[2], images.shape[3], images.device, images.dtype)
                                .expand(images.shape[0], -1, -1, -1)], dim=1)
    else:
        ones_base = _get_ones_base(images.shape[2], images.shape[3], images.device, images.dtype)
        mask = ones_base.expand(images.shape[0], -1, -1, -1)
        sample = torch.cat([images, mask], dim=1)
    warped = F.grid_sample(sample, grid, mode=mode, padding_mode="zeros", align_corners=align_corners)

    mask = warped[:, -1:, :, :]
    warped = warped[:, :-1, :, :]
    fill_base = _prepare_fill_base(fill_value, warped.shape[1], warped.device, warped.dtype)
    fill = fill_base.expand(warped.shape[0], warped.shape[1], warped.shape[2], warped.shape[3])

    if inplace:
        warped.mul_(mask)
        mask = mask.neg_().add_(1.0)
        warped.addcmul_(fill, mask)
        return warped

    return warped * mask + fill * (1.0 - mask)


def warp_points(pts: Tensor, matrix: Tensor) -> Tensor:
    if pts.ndim != 3 or pts.shape[-1] != 2:
        raise ValueError(f"warp_points expects (B,N,2), got {tuple(pts.shape)}.")

    if pts.device != matrix.device or pts.dtype != matrix.dtype:
        pts = pts.to(device=matrix.device, dtype=matrix.dtype)

    ones = torch.ones_like(pts[..., :1])
    pts_h = torch.cat([pts, ones], dim=-1)
    warped = torch.bmm(pts_h, matrix.transpose(1, 2))[..., :2]

    return warped
