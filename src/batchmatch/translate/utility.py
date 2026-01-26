from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import torch

warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
    category=UserWarning,
)

Tensor = torch.Tensor


def _fft2d(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        return torch.fft.fft2(x, dim=(-2, -1))
    torch.fft.fft2(x, dim=(-2, -1), out=out)
    return out


def _rfft2d(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        return torch.fft.rfft2(x, dim=(-2, -1))
    torch.fft.rfft2(x, dim=(-2, -1), out=out)
    return out


def _ifft2d(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        return torch.fft.ifft2(x, dim=(-2, -1))
    torch.fft.ifft2(x, dim=(-2, -1), out=out)
    return out


def _irfft2d(x: Tensor, *, s: Optional[Tuple[int, int]] = None, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        return torch.fft.irfft2(x, s=s, dim=(-2, -1))
    torch.fft.irfft2(x, s=s, dim=(-2, -1), out=out)
    return out


def _fftshift2d(x: Tensor) -> Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))


def _corr(F_left: Tensor, F_right: Tensor) -> Tensor:
    cc = _ifft2d(F_left * torch.conj(F_right))
    cc = _fftshift2d(cc).real
    return cc.sum(dim=1)


def _rcorr(
    F_left: Tensor,
    F_right: Tensor,
    s: Optional[Tuple[int, int]] = None,
) -> Tensor:
    cc = _irfft2d(F_left * torch.conj(F_right), s=s)
    cc = _fftshift2d(cc).real
    return cc.sum(dim=1)


def _cross_correlation_surface(ref: Tensor, mov: Tensor) -> Tensor:
    F_ref = _fft2d(ref)
    F_mov = _fft2d(mov)
    return _corr(F_ref, F_mov)


def _rcross_correlation_surface(
    ref: Tensor,
    mov: Tensor,
    F_ref: Optional[Tensor] = None,
    F_mov_out: Optional[Tensor] = None,
) -> Tensor:
    if F_ref is None:
        F_ref = _rfft2d(ref)
    F_mov = _rfft2d(mov, out=F_mov_out)
    return _rcorr(F_ref, F_mov, s=ref.shape[-2:])


def _overlap_area(
    ref_mask: Tensor, 
    mov_mask: Tensor,
    F_ref_mask: Optional[Tensor] = None
) -> Tensor:
    if F_ref_mask is None:
        F_ref_mask = _rfft2d(ref_mask)
    F_mov_mask = _rfft2d(mov_mask)
    area_cc = _irfft2d(F_ref_mask * torch.conj(F_mov_mask), s=ref_mask.shape[-2:])
    area_cc = _fftshift2d(area_cc).real
    area = area_cc.sum(dim=1)
    return area


def _normalize_by_overlap(
    cc: Tensor,
    ref_mask: Tensor,
    mov_mask: Tensor,
    *,
    min_area: Union[float, Tensor] = 1.0,
    F_ref_mask: Optional[Tensor] = None
) -> Tensor:
    area = _overlap_area(ref_mask, mov_mask, F_ref_mask=F_ref_mask)

    if isinstance(min_area, torch.Tensor):
        min_area_tensor = min_area.to(dtype=area.dtype, device=area.device)
    else:
        min_area_tensor = area.new_tensor(min_area)

    safe_area = torch.where(area > 0, area, torch.ones_like(area))
    safe_area = torch.where(
        min_area_tensor > 0,
        torch.maximum(safe_area, min_area_tensor),
        safe_area,
    )
    cc = cc / safe_area
    mask = (area < min_area_tensor) | (area <= 0)
    cc = cc.masked_fill(mask, 0.0)
    return cc


def _fractional_overlap_threshold(mask: Tensor, fraction: float) -> Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected mask with rank 4, got {tuple(mask.shape)}")
    if fraction <= 0:
        return mask.new_zeros((mask.shape[0], 1, 1))

    mask_values = mask if mask.dtype.is_floating_point else mask.to(torch.float32)
    totals = mask_values.reshape(mask_values.shape[0], -1).sum(dim=1)
    threshold = totals * mask_values.new_tensor(fraction)
    return threshold.reshape(mask_values.shape[0], 1, 1)


def _extract_peak_from_surface(surface: Tensor, H: int, W: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, _, _ = surface.shape
    flat = surface.reshape(B, -1)
    scores, idx = flat.max(dim=1)
    mins = flat.min(dim=1).values
    peak_y = idx // W
    peak_x = idx % W
    center_y = (H // 2)
    center_x = (W // 2)

    constant = scores == mins
    center_idx = int(center_y * W + center_x)
    idx = torch.where(constant, torch.as_tensor(center_idx, device=idx.device, dtype=idx.dtype), idx)
    peak_y = idx // W
    peak_x = idx % W

    ty = peak_y.to(torch.float32) - float(center_y)
    tx = peak_x.to(torch.float32) - float(center_x)
    return ty, tx, scores, peak_y, peak_x


def _extract_peak_from_unshifted_surface(
    surface: Tensor, H: int, W: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, _, _ = surface.shape
    flat = surface.reshape(B, -1)
    scores, idx = flat.max(dim=1)
    mins = flat.min(dim=1).values
    peak_y = idx // W
    peak_x = idx % W

    constant = scores == mins
    idx = torch.where(
        constant,
        torch.zeros_like(idx),
        idx,
    )
    peak_y = idx // W
    peak_x = idx % W

    peak_y_f = peak_y.to(torch.float32)
    peak_x_f = peak_x.to(torch.float32)

    half_H = (H + 1) // 2
    half_W = (W + 1) // 2

    ty = torch.where(peak_y >= half_H, peak_y_f - float(H), peak_y_f)
    tx = torch.where(peak_x >= half_W, peak_x_f - float(W), peak_x_f)

    return ty, tx, scores, peak_y, peak_x


def _softmax_surface(surface: Tensor, *, dim: int = -1) -> Tensor:
    max_vals, _ = surface.max(dim=dim, keepdim=True)
    exp_surface = torch.exp(surface - max_vals)
    sum_exp = exp_surface.sum(dim=dim, keepdim=True)
    softmaxed = exp_surface / sum_exp
    return softmaxed


def _extract_softmax_peak_from_surface(surface: Tensor, H: int, W: int) -> Tuple[Tensor, Tensor, Tensor]:
    B, _, _ = surface.shape
    softmaxed = _softmax_surface(surface, dim=(-2, -1))
    grid_y = torch.arange(H, device=surface.device, dtype=surface.dtype).view(1, H, 1)
    grid_x = torch.arange(W, device=surface.device, dtype=surface.dtype).view(1, 1, W)
    exp_y = (softmaxed * grid_y).sum(dim=(-2, -1))
    exp_x = (softmaxed * grid_x).sum(dim=(-2, -1))
    center_y = (H // 2)
    center_x = (W // 2)
    ty = exp_y - float(center_y)
    tx = exp_x - float(center_x)
    scores = (softmaxed * surface).sum(dim=(-2, -1))
    return ty, tx, scores
