from __future__ import annotations

from typing import Optional, Sequence

import math

import torch

from .utility import _fftshift2d, _irfft2d, _rfft2d

Tensor = torch.Tensor


def _normalized_gradient_fields_surface(
    ref_gx: Tensor,
    ref_gy: Tensor,
    ref_window: Tensor,
    mov_gx: Tensor,
    mov_gy: Tensor,
    mov_window: Tensor,
    # Cached components for Reference
    F_ref_gx_2: Optional[Tensor] = None,
    F_ref_gy_2: Optional[Tensor] = None,
    F_ref_cross: Optional[Tensor] = None,
    # Optional buffers for Moving FFTs
    F_mov_gx_2_out: Optional[Tensor] = None,
    F_mov_gy_2_out: Optional[Tensor] = None,
    F_mov_cross_out: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute squared dot-product NGF surface via FFTs.

    All inputs must be BCHW tensors with matching shapes.

    Args:
        ref_gx: Reference x-gradient [B, C, H, W].
        ref_gy: Reference y-gradient [B, C, H, W].
        mov_gx: Moving x-gradient [B, C, H, W].
        mov_gy: Moving y-gradient [B, C, H, W].
        F_ref_gx_2: Precomputed FFT of (ref_gx)^2.
        F_ref_gy_2: Precomputed FFT of (ref_gy)^2.
        F_ref_cross: Precomputed FFT of (ref_gy * ref_gx).
        F_mov_gx_2_out: Optional buffer for moving gx^2 FFT [B, C, H, W//2+1].
        F_mov_gy_2_out: Optional buffer for moving gy^2 FFT [B, C, H, W//2+1].
        F_mov_cross_out: Optional buffer for moving gx*gy FFT [B, C, H, W//2+1].

    Returns:
        NGF surface [B, H, W].
    """
    if F_ref_gx_2 is None or F_ref_gy_2 is None or F_ref_cross is None:
        ref_gx_2 = ref_gx * ref_gx * ref_window
        ref_gy_2 = ref_gy * ref_gy * ref_window
        ref_cross = ref_gy * ref_gx * ref_window

        if F_ref_gx_2 is None:
            F_ref_gx_2 = _rfft2d(ref_gx_2)
        if F_ref_gy_2 is None:
            F_ref_gy_2 = _rfft2d(ref_gy_2)
        if F_ref_cross is None:
            F_ref_cross = _rfft2d(ref_cross)

    #Note: Doing windowing after squaring is important to prevent significant bias
    mov_gx_2 = mov_gx * mov_gx * mov_window
    mov_gy_2 = mov_gy * mov_gy * mov_window
    mov_cross = mov_gy * mov_gx * mov_window

    F_mov_gx_2 = _rfft2d(mov_gx_2, out=F_mov_gx_2_out)
    F_mov_gy_2 = _rfft2d(mov_gy_2, out=F_mov_gy_2_out)
    F_mov_cross = _rfft2d(mov_cross, out=F_mov_cross_out)

    F_ngf = (
        (F_ref_gy_2 * torch.conj(F_mov_gy_2))
        + (F_ref_gx_2 * torch.conj(F_mov_gx_2))
        + 2.0 * (F_ref_cross * torch.conj(F_mov_cross))
    )

    cc = _irfft2d(F_ngf, s=ref_gx.shape[-2:])
    cc = _fftshift2d(cc).real
    return cc.sum(dim=1)


def _generalized_ngf_surface(
    ref_gx: Tensor,
    ref_gy: Tensor,
    ref_window: Tensor,
    mov_gx: Tensor,
    mov_gy: Tensor,
    mov_window: Tensor,
    *,
    p: int,
    F_ref_terms: Optional[Sequence[Optional[Tensor]]] = None,
    F_mov_terms_out: Optional[Sequence[Optional[Tensor]]] = None,
) -> Tensor:
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    if F_ref_terms is None:
        ref_terms = [None] * (p + 1)
    else:
        if len(F_ref_terms) != p + 1:
            raise ValueError(
                f"Expected {p + 1} reference FFT terms, got {len(F_ref_terms)}"
            )
        ref_terms = list(F_ref_terms)

    if F_mov_terms_out is not None and len(F_mov_terms_out) != p + 1:
        raise ValueError(
            f"Expected {p + 1} moving FFT buffers, got {len(F_mov_terms_out)}"
        )

    F_sum: Optional[Tensor] = None
    for i in range(p + 1):
        exp_x = p - i
        exp_y = i

        F_ref_term = ref_terms[i]
        if F_ref_term is None:
            ref_term = (ref_gx ** exp_x) * (ref_gy ** exp_y) * ref_window
            F_ref_term = _rfft2d(ref_term)
            ref_terms[i] = F_ref_term

        mov_term = (mov_gx ** exp_x) * (mov_gy ** exp_y) * mov_window
        out = None if F_mov_terms_out is None else F_mov_terms_out[i]
        F_mov_term = _rfft2d(mov_term, out=out)

        coeff = math.comb(p, i)
        term = F_ref_term * torch.conj(F_mov_term)
        if coeff != 1:
            term = term * coeff

        if F_sum is None:
            F_sum = term
        else:
            F_sum = F_sum + term

    cc = _irfft2d(F_sum, s=ref_gx.shape[-2:])
    cc = _fftshift2d(cc).real
    return cc.sum(dim=1)


#TODO(wlr): Implement mean-centered and normalized NGF variants from dissertation
#Currently only per-pixel normalized power NGF is implemented