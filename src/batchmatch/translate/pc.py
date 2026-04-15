from __future__ import annotations

import torch

from .utility import _fft2d, _fftshift2d, _ifft2d, _irfft2d, _rfft2d

Tensor = torch.Tensor


def _phase_correlation_surface(
    ref: Tensor,
    mov: Tensor,
    *,
    p : float = 1.0,
    q: float = 1.0,
    eps: float = 1e-8,
    F_ref: Tensor | None = None,
    F_mov_out: Tensor | None = None,
    cross_power_out: Tensor | None = None,
    surface_out: Tensor | None = None,
) -> Tensor:
    """
    Compute phase correlation surface using full FFT.
    """
    if F_ref is None:
        F_ref = _fft2d(ref)

    if F_mov_out is not None:
        torch.fft.fft2(mov, dim=(-2, -1), out=F_mov_out)
        F_mov = F_mov_out
    else:
        F_mov = _fft2d(mov)

    if cross_power_out is not None:
        torch.mul(F_ref, torch.conj(F_mov), out=cross_power_out)
        cross_power_spectrum = cross_power_out
    else:
        cross_power_spectrum = F_ref * torch.conj(F_mov)


    #TODO(wlr): Fix p,q centering to match dissertation values (for overlap normalization)
    #Currently there is no scale invariance (e.g. p=q=0 is unnormalied cross-correlation)
    
    if p != 1.0 or q != 1.0:
        M_mov = torch.abs(F_mov).pow(q)
        M_ref = torch.abs(F_ref).pow(p)
        M = M_mov * M_ref
    else:
        M = torch.abs(cross_power_spectrum)

    normalized_cps = cross_power_spectrum / torch.clamp(M, min=eps)
    pc = _ifft2d(normalized_cps)
    pc = _fftshift2d(pc).real
    result = pc.sum(dim=1)
    if surface_out is not None:
        surface_out.copy_(result)
        return surface_out
    return result


def _rphase_correlation_surface(
    ref: Tensor,
    mov: Tensor,
    *,
    p: float = 1.0,
    q: float = 1.0,
    eps: float = 1e-8,
    F_ref: Tensor | None = None,
    F_mov_out: Tensor | None = None,
    cross_power_out: Tensor | None = None,
    surface_out: Tensor | None = None,
) -> Tensor:
    """
    Compute phase correlation surface using real FFT.
    """
    if F_ref is None:
        F_ref = _rfft2d(ref)

    if F_mov_out is not None:
        torch.fft.rfft2(mov, dim=(-2, -1), out=F_mov_out)
        F_mov = F_mov_out
    else:
        F_mov = _rfft2d(mov)

    #TODO(wlr): Fix p,q centering to match dissertation values (for overlap normalization)
    #Currently there is no scale invariance (e.g. p=q=0 is unnormalied cross-correlation)

    if cross_power_out is not None:
        torch.mul(F_ref, torch.conj(F_mov), out=cross_power_out)
        cross_power_spectrum = cross_power_out
    else:
        cross_power_spectrum = F_ref * torch.conj(F_mov)
    magnitude = torch.abs(cross_power_spectrum)
    normalized_cps = cross_power_spectrum / torch.clamp(magnitude, min=eps)
    pc = _irfft2d(normalized_cps, s=ref.shape[-2:])
    pc = _fftshift2d(pc).real
    result = pc.sum(dim=1)
    if surface_out is not None:
        surface_out.copy_(result)
        return surface_out
    return result


def _rphase_correlation_surface_unshifted(
    ref: Tensor,
    mov: Tensor,
    *,
    p: float = 1.0,
    q: float = 1.0,
    eps: float = 1e-8,
    F_ref: Tensor | None = None,
    F_mov_out: Tensor | None = None,
    cross_power_out: Tensor | None = None,
    surface_out: Tensor | None = None,
) -> Tensor:
    """
    Compute phase correlation surface WITHOUT fftshift (prevents allocation).
    """
    if F_ref is None:
        F_ref = _rfft2d(ref)

    if F_mov_out is not None:
        torch.fft.rfft2(mov, dim=(-2, -1), out=F_mov_out)
        F_mov = F_mov_out
    else:
        F_mov = _rfft2d(mov)

    #TODO(wlr): Fix p,q centering to match dissertation values (for overlap normalization)
    #Currently there is no scale invariance (e.g. p=q=0 is unnormalied cross-correlation)

    if cross_power_out is not None:
        torch.mul(F_ref, torch.conj(F_mov), out=cross_power_out)
        cross_power_spectrum = cross_power_out
    else:
        cross_power_spectrum = F_ref * torch.conj(F_mov)

    magnitude = torch.abs(cross_power_spectrum)
    normalized_cps = cross_power_spectrum / torch.clamp(magnitude, min=eps)
    pc = _irfft2d(normalized_cps, s=ref.shape[-2:])

    result = pc.real.sum(dim=1)

    if surface_out is not None:
        surface_out.copy_(result)
        return surface_out
    return result


def _phase_correlation_surface_unshifted(
    ref: Tensor,
    mov: Tensor,
    *,
    eps: float = 1e-8,
    p: float = 1.0,
    q: float = 1.0,
    F_ref: Tensor | None = None,
    F_mov_out: Tensor | None = None,
    cross_power_out: Tensor | None = None,
    surface_out: Tensor | None = None,
) -> Tensor:
    """
    Compute phase correlation surface WITHOUT fftshift (prevents allocation).
    """
    if F_ref is None:
        F_ref = _fft2d(ref)

    if F_mov_out is not None:
        torch.fft.fft2(mov, dim=(-2, -1), out=F_mov_out)
        F_mov = F_mov_out
    else:
        F_mov = _fft2d(mov)

    #TODO(wlr): Fix p,q centering to match dissertation values (for overlap normalization)
    #Currently there is no scale invariance (e.g. p=q=0 is unnormalied cross-correlation)

    if cross_power_out is not None:
        torch.mul(F_ref, torch.conj(F_mov), out=cross_power_out)
        cross_power_spectrum = cross_power_out
    else:
        cross_power_spectrum = F_ref * torch.conj(F_mov)

    magnitude = torch.abs(cross_power_spectrum)
    normalized_cps = cross_power_spectrum / torch.clamp(magnitude, min=eps)
    pc = _ifft2d(normalized_cps)

    result = pc.real.sum(dim=1)

    if surface_out is not None:
        surface_out.copy_(result)
        return surface_out
    return result
