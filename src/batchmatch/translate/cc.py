from __future__ import annotations

import torch

from batchmatch.helpers.tensor import expand_mask_to_image
from .utility import _rcorr, _rfft2d

Tensor = torch.Tensor


def _masked_ncc_surface(
    ref: Tensor,
    mov: Tensor,
    ref_mask: Tensor,
    mov_mask: Tensor,
    *,
    eps: float = 1e-6,
    min_count: float | Tensor = 0.0,
    F_A: Tensor | None = None,
    F_Rm: Tensor | None = None,
    F_Rm2: Tensor | None = None,
    F_B_out: Tensor | None = None,
    F_Im_out: Tensor | None = None,
    F_Im2_out: Tensor | None = None,
    surface_out: Tensor | None = None,
) -> Tensor:
    """
    Compute masked normalized cross-correlation surface.

    All inputs must be BCHW tensors.

    Args:
        ref: Reference image [B, C, H, W].
        mov: Moving image [B, C, H, W].
        ref_mask: Reference mask [B, 1, H, W] or [B, C, H, W].
        mov_mask: Moving mask [B, 1, H, W] or [B, C, H, W].
        eps: Small value for numerical stability.
        min_count: Minimum overlap count threshold.
        F_A: Precomputed FFT of ref_mask.
        F_Rm: Precomputed FFT of (ref * ref_mask).
        F_Rm2: Precomputed FFT of ref^2 * ref_mask.
        F_B_out: Optional buffer for moving mask FFT [B, C, H, W//2+1].
        F_Im_out: Optional buffer for moving masked image FFT [B, C, H, W//2+1].
        F_Im2_out: Optional buffer for moving masked image^2 FFT [B, C, H, W//2+1].
        surface_out: Optional buffer for output surface [B, H, W].

    Returns:
        NCC surface [B, H, W].
    """

    ref_mask = expand_mask_to_image(ref_mask, ref)
    mov_mask = expand_mask_to_image(mov_mask, mov)

    # Drop cached FFTs if their channel count no longer matches
    if F_A is not None and F_A.shape[1] != ref_mask.shape[1]:
        F_A = None
    if F_Rm is not None and F_Rm.shape[1] != ref.shape[1]:
        F_Rm = None
    if F_Rm2 is not None and F_Rm2.shape[1] != ref.shape[1]:
        F_Rm2 = None

    if F_A is None or F_Rm is None or F_Rm2 is None:
        Rm = ref * ref_mask
        Rm2 = ref * ref * ref_mask
        
        if F_A is None:
            F_A = _rfft2d(ref_mask)
        if F_Rm is None:
            F_Rm = _rfft2d(Rm)
        if F_Rm2 is None:
            F_Rm2 = _rfft2d(Rm2)
            
    # Compute moving components
    Im = mov * mov_mask
    Im2 = mov * mov * mov_mask

    F_B = _rfft2d(mov_mask, out=F_B_out)
    F_Im = _rfft2d(Im, out=F_Im_out)
    F_Im2 = _rfft2d(Im2, out=F_Im2_out)

    s = ref.shape[-2:]
    N = _rcorr(F_A, F_B, s=s)       # overlap count
    S_R = _rcorr(F_Rm, F_B, s=s)    # R
    S_I = _rcorr(F_A, F_Im, s=s)    # I
    S_R2 = _rcorr(F_Rm2, F_B, s=s)  # R^2
    S_I2 = _rcorr(F_A, F_Im2, s=s)  # I^2
    S_RI = _rcorr(F_Rm, F_Im, s=s)  # R*I

    N_raw = N.clone()
    if isinstance(min_count, torch.Tensor):
        min_count_value = min_count.to(dtype=N.dtype, device=N.device)
    else:
        min_count_value = N.new_tensor(min_count)

    N_clamped = torch.clamp(N, min=min_count_value)

    mu_R = S_R / N_clamped
    mu_I = S_I / N_clamped

    var_R = S_R2 / N_clamped - mu_R * mu_R
    var_I = S_I2 / N_clamped - mu_I * mu_I

    var_R = torch.clamp(var_R, min=0.0)
    var_I = torch.clamp(var_I, min=0.0)

    std_R = torch.sqrt(var_R)
    std_I = torch.sqrt(var_I)

    cov_RI = S_RI / N_clamped - mu_R * mu_I

    denom = torch.clamp(std_R * std_I, min=eps)
    ncc = cov_RI / denom

    invalid = (
        (N_raw <= 0)
        | (N_raw < min_count_value)
        | ~torch.isfinite(ncc)
    )

    ncc = ncc.masked_fill(invalid, 0.0)
    if surface_out is not None:
        surface_out.copy_(ncc)
        return surface_out
    return ncc
