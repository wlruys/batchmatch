from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from batchmatch.helpers.tensor import (
    to_bchw,
    to_bchw_flexible,
    to_common_device,
    as_bool_mask,
    reduce_batch,
)
from batchmatch.helpers.math import clamp_div
from batchmatch.process.crop import compute_intersection_mask

Tensor = torch.Tensor

__all__ = [
    "mse",
    "mae",
    "cross_correlation",
    "local_cross_correlation",
    "normalized_gradient_fields",
    "soft_mutual_information",
    "soft_nmi",
    "mutual_information",
    "nmi",
    "prepare_metric_mask",
    "combine_detail_masks",
]

def prepare_metric_mask(mask: Optional[Tensor], target: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None

    mask_t = to_bchw_flexible(mask).to(device=target.device)

    if mask_t.shape[-2:] != tuple(target.shape[-2:]):
        raise ValueError(
            f"Mask spatial dims {tuple(mask_t.shape[-2:])} must match "
            f"target dims {tuple(target.shape[-2:])}."
        )

    B = target.shape[0]
    if mask_t.shape[0] not in (1, B):
        raise ValueError(
            f"Mask batch {mask_t.shape[0]} must be 1 or match target batch {B}."
        )
    if mask_t.shape[0] == 1 and B > 1:
        mask_t = mask_t.expand(B, mask_t.shape[1], *mask_t.shape[-2:])

    C = target.shape[1]
    if mask_t.shape[1] == 1:
        mask_t = mask_t.expand(B, C, *mask_t.shape[-2:])
    elif mask_t.shape[1] != C:
        raise ValueError(
            f"Mask channels {mask_t.shape[1]} must be 1 or match target channels {C}."
        )
    return mask_t.to(dtype=target.dtype)


def _mask_like_image_internal(image: Tensor, mask: Optional[Tensor]) -> Tensor:
    B, C, H, W = image.shape
    if mask is None:
        return torch.ones((B, 1, H, W), device=image.device, dtype=image.dtype)

    mask_t = to_bchw_flexible(mask).to(device=image.device)

    if mask_t.shape[-2:] != (H, W):
        raise ValueError(
            f"Mask spatial dims {tuple(mask_t.shape[-2:])} must match image dims {(H, W)}"
        )
    if mask_t.shape[0] not in (1, B):
        raise ValueError(
            f"Mask batch {mask_t.shape[0]} must be 1 or match image batch {B}."
        )
    if mask_t.shape[0] == 1 and B > 1:
        mask_t = mask_t.expand(B, mask_t.shape[1], H, W)

    if mask_t.shape[1] == 1:
        pass
    elif mask_t.shape[1] == C:
        mask_t = mask_t.mean(dim=1, keepdim=True)
    else:
        raise ValueError(
            f"Mask channels {mask_t.shape[1]} must be 1 or match image channels {C}."
        )

    return mask_t.to(dtype=image.dtype)


#TODO(wlr): Replace with utlities from crop module
def combine_detail_masks(
    ref_image: Tensor,
    ref_mask: Optional[Tensor],
    mov_image: Tensor,
    mov_mask: Optional[Tensor],
    *,
    detach: bool = True,
) -> Optional[Tensor]:
    """
    Combine reference and moving masks into a single mask (intersection).
    """
    if ref_mask is None and mov_mask is None:
        return None

    B = ref_image.shape[0]
    H, W = ref_image.shape[-2:]

    ref_m = _mask_like_image_internal(ref_image, ref_mask)
    mov_m = _mask_like_image_internal(mov_image, mov_mask)

    combined_list = []
    for b in range(B):
        masks_at_b = [ref_m[b : b + 1], mov_m[b : b + 1]]
        intersection = compute_intersection_mask(masks_at_b)
        combined_list.append(intersection)

    combined = torch.cat(combined_list, dim=0)
    combined = combined.to(dtype=ref_image.dtype)
    return combined.detach() if detach else combined

def mse(
    x: Tensor,
    y: Tensor,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    x, y = to_common_device(x, y)
    x = to_bchw(x)
    y = to_bchw(y)

    if x.shape != y.shape:
        raise ValueError(f"Shapes must match, got {x.shape} vs {y.shape}")

    diff2 = (x.to(torch.float32) - y.to(torch.float32)).pow(2)
    mask_t = prepare_metric_mask(mask, x) if mask is not None else None

    if mask_t is not None:
        #TODO(wlr): Replace with masked mean helper utility
        weights = mask_t.to(torch.float32)
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_b = (diff2 * weights).sum(dim=(1, 2, 3)) / denom
    else:
        per_b = diff2.mean(dim=(1, 2, 3))

    return reduce_batch(per_b, reduction)


def mae(
    x: Tensor,
    y: Tensor,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    x, y = to_common_device(x, y)
    x = to_bchw(x)
    y = to_bchw(y)

    if x.shape != y.shape:
        raise ValueError(f"Shapes must match, got {x.shape} vs {y.shape}")

    abs_diff = torch.abs(x.to(torch.float32) - y.to(torch.float32))
    mask_t = prepare_metric_mask(mask, x) if mask is not None else None

    if mask_t is not None:
        #TODO(wlr): Replace with masked mean helper utility
        weights = mask_t.to(torch.float32)
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_b = (abs_diff * weights).sum(dim=(1, 2, 3)) / denom
    else:
        per_b = abs_diff.mean(dim=(1, 2, 3))

    return reduce_batch(per_b, reduction)


def cross_correlation(
    x: Tensor,
    y: Tensor,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    x, y = to_common_device(x, y)
    x = to_bchw(x)
    y = to_bchw(y)

    if x.shape != y.shape:
        raise ValueError(f"Shapes must match, got {x.shape} vs {y.shape}")

    B = x.shape[0]
    mask_t = prepare_metric_mask(mask, x) if mask is not None else None

    if mask_t is not None:
        #TODO(wlr): Replace with masked mean helper utility
        weights = mask_t.to(torch.float32)
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        x_mean = (x * weights).sum(dim=(1, 2, 3), keepdim=True) / denom.view(B, 1, 1, 1)
        y_mean = (y * weights).sum(dim=(1, 2, 3), keepdim=True) / denom.view(B, 1, 1, 1)
    else:
        x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        y_mean = y.mean(dim=(1, 2, 3), keepdim=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    if mask_t is not None:
        #TODO(wlr): Replace with masked mean helper utility
        numerator = (weights * x_centered * y_centered).sum(dim=(1, 2, 3))
        denom_x = torch.sqrt((weights * x_centered * x_centered).sum(dim=(1, 2, 3)).clamp_min(0.0))
        denom_y = torch.sqrt((weights * y_centered * y_centered).sum(dim=(1, 2, 3)).clamp_min(0.0))
    else:
        numerator = (x_centered * y_centered).sum(dim=(1, 2, 3))
        denom_x = torch.sqrt((x_centered * x_centered).sum(dim=(1, 2, 3)))
        denom_y = torch.sqrt((y_centered * y_centered).sum(dim=(1, 2, 3)))

    denominator = denom_x * denom_y
    ncc = clamp_div(numerator, denominator)

    return reduce_batch(ncc, reduction)


def local_cross_correlation(
    x: Tensor,
    y: Tensor,
    window_size: int = 9,
    window_type: str = "gaussian",
    sigma: float = 1.5,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    x, y = to_common_device(x, y)
    x = to_bchw(x).to(torch.float32)
    y = to_bchw(y).to(torch.float32)

    if x.shape != y.shape:
        raise ValueError(f"Shapes must match, got {x.shape} vs {y.shape}")

    B, C, H, W = x.shape

    if window_type == "gaussian":
        coords = torch.arange(window_size, dtype=torch.float32, device=x.device)
        coords = coords - (window_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window / window.sum()
    elif window_type == "uniform":
        window = torch.ones(window_size, window_size, device=x.device)
        window = window / (window_size**2)
    else:
        raise ValueError(f"window_type must be 'gaussian' or 'uniform', got {window_type}")

    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    #TODO(wlr): Should probably mask before computing local stats. Currently only valid for interior regions.

    padding = window_size // 2
    x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    y_pad = F.pad(y, (padding, padding, padding, padding), mode="reflect")

    # local means
    mu_x = F.conv2d(x_pad, window, groups=C)
    mu_y = F.conv2d(y_pad, window, groups=C)

    # local centered images
    x_centered = x - mu_x
    y_centered = y - mu_y
    x_c_pad = F.pad(x_centered, (padding, padding, padding, padding), mode="reflect")
    y_c_pad = F.pad(y_centered, (padding, padding, padding, padding), mode="reflect")

    # local variances and covariance
    var_x = F.conv2d(x_c_pad**2, window, groups=C)
    var_y = F.conv2d(y_c_pad**2, window, groups=C)
    cov_xy = F.conv2d(x_c_pad * y_c_pad, window, groups=C)

    eps = 1e-8
    local_ncc = cov_xy / (torch.sqrt(var_x * var_y) + eps)

    mask_t = prepare_metric_mask(mask, local_ncc) if mask is not None else None

    if mask_t is not None:
        #TODO(wlr): Replace with masked mean helper utility
        weights = mask_t.to(torch.float32)
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_b = (local_ncc * weights).sum(dim=(1, 2, 3)) / denom
    else:
        per_b = local_ncc.mean(dim=(1, 2, 3))

    return reduce_batch(per_b, reduction)

def normalized_gradient_fields(
    g: Tuple[Tensor, Tensor],
    h: Tuple[Tensor, Tensor],
    eps: float = 1e-12,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    gx, gy = g
    hx, hy = h

    gx, hx = to_common_device(gx, hx)
    gy, hy = to_common_device(gy, hy)
    gx = to_bchw(gx).to(torch.float32)
    gy = to_bchw(gy).to(torch.float32)
    hx = to_bchw(hx).to(torch.float32)
    hy = to_bchw(hy).to(torch.float32)

    if gx.shape != hx.shape or gy.shape != hy.shape:
        raise ValueError(
            f"Gradient tensors must share shape. Got gx {tuple(gx.shape)}, "
            f"hx {tuple(hx.shape)}, gy {tuple(gy.shape)}, hy {tuple(hy.shape)}"
        )

    mask_t = prepare_metric_mask(mask, gx) if mask is not None else None

    dot = gx * hx + gy * hy
    g_norm = torch.sqrt(torch.clamp(gx * gx + gy * gy, min=eps))
    h_norm = torch.sqrt(torch.clamp(hx * hx + hy * hy, min=eps))
    cos = clamp_div(dot, g_norm * h_norm, eps=eps)
    ngf_map = 1.0 - cos.pow(2)

    if mask_t is not None:
        #TODO(wlr): Replace with masked mean helper utility
        weights = mask_t.to(torch.float32)
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_b = (ngf_map * weights).sum(dim=(1, 2, 3)) / denom
    else:
        per_b = ngf_map.mean(dim=(1, 2, 3))

    return reduce_batch(per_b, reduction)


def soft_mutual_information(
    x: Tensor,
    y: Tensor,
    bins: int = 64,
    bandwidth: float = 0.2,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Differentiable mutual information using soft binning.
    """
    X = to_bchw(x).to(torch.float32)
    Y = to_bchw(y).to(torch.float32)

    if X.shape != Y.shape:
        raise ValueError(f"Shapes must match, got {X.shape} vs {Y.shape}")

    mask_t = prepare_metric_mask(mask, X) if mask is not None else None

    B, C, H, W = X.shape
    X_flat = X.reshape(B, -1)
    Y_flat = Y.reshape(B, -1)

    # Normalize to [0, 1] per batch
    vmin_x = X_flat.min(dim=1, keepdim=True)[0]
    vmax_x = X_flat.max(dim=1, keepdim=True)[0]
    vmin_y = Y_flat.min(dim=1, keepdim=True)[0]
    vmax_y = Y_flat.max(dim=1, keepdim=True)[0]

    rx = (vmax_x - vmin_x).clamp_min(1e-12)
    ry = (vmax_y - vmin_y).clamp_min(1e-12)

    X_norm = (X_flat - vmin_x) / rx
    Y_norm = (Y_flat - vmin_y) / ry

    bin_centers = torch.linspace(0, 1, bins, device=X.device, dtype=torch.float32)

    # Soft assignments
    dist_x = (X_norm.unsqueeze(-1) - bin_centers) ** 2
    dist_y = (Y_norm.unsqueeze(-1) - bin_centers) ** 2

    weights_x = torch.exp(-dist_x / (2 * bandwidth**2))
    weights_y = torch.exp(-dist_y / (2 * bandwidth**2))

    weights_x = weights_x / (weights_x.sum(dim=-1, keepdim=True) + 1e-12)
    weights_y = weights_y / (weights_y.sum(dim=-1, keepdim=True) + 1e-12)

    if mask_t is not None:
        mask_flat = mask_t.reshape(B, -1).unsqueeze(-1)
        weights_x = weights_x * mask_flat
        weights_y = weights_y * mask_flat

    # Joint
    pxy = torch.einsum("bni,bnj->bij", weights_x, weights_y)
    pxy = pxy / (pxy.sum(dim=(1, 2), keepdim=True) + 1e-12)

    # Marginal
    px = pxy.sum(dim=2, keepdim=True)
    py = pxy.sum(dim=1, keepdim=True)

    # I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    log_pxy = torch.log(pxy.clamp_min(1e-30))
    log_px = torch.log(px.clamp_min(1e-30))
    log_py = torch.log(py.clamp_min(1e-30))

    mi = (pxy * (log_pxy - log_px - log_py)).sum(dim=(1, 2))

    return reduce_batch(mi, reduction)


def soft_nmi(
    x: Tensor,
    y: Tensor,
    bins: int = 64,
    bandwidth: float = 0.2,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Differentiable normalized mutual information using soft binning.
    """
    X = to_bchw(x).to(torch.float32)
    Y = to_bchw(y).to(torch.float32)

    if X.shape != Y.shape:
        raise ValueError(f"Shapes must match, got {X.shape} vs {Y.shape}")

    mask_t = prepare_metric_mask(mask, X) if mask is not None else None

    B, C, H, W = X.shape
    X_flat = X.reshape(B, -1)
    Y_flat = Y.reshape(B, -1)

    # Normalize to [0, 1] per batch
    vmin_x = X_flat.min(dim=1, keepdim=True)[0]
    vmax_x = X_flat.max(dim=1, keepdim=True)[0]
    vmin_y = Y_flat.min(dim=1, keepdim=True)[0]
    vmax_y = Y_flat.max(dim=1, keepdim=True)[0]

    rx = (vmax_x - vmin_x).clamp_min(1e-12)
    ry = (vmax_y - vmin_y).clamp_min(1e-12)

    X_norm = (X_flat - vmin_x) / rx
    Y_norm = (Y_flat - vmin_y) / ry

    bin_centers = torch.linspace(0, 1, bins, device=X.device, dtype=torch.float32)

    # Soft assignments
    dist_x = (X_norm.unsqueeze(-1) - bin_centers) ** 2
    dist_y = (Y_norm.unsqueeze(-1) - bin_centers) ** 2

    weights_x = torch.exp(-dist_x / (2 * bandwidth**2))
    weights_y = torch.exp(-dist_y / (2 * bandwidth**2))

    weights_x = weights_x / (weights_x.sum(dim=-1, keepdim=True) + 1e-12)
    weights_y = weights_y / (weights_y.sum(dim=-1, keepdim=True) + 1e-12)

    if mask_t is not None:
        mask_flat = mask_t.reshape(B, -1).unsqueeze(-1)
        weights_x = weights_x * mask_flat
        weights_y = weights_y * mask_flat

    # Joint
    pxy = torch.einsum("bni,bnj->bij", weights_x, weights_y)
    pxy = pxy / (pxy.sum(dim=(1, 2), keepdim=True) + 1e-12)

    # Marginal
    px = pxy.sum(dim=2)
    py = pxy.sum(dim=1)

    # Entropies
    Hx = -(px.clamp_min(1e-30) * px.clamp_min(1e-30).log()).sum(dim=1)
    Hy = -(py.clamp_min(1e-30) * py.clamp_min(1e-30).log()).sum(dim=1)
    Hxy = -(pxy.clamp_min(1e-30) * pxy.clamp_min(1e-30).log()).sum(dim=(1, 2))

    # NMI = (H(X) + H(Y)) / H(X,Y)
    nmi_val = (Hx + Hy) / Hxy.clamp_min(1e-30)

    return reduce_batch(nmi_val, reduction)


def mutual_information(
    x: Tensor,
    y: Tensor,
    bins: int = 256,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute mutual information between two images.
    """
    X = to_bchw(x).to(torch.float32)
    Y = to_bchw(y).to(torch.float32)

    if X.shape != Y.shape:
        raise ValueError(f"Shapes must match, got {X.shape} vs {Y.shape}")

    mask_t = prepare_metric_mask(mask, X) if mask is not None else None

    X = X.flatten(1)
    Y = Y.flatten(1)
    B = X.shape[0]

    vmin_x, _ = X.min(dim=1, keepdim=True)
    vmax_x, _ = X.max(dim=1, keepdim=True)
    vmin_y, _ = Y.min(dim=1, keepdim=True)
    vmax_y, _ = Y.max(dim=1, keepdim=True)

    rx = (vmax_x - vmin_x).clamp_min(1e-12)
    ry = (vmax_y - vmin_y).clamp_min(1e-12)

    qx = ((X - vmin_x) / rx * (bins - 1)).round().clamp(0, bins - 1).to(torch.long)
    qy = ((Y - vmin_y) / ry * (bins - 1)).round().clamp(0, bins - 1).to(torch.long)

    # joint
    batch_ids = torch.arange(B, device=X.device).repeat_interleave(X.shape[1])
    joint_index = (qx * bins + qy).view(-1)
    global_index = batch_ids * (bins * bins) + joint_index

    if mask_t is not None:
        weights = mask_t.view(B, -1).to(torch.float32).reshape(-1)
    else:
        weights = None

    hist = torch.bincount(
        global_index,
        minlength=B * bins * bins,
        weights=weights,
    ).to(torch.float32)

    pxy = hist.view(B, bins, bins)
    pxy = pxy / pxy.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)

    px = pxy.sum(dim=2, keepdim=True)
    py = pxy.sum(dim=1, keepdim=True)

    log = torch.log
    mi = (
        pxy.clamp_min(1e-30)
        * (log(pxy.clamp_min(1e-30)) - log(px.clamp_min(1e-30)) - log(py.clamp_min(1e-30)))
    ).sum(dim=(1, 2))

    return reduce_batch(mi, reduction)


def nmi(
    x: Tensor,
    y: Tensor,
    bins: int = 256,
    reduction: str = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    X = to_bchw(x).to(torch.float32)
    Y = to_bchw(y).to(torch.float32)

    if X.shape != Y.shape:
        raise ValueError(f"Shapes must match, got {X.shape} vs {Y.shape}")

    mask_t = prepare_metric_mask(mask, X) if mask is not None else None

    X = X.flatten(1)
    Y = Y.flatten(1)
    B = X.shape[0]

    vmin_x, _ = X.min(dim=1, keepdim=True)
    vmax_x, _ = X.max(dim=1, keepdim=True)
    vmin_y, _ = Y.min(dim=1, keepdim=True)
    vmax_y, _ = Y.max(dim=1, keepdim=True)

    rx = (vmax_x - vmin_x).clamp_min(1e-12)
    ry = (vmax_y - vmin_y).clamp_min(1e-12)

    qx = ((X - vmin_x) / rx * (bins - 1)).round().clamp(0, bins - 1).to(torch.long)
    qy = ((Y - vmin_y) / ry * (bins - 1)).round().clamp(0, bins - 1).to(torch.long)

    # joint
    batch_ids = torch.arange(B, device=X.device).repeat_interleave(X.shape[1])
    joint_index = (qx * bins + qy).view(-1)
    global_index = batch_ids * (bins * bins) + joint_index

    if mask_t is not None:
        weights = mask_t.view(B, -1).to(torch.float32).reshape(-1)
    else:
        weights = None

    hist = torch.bincount(
        global_index,
        minlength=B * bins * bins,
        weights=weights,
    ).to(torch.float32)

    pxy = hist.view(B, bins, bins)
    pxy = pxy / pxy.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)

    px = pxy.sum(dim=2)
    py = pxy.sum(dim=1)

    # Entropies
    Hx = -(px.clamp_min(1e-30) * px.clamp_min(1e-30).log()).sum(dim=1)
    Hy = -(py.clamp_min(1e-30) * py.clamp_min(1e-30).log()).sum(dim=1)
    Hxy = -(pxy.clamp_min(1e-30) * pxy.clamp_min(1e-30).log()).sum(dim=(1, 2))

    nmi_val = (Hx + Hy) / Hxy.clamp_min(1e-30)

    return reduce_batch(nmi_val, reduction)
