from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import matplotlib.pyplot as plt

from batchmatch.helpers.tensor import to_bchw, to_chw
from batchmatch.helpers.math import quantile

#TODO(wlr): These defaults aren't very good. Add better documentation and tuning later.
#NOTE(wlr): This is just a coarse segmentation to estimate cell sizes. Works "ok enough" on these images. Not very accurate or robust.
#NOTE(wlr): Future improvements could be proper accurate segementation using modern methods. 

@dataclass(frozen=True)
class CellSizeCfg:
    clip_lo: float = 0.02
    clip_hi: float = 0.98
    denoise_sigma: float = 1.0                
    large_bg_sigma: Optional[float] = None

    channel_mode: Literal["mean", "max", "weighted"] = "mean"
    channel_weights: Optional[Sequence[float]] = None

    grad_weight: float = 0.5
    use_log: bool = True
    log_sigmas: Tuple[float, ...] = (2., 3., 4., 6., 8.)
    blob_weight: float = 0.5

    morph_radius_px: int = 3

    radius_hint_px: Optional[float] = None
    peak_min_distance_frac: float = 0.6
    peak_rel_threshold: float = 0.3

    min_radius_px: float = 2.0
    max_radius_px: float = 128.0
    trim_low_q: float = 0.1
    trim_high_q: float = 0.9
    max_dim: Optional[int] = 512
    return_debug: bool = True



def _resize_longest_side(x: Tensor, max_dim: Optional[int], *, is_mask: bool = False) -> Tuple[Tensor, float]:
    if max_dim is None or max_dim <= 0:
        return x, 1.0
    bchw = to_bchw(x)
    _, _, H, W = bchw.shape
    if max(H, W) <= max_dim:
        return x, 1.0
    scale = max_dim / float(max(H, W))
    out_h = max(1, int(round(H * scale)))
    out_w = max(1, int(round(W * scale)))
    mode = "nearest" if is_mask else "bilinear"
    y = F.interpolate(
        bchw.float(),
        size=(out_h, out_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if x.ndim == 2:
        return y[0, 0], scale
    if x.ndim == 3:
        return y[0], scale
    return y, scale

def _gaussian_kernel2d(sigma: float, device, dtype) -> Tensor:
    r = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    ys = xs[:, None]
    g = torch.exp(-(xs**2 + ys**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g[None, None, :, :]

def _laplacian_kernel2d(device, dtype) -> Tensor:
    k = torch.tensor([[0.,  1., 0.],
                      [1., -4., 1.],
                      [0.,  1., 0.]], device=device, dtype=dtype)
    return k[None, None, :, :]

def _binary_opening(mask: Tensor, r: int) -> Tensor:
    if r <= 0:
        return mask
    k = 2 * r + 1
    er = -F.max_pool2d(-mask, k, stride=1, padding=r)
    op =  F.max_pool2d(er,  k, stride=1, padding=r)
    return op


def _binary_closing(mask: Tensor, r: int) -> Tensor:
    if r <= 0:
        return mask
    k = 2 * r + 1
    di =  F.max_pool2d(mask, k, stride=1, padding=r)
    cl = -F.max_pool2d(-di,  k, stride=1, padding=r)
    return cl

def preprocess_to_gray(img: Tensor, cfg: CellSizeCfg) -> Tensor:
    x = to_bchw(img).float()
    B, C, H, W = x.shape

    flat = x.flatten()
    lo = torch.quantile(flat, cfg.clip_lo)
    hi = torch.quantile(flat, cfg.clip_hi)
    x = (x - lo) / (hi - lo).clamp_min(1e-6)
    x = x.clamp_(0.0, 1.0)

    if cfg.denoise_sigma and cfg.denoise_sigma > 0:
        g = _gaussian_kernel2d(cfg.denoise_sigma, x.device, x.dtype)
        x = F.conv2d(x.view(B * C, 1, H, W), g, padding=g.shape[-1] // 2).view(B, C, H, W)

    if cfg.large_bg_sigma is not None and cfg.large_bg_sigma > 0:
        gL = _gaussian_kernel2d(cfg.large_bg_sigma, x.device, x.dtype)
        bg = F.conv2d(x.view(B * C, 1, H, W), gL, padding=gL.shape[-1] // 2).view(B, C, H, W)
        x = x - bg
        flat2 = x.flatten()
        lo2 = quantile(flat2, cfg.clip_lo)
        hi2 = quantile(flat2, cfg.clip_hi)
        x = (x - lo2) / (hi2 - lo2).clamp_min(1e-6)
        x = x.clamp_(0.0, 1.0)

    if C == 1:
        gray = x
    else:
        if cfg.channel_mode == "mean":
            gray = x.mean(dim=1, keepdim=True)
        elif cfg.channel_mode == "max":
            gray = x.max(dim=1, keepdim=True).values
        elif cfg.channel_mode == "weighted":
            if cfg.channel_weights is None or len(cfg.channel_weights) != C:
                raise ValueError(f"channel_weights of length {C} required for weighted mode")
            w = torch.tensor(cfg.channel_weights, device=x.device, dtype=x.dtype).view(1, C, 1, 1)
            gray = (x * w).sum(dim=1, keepdim=True) / (w.sum() + 1e-8)
        else:
            raise ValueError(f"Unknown channel_mode: {cfg.channel_mode}")

    return gray.clamp_(0.0, 1.0)


def sobel_grad_mag(gray: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    device, dtype = gray.device, gray.dtype
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype)[None, None]
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype)[None, None]
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)
    q = torch.quantile(mag.flatten(), 0.99).clamp_min(1e-6)
    mag_norm = (mag / q).clamp_(0.0, 1.0)
    return mag_norm, gx, gy


def foreground_score(gray: Tensor, cfg: CellSizeCfg) -> Tuple[Tensor, Dict[str, Tensor]]:
    I = gray
    G, gx, gy = sobel_grad_mag(gray)
    score_raw = I + cfg.grad_weight * (1.0 - G)

    blob = torch.zeros_like(gray)
    if cfg.use_log and cfg.blob_weight > 0:
        lap = _laplacian_kernel2d(gray.device, gray.dtype)
        for s in cfg.log_sigmas:
            g = _gaussian_kernel2d(float(s), gray.device, gray.dtype)
            blur = F.conv2d(gray, g, padding=g.shape[-1] // 2)
            lapr = F.conv2d(blur, lap, padding=1)
            resp = -(float(s) ** 2) * lapr
            resp.clamp_min_(0.0)
            blob = torch.maximum(blob, resp)

        bq = quantile(blob.flatten(), 0.99)
        if float(bq) > 0:
            blob = (blob / bq).clamp_(0.0, 1.0)
        score_raw = score_raw + cfg.blob_weight * blob

    lo = quantile(score_raw.flatten(), 0.02)
    hi = quantile(score_raw.flatten(), 0.98)
    score = ((score_raw - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0.0, 1.0)

    comps = {
        "I": I,
        "G": G,
        "gx": gx,
        "gy": gy,
        "blob": blob,
        "score_raw": score_raw,
        "score": score,
    }
    return score, comps

def otsu_threshold(img01: Tensor, nbins: int = 256) -> Union[float, Tensor]:
    x = img01.detach().clamp(0.0, 1.0)
    if x.ndim == 4 and x.shape[1] == 1:
        B = x.shape[0]
        thr = torch.empty((B,), device=x.device, dtype=x.dtype)
        edges = torch.linspace(0.0, 1.0, nbins + 1, device=x.device, dtype=x.dtype)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for b in range(B):
            xb = x[b].flatten()
            hist = torch.histc(xb, bins=nbins, min=0.0, max=1.0)
            p = hist / hist.sum().clamp_min(1e-6)
            omega = p.cumsum(0)
            mu = (p * centers).cumsum(0)
            mu_t = mu[-1]
            denom = (omega * (1.0 - omega)).clamp_min(1e-6)
            sigma_b2 = (mu_t * omega - mu) ** 2 / denom
            idx = int(torch.argmax(sigma_b2))
            thr[b] = centers[idx]
        return thr
    x = x.flatten()
    hist = torch.histc(x, bins=nbins, min=0.0, max=1.0)
    edges = torch.linspace(0.0, 1.0, nbins + 1, device=x.device, dtype=x.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / hist.sum().clamp_min(1e-6)
    omega = p.cumsum(0)
    mu = (p * centers).cumsum(0)
    mu_t = mu[-1]
    denom = (omega * (1.0 - omega)).clamp_min(1e-6)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(torch.argmax(sigma_b2))
    return float(centers[idx])

def _shift2d(t: Tensor, dy: int, dx: int, fill: float) -> Tensor:
    *prefix, H, W = t.shape
    pad_top = max(0, dy)
    pad_bottom = max(0, -dy)
    pad_left = max(0, dx)
    pad_right = max(0, -dx)
    tpad = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=fill)
    y0 = pad_bottom
    x0 = pad_right
    return tpad[..., y0:y0 + H, x0:x0 + W]

def edt_torch_jump_flood(mask: Tensor) -> Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W].")

    B, _, H, W = mask.shape
    device = mask.device

    fg = mask > 0.5
    bg = ~fg

    ys = torch.arange(H, device=device, dtype=torch.int32).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=device, dtype=torch.int32).view(1, 1, W).expand(B, H, W)

    seed_y = torch.where(bg[:, 0], ys, torch.full_like(ys, -1))
    seed_x = torch.where(bg[:, 0], xs, torch.full_like(xs, -1))
    valid = bg[:, 0].to(torch.bool)

    inf = torch.tensor(2**30, device=device, dtype=torch.int64)
    dy0 = (seed_y.to(torch.int64) - ys.to(torch.int64))
    dx0 = (seed_x.to(torch.int64) - xs.to(torch.int64))
    best_d2 = torch.where(valid, dy0 * dy0 + dx0 * dx0, inf)
    bg_any = bg.view(B, -1).any(dim=1)

    max_hw = max(H, W)
    step = 1 << (int(math.floor(math.log2(max_hw))) if max_hw > 1 else 0)

    offsets = [(oy, ox) for oy in (-1, 0, 1) for ox in (-1, 0, 1) if not (oy == 0 and ox == 0)]

    while step >= 1:
        for oy, ox in offsets:
            dy = oy * step
            dx = ox * step

            cand_y = _shift2d(seed_y, dy, dx, fill=-1)
            cand_x = _shift2d(seed_x, dy, dx, fill=-1)
            cand_v = _shift2d(valid.to(torch.uint8), dy, dx, fill=0).to(torch.bool)

            cdy = (cand_y.to(torch.int64) - ys.to(torch.int64))
            cdx = (cand_x.to(torch.int64) - xs.to(torch.int64))
            cand_d2 = torch.where(cand_v, cdy * cdy + cdx * cdx, inf)

            better = cand_d2 < best_d2
            best_d2 = torch.where(better, cand_d2, best_d2)
            seed_y = torch.where(better, cand_y, seed_y)
            seed_x = torch.where(better, cand_x, seed_x)
            valid = torch.where(better, cand_v, valid)

        step //= 2

    if not bg_any.all():
        best_d2[~bg_any] = 0
    dist = torch.sqrt(best_d2.clamp_max(int(2**31 - 1)).to(torch.float32))
    dist = torch.where(bg[:, 0], torch.zeros_like(dist), dist)
    return dist[:, None, :, :]


def distance_peaks(dist: Tensor, cfg: CellSizeCfg) -> Tensor:
    r_hint = cfg.radius_hint_px if cfg.radius_hint_px is not None else 5.0
    min_dist = max(3, int(round(cfg.peak_min_distance_frac * r_hint)))
    k = 2 * min_dist + 1

    pooled = F.max_pool2d(dist, kernel_size=k, stride=1, padding=min_dist)
    ismax = (dist == pooled)

    dmax = dist.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    ismax = ismax & (dist >= cfg.peak_rel_threshold * dmax)

    border = 2
    ismax[..., :border, :] = False
    ismax[..., -border:, :] = False
    ismax[..., :, :border] = False
    ismax[..., :, -border:] = False
    return ismax

def _trimmed_indices(n: int, q_lo: float, q_hi: float) -> Tuple[int, int]:
    if n <= 1:
        return 0, 0
    lo = int(q_lo * (n - 1))
    hi = int(q_hi * (n - 1))
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))
    if hi < lo:
        hi = lo
    return lo, hi

def robust_trimmed_mean(radii: Tensor, cfg: CellSizeCfg) -> Tuple[float, Tensor, Tensor]:
    r = radii[(radii >= cfg.min_radius_px) & (radii <= cfg.max_radius_px)]
    if r.numel() == 0:
        raise RuntimeError("No radii left after min/max filtering.")
    r_sorted, _ = torch.sort(r)
    n = r_sorted.numel()
    lo, hi = _trimmed_indices(n, cfg.trim_low_q, cfg.trim_high_q)
    r_trim = r_sorted[lo:hi + 1]
    return float(r_trim.mean().item()), r_sorted, r_trim

@torch.inference_mode()
def estimate_cell_radius_px(
    img: Tensor,
    cfg: CellSizeCfg = CellSizeCfg(),
) -> Dict[str, Union[float, Tensor, Dict[str, Tensor]]]:
    img_small, scale = _resize_longest_side(img, cfg.max_dim, is_mask=False)

    cfg_eff = cfg
    if scale != 1.0:
        rh = None if cfg.radius_hint_px is None else cfg.radius_hint_px * scale
        cfg_eff = replace(
            cfg,
            radius_hint_px=rh,
            min_radius_px=cfg.min_radius_px * scale,
            max_radius_px=cfg.max_radius_px * scale,
            morph_radius_px=max(1, int(round(cfg.morph_radius_px * scale))),
        )

    gray = preprocess_to_gray(img_small, cfg_eff)                 
    score, comps = foreground_score(gray, cfg_eff)               

    thr = otsu_threshold(score)
    if isinstance(thr, Tensor):
        mask_raw = (score >= thr.view(-1, 1, 1, 1)).float()
    else:
        mask_raw = (score >= thr).float()
    mask = _binary_opening(mask_raw, cfg_eff.morph_radius_px)
    mask = _binary_closing(mask, cfg_eff.morph_radius_px)

    dist = edt_torch_jump_flood(mask)
    peaks = distance_peaks(dist, cfg_eff)

    radii = dist[peaks]
    if radii.numel() == 0:
        raise RuntimeError(
            "No distance peaks found. Try lowering peak_rel_threshold, "
            "reducing morph_radius_px, or setting a radius_hint_px."
        )

    radius_small, r_sorted, r_trim = robust_trimmed_mean(radii, cfg_eff)
    radius_px = radius_small / scale
    diameter_px = 2.0 * radius_px

    out: Dict[str, Union[float, Tensor, Dict[str, Tensor]]] = {
        "radius_px": float(radius_px),
        "diameter_px": float(diameter_px),
        "threshold": float(thr) if not isinstance(thr, Tensor) else thr.detach().cpu().tolist(),
        "scale_used": float(scale),
    }

    if cfg.return_debug:
        out["debug"] = {
            "gray": gray.detach().cpu(),
            "score": score.detach().cpu(),
            "mask_raw": mask_raw.detach().cpu(),
            "mask": mask.detach().cpu(),
            "dist": dist.detach().cpu(),
            "peaks": peaks.detach().cpu(),
            "radii_samples": radii.detach().cpu(),
            "radii_sorted": r_sorted.detach().cpu(),
            "radii_trimmed": r_trim.detach().cpu(),
            "I": comps["I"].detach().cpu(),
            "G": comps["G"].detach().cpu(),
            "blob": comps["blob"].detach().cpu(),
        }

    return out
