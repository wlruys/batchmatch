from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail
from batchmatch.helpers.tensor import to_bchw_flexible
from .config import ChannelSelection, OverlaySpec, EdgeOverlaySpec, CheckerboardSpec
from . import render

__all__ = [
    "render_overlay",
    "render_edge_overlay",
    "render_checkerboard",
    "render_side_by_side",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_edges_from_detail(detail: ImageDetail, fallback_img: Tensor = None) -> Tensor:
    """Compute normalized edge magnitude from an ImageDetail.

    Uses stored gradients when available, otherwise falls back to Sobel on the
    image tensor.  When a domain mask is present the normalization ignores
    boundary pixels so that padding does not dominate the range.
    """
    if ImageDetail.Keys.GRAD.X in detail:
        gx = render.to_chw(detail.get(ImageDetail.Keys.GRAD.X))
        gy = render.to_chw(detail.get(ImageDetail.Keys.GRAD.Y))
        if gx.shape[0] > 1:
            gx = gx[:1]
        if gy.shape[0] > 1:
            gy = gy[:1]
        mag = render.gradient_magnitude(gx, gy)
    else:
        img = fallback_img if fallback_img is not None else detail.get(ImageDetail.Keys.IMAGE)
        mag = render.compute_edge_magnitude(img)

    mask = detail.get(ImageDetail.Keys.DOMAIN.MASK, default=None)
    if mask is None:
        return render.normalize_minmax(mag)

    # Handle BHW or BCHW masks -> CHW
    mask = to_bchw_flexible(mask)[0]
    if mask.shape[-2:] != mag.shape[-2:]:
        mask = F.interpolate(
            mask.unsqueeze(0), size=mag.shape[-2:], mode="nearest"
        ).squeeze(0)

    # Erode mask slightly to exclude boundary pixels from normalization
    if mask.shape[0] == 1:
        k = 5
        kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype)
        eroded = F.conv2d(mask.unsqueeze(0), kernel, padding=k // 2)
        eroded = (eroded >= k * k).float().squeeze(0)
    else:
        eroded = mask

    # Normalize within valid region only
    valid_vals = (mag * eroded)[eroded > 0.5]
    if valid_vals.numel() > 0:
        vmin, vmax = valid_vals.min(), valid_vals.max()
        mag = ((mag - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    else:
        mag = render.normalize_minmax(mag)
    return mag


def _dilate_edge_mask(edge_mask: Tensor, thickness: int) -> Tensor:
    """Dilate a CHW edge mask using a box filter when thickness > 1."""
    if thickness <= 1:
        return edge_mask
    k = 2 * thickness + 1
    kernel = torch.ones(1, 1, k, k, dtype=edge_mask.dtype, device=edge_mask.device)
    mask_4d = edge_mask.unsqueeze(0) if edge_mask.ndim == 3 else edge_mask.unsqueeze(0).unsqueeze(0)
    dilated = F.conv2d(mask_4d, kernel, padding=thickness)
    dilated = (dilated / kernel.numel()).clamp(0, 1).squeeze()
    if dilated.ndim == 2:
        dilated = dilated.unsqueeze(0)
    return dilated


def _blend_edges(
    image: Tensor,
    edges: Tensor,
    color: Tensor,
    spec: EdgeOverlaySpec,
    target_hw: tuple[int, int],
) -> Tensor:
    """Apply gamma, threshold, dilate and blend edge mask onto ``image``."""
    H, W = target_hw
    if edges.shape[-2:] != (H, W):
        edges = F.interpolate(
            edges.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)

    edges = render.apply_gamma(edges, spec.gamma)

    if spec.edge_threshold is not None:
        edge_mask = render.threshold_edges(edges, spec.edge_threshold)
    else:
        edge_mask = edges

    edge_mask = _dilate_edge_mask(edge_mask, spec.edge_thickness)
    alpha = (spec.edge_alpha * edge_mask).clamp(0, 1)
    return render.blend_alpha(image, color.expand(-1, H, W), alpha)


def _normalize_pair(
    ref: Tensor,
    mov: Tensor,
    mode: str,
    per_image: bool = False,
) -> tuple[Tensor, Tensor]:
    if mode == "none":
        return ref.clamp(0, 1), mov.clamp(0, 1)
    if mode == "minmax":
        if per_image and ref.ndim == 4:
            ref_min = ref.amin(dim=(1, 2, 3), keepdim=True)
            ref_max = ref.amax(dim=(1, 2, 3), keepdim=True)
            mov_min = mov.amin(dim=(1, 2, 3), keepdim=True)
            mov_max = mov.amax(dim=(1, 2, 3), keepdim=True)
            ref_norm = (ref - ref_min) / (ref_max - ref_min + 1e-8)
            mov_norm = (mov - mov_min) / (mov_max - mov_min + 1e-8)
            return ref_norm, mov_norm
        return render.normalize_minmax(ref), render.normalize_minmax(mov)
    if mode == "joint":
        if per_image and ref.ndim == 4:
            combined = torch.cat([ref, mov], dim=1)
            vmin = combined.amin(dim=(1, 2, 3), keepdim=True)
            vmax = combined.amax(dim=(1, 2, 3), keepdim=True)
            ref_norm = (ref - vmin) / (vmax - vmin + 1e-8)
            mov_norm = (mov - vmin) / (vmax - vmin + 1e-8)
            return ref_norm, mov_norm
        combined = torch.cat([ref.flatten(), mov.flatten()])
        vmin = combined.min()
        vmax = combined.max()
        ref_norm = (ref - vmin) / (vmax - vmin + 1e-8)
        mov_norm = (mov - vmin) / (vmax - vmin + 1e-8)
        return ref_norm, mov_norm
    raise ValueError(f"Unknown normalize mode: {mode}")


def _resize_to_match(mov: Tensor, ref: Tensor) -> Tensor:
    if mov.shape[-2:] == ref.shape[-2:]:
        return mov
    squeeze = mov.ndim == 3
    if squeeze:
        mov = mov.unsqueeze(0)
    mov = F.interpolate(mov, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    if squeeze:
        mov = mov.squeeze(0)
    return mov


def render_overlay(
    reference: ImageDetail,
    moving: ImageDetail,
    spec: OverlaySpec = OverlaySpec(),
) -> Tensor:
    ref_img = reference.get(ImageDetail.Keys.IMAGE)
    mov_img = moving.get(ImageDetail.Keys.IMAGE)

    # Multi-channel: select before layout conversion
    ref_img = render.select_channels(ref_img, spec.channel)
    mov_img = render.select_channels(mov_img, spec.channel)

    if spec.normalize_per_image:
        ref_img = render.to_bchw(ref_img)
        mov_img = render.to_bchw(mov_img)
    else:
        ref_img = render.to_chw(ref_img)
        mov_img = render.to_chw(mov_img)

    mov_img = _resize_to_match(mov_img, ref_img)

    if spec.grayscale:
        ref_img = render.to_grayscale(ref_img)
        mov_img = render.to_grayscale(mov_img)

    ref_norm, mov_norm = _normalize_pair(
        ref_img, mov_img, spec.normalize, spec.normalize_per_image
    )

    if spec.normalize_per_image:
        ref_norm = render.to_chw(ref_norm)
        mov_norm = render.to_chw(mov_norm)

    ref_rgb = render.to_rgb(ref_norm)
    mov_rgb = render.to_rgb(mov_norm)

    mov_tinted = render.apply_tint(mov_rgb, spec.moving_color, 1.0)

    return render.blend_alpha(ref_rgb, mov_tinted, spec.alpha)


def render_edge_overlay(
    reference: ImageDetail,
    moving: ImageDetail,
    spec: EdgeOverlaySpec = EdgeOverlaySpec(),
) -> Tensor:
    ref_img = render.select_channels(reference.get(ImageDetail.Keys.IMAGE), spec.channel)
    mov_img = render.select_channels(moving.get(ImageDetail.Keys.IMAGE), spec.channel)
    mov_img = _resize_to_match(mov_img, ref_img)

    base_rgb = render.to_rgb(render.normalize_minmax(ref_img))
    _, H, W = base_rgb.shape

    if spec.edge_source == "none":
        return base_rgb

    out = base_rgb.clone()
    if spec.dim_under_edges < 1.0:
        out = out * spec.dim_under_edges

    edge_color = torch.tensor(
        spec.edge_color, dtype=out.dtype, device=out.device
    ).view(3, 1, 1)

    if spec.edge_source in ("mov", "both"):
        mov_edges = _get_edges_from_detail(moving, mov_img)
        out = _blend_edges(out, mov_edges, edge_color, spec, (H, W))

    if spec.edge_source in ("ref", "both"):
        ref_edges = _get_edges_from_detail(reference, ref_img)
        if spec.edge_source == "both":
            ref_color = torch.tensor(
                (0.0, 0.5, 1.0), dtype=out.dtype, device=out.device
            ).view(3, 1, 1)
        else:
            ref_color = edge_color
        out = _blend_edges(out, ref_edges, ref_color, spec, (H, W))

    return out.clamp(0, 1)


def _compute_tiles(
    height: int,
    width: int,
    tiles: Optional[tuple[int, int]],
    tilesize: Optional[tuple[int, int]],
) -> tuple[int, int]:
    """Compute tile count from tiles or tilesize specification.

    If tilesize is specified, compute tile count based on image dimensions.
    Otherwise use tiles directly. At least one must be specified.
    """
    if tilesize is not None:
        if tiles is not None and tiles != (8, 8):
            raise ValueError("Cannot specify both tiles and tilesize")
        tiles_y = max(1, (height + tilesize[0] - 1) // tilesize[0])
        tiles_x = max(1, (width + tilesize[1] - 1) // tilesize[1])
        return (tiles_y, tiles_x)
    if tiles is not None:
        return tiles
    raise ValueError("Must specify either tiles or tilesize")


def render_checkerboard(
    reference: ImageDetail,
    moving: ImageDetail,
    spec: CheckerboardSpec = CheckerboardSpec(),
) -> Tensor:
    ref_img = reference.get(ImageDetail.Keys.IMAGE)
    mov_img = moving.get(ImageDetail.Keys.IMAGE)

    ref_img = render.select_channels(ref_img, spec.channel)
    mov_img = render.select_channels(mov_img, spec.channel)

    if spec.normalize_per_image:
        ref_img = render.to_bchw(ref_img)
        mov_img = render.to_bchw(mov_img)
    else:
        ref_img = render.to_chw(ref_img)
        mov_img = render.to_chw(mov_img)

    mov_img = _resize_to_match(mov_img, ref_img)

    if spec.grayscale:
        ref_img = render.to_grayscale(ref_img)
        mov_img = render.to_grayscale(mov_img)

    ref_norm, mov_norm = _normalize_pair(
        ref_img, mov_img, spec.normalize, spec.normalize_per_image
    )

    ref_norm = render.apply_gamma(ref_norm, spec.gamma)
    mov_norm = render.apply_gamma(mov_norm, spec.gamma)

    ref_scaled = ref_norm * spec.ref_gain
    mov_scaled = mov_norm * spec.mov_gain

    if spec.normalize_per_image:
        ref_scaled = render.to_chw(ref_scaled)
        mov_scaled = render.to_chw(mov_scaled)
    ref_rgb = render.to_rgb(ref_scaled)
    mov_rgb = render.to_rgb(mov_scaled)
    ref_rgb = render.apply_tint(ref_rgb, spec.ref_tint, spec.ref_tint_strength)
    mov_rgb = render.apply_tint(mov_rgb, spec.mov_tint, spec.mov_tint_strength)

    _, H, W = ref_rgb.shape
    tiles = _compute_tiles(H, W, spec.tiles, spec.tilesize)

    mask = render.create_checkerboard_mask(H, W, tiles[0], tiles[1], spec.align, ref_img.device)

    if spec.feather_px > 0:
        weights = render.create_feather_weights(H, W, tiles[0], tiles[1], spec.feather_px, ref_img.device)
        if weights is not None:
            w_ref = mask * (1 - weights * 0.5) + (1 - mask) * (weights * 0.5)
            out = w_ref * ref_rgb + (1 - w_ref) * mov_rgb
        else:
            out = torch.where(mask.bool().expand_as(ref_rgb), ref_rgb, mov_rgb)
    else:
        out = torch.where(mask.bool().expand_as(ref_rgb), ref_rgb, mov_rgb)

    if spec.edge_overlay.edge_source != "none":
        out = _apply_edge_overlay(out, reference, moving, spec.edge_overlay)

    if spec.grid:
        out = render.draw_grid_lines(
            out, tiles, spec.grid_color, spec.grid_alpha, spec.grid_thickness
        )

    return out.clamp(0, 1)


def _apply_edge_overlay(
    image: Tensor,
    reference: ImageDetail,
    moving: ImageDetail,
    spec: EdgeOverlaySpec,
) -> Tensor:
    _, H, W = image.shape
    out = image.clone()
    edge_color = torch.tensor(
        spec.edge_color, dtype=out.dtype, device=out.device
    ).view(3, 1, 1)

    if spec.edge_source in ("mov", "both"):
        mov_edges = _get_edges_from_detail(moving)
        out = _blend_edges(out, mov_edges, edge_color, spec, (H, W))

    if spec.edge_source in ("ref", "both"):
        ref_edges = _get_edges_from_detail(reference)
        ref_color = torch.tensor(
            (0.2, 0.2, 1.0), dtype=out.dtype, device=out.device
        ).view(3, 1, 1)
        out = _blend_edges(out, ref_edges, ref_color, spec, (H, W))

    return out


def render_side_by_side(
    reference: ImageDetail,
    moving: ImageDetail,
    gap: int = 4,
    gap_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    channel: Optional[ChannelSelection] = None,
    max_display_size: Optional[int] = None,
) -> Tensor:
    ref_img = reference.get(ImageDetail.Keys.IMAGE)
    mov_img = moving.get(ImageDetail.Keys.IMAGE)

    ref_img = render.select_channels(ref_img, channel)
    mov_img = render.select_channels(mov_img, channel)
    ref_img = render.downsample_for_display(ref_img, max_display_size)
    mov_img = render.downsample_for_display(mov_img, max_display_size)

    ref_img = render.to_chw(ref_img)
    mov_img = render.to_chw(mov_img)

    ref_h = ref_img.shape[-2]
    mov_h = mov_img.shape[-2]
    if ref_h != mov_h:
        target_h = max(ref_h, mov_h)
        if ref_h != target_h:
            scale = target_h / ref_h
            ref_img = F.interpolate(
                ref_img.unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if mov_h != target_h:
            scale = target_h / mov_h
            mov_img = F.interpolate(
                mov_img.unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    ref_rgb = render.to_rgb(render.normalize_minmax(ref_img))
    mov_rgb = render.to_rgb(render.normalize_minmax(mov_img))

    _, H, W_ref = ref_rgb.shape
    _, _, W_mov = mov_rgb.shape

    out_w = W_ref + gap + W_mov
    out = torch.zeros(3, H, out_w, dtype=ref_rgb.dtype, device=ref_rgb.device)

    gap_tensor = torch.tensor(gap_color, dtype=out.dtype, device=out.device).view(3, 1, 1)
    out[:, :, W_ref:W_ref+gap] = gap_tensor

    out[:, :, :W_ref] = ref_rgb
    out[:, :, W_ref+gap:] = mov_rgb

    return out
