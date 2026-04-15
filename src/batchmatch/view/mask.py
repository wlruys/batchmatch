from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail
from .config import ChannelSelection, MaskViewSpec, MaskOverlaySpec
from . import render

__all__ = [
    "render_mask",
    "render_mask_overlay",
    "render_mask_contour_overlay",
    "render_mask_from_tensor",
]


def _blend_mask(
    base: Tensor,
    overlay: Tensor,
    alpha: Tensor,
    mode: str,
) -> Tensor:
    if mode == "alpha":
        return render.blend_alpha(base, overlay, alpha)
    if mode == "multiply":
        blended = render.blend_multiply(base, overlay)
        return render.blend_alpha(base, blended, alpha)
    if mode == "screen":
        blended = 1.0 - (1.0 - base) * (1.0 - overlay)
        return render.blend_alpha(base, blended, alpha)
    raise ValueError(f"Unknown blend mode: {mode}")


def render_mask(
    mask: Tensor,
    spec: MaskViewSpec = MaskViewSpec(),
) -> Tensor:
    mask = render.to_chw(mask)
    if mask.shape[0] != 1:
        mask = mask[:1]

    mask_float = mask.float()

    if spec.invert:
        mask_float = 1.0 - mask_float

    if spec.mode == "binary":
        return render.apply_colormap(mask_float, spec.binary_colormap)

    elif spec.mode == "overlay":
        color = torch.tensor(spec.overlay_color, dtype=mask_float.dtype, device=mask_float.device)
        color = color.view(3, 1, 1).expand(-1, mask.shape[1], mask.shape[2])
        return color * mask_float

    elif spec.mode == "contour":
        contour = render.mask_to_contour(mask_float, spec.contour_thickness)
        color = torch.tensor(spec.contour_color, dtype=mask_float.dtype, device=mask_float.device)
        color = color.view(3, 1, 1).expand(-1, mask.shape[1], mask.shape[2])
        return contour * color

    elif spec.mode == "outline":
        contour = render.mask_to_contour(mask_float, spec.outline_thickness)
        color = torch.tensor(spec.outline_color, dtype=mask_float.dtype, device=mask_float.device)
        color = color.view(3, 1, 1).expand(-1, mask.shape[1], mask.shape[2])
        return contour * color

    elif spec.mode == "alpha":
        bg = torch.tensor(spec.background_color, dtype=mask_float.dtype, device=mask_float.device)
        bg = bg.view(3, 1, 1).expand(-1, mask.shape[1], mask.shape[2])
        fg = torch.ones(3, mask.shape[1], mask.shape[2], dtype=mask_float.dtype, device=mask_float.device)
        return render.blend_alpha(bg, fg, mask_float)

    else:
        raise ValueError(f"Unknown mask mode: {spec.mode}")


# ---------------------------------------------------------------------------
# Shared helpers for overlay rendering
# ---------------------------------------------------------------------------

def _prepare_mask(mask: Tensor, invert: bool) -> Tensor:
    """Convert mask to single-channel CHW float, optionally inverted."""
    mask = render.to_chw(mask)
    if mask.shape[0] != 1:
        mask = mask[:1]
    mask_float = mask.float()
    if invert:
        mask_float = 1.0 - mask_float
    return mask_float


def _prepare_image_rgb(image: Tensor, spec: MaskOverlaySpec) -> Tensor:
    """Normalize an image tensor and convert to RGB for mask compositing."""
    img_norm = render.prepare_for_display(
        image,
        spec.image_spec.normalize,
        spec.image_spec.percentile_low,
        spec.image_spec.percentile_high,
        spec.image_spec.vmin,
        spec.image_spec.vmax,
        spec.image_spec.gamma,
        spec.image_spec.normalize_per_image,
    )
    return render.to_rgb(img_norm)


def _resize_mask_to_image(mask: Tensor, image: Tensor) -> Tensor:
    """Resize mask to match the spatial dimensions of *image* (nearest interp)."""
    mask_hw = (mask.shape[-2], mask.shape[-1])
    img_hw = (image.shape[-2], image.shape[-1])
    if mask_hw == img_hw:
        return mask
    ndim = mask.ndim
    m = mask.float()
    while m.ndim < 4:
        m = m.unsqueeze(0)
    m = F.interpolate(m, size=img_hw, mode="nearest")
    while m.ndim > ndim:
        m = m.squeeze(0)
    return m


def _apply_mask_to_rgb(
    img_rgb: Tensor,
    mask_float: Tensor,
    spec: MaskOverlaySpec,
) -> Tensor:
    """Blend a prepared mask onto a prepared RGB image according to the spec mode."""
    H, W = mask_float.shape[-2:]

    if spec.mask_spec.mode == "overlay":
        color = torch.tensor(
            spec.mask_spec.overlay_color,
            dtype=img_rgb.dtype,
            device=img_rgb.device,
        ).view(3, 1, 1)
        mask_rgb = color.expand(-1, H, W)
        alpha = spec.mask_spec.overlay_alpha * mask_float
        return _blend_mask(img_rgb, mask_rgb, alpha, spec.blend_mode)

    if spec.mask_spec.mode == "contour":
        contour = render.mask_to_contour(mask_float, spec.mask_spec.contour_thickness)
        color = torch.tensor(
            spec.mask_spec.contour_color,
            dtype=img_rgb.dtype,
            device=img_rgb.device,
        ).view(3, 1, 1)
        return _blend_mask(img_rgb, color.expand(-1, H, W), contour, spec.blend_mode)

    if spec.mask_spec.mode == "outline":
        contour = render.mask_to_contour(mask_float, spec.mask_spec.outline_thickness)
        color = torch.tensor(
            spec.mask_spec.outline_color,
            dtype=img_rgb.dtype,
            device=img_rgb.device,
        ).view(3, 1, 1)
        return _blend_mask(img_rgb, color.expand(-1, H, W), contour, spec.blend_mode)

    if spec.mask_spec.mode == "alpha":
        bg = torch.tensor(
            spec.mask_spec.background_color,
            dtype=img_rgb.dtype,
            device=img_rgb.device,
        ).view(3, 1, 1).expand_as(img_rgb)
        return render.blend_alpha(bg, img_rgb, mask_float)

    if spec.mask_spec.mode == "binary":
        return render.apply_colormap(mask_float, spec.mask_spec.binary_colormap)

    raise ValueError(f"Unknown mask mode: {spec.mask_spec.mode}")


# ---------------------------------------------------------------------------
# Public overlay functions
# ---------------------------------------------------------------------------

def render_mask_overlay(
    detail: ImageDetail,
    spec: MaskOverlaySpec = MaskOverlaySpec(),
    image_key: str = None,
    mask_key: str = None,
) -> Tensor:
    image_key = image_key or ImageDetail.Keys.IMAGE
    mask_key = mask_key or ImageDetail.Keys.DOMAIN.MASK

    image = detail.get(image_key)
    mask = detail.get(mask_key)

    image = render.select_channels(image, spec.image_spec.channel)
    image = render.downsample_for_display(image, spec.image_spec.max_display_size)
    mask = _resize_mask_to_image(mask, image)

    img_rgb = _prepare_image_rgb(image, spec)
    mask_float = _prepare_mask(mask, spec.mask_spec.invert)
    return _apply_mask_to_rgb(img_rgb, mask_float, spec)


def render_mask_contour_overlay(
    detail: ImageDetail,
    contour_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    thickness: int = 2,
    image_key: str = None,
    mask_key: str = None,
) -> Tensor:
    spec = MaskOverlaySpec(
        mask_spec=MaskViewSpec(
            mode="contour",
            contour_color=contour_color,
            contour_thickness=thickness,
        ),
    )
    return render_mask_overlay(detail, spec, image_key, mask_key)


def render_mask_from_tensor(
    image: Tensor,
    mask: Tensor,
    spec: MaskOverlaySpec = MaskOverlaySpec(),
) -> Tensor:
    image = render.select_channels(image, spec.image_spec.channel)
    image = render.downsample_for_display(image, spec.image_spec.max_display_size)
    mask = _resize_mask_to_image(mask, image)
    img_rgb = _prepare_image_rgb(image, spec)
    mask_float = _prepare_mask(mask, spec.mask_spec.invert)
    return _apply_mask_to_rgb(img_rgb, mask_float, spec)
