from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "normalize_minmax",
    "normalize_percentile",
    "normalize_symmetric",
    "normalize_for_display",
    "prepare_for_display",
    "apply_gamma",
    "apply_tint",
    "to_grayscale",
    "to_rgb",
    "to_uint8",
    "to_chw",
    "to_bchw",
    "select_channels",
    "downsample_for_display",
    "apply_colormap",
    "apply_cyclic_colormap",
    "blend_alpha",
    "blend_multiply",
    "gradient_magnitude",
    "gradient_orientation",
    "gradient_to_rgb",
    "orientation_to_rgb",
    "mask_to_contour",
    "mask_to_overlay",
    "create_checkerboard_mask",
    "create_feather_weights",
    "draw_grid_lines",
    "compute_edge_magnitude",
    "threshold_edges",
]

from batchmatch.helpers.tensor import to_chw, to_bchw


def normalize_minmax(image: Tensor, per_channel: bool = False) -> Tensor:
    if per_channel and image.ndim >= 3:
        dims = tuple(range(image.ndim - 2, image.ndim))
        vmin = image.amin(dim=dims, keepdim=True)
        vmax = image.amax(dim=dims, keepdim=True)
    else:
        vmin = image.min()
        vmax = image.max()
    return (image - vmin) / (vmax - vmin + 1e-8)


def normalize_percentile(
    image: Tensor,
    low: float = 2.0,
    high: float = 98.0,
) -> Tensor:
    flat = image.reshape(-1)
    low_val = torch.quantile(flat, low / 100.0)
    high_val = torch.quantile(flat, high / 100.0)
    out = (image - low_val) / (high_val - low_val + 1e-8)
    return out.clamp(0, 1)


def normalize_symmetric(image: Tensor, per_image: bool = False) -> Tensor:
    if per_image and image.ndim == 4:
        max_abs = image.abs().amax(dim=(1, 2, 3), keepdim=True)
    else:
        max_abs = image.abs().max()
    if max_abs < 1e-8:
        return image
    return image / max_abs


def normalize_for_display(
    image: Tensor,
    mode: str = "minmax",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    per_image: bool = False,
) -> Tensor:
    if vmin is not None and vmax is not None:
        out = (image - vmin) / (vmax - vmin + 1e-8)
        return out.clamp(0, 1)

    if mode == "none":
        return image.clamp(0, 1)
    if mode == "minmax":
        if per_image and image.ndim == 4:
            vmin = image.amin(dim=(1, 2, 3), keepdim=True)
            vmax = image.amax(dim=(1, 2, 3), keepdim=True)
            return (image - vmin) / (vmax - vmin + 1e-8)
        return normalize_minmax(image)
    if mode == "percentile":
        if per_image and image.ndim == 4:
            low_val = torch.quantile(image, percentile_low / 100.0, dim=(1, 2, 3), keepdim=True)
            high_val = torch.quantile(image, percentile_high / 100.0, dim=(1, 2, 3), keepdim=True)
            out = (image - low_val) / (high_val - low_val + 1e-8)
            return out.clamp(0, 1)
        return normalize_percentile(image, percentile_low, percentile_high)
    if mode == "abs":
        image_abs = image.abs()
        if per_image and image_abs.ndim == 4:
            vmin = image_abs.amin(dim=(1, 2, 3), keepdim=True)
            vmax = image_abs.amax(dim=(1, 2, 3), keepdim=True)
            return (image_abs - vmin) / (vmax - vmin + 1e-8)
        return normalize_minmax(image_abs)
    raise ValueError(f"Unknown normalize mode: {mode}")


def apply_gamma(image: Tensor, gamma: float) -> Tensor:
    if gamma == 1.0:
        return image
    if gamma <= 0:
        return (image > 0).float()
    return image.clamp(0, 1).pow(1.0 / gamma)


def prepare_for_display(
    image: Tensor,
    mode: str = "minmax",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gamma: float = 1.0,
    per_image: bool = False,
) -> Tensor:
    if per_image and image.ndim == 4:
        image = to_bchw(image)
        rendered = normalize_for_display(
            image,
            mode,
            percentile_low,
            percentile_high,
            vmin,
            vmax,
            per_image=True,
        )
        rendered = apply_gamma(rendered, gamma)
        return to_chw(rendered)

    image = to_chw(image)
    rendered = normalize_for_display(
        image, mode, percentile_low, percentile_high, vmin, vmax, per_image=False
    )
    return apply_gamma(rendered, gamma)


def apply_tint(
    image: Tensor, tint: Tuple[float, float, float], strength: float
) -> Tensor:
    if strength <= 0:
        return image
    tint_tensor = torch.tensor(tint, dtype=image.dtype, device=image.device)
    while tint_tensor.ndim < image.ndim:
        tint_tensor = tint_tensor.unsqueeze(-1)
    return (1.0 - strength) * image + strength * (image * tint_tensor)


def to_rgb(image: Tensor) -> Tensor:
    image = to_chw(image)
    C = image.shape[0]
    if C == 1:
        return image.expand(3, -1, -1)
    if C == 2:
        # Pad with a zero channel so 2-channel images can still be viewed.
        return torch.cat(
            [image, torch.zeros_like(image[:1])], dim=0
        )
    if C == 3:
        return image
    if C > 3:
        return image[:3]
    raise ValueError(f"Cannot convert {C}-channel image to RGB")


ChannelSelection = Union[int, Tuple[int, ...]]


def select_channels(
    image: Tensor,
    channel: Optional[ChannelSelection] = None,
) -> Tensor:
    """Select channel(s) from a CHW tensor for display.

    Parameters
    ----------
    image : Tensor
        CHW image tensor with arbitrary number of channels.
    channel : int, tuple[int, ...], or None
        * ``None`` — auto-select: C==1 pass-through, C<=3 keep as-is,
          C>3 take first channel.
        * ``int`` — extract a single channel (returns 1×H×W).
        * ``tuple`` of 1–3 ints — gather those channels in order.

    Returns
    -------
    Tensor
        CHW tensor with C in {1, 2, 3}.
    """
    image = to_chw(image)
    C = image.shape[0]

    if channel is None:
        if C <= 3:
            return image
        return image[:1]

    if isinstance(channel, int):
        if channel < 0 or channel >= C:
            raise IndexError(
                f"channel {channel} out of range for {C}-channel image"
            )
        return image[channel : channel + 1]

    # tuple of indices
    indices = list(channel)
    for idx in indices:
        if idx < 0 or idx >= C:
            raise IndexError(
                f"channel {idx} out of range for {C}-channel image"
            )
    return image[indices]


def downsample_for_display(
    image: Tensor,
    max_size: Optional[int] = None,
) -> Tensor:
    """Downsample a CHW or BCHW tensor so max(H, W) <= *max_size*.

    Uses area interpolation for clean downsampling.  Returns the image
    unchanged when it is already within limits or *max_size* is ``None``.
    """
    if max_size is None:
        return image
    H, W = image.shape[-2], image.shape[-1]
    long_edge = max(H, W)
    if long_edge <= max_size:
        return image

    scale = max_size / long_edge
    new_h = max(1, int(H * scale))
    new_w = max(1, int(W * scale))
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    image = F.interpolate(image, size=(new_h, new_w), mode="area")
    if squeeze:
        image = image.squeeze(0)
    return image


def to_grayscale(image: Tensor) -> Tensor:
    if image.ndim == 3:
        image = to_chw(image)
        if image.shape[0] == 1:
            return image
        return image.mean(dim=0, keepdim=True)
    if image.ndim == 4:
        image = to_bchw(image)
        if image.shape[1] == 1:
            return image
        return image.mean(dim=1, keepdim=True)
    raise ValueError(f"Expected 3D or 4D image tensor, got {image.ndim}D.")


def to_uint8(image: Tensor) -> Tensor:
    return (image.clamp(0, 1) * 255).to(torch.uint8)


_COLORMAP_CACHE: dict[str, Tensor] = {}


def _get_colormap_lut(name: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    cache_key = f"{name}_{device}_{dtype}"
    if cache_key in _COLORMAP_CACHE:
        return _COLORMAP_CACHE[cache_key]

    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(name)
        lut = torch.tensor(
            [cmap(i / 255.0)[:3] for i in range(256)],
            dtype=dtype,
            device=device,
        )
    except ImportError:
        lut = torch.linspace(0, 1, 256, dtype=dtype, device=device).unsqueeze(1)
        lut = lut.expand(-1, 3)

    _COLORMAP_CACHE[cache_key] = lut
    return lut


def apply_colormap(image: Tensor, colormap: str = "viridis") -> Tensor:
    image = to_chw(image)
    if image.shape[0] != 1:
        image = image.mean(dim=0, keepdim=True)

    img_norm = normalize_minmax(image)

    lut = _get_colormap_lut(colormap, image.device, image.dtype)

    indices = (img_norm[0] * 255).clamp(0, 255).long()

    rgb = lut[indices]
    return rgb.permute(2, 0, 1)


def apply_cyclic_colormap(image: Tensor, colormap: str = "hsv") -> Tensor:
    return apply_colormap(image, colormap)


def blend_alpha(
    base: Tensor, overlay: Tensor, alpha: float | Tensor
) -> Tensor:
    if isinstance(alpha, (int, float)):
        return (1.0 - alpha) * base + alpha * overlay
    alpha = alpha.to(base.dtype)
    while alpha.ndim < base.ndim:
        alpha = alpha.unsqueeze(0)
    return (1.0 - alpha) * base + alpha * overlay


def blend_multiply(base: Tensor, overlay: Tensor) -> Tensor:
    return base * overlay


def gradient_magnitude(gx: Tensor, gy: Tensor) -> Tensor:
    return torch.sqrt(gx * gx + gy * gy)


def gradient_orientation(gx: Tensor, gy: Tensor) -> Tensor:
    return torch.atan2(gy, gx) + math.pi


def orientation_to_rgb(
    orientation: Tensor, magnitude: Optional[Tensor] = None
) -> Tensor:
    orientation = to_chw(orientation)
    if orientation.shape[0] != 1:
        orientation = orientation[:1]

    hue = orientation[0] / (2 * math.pi)
    hue = hue.clamp(0, 1)

    sat = torch.ones_like(hue)

    if magnitude is not None:
        magnitude = to_chw(magnitude)
        if magnitude.shape[0] != 1:
            magnitude = magnitude[:1]
        val = normalize_minmax(magnitude[0])
    else:
        val = torch.ones_like(hue)

    h = hue * 6.0
    i = h.floor()
    f = h - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))

    i = i.long() % 6

    r = torch.zeros_like(hue)
    g = torch.zeros_like(hue)
    b = torch.zeros_like(hue)

    mask0 = i == 0
    mask1 = i == 1
    mask2 = i == 2
    mask3 = i == 3
    mask4 = i == 4
    mask5 = i == 5

    r[mask0] = val[mask0]
    g[mask0] = t[mask0]
    b[mask0] = p[mask0]

    r[mask1] = q[mask1]
    g[mask1] = val[mask1]
    b[mask1] = p[mask1]

    r[mask2] = p[mask2]
    g[mask2] = val[mask2]
    b[mask2] = t[mask2]

    r[mask3] = p[mask3]
    g[mask3] = q[mask3]
    b[mask3] = val[mask3]

    r[mask4] = t[mask4]
    g[mask4] = p[mask4]
    b[mask4] = val[mask4]

    r[mask5] = val[mask5]
    g[mask5] = p[mask5]
    b[mask5] = q[mask5]

    return torch.stack([r, g, b], dim=0)


def gradient_to_rgb(
    gx: Tensor,
    gy: Tensor,
    mode: str = "hsv",
    normalize: bool = True,
    per_image: bool = False,
) -> Tensor:
    if per_image:
        gx = to_bchw(gx)
        gy = to_bchw(gy)
        if gx.shape[1] != 1:
            gx = gx[:, :1]
        if gy.shape[1] != 1:
            gy = gy[:, :1]
    else:
        gx = to_chw(gx)
        gy = to_chw(gy)
        if gx.shape[0] != 1:
            gx = gx[:1]
        if gy.shape[0] != 1:
            gy = gy[:1]

    if mode == "hsv":
        orientation = gradient_orientation(gx, gy)
        magnitude = gradient_magnitude(gx, gy)
        if per_image:
            magnitude = normalize_for_display(magnitude, "minmax", per_image=True)
            orientation = to_chw(orientation)
            magnitude = to_chw(magnitude)
        return orientation_to_rgb(orientation, magnitude)

    elif mode == "rg":
        if normalize:
            gx_norm = normalize_symmetric(gx, per_image=per_image)
            gy_norm = normalize_symmetric(gy, per_image=per_image)
        else:
            gx_norm = gx.clamp(-1, 1)
            gy_norm = gy.clamp(-1, 1)
        gx_norm = gx_norm * 0.5 + 0.5
        gy_norm = gy_norm * 0.5 + 0.5
        mag = normalize_for_display(gradient_magnitude(gx, gy), "minmax", per_image=per_image)
        gx_norm = to_chw(gx_norm)[0]
        gy_norm = to_chw(gy_norm)[0]
        mag = to_chw(mag)[0]
        return torch.stack([gx_norm, gy_norm, mag], dim=0)

    else:
        raise ValueError(f"Unknown gradient visualization mode: {mode}")



def mask_to_contour(mask: Tensor, thickness: int = 1) -> Tensor:
    mask = to_chw(mask)
    if mask.shape[0] != 1:
        mask = mask[:1]

    mask_float = mask.float()

    k = 2 * thickness + 1
    kernel = torch.ones(1, 1, k, k, dtype=mask_float.dtype, device=mask_float.device)

    padded = F.pad(mask_float.unsqueeze(0), (thickness, thickness, thickness, thickness), mode="constant", value=0)
    dilated = F.conv2d(padded, kernel, padding=0)
    dilated = (dilated > 0).float()

    padded = F.pad(mask_float.unsqueeze(0), (thickness, thickness, thickness, thickness), mode="constant", value=1)
    eroded = F.conv2d(padded, kernel, padding=0)
    eroded = (eroded >= k * k).float()

    contour = (dilated - eroded).clamp(0, 1)
    return contour[0]


def mask_to_overlay(
    image: Tensor,
    mask: Tensor,
    color: Tuple[float, float, float],
    alpha: float,
) -> Tensor:
    image = to_chw(image)
    mask = to_chw(mask)

    image_rgb = to_rgb(image)

    if mask.shape[0] != 1:
        mask = mask[:1]

    color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(
        3, 1, 1
    )
    colored_mask = color_tensor.expand(-1, mask.shape[1], mask.shape[2])

    return blend_alpha(image_rgb, colored_mask, alpha * mask)


def create_checkerboard_mask(
    height: int,
    width: int,
    tiles_y: int,
    tiles_x: int,
    align: str = "ul",
    device: Optional[torch.device] = None,
) -> Tensor:
    ty = torch.arange(height, device=device) * tiles_y // height
    tx = torch.arange(width, device=device) * tiles_x // width
    yy = ty.view(-1, 1)
    xx = tx.view(1, -1)
    mask = ((yy + xx) % 2 == 0).float()
    if align == "center":
        mask = torch.roll(mask, shifts=(tiles_y // 2, tiles_x // 2), dims=(0, 1))
    return mask.unsqueeze(0)


def create_feather_weights(
    height: int,
    width: int,
    tiles_y: int,
    tiles_x: int,
    feather_px: int,
    device: Optional[torch.device] = None,
) -> Optional[Tensor]:
    if feather_px <= 0:
        return None

    by = torch.linspace(0, height, tiles_y + 1, device=device)
    bx = torch.linspace(0, width, tiles_x + 1, device=device)
    yy = torch.arange(height, device=device).float().view(-1, 1).expand(-1, width)
    xx = torch.arange(width, device=device).float().view(1, -1).expand(height, -1)

    dy = torch.zeros_like(yy)
    for i in range(tiles_y + 1):
        dy = torch.minimum(dy if i > 0 else torch.full_like(yy, float("inf")), (yy - by[i]).abs())

    dx = torch.zeros_like(xx)
    for i in range(tiles_x + 1):
        dx = torch.minimum(dx if i > 0 else torch.full_like(xx, float("inf")), (xx - bx[i]).abs())

    d = torch.minimum(dy, dx)
    t = (d / float(feather_px)).clamp(0, 1)
    w_ref = 0.5 + 0.5 * torch.cos(math.pi * (1 - t))
    return w_ref.unsqueeze(0)


def draw_grid_lines(
    image: Tensor,
    tiles: Tuple[int, int],
    color: Tuple[float, float, float],
    alpha: float,
    thickness: int,
) -> Tensor:
    if alpha <= 0 or thickness <= 0:
        return image

    image = to_chw(image)
    C, H, W = image.shape
    out = image.clone()

    ty, tx = tiles
    gy = torch.linspace(0, H, ty + 1).round().long().tolist()
    gx = torch.linspace(0, W, tx + 1).round().long().tolist()

    color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(
        3, 1, 1
    )

    if C == 1:
        out = out.expand(3, -1, -1).clone()
        C = 3

    half = thickness // 2

    for y in gy:
        y0 = max(0, y - half)
        y1 = min(H, y0 + thickness)
        out[:, y0:y1, :] = (1 - alpha) * out[:, y0:y1, :] + alpha * color_tensor

    for x in gx:
        x0 = max(0, x - half)
        x1 = min(W, x0 + thickness)
        out[:, :, x0:x1] = (1 - alpha) * out[:, :, x0:x1] + alpha * color_tensor

    return out


#TODO(wlr): Redundant with gradient module, but just eh, keep here for now. Cleanup later.

def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3) / 4.0
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3) / 4.0
    return kx, ky


def compute_edge_magnitude(image: Tensor) -> Tensor:
    image = to_bchw(image)

    if image.shape[1] > 1:
        gray = image.mean(dim=1, keepdim=True)
    else:
        gray = image

    kx, ky = _sobel_kernels(image.device, image.dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)

    mag = mag / (mag.max() + 1e-8)
    return mag[0]


def threshold_edges(magnitude: Tensor, threshold: float) -> Tensor:
    return (magnitude >= threshold).float()
