from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail
from .config import GallerySpec, ImageViewSpec
from . import render

__all__ = [
    "create_image_grid",
    "render_detail_gallery",
    "render_detail_components",
    "render_batch_gallery",
    "render_tensor_gallery",
    "render_channel_gallery",
]


def _render_for_gallery(tensor: Tensor, spec: ImageViewSpec) -> Tensor:
    """Select channels, downsample, normalize, and convert to RGB."""
    tensor = render.select_channels(tensor, spec.channel)
    tensor = render.downsample_for_display(tensor, spec.max_display_size)
    rendered = render.prepare_for_display(
        tensor,
        spec.normalize,
        spec.percentile_low,
        spec.percentile_high,
        spec.vmin,
        spec.vmax,
        spec.gamma,
        spec.normalize_per_image,
    )
    return render.to_rgb(rendered)


def create_image_grid(
    images: Sequence[Tensor],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: int = 2,
    pad_value: float = 0.0,
) -> Tensor:
    if not images:
        raise ValueError("Cannot create grid from empty image list")

    n = len(images)
    nrows, ncols = _compute_grid_layout(n, nrows, ncols)

    first = render.to_chw(images[0])
    C, H, W = first.shape

    resized = []
    for img in images:
        img = render.to_chw(img)
        if img.shape[-2:] != (H, W):
            img = F.interpolate(
                img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)
        if img.shape[0] == 1 and C == 3:
            img = img.expand(3, -1, -1)
        elif img.shape[0] == 3 and C == 1:
            img = img.mean(dim=0, keepdim=True)
        resized.append(img)

    grid_h = nrows * H + (nrows - 1) * padding
    grid_w = ncols * W + (ncols - 1) * padding
    grid = torch.full(
        (C, grid_h, grid_w),
        pad_value,
        dtype=first.dtype,
        device=first.device,
    )

    for i, img in enumerate(resized):
        row = i // ncols
        col = i % ncols
        y0 = row * (H + padding)
        x0 = col * (W + padding)
        grid[:, y0:y0+H, x0:x0+W] = img

    return grid


def render_detail_gallery(
    details: Sequence[ImageDetail],
    key: str = None,
    spec: GallerySpec = GallerySpec(),
) -> list[Tensor]:
    key = key or ImageDetail.Keys.IMAGE

    results = []
    for detail in details:
        if key not in detail:
            continue
        img = detail.get(key)
        results.append(_render_for_gallery(img, spec.image_spec))
    return results


def render_detail_components(
    detail: ImageDetail,
    keys: Sequence[str],
    specs: Optional[Sequence[ImageViewSpec]] = None,
) -> list[Tensor]:
    results = []
    for i, key in enumerate(keys):
        if key not in detail:
            continue
        tensor = detail.get(key)
        spec = specs[i] if specs and i < len(specs) else ImageViewSpec()
        results.append(_render_for_gallery(tensor, spec))
    return results


def render_batch_gallery(
    detail: ImageDetail,
    key: str = None,
    spec: GallerySpec = GallerySpec(),
) -> list[Tensor]:
    key = key or ImageDetail.Keys.IMAGE

    tensor = detail.get(key)
    tensor = render.to_bchw(tensor)
    B = tensor.shape[0]

    results = []
    for b in range(B):
        results.append(_render_for_gallery(tensor[b], spec.image_spec))
    return results


def render_tensor_gallery(
    tensors: Sequence[Tensor],
    spec: GallerySpec = GallerySpec(),
) -> list[Tensor]:
    return [_render_for_gallery(t, spec.image_spec) for t in tensors]


def render_channel_gallery(
    image: Union[Tensor, "ImageDetail"],
    spec: GallerySpec = GallerySpec(),
    channels: Optional[Sequence[int]] = None,
) -> list[Tensor]:
    """Split a multi-channel image into per-channel displayable RGB tensors.

    Each channel is independently normalized and colormapped.  Useful for
    OME-TIFF images where each channel is a distinct modality.

    Parameters
    ----------
    image : Tensor or ImageDetail
        A CHW or BCHW tensor with any number of channels.
    spec : GallerySpec
        Controls normalization, colormap, and layout.  ``image_spec.channel``
        is ignored — each channel is rendered individually.
    channels : list of int, optional
        Which channels to include.  ``None`` renders all channels.

    Returns
    -------
    list[Tensor]
        One 3×H×W RGB tensor per selected channel.
    """
    if hasattr(image, 'get'):
        tensor = image.get(ImageDetail.Keys.IMAGE)
    else:
        tensor = image

    tensor = render.to_chw(tensor)
    C = tensor.shape[0]

    if channels is None:
        channels = list(range(C))

    from dataclasses import replace as dc_replace
    base_spec = dc_replace(spec.image_spec, channel=None)

    results = []
    for ch in channels:
        if ch < 0 or ch >= C:
            raise IndexError(f"channel {ch} out of range for {C}-channel image")
        results.append(_render_for_gallery(tensor[ch:ch+1], base_spec))
    return results


def _compute_grid_layout(
    n: int, nrows: Optional[int], ncols: Optional[int]
) -> tuple[int, int]:
    if nrows and ncols:
        return nrows, ncols
    if nrows:
        return nrows, (n + nrows - 1) // nrows
    if ncols:
        return (n + ncols - 1) // ncols, ncols
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols
