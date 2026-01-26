from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail
from .config import (
    DisplaySpec,
    ImageViewSpec,
    GallerySpec,
    GradientViewSpec,
    GradientGallerySpec,
    MaskViewSpec,
    MaskOverlaySpec,
    CheckerboardSpec,
    OverlaySpec,
    EdgeOverlaySpec,
    QuadAnnotationSpec,
    BoxAnnotationSpec,
    PointAnnotationSpec,
)
from . import render
from . import gradient as gradient_module
from . import mask as mask_module
from . import composite
from . import gallery as gallery_module
from . import annotate

__all__ = [
    "show_image",
    "show_tensor",
    "show_images",
    "show_detail",
    "show_gradient",
    "show_gradients",
    "show_gradient_hsv",
    "show_mask",
    "show_mask_overlay",
    "show_comparison",
]


def _get_figure_axes(ax, display: DisplaySpec):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    figsize = display.figsize or (6, 5)
    fig = plt.figure(figsize=figsize, dpi=display.dpi)
    return fig, fig.gca()


def _get_image_hw(image: Union[Tensor, ImageDetail]) -> tuple[int, int]:
    if isinstance(image, ImageDetail):
        tensor = image.get(ImageDetail.Keys.IMAGE)
    else:
        tensor = image

    if tensor.ndim >= 2:
        return int(tensor.shape[-2]), int(tensor.shape[-1])
    raise ValueError(f"Expected tensor with at least 2 dims, got {tuple(tensor.shape)}.")


def _compute_gallery_figsize(
    sizes: Sequence[tuple[int, int]],
    nrows: int,
    ncols: int,
    per_image_size: tuple[float, float],
) -> tuple[tuple[float, float], list[int], list[int]]:
    grid = [[(1, 1) for _ in range(ncols)] for _ in range(nrows)]
    for i, (h, w) in enumerate(sizes):
        if i >= nrows * ncols:
            break
        row = i // ncols
        col = i % ncols
        grid[row][col] = (max(1, h), max(1, w))

    row_heights = [max(h for h, _ in row) for row in grid]
    col_widths = [max(grid[r][c][1] for r in range(nrows)) for c in range(ncols)]

    max_h = max(row_heights)
    max_w = max(col_widths)
    width_scale = per_image_size[0] / max_w
    height_scale = per_image_size[1] / max_h

    fig_w = sum(col_widths) * width_scale
    fig_h = sum(row_heights) * height_scale
    return (fig_w, fig_h), row_heights, col_widths


def _plot_tensor(ax, tensor: Tensor, cmap: str = "gray", interpolation: str = "nearest"):
    tensor = render.to_chw(tensor)
    if tensor.shape[0] == 1:
        ax.imshow(tensor[0].detach().cpu().numpy(), cmap=cmap, interpolation=interpolation)
    else:
        ax.imshow(tensor.permute(1, 2, 0).detach().cpu().numpy(), interpolation=interpolation)


def _finalize_axes(ax, display: DisplaySpec):
    if display.title:
        ax.set_title(display.title)
    ax.axis("off")


def _finalize_figure(fig, display: DisplaySpec):
    import matplotlib.pyplot as plt

    if display.tight_layout:
        fig.tight_layout()
    if display.save_path:
        fig.savefig(display.save_path, dpi=display.dpi, bbox_inches="tight")
    if display.show:
        plt.show()


def _compute_grid_layout(n: int, nrows: Optional[int], ncols: Optional[int]) -> tuple[int, int]:
    if nrows and ncols:
        return nrows, ncols
    if nrows:
        return nrows, (n + nrows - 1) // nrows
    if ncols:
        return (n + ncols - 1) // ncols, ncols
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def show_image(
    image: Union[Tensor, ImageDetail],
    spec: ImageViewSpec = ImageViewSpec(),
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    if isinstance(image, ImageDetail):
        img = image.get(ImageDetail.Keys.IMAGE)
    else:
        img = image

    rendered = render.prepare_for_display(
        img,
        spec.normalize,
        spec.percentile_low,
        spec.percentile_high,
        spec.vmin,
        spec.vmax,
        spec.gamma,
        spec.normalize_per_image,
    )

    if rendered.shape[0] == 1 and spec.colormap != "gray":
        rendered = render.apply_colormap(rendered, spec.colormap)
    else:
        rendered = render.to_rgb(rendered)

    fig, ax = _get_figure_axes(ax, display)
    _plot_tensor(ax, rendered, spec.colormap, spec.interpolation)
    _finalize_axes(ax, display)

    if spec.show_colorbar:
        import matplotlib.pyplot as plt
        cbar = plt.colorbar(ax.images[0], ax=ax)
        if spec.colorbar_label:
            cbar.set_label(spec.colorbar_label)

    _finalize_figure(fig, display)
    return ax


def show_tensor(
    tensor: Tensor,
    spec: ImageViewSpec = ImageViewSpec(),
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    return show_image(tensor, spec, display, ax)


def show_images(
    images: Sequence[Union[Tensor, ImageDetail]],
    spec: GallerySpec = GallerySpec(),
    display: DisplaySpec = DisplaySpec(),
    *,
    title: str | None = None,
):
    import matplotlib.pyplot as plt

    n = len(images)
    if n == 0:
        raise ValueError("No images to display")
    
    if title is not None:
        display = replace(display, title=title)

    nrows, ncols = _compute_grid_layout(n, spec.nrows, spec.ncols)
    if spec.preserve_relative_size:
        sizes = [_get_image_hw(img) for img in images]
        figsize, row_heights, col_widths = _compute_gallery_figsize(
            sizes, nrows, ncols, spec.per_image_size
        )
        fig = plt.figure(figsize=display.figsize or figsize, dpi=display.dpi)
        gs = fig.add_gridspec(
            nrows,
            ncols,
            width_ratios=col_widths,
            height_ratios=row_heights,
        )
        axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    else:
        figsize = display.figsize or (
            ncols * spec.per_image_size[0],
            nrows * spec.per_image_size[1],
        )
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=display.dpi)
    fig.subplots_adjust(wspace=spec.spacing, hspace=spec.spacing)

    image_spec = spec.image_spec
    if spec.share_colorbar and spec.image_spec.show_colorbar:
        image_spec = replace(spec.image_spec, show_colorbar=False)

    if not spec.preserve_relative_size:
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for i, img in enumerate(images):
        if i >= len(flat_axes):
            break
        ax = flat_axes[i]
        title = spec.titles[i] if spec.titles and i < len(spec.titles) else None
        show_image(
            img,
            image_spec,
            DisplaySpec(title=title, show=False),
            ax=ax,
        )

    for i in range(len(images), len(flat_axes)):
        flat_axes[i].axis("off")

    if spec.share_colorbar:
        first_with_image = next((ax for ax in flat_axes if ax.images), None)
        if first_with_image is not None:
            cbar = fig.colorbar(first_with_image.images[0], ax=flat_axes)
            if spec.image_spec.colorbar_label:
                cbar.set_label(spec.image_spec.colorbar_label)

    _finalize_figure(fig, display)
    return fig


def show_detail(
    detail: ImageDetail,
    keys: Sequence[str] = None,
    specs: Optional[Sequence[ImageViewSpec]] = None,
    titles: Optional[Sequence[str]] = None,
    display: DisplaySpec = DisplaySpec(),
):
    keys = keys or [ImageDetail.Keys.IMAGE]
    tensors = gallery_module.render_detail_components(detail, keys, specs)

    gallery_spec = GallerySpec(
        image_spec=ImageViewSpec(normalize="none"),
        titles=titles,
    )
    return show_images(tensors, gallery_spec, display)


def show_gradient(
    detail: ImageDetail,
    component: str = "norm",
    spec: GradientViewSpec = GradientViewSpec(),
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    rendered = gradient_module.render_gradient_component(detail, component, spec)
    rendered = render.to_uint8(rendered)

    fig, ax = _get_figure_axes(ax, display)
    _plot_tensor(ax, rendered)
    _finalize_axes(ax, display)
    _finalize_figure(fig, display)
    return ax


def show_gradients(
    detail: ImageDetail,
    spec: GradientGallerySpec = GradientGallerySpec(),
    display: DisplaySpec = DisplaySpec(),
):
    rendered = gradient_module.render_gradient_gallery(detail, spec)
    titles = [f"Gradient {c}" for c in spec.show_components]
    gallery_spec = GallerySpec(titles=titles)

    if spec.layout == "row":
        gallery_spec = GallerySpec(nrows=1, titles=titles)
    elif spec.layout == "column":
        gallery_spec = GallerySpec(ncols=1, titles=titles)
    else:
        gallery_spec = GallerySpec(titles=titles)

    return show_images(rendered, gallery_spec, display)


def show_gradient_hsv(
    detail: ImageDetail,
    per_image: bool = False,
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    rendered = gradient_module.render_gradient_hsv(detail, per_image=per_image)
    return show_image(rendered, display=display, ax=ax)



def show_mask(
    mask: Union[Tensor, ImageDetail],
    spec: MaskViewSpec = MaskViewSpec(),
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    if isinstance(mask, ImageDetail):
        m = mask.get(ImageDetail.Keys.DOMAIN.MASK)
    else:
        m = mask

    rendered = mask_module.render_mask(m, spec)

    fig, ax = _get_figure_axes(ax, display)
    _plot_tensor(ax, rendered)
    _finalize_axes(ax, display)
    _finalize_figure(fig, display)
    return ax


def show_mask_overlay(
    detail: ImageDetail,
    spec: MaskOverlaySpec = MaskOverlaySpec(),
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    rendered = mask_module.render_mask_overlay(detail, spec)

    fig, ax = _get_figure_axes(ax, display)
    _plot_tensor(ax, rendered)
    _finalize_axes(ax, display)
    _finalize_figure(fig, display)
    return ax



def show_comparison(
    reference: ImageDetail,
    moving: ImageDetail,
    mode: str = "checkerboard",
    spec: Optional[object] = None,
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    if mode == "overlay":
        spec = spec or OverlaySpec()
        rendered = composite.render_overlay(reference, moving, spec)
    elif mode == "edge_overlay":
        spec = spec or EdgeOverlaySpec()
        rendered = composite.render_edge_overlay(reference, moving, spec)
    elif mode == "checkerboard":
        spec = spec or CheckerboardSpec()
        rendered = composite.render_checkerboard(reference, moving, spec)
    elif mode == "side_by_side":
        ref_img = reference.get(ImageDetail.Keys.IMAGE)
        mov_img = moving.get(ImageDetail.Keys.IMAGE)
        gallery_spec = spec if isinstance(spec, GallerySpec) else GallerySpec()
        if gallery_spec.titles is None:
            gallery_spec = replace(gallery_spec, titles=["Reference", "Moving"])
        gallery_spec = replace(gallery_spec, nrows=1)
        return show_images(
            [ref_img, mov_img],
            gallery_spec,
            display,
        )
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")

    return show_image(rendered, display=display, ax=ax)


def show_annotated(
    detail: ImageDetail,
    image_spec: ImageViewSpec = ImageViewSpec(),
    quad_spec: Optional[QuadAnnotationSpec] = None,
    box_spec: Optional[BoxAnnotationSpec] = None,
    point_spec: Optional[PointAnnotationSpec] = None,
    display: DisplaySpec = DisplaySpec(),
    ax=None,
):
    img = detail.get(ImageDetail.Keys.IMAGE)
    rendered = render.prepare_for_display(
        img,
        image_spec.normalize,
        image_spec.percentile_low,
        image_spec.percentile_high,
        image_spec.vmin,
        image_spec.vmax,
        image_spec.gamma,
        image_spec.normalize_per_image,
    )
    rendered = render.to_rgb(rendered)

    annotated = annotate.annotate_from_detail(
        rendered, detail, quad_spec, box_spec, point_spec
    )

    fig, ax = _get_figure_axes(ax, display)
    _plot_tensor(ax, annotated)
    _finalize_axes(ax, display)
    _finalize_figure(fig, display)
    return ax
