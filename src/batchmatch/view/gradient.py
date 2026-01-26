from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail
from .config import GradientViewSpec, GradientGallerySpec
from . import render

__all__ = [
    "render_gradient_component",
    "render_gradient_gallery",
    "render_gradient_hsv",
    "render_gradient_with_quiver",
    "get_quiver_data",
]


def render_gradient_component(
    detail: ImageDetail,
    component: str,
    spec: GradientViewSpec = GradientViewSpec(),
) -> Tensor:
    gx = detail.get(ImageDetail.Keys.GRAD.X)
    gy = detail.get(ImageDetail.Keys.GRAD.Y)

    if spec.normalize_per_image:
        gx = render.to_bchw(gx)
        gy = render.to_bchw(gy)
        if gx.shape[1] > 1:
            gx = gx[:, :1]
        if gy.shape[1] > 1:
            gy = gy[:, :1]
    else:
        gx = render.to_chw(gx)
        gy = render.to_chw(gy)
        if gx.shape[0] > 1:
            gx = gx[:1]
        if gy.shape[0] > 1:
            gy = gy[:1]

    if component == "x":
        grad = gx
        if spec.symmetric_range:
            grad = render.normalize_symmetric(grad, per_image=spec.normalize_per_image)
            grad = grad * 0.5 + 0.5
        else:
            grad = render.normalize_for_display(
                grad, spec.normalize, per_image=spec.normalize_per_image
            )
        grad = render.to_chw(grad)
        return render.apply_colormap(grad, spec.signed_colormap)

    elif component == "y":
        grad = gy
        if spec.symmetric_range:
            grad = render.normalize_symmetric(grad, per_image=spec.normalize_per_image)
            grad = grad * 0.5 + 0.5
        else:
            grad = render.normalize_for_display(
                grad, spec.normalize, per_image=spec.normalize_per_image
            )
        grad = render.to_chw(grad)
        return render.apply_colormap(grad, spec.signed_colormap)

    elif component == "norm":
        norm = render.gradient_magnitude(gx, gy)
        norm = render.normalize_for_display(
            norm, spec.normalize, per_image=spec.normalize_per_image
        )
        norm = render.apply_gamma(norm, spec.norm_gamma)
        norm = render.to_chw(norm)
        return render.apply_colormap(norm, spec.norm_colormap)

    elif component == "orientation":
        orientation = render.gradient_orientation(gx, gy)
        if spec.orientation_as_color:
            magnitude = render.gradient_magnitude(gx, gy)
            if spec.normalize_per_image:
                magnitude = render.normalize_for_display(
                    magnitude, "minmax", per_image=True
                )
            orientation = render.to_chw(orientation)
            magnitude = render.to_chw(magnitude)
            return render.orientation_to_rgb(orientation, magnitude)
        else:
            orientation_norm = orientation / (2 * math.pi)
            orientation_norm = render.to_chw(orientation_norm)
            return render.apply_cyclic_colormap(orientation_norm, spec.orientation_colormap)

    else:
        raise ValueError(f"Unknown gradient component: {component}")


def render_gradient_gallery(
    detail: ImageDetail,
    spec: GradientGallerySpec = GradientGallerySpec(),
) -> list[Tensor]:
    results = []
    for comp in spec.show_components:
        comp_spec = GradientViewSpec()
        if spec.component_specs and comp in spec.component_specs:
            comp_spec = spec.component_specs[comp]
        results.append(render_gradient_component(detail, comp, comp_spec))
    return results


def render_gradient_hsv(detail: ImageDetail, per_image: bool = False) -> Tensor:
    gx = detail.get(ImageDetail.Keys.GRAD.X)
    gy = detail.get(ImageDetail.Keys.GRAD.Y)
    return render.gradient_to_rgb(gx, gy, mode="hsv", per_image=per_image)


def render_gradient_with_quiver(
    detail: ImageDetail,
    spec: GradientViewSpec = GradientViewSpec(),
) -> Tensor:
    gx = detail.get(ImageDetail.Keys.GRAD.X)
    gy = detail.get(ImageDetail.Keys.GRAD.Y)

    if spec.normalize_per_image:
        gx = render.to_bchw(gx)
        gy = render.to_bchw(gy)
        if gx.shape[1] > 1:
            gx = gx[:, :1]
        if gy.shape[1] > 1:
            gy = gy[:, :1]
    else:
        gx = render.to_chw(gx)
        gy = render.to_chw(gy)
        if gx.shape[0] > 1:
            gx = gx[:1]
        if gy.shape[0] > 1:
            gy = gy[:1]

    norm = render.gradient_magnitude(gx, gy)
    norm = render.normalize_for_display(
        norm, spec.normalize, per_image=spec.normalize_per_image
    )
    norm = render.apply_gamma(norm, spec.norm_gamma)
    norm = render.to_chw(norm)
    return render.apply_colormap(norm, spec.norm_colormap)


def get_quiver_data(
    detail: ImageDetail,
    step: int = 16,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    gx = detail.get(ImageDetail.Keys.GRAD.X)
    gy = detail.get(ImageDetail.Keys.GRAD.Y)

    gx = render.to_chw(gx)
    gy = render.to_chw(gy)

    if gx.shape[0] > 1:
        gx = gx[:1]
    if gy.shape[0] > 1:
        gy = gy[:1]

    _, H, W = gx.shape

    y_coords = torch.arange(step // 2, H, step)
    x_coords = torch.arange(step // 2, W, step)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    U = gx[0, y_coords][:, x_coords] * scale
    V = gy[0, y_coords][:, x_coords] * scale

    return X.cpu(), Y.cpu(), U.cpu(), V.cpu()


def render_gradient_from_tensors(
    gx: Tensor,
    gy: Tensor,
    component: str = "norm",
    spec: GradientViewSpec = GradientViewSpec(),
) -> Tensor:
    if spec.normalize_per_image:
        gx = render.to_bchw(gx)
        gy = render.to_bchw(gy)
        if gx.shape[1] > 1:
            gx = gx[:, :1]
        if gy.shape[1] > 1:
            gy = gy[:, :1]
    else:
        gx = render.to_chw(gx)
        gy = render.to_chw(gy)
        if gx.shape[0] > 1:
            gx = gx[:1]
        if gy.shape[0] > 1:
            gy = gy[:1]

    if component == "x":
        grad = gx
        if spec.symmetric_range:
            grad = render.normalize_symmetric(
                grad, per_image=spec.normalize_per_image
            ) * 0.5 + 0.5
        else:
            grad = render.normalize_for_display(
                grad, spec.normalize, per_image=spec.normalize_per_image
            )
        grad = render.to_chw(grad)
        return render.apply_colormap(grad, spec.signed_colormap)

    elif component == "y":
        grad = gy
        if spec.symmetric_range:
            grad = render.normalize_symmetric(
                grad, per_image=spec.normalize_per_image
            ) * 0.5 + 0.5
        else:
            grad = render.normalize_for_display(
                grad, spec.normalize, per_image=spec.normalize_per_image
            )
        grad = render.to_chw(grad)
        return render.apply_colormap(grad, spec.signed_colormap)

    elif component == "norm":
        norm = render.gradient_magnitude(gx, gy)
        norm = render.normalize_for_display(
            norm, spec.normalize, per_image=spec.normalize_per_image
        )
        norm = render.apply_gamma(norm, spec.norm_gamma)
        norm = render.to_chw(norm)
        return render.apply_colormap(norm, spec.norm_colormap)

    elif component == "orientation":
        orientation = render.gradient_orientation(gx, gy)
        if spec.orientation_as_color:
            magnitude = render.gradient_magnitude(gx, gy)
            if spec.normalize_per_image:
                magnitude = render.normalize_for_display(
                    magnitude, "minmax", per_image=True
                )
            orientation = render.to_chw(orientation)
            magnitude = render.to_chw(magnitude)
            return render.orientation_to_rgb(orientation, magnitude)
        else:
            orientation_norm = orientation / (2 * math.pi)
            orientation_norm = render.to_chw(orientation_norm)
            return render.apply_cyclic_colormap(orientation_norm, spec.orientation_colormap)

    else:
        raise ValueError(f"Unknown gradient component: {component}")
