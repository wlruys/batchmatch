from __future__ import annotations

from batchmatch.helpers.math import masked_mean, batched_quantile_with_threshold_and_counts
import torch
from batchmatch.base.pipeline import Stage, StageRegistry, _register_scalar
from batchmatch.base.tensordicts import ImageDetail, NestedKey

Tensor = torch.Tensor

process_registry = StageRegistry("process")

@process_registry.register("apply_mask", "mask_gradient")
class ApplyGradientMask(Stage):
    """
    Apply a binary mask to the gradients, zeroing out gradients where the mask is zero. (Note: deprecated, translation stages now handle this directly.)
    """
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.DOMAIN.MASK,
    })
    sets: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
    })

    def __init__(
        self,
        *,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self._mask_key = ImageDetail.Keys.DOMAIN.MASK
        self._inplace = inplace

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        mask = image.get(self._mask_key)

        if self._inplace:
            gx.mul_(mask)
            gy.mul_(mask)
        else:
            gx = gx * mask
            gy = gy * mask

        image.set(ImageDetail.Keys.GRAD.X, gx)
        image.set(ImageDetail.Keys.GRAD.Y, gy)
        return image
    
@process_registry.register("apply_window", "window_gradient")
class ApplyGradientWindow(Stage):
    """
    Apply a window function to the gradients. Note: Use is deprecated (done in translation stages directly).
    """
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.DOMAIN.WINDOW,
    })
    sets: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
    })

    def __init__(
        self,
        *,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self._window_key = ImageDetail.Keys.DOMAIN.WINDOW
        self._inplace = inplace

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        window = image.get(self._window_key)

        if self._inplace:
            gx.mul_(window)
            gy.mul_(window)
        else:
            gx = gx * window
            gy = gy * window

        image.set(ImageDetail.Keys.GRAD.X, gx)
        image.set(ImageDetail.Keys.GRAD.Y, gy)
        return image


@process_registry.register("flip", "flip_gradients")
class FlipGradients(Stage):
    """
    Flip gradient directions horizontally and/or vertically.
    """
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})

    def __init__(
        self,
        *,
        horizontal: bool = False,
        vertical: bool = False,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self._horizontal = horizontal
        self._vertical = vertical
        self._inplace = inplace
    
    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)

        if self._horizontal:
            if self._inplace:
                gx.neg_()
            else:
                gx = -gx

        if self._vertical:
            if self._inplace:
                gy.neg_()
            else:
                gy = -gy

        image.set(ImageDetail.Keys.GRAD.X, gx)
        image.set(ImageDetail.Keys.GRAD.Y, gy)
        return image

@process_registry.register("mean_whiten", "whiten")
class MeanWhitenGradient(Stage):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})

    def __init__(self, threshold: float = 1e-6, **kwargs) -> None:
        super().__init__()
        _register_scalar(self, "threshold", threshold, persistent=True)

    def forward(self, image: ImageDetail):
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)

        thr = self.threshold
        gx_mask = gx.abs() >= thr
        mean_x = masked_mean(gx, gx_mask, dim=(-2, -1), keepdim=True, threshold=1e-12)

        gy_mask = gy.abs() >= thr
        mean_y = masked_mean(gy, gy_mask, dim=(-2, -1), keepdim=True, threshold=1e-12)

        if self._inplace:
            gx.sub_(mean_x)
            gy.sub_(mean_y)
        else:
            gx = gx - mean_x
            gy = gy - mean_y

        image.update({
            ImageDetail.Keys.GRAD.X: gx,
            ImageDetail.Keys.GRAD.Y: gy,
        })
        return image
    
@process_registry.register("mean_scale")
class MeanScaleGradient(Stage):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})

    def __init__(self, scale: float = 1.0, threshold: float = 1e-6, **kwargs) -> None:
        super().__init__()
        _register_scalar(self, "scale", scale, persistent=True)
        _register_scalar(self, "threshold", threshold, persistent=True)

    def forward(self, image: ImageDetail):
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)

        thr = self.threshold
        scale = self.scale
        
        gx_mask = gx.abs() >= thr
        mean_x = masked_mean(gx.abs(), gx_mask, dim=(-2, -1), keepdim=True, threshold=1e-12)

        gy_mask = gy.abs() >= thr
        mean_y = masked_mean(gy.abs(), gy_mask, dim=(-2, -1), keepdim=True, threshold=1e-12)

        scale_x = scale / (mean_x + thr)
        scale_y = scale / (mean_y + thr)

        if self._inplace:
            gx.mul_(scale_x)
            gy.mul_(scale_y)
        else:
            gx = gx * scale_x
            gy = gy * scale_y

        image.update({
            ImageDetail.Keys.GRAD.X: gx,
            ImageDetail.Keys.GRAD.Y: gy,
        })
        return image
    
@process_registry.register("norm_scale")
class NormScaleGradient(Stage):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y, ImageDetail.Keys.GRAD.NORM})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})

    def __init__(self, scale: float = 1.0, threshold: float = 1e-6, **kwargs) -> None:
        super().__init__()
        _register_scalar(self, "scale", scale, persistent=True)
        _register_scalar(self, "threshold", threshold, persistent=True)

    def forward(self, image: ImageDetail):
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        norm = image.get(ImageDetail.Keys.GRAD.NORM)

        # Use registered buffers directly
        thr = self.threshold
        scale = self.scale
        
        norm_mask = norm >= thr
        mean_norm = masked_mean(norm, norm_mask, dim=(-2, -1), keepdim=True, threshold=1e-12)

        scale_factor = scale / (mean_norm + thr)

        if self._inplace:
            gx.mul_(scale_factor)
            gy.mul_(scale_factor)
        else:
            gx = gx * scale_factor
            gy = gy * scale_factor

        image.update({
            ImageDetail.Keys.GRAD.X: gx,
            ImageDetail.Keys.GRAD.Y: gy,
        })
        return image

@process_registry.register("complex", "complex_gradient")
class ComplexGradient(Stage):
    """Convert separate x and y gradients into a complex gradient representation."""
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.I})

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)

        grad_i = torch.complex(gx, gy)
        image.set(ImageDetail.Keys.GRAD.I, grad_i)
        return image


@process_registry.register("mask_outliers", "outlier_mask")
class MaskGradientOutliers(Stage):
    """
    Detect and zero out outlier gradients via quantile thresholding.

    Computes the specified quantile over gradient magnitudes above `magnitude_threshold`,
    then zeros pixels where magnitude exceeds `quantile_multiplier * quantile_value`.
    """
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.GRAD.NORM,
    })
    sets: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
    })

    def __init__(
        self,
        *,
        quantile: float = 0.99,
        quantile_multiplier: float = 3.0,
        magnitude_threshold: float = 1e-6,
        per_image: bool = False,
        min_valid_fraction: float = 0.01,
        seed: int | None = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self._quantile = float(quantile)
        self._quantile_multiplier = float(quantile_multiplier)
        self._magnitude_threshold = float(magnitude_threshold)
        self._per_image = per_image
        self._min_valid_fraction = float(min_valid_fraction)
        self._seed = seed
        self._inplace = inplace

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        norm = image.get(ImageDetail.Keys.GRAD.NORM)

        B, C, H, W = norm.shape
        N_spatial = H * W

        if self._per_image:
            norm_flat = norm.view(B, -1)
            N_elements = C * N_spatial
        else:
            norm_flat = norm.view(B * C, -1)
            N_elements = N_spatial

        quantile_values, valid_counts = batched_quantile_with_threshold_and_counts(
            norm_flat, self._quantile, self._magnitude_threshold, seed=self._seed
        )

        min_valid = int(self._min_valid_fraction * N_elements)
        quantile_values = torch.where(
            valid_counts >= min_valid,
            quantile_values,
            torch.full_like(quantile_values, float('inf'))
        )

        outlier_threshold = quantile_values * self._quantile_multiplier
        if self._per_image:
            outlier_threshold = outlier_threshold.view(B, 1, 1, 1)
        else:
            outlier_threshold = outlier_threshold.view(B, C, 1, 1)

        is_outlier = norm > outlier_threshold

        if self._inplace:
            gx.masked_fill_(is_outlier, 0.0)
            gy.masked_fill_(is_outlier, 0.0)
        else:
            gx = gx.masked_fill(is_outlier, 0.0)
            gy = gy.masked_fill(is_outlier, 0.0)

        image.set(ImageDetail.Keys.GRAD.X, gx)
        image.set(ImageDetail.Keys.GRAD.Y, gy)
        return image

    def extra_repr(self) -> str:
        parts = [f"q={self._quantile:.2f}", f"mult={self._quantile_multiplier:.1f}"]
        if self._per_image:
            parts.append("per_image")
        if self._min_valid_fraction != 0.01:
            parts.append(f"min_valid={self._min_valid_fraction:.0%}")
        if self._seed is not None:
            parts.append(f"seed={self._seed}")
        if self._inplace:
            parts.append("inplace")
        return ", ".join(parts)
