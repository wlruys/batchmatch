"""ImageMetric implementations for computing similarity between ImageDetail pairs."""

from __future__ import annotations

from typing import Optional

import torch

from batchmatch.base.tensordicts import ImageDetail

from .base import ImageMetric, register_image_metric
from .functions import (
    mse,
    mae,
    cross_correlation,
    local_cross_correlation,
    normalized_gradient_fields,
    mutual_information,
    nmi,
    soft_mutual_information,
    soft_nmi,
    combine_detail_masks,
)

Tensor = torch.Tensor

__all__ = [
    "MeanSquaredErrorMetric",
    "MeanAbsoluteErrorMetric",
    "NormalizedCrossCorrelationMetric",
    "LocalNormalizedCrossCorrelationMetric",
    "NormalizedGradientFieldsMetric",
    "DifferentiableMutualInformationMetric",
    "DifferentiableNormalizedMutualInformationMetric",
    "MutualInformationMetric",
    "NormalizedMutualInformationMetric",
]

def _ensure_matching_image_shape(
    reference: ImageDetail,
    moving: ImageDetail,
    metric_name: str,
) -> None:
    """Validate that reference and moving images have matching shapes."""
    if reference.image.shape != moving.image.shape:
        raise ValueError(
            f"{metric_name} requires images with matching shape. "
            f"Got {tuple(reference.image.shape)} vs {tuple(moving.image.shape)}."
        )


def _require_detail_field(
    detail: ImageDetail,
    attr: str,
    metric_name: str,
) -> Tensor:
    """Require that an ImageDetail has a specific field populated."""
    value = getattr(detail, attr, None)
    if value is None:
        raise ValueError(f"{metric_name} requires ImageDetail.{attr} to be populated.")
    return value


def _detail_mask(reference: ImageDetail, moving: ImageDetail) -> Optional[Tensor]:
    """Get combined mask from reference and moving ImageDetails."""
    return combine_detail_masks(
        reference.image,
        reference.mask,
        moving.image,
        moving.mask,
    )


@register_image_metric("mse", help="Mean squared error over intensity images.")
class MeanSquaredErrorMetric(ImageMetric):
    """
    Mean Squared Error metric.
    """

    name = "MeanSquaredError"
    differentiable = True
    maximize = False
    reduction: str

    def __init__(self, *, reduction: str = "none"):
        super().__init__()
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return mse(reference.image, moving.image, reduction=self.reduction, mask=mask)


@register_image_metric("mae", help="Mean absolute error (L1 loss) over intensity images.")
class MeanAbsoluteErrorMetric(ImageMetric):
    """
    Mean Absolute Error (L1) metric.
    """

    name = "MeanAbsoluteError"
    differentiable = True
    maximize = False
    reduction: str

    def __init__(self, *, reduction: str = "none"):
        super().__init__()
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return mae(reference.image, moving.image, reduction=self.reduction, mask=mask)


@register_image_metric("ncc", help="Normalized cross-correlation over intensity images.")
class NormalizedCrossCorrelationMetric(ImageMetric):
    """
    Normalized Cross-Correlation metric.
    """

    name = "NormalizedCrossCorrelation"
    differentiable = True
    maximize = True
    reduction: str

    def __init__(self, *, reduction: str = "none"):
        super().__init__()
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return cross_correlation(
            reference.image, moving.image, reduction=self.reduction, mask=mask
        )


@register_image_metric(
    "local_ncc", help="Local normalized cross-correlation for multi-modal registration."
)
class LocalNormalizedCrossCorrelationMetric(ImageMetric):
    """
    Locallly Normalized Cross-Correlation metric.
    """

    name = "LocalNormalizedCrossCorrelation"
    differentiable = True
    maximize = True
    window_size: int
    window_type: str
    sigma: float
    reduction: str

    def __init__(
        self,
        *,
        window_size: int = 9,
        window_type: str = "gaussian",
        sigma: float = 1.5,
        reduction: str = "none",
    ):
        super().__init__()
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "window_type", window_type)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return local_cross_correlation(
            reference.image,
            moving.image,
            window_size=self.window_size,
            window_type=self.window_type,
            sigma=self.sigma,
            reduction=self.reduction,
            mask=mask,
        )

@register_image_metric("ngf", help="Normalized gradient fields metric on gx/gy.")
class NormalizedGradientFieldsMetric(ImageMetric):
    """
    Normalized Gradient Fields metric.
    """

    name = "NormalizedGradientFields"
    differentiable = True
    requires_gradients = True
    maximize = False  # NGF returns 1 - cos², which is 0 when aligned
    reduction: str
    eps: float

    def __init__(self, *, reduction: str = "none", eps: float = 1e-12):
        super().__init__()
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "eps", eps)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        ref_gx = _require_detail_field(reference, "gx", self.name)
        ref_gy = _require_detail_field(reference, "gy", self.name)
        mov_gx = _require_detail_field(moving, "gx", self.name)
        mov_gy = _require_detail_field(moving, "gy", self.name)
        mask = _detail_mask(reference, moving)
        return normalized_gradient_fields(
            (ref_gx, ref_gy),
            (mov_gx, mov_gy),
            eps=self.eps,
            reduction=self.reduction,
            mask=mask,
        )


@register_image_metric("diff_mi", help="Differentiable mutual information using soft binning.")
class DifferentiableMutualInformationMetric(ImageMetric):
    """
    Differentiable Mutual Information using soft binning.
    """

    name = "DifferentiableMutualInformation"
    differentiable = True
    maximize = True
    bins: int
    bandwidth: float
    reduction: str

    def __init__(
        self, *, bins: int = 64, bandwidth: float = 0.2, reduction: str = "none"
    ):
        super().__init__()
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return soft_mutual_information(
            reference.image,
            moving.image,
            bins=self.bins,
            bandwidth=self.bandwidth,
            reduction=self.reduction,
            mask=mask,
        )


@register_image_metric(
    "diff_nmi", help="Differentiable normalized mutual information using soft binning."
)
class DifferentiableNormalizedMutualInformationMetric(ImageMetric):
    """
    Differentiable Normalized Mutual Information using soft binning.
    """

    name = "DifferentiableNormalizedMutualInformation"
    differentiable = True
    maximize = True
    bins: int
    bandwidth: float
    reduction: str

    def __init__(
        self, *, bins: int = 64, bandwidth: float = 0.2, reduction: str = "none"
    ):
        super().__init__()
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return soft_nmi(
            reference.image,
            moving.image,
            bins=self.bins,
            bandwidth=self.bandwidth,
            reduction=self.reduction,
            mask=mask,
        )

@register_image_metric("mi", help="Mutual information between intensity images.")
class MutualInformationMetric(ImageMetric):
    """
    Mutual Information metric using hard binning.
    """

    name = "MutualInformation"
    differentiable = False
    maximize = True
    bins: int
    reduction: str

    def __init__(self, *, bins: int = 256, reduction: str = "none"):
        super().__init__()
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return mutual_information(
            reference.image,
            moving.image,
            bins=self.bins,
            reduction=self.reduction,
            mask=mask,
        )


@register_image_metric("nmi", help="Normalized mutual information between intensity images.")
class NormalizedMutualInformationMetric(ImageMetric):
    """
    Normalized Mutual Information metric using hard binning.
    """

    name = "NormalizedMutualInformation"
    differentiable = False
    maximize = True
    bins: int
    reduction: str

    def __init__(self, *, bins: int = 256, reduction: str = "none"):
        super().__init__()
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "reduction", reduction)

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        _ensure_matching_image_shape(reference, moving, self.name)
        mask = _detail_mask(reference, moving)
        return nmi(
            reference.image,
            moving.image,
            bins=self.bins,
            reduction=self.reduction,
            mask=mask,
        )
