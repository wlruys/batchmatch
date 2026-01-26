from __future__ import annotations

from abc import ABC
from typing import Optional, Tuple
from batchmatch.helpers.math import masked_mean, safe_divide, safe_divide_
import torch
from batchmatch.base.pipeline import Stage, StageRegistry, _register_scalar
from batchmatch.base.tensordicts import ImageDetail, NestedKey

Tensor = torch.Tensor

eta_registry = StageRegistry("eta")
normalize_registry = StageRegistry("normalize")


class ComputeEtaBase(Stage, ABC):
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.ETA})

    def __init__(self) -> None:
        super().__init__()


@eta_registry.register("fixed")
class FixedEtaStage(ComputeEtaBase):
    def __init__(
        self,
        *,
        eta: float = 1e-3,
    ) -> None:
        super().__init__()
        self._eta_value = float(eta)

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        B, C, H, W = gx.shape
        eta_tensor = gx.new_full((1, 1, 1, 1), self._eta_value).expand(B, C, 1, 1)
        image.set(ImageDetail.Keys.GRAD.ETA, eta_tensor)
        return image


@eta_registry.register("mean")
class MeanEta(ComputeEtaBase):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.NORM})

    def __init__(self, scale: float = 0.20, threshold: float = 1e-6, **kwargs) -> None:
        super().__init__()
        _register_scalar(self, "scale", scale, persistent=True)
        _register_scalar(self, "threshold", threshold, persistent=True)

    def forward(self, image: ImageDetail) -> ImageDetail:
        norm = image.get(ImageDetail.Keys.GRAD.NORM)
        thr = self.threshold
        scale = self.scale
        mean_norm = masked_mean(norm, norm >= thr, dim=(-2, -1), keepdim=False, threshold=1e-12)
        mean_norm = mean_norm.unsqueeze(-1).unsqueeze(-1)
        image.set(ImageDetail.Keys.GRAD.ETA, scale * mean_norm)
        return image


@normalize_registry.register("normalize")
class NormalizeGradient(Stage):
    """
    Normalize gradients by their norm: `gx = gx / norm`, `gy = gy / norm`.
    """
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.GRAD.NORM,
    })
    invalidates: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.NORM,
        ImageDetail.Keys.GRAD.ETA,
    })

    def __init__(
        self,
        *,
        inplace: bool = False,
        output_keys: Optional[Tuple[NestedKey, NestedKey]] = None,
        threshold: float = 1e-3,
    ) -> None:
        super().__init__()
        self._inplace = inplace
        _register_scalar(self, "threshold", threshold, persistent=True)
        if output_keys is None:
            self._out_gx_key = ImageDetail.Keys.GRAD.X
            self._out_gy_key = ImageDetail.Keys.GRAD.Y
        else:
            self._out_gx_key = output_keys[0]
            self._out_gy_key = output_keys[1]
            self._out_keys = [self._out_gx_key, self._out_gy_key]

    def _apply_normalization(self, gx: Tensor, gy: Tensor, norm: Tensor, threshold: float = 1e-8) -> Tuple[Tensor, Tensor]:
        if self._inplace:
            safe_divide_(gx, norm, threshold)
            safe_divide_(gy, norm, threshold)
            return gx, gy
        else:
            normed_gx = safe_divide(gx, norm, threshold)
            normed_gy = safe_divide(gy, norm, threshold)
            return normed_gx, normed_gy

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        norm = image.get(ImageDetail.Keys.GRAD.NORM)

        normed_gx, normed_gy = self._apply_normalization(gx, gy, norm, threshold=self.threshold)

        image.set(self._out_gx_key, normed_gx)
        image.set(self._out_gy_key, normed_gy)
        return image

@normalize_registry.register("power_normalize")
class PowerNormalizeGradient(NormalizeGradient):
    """
    Normalize gradients by a powered norm. gx = gx / norm**power, gy = gy / norm**power.
    """
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.GRAD.NORM,
    })
    invalidates: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.NORM,
        ImageDetail.Keys.GRAD.ETA,
    })

    def __init__(
        self,
        *,
        power: float = 1.0,
        inplace: bool = False,
        output_keys: Optional[Tuple[NestedKey, NestedKey]] = None,
    ) -> None:
        super().__init__()
        self._power = power
        self._inplace = inplace
        if output_keys is None:
            self._out_gx_key = ImageDetail.Keys.GRAD.X
            self._out_gy_key = ImageDetail.Keys.GRAD.Y
        else:
            self._out_gx_key = output_keys[0]
            self._out_gy_key = output_keys[1]
            self._out_keys = [self._out_gx_key, self._out_gy_key]

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        norm = image.get(ImageDetail.Keys.GRAD.NORM)

        norm_power = norm.pow(self._power)

        normed_gx, normed_gy = self._apply_normalization(gx, gy, norm_power)

        image.set(self._out_gx_key, normed_gx)
        image.set(self._out_gy_key, normed_gy)
        return image



def build_eta_operator(eta_type: str, **kwargs) -> ComputeEtaBase:
    return eta_registry.build(eta_type, **kwargs)


def build_gradient_normalize_operator(norm_type: str, **kwargs) -> Stage:
    return normalize_registry.build(norm_type, **kwargs)
