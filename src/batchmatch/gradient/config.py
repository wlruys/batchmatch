from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Sequence

__all__ = [
    "ApplyGradientMaskConfig",
    "ApplyGradientWindowConfig",
    "BoxRatioGradientConfig",
    "CDGradientConfig",
    "ComplexGradientConfig",
    "EtaConfig",
    "FixedEtaConfig",
    "FlipGradientsConfig",
    "MeanEtaConfig",
    "MeanScaleGradientConfig",
    "MeanWhitenGradientConfig",
    "NormConfig",
    "NormScaleGradientConfig",
    "L2NormConfig",
    "MaskGradientOutliersConfig",
    "NormalizeConfig",
    "PowerNormalizeConfig",
    "GradientStage",
    "GradientPipelineConfig",
    "GradientMethodConfig",
    "ROEWAGradientConfig",
    "SobelGradientConfig",
]

def _merge_params(explicit: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    params = dict(explicit)
    for key, value in (extras or {}).items():
        if key not in params:
            params[key] = value
    return params


def _coerce_stage(value: Any) -> Optional["GradientStage"]:
    if value is None:
        return None
    if isinstance(value, GradientStage):
        return value
    to_stage = getattr(value, "to_stage", None)
    if callable(to_stage):
        return to_stage()
    if isinstance(value, str):
        return GradientStage(type=value)
    raise TypeError(f"Expected GradientStage or config, got {type(value).__name__}")


def _coerce_stage_list(values: Optional[Sequence[Any]]) -> Optional[list["GradientStage"]]:
    if values is None:
        return None
    return [_coerce_stage(v) for v in values]  # type: ignore[list-item]


@dataclass
class NormConfig:
    """Config for gradient norm stages."""

    type: str = "l2"
    params: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> "GradientStage":
        return GradientStage(type=self.type, params=_merge_params(self.params, self.extras))


@dataclass
class L2NormConfig:
    """Typed config for L2 gradient norm."""

    eps: float = 1e-8
    inplace: bool = False
    output_key: Optional[str] = None
    squared: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> "GradientStage":
        params = {
            "eps": self.eps,
            "inplace": self.inplace,
            "squared": self.squared,
        }
        if self.output_key is not None:
            params["output_key"] = self.output_key
        return GradientStage(type="l2", params=_merge_params(params, self.extras))


class EtaConfig:

    @staticmethod
    def from_mean(
        *,
        scale: float = 0.2,
        threshold: float = 1e-6,
        norm: NormConfig | L2NormConfig | GradientStage | str | None = None,
        **extras: Any,
    ) -> "MeanEtaConfig":
        return MeanEtaConfig(scale=scale, threshold=threshold, norm=norm, extras=extras)

    @staticmethod
    def from_fixed(
        *,
        eta: float = 1e-3,
        **extras: Any,
    ) -> "FixedEtaConfig":
        return FixedEtaConfig(eta=eta, extras=extras)


@dataclass
class MeanEtaConfig:

    scale: float = 0.2
    threshold: float = 1e-6
    norm: NormConfig | L2NormConfig | GradientStage | str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> "GradientStage":
        params = {
            "scale": self.scale,
            "threshold": self.threshold,
        }
        return GradientStage(
            type="mean",
            params=_merge_params(params, self.extras),
            norm=_coerce_stage(self.norm),
        )


@dataclass
class FixedEtaConfig:

    eta: float = 1e-3
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> "GradientStage":
        params = {"eta": self.eta}
        return GradientStage(type="fixed", params=_merge_params(params, self.extras))


@dataclass
class NormalizeConfig:

    threshold: float = 1e-3
    inplace: bool = False
    output_keys: Optional[tuple[str, str]] = None
    norm: NormConfig | L2NormConfig | GradientStage | str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> "GradientStage":
        params = {
            "threshold": self.threshold,
            "inplace": self.inplace,
        }
        if self.output_keys is not None:
            params["output_keys"] = self.output_keys
        return GradientStage(
            type="normalize",
            params=_merge_params(params, self.extras),
            norm=_coerce_stage(self.norm),
        )


@dataclass
class PowerNormalizeConfig:

    power: float = 1.0
    inplace: bool = False
    output_keys: Optional[tuple[str, str]] = None
    norm: NormConfig | L2NormConfig | GradientStage | str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> "GradientStage":
        params = {
            "power": self.power,
            "inplace": self.inplace,
        }
        if self.output_keys is not None:
            params["output_keys"] = self.output_keys
        return GradientStage(
            type="power_normalize",
            params=_merge_params(params, self.extras),
            norm=_coerce_stage(self.norm),
        )


@dataclass
class GradientStage:

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    norm: Optional["GradientStage"] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def _params_dict(self) -> dict[str, Any]:
        params = dict(self.extras or {})
        params.update(self.params or {})
        return params

    def to_spec_mapping(self, *, include_norm: bool = True) -> dict[str, Any]:
        mapping: dict[str, Any] = {"type": self.type}
        mapping.update(self._params_dict())
        if include_norm and self.norm is not None:
            return {
                "stage": mapping,
                "norm": self.norm.to_spec_mapping(include_norm=False),
            }
        return mapping


@dataclass
class GradientPipelineConfig:

    gradient: GradientStage
    preprocess: Optional[list[GradientStage]] = None
    eta: Optional[GradientStage] = None
    normalize: Optional[GradientStage] = None
    postprocess: Optional[list[GradientStage]] = None
    build_complex: bool = False

    def __post_init__(self) -> None:
        self.gradient = _coerce_stage(self.gradient)  # type: ignore[assignment]
        self.preprocess = _coerce_stage_list(self.preprocess)
        self.eta = _coerce_stage(self.eta)
        self.normalize = _coerce_stage(self.normalize)
        self.postprocess = _coerce_stage_list(self.postprocess)

    def to_spec_dict(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "gradient": self.gradient.to_spec_mapping(),
        }
        if self.preprocess:
            spec["preprocess"] = [stage.to_spec_mapping() for stage in self.preprocess]
        if self.eta is not None:
            spec["eta"] = self.eta.to_spec_mapping()
        if self.normalize is not None:
            spec["normalize"] = self.normalize.to_spec_mapping()
        if self.postprocess:
            spec["postprocess"] = [stage.to_spec_mapping() for stage in self.postprocess]
        return spec

    def to_method_and_params(self) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = dict(self.gradient._params_dict())
        if self.preprocess:
            params["preprocess"] = [stage.to_spec_mapping() for stage in self.preprocess]
        if self.eta is not None:
            params["eta"] = self.eta.to_spec_mapping()
        if self.normalize is not None:
            params["normalize"] = self.normalize.to_spec_mapping()
        if self.postprocess:
            params["postprocess"] = [stage.to_spec_mapping() for stage in self.postprocess]
        if self.build_complex:
            params["build_complex"] = True
        return self.gradient.type, params


@dataclass
class GradientMethodConfig:
    """Typed config for a named gradient method plus pipeline stages."""

    METHOD: ClassVar[str] = ""

    params: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    preprocess: Optional[list[Any]] = None
    eta: Optional[Any] = None
    normalize: Optional[Any] = None
    postprocess: Optional[list[Any]] = None
    build_complex: bool = False

    def _gradient_stage(self) -> GradientStage:
        return GradientStage(type=self.METHOD, params=_merge_params(self.params, self.extras))

    def to_pipeline_config(self) -> GradientPipelineConfig:
        return GradientPipelineConfig(
            gradient=self._gradient_stage(),
            preprocess=_coerce_stage_list(self.preprocess),
            eta=_coerce_stage(self.eta),
            normalize=_coerce_stage(self.normalize),
            postprocess=_coerce_stage_list(self.postprocess),
            build_complex=self.build_complex,
        )

    def to_method_and_params(self) -> tuple[str, dict[str, Any]]:
        return self.to_pipeline_config().to_method_and_params()


@dataclass
class CDGradientConfig(GradientMethodConfig):
    METHOD: ClassVar[str] = "cd"
    stencil_width: int = 3

    def _gradient_stage(self) -> GradientStage:
        params = {"stencil_width": self.stencil_width}
        return GradientStage(type=self.METHOD, params=_merge_params(params, self.extras))


@dataclass
class SobelGradientConfig(GradientMethodConfig):
    METHOD: ClassVar[str] = "sobel"


@dataclass
class BoxRatioGradientConfig(GradientMethodConfig):
    METHOD: ClassVar[str] = "box_ratio"
    width: int = 11
    eps: float = 1e-6
    mode: str = "sym"
    require_positive: bool = False
    padding_mode: str = "zeros"

    def _gradient_stage(self) -> GradientStage:
        params = {
            "width": self.width,
            "eps": self.eps,
            "mode": self.mode,
            "require_positive": self.require_positive,
            "padding_mode": self.padding_mode,
        }
        return GradientStage(type=self.METHOD, params=_merge_params(params, self.extras))


@dataclass
class ROEWAGradientConfig(GradientMethodConfig):
    METHOD: ClassVar[str] = "roewa"

    alpha: float = 0.9
    eps: float = 1e-12
    mode: str = "log"
    orthogonal_smooth: bool = True

    def _gradient_stage(self) -> GradientStage:
        params = {
            "alpha": self.alpha,
            "eps": self.eps,
            "mode": self.mode,
            "orthogonal_smooth": self.orthogonal_smooth,
        }
        return GradientStage(type=self.METHOD, params=_merge_params(params, self.extras))


@dataclass
class ApplyGradientMaskConfig:
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {"inplace": self.inplace}
        return GradientStage(type="apply_mask", params=_merge_params(params, self.extras))


@dataclass
class ApplyGradientWindowConfig:
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {"inplace": self.inplace}
        return GradientStage(type="apply_window", params=_merge_params(params, self.extras))


@dataclass
class FlipGradientsConfig:
    horizontal: bool = False
    vertical: bool = False
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {
            "horizontal": self.horizontal,
            "vertical": self.vertical,
            "inplace": self.inplace,
        }
        return GradientStage(type="flip", params=_merge_params(params, self.extras))


@dataclass
class MeanWhitenGradientConfig:
    threshold: float = 1e-6
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {
            "threshold": self.threshold,
            "inplace": self.inplace,
        }
        return GradientStage(type="mean_whiten", params=_merge_params(params, self.extras))


@dataclass
class MeanScaleGradientConfig:
    scale: float = 1.0
    threshold: float = 1e-6
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {
            "scale": self.scale,
            "threshold": self.threshold,
            "inplace": self.inplace,
        }
        return GradientStage(type="mean_scale", params=_merge_params(params, self.extras))


@dataclass
class NormScaleGradientConfig:
    scale: float = 1.0
    threshold: float = 1e-6
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {
            "scale": self.scale,
            "threshold": self.threshold,
            "inplace": self.inplace,
        }
        return GradientStage(type="norm_scale", params=_merge_params(params, self.extras))


@dataclass
class ComplexGradientConfig:
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        return GradientStage(type="complex", params=_merge_params({}, self.extras))


@dataclass
class MaskGradientOutliersConfig:
    quantile: float = 0.99
    quantile_multiplier: float = 3.0
    magnitude_threshold: float = 1e-6
    per_image: bool = False
    min_valid_fraction: float = 0.01
    seed: int | None = None
    inplace: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_stage(self) -> GradientStage:
        params = {
            "quantile": self.quantile,
            "quantile_multiplier": self.quantile_multiplier,
            "magnitude_threshold": self.magnitude_threshold,
            "per_image": self.per_image,
            "min_valid_fraction": self.min_valid_fraction,
            "seed": self.seed,
            "inplace": self.inplace,
        }
        return GradientStage(type="mask_outliers", params=_merge_params(params, self.extras))
