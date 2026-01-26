from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

__all__ = [
    "TranslationConfig",
    "CCTranslationConfig",
    "MeanCCTranslationConfig",
    "GCCTranslationConfig",
    "NCCTranslationConfig",
    "PCTranslationConfig",
    "GPCTranslationConfig",
    "NGFTranslationConfig",
    "GNGFTranslationConfig",
]


def _validate_overlap_fraction(value: Optional[float]) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"overlap_fraction must be in [0, 1], got {value}")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass
class TranslationConfig:
    save_surface: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    METHOD: ClassVar[str] = ""

    def _explicit_params(self) -> dict[str, Any]:
        return {}

    def _merge_params(self, explicit: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"save_surface": self.save_surface}
        for key, value in explicit.items():
            if value is not None:
                params[key] = value
        extras = self.extras or {}
        for key, value in extras.items():
            if key not in params:
                params[key] = value
        return params

    def to_method_and_params(self) -> tuple[str, dict[str, Any]]:
        return self.METHOD, self._merge_params(self._explicit_params())


@dataclass
class CCTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "cc"

    overlap_fraction: Optional[float] = None
    mean_centered: bool = False

    def __post_init__(self) -> None:
        _validate_overlap_fraction(self.overlap_fraction)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "overlap_fraction": self.overlap_fraction,
            "mean_centered": self.mean_centered,
        }


@dataclass
class MeanCCTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "mean_cc"

    overlap_fraction: Optional[float] = None
    mean_centered: bool = True

    def __post_init__(self) -> None:
        _validate_overlap_fraction(self.overlap_fraction)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "overlap_fraction": self.overlap_fraction,
            "mean_centered": self.mean_centered,
        }


@dataclass
class GCCTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "gcc"

    overlap_fraction: Optional[float] = 0.99

    def __post_init__(self) -> None:
        _validate_overlap_fraction(self.overlap_fraction)

    def _explicit_params(self) -> dict[str, Any]:
        return {"overlap_fraction": self.overlap_fraction}


@dataclass
class NCCTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "ncc"

    overlap_fraction: Optional[float] = 0.99
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _validate_overlap_fraction(self.overlap_fraction)
        _validate_positive("eps", self.eps)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "overlap_fraction": self.overlap_fraction,
            "eps": self.eps,
        }


@dataclass
class PCTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "pc"

    eps: float = 1e-8
    skip_fftshift: bool = True

    def __post_init__(self) -> None:
        _validate_positive("eps", self.eps)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "skip_fftshift": self.skip_fftshift,
        }


@dataclass
class GPCTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "gpc"

    eps: float = 1e-8
    skip_fftshift: bool = True
    p: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive("eps", self.eps)
        _validate_positive("p", self.p)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "skip_fftshift": self.skip_fftshift,
            "p": self.p,
        }


@dataclass
class NGFTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "ngf"

    overlap_fraction: Optional[float] = 0.99
    weight_by_gradient_norm: bool = False
    gradient_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        _validate_overlap_fraction(self.overlap_fraction)
        _validate_positive("gradient_norm_eps", self.gradient_norm_eps)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "overlap_fraction": self.overlap_fraction,
            "weight_by_gradient_norm": self.weight_by_gradient_norm,
            "gradient_norm_eps": self.gradient_norm_eps,
        }


@dataclass
class GNGFTranslationConfig(TranslationConfig):
    METHOD: ClassVar[str] = "gngf"

    overlap_fraction: Optional[float] = 0.99
    p: int = 2

    def __post_init__(self) -> None:
        _validate_overlap_fraction(self.overlap_fraction)
        _validate_positive_int("p", self.p)

    def _explicit_params(self) -> dict[str, Any]:
        return {
            "overlap_fraction": self.overlap_fraction,
            "p": self.p,
        }
