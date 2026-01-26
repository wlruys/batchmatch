from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

import torch
import torch.nn as nn

from batchmatch.base.tensordicts import ImageDetail

Tensor = torch.Tensor

__all__ = [
    "ImageMetric",
    "ImageMetricEntry",
    "ImageMetricSpec",
    "register_image_metric",
    "build_image_metric",
    "available_image_metrics",
]


@dataclass(frozen=True)
class ImageMetricEntry:
    key: str
    cls: Type["ImageMetric"]
    help: str = ""


_IMAGE_METRIC_REGISTRY: Dict[str, ImageMetricEntry] = {}


def register_image_metric(key: str, *, help: str = ""):
    key_l = key.lower()

    def decorator(cls: Type["ImageMetric"]) -> Type["ImageMetric"]:
        if key_l in _IMAGE_METRIC_REGISTRY:
            raise KeyError(f"ImageMetric '{key_l}' already registered")

        entry = ImageMetricEntry(key=key_l, cls=cls, help=help)
        _IMAGE_METRIC_REGISTRY[key_l] = entry
        setattr(cls, "registry_key", key_l)
        return cls

    return decorator


@dataclass
class ImageMetricSpec:
    name: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> "ImageMetric":
        return build_image_metric(self)


def build_image_metric(spec: ImageMetricSpec) -> "ImageMetric":
    key = spec.name.lower()
    try:
        entry = _IMAGE_METRIC_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_IMAGE_METRIC_REGISTRY)) or "<empty>"
        raise ValueError(
            f"Unknown ImageMetric '{spec.name}'. Available: {available}"
        ) from exc

    kwargs = dict(spec.extra)
    return entry.cls(**kwargs)


def available_image_metrics() -> Dict[str, ImageMetricEntry]:
    return dict(_IMAGE_METRIC_REGISTRY)


class ImageMetric(nn.Module):
    name: str = "ImageMetric"
    differentiable: bool = True
    requires_gradients: bool = False
    requires_complex_gradients: bool = False
    maximize: bool = False

    def __init__(self):
        super().__init__()

    def compute(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        raise NotImplementedError

    def forward(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        return self.compute(reference, moving)
