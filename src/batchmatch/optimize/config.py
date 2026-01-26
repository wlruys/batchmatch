from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch

from batchmatch.gradient.config import GradientPipelineConfig, GradientMethodConfig
from batchmatch.metric import ImageMetricSpec
from batchmatch.warp import WarpPipelineConfig

__all__ = [
    "MetricConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizeConfig",
]


@dataclass(frozen=True)
class MetricConfig:
    spec: ImageMetricSpec
    maximize: Optional[bool] = None


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adam"
    params: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-1})
    param_groups: Optional[list[dict[str, Any]]] = None


@dataclass(frozen=True)
class SchedulerConfig:
    type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizeConfig:
    iterations: int = 300
    device: Union[str, torch.device] = "cpu"
    dtype: torch.dtype = torch.float32

    metric: MetricConfig = field(default_factory=lambda: MetricConfig(ImageMetricSpec("mse")))
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    gradient: Optional[GradientPipelineConfig | GradientMethodConfig] = None
    warp: Optional[WarpPipelineConfig] = None

    grad_clip: Optional[float] = 5.0
    progress_enabled: bool = True
    progress_transient: bool = False

    def resolve_device(self) -> torch.device:
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)
