from __future__ import annotations

from batchmatch.optimize.config import (
    MetricConfig,
    OptimizerConfig,
    SchedulerConfig,
    OptimizeConfig,
)
from batchmatch.optimize.optimizer import AffineWarpOptimize, OptimizationStats
from batchmatch.optimize.pipeline import build_reference_pipeline, build_moving_pipeline
from batchmatch.optimize.result import OptimizeResult

__all__ = [
    "MetricConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizeConfig",
    "AffineWarpOptimize",
    "OptimizationStats",
    "OptimizeResult",
    "build_reference_pipeline",
    "build_moving_pipeline",
]

#Note(wlr): This is very experimental and incomplete. Its mainly a test of how well torch autograd can recover affine parameters
#TODO(wlr): Needs more work to be generally useful. Really needs regulairization and multi-scale optimization