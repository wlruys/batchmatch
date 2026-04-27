"""Gradient-based affine registration via PyTorch autograd.

Experimental: this module is primarily a test of how well torch autograd
can recover affine parameters. Production registration should use
:mod:`batchmatch.search`. Future work: regularization and multi-scale
optimization.
"""
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