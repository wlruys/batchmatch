from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from batchmatch.base import (
    CacheTD,
    ImageDetail,
    WarpParams,
    TranslationResults,
    NestedKey,
    build_image_td,
    validate_image_td_shape,
    Stage,
    Pipeline,
    StageSpec,
    StageSpecConf,
    StageRegistry,
    coerce_stage_spec,
    coerce_stage_list,
    should_build_stage,
)
from batchmatch.helpers.device import auto_device
from batchmatch.optimize import (
    AffineWarpOptimize,
    MetricConfig,
    OptimizeConfig,
    OptimizeResult,
    OptimizerConfig,
    SchedulerConfig,
)

try:
    __version__ = _pkg_version("batchmatch")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "auto_device",
    "CacheTD",
    "ImageDetail",
    "WarpParams",
    "TranslationResults",
    "NestedKey",
    "build_image_td",
    "validate_image_td_shape",
    "Stage",
    "Pipeline",
    "StageSpec",
    "StageSpecConf",
    "StageRegistry",
    "coerce_stage_spec",
    "coerce_stage_list",
    "should_build_stage",
    "AffineWarpOptimize",
    "MetricConfig",
    "OptimizeConfig",
    "OptimizeResult",
    "OptimizerConfig",
    "SchedulerConfig",
]
