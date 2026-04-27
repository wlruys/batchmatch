"""Affine warp specifications, stages, and resampling pipelines."""
from __future__ import annotations
from batchmatch.base.tensordicts import WarpParams
from batchmatch.warp.specs import (
    apply_warp_grid,
    compute_warp_grid,
    compute_warp_matrices,
    warp_params_from_image_detail,
    warp_params_to_image_detail,
    warp_points,
)

from batchmatch.warp.stages import (
    PrepareWarpStage,
    WarpAuxBoxesStage,
    WarpAuxQuadsStage,
    WarpBoxesStage,
    WarpImageStage,
    WarpMaskStage,
    WarpPointsStage,
    WarpQuadStage,
    WarpTensorStageBase,
    WarpWindowStage,
    build_warp_stage,
    warp_registry,
)

from batchmatch.warp.config import WarpPipelineConfig
from batchmatch.warp.base import (
    WarpPipelineSpec,
    build_warp_pipeline,
    coerce_warp_pipeline_spec,
)

__all__ = [
    "WarpParams",
    "apply_warp_grid",
    "compute_warp_grid",
    "compute_warp_matrices",
    "warp_params_from_image_detail",
    "warp_params_to_image_detail",
    "warp_points",
    "WarpTensorStageBase",
    "PrepareWarpStage",
    "WarpAuxBoxesStage",
    "WarpAuxQuadsStage",
    "WarpBoxesStage",
    "WarpImageStage",
    "WarpMaskStage",
    "WarpPointsStage",
    "WarpQuadStage",
    "WarpWindowStage",
    "build_warp_stage",
    "warp_registry",
    "WarpPipelineSpec",
    "WarpPipelineConfig",
    "build_warp_pipeline",
    "coerce_warp_pipeline_spec",
]
