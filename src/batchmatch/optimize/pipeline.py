from __future__ import annotations

from typing import List

from batchmatch.base.pipeline import Pipeline, Stage
from batchmatch.gradient.base import build_gradient_pipeline
from batchmatch.gradient.config import GradientPipelineConfig, GradientMethodConfig
from batchmatch.warp import WarpPipelineConfig, build_warp_pipeline

from .config import OptimizeConfig

__all__ = [
    "build_reference_pipeline",
    "build_moving_pipeline",
]


def _default_warp_config() -> WarpPipelineConfig:
    return WarpPipelineConfig(
        outputs=["image", "mask", "window", "boxes"],
        prepare={"type": "prepare", "inverse": True, "align_corners": False, "inplace": False},
        stages={
            "image": {
                "type": "image",
                "mode": "bilinear",
                "fill_value": 0.0,
                "align_corners": False,
                "inplace": False,
            },
            "mask": {
                "type": "mask",
                "fill_value": 0.0,
                "align_corners": False,
                "inplace": False,
            },
            "window": {
                "type": "window",
                "mode": "bilinear",
                "fill_value": 0.0,
                "align_corners": False,
                "inplace": False,
            },
            "boxes": {
                "type": "boxes",
            },
        },
    )


def _build_gradient_stage(
    config: OptimizeConfig,
    *,
    require_complex: bool = False,
) -> Stage:
    method: GradientPipelineConfig | GradientMethodConfig | str
    if config.gradient is None:
        method = "sobel"
        params = {}
    else:
        method = config.gradient
        params = {}
    return build_gradient_pipeline(method, build_complex=require_complex, **params)


def build_reference_pipeline(
    config: OptimizeConfig,
    *,
    requires_gradients: bool = False,
    requires_complex_gradients: bool = False,
) -> Pipeline:
    stages: List[Stage] = []

    if requires_gradients:
        stages.append(
            _build_gradient_stage(
                config,
                require_complex=requires_complex_gradients,
            )
        )

    return Pipeline(stages)


def build_moving_pipeline(
    config: OptimizeConfig,
    *,
    requires_gradients: bool = False,
    requires_complex_gradients: bool = False,
) -> Pipeline:
    stages: List[Stage] = []

    warp_cfg = config.warp or _default_warp_config()
    stages.append(build_warp_pipeline(warp_cfg))

    if requires_gradients:
        stages.append(
            _build_gradient_stage(
                config,
                require_complex=requires_complex_gradients,
            )
        )

    return Pipeline(stages)
