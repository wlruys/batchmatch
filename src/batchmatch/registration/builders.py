from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from batchmatch.base.pipeline import Pipeline
from batchmatch.io.config import OutputConfig
from batchmatch.process import CenterPad, ProcessConfig, ScaleResize, TargetResize
from batchmatch.process.builders import build_crop_stage_from_config
from batchmatch.search import ExhaustiveSearchConfig, SearchGridConfig
from batchmatch.search.product import build_product_pipeline

__all__ = [
    "build_preprocessing_pipeline",
    "build_search_params",
    "build_search_config",
    "build_product_pipeline_from_output",
]

#TODO(wlr): This is mainly for Hydra integration, currently unused elsewhere


def build_preprocessing_pipeline(process: ProcessConfig) -> Pipeline:
    stages = []

    crop_stage = build_crop_stage_from_config(process.crop)
    if crop_stage is not None:
        stages.append(crop_stage)

    if process.resize is not None:
        resize_cfg = process.resize
        if resize_cfg.method == "scale":
            stages.append(ScaleResize(scale=resize_cfg.scale, outputs=list(resize_cfg.outputs)))
        elif resize_cfg.method == "target":
            stages.append(
                TargetResize(
                    target_width=resize_cfg.target_width,
                    target_height=resize_cfg.target_height,
                    outputs=list(resize_cfg.outputs),
                )
            )

    if process.pad is not None:
        pad_cfg = process.pad
        stages.append(
            CenterPad(
                scale=pad_cfg.scale,
                window_alpha=pad_cfg.window_alpha,
                pad_to_pow2=pad_cfg.pad_to_pow2,
                pad_to_even=pad_cfg.pad_to_even,
                shrink_by=pad_cfg.shrink_by,
                outputs=list(pad_cfg.outputs),
            )
        )

    return Pipeline(stages)


def build_search_params(grid: SearchGridConfig):
    if not isinstance(grid, SearchGridConfig):
        raise TypeError("grid must be a SearchGridConfig")
    return grid.to_search_params()


def build_search_config(
    search: ExhaustiveSearchConfig,
    *,
    translation_method: Optional[str] = None,
    translation_params: Optional[dict[str, Any]] = None,
    gradient_method: Optional[str] = None,
    gradient_params: Optional[dict[str, Any]] = None,
    device: Optional[str] = None,
) -> ExhaustiveSearchConfig:
    if not isinstance(search, ExhaustiveSearchConfig):
        raise TypeError("search must be an ExhaustiveSearchConfig")

    overrides: dict[str, Any] = {}
    if translation_method is not None:
        overrides["translation_method"] = translation_method
        if translation_params is not None:
            overrides["translation_params"] = translation_params
    if gradient_method is not None:
        overrides["gradient_method"] = gradient_method
        if gradient_params is not None:
            overrides["gradient_params"] = gradient_params
    if device is not None:
        overrides["device"] = device

    if not overrides:
        return search

    return replace(search, **overrides)


def build_product_pipeline_from_output(result, output: OutputConfig):
    if not isinstance(output, OutputConfig):
        raise TypeError("output must be an OutputConfig")
    return build_product_pipeline(
        result,
        scale_translation=output.scale_translation,
        scale_warp_translation=output.scale_warp_translation,
        apply_pad=output.apply_pad,
        apply_warp=output.apply_warp,
        apply_shift=output.apply_shift,
    )
