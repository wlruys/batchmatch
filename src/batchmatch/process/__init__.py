"""
Processing stages for image transformations.

Exports process-stage registries and helper constructors.
"""
from batchmatch.process.config import (
    CropConfig,
    CropOutputConfig,
    MaskCropConfig,
    PadConfig,
    ProcessConfig,
    RandomCropConfig,
    ResizeConfig,
)
from batchmatch.process.resize import (
    CellUnitResize,
    ScaleResize,
    TargetResize,
    build_resize_pipeline,
    resize_registry,
    get_scales_between,
)
from batchmatch.process.shift import (
    ShiftStage,
    SubpixelShiftStage,
    build_shift_pipeline,
    build_shift_stage,
    build_subpixel_shift_pipeline,
    get_translation_offsets,
    shift_registry,
    shift_spatial_batch,
)
from batchmatch.process.crop import (
    CropStageBase,
    CropUnionStage,
    CropIntersectionStage,
    RandomCropStage,
    build_crop_stage,
    crop_registry,
    invalidate_warp_computed,
)
from batchmatch.process.builders import build_crop_stage_from_config
from batchmatch.process.pad import (
    CenterPad,
    build_pad_pipeline,
    pad_registry,
)
from batchmatch.process.window import (
    ApplyWindowStage,
    TukeyBoxWindow,
    TukeyQuadWindow,
    window_registry,
    build_window_operator,
    WindowPipelineSpec,
    build_window_pipeline,
)

__all__ = [
    # Config classes
    "CropConfig",
    "CropOutputConfig",
    "MaskCropConfig",
    "PadConfig",
    "ProcessConfig",
    "RandomCropConfig",
    "ResizeConfig",
    # Resize
    "ScaleResize",
    "CellUnitResize",
    "TargetResize",
    "build_resize_pipeline",
    "resize_registry",
    # Shift
    "ShiftStage",
    "SubpixelShiftStage",
    "build_shift_pipeline",
    "build_shift_stage",
    "build_subpixel_shift_pipeline",
    "get_translation_offsets",
    "shift_registry",
    "shift_spatial_batch",
    # Crop
    "CropStageBase",
    "CropUnionStage",
    "CropIntersectionStage",
    "RandomCropStage",
    "build_crop_stage",
    "build_crop_stage_from_config",
    "crop_registry",
    # Pad
    "CenterPad",
    "build_pad_pipeline",
    "pad_registry",
    # Window
    "ApplyWindowStage",
    "TukeyBoxWindow",
    "TukeyQuadWindow",
    "window_registry",
    "build_window_operator",
    "WindowPipelineSpec",
    "build_window_pipeline",
    # Utilities
    "get_scales_between",
    "invalidate_warp_computed",
]
