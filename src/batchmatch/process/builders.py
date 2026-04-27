from __future__ import annotations

from typing import Optional, Sequence

from batchmatch.base.pipeline import Stage
from batchmatch.process.config import CropConfig, CropOutputConfig, MaskCropConfig, RandomCropConfig
from batchmatch.process.crop import build_crop_stage

__all__ = ["build_crop_stage_from_config"]


def _normalize_outputs(outputs: list[str] | CropOutputConfig) -> list[str]:
    if isinstance(outputs, CropOutputConfig):
        return outputs.to_outputs_list()
    return outputs


def build_crop_stage_from_config(crop: CropConfig | None) -> Stage | None:
    if crop is None:
        return None

    crop_type = (crop.type or "none").lower()
    if crop_type in {"none", "null", "skip", "identity"}:
        return None

    if crop_type == "random":
        random_cfg = crop.random or RandomCropConfig()
        outputs = _normalize_outputs(random_cfg.outputs)
        generator = None
        if random_cfg.seed is not None:
            import torch

            generator = torch.Generator()
            generator.manual_seed(int(random_cfg.seed))

        return build_crop_stage(
            "crop_random",
            min_size=random_cfg.min_size,
            max_size=random_cfg.max_size,
            min_area=random_cfg.min_area,
            max_area=random_cfg.max_area,
            min_aspect=random_cfg.min_aspect,
            max_aspect=random_cfg.max_aspect,
            max_attempts=random_cfg.max_attempts,
            allow_full_image=random_cfg.allow_full_image,
            outputs=outputs,
            clip_geometry=random_cfg.clip_geometry,
            invalidate_warp=random_cfg.invalidate_warp,
            generator=generator,
        )

    if crop_type in {"union", "intersection"}:
        mask_cfg = crop.mask or MaskCropConfig(method=crop_type)
        outputs = _normalize_outputs(mask_cfg.outputs)
        name = "crop_union" if crop_type == "union" else "crop_intersection"
        return build_crop_stage(
            name,
            outputs=outputs,
            clip_geometry=mask_cfg.clip_geometry,
            invalidate_warp=mask_cfg.invalidate_warp,
        )

    return None
