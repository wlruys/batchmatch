"""Register a synthetic crop+warp via exhaustive warp search.

Demonstrates the full SpatialImage → RegistrationTransform → ProductIO
pipeline on images created in-memory (crop from reference + artificial
rotation).

Run:
    uv run examples/registration/register_crop_warp.py
"""

import time
from pathlib import Path

import torch

from batchmatch import auto_device
from batchmatch.base import ImageDetail, build_image_td
from batchmatch.io import ImageIO, ProductIO, export_registered
from batchmatch.io.space import ImageSpace, RegionYXHW, SourceInfo, SpatialImage
from batchmatch.process.crop import RandomCropStage
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import ScaleResize
from batchmatch.process.spatial_stages import SpatialCenterPad
from batchmatch.search import (
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    AngleRange,
    SearchParams,
)
from batchmatch.search.transform import RegistrationTransform
from batchmatch.translate.config import GNGFTranslationConfig
from batchmatch.gradient import (
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
)
from batchmatch.view.config import CheckerboardSpec, OverlaySpec
from batchmatch.view.display import show_comparison
from batchmatch.warp import WarpPipelineConfig, build_warp_pipeline

import numpy as np


def _make_source(label: str, h: int, w: int) -> SourceInfo:
    """Build a synthetic SourceInfo for in-memory images."""
    return SourceInfo(
        source_path=f"<synthetic:{label}>",
        series_index=0,
        level_count=1,
        level_shapes=((h, w),),
        axes="YX",
        dtype="float32",
        format="raster",
    )


def _wrap_spatial(detail: ImageDetail, label: str) -> SpatialImage:
    """Wrap a plain ImageDetail into a SpatialImage with identity geometry."""
    h, w = detail.image.shape[-2], detail.image.shape[-1]
    src = _make_source(label, h, w)
    region = RegionYXHW(y=0, x=0, h=h, w=w)
    space = ImageSpace(
        source=src,
        pyramid_level=0,
        region=region,
        downsample=1,
        shape_hw=(h, w),
        matrix_image_from_source=np.eye(3, dtype=np.float64),
    )
    return SpatialImage(detail=detail, space=space)


def main() -> None:
    reference_detail = ImageIO(grayscale=True).load("img/test.jpg").detail

    # Build random moving image by cropping from reference
    max_H, max_W = reference_detail.image.shape[-2:]
    min_crop = max_H // 8
    max_crop = max_H // 4

    generator = torch.Generator().manual_seed(123)
    crop_stage = RandomCropStage(
        min_size=(min_crop, min_crop),
        max_size=(max_crop, max_crop),
        generator=generator,
    )

    moving_detail = crop_stage(reference_detail)

    pad_to_warp = CenterPad(
        scale=1.3,
        pad_to_pow2=False,
        outputs=["image"],
    )

    applied_angle = 10.0
    moving_detail.add_warp_params(angle=applied_angle)
    warp_pipe = build_warp_pipeline(WarpPipelineConfig(outputs=["image"]))
    pad_and_warp = pad_to_warp >> warp_pipe
    moving_detail = pad_and_warp(moving_detail)
    moving_detail.clear_warp_params()

    print(f"Applied artificial rotation: {applied_angle} degrees")

    # Wrap bare ImageDetails as SpatialImages
    reference = _wrap_spatial(reference_detail, "reference")
    moving = _wrap_spatial(moving_detail, "moving")

    print(f"Reference shape: {tuple(reference.detail.image.shape)}")
    print(f"Moving shape (after warp): {tuple(moving.detail.image.shape)}")

    # Prepare low-resolution padded versions for search
    pad = SpatialCenterPad(
        scale=2,
        window_alpha=0.05,
        pad_to_pow2=False,
        outputs=["image", "box", "mask", "quad", "window"],
    )
    ref_search, mov_search = pad([reference.clone(), moving.clone()])

    print(f"Search reference shape: {tuple(ref_search.detail.image.shape)}")
    print(f"Search moving shape: {tuple(mov_search.detail.image.shape)}")

    show_comparison(
        ref_search.detail,
        mov_search.detail,
        mode="overlay",
        spec=OverlaySpec(),
    )

    search_params = SearchParams(
        rotation=AngleRange(min_angle=-15.0, max_angle=15.0, step=0.5),
    )
    config = ExhaustiveSearchConfig(
        translation=GNGFTranslationConfig(overlap_fraction=0.99, p=2),
        batch_size=32,
        progress_enabled=True,
        gradient=CDGradientConfig(
            eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
            normalize=NormalizeConfig(norm="l2", threshold=1e-3),
        ),
        use_moving_cache=False,
        use_reference_cache=False,
    )
    search = ExhaustiveWarpSearch(search_params, config)

    device = auto_device("auto")
    ref_search = ref_search.to(device)
    mov_search = mov_search.to(device)
    search = search.to(device)

    start_t = time.perf_counter()
    result = search(ref_search.detail, mov_search.detail, top_k=1, progress=True)
    end_t = time.perf_counter()

    print(f"Search took {end_t - start_t:.2f} seconds.")

    warp = result.warp
    translation = result.translation_results
    recovered_angle = warp.angle.item()
    expected_angle = -applied_angle
    print(f"Expected rotation angle: {expected_angle}")
    print(f"Recovered rotation angle: {recovered_angle:.2f}")
    print(f"Angle error: {abs(recovered_angle - expected_angle):.2f} degrees")
    print(f"Estimated translation: tx={translation.tx.item():.1f}, ty={translation.ty.item():.1f}")

    # Build RegistrationTransform
    result_cpu = result.to("cpu")
    result_cpu.add_warp_params(
        angle=0.0, scale_x=1.0, scale_y=1.0,
        shear_x=0.0, shear_y=0.0, tx=0.0, ty=0.0,
    )  # ensure warp params exist if search only had rotation

    transform = RegistrationTransform.from_search(
        moving=mov_search.to("cpu"),
        reference=ref_search.to("cpu"),
        search_result=result_cpu,
    )

    # Save manifest
    out_dir = Path("outputs") / "register_crop_warp"
    store = ProductIO(out_dir)
    manifest_path = store.save(transform, overwrite=True)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
