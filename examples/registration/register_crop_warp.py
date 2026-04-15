import time
from pathlib import Path
import torch
import torch.nn.functional as F

from batchmatch import auto_device
from batchmatch.base import ImageDetail, build_image_td
from batchmatch.io import ImageIO, ProductIO
from batchmatch.process.crop import RandomCropStage
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import ScaleResize
from batchmatch.translate import NGFTranslationConfig
from batchmatch.gradient import (
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
)
from batchmatch.search import (
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    AngleRange,
    SearchParams,
    build_product_pipeline,
)
from batchmatch.search.config import ScaleRange
from batchmatch.translate.config import GNGFTranslationConfig
from batchmatch.view.config import CheckerboardSpec, OverlaySpec
from batchmatch.view.display import show_comparison
from batchmatch.warp import WarpPipelineConfig, build_warp_pipeline


def main() -> None:
    reference = ImageIO(grayscale=True).load("img/test.jpg").detail


    #Build random moving image by cropping from reference
    max_H, max_W = reference.image.shape[-2:]
    min_crop = max_H // 8
    max_crop = max_H // 4

    generator = torch.Generator().manual_seed(123)
    crop_stage = RandomCropStage(
        min_size=(min_crop, min_crop),
        max_size=(max_crop, max_crop),
        generator=generator,
    )

    moving = crop_stage(reference)

    pad_to_warp = CenterPad(
        scale=1.3,
        pad_to_pow2=False,
        outputs=["image"],
    )

    applied_angle = 10.0
    moving.add_warp_params(angle=applied_angle)
    warp_pipe = build_warp_pipeline(WarpPipelineConfig(outputs=["image"]))
    pad_and_warp = pad_to_warp >> warp_pipe
    moving = pad_and_warp(moving)
    moving.clear_warp_params()

    print(f"Applied artificial rotation: {applied_angle} degrees")

    reference_fullres = reference.clone()
    moving_fullres = moving.clone()
    moving_unregistered = moving_fullres.clone()

    print(f"Full-res reference shape: {reference_fullres.image.shape}")
    print(f"Full-res moving shape (after warp): {moving_fullres.image.shape}")

    # Prepare low-resolution padded versions for search
    scale = 0.128
    resize_pipe = ScaleResize(scale=scale)
    pad_pipe = CenterPad(
        scale=2,
        window_alpha=0.05,
        pad_to_pow2=False,
        outputs=["image", "box", "mask", "quad", "window"],
    )
    prepare_pipe = resize_pipe >> pad_pipe
    reference_lowres, moving_lowres = prepare_pipe(reference, moving)

    print(f"Low-res reference shape: {reference_lowres.image.shape}")
    print(f"Low-res moving shape: {moving_lowres.image.shape}")

    show_comparison(
        reference_lowres,
        moving_lowres,
        mode="overlay",
        spec=OverlaySpec(),
    )
    
    search_params = SearchParams(
        rotation=AngleRange(min_angle=-15.0, max_angle=15.0, step=0.5),
        #scale_x=ScaleRange(min_scale=1.0, max_scale=1.1, step=0.01),
        #scale_y=ScaleRange(min_scale=1.0, max_scale=1.1, step=0.01),
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

    reference_lowres = reference_lowres.to(device)
    moving_lowres = moving_lowres.to(device)
    search = search.to(device)

    start_t = time.perf_counter()
    result = search(reference_lowres, moving_lowres, top_k=1, progress=True)
    end_t = time.perf_counter()

    print(f"Search took {end_t - start_t:.2f} seconds.")

    warp = result.warp
    translation = result.translation_results
    recovered_angle = warp.angle.item()
    expected_angle = -applied_angle
    print(f"Expected rotation angle: {expected_angle}")
    print(f"Recovered rotation angle: {recovered_angle:.2f}")
    print(f"Angle error: {abs(recovered_angle - expected_angle):.2f} degrees")
    print(f"Estimated translation (low-res): tx={translation.tx.item():.1f}, ty={translation.ty.item():.1f}")

    # Apply search result to full-resolution images (only first image in args is transformed)
    product_pipeline = build_product_pipeline(result, crop_mode="union")
    registered, reference_out, original_mov = product_pipeline(
        moving_fullres.clone(),
        reference_fullres.clone(),
        moving_fullres.clone(),
    )

    print(f"Scaled translation (full-res): tx={registered.translation_results.tx.item():.1f}, ty={registered.translation_results.ty.item():.1f}")

    print(f"Registered shape: {registered.image.shape}")
    print(f"Reference output shape: {reference_out.image.shape}")

    show_comparison(
        reference_out,
        registered,
        mode="overlay",
        spec=OverlaySpec(),
    )

    out_dir = Path("outputs") / "register_crop_warp" 

    moving_save = original_mov.clone()
    reference_save = reference_out.clone()
    registered_save = registered.clone()
    
    ProductIO(out_dir).save(
        moving=moving_save,
        reference=reference_save,
        registered=registered_save,
        search_result=result,
        overwrite=True,
    )
    print(f"Saved product images to: {out_dir}")


if __name__ == "__main__":
    main()
