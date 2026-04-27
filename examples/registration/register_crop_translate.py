import torch
import torch.nn.functional as F

from batchmatch.base import ImageDetail, build_image_td
from batchmatch.gradient import (
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
    build_gradient_pipeline,
)
from batchmatch.io import ImageIO
from batchmatch.process.crop import RandomCropStage
from batchmatch.process.pad import build_pad_pipeline
from batchmatch.process.resize import build_resize_pipeline
from batchmatch.process.shift import build_shift_pipeline
from batchmatch.process.window import build_window_pipeline
from batchmatch.translate import build_translation_stage
from batchmatch.view.composite import render_checkerboard
from batchmatch.view.config import GallerySpec, ImageViewSpec
from batchmatch.view.display import DisplaySpec, show_comparison, show_images
from batchmatch.warp import build_warp_pipeline


def main():
    import matplotlib.pyplot as plt

    reference = ImageIO(grayscale=True).load("img/test.jpg").detail

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

    resize_pipe = build_resize_pipeline(
        "target_resize",
        target_width=256,
    )
    pad_pipe = build_pad_pipeline(
        "center_pad",
        scale=2,
    )
    prepare_pipe = resize_pipe >> pad_pipe
    reference, moving = prepare_pipe(reference, moving)

    show_comparison(
        reference,
        moving,
        mode="overlay",
    )

    search = build_translation_stage("ngf")
    shift = build_shift_pipeline(source="translation", negate=False)

    if search.requires_gradients:
        grad_op = build_gradient_pipeline(
            CDGradientConfig(
                eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
                normalize=NormalizeConfig(norm="l2", threshold=1e-3),
                build_complex=search.requires_complex_gradients,
            )
        )
        reference, moving = grad_op(reference, moving)

    moving = search(reference, moving)
    print("Estimated translation tx:", moving.translation_results.tx)
    print("Estimated translation ty:", moving.translation_results.ty)

    shifted_moving = shift(moving)

    show_comparison(
        reference,
        shifted_moving,
        mode="overlay",
    )

if __name__ == "__main__":
    main()
