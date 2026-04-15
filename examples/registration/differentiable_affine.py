from __future__ import annotations

import torch

from batchmatch import auto_device
from batchmatch.base import build_image_td
from batchmatch.metric import ImageMetricSpec, cross_correlation
from batchmatch.optimize import AffineWarpOptimize, MetricConfig, OptimizeConfig, OptimizerConfig
from batchmatch.process.pad import CenterPad
from batchmatch.view.config import DisplaySpec, CheckerboardSpec
from batchmatch.view.display import show_comparison
from batchmatch.warp import WarpPipelineConfig, build_warp_pipeline


def make_gaussian_mixture(size: int = 256, n_gaussians: int = 5, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    image = torch.zeros(1, 1, size, size)
    for _ in range(n_gaussians):
        cx, cy = torch.rand(2) * 1.6 - 0.8
        sigma = 0.1 + torch.rand(1).item() * 0.3
        amp = 0.5 + torch.rand(1).item() * 0.5
        image += amp * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return image / image.max()


def main(metric: str = "ncc", iterations: int = 200, show: bool = True) -> None:
    device = auto_device("auto")
    gt_angle, gt_tx, gt_ty = 8.0, 10.0, -6.0

    reference_img = make_gaussian_mixture(size=256, seed=42).to(device)
    reference = build_image_td(reference_img)
    pad_stage = CenterPad(scale=1.3, outputs=["image", "mask"], pad_to_pow2=False)
    reference = pad_stage(reference)

    moving = reference.clone()
    moving.add_warp_params(angle=gt_angle, tx=gt_tx, ty=gt_ty)
    warp_pipe = build_warp_pipeline(WarpPipelineConfig(outputs=["image", "mask"]))
    moving = warp_pipe(moving)
    moving.clear_warp_params()

    config = OptimizeConfig(
        iterations=iterations,
        device=device,
        metric=MetricConfig(spec=ImageMetricSpec(metric, {"reduction": "mean"})),
        optimizer=OptimizerConfig(
            params={"lr": 0.1},
            param_groups=[
                {"params": ["angle"], "lr": 0.05},
                {"params": ["tx", "ty"], "lr": 0.2},
                {"params": ["scale_x", "scale_y", "shear_x", "shear_y"], "lr": 0.0},
            ],
        ),
        warp=WarpPipelineConfig(outputs=["image", "mask"]),
    )
    result = AffineWarpOptimize(config).optimize(reference, moving)

    ncc_before = cross_correlation(reference.image, moving.image).item()
    ncc_after = cross_correlation(reference.image, result.registered.image).item()
    w = result.warp
    print(f"NCC: {ncc_before:.4f} -> {ncc_after:.4f}")
    print(f"Applied:   angle={gt_angle:.1f}°, tx={gt_tx:.1f}, ty={gt_ty:.1f}")
    print(f"Recovered: angle={w.angle.item():.1f}°, tx={w.tx.item():.1f}, ty={w.ty.item():.1f}")

    if show:
        show_comparison(reference, moving, mode="checkerboard", spec=CheckerboardSpec(),
                        display=DisplaySpec(title="Before"))
        show_comparison(reference, result.registered, mode="checkerboard", spec=CheckerboardSpec(),
                        display=DisplaySpec(title="After"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="mse", choices=["mse", "ncc", "ngf"])
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--no-show", dest="show", action="store_false")
    args = p.parse_args()
    main(args.metric, args.iterations, args.show)
