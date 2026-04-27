"""Register a synthetic crop+warp via exhaustive warp search.

This example builds a moving image from a known crop of ``img/test.jpg``,
applies a known rotation, and recovers the crop placement plus rotation with
``ExhaustiveWarpSearch``.

Run:
    uv run python examples/registration/register_crop_warp.py --device cpu
    uv run python examples/registration/register_crop_warp.py --device mps
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from batchmatch import auto_device
from batchmatch.base import ImageDetail, WarpParams, build_image_td
from batchmatch.helpers.affine import mat_pad
from batchmatch.io import ImageIO, ProductIO
from batchmatch.io.space import ImageSpace, RegionYXHW, SourceInfo, SpatialImage
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import TargetResize
from batchmatch.process.spatial_stages import SpatialCenterPad
from batchmatch.search import (
    AngleRange,
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    SearchParams,
)
from batchmatch.search.transform import RegistrationTransform
from batchmatch.view.config import CheckerboardSpec, DisplaySpec, EdgeOverlaySpec, OverlaySpec
from batchmatch.view.display import show_comparison, show_images
from batchmatch.warp import WarpPipelineConfig, build_warp_pipeline
from batchmatch.warp.resample import warp_to_reference
from batchmatch.warp.specs import compute_warp_matrices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("img/test.jpg"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "register_crop_warp")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, mps, cuda, or auto.")
    parser.add_argument("--reference-width", type=int, default=512)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y", "X", "H", "W"),
        default=(112, 164, 168, 192),
        help="Crop from the resized reference, in y x h w pixels.",
    )
    parser.add_argument("--applied-angle", type=float, default=10.0)
    parser.add_argument("--rotation-min", type=float, default=-15.0)
    parser.add_argument("--rotation-max", type=float, default=15.0)
    parser.add_argument("--rotation-step", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pad-scale", type=float, default=1.25)
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows.")
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=0.75,
        help="Maximum allowed rotation error in degrees.",
    )
    parser.add_argument(
        "--matrix-tolerance",
        type=float,
        default=3.0,
        help="Maximum allowed full-resolution matrix coefficient error in pixels.",
    )
    return parser.parse_args()


def _make_source(label: str, h: int, w: int) -> SourceInfo:
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
    h, w = detail.image.shape[-2], detail.image.shape[-1]
    src = _make_source(label, h, w)
    space = ImageSpace(
        source=src,
        pyramid_level=0,
        region=RegionYXHW(y=0, x=0, h=h, w=w),
        downsample=1,
        shape_hw=(h, w),
        matrix_image_from_source=np.eye(3, dtype=np.float64),
    )
    return SpatialImage(detail=detail, space=space)


def _load_reference(path: Path, target_width: int) -> ImageDetail:
    detail = ImageIO(grayscale=True).load(path).detail
    if target_width > 0:
        detail = TargetResize(target_width=target_width, outputs=["image"])(detail)
    return detail


def _crop_reference(reference: ImageDetail, crop_yxhw: tuple[int, int, int, int]) -> ImageDetail:
    y, x, h, w = crop_yxhw
    ref_h, ref_w = reference.image.shape[-2:]
    if y < 0 or x < 0 or h <= 0 or w <= 0 or y + h > ref_h or x + w > ref_w:
        raise ValueError(
            f"Crop yxhw={crop_yxhw} is outside resized reference shape {(ref_h, ref_w)}."
        )
    return build_image_td(reference.image[..., y : y + h, x : x + w].clone())


def _make_warped_moving(
    crop: ImageDetail,
    *,
    angle: float,
    pad_scale: float,
) -> tuple[ImageDetail, tuple[int, int, int, int], np.ndarray]:
    pad_stage = CenterPad(scale=pad_scale, pad_to_pow2=False, outputs=["image"])
    padded = pad_stage(crop.clone())
    crop_h, crop_w = crop.image.shape[-2:]
    mov_h, mov_w = padded.image.shape[-2:]
    left = (mov_w - crop_w) // 2
    top = (mov_h - crop_h) // 2
    right = mov_w - crop_w - left
    bottom = mov_h - crop_h - top

    padded.add_warp_params(angle=angle)
    warped = build_warp_pipeline(WarpPipelineConfig(outputs=["image"]))(padded)
    warped.clear_warp_params()

    warp = WarpParams.from_components(
        angle=torch.tensor([angle], dtype=torch.float32),
        scale_x=torch.tensor([1.0], dtype=torch.float32),
        scale_y=torch.tensor([1.0], dtype=torch.float32),
        shear_x=torch.tensor([0.0], dtype=torch.float32),
        shear_y=torch.tensor([0.0], dtype=torch.float32),
        tx=torch.tensor([0.0], dtype=torch.float32),
        ty=torch.tensor([0.0], dtype=torch.float32),
    )
    M_applied, _ = compute_warp_matrices(warp, mov_h, mov_w, inplace=False)
    return warped, (left, top, right, bottom), M_applied[0].numpy().astype(np.float64)


def _expected_matrix_ref_from_moving(
    *,
    crop_yxhw: tuple[int, int, int, int],
    crop_pad_ltrb: tuple[int, int, int, int],
    applied_sampling_matrix: np.ndarray,
) -> np.ndarray:
    y, x, _, _ = crop_yxhw
    left, top, _, _ = crop_pad_ltrb
    return mat_pad(float(x - left), float(y - top), 0.0, 0.0) @ applied_sampling_matrix


def _max_abs_matrix_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> None:
    args = _parse_args()
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device(args.device)
    crop_yxhw = tuple(int(v) for v in args.crop)

    reference_detail = _load_reference(args.image, args.reference_width)
    crop_detail = _crop_reference(reference_detail, crop_yxhw)
    moving_detail, crop_pad_ltrb, applied_sampling_matrix = _make_warped_moving(
        crop_detail,
        angle=args.applied_angle,
        pad_scale=args.pad_scale,
    )

    reference = _wrap_spatial(reference_detail, "reference")
    moving = _wrap_spatial(moving_detail, "moving")

    print(f"device: {device}")
    print(f"reference shape: {tuple(reference.detail.image.shape)}")
    print(f"crop yxhw in resized reference: {crop_yxhw}")
    print(f"moving shape after crop pad + warp: {tuple(moving.detail.image.shape)}")
    print(f"applied moving rotation: {args.applied_angle:.2f} deg")

    show_images(
        [reference.detail.image.cpu(), moving.detail.image.cpu()],
        display=DisplaySpec(
            title="Synthetic Crop-Warp Inputs",
            show=args.show,
            save_path=str(args.output_dir / "preview_inputs.png"),
        ),
    )

    search_pad = SpatialCenterPad(
        scale=args.pad_scale,
        window_alpha=0.05,
        pad_to_pow2=False,
        shrink_by=0,
        outputs=["image", "box", "mask", "quad", "window"],
    )
    ref_search, mov_search = search_pad([reference.clone(), moving.clone()])

    print(f"search reference shape: {tuple(ref_search.detail.image.shape)}")
    print(f"search moving shape:    {tuple(mov_search.detail.image.shape)}")

    show_comparison(
        ref_search.detail,
        mov_search.detail,
        mode="overlay",
        spec=OverlaySpec(alpha=0.5),
        display=DisplaySpec(
            title="Search Space Before Registration",
            show=args.show,
            save_path=str(args.output_dir / "preview_search_before_overlay.png"),
        ),
    )

    search_params = SearchParams(
        rotation=AngleRange(args.rotation_min, args.rotation_max, args.rotation_step),
    )
    config = ExhaustiveSearchConfig(
        translation_method="ncc",
        translation_params={"overlap_fraction": 0.5},
        batch_size=args.batch_size,
        progress_enabled=True,
        device=device,
        use_moving_cache=False,
        use_reference_cache=False,
    )
    search = ExhaustiveWarpSearch(search_params, config).to(device)

    start_t = time.perf_counter()
    result = search(
        ref_search.to(device).detail,
        mov_search.to(device).detail,
        top_k=1,
        progress=True,
    )
    print(f"search took {time.perf_counter() - start_t:.2f} s")

    result_cpu = result.to("cpu")
    warp = result_cpu.warp
    translation = result_cpu.translation_results
    recovered_angle = float(warp.angle[0].item())
    expected_angle = -float(args.applied_angle)
    angle_error = abs(recovered_angle - expected_angle)
    print(f"expected recovery rotation: {expected_angle:.2f} deg")
    print(f"recovered rotation:         {recovered_angle:.2f} deg")
    print(f"angle error:                {angle_error:.2f} deg")
    print(
        "estimated translation in search pixels: "
        f"tx={translation.tx[0].item():.2f}, ty={translation.ty[0].item():.2f}, "
        f"score={translation.score[0].item():.4f}"
    )

    ref_search_cpu = ref_search.to("cpu")
    mov_search_cpu = mov_search.to("cpu")
    transform = RegistrationTransform.from_search(
        moving=mov_search_cpu,
        reference=ref_search_cpu,
        search_result=result_cpu,
    )

    expected_full = _expected_matrix_ref_from_moving(
        crop_yxhw=crop_yxhw,
        crop_pad_ltrb=crop_pad_ltrb,
        applied_sampling_matrix=applied_sampling_matrix,
    )
    matrix_error = _max_abs_matrix_error(
        transform.matrix_ref_full_from_mov_full,
        expected_full,
    )
    print(f"expected ref<-moving matrix:\n{expected_full}")
    print(f"recovered ref<-moving matrix:\n{transform.matrix_ref_full_from_mov_full}")
    print(f"max matrix coefficient error: {matrix_error:.2f} px")

    low_registered = warp_to_reference(
        mov_search_cpu.detail,
        transform.matrix_ref_search_from_mov_search,
        out_hw=ref_search_cpu.shape_hw,
    )
    show_comparison(
        ref_search_cpu.detail,
        low_registered,
        mode="overlay",
        spec=OverlaySpec(alpha=0.5),
        display=DisplaySpec(
            title="Search Space After Registration",
            show=args.show,
            save_path=str(args.output_dir / "preview_search_registered_overlay.png"),
        ),
    )
    show_comparison(
        ref_search_cpu.detail,
        low_registered,
        mode="checkerboard",
        spec=CheckerboardSpec(
            tiles=(32, 32),
            edge_overlay=EdgeOverlaySpec(edge_source="mov", edge_threshold=0.3),
        ),
        display=DisplaySpec(
            title="Search Space Registered Checkerboard",
            show=args.show,
            save_path=str(args.output_dir / "preview_search_registered_checkerboard.png"),
        ),
    )

    store = ProductIO(args.output_dir)
    manifest_path = store.save(
        transform,
        debug_details={"search_result": result_cpu},
        extra_artifacts={
            "synthetic": {
                "source_image": str(args.image),
                "reference_width": int(args.reference_width),
                "crop_yxhw": list(crop_yxhw),
                "crop_pad_ltrb": list(crop_pad_ltrb),
                "applied_angle": float(args.applied_angle),
                "expected_recovery_angle": expected_angle,
                "angle_error": angle_error,
                "expected_matrix_ref_full_from_mov_full": expected_full.tolist(),
                "max_matrix_error_px": matrix_error,
            }
        },
        overwrite=True,
    )
    print(f"manifest: {manifest_path}")

    if angle_error > args.angle_tolerance:
        raise RuntimeError(
            f"Recovered angle error {angle_error:.2f} exceeds tolerance "
            f"{args.angle_tolerance:.2f}."
        )
    if matrix_error > args.matrix_tolerance:
        raise RuntimeError(
            f"Recovered matrix error {matrix_error:.2f} px exceeds tolerance "
            f"{args.matrix_tolerance:.2f} px."
        )


if __name__ == "__main__":
    main()
