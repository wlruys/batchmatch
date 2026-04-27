"""Register calibrated TIFF inputs and export the aligned result as OME-TIFF.

Default run uses two overlapping crops of ``img/FISH.tif`` so the example
works against sample data in the repo. Point ``--reference`` and
``--moving`` at separate calibrated TIFF/OME-TIFF files for a real run.

Run:
    uv run python examples/registration/register_tiff.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from batchmatch import auto_device
from batchmatch.io import (
    ProductIO,
    RegionYXHW,
    SourceInfo,
    export_moving_mask_tiff,
    export_stacked_registered_tiff,
    load_image,
    open_image,
)
from batchmatch.io.space import SpatialImage
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import TargetResize
from batchmatch.process.spatial_stages import (
    SpatialCenterPad,
    SpatialPhysicalResize,
    SpatialTargetResize,
)
from batchmatch.search import (
    AngleRange,
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    ScaleRange,
    SearchParams,
    ShearRange,
)
from batchmatch.search.transform import RegistrationTransform
from batchmatch.view.config import (
    CheckerboardSpec,
    DisplaySpec,
    EdgeOverlaySpec,
    GallerySpec,
    ImageViewSpec,
    OverlaySpec,
)
from batchmatch.view.display import show_comparison, show_images
from batchmatch.view.preview import render_registration_preview
from batchmatch.warp import PrepareWarpStage, WarpImageStage
from batchmatch.warp.resample import warp_to_reference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("img/FISH.tif"))
    parser.add_argument("--moving", type=Path, default=None)
    parser.add_argument("--reference-channel", type=str, default="DAPI")
    parser.add_argument("--moving-channel", type=str, default=None)
    parser.add_argument(
        "--reference-region",
        type=int,
        nargs=4,
        metavar=("Y", "X", "H", "W"),
        default=(0, 0, 1024, 1024),
    )
    parser.add_argument(
        "--moving-region",
        type=int,
        nargs=4,
        metavar=("Y", "X", "H", "W"),
        default=(128, 192, 1024, 1024),
    )
    parser.add_argument("--search-dim", type=int, default=512)
    parser.add_argument("--pad-scale", type=float, default=2.0)
    parser.add_argument("--metric", choices=("ncc", "pc", "cc", "gcc", "gpc", "ngf", "gngf"), default="ncc")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rotation", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(-2.0, 2.0, 1.0))
    parser.add_argument("--scale-x", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.95, 1.05, 0.025))
    parser.add_argument("--scale-y", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.95, 1.05, 0.025))
    parser.add_argument("--shear-x", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.0, 0.0, 1.0))
    parser.add_argument("--shear-y", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.0, 0.0, 1.0))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs") / "register_tiff"
    )
    parser.add_argument("--export-name", type=str, default="registered_union.ome.tif")
    parser.add_argument("--before-export-name", type=str, default="unregistered_union.ome.tif")
    parser.add_argument("--mask-name", type=str, default="registered_moving_mask_union.ome.tif")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip full-resolution OME-TIFF export.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows in addition to saving preview PNGs.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(1024, 1024),
        help="Downsize full-resolution preview images to this H W before saving.",
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=1200,
        help="Max image extent for saved input/search preview figures.",
    )
    parser.add_argument(
        "--no-full-preview",
        action="store_true",
        help="Skip full-resolution registration preview rendering.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "Optional output tile size for full-resolution warp/export. "
            "Streaming TIFF export defaults to 512 and rounds to a multiple of 16."
        ),
    )
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        default=0,
        help="Number of 2x OME-TIFF SubIFD pyramid levels for exported TIFFs.",
    )
    parser.add_argument(
        "--eager-export",
        action="store_true",
        help="Disable streamed tile/channel TIFF export and use the legacy eager path.",
    )
    parser.add_argument(
        "--export-compression",
        type=str,
        default=None,
        help="Optional lossless tifffile compression codec, e.g. zlib or zstd.",
    )
    parser.add_argument(
        "--export-channels",
        type=str,
        default="all",
        help="Deprecated; stacked union export always writes all reference and moving channels.",
    )
    parser.add_argument(
        "--synthetic-moving-warp",
        type=float,
        nargs=7,
        metavar=("ANGLE", "SCALE_X", "SCALE_Y", "SHEAR_X", "SHEAR_Y", "TX", "TY"),
        default=None,
        help=(
            "Artificially warp the loaded moving tile before search. "
            "Useful for synthetic affine recovery tests."
        ),
    )
    return parser.parse_args()


def _parse_channel(value: str | None) -> int | str | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _range3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have MIN MAX STEP.")
    start, end, step = (float(v) for v in values)
    if step <= 0:
        raise ValueError(f"{name} step must be positive.")
    return start, end, step


def _print_source_summary(label: str, path: Path) -> None:
    with open_image(path) as src:
        info = src.source
    print(f"{label}: {path}")
    print(f"  axes={info.axes} levels={info.level_count} shape={info.base_shape_hw}")
    if info.channel_names:
        print(f"  channels={info.channel_names}")
    if info.pixel_size_xy is not None:
        print(f"  pixel_size_xy={info.pixel_size_xy} {info.unit or ''}".rstrip())
    if info.origin_xy is not None:
        print(f"  origin_xy={info.origin_xy}")
    if info.physical_extent_xy is not None:
        print(f"  physical_extent_xy={info.physical_extent_xy}")


def _validate_physical_metadata(reference: SpatialImage, moving: SpatialImage) -> None:
    ref_src = reference.space.source
    mov_src = moving.space.source
    missing = [
        label
        for label, src in (("reference", ref_src), ("moving", mov_src))
        if src.pixel_size_xy is None
    ]
    if missing:
        raise ValueError(
            "TIFF physical pixel-size metadata is required for metadata-based "
            f"prescaling. Missing pixel_size_xy on: {', '.join(missing)}."
        )
    if (ref_src.unit or "") != (mov_src.unit or ""):
        raise ValueError(
            "Reference and moving TIFF physical units must match. "
            f"Got {ref_src.unit!r} and {mov_src.unit!r}."
        )


def _build_search_params(args: argparse.Namespace) -> SearchParams:
    r0, r1, rs = _range3(args.rotation, "--rotation")
    sx0, sx1, sxs = _range3(args.scale_x, "--scale-x")
    sy0, sy1, sys = _range3(args.scale_y, "--scale-y")
    hx0, hx1, hxs = _range3(args.shear_x, "--shear-x")
    hy0, hy1, hys = _range3(args.shear_y, "--shear-y")
    return SearchParams(
        rotation=AngleRange(r0, r1, rs),
        scale_x=ScaleRange(sx0, sx1, sxs),
        scale_y=ScaleRange(sy0, sy1, sys),
        shear_x=ShearRange(hx0, hx1, hxs),
        shear_y=ShearRange(hy0, hy1, hys),
    )


def _save_pair_preview(
    reference: SpatialImage,
    moving: SpatialImage,
    *,
    path: Path,
    title: str,
    show: bool,
    max_display_size: int | None,
) -> None:
    spec = GallerySpec(
        nrows=1,
        titles=["Reference", "Moving"],
        image_spec=ImageViewSpec(
            normalize="minmax",
            colormap="gray",
            max_display_size=max_display_size,
        ),
    )
    show_images(
        [reference.detail.image.cpu(), moving.detail.image.cpu()],
        spec=spec,
        display=DisplaySpec(title=title, show=show, save_path=str(path)),
    )


def _physical_ref_from_mov(ref: SourceInfo, mov: SourceInfo) -> np.ndarray:
    if ref.pixel_size_xy is None or mov.pixel_size_xy is None:
        raise ValueError("Physical placement requires pixel_size_xy metadata.")
    ref_sx, ref_sy = ref.pixel_size_xy
    mov_sx, mov_sy = mov.pixel_size_xy
    ref_ox, ref_oy = ref.origin_xy or (0.0, 0.0)
    mov_ox, mov_oy = mov.origin_xy or (0.0, 0.0)
    return np.array(
        [
            [float(mov_sx) / float(ref_sx), 0.0, (float(mov_ox) - float(ref_ox)) / float(ref_sx)],
            [0.0, float(mov_sy) / float(ref_sy), (float(mov_oy) - float(ref_oy)) / float(ref_sy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _apply_synthetic_moving_warp(
    moving: SpatialImage,
    params: Sequence[float],
) -> SpatialImage:
    angle, scale_x, scale_y, shear_x, shear_y, tx, ty = (float(v) for v in params)
    detail = moving.detail.clone()
    dtype = detail.image.dtype
    device = detail.image.device
    detail.add_warp_params(
        angle=torch.tensor([angle], dtype=dtype, device=device),
        scale_x=torch.tensor([scale_x], dtype=dtype, device=device),
        scale_y=torch.tensor([scale_y], dtype=dtype, device=device),
        shear_x=torch.tensor([shear_x], dtype=dtype, device=device),
        shear_y=torch.tensor([shear_y], dtype=dtype, device=device),
        tx=torch.tensor([tx], dtype=dtype, device=device),
        ty=torch.tensor([ty], dtype=dtype, device=device),
    )
    detail = PrepareWarpStage(inverse=True, inplace=False)(detail)
    detail = WarpImageStage(inplace=False)(detail)
    return moving.with_detail(detail)


def main() -> None:
    args = _parse_args()
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = auto_device(args.device)
    moving_path = args.moving or args.reference
    ref_channel = _parse_channel(args.reference_channel)
    mov_channel = (
        _parse_channel(args.moving_channel) if args.moving_channel is not None else ref_channel
    )

    _print_source_summary("reference source", args.reference)
    if moving_path != args.reference:
        _print_source_summary("moving source", moving_path)

    reference = load_image(
        args.reference,
        channels=ref_channel,
        region=RegionYXHW.from_list(args.reference_region),
        downsample=1,
        grayscale=False,
    )
    moving = load_image(
        moving_path,
        channels=mov_channel,
        region=RegionYXHW.from_list(args.moving_region),
        downsample=1,
        grayscale=False,
    )
    _validate_physical_metadata(reference, moving)

    synthetic_artifact: dict[str, object] | None = None
    if args.synthetic_moving_warp is not None:
        moving = _apply_synthetic_moving_warp(moving, args.synthetic_moving_warp)
        synthetic_artifact = {
            "angle": float(args.synthetic_moving_warp[0]),
            "scale_x": float(args.synthetic_moving_warp[1]),
            "scale_y": float(args.synthetic_moving_warp[2]),
            "shear_x": float(args.synthetic_moving_warp[3]),
            "shear_y": float(args.synthetic_moving_warp[4]),
            "tx": float(args.synthetic_moving_warp[5]),
            "ty": float(args.synthetic_moving_warp[6]),
            "scope": "loaded moving tile before search",
        }
        print(
            "Applied synthetic moving tile warp before search: "
            f"{synthetic_artifact}. Full-resolution TIFF exports still use the "
            "original source files."
        )

    _save_pair_preview(
        reference,
        moving,
        path=args.output_dir / "preview_inputs.png",
        title="TIFF Inputs",
        show=args.show,
        max_display_size=args.max_display_size,
    )

    prepare = (
        SpatialPhysicalResize(reference_index=0)
        >> SpatialTargetResize(inner=TargetResize(target_width=args.search_dim))
        >> SpatialCenterPad(
            inner=CenterPad(
                scale=args.pad_scale,
                window_alpha=0.05,
                pad_to_pow2=False,
                outputs=["image", "box", "mask", "quad", "window"],
            )
        )
    )
    reference_search, moving_search = prepare([reference.clone(), moving.clone()])

    _save_pair_preview(
        reference_search,
        moving_search,
        path=args.output_dir / "preview_search_space.png",
        title="Search Space After Physical Resize, Low-Res Resize, and Padding",
        show=args.show,
        max_display_size=args.max_display_size,
    )

    print(f"search reference shape: {tuple(reference_search.detail.image.shape)}")
    print(f"search moving shape:    {tuple(moving_search.detail.image.shape)}")
    print(f"device: {device}")

    search_params = _build_search_params(args)
    config = ExhaustiveSearchConfig(
        batch_size=args.batch_size,
        translation_method=args.metric,
        progress_enabled=True,
        device=device,
    )
    search = ExhaustiveWarpSearch(search_params, config).to(device)

    start_t = time.perf_counter()
    result = search.search(
        reference_search.to(device).detail,
        moving_search.to(device).detail,
        top_k=args.top_k,
        progress=True,
    )
    print(f"Search took {time.perf_counter() - start_t:.2f} s.")

    result_cpu = result.to("cpu")
    warp = result_cpu.warp
    tr = result_cpu.translation_results
    print(
        "best warp: "
        f"angle={warp.angle[0].item():.3f}, "
        f"scale_x={warp.scale_x[0].item():.5f}, "
        f"scale_y={warp.scale_y[0].item():.5f}, "
        f"shear_x={warp.shear_x[0].item():.3f}, "
        f"shear_y={warp.shear_y[0].item():.3f}"
    )
    print(
        "best translation (search pixels): "
        f"tx={tr.tx[0].item():.2f}, ty={tr.ty[0].item():.2f}, "
        f"score={tr.score[0].item():.4f}"
    )

    reference_search_cpu = reference_search.to("cpu")
    moving_search_cpu = moving_search.to("cpu")
    transform = RegistrationTransform.from_search(
        moving=moving_search_cpu,
        reference=reference_search_cpu,
        search_result=result_cpu,
    )

    low_registered = warp_to_reference(
        moving_search_cpu.detail,
        transform.matrix_ref_search_from_mov_search,
        out_hw=reference_search_cpu.shape_hw,
    )
    overlay_spec = OverlaySpec(alpha=0.5)
    checkerboard_spec = CheckerboardSpec(
        tiles=(32, 32),
        edge_overlay=EdgeOverlaySpec(edge_source="mov", edge_threshold=0.3),
    )
    show_comparison(
        reference_search_cpu.detail,
        low_registered,
        mode="overlay",
        spec=overlay_spec,
        display=DisplaySpec(
            title="Low-Resolution Registered Overlay",
            show=args.show,
            save_path=str(args.output_dir / "preview_search_registered_overlay.png"),
        ),
    )
    show_comparison(
        reference_search_cpu.detail,
        low_registered,
        mode="checkerboard",
        spec=checkerboard_spec,
        display=DisplaySpec(
            title="Low-Resolution Registered Checkerboard",
            show=args.show,
            save_path=str(args.output_dir / "preview_search_registered_checkerboard.png"),
        ),
    )

    full_preview = None
    if not args.no_full_preview:
        full_reference = load_image(args.reference, channels=ref_channel, downsample=1, grayscale=False)
        full_moving = load_image(moving_path, channels=mov_channel, downsample=1, grayscale=False)
        full_preview = render_registration_preview(
            transform,
            full_moving.to("cpu"),
            full_reference.to("cpu"),
            crop_mode="union",
            preview_size=tuple(args.preview_size),
            tile_size=args.tile_size,
        )

    export_artifacts: dict[str, object] = {}
    if synthetic_artifact is not None:
        export_artifacts["synthetic_moving_warp"] = synthetic_artifact

    if not args.no_export:
        if args.export_channels != "all":
            print(
                "--export-channels is deprecated for register_tiff stacked export; "
                "writing all reference and moving channels."
            )

        after_path = args.output_dir / args.export_name
        after_artifact = export_stacked_registered_tiff(
            transform,
            reference_path=args.reference,
            moving_path=moving_path,
            output_path=after_path,
            tile_size=args.tile_size,
            pyramid_levels=args.pyramid_levels,
            streaming=not args.eager_export,
            overwrite=True,
            compression=args.export_compression,
        )
        after_bbox = tuple(int(v) for v in after_artifact["bbox_ref_full_xyxy"])
        mask_artifact = export_moving_mask_tiff(
            transform,
            moving_path=moving_path,
            reference_path=args.reference,
            output_path=args.output_dir / args.mask_name,
            bbox_ref_full_xyxy=after_bbox,
            tile_size=args.tile_size,
            pyramid_levels=args.pyramid_levels,
            streaming=not args.eager_export,
            overwrite=True,
            compression=args.export_compression,
        )

        before_matrix = _physical_ref_from_mov(
            transform.reference.source,
            transform.moving.source,
        )
        before_artifact = export_stacked_registered_tiff(
            transform,
            reference_path=args.reference,
            moving_path=moving_path,
            matrix_ref_from_mov=before_matrix,
            output_path=args.output_dir / args.before_export_name,
            tile_size=args.tile_size,
            pyramid_levels=args.pyramid_levels,
            streaming=not args.eager_export,
            overwrite=True,
            compression=args.export_compression,
        )
        export_artifacts["exports"] = {
            "registered_union": after_artifact,
            "unregistered_union": before_artifact,
            "registered_moving_mask": mask_artifact,
        }
        print(f"exported registered union stack: {after_path}")
        print(f"exported unregistered union stack: {args.output_dir / args.before_export_name}")
        print(f"exported moving mask: {args.output_dir / args.mask_name}")
        print(f"registered bbox_ref_full_xyxy: {after_artifact['bbox_ref_full_xyxy']}")
        print(f"unregistered bbox_ref_full_xyxy: {before_artifact['bbox_ref_full_xyxy']}")

    store = ProductIO(args.output_dir)
    manifest_path = store.save(
        transform,
        preview=full_preview,
        debug_details={"search_result": result_cpu},
        extra_artifacts=export_artifacts,
        overwrite=True,
    )
    print(f"manifest: {manifest_path}")
    if full_preview is not None:
        print(f"full preview output_hw: {full_preview.output_hw}")


if __name__ == "__main__":
    main()
