"""Register TIFF inputs and export the aligned result as TIFF.

Default run uses two overlapping crops of ``img/FISH.tif`` so the example
works against the sample data in the repo. Point ``--reference`` and
``--moving`` at separate TIFFs for a real run.

Run:
    uv run examples/registration/register_tiff.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from batchmatch import auto_device
from batchmatch.io import ProductIO, RegionYXHW, export_registered, load_image, open_image
from batchmatch.io.space import SpatialImage
from batchmatch.process.spatial_stages import SpatialCenterPad
from batchmatch.search.transform import RegistrationTransform
from batchmatch.translate import build_translation_stage
from batchmatch.view.preview import render_registration_preview


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
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--metric", choices=("ncc", "pc", "cc"), default="ncc")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs") / "register_tiff"
    )
    parser.add_argument("--export-name", type=str, default="registered_fullres.ome.tif")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip full-resolution export (manifest only).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a full-resolution preview PNG (memory-intensive).",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(1024, 1024),
        help="Downsize preview images to this H W before saving.",
    )
    parser.add_argument(
        "--export-channels",
        type=str,
        default="registration",
        help="'all' or 'registration' (reopen moving with the registration channel only).",
    )
    return parser.parse_args()


def _parse_channel(value: str | None) -> int | str | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _print_source_summary(label: str, path: Path) -> None:
    with open_image(path) as src:
        info = src.source
    print(f"{label}: {path}")
    print(f"  axes={info.axes} levels={info.level_count}")
    if info.channel_names:
        print(f"  channels={info.channel_names}")
    if info.pixel_size_xy is not None:
        print(f"  pixel_size_xy={info.pixel_size_xy} {info.unit or ''}".rstrip())


def main() -> None:
    args = _parse_args()
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
        downsample=args.downsample,
        grayscale=False,
    ).to(device)
    moving = load_image(
        moving_path,
        channels=mov_channel,
        region=RegionYXHW.from_list(args.moving_region),
        downsample=args.downsample,
        grayscale=False,
    ).to(device)

    if reference.detail.image.shape != moving.detail.image.shape:
        raise ValueError(
            "Reference and moving images must match after load/downsample. "
            f"Got {tuple(reference.detail.image.shape)} vs {tuple(moving.detail.image.shape)}."
        )
    print(f"search image shape (pre-pad): {tuple(reference.detail.image.shape)}")

    pad = SpatialCenterPad(scale=2, outputs=["image", "mask", "window"])
    ref_search, mov_search = pad([reference.clone(), moving.clone()])

    result = build_translation_stage(args.metric)(ref_search.detail, mov_search.detail)
    result.add_warp_params(
        angle=0.0, scale_x=1.0, scale_y=1.0, shear_x=0.0, shear_y=0.0, tx=0.0, ty=0.0
    )
    tr = result.translation_results
    print(
        "estimated translation (search pixels): "
        f"tx={tr.tx.item():.2f}, ty={tr.ty.item():.2f}, score={tr.score.item():.4f}"
    )

    transform = RegistrationTransform.from_search(
        moving=mov_search, reference=ref_search, search_result=result.to("cpu")
    )

    preview = None
    if args.preview:
        full_ref = load_image(args.reference, channels=ref_channel, downsample=1, grayscale=False)
        full_mov = load_image(moving_path, channels=mov_channel, downsample=1, grayscale=False)
        preview = render_registration_preview(
            transform,
            full_mov,
            full_ref,
            crop_mode="union",
            preview_size=tuple(args.preview_size),
        )

    store = ProductIO(args.output_dir)
    manifest_path = store.save(transform, preview=preview, overwrite=True)
    print(f"manifest: {manifest_path}")

    if not args.no_export:
        channels: str | list = "all"
        if args.export_channels == "registration":
            channels = (
                [mov_channel] if mov_channel is not None else "all"
            )
        export_path = export_registered(
            transform,
            output_path=args.output_dir / args.export_name,
            channels=channels,
            overwrite=True,
        )
        print(f"exported: {export_path}")

    if preview is not None:
        print(f"preview output_hw: {preview.output_hw}")


if __name__ == "__main__":
    main()
