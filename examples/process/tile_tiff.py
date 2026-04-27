"""Tile a TIFF image into rectangular tiles based on physical size.
Run:
    uv run python examples/process/tile_tiff.py
    uv run python examples/process/tile_tiff.py --image path/to/large.tif --tile-size 100 100
    uv run python examples/process/tile_tiff.py --tile-size-pixels 512 512
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from batchmatch.io import (
    PreviewConfig,
    RegionYXHW,
    TiffExportConfig,
    open_image,
    save_preview,
    save_tiff,
)
from batchmatch.io.images import ImagePolicy
from batchmatch.io.space import SourceInfo


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=root / "img" / "FISH.tif",
        help="Path to the input TIFF image.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help=(
            "Tile size in physical (spatial) units matching the TIFF's "
            "pixel_size_xy (e.g. microns).  Mutually exclusive with "
            "--tile-size-pixels."
        ),
    )
    parser.add_argument(
        "--tile-size-pixels",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help=(
            "Tile size in pixels.  Used when the TIFF lacks spatial "
            "calibration or when you want exact pixel dimensions."
        ),
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help=(
            "Channel name or integer index.  Repeat to select multiple "
            "channels.  Defaults to all channels."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help=(
            "Overlap between adjacent tiles, expressed in the same units "
            "as the tile size (physical or pixel).  Default: 0."
        ),
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Integer downsample factor applied when reading each tile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "tiles",
        help="Directory to write tile TIFFs and manifest.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tile",
        help="Filename prefix for tile TIFFs (default: 'tile').",
    )
    # --- Preview options --------------------------------------------------
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a PNG preview alongside each tile.",
    )
    parser.add_argument(
        "--preview-normalize",
        type=str,
        default="minmax",
        choices=["minmax", "none", "percentile", "abs"],
        help="Normalization mode for previews (default: minmax).",
    )
    parser.add_argument(
        "--preview-gamma",
        type=float,
        default=1.0,
        help="Gamma correction for previews (default: 1.0).",
    )
    parser.add_argument(
        "--preview-colormap",
        type=str,
        default=None,
        help="Matplotlib colormap for single-channel previews.",
    )
    parser.add_argument(
        "--preview-max-size",
        type=int,
        default=None,
        help="Cap max(H,W) of preview images.",
    )
    return parser.parse_args()


def _parse_channel(token: str) -> int | str:
    try:
        return int(token)
    except ValueError:
        return token


def _physical_to_pixels(
    tile_phys_h: float,
    tile_phys_w: float,
    pixel_size_xy: tuple[float, float],
) -> tuple[int, int]:
    """Convert a physical tile size to pixels using the image calibration."""
    px_w, px_h = pixel_size_xy  # pixel_size_xy is (x, y)
    tile_h_px = max(1, round(tile_phys_h / px_h))
    tile_w_px = max(1, round(tile_phys_w / px_w))
    return tile_h_px, tile_w_px


def _compute_tile_grid(
    image_h: int,
    image_w: int,
    tile_h: int,
    tile_w: int,
    overlap_h: int,
    overlap_w: int,
) -> list[RegionYXHW]:
    """Return a list of non-overflowing tile regions covering the image."""
    step_h = max(1, tile_h - overlap_h)
    step_w = max(1, tile_w - overlap_w)

    tiles: list[RegionYXHW] = []
    y = 0
    while y < image_h:
        h = min(tile_h, image_h - y)
        x = 0
        while x < image_w:
            w = min(tile_w, image_w - x)
            tiles.append(RegionYXHW(y=y, x=x, h=h, w=w))
            x += step_w
        y += step_h
    return tiles


def _build_tile_source_info(
    source: SourceInfo,
    region: RegionYXHW,
) -> SourceInfo:
    """Derive a SourceInfo for a tile, adjusting origin to its position."""
    origin_xy = None
    physical_extent = None
    if source.pixel_size_xy is not None:
        px_w, px_h = source.pixel_size_xy
        ox = (source.origin_xy[0] if source.origin_xy else 0.0) + region.x * px_w
        oy = (source.origin_xy[1] if source.origin_xy else 0.0) + region.y * px_h
        origin_xy = (ox, oy)
        physical_extent = (region.w * px_w, region.h * px_h)

    from dataclasses import replace

    return replace(
        source,
        level_shapes=((region.h, region.w),),
        level_count=1,
        origin_xy=origin_xy,
        physical_extent_xy=physical_extent,
    )


def main() -> None:
    args = _parse_args()

    src = open_image(args.image)
    source = src.source
    base_h, base_w = source.base_shape_hw

    print(f"TIFF: {args.image}")
    print(f"  axes={source.axes}  shape=({base_h}, {base_w})")
    print(f"  levels={source.level_count}  dtype={source.dtype}")
    if source.channel_names:
        print(f"  channels={list(source.channel_names)}")
    if source.pixel_size_xy is not None:
        print(f"  pixel_size_xy={source.pixel_size_xy} {source.unit or ''}".rstrip())

    if args.tile_size is not None and args.tile_size_pixels is not None:
        raise ValueError("Specify --tile-size or --tile-size-pixels, not both.")

    if args.tile_size is not None:
        if source.pixel_size_xy is None:
            raise ValueError(
                "--tile-size requires spatial calibration (pixel_size_xy) in "
                "the TIFF metadata.  Use --tile-size-pixels instead."
            )
        tile_h_px, tile_w_px = _physical_to_pixels(
            args.tile_size[0], args.tile_size[1], source.pixel_size_xy
        )
        overlap_h_phys = args.overlap
        overlap_w_phys = args.overlap
        overlap_h = max(0, round(overlap_h_phys / source.pixel_size_xy[1]))
        overlap_w = max(0, round(overlap_w_phys / source.pixel_size_xy[0]))
        print(
            f"  tile (physical): {args.tile_size[0]} x {args.tile_size[1]} "
            f"{source.unit or 'units'} -> {tile_h_px} x {tile_w_px} px"
        )
    elif args.tile_size_pixels is not None:
        tile_h_px, tile_w_px = args.tile_size_pixels
        overlap_h = max(0, int(args.overlap))
        overlap_w = max(0, int(args.overlap))
    else:
        # Default: split into ~4 tiles
        tile_h_px = max(1, math.ceil(base_h / 2))
        tile_w_px = max(1, math.ceil(base_w / 2))
        overlap_h = 0
        overlap_w = 0
        print(f"  [default] tile size: {tile_h_px} x {tile_w_px} px (2x2 grid)")

    tiles = _compute_tile_grid(base_h, base_w, tile_h_px, tile_w_px, overlap_h, overlap_w)
    print(f"  tiles: {len(tiles)} ({tile_h_px}x{tile_w_px} px, overlap {overlap_h}x{overlap_w} px)")

    channels = None
    if args.channel:
        channels = [_parse_channel(c) for c in args.channel]
        print(f"  channels: {channels}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiff_config = TiffExportConfig(
        format="ome-tiff",
        photometric="auto",
        overwrite=True,
    )
    preview_config: PreviewConfig | None = None
    if args.preview:
        preview_config = PreviewConfig(
            normalize=args.preview_normalize,
            gamma=args.preview_gamma,
            colormap=args.preview_colormap,
            max_size=args.preview_max_size,
            overwrite=True,
        )
    manifest_tiles: list[dict] = []

    for idx, region in enumerate(tiles):
        spatial_img = src.read(
            region=region,
            downsample=args.downsample,
            channels=channels,
            policy=ImagePolicy(grayscale=False, dtype=None),
        )

        tile_name = f"{args.prefix}_{idx:04d}.tif"
        tile_path = args.output_dir / tile_name

        tile_source = _build_tile_source_info(source, region)

        save_tiff(
            spatial_img,
            tile_path,
            config=tiff_config,
            source=tile_source,
        )

        preview_name: str | None = None
        if preview_config is not None:
            preview_name = f"{args.prefix}_{idx:04d}.png"
            preview_path = args.output_dir / preview_name
            save_preview(spatial_img, preview_path, config=preview_config)

        tile_entry = {
            "index": idx,
            "filename": tile_name,
            "preview": preview_name,
            "region_yxhw": region.to_list(),
            "shape_hw": [
                int(spatial_img.detail.image.shape[-2]),
                int(spatial_img.detail.image.shape[-1]),
            ],
            "downsample": args.downsample,
        }
        if tile_source.origin_xy is not None:
            tile_entry["origin_xy"] = list(tile_source.origin_xy)
        if tile_source.physical_extent_xy is not None:
            tile_entry["physical_extent_xy"] = list(tile_source.physical_extent_xy)

        manifest_tiles.append(tile_entry)
        print(
            f"  [{idx + 1}/{len(tiles)}] {tile_name}  "
            f"region=({region.y},{region.x},{region.h},{region.w})  "
            f"shape={tile_entry['shape_hw']}"
        )

    manifest = {
        "source": {
            "path": str(args.image.resolve()),
            "shape_hw": [base_h, base_w],
            "axes": source.axes,
            "channel_names": list(source.channel_names),
            "pixel_size_xy": list(source.pixel_size_xy) if source.pixel_size_xy else None,
            "unit": source.unit,
            "format": source.format,
        },
        "tiling": {
            "tile_size_hw_px": [tile_h_px, tile_w_px],
            "overlap_hw_px": [overlap_h, overlap_w],
            "downsample": args.downsample,
            "channels": (
                [str(c) for c in channels] if channels else None
            ),
        },
        "tiles": manifest_tiles,
    }
    manifest_path = args.output_dir / "tiles.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Wrote {len(tiles)} tiles to {args.output_dir}")

    src.close()


if __name__ == "__main__":
    main()
