"""Load a TIFF, inspect its metadata, and render per-channel previews.

Run:
  uv run python examples/process/tiff.py
  uv run python examples/process/tiff.py --show
  uv run python examples/process/tiff.py --channel DAPI --channel FITC
  uv run python examples/process/tiff.py --dpi 300 --figsize 20 20   # high-res output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from batchmatch.io import load_image, open_image
from batchmatch.view.config import DisplaySpec, GallerySpec, ImageViewSpec
from batchmatch.view.display import show_image, show_images


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=root / "img" / "FISH.tif", help="Path to a TIFF file.")
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help="Channel name or integer index. Repeat to select multiple channels. Defaults to all named channels.",
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("Y", "X", "H", "W"),
        default=None,
        help="Optional crop as y x h w before rendering.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Integer downsample factor applied on load.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "process",
        help="Directory to write preview figures.",
    )
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows (interactive).")
    parser.add_argument("--dpi", type=int, default=110, help="DPI for saved figures (default: 110).")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Figure size in inches as width height (default: auto).",
    )
    parser.add_argument(
        "--per-image-size",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(3.5, 3.5),
        help="Per-image size in inches for gallery views (default: 3.5 3.5).",
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=None,
        help="Max pixel extent for display; None keeps native resolution (default: None).",
    )
    return parser.parse_args()


def _parse_channel(token: str) -> int | str:
    try:
        return int(token)
    except ValueError:
        return token


def _default_channels(channel_names: list[str]) -> list[int | str]:
    if channel_names:
        return [name for name in channel_names if name]
    return [0]


def _channel_label(channel: int | str) -> str:
    return f"channel_{channel}" if isinstance(channel, int) else channel


def _rgb_from_channels(images: list[torch.Tensor]) -> torch.Tensor | None:
    if len(images) < 3:
        return None
    rgb = torch.cat([img[:, :1] for img in images[:3]], dim=1)
    return rgb


def main() -> None:
    args = _parse_args()
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    handle = open_image(args.image)
    source = handle.source
    channels = [_parse_channel(token) for token in args.channel] if args.channel else _default_channels(source.channel_names)

    print(f"TIFF: {args.image}")
    print(f"  axes={source.axes}")
    print(f"  shape={source.base_shape_hw}")
    print(f"  dtype={source.dtype}")
    print(f"  levels={source.level_count}")
    if source.channel_names:
        print(f"  channel_names={list(source.channel_names)}")
    if source.pixel_size_xy is not None:
        print(f"  pixel_size_xy={source.pixel_size_xy} {source.unit or ''}".rstrip())
    if args.region is not None:
        print(f"  region={tuple(args.region)}")
    print(f"  downsample={args.downsample}")

    loaded_channels: list[torch.Tensor] = []
    for channel in channels:
        loaded = load_image(
            args.image,
            channels=channel,
            region=tuple(args.region) if args.region is not None else None,
            downsample=args.downsample,
            grayscale=False,
        )
        loaded_channels.append(loaded.detail.image.cpu())
        print(f"  loaded {_channel_label(channel)} -> shape={tuple(loaded.detail.image.shape)}")

    image_spec = ImageViewSpec(normalize="minmax", colormap="magma", max_display_size=args.max_display_size)
    figsize = tuple(args.figsize) if args.figsize else None

    show_images(
        loaded_channels,
        spec=GallerySpec(
            ncols=min(4, len(loaded_channels)),
            per_image_size=tuple(args.per_image_size),
            image_spec=image_spec,
        ),
        display=DisplaySpec(
            title="TIFF Channel Gallery",
            show=args.show,
            save_path=str(args.output_dir / "tiff_channel_gallery.png"),
            dpi=args.dpi,
            figsize=figsize,
        ),
    )

    rgb = _rgb_from_channels(loaded_channels)
    if rgb is not None:
        show_image(
            rgb,
            spec=ImageViewSpec(normalize="minmax", max_display_size=args.max_display_size),
            display=DisplaySpec(
                title="TIFF RGB Composite",
                show=args.show,
                save_path=str(args.output_dir / "tiff_rgb_composite.png"),
                dpi=args.dpi,
                figsize=figsize,
            ),
        )

    first = loaded_channels[0]
    show_image(
        first,
        spec=ImageViewSpec(normalize="minmax", colormap="gray", max_display_size=args.max_display_size),
        display=DisplaySpec(
            title=f"TIFF Preview: {_channel_label(channels[0])}",
            show=args.show,
            save_path=str(args.output_dir / "tiff_single_channel.png"),
            dpi=args.dpi,
            figsize=figsize,
        ),
    )

    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
