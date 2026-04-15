"""Load a TIFF, inspect its metadata, and render per-channel previews.

Run:
  uv run python examples/process/tiff.py
  uv run python examples/process/tiff.py --show
  uv run python examples/process/tiff.py --channel DAPI --channel FITC
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
    meta = handle.meta
    channels = [_parse_channel(token) for token in args.channel] if args.channel else _default_channels(meta.channel_names)

    print(f"TIFF: {args.image}")
    print(f"  axes={meta.axes}")
    print(f"  shape={meta.shape}")
    print(f"  dtype={meta.dtype}")
    print(f"  levels={meta.level_count}")
    if meta.channel_names:
        print(f"  channel_names={meta.channel_names}")
    if meta.pixel_size_xy is not None:
        print(f"  pixel_size_xy={meta.pixel_size_xy} {meta.unit or ''}".rstrip())
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

    show_images(
        loaded_channels,
        spec=GallerySpec(
            ncols=min(4, len(loaded_channels)),
            per_image_size=(3.5, 3.5),
            image_spec=ImageViewSpec(normalize="minmax", colormap="magma"),
        ),
        display=DisplaySpec(
            title="TIFF Channel Gallery",
            show=args.show,
            save_path=str(args.output_dir / "tiff_channel_gallery.png"),
        ),
    )

    rgb = _rgb_from_channels(loaded_channels)
    if rgb is not None:
        show_image(
            rgb,
            spec=ImageViewSpec(normalize="minmax"),
            display=DisplaySpec(
                title="TIFF RGB Composite",
                show=args.show,
                save_path=str(args.output_dir / "tiff_rgb_composite.png"),
            ),
        )

    first = loaded_channels[0]
    show_image(
        first,
        spec=ImageViewSpec(normalize="minmax", colormap="gray"),
        display=DisplaySpec(
            title=f"TIFF Preview: {_channel_label(channels[0])}",
            show=args.show,
            save_path=str(args.output_dir / "tiff_single_channel.png"),
        ),
    )

    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
