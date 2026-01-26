from __future__ import annotations

"""
Randomly crop an image multiple times and display a gallery.

Run:
  uv run python examples/process/crop_demo.py
  uv run python examples/process/crop_demo.py --show
"""

import argparse
import os
from pathlib import Path

import torch

from batchmatch.base import build_image_td
from batchmatch.io import ImageIO
from batchmatch.process.crop import RandomCropStage
from batchmatch.view.config import DisplaySpec, GallerySpec, ImageViewSpec
from batchmatch.view.display import show_images


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=root / "FISH.jpg", help="Path to an input image.")
    parser.add_argument("--num-crops", type=int, default=8, help="Number of random crops to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cropping.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "process",
        help="Directory to write output figures.",
    )
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows (interactive).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    img = ImageIO(grayscale=True).load(args.image)
    detail = build_image_td(img).repeat(args.num_crops)

    min_side = min(detail.H, detail.W)
    min_crop = max(8, min_side // 4)
    min_crop = min(min_crop, min_side)
    max_crop = max(min_crop, min_side // 2)
    max_crop = min(max_crop, min_side)

    generator = torch.Generator().manual_seed(args.seed)
    crop_stage = RandomCropStage(
        min_size=(min_crop, min_crop),
        max_size=(max_crop, max_crop),
        generator=generator,
    )

    crops = crop_stage(detail)

    gallery_spec = GallerySpec(
        ncols=4,
        per_image_size=(3.5, 3.5),
        image_spec=ImageViewSpec(normalize="minmax"),
    )
    show_images(
        [crop.image for crop in crops],
        spec=gallery_spec,
        display=DisplaySpec(
            title="Random Crops",
            show=args.show,
            save_path=str(args.output_dir / "crop_gallery.png"),
        ),
    )
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
