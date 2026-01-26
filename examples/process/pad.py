"""
Pad an image and visualize the domain (quad/box/mask/window).

Run:
  uv run python examples/process/pad_demo.py
  uv run python examples/process/pad_demo.py --show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from batchmatch.base import build_image_td
from batchmatch.io import ImageIO
from batchmatch.process.pad import build_pad_pipeline
from batchmatch.view.annotate import annotate_from_detail
from batchmatch.view.config import (
    BoxAnnotationSpec,
    DisplaySpec,
    ImageViewSpec,
    MaskOverlaySpec,
    MaskViewSpec,
    QuadAnnotationSpec,
)
from batchmatch.view.display import show_image, show_mask_overlay


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=root / "img/test.jpg", help="Path to an input image.")
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

    img = ImageIO(grayscale=False).load(args.image)

    td = build_image_td(img)

    pad_pipe = build_pad_pipeline(
        "center_pad",
        scale=1.5,
        create_quad=True,
        create_box=True,
        create_mask=True,
        create_window=True,
    )

    padded_td = pad_pipe(td)

    quad_spec = QuadAnnotationSpec(
        color=(1.0, 0.0, 0.0),
        thickness=2,
        fill=False,
    )
    annotated_with_quad = annotate_from_detail(padded_td, quad_spec=quad_spec)
    show_image(
        annotated_with_quad,
        display=DisplaySpec(
            title="Padded Image with Quad",
            show=args.show,
            figsize=(8, 8),
            save_path=str(args.output_dir / "pad_demo_quad.png"),
        ),
    )

    box_spec = BoxAnnotationSpec(
        color=(0.0, 0.0, 1.0),
        thickness=2,
        fill=False,
    )
    annotated_with_box = annotate_from_detail(padded_td, box_spec=box_spec)
    show_image(
        annotated_with_box,
        display=DisplaySpec(
            title="Padded Image with Box",
            show=args.show,
            figsize=(8, 8),
            save_path=str(args.output_dir / "pad_demo_box.png"),
        ),
    )

    overlay_spec = MaskOverlaySpec(
        mask_spec=MaskViewSpec(
            mode="overlay",
            overlay_color=(0.0, 1.0, 0.0),  # Green
            overlay_alpha=0.3,
        ),
    )
    show_mask_overlay(
        padded_td,
        spec=overlay_spec,
        display=DisplaySpec(
            title="Mask Overlay",
            show=args.show,
            figsize=(8, 8),
            save_path=str(args.output_dir / "pad_demo_mask_overlay.png"),
        ),
    )

    if padded_td.window is not None:
        show_image(
            padded_td.window,
            spec=ImageViewSpec(colormap="viridis", show_colorbar=True),
            display=DisplaySpec(
                title="Window",
                show=args.show,
                figsize=(8, 8),
                save_path=str(args.output_dir / "pad_demo_window.png"),
            ),
        )

    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
