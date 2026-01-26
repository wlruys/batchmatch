"""
Warp a batch of images and visualize quad, box, masks, and windows.

Run:
  uv run python examples/process/warp_demo.py
  uv run python examples/process/warp_demo.py --show
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from batchmatch.base import build_image_td
from batchmatch.io import ImageIO
from batchmatch.process.pad import build_pad_pipeline
from batchmatch.process.resize import build_resize_pipeline
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
from batchmatch.warp import build_warp_pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=root / "img/test.jpg", help="Path to an input image.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of warped copies to generate.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for processing (auto|cpu|cuda|mps).",
    )
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

    device = _resolve_device(args.device)
    img = ImageIO(grayscale=False, device=device).load(args.image)
    td = build_image_td(img)

    resize_pipe = build_resize_pipeline(
        "target_resize",
        target_width=256,
    )

    pad_pipe = build_pad_pipeline(
        "center_pad",
        scale=1.5,
        create_quad=True,
        create_box=True,
        create_mask=True,
        create_window=True,
    )

    warp_pipe = build_warp_pipeline(
        {
            "prepare": {"type": "prepare", "inverse": True},
            "image": {"type": "image", "mode": "bilinear"},
            "mask": {"type": "mask"},
            "window": {"type": "window"},
            "boxes": {"type": "boxes"},
            "quad": {"type": "quad"},
        }
    )

    B = max(1, int(args.batch_size))
    batch_td = td.repeat(B)

    dtype = batch_td.image.dtype
    if B == 1:
        angles = torch.tensor([0.0], device=device, dtype=dtype)
        scales = torch.tensor([1.0], device=device, dtype=dtype)
    else:
        angles = torch.linspace(0.0, 45.0, B, device=device, dtype=dtype)
        scales = torch.linspace(1.0, 0.85, B, device=device, dtype=dtype)

    batch_td.add_warp_params(angle=angles, scale_x=scales, scale_y=scales)

    pipe = resize_pipe >> pad_pipe >> warp_pipe
    warped = pipe(batch_td)

    quad_spec = QuadAnnotationSpec(color=(1.0, 0.0, 0.0), thickness=2, fill=False)
    box_spec = BoxAnnotationSpec(color=(0.0, 0.0, 1.0), thickness=2, fill=False)
    annotated = annotate_from_detail(warped, quad_spec=quad_spec, box_spec=box_spec)

    figsize = (max(8.0, 4.0 * B), 4.0)
    show_image(
        annotated,
        display=DisplaySpec(
            title="Warped Images (Quad + Box)",
            show=args.show,
            figsize=figsize,
            save_path=str(args.output_dir / "warp_demo_annotated.png"),
        ),
    )

    overlay_spec = MaskOverlaySpec(
        mask_spec=MaskViewSpec(
            mode="overlay",
            overlay_color=(0.0, 1.0, 0.0),
            overlay_alpha=0.3,
        ),
    )
    show_mask_overlay(
        warped,
        spec=overlay_spec,
        display=DisplaySpec(
            title="Warped Images (Mask Overlay)",
            show=args.show,
            figsize=figsize,
            save_path=str(args.output_dir / "warp_demo_mask_overlay.png"),
        ),
    )

    if warped.window is not None:
        show_image(
            warped.window,
            spec=ImageViewSpec(colormap="viridis", show_colorbar=True),
            display=DisplaySpec(
                title="Warped Windows",
                show=args.show,
                figsize=figsize,
                save_path=str(args.output_dir / "warp_demo_window.png"),
            ),
        )

    print(f"Device: {device}")
    print(f"Wrote outputs to: {args.output_dir}")

if __name__ == "__main__":
    main()
