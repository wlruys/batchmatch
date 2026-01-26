"""
Compute image gradients and visualize the components.

Run this script from the repository root to load a sample image, compute
Sobel gradients, and display multiple gradient views.

Run:
  uv run python examples/process/gradient_demo.py
  uv run python examples/process/gradient_demo.py --show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from batchmatch.base import build_image_td
from batchmatch.gradient import build_gradient_pipeline
from batchmatch.io import ImageIO
from batchmatch.view.config import DisplaySpec, GradientGallerySpec, GradientViewSpec
from batchmatch.view.display import show_gradient, show_gradient_hsv, show_gradients


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=root / "FISH.jpg", help="Path to an input image.")
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

    image = ImageIO(grayscale=True).load(args.image)
    detail = build_image_td(image)

    gradient_pipe = build_gradient_pipeline(
        "cd",
        eta={"type": "mean", "scale": 0.01, "norm": "l2"},
        normalize={"type": "normalize", "norm": "l2"},
    )
    detail = gradient_pipe(detail)

    show_gradients(
        detail,
        spec=GradientGallerySpec(layout="row"),
        display=DisplaySpec(
            title="Gradient Components",
            show=args.show,
            figsize=(14, 4),
            save_path=str(args.output_dir / "gradient_demo_components.png"),
        ),
    )

    show_gradient(
        detail,
        component="norm",
        spec=GradientViewSpec(norm_colormap="magma"),
        display=DisplaySpec(
            title="Gradient Magnitude",
            show=args.show,
            figsize=(6, 6),
            save_path=str(args.output_dir / "gradient_demo_magnitude.png"),
        ),
    )

    show_gradient_hsv(
        detail,
        display=DisplaySpec(
            title="Gradient Orientation (HSV)",
            show=args.show,
            figsize=(6, 6),
            save_path=str(args.output_dir / "gradient_demo_hsv.png"),
        ),
    )
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
