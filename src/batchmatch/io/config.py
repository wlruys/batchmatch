"""Configuration dataclasses for input/output operations."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "IOConfig",
    "OutputConfig",
]


@dataclass
class IOConfig:
    reference: str = "reference.png"
    moving: str = "moving.png"
    output_dir: str = "outputs/register"
    grayscale: bool = True
    timestamp_output: bool = True
    save_images: bool = True
    save_product_json: bool = True


@dataclass
class OutputConfig:
    apply_pad: bool = True
    apply_warp: bool = True
    apply_shift: bool = True
    scale_translation: bool = True
    scale_warp_translation: bool = True
    show_overlay: bool = True
