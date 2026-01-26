from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

__all__ = [
    "ResizeConfig",
    "PadConfig",
    "CropOutputConfig",
    "RandomCropConfig",
    "MaskCropConfig",
    "CropConfig",
    "ProcessConfig",
]


@dataclass
class ResizeConfig:
    method: Literal["scale", "target"] = "scale"
    scale: Optional[float] = None
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    outputs: List[str] = field(default_factory=lambda: ["image"])

    def __post_init__(self):
        if self.method == "scale" and self.scale is None:
            raise ValueError("scale must be specified when method='scale'")
        if (
            self.method == "target"
            and self.target_width is None
            and self.target_height is None
        ):
            raise ValueError(
                "target_width or target_height must be specified when method='target'"
            )

@dataclass
class PadConfig:
    method: Literal["center"] = "center"
    scale: float = 2.0
    window_alpha: float = 0.05
    pad_to_pow2: bool = True
    pad_to_even: bool = False
    shrink_by: int = 4
    outputs: List[str] = field(
        default_factory=lambda: ["image", "box", "mask", "quad", "window"]
    )

@dataclass
class CropOutputConfig:
    """
    Note(wlr): This is a legacy config format for specifying crop outputs using boolean flags.
    It is recommended to use List[str] format instead. This is just an adapter for backward compatibility on test scripts.
    """
    crop_image: bool = True
    crop_mask: bool = True
    crop_window: bool = False
    crop_gradients: bool = False
    adjust_box: bool = False
    adjust_quad: bool = False
    adjust_points: bool = False
    clip_geometry: bool = True
    invalidate_warp: bool = True

    def to_outputs_list(self) -> List[str]:
        outputs = []
        if self.crop_image:
            outputs.append("image")
        if self.crop_mask:
            outputs.append("mask")
        if self.crop_window:
            outputs.append("window")
        if self.crop_gradients:
            outputs.extend(["gx", "gy"])
        if self.adjust_box:
            outputs.append("box")
        if self.adjust_quad:
            outputs.append("quad")
        if self.adjust_points:
            outputs.append("points")
        return outputs


@dataclass
class RandomCropConfig:
    min_size: Union[int, Tuple[int, int]] = 16
    max_size: Optional[Union[int, Tuple[int, int]]] = None
    min_area: Optional[int] = None
    max_area: Optional[int] = None
    min_aspect: Optional[float] = None
    max_aspect: Optional[float] = None
    max_attempts: int = 50
    allow_full_image: bool = False
    outputs: List[str] = field(default_factory=lambda: ["image", "mask"])
    clip_geometry: bool = True
    invalidate_warp: bool = True
    seed: Optional[int] = None


@dataclass
class MaskCropConfig:
    method: Literal["union", "intersection"] = "intersection"
    outputs: List[str] = field(default_factory=lambda: ["image", "mask"])
    clip_geometry: bool = True
    invalidate_warp: bool = True


@dataclass
class CropConfig:
    type: Literal["random", "union", "intersection", "none"] = "none"
    random: Optional[RandomCropConfig] = None
    mask: Optional[MaskCropConfig] = None

    def __post_init__(self):
        if self.type == "random" and self.random is None:
            self.random = RandomCropConfig()
        elif self.type in ("union", "intersection") and self.mask is None:
            self.mask = MaskCropConfig(method=self.type)


@dataclass
class ProcessConfig:
    crop: Optional[CropConfig] = None
    resize: Optional[ResizeConfig] = None
    pad: Optional[PadConfig] = field(default_factory=PadConfig)
