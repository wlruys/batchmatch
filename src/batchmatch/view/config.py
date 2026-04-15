from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple, Union

Color = Tuple[float, float, float]
ChannelSelection = Union[int, Tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class DisplaySpec:
    figsize: Optional[Tuple[float, float]] = None
    dpi: int = 110
    title: Optional[str] = None
    show: bool = True
    save_path: Optional[str] = None
    tight_layout: bool = True


@dataclass(frozen=True, slots=True)
class ImageViewSpec:
    normalize: Literal["minmax", "none", "percentile", "abs"] = "abs"
    normalize_per_image: bool = False
    gamma: float = 1.0
    colormap: str = "gray"
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    interpolation: str = "nearest"
    show_colorbar: bool = False
    colorbar_label: Optional[str] = None

    # Multi-channel support: select which channel(s) to display.
    #   None  -> auto (C==1: grayscale, C<=3: RGB, C>3: first channel)
    #   int   -> single channel, rendered with *colormap*
    #   (i,j,k) -> three channels mapped to (R, G, B)
    channel: Optional[ChannelSelection] = None

    # High-resolution support: downsample for display when max(H,W)
    # exceeds this limit.  None disables downsampling.
    max_display_size: Optional[int] = None


@dataclass(frozen=True, slots=True)
class GallerySpec:
    nrows: Optional[int] = None
    ncols: Optional[int] = None
    image_spec: ImageViewSpec = field(default_factory=ImageViewSpec)
    titles: Optional[Sequence[str]] = None
    share_colorbar: bool = False
    spacing: float = 0.1
    per_image_size: Tuple[float, float] = (4.0, 4.0)
    preserve_relative_size: bool = False
    channel_names: Optional[Sequence[str]] = None


@dataclass(frozen=True, slots=True)
class GradientViewSpec:
    component: Literal["x", "y", "norm", "orientation", "all"] = "all"

    signed_colormap: str = "RdBu_r"
    symmetric_range: bool = True

    norm_colormap: str = "viridis"
    norm_gamma: float = 1.0

    orientation_colormap: str = "hsv"
    orientation_as_color: bool = True

    show_quiver: bool = False
    quiver_step: int = 16
    quiver_scale: float = 1.0
    quiver_color: Color = (0.0, 0.0, 0.0)

    normalize: Literal["minmax", "percentile", "none"] = "minmax"
    normalize_per_image: bool = False
    percentile: float = 99.0


@dataclass(frozen=True, slots=True)
class GradientGallerySpec:
    show_components: Tuple[str, ...] = ("x", "y", "norm", "orientation")
    component_specs: Optional[dict] = None
    layout: Literal["row", "column", "grid"] = "row"
    display: DisplaySpec = field(default_factory=DisplaySpec)


@dataclass(frozen=True, slots=True)
class MaskViewSpec:
    mode: Literal["binary", "overlay", "contour", "alpha", "outline"] = "overlay"

    binary_colormap: str = "gray"
    invert: bool = False

    overlay_color: Color = (1.0, 0.0, 0.0)
    overlay_alpha: float = 0.4

    contour_color: Color = (0.0, 1.0, 0.0)
    contour_thickness: int = 2
    contour_levels: int = 1

    background_color: Color = (0.2, 0.2, 0.2)

    outline_color: Color = (1.0, 1.0, 0.0)
    outline_thickness: int = 2


@dataclass(frozen=True, slots=True)
class MaskOverlaySpec:
    mask_spec: MaskViewSpec = field(default_factory=MaskViewSpec)
    image_spec: ImageViewSpec = field(default_factory=ImageViewSpec)
    blend_mode: Literal["alpha", "multiply", "screen"] = "alpha"


@dataclass(frozen=True, slots=True)
class OverlaySpec:
    alpha: float = 0.5
    moving_color: Color = (1.0, 0.0, 0.0)
    normalize: Literal["minmax", "joint", "none"] = "minmax"
    normalize_per_image: bool = False
    grayscale: bool = True
    channel: Optional[ChannelSelection] = None


@dataclass(frozen=True, slots=True)
class EdgeOverlaySpec:
    edge_source: Literal["ref", "mov", "both", "none"] = "mov"
    edge_color: Color = (0.0, 1.0, 0.0)
    edge_alpha: float = 1.0
    edge_threshold: Optional[float] = 0.05
    edge_thickness: int = 3
    gamma: float = 0.6
    glow_radius: int = 0
    dim_under_edges: float = 0.7

    # Multi-channel support: select which channel(s) to use as the background
    # image when rendering the edge overlay.  None → auto (C<=3 keep as-is,
    # C>3 take first channel).
    channel: Optional[ChannelSelection] = None


@dataclass(frozen=True, slots=True)
class CheckerboardSpec:
    tiles: Optional[Tuple[int, int]] = (8, 8)
    tilesize: Optional[Tuple[int, int]] = None
    align: Literal["ul", "center"] = "ul"
    feather_px: int = 0

    normalize: Literal["minmax", "joint", "none"] = "minmax"
    normalize_per_image: bool = False
    gamma: float = 1.0
    grayscale: bool = True
    channel: Optional[ChannelSelection] = None

    ref_tint: Color = (1.0, 0.8, 0.7)
    mov_tint: Color = (0.7, 0.85, 1.0)
    ref_tint_strength: float = 0.6
    mov_tint_strength: float = 0.6
    ref_gain: float = 1.0
    mov_gain: float = 1.0

    grid: bool = True
    grid_color: Color = (0.0, 0.0, 0.0)
    grid_alpha: float = 0.35
    grid_thickness: int = 1

    edge_overlay: EdgeOverlaySpec = field(
        default_factory=lambda: EdgeOverlaySpec(edge_source="none")
    )


@dataclass(frozen=True, slots=True)
class QuadAnnotationSpec:
    color: Color = (1.0, 0.0, 0.0)
    thickness: int = 2
    fill: bool = False
    fill_alpha: float = 0.3
    label_key: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PointAnnotationSpec:
    color: Color = (0.0, 1.0, 0.0)
    radius: int = 3
    marker: Literal["circle", "cross", "x", "dot"] = "circle"


@dataclass(frozen=True, slots=True)
class BoxAnnotationSpec:
    color: Color = (0.0, 0.0, 1.0)
    thickness: int = 2
    fill: bool = False
    fill_alpha: float = 0.2
