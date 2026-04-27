"""
Window generation stages for ImageDetail containers.

Includes Tukey window helpers and stages for box/quad windows.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail, NestedKey
from batchmatch.base.pipeline import Pipeline, Stage, StageRegistry, StageSpec, coerce_stage_spec

#Note: Cache is bc common boxes may be re-used often, for example when refilling an expanded image in a search
window_registry = StageRegistry("window")
_WINDOW_CACHE_SIZE = 32


@lru_cache(maxsize=_WINDOW_CACHE_SIZE)
def _cached_tukey_2d(
    H: int,
    W: int,
    alpha: float,
    device_type: str,
    device_index: Optional[int],
    dtype: torch.dtype,
) -> Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    return tukey_2d(H, W, alpha, device=device, dtype=dtype)


def get_cached_window(
    H: int,
    W: int,
    alpha: float = 0.5,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return _cached_tukey_2d(H, W, alpha, device.type, device.index, dtype)


def tukey_1d(
    N: int,
    alpha: float = 0.5,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.float32

    if N <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if N == 1:
        return torch.ones(1, device=device, dtype=dtype)

    alpha = max(0.0, min(1.0, alpha))
    if alpha == 0.0:
        return torch.ones(N, device=device, dtype=dtype)

    n = torch.arange(N, device=device, dtype=dtype)
    width = alpha * (N - 1) / 2.0

    t_left = n / width
    t_right = ((N - 1) - n) / width

    #taper: 0.5 * (1 - cos(pi * t))
    left_taper = 0.5 * (1.0 - torch.cos(math.pi * t_left))
    right_taper = 0.5 * (1.0 - torch.cos(math.pi * t_right))

    window = torch.where(n < width, left_taper,
              torch.where(n > (N - 1) - width, right_taper,
              torch.ones_like(n)))

    return window


def tukey_2d(
    H: int,
    W: int,
    alpha: float = 0.5,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    wy = tukey_1d(H, alpha, device=device, dtype=dtype)
    wx = tukey_1d(W, alpha, device=device, dtype=dtype)
    return wy.unsqueeze(1) * wx.unsqueeze(0)


def _tukey_from_t(t: Tensor, alpha: float) -> Tensor:
    if alpha <= 0:
        return torch.ones_like(t)

    alpha = min(1.0, alpha)
    half_alpha = alpha / 2.0

    t_left = t / half_alpha 
    t_right = (1.0 - t) / half_alpha

    left_taper = 0.5 * (1.0 - torch.cos(math.pi * t_left.clamp(0, 1)))
    right_taper = 0.5 * (1.0 - torch.cos(math.pi * t_right.clamp(0, 1)))

    result = torch.where(
        t < half_alpha,
        left_taper,
        torch.where(t > 1.0 - half_alpha, right_taper, torch.ones_like(t))
    )

    return result

def make_box_window_batched(
    H: int,
    W: int,
    boxes: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    device = boxes.device
    dtype = torch.float32

    if boxes.ndim == 3:
        boxes = boxes[:, 0, :]  # Take first box
    B = boxes.shape[0]

    x0 = boxes[:, 0].round().long().clamp(0, W)
    y0 = boxes[:, 1].round().long().clamp(0, H)
    x1 = boxes[:, 2].round().long().clamp(0, W)
    y1 = boxes[:, 3].round().long().clamp(0, H)

    gy = torch.arange(H, device=device, dtype=dtype)
    gx = torch.arange(W, device=device, dtype=dtype)

    gy_exp = gy.view(1, H, 1)
    gx_exp = gx.view(1, 1, W)

    x0_exp = x0.view(B, 1, 1).float()
    y0_exp = y0.view(B, 1, 1).float()
    x1_exp = x1.view(B, 1, 1).float()
    y1_exp = y1.view(B, 1, 1).float()

    inside_x = (gx_exp >= x0_exp) & (gx_exp < x1_exp)
    inside_y = (gy_exp >= y0_exp) & (gy_exp < y1_exp)
    inside = inside_x & inside_y

    # positions within box [0, 1]
    box_w = (x1_exp - x0_exp).clamp(min=1)
    box_h = (y1_exp - y0_exp).clamp(min=1)
    t_x = (gx_exp - x0_exp) / (box_w - 1).clamp(min=1)
    t_y = (gy_exp - y0_exp) / (box_h - 1).clamp(min=1)

    w_x = _tukey_from_t(t_x, alpha)
    w_y = _tukey_from_t(t_y, alpha)

    window = w_x * w_y
    window = torch.where(inside, window, torch.zeros_like(window))

    return window


def make_box_window(
    H: int,
    W: int,
    box: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    return make_box_window_batched(H, W, box.unsqueeze(0), alpha)[0]

def _make_quad_window_general(
    H: int,
    W: int,
    corners: Tensor,
    alpha: float,
) -> Tensor:
    device = corners.device
    dtype = torch.float32
    B = corners.shape[0]

    c0, c1, c2, c3 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    e0, e1, e2, e3 = c1 - c0, c2 - c1, c3 - c2, c0 - c3

    len0 = torch.sqrt(e0[:, 0] ** 2 + e0[:, 1] ** 2).clamp(min=1e-8)
    len1 = torch.sqrt(e1[:, 0] ** 2 + e1[:, 1] ** 2).clamp(min=1e-8)
    len2 = torch.sqrt(e2[:, 0] ** 2 + e2[:, 1] ** 2).clamp(min=1e-8)
    len3 = torch.sqrt(e3[:, 0] ** 2 + e3[:, 1] ** 2).clamp(min=1e-8)

    nx0, ny0 = -e0[:, 1] / len0, e0[:, 0] / len0
    nx1, ny1 = -e1[:, 1] / len1, e1[:, 0] / len1
    nx2, ny2 = -e2[:, 1] / len2, e2[:, 0] / len2
    nx3, ny3 = -e3[:, 1] / len3, e3[:, 0] / len3

    x0_0, y0_0 = c0[:, 0], c0[:, 1]
    x0_1, y0_1 = c1[:, 0], c1[:, 1]
    x0_2, y0_2 = c2[:, 0], c2[:, 1]
    x0_3, y0_3 = c3[:, 0], c3[:, 1]

    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    gx = gx.unsqueeze(0)
    gy = gy.unsqueeze(0)

    nx0, ny0 = nx0.view(B, 1, 1), ny0.view(B, 1, 1)
    nx1, ny1 = nx1.view(B, 1, 1), ny1.view(B, 1, 1)
    nx2, ny2 = nx2.view(B, 1, 1), ny2.view(B, 1, 1)
    nx3, ny3 = nx3.view(B, 1, 1), ny3.view(B, 1, 1)

    x0_0, y0_0 = x0_0.view(B, 1, 1), y0_0.view(B, 1, 1)
    x0_1, y0_1 = x0_1.view(B, 1, 1), y0_1.view(B, 1, 1)
    x0_2, y0_2 = x0_2.view(B, 1, 1), y0_2.view(B, 1, 1)
    x0_3, y0_3 = x0_3.view(B, 1, 1), y0_3.view(B, 1, 1)

    #distances to edges
    d0 = (gx - x0_0) * nx0 + (gy - y0_0) * ny0
    d1 = (gx - x0_1) * nx1 + (gy - y0_1) * ny1
    d2 = (gx - x0_2) * nx2 + (gy - y0_2) * ny2
    d3 = (gx - x0_3) * nx3 + (gy - y0_3) * ny3

    # check if inside quad
    inside = (d0 > 0) & (d1 > 0) & (d2 > 0) & (d3 > 0)

    #positions between opposite edges
    t1 = d0 / (d0 + d2).clamp(min=1e-8)
    t2 = d3 / (d3 + d1).clamp(min=1e-8)

    if alpha <= 0:
        return inside.to(dtype)

    half_alpha = min(1.0, alpha) / 2.0

    tl1 = (t1 / half_alpha).clamp_(0, 1)
    tl2 = (t2 / half_alpha).clamp_(0, 1)
    tr1 = ((1.0 - t1) / half_alpha).clamp_(0, 1)
    tr2 = ((1.0 - t2) / half_alpha).clamp_(0, 1)

    taper1 = torch.minimum(tl1, tr1)
    taper2 = torch.minimum(tl2, tr2)

    w1 = 0.5 * (1.0 - torch.cos(math.pi * taper1))
    w2 = 0.5 * (1.0 - torch.cos(math.pi * taper2))

    window = w1 * w2 * inside.to(dtype)
    return window


def make_quad_window_batched(
    H: int,
    W: int,
    quads: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    if quads.ndim == 3:
        quads = quads[:, 0, :]
    B = quads.shape[0]

    #TODO(wlr): Implement fast path for axis-aligned quads
    corners = quads.view(B, 4, 2).to(dtype=torch.float32)
    return _make_quad_window_general(H, W, corners, alpha)


def make_quad_window(
    H: int,
    W: int,
    quad: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    return make_quad_window_batched(H, W, quad.unsqueeze(0), alpha)[0]


@window_registry.register("tukey_box", "box_tukey")
class TukeyBoxWindow(Stage):
    requires: ClassVar[frozenset[NestedKey]] = frozenset({
        ImageDetail.Keys.IMAGE,
        ImageDetail.Keys.DOMAIN.BOX,
    })
    sets: ClassVar[frozenset[NestedKey]] = frozenset({
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(self, *, alpha: float = 0.5) -> None:
        super().__init__()
        self._alpha = float(alpha)

    def forward(self, image: ImageDetail) -> ImageDetail:
        img = image.get(ImageDetail.Keys.IMAGE)
        box = image.get(ImageDetail.Keys.DOMAIN.BOX)

        B, C, H, W = img.shape

        window = make_box_window_batched(H, W, box, alpha=self._alpha)
        window = window.unsqueeze(1).to(dtype=img.dtype)
        image.set(ImageDetail.Keys.DOMAIN.WINDOW, window)
        return image

    def __repr__(self) -> str:
        return f"TukeyBoxWindow(alpha={self._alpha})"


@window_registry.register("tukey_quad", "quad_tukey")
class TukeyQuadWindow(Stage):
    requires: ClassVar[frozenset[NestedKey]] = frozenset({
        ImageDetail.Keys.IMAGE,
        ImageDetail.Keys.DOMAIN.QUAD,
    })
    sets: ClassVar[frozenset[NestedKey]] = frozenset({
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(self, *, alpha: float = 0.5) -> None:
        super().__init__()
        self._alpha = float(alpha)

    def forward(self, image: ImageDetail) -> ImageDetail:
        img = image.get(ImageDetail.Keys.IMAGE)
        quad = image.get(ImageDetail.Keys.DOMAIN.QUAD)

        B, C, H, W = img.shape
        window = make_quad_window_batched(H, W, quad, alpha=self._alpha)
        window = window.unsqueeze(1).to(dtype=img.dtype)

        image.set(ImageDetail.Keys.DOMAIN.WINDOW, window)
        return image

    def __repr__(self) -> str:
        return f"TukeyQuadWindow(alpha={self._alpha})"


@window_registry.register("apply", "apply_window")
class ApplyWindowStage(Stage):
    """Multiply targets by the domain window.

    .. deprecated::
        Windowing is now applied inside the search pipeline.
        This stage will be removed in a future release.
    """

    requires: ClassVar[frozenset[NestedKey]] = frozenset({
        ImageDetail.Keys.DOMAIN.WINDOW,
    })
    sets: ClassVar[frozenset[NestedKey]] = frozenset()

    def __init__(
        self,
        *,
        targets: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        warnings.warn(
            "ApplyWindowStage is deprecated; windowing is now applied "
            "inside the search pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )

        if targets is None:
            targets = ["image"]

        target_set = {t.lower() for t in targets}
        self._apply_image = "image" in target_set
        self._apply_gx = "gx" in target_set
        self._apply_gy = "gy" in target_set
        self._apply_gi = "gi" in target_set

        requires = set(self.requires)
        sets = set(self.sets)

        if self._apply_image:
            requires.add(ImageDetail.Keys.IMAGE)
            sets.add(ImageDetail.Keys.IMAGE)
        if self._apply_gx:
            requires.add(ImageDetail.Keys.GRAD.X)
            sets.add(ImageDetail.Keys.GRAD.X)
        if self._apply_gy:
            requires.add(ImageDetail.Keys.GRAD.Y)
            sets.add(ImageDetail.Keys.GRAD.Y)
        if self._apply_gi:
            requires.add(ImageDetail.Keys.GRAD.I)
            sets.add(ImageDetail.Keys.GRAD.I)

        self.requires = frozenset(requires)
        self.sets = frozenset(sets)

    def forward(self, image: ImageDetail) -> ImageDetail:
        window = image.get(ImageDetail.Keys.DOMAIN.WINDOW)

        if self._apply_image:
            img = image.get(ImageDetail.Keys.IMAGE)
            image.set(ImageDetail.Keys.IMAGE, img * window)

        if self._apply_gx and ImageDetail.Keys.GRAD.X in image:
            gx = image.get(ImageDetail.Keys.GRAD.X)
            image.set(ImageDetail.Keys.GRAD.X, gx * window)
        
        if self._apply_gy and ImageDetail.Keys.GRAD.Y in image:
            gy = image.get(ImageDetail.Keys.GRAD.Y)
            image.set(ImageDetail.Keys.GRAD.Y, gy * window)

        if self._apply_gi and ImageDetail.Keys.GRAD.I in image:
            gi = image.get(ImageDetail.Keys.GRAD.I)
            image.set(ImageDetail.Keys.GRAD.I, gi * window)

        return image


def build_window_operator(name: str, **kwargs) -> Stage:
    return window_registry.build(name, **kwargs)


@dataclass(frozen=True)
class WindowPipelineSpec:
    stage: Optional[StageSpec] = None


def coerce_window_pipeline_spec(
    value: object,
    *,
    field_name: str = "window",
) -> WindowPipelineSpec:
    if isinstance(value, WindowPipelineSpec):
        return value

    if isinstance(value, str):
        stage = coerce_stage_spec(value, field_name=f"{field_name}.stage", allow_none=False)
        return WindowPipelineSpec(stage=stage)

    if isinstance(value, Mapping):
        raw = dict(value)

        if "stage" in raw:
            stage = coerce_stage_spec(
                raw.pop("stage"),
                field_name=f"{field_name}.stage",
                allow_none=False,
            )
            return WindowPipelineSpec(stage=stage)

        if "type" in raw or "name" in raw:
            stage = coerce_stage_spec(
                raw,
                field_name=f"{field_name}.stage",
                allow_none=False,
            )
            return WindowPipelineSpec(stage=stage)

    raise TypeError(
        f"Cannot coerce {value!r} to WindowPipelineSpec for field '{field_name}'."
    )


def build_window_pipeline(
    spec: Union[WindowPipelineSpec, Mapping, str],
    **kwargs,
) -> Pipeline:
    if isinstance(spec, str) and kwargs:
        spec = {"type": spec, **kwargs}

    if isinstance(spec, WindowPipelineSpec):
        pipeline_spec = spec
    else:
        pipeline_spec = coerce_window_pipeline_spec(spec)

    stages: list[Stage] = []
    if pipeline_spec.stage is not None:
        stage_spec = pipeline_spec.stage
        if not isinstance(stage_spec, StageSpec):
            stage_spec = coerce_stage_spec(stage_spec, field_name="window.stage", allow_none=False)
        stage = window_registry.build(
            stage_spec.type, **stage_spec.params
        )
        stages.append(stage)

    return Pipeline(stages)
