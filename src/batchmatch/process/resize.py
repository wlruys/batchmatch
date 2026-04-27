from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail, NestedKey, WarpParams, TranslationResults, _normalize_keys
from batchmatch.base.pipeline import Pipeline, Stage, StageRegistry, StageSpec, coerce_stage_spec
from batchmatch.helpers.box import scale_quad, scale_xyxy
from batchmatch.helpers.tensor import scale_points

resize_registry = StageRegistry("resize")

if TYPE_CHECKING:
    from batchmatch.process.cells import CellSizeCfg

_RESIZE_OUTPUTS = ("image", "mask", "box", "quad", "points", "warp", "translation")
_RESIZE_OUTPUT_ALIASES = {
    "boxes": "box",
    "box": "box",
    "quads": "quad",
    "quad": "quad",
    "point": "points",
    "points": "points",
    "image": "image",
    "mask": "mask",
    "warp": "warp",
    "translation": "translation",
}

def get_scales_between(details: Sequence[ImageDetail], *rest):
    """
    Compute scale factors to first image in args.

    Args:
        details: Sequence of ImageDetail objects.
        rest: Additional ImageDetail objects.

    Returns:
        Tuple of (scale_x, scale_y) factors.
    """
    all_details = list(details) + list(rest)
    ref_detail = all_details[0]
    ref_h, ref_w = ref_detail.H, ref_detail.W
    scales = []
    for i, detail in enumerate(all_details):
        h, w = detail.H, detail.W
        scale_x = ref_w / float(w)
        scale_y = ref_h / float(h)
        scales.append((float(scale_x), float(scale_y)))
    return scales


def _normalize_resize_outputs(outputs: Optional[Sequence[str]]) -> Optional[set[str]]:
    if outputs is None:
        return None
    if isinstance(outputs, str):
        normalized = [outputs]
    else:
        normalized = [str(o) for o in outputs]
    lowered = {o.lower() for o in normalized}
    if "all" in lowered:
        return set(_RESIZE_OUTPUTS)
    unknown = {o for o in lowered if o not in _RESIZE_OUTPUT_ALIASES}
    if unknown:
        raise ValueError(
            f"Unknown outputs: {sorted(unknown)}. Valid: {sorted(_RESIZE_OUTPUTS)}"
        )
    return { _RESIZE_OUTPUT_ALIASES[o] for o in lowered }


def _interpolate(
    x: Tensor,
    out_hw: tuple[int, int],
    *,
    mode: str = "bilinear",
    antialias: bool = False,
) -> Tensor:
    align_corners = False if mode in ("bilinear", "bicubic") else None

    if mode == "nearest":
        align_corners = None
        antialias = False

    return F.interpolate(
        x, size=out_hw, mode=mode, align_corners=align_corners, antialias=antialias
    )

def scale_resize(
    image: Tensor,
    scale_x: float,
    scale_y: float,
    *,
    is_mask: bool = False,
    mode: str = "bilinear",
) -> Tensor:
    h, w = image.shape[-2:]
    out_h = int(round(h * scale_y))
    out_w = int(round(w * scale_x))

    if is_mask:
        new_image = _interpolate(image, (out_h, out_w), mode="nearest")
    else:
        new_image = _interpolate(image, (out_h, out_w), mode=mode, antialias=True)
    return new_image

def target_resize(
    image: Tensor,
    target: int,
    *,
    is_mask: bool = False,
    mode: str = "bilinear",
    target_dim: int = 1,
) -> Tuple[Tensor, Tuple[int, int]]:
    h, w = image.shape[-2:]
    if target_dim == 1:
        scale = target / float(w)
        out_w = int(target)
        out_h = int(round(h * scale))
    else:
        scale = target / float(h)
        out_h = int(target)
        out_w = int(round(w * scale))

    if is_mask:
        new_image = _interpolate(image, (out_h, out_w), mode="nearest")
    else:
        new_image = _interpolate(image, (out_h, out_w), mode=mode, antialias=True)
    return new_image, (h, w)


def _down2(x: Tensor, *, is_mask: bool = False) -> Tensor:
    if is_mask:
        y = F.max_pool2d(x, kernel_size=2, stride=2)
    else:
        y = F.avg_pool2d(x, kernel_size=2, stride=2)
    return y

def _up2(x: Tensor, *, is_mask: bool = False) -> Tensor:
    _, _, H, W = x.shape
    target_hw = (H * 2, W * 2)
    if is_mask:
        y = _interpolate(x, target_hw, mode="nearest")
    else:
        y = _interpolate(x, target_hw, mode="bilinear", antialias=False)
    return y


def multilevel_target_resize(
    image: Tensor,
    target: int,
    *,
    is_mask: bool = False,
    target_dim: int = 1,
) -> Tuple[Tensor, Tuple[int, int]]:
    h, w = image.shape[-2:]
    target_size = int(target)

    x = image
    while (x.shape[-2] * 2) <= target_size and (x.shape[-1] * 2) <= target_size:
        x = _up2(x, is_mask=is_mask)
    while (x.shape[-2] // 2) >= target_size and (x.shape[-1] // 2) >= target_size:
        x = _down2(x, is_mask=is_mask)
    if target_dim == 1:
        final = target_resize(x, target_size, is_mask=is_mask, mode="bilinear", target_dim=1)[0]
    else:
        final = target_resize(x, target_size, is_mask=is_mask, mode="bilinear", target_dim=0)[0]
    return final, (h, w)

def multilevel_scale_resize(
    image: Tensor,
    scale_x: float,
    scale_y: float,
    *,
    is_mask: bool = False,
) -> Tuple[Tensor, Tuple[int, int]]:
    h, w = image.shape[-2:]
    target_h = int(round(h * scale_y))
    target_w = int(round(w * scale_x))
    x = image
    while (x.shape[-2] * 2) <= target_h and (x.shape[-1] * 2) <= target_w:
        x = _up2(x, is_mask=is_mask)
    while (x.shape[-2] // 2) >= target_h and (x.shape[-1] // 2) >= target_w:
        x = _down2(x, is_mask=is_mask)
    final = target_resize(x, target_h, is_mask=is_mask, mode="bilinear", target_dim=0)[0]
    return final, (h, w)


def _init_resize_output_keys(
    stage: Stage,
    *,
    outputs: Optional[Sequence[str]],
    image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    mask_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    box_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    quad_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    point_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    warp_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    translation_keys: Optional[Union[NestedKey, Sequence[NestedKey]]],
    extra_requires: Optional[set[NestedKey]] = None,
) -> None:
    """Resolve *outputs* flags and normalise key tuples.

    Sets ``stage.image_keys``, ``stage.mask_keys``, etc. together with
    ``stage._resize_mask``, ``stage._resize_box``, ... boolean flags.
    """
    outputs_set = _normalize_resize_outputs(outputs)
    if outputs_set is None:
        outputs_set = set(_RESIZE_OUTPUTS)

    resize_image = "image" in outputs_set
    resize_mask = "mask" in outputs_set
    resize_box = "box" in outputs_set
    resize_quad = "quad" in outputs_set
    resize_points = "points" in outputs_set
    resize_warp = "warp" in outputs_set
    resize_translation = "translation" in outputs_set

    image_keys = image_keys if image_keys is not None else ImageDetail.Keys.IMAGE
    image_keys = image_keys if resize_image else None

    mask_keys = mask_keys if mask_keys is not None else ImageDetail.Keys.DOMAIN.MASK
    mask_keys = mask_keys if resize_mask else None

    box_keys = box_keys if box_keys is not None else ImageDetail.Keys.DOMAIN.BOX
    box_keys = box_keys if resize_box else None

    quad_keys = quad_keys if quad_keys is not None else ImageDetail.Keys.DOMAIN.QUAD
    quad_keys = quad_keys if resize_quad else None

    point_keys = point_keys if point_keys is not None else ImageDetail.Keys.AUX.POINTS
    point_keys = point_keys if resize_points else None

    warp_keys = warp_keys if warp_keys is not None else ImageDetail.Keys.WARP.ROOT
    warp_keys = warp_keys if resize_warp else None

    translation_keys = translation_keys if translation_keys is not None else ImageDetail.Keys.TRANSLATION.ROOT
    translation_keys = translation_keys if resize_translation else None

    stage._resize_mask = resize_mask
    stage._resize_box = resize_box
    stage._resize_quad = resize_quad
    stage._resize_points = resize_points
    stage._resize_warp = resize_warp
    stage._resize_translation = resize_translation

    stage.image_keys = _normalize_keys(image_keys)
    stage.mask_keys = _normalize_keys(mask_keys)
    stage.box_keys = _normalize_keys(box_keys)
    stage.quad_keys = _normalize_keys(quad_keys)
    stage.point_keys = _normalize_keys(point_keys)
    stage.warp_keys = _normalize_keys(warp_keys)
    stage.translation_keys = _normalize_keys(translation_keys)

    reqs: set[NestedKey] = set()
    if stage.image_keys:
        reqs.update(stage.image_keys)
    if extra_requires:
        reqs.update(extra_requires)

    stage.requires = frozenset(reqs)
    stage.sets = frozenset(reqs)



def _scale_image_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float, *, is_mask: bool,
) -> None:
    img = image.get(key, None)
    if img is None:
        return
    image.set(key, scale_resize(img, sx, sy, is_mask=is_mask))


def _scale_mask_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float,
) -> None:
    mask = image.get(key, None)
    if mask is None:
        return
    image.set(key, scale_resize(mask, sx, sy, is_mask=True))


def _scale_box_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float,
) -> None:
    boxes = image.get(key, None)
    if boxes is None:
        return
    image.set(key, scale_xyxy(boxes, scale_x=sx, scale_y=sy))


def _scale_quad_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float,
) -> None:
    quads = image.get(key, None)
    if quads is None:
        return
    image.set(key, scale_quad(quads, scale_x=sx, scale_y=sy))


def _scale_point_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float,
) -> None:
    pts = image.get(key, None)
    if pts is None:
        return
    image.set(key, scale_points(pts, scale_x=sx, scale_y=sy))


def _scale_warp_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float,
) -> None:
    warp = image.get(key, None)
    if warp is None:
        return

    if WarpParams.Keys.TX in warp.keys():
        warp.set(WarpParams.Keys.TX, warp.get(WarpParams.Keys.TX) * sx)
    if WarpParams.Keys.TY in warp.keys():
        warp.set(WarpParams.Keys.TY, warp.get(WarpParams.Keys.TY) * sy)

    for k in WarpParams.Keys.CENTER:
        if k in warp.keys():
            warp.del_(k)
    for k in WarpParams.Keys.COMPUTED:
        if k in warp.keys():
            warp.del_(k)


def _scale_translation_key(
    image: ImageDetail, key: NestedKey, sx: float, sy: float,
) -> None:
    translation = image.get(key, None)
    if translation is None:
        return
    if TranslationResults.Keys.X in translation.keys():
        translation.set(TranslationResults.Keys.X, translation.get(TranslationResults.Keys.X) * sx)
    if TranslationResults.Keys.Y in translation.keys():
        translation.set(TranslationResults.Keys.Y, translation.get(TranslationResults.Keys.Y) * sy)


def _apply_scale_to_detail(
    stage: "ScaleResize | UnitResize",
    detail: ImageDetail,
    sx: float,
    sy: float,
) -> ImageDetail:
    """Scale every configured field of *detail* by *(sx, sy)* in-place."""
    if stage.image_keys is not None:
        for key in stage.image_keys:
            _scale_image_key(detail, key, sx, sy, is_mask=stage._is_mask)
    if stage._resize_mask and stage.mask_keys is not None:
        for key in stage.mask_keys:
            _scale_mask_key(detail, key, sx, sy)
    if stage._resize_box and stage.box_keys is not None:
        for key in stage.box_keys:
            _scale_box_key(detail, key, sx, sy)
    if stage._resize_quad and stage.quad_keys is not None:
        for key in stage.quad_keys:
            _scale_quad_key(detail, key, sx, sy)
    if stage._resize_points and stage.point_keys is not None:
        for key in stage.point_keys:
            _scale_point_key(detail, key, sx, sy)
    if stage._resize_warp and stage.warp_keys is not None:
        for key in stage.warp_keys:
            _scale_warp_key(detail, key, sx, sy)
    if stage._resize_translation and stage.translation_keys is not None:
        for key in stage.translation_keys:
            _scale_translation_key(detail, key, sx, sy)
    return detail


@resize_registry.register("scale_resize")
class ScaleResize(Stage):

    def __init__(
        self,
        *,
        scale: Optional[float | Tuple[float, float]] = None,
        outputs: Optional[Sequence[str]] = None,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        mask_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        box_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        quad_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        point_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        warp_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        translation_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        is_mask: bool = False,
    ) -> None:
        _init_resize_output_keys(
            self,
            outputs=outputs,
            image_keys=image_keys,
            mask_keys=mask_keys,
            box_keys=box_keys,
            quad_keys=quad_keys,
            point_keys=point_keys,
            warp_keys=warp_keys,
            translation_keys=translation_keys,
        )

        super().__init__()

        self._scale = scale
        self._is_mask = is_mask

        if isinstance(self._scale, (int, float)):
            self._scale_x = float(self._scale)
            self._scale_y = float(self._scale)
        elif isinstance(self._scale, tuple):
            self._scale_x = float(self._scale[0])
            self._scale_y = float(self._scale[1])
        else:
            raise ValueError("Scale must be a float or a tuple of two floats.", self._scale)

    # -- public convenience methods (thin wrappers) -------------------------

    def scale_image(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_image_key(image, key, self._scale_x, self._scale_y, is_mask=self._is_mask)

    def scale_mask(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_mask_key(image, key, self._scale_x, self._scale_y)

    def scale_boxes(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_box_key(image, key, self._scale_x, self._scale_y)

    def scale_quads(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_quad_key(image, key, self._scale_x, self._scale_y)

    def scale_points(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_point_key(image, key, self._scale_x, self._scale_y)

    def scale_warp(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_warp_key(image, key, self._scale_x, self._scale_y)

    def scale_translation(self, image: ImageDetail, key: NestedKey) -> None:
        _scale_translation_key(image, key, self._scale_x, self._scale_y)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        image: ImageDetail | list[ImageDetail],
        *args: ImageDetail,
    ) -> ImageDetail | list[ImageDetail]:
        if isinstance(image, list):
            images = image
        elif len(args) > 0:
            images = [image, *args]
        else:
            images = [image]

        for img in images:
            _apply_scale_to_detail(self, img, self._scale_x, self._scale_y)

        return images if isinstance(image, list) else images[0]

    def forward_at_scale(
        self,
        image: ImageDetail | list[ImageDetail],
        scale_x: float,
        scale_y: float,
        *args: ImageDetail,
    ) -> ImageDetail | list[ImageDetail]:
        """Apply the resize at explicit *scale_x*, *scale_y* without
        permanently mutating ``self._scale*``.

        Always dispatches through :meth:`ScaleResize.forward` so that
        subclass overrides (e.g. ``TargetResize.forward``) are bypassed.
        """
        saved = (self._scale_x, self._scale_y, self._scale)
        try:
            self._scale_x = scale_x
            self._scale_y = scale_y
            self._scale = (scale_x, scale_y)
            return ScaleResize.forward(self, image, *args)
        finally:
            self._scale_x, self._scale_y, self._scale = saved

@resize_registry.register("target_resize")
class TargetResize(ScaleResize):

    def __init__(
        self,
        *,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        outputs: Optional[Sequence[str]] = None,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        mask_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        box_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        quad_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        point_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        warp_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        translation_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        is_mask: bool = False,
    ):
        self._target_width = target_width
        self._target_height = target_height

        self._scale_x = 1.0
        self._scale_y = 1.0

        super().__init__(
            scale=(self._scale_x, self._scale_y),  # placeholder; updated in forward
            outputs=outputs,
            image_keys=image_keys,
            mask_keys=mask_keys,
            box_keys=box_keys,
            quad_keys=quad_keys,
            point_keys=point_keys,
            warp_keys=warp_keys,
            translation_keys=translation_keys,
            is_mask=is_mask,
        )

    def compute_scale_factor(
        self,
        images: list[ImageDetail],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Tuple[float, float]:
        if not target_width and not target_height:
            raise ValueError("At least one of target_width or target_height must be specified.")

        max_width = max(img.W for img in images)
        max_height = max(img.H for img in images)

        scale_x: float | None = None
        scale_y: float | None = None

        if target_width is not None:
            scale_x = target_width / max_width
        if target_height is not None:
            scale_y = target_height / max_height
        if scale_x is None:
            scale_x = scale_y
        if scale_y is None:
            scale_y = scale_x

        assert scale_x is not None and scale_y is not None
        return (float(scale_x), float(scale_y))

    def forward(
        self,
        image: ImageDetail | list[ImageDetail],
        *args: ImageDetail,
    ) -> ImageDetail | list[ImageDetail]:

        if isinstance(image, list):
            images = image
        elif len(args) > 0:
            images = [image, *args]
        else:
            images = [image]

        self._scale_x, self._scale_y = self.compute_scale_factor(
            images,
            target_width=self._target_width,
            target_height=self._target_height,
        )

        self._scale = (self._scale_x, self._scale_y)

        return super().forward(image, *args)

class UnitResize(Stage):

    def __init__(
        self,
        *,
        lengths: Sequence[float],
        widths: Sequence[float] | None = None,
        is_mask: bool = False,
        reference_index: int = 0,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        mask_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        box_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        quad_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        point_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        warp_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        translation_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        radius_key: Optional[NestedKey] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> None:
        effective_radius_key = radius_key if radius_key is not None else ImageDetail.Keys.IMAGE

        _init_resize_output_keys(
            self,
            outputs=outputs,
            image_keys=image_keys,
            mask_keys=mask_keys,
            box_keys=box_keys,
            quad_keys=quad_keys,
            point_keys=point_keys,
            warp_keys=warp_keys,
            translation_keys=translation_keys,
            extra_requires={effective_radius_key},
        )

        super().__init__()

        self._lengths = lengths
        self._widths = widths
        self._is_mask = is_mask
        self._reference_index = reference_index

    def _compute_scale_factors(
        self,
        images: list[ImageDetail],
    ) -> list[Tuple[float, float]]:
        ref_length = self._lengths[self._reference_index]
        ref_width = None
        if self._widths is not None:
            ref_width = self._widths[self._reference_index]

        scales = []
        for i, img in enumerate(images):
            length = self._lengths[i]
            scale_y = ref_length / length

            if self._widths is not None:
                width = self._widths[i]
                scale_x = ref_width / width
            else:
                scale_x = scale_y

            scales.append((float(scale_x), float(scale_y)))
        return scales

    def _forward_single_at_scale(
        self,
        image: ImageDetail,
        scale_x: float,
        scale_y: float,
    ) -> ImageDetail:
        """Apply the resize to a single *image* at explicit scales."""
        return _apply_scale_to_detail(self, image, scale_x, scale_y)

    def forward(
        self,
        image: ImageDetail | list[ImageDetail],
        *args: ImageDetail,
    ) -> ImageDetail | list[ImageDetail]:
        if isinstance(image, list):
            images = image
        elif len(args) > 0:
            images = [image, *args]
        else:
            images = [image]

        scales = self._compute_scale_factors(images)

        for img, (scale_x, scale_y) in zip(images, scales):
            _apply_scale_to_detail(self, img, scale_x, scale_y)

        return images if isinstance(image, list) else images[0]

@resize_registry.register("cell_unit_resize", "cell_radius_resize")
class CellUnitResize(UnitResize):
    """
    Estimate per-image cell radii and normalize sizes relative to a reference.
    """

    def __init__(
        self,
        *,
        cell_cfg: "CellSizeCfg | None" = None,
        radius_key: Optional[NestedKey] = None,
        is_mask: bool = False,
        reference_index: int = 0,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        mask_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        box_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        quad_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        point_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        warp_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        translation_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> None:
        # Use placeholder lengths; actual lengths computed in forward()
        super().__init__(
            lengths=[1.0],
            widths=None,
            is_mask=is_mask,
            reference_index=reference_index,
            image_keys=image_keys,
            mask_keys=mask_keys,
            box_keys=box_keys,
            quad_keys=quad_keys,
            point_keys=point_keys,
            warp_keys=warp_keys,
            translation_keys=translation_keys,
            radius_key=radius_key,
            outputs=outputs,
        )

        self._cell_cfg = cell_cfg
        self._radius_key = radius_key if radius_key is not None else ImageDetail.Keys.IMAGE

    def _estimate_radius(self, image: ImageDetail) -> float:
        img = image.get(self._radius_key, None)
        if img is None:
            raise ValueError(f"Radius key {self._radius_key} not found in ImageDetail.")

        from batchmatch.process.cells import CellSizeCfg, estimate_cell_radius_px

        if self._cell_cfg is None:
            cfg = CellSizeCfg(return_debug=False)
        else:
            cfg = self._cell_cfg

        result = estimate_cell_radius_px(img, cfg)
        radius = result["radius_px"]
        return float(radius)

    def _compute_scale_factors(
        self,
        images: list[ImageDetail],
    ) -> list[Tuple[float, float]]:
        if not images:
            return []

        if not (0 <= self._reference_index < len(images)):
            raise ValueError(
                f"reference_index {self._reference_index} out of range for {len(images)} images."
            )

        # Dynamically compute lengths from estimated cell radii
        self._lengths = [self._estimate_radius(img) for img in images]

        return super()._compute_scale_factors(images)


@resize_registry.register("physical_resize", "pixel_size_resize")
class PhysicalResize(UnitResize):
    """Normalize images so that each pixel covers the same physical area.

    Given per-image ``pixel_size_xy`` tuples (from OME-TIFF / TIFF
    metadata), the stage scales every image so that all images share the
    reference image's physical pixel size.

    Pixel sizes can be supplied explicitly or extracted at call time from
    :class:`~batchmatch.io.space.SpatialImage` metadata via the
    ``SpatialPhysicalResize`` wrapper.
    """

    def __init__(
        self,
        *,
        pixel_sizes_xy: Sequence[tuple[float, float]] | None = None,
        reference_index: int = 0,
        is_mask: bool = False,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        mask_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        box_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        quad_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        point_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        warp_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        translation_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> None:
        # Use placeholder lengths; actual lengths computed from pixel sizes.
        super().__init__(
            lengths=[1.0],
            widths=[1.0],
            is_mask=is_mask,
            reference_index=reference_index,
            image_keys=image_keys,
            mask_keys=mask_keys,
            box_keys=box_keys,
            quad_keys=quad_keys,
            point_keys=point_keys,
            warp_keys=warp_keys,
            translation_keys=translation_keys,
            outputs=outputs,
        )
        self._pixel_sizes_xy = pixel_sizes_xy

    def set_pixel_sizes(self, pixel_sizes_xy: Sequence[tuple[float, float]]) -> None:
        """Set pixel sizes for the next forward pass."""
        self._pixel_sizes_xy = list(pixel_sizes_xy)

    def _compute_scale_factors(
        self,
        images: list[ImageDetail],
    ) -> list[Tuple[float, float]]:
        if self._pixel_sizes_xy is None:
            raise ValueError(
                "PhysicalResize requires pixel_sizes_xy. Supply them at "
                "construction, via set_pixel_sizes(), or use SpatialPhysicalResize."
            )
        n = len(images)
        if len(self._pixel_sizes_xy) != n:
            raise ValueError(
                f"Expected {n} pixel sizes, got {len(self._pixel_sizes_xy)}."
            )
        if not (0 <= self._reference_index < n):
            raise ValueError(
                f"reference_index {self._reference_index} out of range for {n} images."
            )

        # Physical pixel sizes: larger means coarser resolution.
        # To match the reference, scale each image by (its_size / ref_size)
        # so that coarser images get downscaled and finer ones upscaled.
        ref_px, ref_py = self._pixel_sizes_xy[self._reference_index]
        scales: list[Tuple[float, float]] = []
        for px, py in self._pixel_sizes_xy:
            # scale > 1  →  shrink (this image has finer pixels)
            # scale < 1  →  enlarge (this image has coarser pixels)
            sx = px / ref_px
            sy = py / ref_py
            scales.append((float(sx), float(sy)))
        return scales


def build_resize_operator(
    resize_type: str,
    **kwargs,
) -> Stage:
    return resize_registry.build(resize_type, **kwargs)

@dataclass(frozen=True)
class ResizePipelineSpec:
    stage: StageSpec

def coerce_resize_pipeline_spec(
    value: object,
    *,
    field_name: str = "resize_pipeline",
) -> ResizePipelineSpec:
    if isinstance(value, ResizePipelineSpec):
        return value

    if isinstance(value, str):
        stage_spec = coerce_stage_spec(value, field_name=f"{field_name}.stage")
        return ResizePipelineSpec(stage=stage_spec)

    if isinstance(value, Mapping):
        raw = dict(value)

        if "stage" in raw:
            stage = coerce_stage_spec(
                raw.pop("stage"),
                field_name=f"{field_name}.stage",
                allow_none=False,
            )
            return ResizePipelineSpec(stage=stage)

        if "type" in raw or "name" in raw:
             stage = coerce_stage_spec(
                raw,
                field_name=f"{field_name}.stage",
                allow_none=False,
            )
             return ResizePipelineSpec(stage=stage)

    raise TypeError(
        f"Cannot coerce {value!r} to ResizePipelineSpec for field '{field_name}'."
    )

def build_resize_pipeline(
    spec: Union[ResizePipelineSpec, Mapping, str],
    **kwargs,
) -> Pipeline:

    if isinstance(spec, str) and kwargs:
        spec = {"type": spec, **kwargs}

    if isinstance(spec, ResizePipelineSpec):
        pipeline_spec = spec
    else:
        pipeline_spec = coerce_resize_pipeline_spec(spec)

    stages: list[Stage] = []
    if pipeline_spec.stage is not None:
        stage_spec = coerce_stage_spec(
            pipeline_spec.stage,
            field_name="resize_pipeline.stage",
            allow_none=False,
        )
        stage = resize_registry.build(stage_spec.type, **stage_spec.params)
        stages.append(stage)

    return Pipeline(stages)
