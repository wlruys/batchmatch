from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch
from torch import Tensor

from batchmatch.base.pipeline import Pipeline, Stage
from batchmatch.base.tensordicts import ImageDetail, NestedKey, TranslationResults, WarpParams
from batchmatch.helpers.box import adjust_xyxy_to_crop, adjust_quad_to_crop, mask_to_box
from batchmatch.helpers.tensor import adjust_points_to_crop
from batchmatch.process.crop import compute_intersection_mask, compute_union_mask, crop_spatial_to_box
from batchmatch.warp.base import WarpPipelineConfig

__all__ = [
    "ApplySearchResultStage",
    "ApplyToIndexStage",
    "ProductCropStage",
    "ProductPipelineConfig",
    "build_product_pipeline",
]


def _get_content_dims(image: ImageDetail) -> tuple[float, float]:
    box = image.get(ImageDetail.Keys.DOMAIN.BOX, default=None)
    if box is not None:
        x1, y1, x2, y2 = box[0, 0, 0], box[0, 0, 1], box[0, 0, 2], box[0, 0, 3]
        width = float((x2 - x1).item())
        height = float((y2 - y1).item())
        return width, height
    return float(image.W), float(image.H)


class ApplySearchResultStage(Stage):
    _auto_validate: bool = False
    _auto_invalidate: bool = False

    def __init__(
        self,
        result: ImageDetail,
        *,
        idx: int = 0,
        move_scale: tuple[float, float] | None = None,
        ref_scale: tuple[float, float] | None = None,
        scale_translation: bool = True,
        scale_warp_translation: bool = True,
        scale_warp_scale: bool = True,
        reference_index: int = 1,
    ) -> None:
        super().__init__()

        if result.warp is None:
            raise ValueError("Result missing warp parameters.")
        if result.translation_results is None:
            raise ValueError("Result missing translation_results.")

        batch_size = result.batch_size[0]
        if idx < 0:
            idx = batch_size + idx
        if idx < 0 or idx >= batch_size:
            raise IndexError(f"idx {idx} out of range for batch size {batch_size}.")

        self._result = result
        self._idx = idx
        self._scale_translation = scale_translation
        self._scale_warp_translation = scale_warp_translation
        self._scale_warp_scale = scale_warp_scale
        self._reference_index = reference_index or 1
        self._move_scale = move_scale
        self._ref_scale = ref_scale

    def _compute_scale_factors(self, reference: ImageDetail) -> tuple[float, float]:
        tr = self._result.translation_results
        if tr is None:
            return 1.0, 1.0

        search_h = tr.get(TranslationResults.Keys.SEARCH_H, default=None)
        search_w = tr.get(TranslationResults.Keys.SEARCH_W, default=None)
        if search_h is None or search_w is None:
            return 1.0, 1.0

        search_h_val = float(search_h[self._idx].item())
        search_w_val = float(search_w[self._idx].item())
        if search_h_val <= 0 or search_w_val <= 0:
            return 1.0, 1.0

        target_w, target_h = _get_content_dims(reference)
        return target_w / search_w_val, target_h / search_h_val

    def _select_and_expand(
        self,
        tensor: Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        selected = tensor[self._idx : self._idx + 1].to(device=device, dtype=dtype)
        if batch > 1:
            selected = selected.expand(batch).contiguous()
        return selected

    def forward(
        self,
        first: ImageDetail | list[ImageDetail],
        *rest: ImageDetail,
    ) -> list[ImageDetail]:
        if isinstance(first, list):
            images = first
        else:
            images = [first, *rest]

        if len(images) < 2:
            raise ValueError("Expected at least two ImageDetails: [moving, reference].")

        moving = images[0]
        reference = images[self._reference_index]
        device = moving.image.device
        dtype = moving.image.dtype
        batch = moving.B

        scale_x, scale_y = self._compute_scale_factors(reference)

        wp = self._result.warp
        warp_tx = self._select_and_expand(wp.tx, batch, device, dtype)
        warp_ty = self._select_and_expand(wp.ty, batch, device, dtype)

        if self._scale_warp_translation:
            warp_tx = warp_tx * scale_x
            warp_ty = warp_ty * scale_y

        if self._scale_warp_scale:
            ratio_x = (
                self._ref_scale[0] / self._move_scale[0]
                if self._move_scale and self._ref_scale
                else 1.0
            )
            ratio_y = (
                self._ref_scale[1] / self._move_scale[1]
                if self._move_scale and self._ref_scale
                else 1.0
            )
            warp_scale_x = self._select_and_expand(wp.scale_x, batch, device, dtype) * ratio_x
            warp_scale_y = self._select_and_expand(wp.scale_y, batch, device, dtype) * ratio_y
        else:
            warp_scale_x = self._select_and_expand(wp.scale_x, batch, device, dtype)
            warp_scale_y = self._select_and_expand(wp.scale_y, batch, device, dtype)


        cx = wp.get(WarpParams.Keys.CENTER_X, default=None)
        cy = wp.get(WarpParams.Keys.CENTER_Y, default=None)
        if cx is not None:
            cx = self._select_and_expand(cx, batch, device, dtype)
            if self._scale_warp_translation:
                cx = cx * scale_x
        if cy is not None:
            cy = self._select_and_expand(cy, batch, device, dtype)
            if self._scale_warp_translation:
                cy = cy * scale_y

        warp_params = WarpParams.from_components(
            angle=self._select_and_expand(wp.angle, batch, device, dtype),
            scale_x=warp_scale_x,
            scale_y=warp_scale_y,
            shear_x=self._select_and_expand(wp.shear_x, batch, device, dtype),
            shear_y=self._select_and_expand(wp.shear_y, batch, device, dtype),
            tx=warp_tx,
            ty=warp_ty,
            center_x=cx,
            center_y=cy,
        )
        moving.set(ImageDetail.Keys.WARP.ROOT, warp_params)

        tr = self._result.translation_results
        tx = self._select_and_expand(tr.tx, batch, device, dtype)
        ty = self._select_and_expand(tr.ty, batch, device, dtype)
        score = self._select_and_expand(tr.score, batch, device, dtype)

        if self._scale_translation:
            tx = tx * scale_x
            ty = ty * scale_y

        translation = TranslationResults.from_components(
            x=tx,
            y=ty,
            score=score,
            device=device,
            dtype=dtype,
        )
        moving.set(ImageDetail.Keys.TRANSLATION.ROOT, translation)

        return images


class ApplyToIndexStage(Stage):

    _auto_validate: bool = False
    _auto_invalidate: bool = False

    def __init__(self, stage: Stage, *, indices: int | Sequence[int]) -> None:
        super().__init__()
        self.stage = stage
        self._indices = [indices] if isinstance(indices, int) else list(indices)

    def forward(
        self,
        first: ImageDetail | list[ImageDetail],
        *rest: ImageDetail,
    ) -> list[ImageDetail]:
        if isinstance(first, list):
            images = first
        else:
            images = [first, *rest]

        total = len(images)
        for idx in self._indices:
            resolved = idx + total if idx < 0 else idx
            if resolved < 0 or resolved >= total:
                raise IndexError(f"index {idx} out of range [0, {total})")
            images[resolved] = self.stage(images[resolved])

        return images


class ProductCropStage(Stage):
    DEFAULT_IMAGE_KEY: NestedKey = ImageDetail.Keys.IMAGE
    DEFAULT_MASK_KEY: NestedKey = ImageDetail.Keys.DOMAIN.MASK
    DEFAULT_WINDOW_KEY: NestedKey = ImageDetail.Keys.DOMAIN.WINDOW
    DEFAULT_GX_KEY: NestedKey = ImageDetail.Keys.GRAD.X
    DEFAULT_GY_KEY: NestedKey = ImageDetail.Keys.GRAD.Y
    DEFAULT_BOX_KEY: NestedKey = ImageDetail.Keys.DOMAIN.BOX
    DEFAULT_QUAD_KEY: NestedKey = ImageDetail.Keys.DOMAIN.QUAD
    DEFAULT_POINTS_KEY: NestedKey = ImageDetail.Keys.AUX.POINTS

    _auto_validate: bool = False
    _auto_invalidate: bool = False

    def __init__(
        self,
        *,
        mode: Literal["union", "intersection"] = "intersection",
        combine_fn: Callable[[list[Tensor]], Tensor] | None = None,
        crop_image: bool = True,
        crop_mask: bool = True,
        crop_window: bool = False,
        crop_gx: bool = False,
        crop_gy: bool = False,
        adjust_box: bool = True,
        adjust_quad: bool = True,
        adjust_points: bool = False,
        clip_geometry: bool = True,
        invalidate_warp: bool = True,
        image_key: NestedKey | None = None,
        mask_key: NestedKey | None = None,
        window_key: NestedKey | None = None,
        gx_key: NestedKey | None = None,
        gy_key: NestedKey | None = None,
        box_key: NestedKey | None = None,
        quad_key: NestedKey | None = None,
        points_key: NestedKey | None = None,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._combine_fn = combine_fn
        self._clip_geometry = clip_geometry
        self._invalidate_warp = invalidate_warp

        self._image_key = (
            image_key if image_key is not None
            else (self.DEFAULT_IMAGE_KEY if crop_image else None)
        )
        self._mask_key = (
            mask_key if mask_key is not None
            else (self.DEFAULT_MASK_KEY if crop_mask else None)
        )
        self._window_key = (
            window_key if window_key is not None
            else (self.DEFAULT_WINDOW_KEY if crop_window else None)
        )
        self._gx_key = (
            gx_key if gx_key is not None
            else (self.DEFAULT_GX_KEY if crop_gx else None)
        )
        self._gy_key = (
            gy_key if gy_key is not None
            else (self.DEFAULT_GY_KEY if crop_gy else None)
        )
        self._box_key = (
            box_key if box_key is not None
            else (self.DEFAULT_BOX_KEY if adjust_box else None)
        )
        self._quad_key = (
            quad_key if quad_key is not None
            else (self.DEFAULT_QUAD_KEY if adjust_quad else None)
        )
        self._points_key = (
            points_key if points_key is not None
            else (self.DEFAULT_POINTS_KEY if adjust_points else None)
        )

    def _get_mask_key(self) -> NestedKey:
        return self._mask_key if self._mask_key is not None else self.DEFAULT_MASK_KEY

    def _combine_masks(self, masks: list[Tensor]) -> Tensor:
        if self._combine_fn is not None:
            return self._combine_fn(masks)
        if self._mode == "union":
            return compute_union_mask(masks)
        return compute_intersection_mask(masks)

    def _invalidate_warp_data(self, detail: ImageDetail) -> None:
        warp = detail.get(ImageDetail.Keys.WARP.ROOT, default=None)
        if warp is None:
            return

        for center_key in WarpParams.Keys.CENTER:
            if center_key in warp.keys():
                warp.del_(center_key)

        for computed_key in WarpParams.Keys.COMPUTED:
            if computed_key in warp.keys():
                warp.del_(computed_key)

    def _compute_global_crop_box(
        self,
        images: list[ImageDetail],
    ) -> Tensor:
        if len(images) == 0:
            raise ValueError("Expected at least one ImageDetail.")

        B = images[0].B
        H, W = images[0].H, images[0].W
        device = images[0].image.device
        mask_key = self._get_mask_key()

        per_batch_boxes: list[Tensor] = []

        for b in range(B):
            masks_at_b: list[Tensor] = []
            for detail in images:
                mask = detail.get(mask_key, default=None)
                if mask is not None:
                    masks_at_b.append(mask[b : b + 1])
                else:
                    masks_at_b.append(torch.ones(1, 1, H, W, device=device))

            combined = self._combine_masks(masks_at_b)

            if combined.sum() == 0:
                raise ValueError(
                    f"Empty crop region at batch position {b}. "
                    f"Combined mask has no valid pixels."
                )

            box = mask_to_box(combined).squeeze(0).squeeze(0)
            box = box.clone()
            # Adjust box to be inclusive
            box[2] = box[2] + 1
            box[3] = box[3] + 1
            per_batch_boxes.append(box)

        stacked = torch.stack(per_batch_boxes, dim=0)
        global_box = torch.zeros(4, device=device, dtype=stacked.dtype)
        global_box[0] = stacked[:, 0].min()
        global_box[1] = stacked[:, 1].min()
        global_box[2] = stacked[:, 2].max()
        global_box[3] = stacked[:, 3].max()

        return global_box

    def _crop_detail(
        self,
        detail: ImageDetail,
        crop_box: Tensor,
    ) -> ImageDetail:
        new_data: dict[NestedKey, Tensor] = {}

        for key in [self._image_key, self._mask_key, self._window_key, self._gx_key, self._gy_key]:
            if key is None:
                continue
            tensor = detail.get(key, default=None)
            if tensor is None:
                continue
            cropped = crop_spatial_to_box(tensor, crop_box)
            new_data[key] = cropped

        if self._box_key is not None:
            box = detail.get(self._box_key, default=None)
            if box is not None:
                adjusted = adjust_xyxy_to_crop(box, crop_box, clip=self._clip_geometry)
                new_data[self._box_key] = adjusted

        if self._quad_key is not None:
            quad = detail.get(self._quad_key, default=None)
            if quad is not None:
                adjusted = adjust_quad_to_crop(quad, crop_box, clip=self._clip_geometry)
                new_data[self._quad_key] = adjusted

        if self._points_key is not None:
            points = detail.get(self._points_key, default=None)
            if points is not None:
                adjusted = adjust_points_to_crop(points, crop_box, clip=self._clip_geometry)
                new_data[self._points_key] = adjusted

        out = detail.clone()
        for key, value in new_data.items():
            out.set(key, value)

        if self._invalidate_warp:
            self._invalidate_warp_data(out)

        return out

    def forward(
        self,
        first: ImageDetail | list[ImageDetail],
        *rest: ImageDetail,
    ) -> list[ImageDetail]:
        if isinstance(first, list):
            images = list(first)  # Shallow copy to avoid modifying input
        else:
            images = [first, *rest]

        if len(images) == 0:
            return []

        H, W = images[0].H, images[0].W
        B = images[0].B
        for i, detail in enumerate(images):
            if detail.H != H or detail.W != W:
                raise ValueError(
                    f"All inputs must have same spatial dimensions. "
                    f"Input 0 has ({H}, {W}), input {i} has ({detail.H}, {detail.W})"
                )
            if detail.B != B:
                raise ValueError(
                    f"All inputs must have same batch size. "
                    f"Input 0 has B={B}, input {i} has B={detail.B}"
                )

        crop_box = self._compute_global_crop_box(images)
        return [self._crop_detail(detail, crop_box) for detail in images]


@dataclass
class ProductPipelineConfig:
    idx: int = 0
    scale_translation: bool = True
    scale_warp_translation: bool = True
    pad_spec: object | None = field(default_factory=lambda: {
        "type": "center_pad",
        "scale": 2,
        "outputs": ["image", "box", "quad", "mask", "window"],
        "shrink_by": 0,
    })
    warp_spec: object | None = field(
        default_factory=lambda: WarpPipelineConfig(outputs=["image", "mask"])
    )
    shift: bool = True
    shift_params: dict[str, Any] = field(
        default_factory=lambda: {
            "source": "translation",
            "negate": True,
            "shift_window": False,
            "shift_points": False,
        }
    )
    crop_mode: Literal["union", "intersection"] | None = None
    crop_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, result: ImageDetail) -> Pipeline:
        from batchmatch.process.pad import build_pad_pipeline
        from batchmatch.process.shift import SubpixelShiftStage
        from batchmatch.warp.base import build_warp_pipeline

        stages: list[Stage] = [
            ApplySearchResultStage(
                result,
                idx=self.idx,
                scale_translation=self.scale_translation,
                scale_warp_translation=self.scale_warp_translation,
            )
        ]

        if self.pad_spec is not None:
            stages.append(build_pad_pipeline(self.pad_spec))

        if self.warp_spec is not None:
            warp_pipeline = build_warp_pipeline(self.warp_spec)
            stages.append(ApplyToIndexStage(warp_pipeline, indices=0))

        if self.shift:
            shift_stage = SubpixelShiftStage(**self.shift_params)
            stages.append(ApplyToIndexStage(shift_stage, indices=0))

        if self.crop_mode is not None:
            stages.append(ProductCropStage(mode=self.crop_mode, **self.crop_kwargs))

        return Pipeline(stages)


def build_product_pipeline(
    result: ImageDetail,
    *,
    config: ProductPipelineConfig | None = None,
    idx: int = 0,
    move_scale: tuple[float, float] | None = None,
    ref_scale: tuple[float, float] | None = None,
    scale_translation: bool = True,
    scale_warp_translation: bool = True,
    apply_pad: bool = True,
    apply_warp: bool = True,
    apply_shift: bool = True,
    crop_mode: Literal["union", "intersection"] | None = None,
    crop_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Example:
        >>> # Search at low resolution
        >>> result = search(ref_lowres, mov_lowres, top_k=1)
        >>> # Apply to high resolution images with cropping to union of content
        >>> pipeline = build_product_pipeline(result, crop_mode="union")
        >>> [registered, reference] = pipeline([moving_hires, reference_hires])
    """
    if config is not None:
        if (
            idx != 0
            or scale_translation is not True
            or scale_warp_translation is not True
            or apply_pad is not True
            or apply_warp is not True
            or apply_shift is not True
            or crop_mode is not None
            or crop_kwargs is not None
        ):
            raise ValueError("Cannot combine config with legacy arguments.")
        return config.build(result)

    from batchmatch.process.pad import build_pad_pipeline
    from batchmatch.process.shift import build_subpixel_shift_pipeline
    from batchmatch.warp.base import build_warp_pipeline

    stages: list[Stage] = []

    stages.append(
        ApplySearchResultStage(
            result,
            idx=idx,
            move_scale=move_scale,
            ref_scale=ref_scale,
            scale_translation=scale_translation,
            scale_warp_translation=scale_warp_translation,
        )
    )

    if apply_pad:
        pad_pipeline = build_pad_pipeline({
            "type": "center_pad",
            "scale": 2,
            "outputs": ["image", "box", "quad", "mask"],
            "shrink_by": 0,
        })
        stages.append(pad_pipeline)

    if apply_warp:
        warp_pipeline = build_warp_pipeline({"outputs": ["image", "mask"]})
        stages.append(ApplyToIndexStage(warp_pipeline, indices=0))

    if apply_shift:
        shift_pipeline = build_subpixel_shift_pipeline(
            source="translation",
            negate=True,
            shift_window=False,
            shift_points=False,
        )
        stages.append(ApplyToIndexStage(shift_pipeline, indices=0))

    if crop_mode is not None:
        crop_stage = ProductCropStage(mode=crop_mode, **(crop_kwargs or {}))
        stages.append(crop_stage)

    return Pipeline(stages)
