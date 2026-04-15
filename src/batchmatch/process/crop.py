from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail, NestedKey, WarpParams
from batchmatch.base.pipeline import Stage, StageRegistry
from batchmatch.helpers.box import (
    adjust_xyxy_to_crop,
    adjust_quad_to_crop,
    mask_to_box,
)
from batchmatch.helpers.tensor import adjust_points_to_crop

crop_registry = StageRegistry("crop")

_CROP_OUTPUTS = {"image", "mask", "window", "gx", "gy", "box", "quad", "points"}


# ---------------------------------------------------------------------------
# Shared utilities (also imported by pad.py and builders.py)
# ---------------------------------------------------------------------------

def _normalize_stage_outputs(
    outputs: Optional[Sequence[str]],
    valid: set[str],
) -> Optional[set[str]]:
    """Validate and normalize an outputs list against a set of valid names."""
    if outputs is None:
        return None
    normalized = {str(o) for o in outputs}
    if "all" in normalized:
        return set(valid)
    unknown = normalized - valid
    if unknown:
        raise ValueError(f"Unknown outputs: {sorted(unknown)}. Valid: {sorted(valid)}")
    return normalized


def invalidate_warp_computed(warp) -> None:
    """Delete center and computed keys from a WarpParams TensorDict."""
    if warp is None:
        return
    for key in WarpParams.Keys.CENTER:
        if key in warp.keys():
            warp.del_(key)
    for key in WarpParams.Keys.COMPUTED:
        if key in warp.keys():
            warp.del_(key)


def _normalize_crop_outputs(outputs: Optional[Sequence[str]]) -> Optional[set[str]]:
    return _normalize_stage_outputs(outputs, _CROP_OUTPUTS)


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def crop_spatial_to_box(tensor: Tensor, box: Tensor) -> Tensor:
    x0, y0, x1, y1 = box.round().int().tolist()
    return tensor[:, :, y0:y1, x0:x1]


def compute_union_mask(masks: list[Tensor]) -> Tensor:
    stacked = torch.stack(masks, dim=0)
    union = stacked.sum(dim=0).clamp(0, 1)
    return union


def compute_intersection_mask(masks: list[Tensor]) -> Tensor:
    stacked = torch.stack(masks, dim=0)
    intersection = stacked.prod(dim=0).clamp(0, 1)
    return intersection


def _normalize_size(
    value: Optional[int | Tuple[int, int]],
    *,
    default: Tuple[int, int],
) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        return value
    raise ValueError("size must be an int or a tuple of (height, width).")


# ---------------------------------------------------------------------------
# CropStageBase — shared key setup, tensor cropping, and forward loop
# ---------------------------------------------------------------------------

class CropStageBase(Stage):
    DEFAULT_IMAGE_KEY: NestedKey = ImageDetail.Keys.IMAGE
    DEFAULT_MASK_KEY: NestedKey = ImageDetail.Keys.DOMAIN.MASK
    DEFAULT_WINDOW_KEY: NestedKey = ImageDetail.Keys.DOMAIN.WINDOW
    DEFAULT_GX_KEY: NestedKey = ImageDetail.Keys.GRAD.X
    DEFAULT_GY_KEY: NestedKey = ImageDetail.Keys.GRAD.Y
    DEFAULT_BOX_KEY: NestedKey = ImageDetail.Keys.DOMAIN.BOX
    DEFAULT_QUAD_KEY: NestedKey = ImageDetail.Keys.DOMAIN.QUAD
    DEFAULT_POINTS_KEY: NestedKey = ImageDetail.Keys.AUX.POINTS

    def __init__(
        self,
        *,
        outputs: Optional[Sequence[str]] = None,
        crop_image: bool = True,
        crop_mask: bool = True,
        crop_window: bool = False,
        crop_gx: bool = False,
        crop_gy: bool = False,
        adjust_box: bool = False,
        adjust_quad: bool = False,
        adjust_points: bool = False,
        clip_geometry: bool = True,
        invalidate_warp: bool = True,
        image_key: Optional[NestedKey] = None,
        mask_key: Optional[NestedKey] = None,
        window_key: Optional[NestedKey] = None,
        gx_key: Optional[NestedKey] = None,
        gy_key: Optional[NestedKey] = None,
        box_key: Optional[NestedKey] = None,
        quad_key: Optional[NestedKey] = None,
        points_key: Optional[NestedKey] = None,
    ) -> None:
        super().__init__()

        outputs_set = _normalize_crop_outputs(outputs)
        if outputs_set is not None:
            crop_image = "image" in outputs_set
            crop_mask = "mask" in outputs_set
            crop_window = "window" in outputs_set
            crop_gx = "gx" in outputs_set
            crop_gy = "gy" in outputs_set
            adjust_box = "box" in outputs_set
            adjust_quad = "quad" in outputs_set
            adjust_points = "points" in outputs_set

        self._clip_geometry = clip_geometry
        self._invalidate_warp = invalidate_warp

        self._image_key = image_key if image_key is not None else (self.DEFAULT_IMAGE_KEY if crop_image else None)
        self._mask_key = mask_key if mask_key is not None else (self.DEFAULT_MASK_KEY if crop_mask else None)
        self._window_key = window_key if window_key is not None else (self.DEFAULT_WINDOW_KEY if crop_window else None)
        self._gx_key = gx_key if gx_key is not None else (self.DEFAULT_GX_KEY if crop_gx else None)
        self._gy_key = gy_key if gy_key is not None else (self.DEFAULT_GY_KEY if crop_gy else None)
        self._box_key = box_key if box_key is not None else (self.DEFAULT_BOX_KEY if adjust_box else None)
        self._quad_key = quad_key if quad_key is not None else (self.DEFAULT_QUAD_KEY if adjust_quad else None)
        self._points_key = points_key if points_key is not None else (self.DEFAULT_POINTS_KEY if adjust_points else None)

        # Pre-compute active key tuples to avoid repeated None checks per batch element
        self._spatial_keys: tuple[NestedKey, ...] = tuple(
            k for k in (self._image_key, self._mask_key, self._window_key, self._gx_key, self._gy_key)
            if k is not None
        )

    def _get_mask_key(self) -> NestedKey:
        return self._mask_key if self._mask_key is not None else self.DEFAULT_MASK_KEY

    @abstractmethod
    def _combine_masks(self, masks: list[Tensor]) -> Tensor:
        ...

    def _crop_tensors(
        self,
        image_details: list[ImageDetail],
        b: int,
        crop_box: Tensor,
        N: int,
    ) -> ImageDetail:
        """Crop all active keys from each ImageDetail at batch position *b*."""
        cropped_data: dict[NestedKey, list[Tensor]] = {}

        for td in image_details:
            for key in self._spatial_keys:
                tensor = td.get(key, None)
                if tensor is None:
                    continue
                cropped = crop_spatial_to_box(tensor[b : b + 1], crop_box)
                cropped_data.setdefault(key, []).append(cropped)

            if self._box_key is not None:
                box = td.get(self._box_key, None)
                if box is not None:
                    adjusted = adjust_xyxy_to_crop(
                        box[b : b + 1], crop_box, clip=self._clip_geometry
                    )
                    cropped_data.setdefault(self._box_key, []).append(adjusted)

            if self._quad_key is not None:
                quad = td.get(self._quad_key, None)
                if quad is not None:
                    adjusted = adjust_quad_to_crop(
                        quad[b : b + 1], crop_box, clip=self._clip_geometry
                    )
                    cropped_data.setdefault(self._quad_key, []).append(adjusted)

            if self._points_key is not None:
                points = td.get(self._points_key, None)
                if points is not None:
                    adjusted = adjust_points_to_crop(
                        points[b : b + 1], crop_box, clip=self._clip_geometry
                    )
                    cropped_data.setdefault(self._points_key, []).append(adjusted)

        out_data = {}
        for key, tensors in cropped_data.items():
            if len(tensors) == N:
                out_data[key] = torch.cat(tensors, dim=0)

        out_td = ImageDetail(out_data, batch_size=[N])

        if self._invalidate_warp:
            warp = out_td.get(ImageDetail.Keys.WARP.ROOT, None)
            invalidate_warp_computed(warp)

        return out_td

    @staticmethod
    def _parse_inputs(
        first: ImageDetail | list[ImageDetail],
        rest: tuple[ImageDetail, ...],
    ) -> list[ImageDetail]:
        if isinstance(first, list):
            return first
        if rest:
            return [first] + list(rest)
        return [first]

    @staticmethod
    def _validate_inputs(image_details: list[ImageDetail]) -> tuple[int, int, int, int]:
        """Return ``(N, B, H, W)`` after checking batch/spatial consistency."""
        N = len(image_details)
        B = image_details[0].B
        H, W = image_details[0].H, image_details[0].W
        for i, td in enumerate(image_details):
            if td.B != B:
                raise ValueError(
                    f"All inputs must have same batch size. "
                    f"Input 0 has B={B}, input {i} has B={td.B}."
                )
            if td.H != H or td.W != W:
                raise ValueError(
                    f"All inputs must have same spatial dimensions. "
                    f"Input 0 has ({H}, {W}), input {i} has ({td.H}, {td.W})."
                )
        return N, B, H, W

    def forward(
        self,
        first: ImageDetail | list[ImageDetail],
        *rest: ImageDetail,
    ) -> list[ImageDetail]:
        image_details = self._parse_inputs(first, rest)
        if not image_details:
            return []

        N, B, H, W = self._validate_inputs(image_details)
        mask_key = self._get_mask_key()
        output_details: list[ImageDetail] = []

        for b in range(B):
            masks_at_b = []
            for td in image_details:
                mask = td.get(mask_key, None)
                if mask is not None:
                    masks_at_b.append(mask[b : b + 1])
                else:
                    masks_at_b.append(torch.ones(1, 1, H, W, device=td.image.device))

            combined_mask = self._combine_masks(masks_at_b)

            if combined_mask.sum() == 0:
                raise ValueError(
                    f"Empty crop region at batch position {b}. "
                    f"Combined mask may be empty."
                )

            crop_box = mask_to_box(combined_mask).squeeze(0).squeeze(0)
            crop_box = crop_box.clone()
            crop_box[2] += 1
            crop_box[3] += 1

            x0, y0, x1, y1 = crop_box.round().int().tolist()
            if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                raise ValueError(
                    f"Empty crop region at batch position {b}. "
                    f"Combined mask may be empty."
                )

            out_td = self._crop_tensors(image_details, b, crop_box, N)
            output_details.append(out_td)

        return output_details


# ---------------------------------------------------------------------------
# RandomCropStage
# ---------------------------------------------------------------------------

@crop_registry.register("crop_random")
class RandomCropStage(CropStageBase):
    """
    Randomly crop ImageDetails within the intersection of their masks.

    The crop is sampled to satisfy size, area, and aspect constraints and
    is applied per batch element.

    Input: N ImageDetails, each with batch size B.
    Output: List of B ImageDetails, each with batch size N.

    Note this is not the behavior of other stages across the library, which typically take N
    ImageDetails and return inplace modification of those same N ImageDetails.
    Crop stages break this pattern to build crops per batch element.

    Args:
        min_size: Minimum crop size (h, w) or int for square.
        max_size: Maximum crop size (h, w) or int for square.
        min_area: Minimum crop area in pixels.
        max_area: Maximum crop area in pixels.
        min_aspect: Minimum aspect ratio (w / h).
        max_aspect: Maximum aspect ratio (w / h).
        max_attempts: Maximum attempts to sample a valid crop.
    """

    def __init__(
        self,
        *,
        min_size: Optional[int | Tuple[int, int]] = 16,
        max_size: Optional[int | Tuple[int, int]] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        min_aspect: Optional[float] = None,
        max_aspect: Optional[float] = None,
        max_attempts: int = 50,
        allow_full_image: bool = False,
        outputs: Optional[Sequence[str]] = None,
        crop_image: bool = True,
        crop_mask: bool = True,
        crop_window: bool = False,
        crop_gx: bool = False,
        crop_gy: bool = False,
        adjust_box: bool = False,
        adjust_quad: bool = False,
        adjust_points: bool = False,
        clip_geometry: bool = True,
        invalidate_warp: bool = True,
        image_key: Optional[NestedKey] = None,
        mask_key: Optional[NestedKey] = None,
        window_key: Optional[NestedKey] = None,
        gx_key: Optional[NestedKey] = None,
        gy_key: Optional[NestedKey] = None,
        box_key: Optional[NestedKey] = None,
        quad_key: Optional[NestedKey] = None,
        points_key: Optional[NestedKey] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(
            outputs=outputs,
            crop_image=crop_image, crop_mask=crop_mask,
            crop_window=crop_window, crop_gx=crop_gx, crop_gy=crop_gy,
            adjust_box=adjust_box, adjust_quad=adjust_quad, adjust_points=adjust_points,
            clip_geometry=clip_geometry, invalidate_warp=invalidate_warp,
            image_key=image_key, mask_key=mask_key, window_key=window_key,
            gx_key=gx_key, gy_key=gy_key, box_key=box_key,
            quad_key=quad_key, points_key=points_key,
        )

        self._min_size = _normalize_size(min_size, default=(1, 1))
        self._max_size = (
            None if max_size is None else _normalize_size(max_size, default=(0, 0))
        )
        self._min_area = min_area
        self._max_area = max_area
        self._min_aspect = min_aspect
        self._max_aspect = max_aspect
        self._max_attempts = max_attempts
        self._allow_full_image = allow_full_image
        self._generator = generator

        self.requires = frozenset({self.DEFAULT_IMAGE_KEY})

    def _combine_masks(self, masks: list[Tensor]) -> Tensor:
        combined = compute_intersection_mask(masks)
        if combined.sum() == 0 and self._allow_full_image:
            return torch.ones_like(combined)
        return combined

    def _randint(self, low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (1,), generator=self._generator).item())

    def _sample_crop_size(self, max_h: int, max_w: int) -> Tuple[int, int]:
        min_h, min_w = self._min_size
        max_h = max_h if self._max_size is None else min(max_h, self._max_size[0])
        max_w = max_w if self._max_size is None else min(max_w, self._max_size[1])

        min_h = max(1, min_h)
        min_w = max(1, min_w)

        if min_h > max_h or min_w > max_w:
            raise ValueError("No valid crop size within the domain bounds.")

        for _ in range(self._max_attempts):
            h = int(torch.randint(min_h, max_h + 1, (1,), generator=self._generator).item())
            w = int(torch.randint(min_w, max_w + 1, (1,), generator=self._generator).item())
            area = h * w
            if self._min_area is not None and area < self._min_area:
                continue
            if self._max_area is not None and area > self._max_area:
                continue
            aspect = w / float(h)
            if self._min_aspect is not None and aspect < self._min_aspect:
                continue
            if self._max_aspect is not None and aspect > self._max_aspect:
                continue
            return h, w

        raise ValueError("Failed to sample a valid crop size.")

    def forward(
        self,
        first: ImageDetail | list[ImageDetail],
        *rest: ImageDetail,
    ) -> list[ImageDetail]:
        image_details = self._parse_inputs(first, rest)
        if not image_details:
            return []

        N, B, H, W = self._validate_inputs(image_details)
        mask_key = self._get_mask_key()
        output_details: list[ImageDetail] = []

        for b in range(B):
            masks_at_b = []
            for td in image_details:
                mask = td.get(mask_key, None)
                if mask is not None:
                    masks_at_b.append(mask[b : b + 1])
                else:
                    masks_at_b.append(torch.ones(1, 1, H, W, device=td.image.device))

            combined_mask = self._combine_masks(masks_at_b)

            if combined_mask.sum() == 0:
                raise ValueError(
                    f"Empty crop region at batch position {b}. "
                    "Combined mask may be empty."
                )

            domain_box = mask_to_box(combined_mask).squeeze(0).squeeze(0)
            domain_box = domain_box.clone()
            domain_box[2] += 1
            domain_box[3] += 1

            x0, y0, x1, y1 = domain_box.round().int().tolist()
            domain_h = y1 - y0
            domain_w = x1 - x0

            if domain_h <= 0 or domain_w <= 0:
                raise ValueError(
                    f"Domain box has non-positive extent at batch position {b}."
                )

            crop_h, crop_w = self._sample_crop_size(domain_h, domain_w)

            max_x0 = x1 - crop_w
            max_y0 = y1 - crop_h

            if max_x0 < x0 or max_y0 < y0:
                raise ValueError("Crop size exceeds domain bounds.")

            cx0 = self._randint(x0, max_x0)
            cy0 = self._randint(y0, max_y0)

            crop_box = torch.tensor(
                [cx0, cy0, cx0 + crop_w, cy0 + crop_h],
                device=image_details[0].image.device,
                dtype=image_details[0].image.dtype,
            )

            out_td = self._crop_tensors(image_details, b, crop_box, N)
            output_details.append(out_td)

        if len(output_details) == 1:
            return output_details[0]

        return output_details


# ---------------------------------------------------------------------------
# Mask-based crop stages
# ---------------------------------------------------------------------------

@crop_registry.register("crop_union")
class CropUnionStage(CropStageBase):
    def _combine_masks(self, masks: list[Tensor]) -> Tensor:
        return compute_union_mask(masks)


@crop_registry.register("crop_intersection")
class CropIntersectionStage(CropStageBase):
    def _combine_masks(self, masks: list[Tensor]) -> Tensor:
        return compute_intersection_mask(masks)


def build_crop_stage(name: str = "crop_union", **kwargs) -> Stage:
    return crop_registry.build(name, **kwargs)
