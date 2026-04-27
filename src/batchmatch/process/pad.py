from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, ClassVar, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail, NestedKey, WarpParams, _normalize_keys
from batchmatch.base.pipeline import Pipeline, Stage, StageRegistry, StageSpec, coerce_stage_spec
from batchmatch.helpers.box import pad_to_box, pad_to_quad, shrink_xyxy, shrink_quad
from batchmatch.process.crop import invalidate_warp_computed
from batchmatch.process.window import make_box_window_batched

pad_registry = StageRegistry("pad")


def _center_pad(
    H: int,
    W: int,
    Ht: int,
    Wt: int,
) -> tuple[int, int, int, int]:
    if H > Ht or W > Wt:
        raise ValueError(f"Target {(Ht, Wt)} must be >= input {(H, W)}.")

    dy = Ht - H
    dx = Wt - W
    top = dy // 2
    bottom = dy - top
    left = dx // 2
    right = dx - left

    return (left, top, right, bottom)


_PAD_OUTPUTS = {"image", "box", "quad", "mask", "window"}


def _normalize_outputs(outputs: Optional[Sequence[str]]) -> Optional[set[str]]:
    from batchmatch.process.crop import _normalize_stage_outputs
    return _normalize_stage_outputs(outputs, _PAD_OUTPUTS)


@pad_registry.register("center_pad")
class CenterPad(Stage):
    def __init__(
        self,
        *,
        scale: Optional[float | Tuple[float, float]] = None,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        box_key: Optional[NestedKey] = None,
        quad_key: Optional[NestedKey] = None,
        mask_key: Optional[NestedKey] = None,
        window_key: Optional[NestedKey] = None,
        outputs: Optional[Sequence[str]] = None,
        create_box: bool = True,
        create_mask: bool = True,
        create_quad: bool = True,
        create_window: bool = True,
        window_alpha: float = 0.05,
        shrink_by: Optional[int] = 4,
        pad_to_even: bool = False,
        pad_to_pow2: bool = True,
    ):

        outputs_set = _normalize_outputs(outputs)
        if outputs_set is not None:
            create_box = "box" in outputs_set
            create_mask = "mask" in outputs_set
            create_quad = "quad" in outputs_set
            create_window = "window" in outputs_set

        self._scale = scale
        image_keys = image_keys if image_keys is not None else [ImageDetail.Keys.IMAGE]
        self._image_keys = _normalize_keys(image_keys)
        self._box_key = box_key if box_key is not None else ImageDetail.Keys.DOMAIN.BOX
        self._quad_key = quad_key if quad_key is not None else ImageDetail.Keys.DOMAIN.QUAD
        self._mask_key = mask_key if mask_key is not None else ImageDetail.Keys.DOMAIN.MASK
        self._window_key = window_key if window_key is not None else ImageDetail.Keys.DOMAIN.WINDOW
        self._shrink_by = shrink_by
        self._pad_to_even = pad_to_even
        self._pad_to_pow2 = pad_to_pow2
        self._create_box = create_box
        self._create_mask = create_mask
        self._create_quad = create_quad
        self._create_window = create_window
        self._window_alpha = window_alpha

        sets = set()
        reqs = set()

        if self._image_keys:
            sets.update(k for k in self._image_keys)
            reqs.update(k for k in self._image_keys)
        else:
            sets.add(ImageDetail.Keys.IMAGE)
            reqs.add(ImageDetail.Keys.IMAGE)

        if self._create_box:
            sets.add(self._box_key)

        if self._create_quad:
            sets.add(self._quad_key)

        if self._create_mask:
            sets.add(self._mask_key)

        if self._create_window:
            sets.add(self._window_key)

        self.requires: ClassVar[frozenset[NestedKey]] = frozenset(reqs)
        self.sets: ClassVar[frozenset[NestedKey]] = frozenset(sets)

        super().__init__()


    def get_scale(self)-> Tuple[float, float]:
        if self._scale is None:
            return (1.0, 1.0)
        if isinstance(self._scale, float) or isinstance(self._scale, int):
            return (self._scale, self._scale)
        elif isinstance(self._scale, tuple) and len(self._scale) == 2:
            return self._scale
        else:
            raise ValueError("Scale must be a float or a tuple of two floats.")

    def get_target_size(self, images: list[ImageDetail]) -> tuple[int, int]:
        scale_x, scale_y = self.get_scale()

        max_h = max(img.H for img in images)
        max_w = max(img.W for img in images)

        Ht = int(max_h * scale_y)
        Wt = int(max_w * scale_x)

        if self._pad_to_pow2:
            Ht = 1 << (Ht - 1).bit_length()
            Wt = 1 << (Wt - 1).bit_length()

        if self._pad_to_even:
            if Ht % 2 != 0:
                Ht += 1
            if Wt % 2 != 0:
                Wt += 1

        return Ht, Wt

    def get_pad(self, images: list[ImageDetail]) -> tuple[int, int, int, int]:
        Ht, Wt = self.get_target_size(images)
        max_h = max(img.H for img in images)
        max_w = max(img.W for img in images)
        return _center_pad(max_h, max_w, Ht, Wt)

    def create_padded_image(
        self,
        image: Tensor,
        pad: tuple[int, int, int, int],
        fill: float = 0.0,
    ) -> Tensor:
        # pad is (left, top, right, bottom), F.pad expects (left, right, top, bottom)
        # TODO(wlr): Fix this inconsistency, change our format to match F.pad
        left, top, right, bottom = pad
        return F.pad(image, (left, right, top, bottom), mode="constant", value=fill)

    def create_box(
        self,
        B: int,
        H: int,
        W: int,
        pad: tuple[int, int, int, int],
    ):
        return pad_to_box(B, H, W, pad)

    def create_mask(
        self,
        B: int,
        H: int,
        W: int,
        pad: tuple[int, int, int, int],
    ) -> Tensor:
        left, top, right, bottom = pad
        mask = torch.zeros((B, H, W), dtype=torch.float32)
        mask[:, top:H - bottom, left:W - right] = 1.0
        return mask.unsqueeze(1)

    def create_quad(
        self,
        B: int,
        H: int,
        W: int,
        pad: tuple[int, int, int, int],
    ):
        return pad_to_quad(B, H, W, pad)

    def _invalidate_warp(self, img: ImageDetail) -> None:
        warp = img.get(ImageDetail.Keys.WARP.ROOT, None)
        invalidate_warp_computed(warp)

    def _pad_single(
        self,
        img: ImageDetail,
        Ht: int,
        Wt: int,
    ) -> tuple[ImageDetail, tuple[int, int, int, int]]:
        """Pad a single *img* to target size ``(Ht, Wt)``.

        Returns ``(detail, (left, top, right, bottom))`` so callers that
        need the padding offsets (e.g. spatial wrappers) can obtain them
        without re-deriving them from the target size.
        """
        B, C, H, W = img.image.shape
        pad = _center_pad(H, W, Ht, Wt)

        for ikey in self._image_keys:
            padded_image = self.create_padded_image(img.get(ikey), pad, fill=0.0)
            img.set(ikey, padded_image)

        self._invalidate_warp(img)

        # Create base box for the original image region
        base_box = self.create_box(B, H, W, pad)
        base_quad = self.create_quad(B, H, W, pad)
        device = img.image.device
        base_box = base_box.to(device=device)
        base_quad = base_quad.to(device=device)

        if self._shrink_by is not None and self._shrink_by > 0:
            base_box = shrink_xyxy(base_box, self._shrink_by)
            base_quad = shrink_quad(base_quad, self._shrink_by)

        if self._create_box:
            img.set(self._box_key, base_box)

        if self._create_mask:
            # Create mask from the (possibly shrunk) box
            left, top, right, bottom = pad
            shrink = self._shrink_by if self._shrink_by is not None else 0
            mask = torch.zeros((B, Ht, Wt), dtype=torch.float32, device=device)
            mask[:, top + shrink:Ht - bottom - shrink, left + shrink:Wt - right - shrink] = 1.0
            img.set(self._mask_key, mask.unsqueeze(1))

        if self._create_quad:
            img.set(self._quad_key, base_quad)

        if self._create_window:
            # Create window from the (possibly shrunk) box
            window = make_box_window_batched(Ht, Wt, base_box, alpha=self._window_alpha)
            img.set(self._window_key, window.unsqueeze(1))

        return img, pad

    def forward(
        self,
        image: ImageDetail | list[ImageDetail],
        *args: ImageDetail,
    ) -> ImageDetail | list[ImageDetail]:

        if isinstance(image, list):
            images = image
        else:
            images = [image] + list(args)

        Ht, Wt = self.get_target_size(images)

        out_images = []
        for img in images:
            img, _pad = self._pad_single(img, Ht, Wt)
            out_images.append(img)

        if isinstance(image, list):
            return out_images
        else:
            return out_images[0]


def build_pad_operator(
    pad_type: str,
    **kwargs,
) -> Stage:
    return pad_registry.build(pad_type, **kwargs)

@dataclass(frozen=True)
class PadPipelineSpec:
    stage: StageSpec

def coerce_pad_pipeline_spec(
    value: object,
    *,
    field_name: str = "pad_pipeline",
) -> PadPipelineSpec:
    if isinstance(value, PadPipelineSpec):
        return value

    if isinstance(value, str):
        stage_spec = coerce_stage_spec(value, field_name=f"{field_name}.stage")
        return PadPipelineSpec(stage=stage_spec)

    if isinstance(value, Mapping):
        raw = dict(value)

        if "stage" in raw:
            stage = coerce_stage_spec(
                raw.pop("stage"),
                field_name=f"{field_name}.stage",
                allow_none=False,
            )
            return PadPipelineSpec(stage=stage)

        if "type" in raw or "name" in raw:
             stage = coerce_stage_spec(
                raw,
                field_name=f"{field_name}.stage",
                allow_none=False,
            )
             return PadPipelineSpec(stage=stage)

    raise TypeError(
        f"Cannot coerce {value!r} to PadPipelineSpec for field '{field_name}'."
    )

def build_pad_pipeline(
    spec: Union[PadPipelineSpec, Mapping, str],
    **kwargs,
) -> Pipeline:
    if isinstance(spec, str) and kwargs:
        spec = {"type": spec, **kwargs}

    if isinstance(spec, PadPipelineSpec):
        pipeline_spec = spec
    else:
        pipeline_spec = coerce_pad_pipeline_spec(spec)

    stages: list[Stage] = []
    if pipeline_spec.stage is not None:
        stage_spec = coerce_stage_spec(
            pipeline_spec.stage,
            field_name="pad_pipeline.stage",
            allow_none=False,
        )
        stage = pad_registry.build(stage_spec.type, **stage_spec.params)
        stages.append(stage)

    return Pipeline(stages)
