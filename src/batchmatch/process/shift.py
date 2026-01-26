from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from batchmatch.base.tensordicts import (
    ImageDetail,
    NestedKey,
    WarpParams,
)
from batchmatch.base.pipeline import Pipeline, Stage, StageRegistry
from batchmatch.helpers.box import shift_quad_batch, shift_xyxy_batch
from batchmatch.helpers.tensor import shift_points

shift_registry = StageRegistry("shift")


def shift_spatial_batch(
    tensor: Tensor,
    *,
    dx: Tensor,
    dy: Tensor,
    fill_value: float = 0.0,
) -> Tensor:
    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    dx_int = dx.round().to(torch.int64)
    dy_int = dy.round().to(torch.int64)

    if (dx_int == dx_int[0]).all() and (dy_int == dy_int[0]).all():
        shift_x = int(dx_int[0].item())
        shift_y = int(dy_int[0].item())
        out = torch.roll(tensor, shifts=(shift_y, shift_x), dims=(-2, -1))

        if fill_value == 0.0 and shift_x == 0 and shift_y == 0:
            return out

        if shift_y > 0:
            out[:, :, :shift_y, :] = fill_value
        elif shift_y < 0:
            out[:, :, shift_y:, :] = fill_value

        if shift_x > 0:
            out[:, :, :, :shift_x] = fill_value
        elif shift_x < 0:
            out[:, :, :, shift_x:] = fill_value

        return out

    out = torch.full_like(tensor, fill_value)

    for b in range(B):
        sx = int(dx_int[b].item())
        sy = int(dy_int[b].item())

        if sx == 0 and sy == 0:
            out[b] = tensor[b]
            continue

        # Positive shift moves right/down
        # source is left/top
        src_y_start = max(0, -sy)
        src_y_end = min(H, H - sy)
        src_x_start = max(0, -sx)
        src_x_end = min(W, W - sx)

        dst_y_start = max(0, sy)
        dst_y_end = min(H, H + sy)
        dst_x_start = max(0, sx)
        dst_x_end = min(W, W + sx)

        if src_y_end > src_y_start and src_x_end > src_x_start:
            out[b, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                tensor[b, :, src_y_start:src_y_end, src_x_start:src_x_end]

    return out


OffsetSource = Literal["warp", "translation"]


@shift_registry.register("shift")
class ShiftStage(Stage):
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
        source: OffsetSource = "translation",
        negate: bool = False,
        shift_image: bool = True,
        shift_mask: bool = True,
        shift_window: bool = True,
        shift_gx: bool = True,
        shift_gy: bool = True,
        shift_box: bool = True,
        shift_quad: bool = True,
        shift_points: bool = True,
        image_key: Optional[NestedKey] = None,
        mask_key: Optional[NestedKey] = None,
        window_key: Optional[NestedKey] = None,
        gx_key: Optional[NestedKey] = None,
        gy_key: Optional[NestedKey] = None,
        box_key: Optional[NestedKey] = None,
        quad_key: Optional[NestedKey] = None,
        points_key: Optional[NestedKey] = None,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()

        self._source = source
        self._negate = negate
        self._fill_value = fill_value

        self._image_key = image_key if image_key is not None else (self.DEFAULT_IMAGE_KEY if shift_image else None)
        self._mask_key = mask_key if mask_key is not None else (self.DEFAULT_MASK_KEY if shift_mask else None)
        self._window_key = window_key if window_key is not None else (self.DEFAULT_WINDOW_KEY if shift_window else None)
        self._gx_key = gx_key if gx_key is not None else (self.DEFAULT_GX_KEY if shift_gx else None)
        self._gy_key = gy_key if gy_key is not None else (self.DEFAULT_GY_KEY if shift_gy else None)
        self._box_key = box_key if box_key is not None else (self.DEFAULT_BOX_KEY if shift_box else None)
        self._quad_key = quad_key if quad_key is not None else (self.DEFAULT_QUAD_KEY if shift_quad else None)
        self._points_key = points_key if points_key is not None else (self.DEFAULT_POINTS_KEY if shift_points else None)

    def _get_offsets(self, image: ImageDetail) -> tuple[Tensor, Tensor]:
        if self._source == "warp":
            tx = image.get(ImageDetail.Keys.WARP.TX, None)
            ty = image.get(ImageDetail.Keys.WARP.TY, None)
            if tx is None or ty is None:
                raise ValueError("WarpParams tx/ty not found in ImageDetail")
        elif self._source == "translation":
            tx = image.get(ImageDetail.Keys.TRANSLATION.X, None)
            ty = image.get(ImageDetail.Keys.TRANSLATION.Y, None)
            if tx is None or ty is None:
                raise ValueError("TranslationResults x/y not found in ImageDetail")
        else:
            raise ValueError(f"Unknown offset source: {self._source}")

        if self._negate:
            tx = -tx
            ty = -ty

        return tx, ty

    def _shift_spatial(self, image: ImageDetail, key: Optional[NestedKey], tx: Tensor, ty: Tensor):
        if key is None:
            return
        tensor = image.get(key, None)
        if tensor is None:
            return
        shifted = shift_spatial_batch(tensor, dx=tx, dy=ty, fill_value=self._fill_value)
        image.set(key, shifted)

    def _shift_box(self, image: ImageDetail, key: Optional[NestedKey], tx: Tensor, ty: Tensor):
        if key is None:
            return
        boxes = image.get(key, None)
        if boxes is None:
            return
        shifted = shift_xyxy_batch(boxes, dx=tx, dy=ty)
        image.set(key, shifted)

    def _shift_quad(self, image: ImageDetail, key: Optional[NestedKey], tx: Tensor, ty: Tensor):
        if key is None:
            return
        quads = image.get(key, None)
        if quads is None:
            return
        shifted = shift_quad_batch(quads, dx=tx, dy=ty)
        image.set(key, shifted)

    def _shift_points(self, image: ImageDetail, key: Optional[NestedKey], tx: Tensor, ty: Tensor):
        if key is None:
            return
        points = image.get(key, None)
        if points is None:
            return
        shifted = shift_points(points, dx=tx, dy=ty)
        image.set(key, shifted)

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
            tx, ty = self._get_offsets(img)

            self._shift_spatial(img, self._image_key, tx, ty)
            self._shift_spatial(img, self._mask_key, tx, ty)
            self._shift_spatial(img, self._window_key, tx, ty)
            self._shift_spatial(img, self._gx_key, tx, ty)
            self._shift_spatial(img, self._gy_key, tx, ty)

            self._shift_box(img, self._box_key, tx, ty)
            self._shift_quad(img, self._quad_key, tx, ty)
            self._shift_points(img, self._points_key, tx, ty)

        if isinstance(image, list):
            return images
        else:
            return images[0]


@shift_registry.register("subpixel")
class SubpixelShiftStage(Stage):
    """
    Apply sub-pixel translation using bilinear interpolation.
    """

    _auto_validate: bool = False
    _auto_invalidate: bool = False

    def __init__(
        self,
        *,
        source: OffsetSource = "translation",
        negate: bool = False,
        shift_image: bool = True,
        shift_mask: bool = True,
        shift_window: bool = True,
        shift_box: bool = True,
        shift_quad: bool = True,
        shift_points: bool = True,
        mode: str = "bilinear",
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self._source = source
        self._negate = negate
        self._shift_image = shift_image
        self._shift_mask = shift_mask
        self._shift_window = shift_window
        self._shift_box = shift_box
        self._shift_quad = shift_quad
        self._shift_points = shift_points
        self._mode = mode
        self._fill_value = fill_value

    def _get_offsets(self, image: ImageDetail) -> tuple[Tensor, Tensor]:
        if self._source == "warp":
            tx = image.get(ImageDetail.Keys.WARP.TX, None)
            ty = image.get(ImageDetail.Keys.WARP.TY, None)
            if tx is None or ty is None:
                raise ValueError("WarpParams tx/ty not found in ImageDetail")
        elif self._source == "translation":
            tx = image.get(ImageDetail.Keys.TRANSLATION.X, None)
            ty = image.get(ImageDetail.Keys.TRANSLATION.Y, None)
            if tx is None or ty is None:
                raise ValueError("TranslationResults x/y not found in ImageDetail")
        else:
            raise ValueError(f"Unknown offset source: {self._source}")

        if self._negate:
            tx = -tx
            ty = -ty

        return tx, ty

    def forward(self, image: ImageDetail) -> ImageDetail:
        from batchmatch.warp.base import build_warp_pipeline

        tx, ty = self._get_offsets(image)
        device = image.image.device
        dtype = image.image.dtype
        B = image.B

        # Create identity warp with just translation
        warp_params = WarpParams.from_components(
            angle=torch.zeros(B, device=device, dtype=dtype),
            scale_x=torch.ones(B, device=device, dtype=dtype),
            scale_y=torch.ones(B, device=device, dtype=dtype),
            shear_x=torch.zeros(B, device=device, dtype=dtype),
            shear_y=torch.zeros(B, device=device, dtype=dtype),
            tx=tx,
            ty=ty,
        )
        image.set(ImageDetail.Keys.WARP.ROOT, warp_params)

        # Build warp pipeline only for components that exist
        warp_spec: dict = {"prepare": {"type": "prepare"}}
        #TODO(wlr): This is too expensize, we should only build the warp spec once, but okay for now
        #Note if we need this in an optimization loop this will be very slow

        if self._shift_image and image.get(ImageDetail.Keys.IMAGE, default=None) is not None:
            warp_spec["image"] = {"type": "image", "fill_value": self._fill_value, "mode": self._mode}
        if self._shift_mask and image.get(ImageDetail.Keys.DOMAIN.MASK, default=None) is not None:
            warp_spec["mask"] = {"type": "mask", "fill_value": self._fill_value}
        if self._shift_window and image.get(ImageDetail.Keys.DOMAIN.WINDOW, default=None) is not None:
            warp_spec["window"] = {"type": "window", "fill_value": self._fill_value, "mode": self._mode}
        if self._shift_box and image.get(ImageDetail.Keys.DOMAIN.BOX, default=None) is not None:
            warp_spec["boxes"] = {"type": "boxes"}
        if self._shift_quad and image.get(ImageDetail.Keys.DOMAIN.QUAD, default=None) is not None:
            warp_spec["quad"] = {"type": "quad"}
        if self._shift_points and image.get(ImageDetail.Keys.AUX.POINTS, default=None) is not None:
            warp_spec["points"] = {"type": "points"}

        warp_pipeline = build_warp_pipeline(warp_spec)
        return warp_pipeline(image)


def build_shift_stage(name: str = "shift", **kwargs) -> Stage:
    return shift_registry.build(name, **kwargs)


def build_shift_pipeline(**kwargs) -> Pipeline:
    stage = ShiftStage(**kwargs)
    return Pipeline([stage])


def build_subpixel_shift_pipeline(
    source: OffsetSource = "translation",
    negate: bool = False,
    **kwargs,
) -> Pipeline:
    stage = SubpixelShiftStage(source=source, negate=negate, **kwargs)
    return Pipeline([stage])
