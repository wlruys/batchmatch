from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import torch
from batchmatch.base.tensordicts import ImageDetail, NestedKey, WarpParams
from batchmatch.base.pipeline import Stage, StageRegistry
from batchmatch.helpers.box import box_to_quad, quads_to_boxes
from batchmatch.warp.specs import (
    apply_warp_grid,
    compute_warp_grid,
    compute_warp_matrices,
    warp_params_from_image_detail,
    warp_points,
)

Tensor = torch.Tensor

warp_registry = StageRegistry("warp")


def _normalize_boxes(
    boxes: Tensor,
    *,
    batch: int,
    key: NestedKey,
    allow_unbatched: bool,
) -> tuple[Tensor, bool, bool]:
    if boxes.ndim != 3 or boxes.shape[2] != 4:
        raise ValueError(f"{key!r} must have shape (B,N,4), got {tuple(boxes.shape)}.")
    if boxes.shape[0] != batch:
        raise ValueError(
            f"{key!r} batch {boxes.shape[0]} must match transform batch {batch}."
        )
    return boxes, True, False


def _normalize_quads(
    quads: Tensor,
    *,
    batch: int,
    key: NestedKey,
    allow_unbatched: bool,
) -> tuple[Tensor, bool, bool]:
    if quads.ndim != 3 or quads.shape[2] != 8:
        raise ValueError(f"{key!r} must have shape (B,N,8), got {tuple(quads.shape)}.")
    if quads.shape[0] != batch:
        raise ValueError(
            f"{key!r} batch {quads.shape[0]} must match transform batch {batch}."
        )
    return quads, True, False


@warp_registry.register("prepare")
class PrepareWarpStage(Stage):
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.IMAGE,
        ImageDetail.Keys.WARP.ANGLE,
        ImageDetail.Keys.WARP.SCALE_X,
        ImageDetail.Keys.WARP.SCALE_Y,
        ImageDetail.Keys.WARP.SHEAR_X,
        ImageDetail.Keys.WARP.SHEAR_Y,
        ImageDetail.Keys.WARP.TX,
        ImageDetail.Keys.WARP.TY,
    })
    sets: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.WARP.GRID,
        ImageDetail.Keys.WARP.M_FWD,
        ImageDetail.Keys.WARP.M_INV,
    })

    def __init__(
        self,
        *,
        align_corners: bool = False,
        inverse: bool = True,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self._align_corners = align_corners
        self._inverse = inverse
        self._inplace = inplace

    def forward(self, image: ImageDetail) -> ImageDetail:
        img = image.get(ImageDetail.Keys.IMAGE)
        B, C, H, W = img.shape
        
        angle = image.get(ImageDetail.Keys.WARP.ANGLE)
        device = angle.device
        dtype = angle.dtype

        m_fwd_out = image.get(ImageDetail.Keys.WARP.M_FWD, None) if self._inplace else None
        m_inv_out = image.get(ImageDetail.Keys.WARP.M_INV, None) if self._inplace else None
        grid_out = image.get(ImageDetail.Keys.WARP.GRID, None) if self._inplace else None
        
        if m_fwd_out is not None:
            if m_fwd_out.shape[0] >= B and m_fwd_out.shape[1:] == (3, 3) and m_fwd_out.device == device and m_fwd_out.dtype == dtype:
                m_fwd_out = m_fwd_out[:B]
            else:
                m_fwd_out = None
        if m_inv_out is not None:
            if m_inv_out.shape[0] >= B and m_inv_out.shape[1:] == (3, 3) and m_inv_out.device == device and m_inv_out.dtype == dtype:
                m_inv_out = m_inv_out[:B]
            else:
                m_inv_out = None
        if grid_out is not None:
            if (grid_out.shape[0] >= B 
                and grid_out.shape[1:] == (H, W, 2)
                and grid_out.device == device
                and grid_out.dtype == dtype
                and grid_out.is_contiguous()):
                grid_out = grid_out[:B]
            else:
                grid_out = None

        cx_vals = image.get(ImageDetail.Keys.WARP.CX, None)
        cy_vals = image.get(ImageDetail.Keys.WARP.CY, None)

        if cx_vals is None or cx_vals.shape != (B,) or cx_vals.device != device or cx_vals.dtype != dtype:
            cx_vals = torch.ones_like(angle) * ((W - 1.0) * 0.5)
            image.set(ImageDetail.Keys.WARP.CX, cx_vals)

        if cy_vals is None or cy_vals.shape != (B,) or cy_vals.device != device or cy_vals.dtype != dtype:
            cy_vals = torch.ones_like(angle) * ((H - 1.0) * 0.5)
            image.set(ImageDetail.Keys.WARP.CY, cy_vals)

        params = warp_params_from_image_detail(image)

        M_fwd, M_inv = compute_warp_matrices(
            params, H, W, 
            align_corners=self._align_corners,
            out_fwd=m_fwd_out,
            out_inv=m_inv_out,
            inplace=self._inplace,
        )

        matrix = M_fwd if self._inverse else M_inv
        
        grid = compute_warp_grid(
            matrix, H, W, 
            align_corners=self._align_corners,
            out=grid_out,
            inplace=self._inplace,
        )

        if grid_out is None or not self._inplace:
            image.set(ImageDetail.Keys.WARP.GRID, grid)
        if m_fwd_out is None or not self._inplace:
            image.set(ImageDetail.Keys.WARP.M_FWD, M_fwd)
        if m_inv_out is None or not self._inplace:
            image.set(ImageDetail.Keys.WARP.M_INV, M_inv)

        return image


class WarpTensorStageBase(Stage):
    def __init__(
        self,
        *,
        mode: str = "bilinear",
        fill_value: float | Sequence[float] | None = 0.0,
        align_corners: bool = False,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._fill_value = fill_value
        self._align_corners = align_corners
        self._inplace = inplace

    @property
    @abstractmethod
    def _tensor_key(self) -> NestedKey:
        ...

    def forward(self, image: ImageDetail) -> ImageDetail:
        tensor = image.get(self._tensor_key)
        grid = image.get(ImageDetail.Keys.WARP.GRID)
        sample_out = None
        if self._inplace and self._tensor_key == ImageDetail.Keys.IMAGE:
            sample_out = image.get(ImageDetail.Keys.WARP.SAMPLE, None)

        warped = apply_warp_grid(
            tensor, grid,
            mode=self._mode,
            fill_value=self._fill_value,
            align_corners=self._align_corners,
            inplace=self._inplace,
            sample_out=sample_out,
        )

        image.set(self._tensor_key, warped)
        return image


@warp_registry.register("image")
class WarpImageStage(WarpTensorStageBase):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.IMAGE, ImageDetail.Keys.WARP.GRID})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.IMAGE})

    @property
    def _tensor_key(self) -> NestedKey:
        return ImageDetail.Keys.IMAGE


@warp_registry.register("window")
class WarpWindowStage(WarpTensorStageBase):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.DOMAIN.WINDOW, ImageDetail.Keys.WARP.GRID})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.DOMAIN.WINDOW})

    @property
    def _tensor_key(self) -> NestedKey:
        return ImageDetail.Keys.DOMAIN.WINDOW


@warp_registry.register("gx")
class WarpGxStage(WarpTensorStageBase):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.WARP.GRID})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X})

    @property
    def _tensor_key(self) -> NestedKey:
        return ImageDetail.Keys.GRAD.X


@warp_registry.register("gy")
class WarpGyStage(WarpTensorStageBase):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.Y, ImageDetail.Keys.WARP.GRID})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.Y})

    @property
    def _tensor_key(self) -> NestedKey:
        return ImageDetail.Keys.GRAD.Y


@warp_registry.register("mask")
class WarpMaskStage(WarpTensorStageBase):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.DOMAIN.MASK, ImageDetail.Keys.WARP.GRID})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.DOMAIN.MASK})

    def __init__(
        self,
        *,
        fill_value: float | Sequence[float] | None = 0.0,
        align_corners: bool = False,
        inplace: bool = True,
    ) -> None:
        # Masks always use nearest neighbor interpolation
        super().__init__(
            mode="nearest",
            fill_value=fill_value,
            align_corners=align_corners,
            inplace=inplace,
        )

    @property
    def _tensor_key(self) -> NestedKey:
        return ImageDetail.Keys.DOMAIN.MASK

    def forward(self, image: ImageDetail) -> ImageDetail:
        mask = image.get(self._tensor_key)
        grid = image.get(ImageDetail.Keys.WARP.GRID)

        was_bool = mask.dtype == torch.bool
        if was_bool:
            mask = mask.float()

        warped = apply_warp_grid(
            mask, grid,
            mode=self._mode,
            fill_value=self._fill_value,
            align_corners=self._align_corners,
            inplace=self._inplace,
        )

        if was_bool:
            warped = warped > 0.5

        image.set(self._tensor_key, warped)
        return image


@warp_registry.register("points")
class WarpPointsStage(Stage):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.AUX.POINTS, ImageDetail.Keys.WARP.M_INV})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.AUX.POINTS})

    def forward(self, image: ImageDetail) -> ImageDetail:
        pts = image.get(ImageDetail.Keys.AUX.POINTS)
        matrix = image.get(ImageDetail.Keys.WARP.M_INV)

        warped = warp_points(pts, matrix)

        image.set(ImageDetail.Keys.AUX.POINTS, warped)
        return image


class _WarpBoxesStageBase(Stage):
    def __init__(
        self,
        *,
        box_key: NestedKey,
        quad_key: NestedKey,
        allow_unbatched: bool = False,
    ) -> None:
        self._box_key = box_key
        self._quad_key = quad_key
        self._allow_unbatched = allow_unbatched
        super().__init__()

    @property
    def requires(self) -> frozenset[NestedKey]:  # type: ignore[override]
        return frozenset({self._box_key, ImageDetail.Keys.WARP.M_INV})

    @property
    def sets(self) -> frozenset[NestedKey]:  # type: ignore[override]
        return frozenset({self._box_key})

    @property
    def invalidates(self) -> frozenset[NestedKey]:  # type: ignore[override]
        return frozenset()  # Don't invalidate quad - they can coexist

    def forward(self, image: ImageDetail) -> ImageDetail:
        boxes = image.get(self._box_key)
        matrix = image.get(ImageDetail.Keys.WARP.M_INV)

        B = matrix.shape[0]
        boxes_bxn4, _, _ = _normalize_boxes(
            boxes,
            batch=B,
            key=self._box_key,
            allow_unbatched=self._allow_unbatched,
        )
        N = boxes_bxn4.shape[1]

        boxes_flat = boxes_bxn4.reshape(B * N, 4)
        quads_flat = box_to_quad(boxes_flat)
        quads = quads_flat.reshape(B, N, 8)

        pts = quads.view(B, N * 4, 2)
        warped_pts = warp_points(pts, matrix)
        warped_quads = warped_pts.reshape(B, N, 8)

        warped_boxes_flat = quads_to_boxes(warped_quads.reshape(B * N, 8))
        warped_boxes = warped_boxes_flat.view(B, N, 4)

        image.set(self._box_key, warped_boxes)
        return image


@warp_registry.register("boxes")
class WarpBoxesStage(_WarpBoxesStageBase):
    def __init__(self) -> None:
        super().__init__(
            box_key=ImageDetail.Keys.DOMAIN.BOX,
            quad_key=ImageDetail.Keys.DOMAIN.QUAD,
            allow_unbatched=False,
        )


@warp_registry.register("aux_boxes")
class WarpAuxBoxesStage(_WarpBoxesStageBase):
    def __init__(self) -> None:
        super().__init__(
            box_key=ImageDetail.Keys.AUX.BOXES,
            quad_key=ImageDetail.Keys.AUX.QUADS,
            allow_unbatched=True,
        )


class _WarpQuadStageBase(Stage):
    def __init__(
        self,
        *,
        quad_key: NestedKey,
        box_key: NestedKey,
        allow_unbatched: bool = False,
        sets_box: bool = False,
    ) -> None:
        self._quad_key = quad_key
        self._box_key = box_key
        self._allow_unbatched = allow_unbatched
        self._sets_box = sets_box
        super().__init__()

    @property
    def requires(self) -> frozenset[NestedKey]:  # type: ignore[override]
        return frozenset({self._quad_key, ImageDetail.Keys.WARP.M_INV})

    @property
    def sets(self) -> frozenset[NestedKey]:  # type: ignore[override]
        if self._sets_box:
            return frozenset({self._quad_key, self._box_key})
        return frozenset({self._quad_key})

    @property
    def invalidates(self) -> frozenset[NestedKey]:  # type: ignore[override]
        return frozenset()  # Don't invalidate box - they can coexist

    def forward(self, image: ImageDetail) -> ImageDetail:
        quads = image.get(self._quad_key)
        matrix = image.get(ImageDetail.Keys.WARP.M_INV)

        B = matrix.shape[0]
        quads_bxn8, _, _ = _normalize_quads(
            quads,
            batch=B,
            key=self._quad_key,
            allow_unbatched=self._allow_unbatched,
        )
        N = quads_bxn8.shape[1]

        pts = quads_bxn8.view(B, N * 4, 2)
        warped_pts = warp_points(pts, matrix)
        warped_quads = warped_pts.reshape(B, N, 8)

        image.set(self._quad_key, warped_quads)

        if self._sets_box:
            warped_boxes_flat = quads_to_boxes(warped_quads.reshape(B * N, 8))
            warped_boxes = warped_boxes_flat.view(B, N, 4)
            image.set(self._box_key, warped_boxes)

        return image


@warp_registry.register("quad")
class WarpQuadStage(_WarpQuadStageBase):
    def __init__(self, *, sets_box: bool = False) -> None:
        super().__init__(
            quad_key=ImageDetail.Keys.DOMAIN.QUAD,
            box_key=ImageDetail.Keys.DOMAIN.BOX,
            allow_unbatched=False,
            sets_box=sets_box,
        )


@warp_registry.register("aux_quads")
class WarpAuxQuadsStage(_WarpQuadStageBase):
    def __init__(self, *, sets_box: bool = False) -> None:
        super().__init__(
            quad_key=ImageDetail.Keys.AUX.QUADS,
            box_key=ImageDetail.Keys.AUX.BOXES,
            allow_unbatched=True,
            sets_box=sets_box,
        )


def build_warp_stage(name: str, **kwargs) -> Stage:
    return warp_registry.build(name, **kwargs)
