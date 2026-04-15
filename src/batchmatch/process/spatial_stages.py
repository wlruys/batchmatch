"""SpatialImage-aware wrappers for the process stages used in the
registration path (pad, resize). Each wrapper delegates tensor work to the
existing :class:`Stage` and reports the exact ``M_next_from_current``.

These stages flow through the same pipeline shape as the ``Stage``
hierarchy but on :class:`SpatialImage`, composing into
:attr:`ImageSpace.matrix_image_from_source`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from batchmatch.base.spatial import SpatialStage, _compose
from batchmatch.base.tensordicts import ImageDetail, NestedKey
from batchmatch.helpers.affine import mat_pad, mat_resize
from batchmatch.io.space import SpatialImage
from batchmatch.process.pad import CenterPad as _CenterPad
from batchmatch.process.resize import (
    ScaleResize as _ScaleResize,
    TargetResize as _TargetResize,
    UnitResize as _UnitResize,
    PhysicalResize as _PhysicalResize,
)

__all__ = [
    "SpatialCenterPad",
    "SpatialScaleResize",
    "SpatialTargetResize",
    "SpatialUnitResize",
    "SpatialPhysicalResize",
]


class SpatialCenterPad(SpatialStage):
    """Pad each :class:`SpatialImage` to a shared centered canvas.

    Uses the same sizing rules as :class:`batchmatch.process.pad.CenterPad`
    (optional scale, pow2/even rounding). On a list input, the target size
    is shared across all images; single-image input scales the image alone.
    """

    def __init__(
        self,
        *,
        inner: _CenterPad | None = None,
        scale: Optional[float | Tuple[float, float]] = None,
        image_keys: Optional[Union[NestedKey, Sequence[NestedKey]]] = None,
        outputs: Optional[Sequence[str]] = None,
        create_box: bool = True,
        create_mask: bool = True,
        create_quad: bool = True,
        create_window: bool = True,
        window_alpha: float = 0.05,
        shrink_by: Optional[int] = 4,
        pad_to_even: bool = False,
        pad_to_pow2: bool = True,
    ) -> None:
        if inner is not None:
            self._inner = inner
        else:
            self._inner = _CenterPad(
                scale=scale,
                image_keys=image_keys,
                outputs=outputs,
                create_box=create_box,
                create_mask=create_mask,
                create_quad=create_quad,
                create_window=create_window,
                window_alpha=window_alpha,
                shrink_by=shrink_by,
                pad_to_even=pad_to_even,
                pad_to_pow2=pad_to_pow2,
            )

    def _handles_list(self) -> bool:
        return True

    def _apply(self, detail: ImageDetail) -> tuple[ImageDetail, np.ndarray]:
        results = self._apply_many([detail])
        return results[0]

    def _apply_many(
        self,
        details: list[ImageDetail],
    ) -> list[tuple[ImageDetail, np.ndarray]]:
        Ht, Wt = self._inner.get_target_size(details)
        out: list[tuple[ImageDetail, np.ndarray]] = []
        for detail in details:
            new_detail, pad = self._inner._pad_single(detail, Ht, Wt)
            left, top, right, bottom = pad
            matrix = mat_pad(float(left), float(top), float(right), float(bottom))
            out.append((new_detail, matrix))
        return out


class SpatialScaleResize(SpatialStage):
    """Spatial wrapper around :class:`ScaleResize`.

    Accepts either an explicit *inner* stage or keyword arguments
    forwarded to the :class:`ScaleResize` constructor. Passing a
    pre-built stage lets callers use any :class:`ScaleResize` subclass.
    """

    def __init__(
        self,
        *,
        inner: _ScaleResize | None = None,
        scale: float | Tuple[float, float] = 1.0,
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
        if inner is not None:
            self._inner = inner
        else:
            self._inner = _ScaleResize(
                scale=scale,
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

    def _apply(self, detail: ImageDetail) -> tuple[ImageDetail, np.ndarray]:
        H_in, W_in = int(detail.H), int(detail.W)
        new_detail = self._inner.forward(detail)
        H_out, W_out = int(new_detail.H), int(new_detail.W)
        matrix = mat_resize(W_in, H_in, W_out, H_out)
        return new_detail, matrix


class SpatialTargetResize(SpatialStage):
    """Spatial wrapper around :class:`TargetResize` (shared scale across inputs).

    Accepts either an explicit *inner* :class:`TargetResize` (or any
    :class:`ScaleResize` subclass that implements ``compute_scale_factor``)
    or keyword arguments forwarded to the :class:`TargetResize` constructor.
    """

    def __init__(
        self,
        *,
        inner: _TargetResize | None = None,
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
    ) -> None:
        if inner is not None:
            self._inner = inner
        else:
            self._inner = _TargetResize(
                target_width=target_width,
                target_height=target_height,
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

    def _handles_list(self) -> bool:
        return True

    def _apply(self, detail: ImageDetail) -> tuple[ImageDetail, np.ndarray]:
        results = self._apply_many([detail])
        return results[0]

    def _apply_many(
        self,
        details: list[ImageDetail],
    ) -> list[tuple[ImageDetail, np.ndarray]]:
        sx, sy = self._inner.compute_scale_factor(
            details,
            target_width=self._inner._target_width,
            target_height=self._inner._target_height,
        )
        out: list[tuple[ImageDetail, np.ndarray]] = []
        for detail in details:
            H_in, W_in = int(detail.H), int(detail.W)
            new_detail = self._inner.forward_at_scale(detail, sx, sy)
            H_out, W_out = int(new_detail.H), int(new_detail.W)
            matrix = mat_resize(W_in, H_in, W_out, H_out)
            out.append((new_detail, matrix))
        return out


class SpatialUnitResize(SpatialStage):
    """Spatial wrapper around :class:`UnitResize` (per-image scale factors).

    Works with any :class:`UnitResize` subclass, including
    :class:`CellUnitResize`, because scale-factor computation is delegated
    to :meth:`UnitResize._compute_scale_factors` (which ``CellUnitResize``
    overrides to estimate cell radii first).
    """

    def __init__(self, *, inner: _UnitResize) -> None:
        self._inner = inner

    def _handles_list(self) -> bool:
        return True

    def _apply(self, detail: ImageDetail) -> tuple[ImageDetail, np.ndarray]:
        results = self._apply_many([detail])
        return results[0]

    def _apply_many(
        self,
        details: list[ImageDetail],
    ) -> list[tuple[ImageDetail, np.ndarray]]:
        scales = self._inner._compute_scale_factors(details)
        out: list[tuple[ImageDetail, np.ndarray]] = []
        for detail, (sx, sy) in zip(details, scales):
            H_in, W_in = int(detail.H), int(detail.W)
            new_detail = self._inner._forward_single_at_scale(detail, sx, sy)
            H_out, W_out = int(new_detail.H), int(new_detail.W)
            matrix = mat_resize(W_in, H_in, W_out, H_out)
            out.append((new_detail, matrix))
        return out


class SpatialPhysicalResize(SpatialStage):
    """Normalize images to a common physical pixel size using TIFF metadata.

    Reads ``pixel_size_xy`` from each :class:`SpatialImage`'s
    :attr:`~ImageSpace.source` and scales images so that every pixel
    covers the same physical area as the reference image.

    Unlike other spatial wrappers this stage needs the full
    :class:`SpatialImage` (not just :class:`ImageDetail`) to access
    source metadata, so it overrides :meth:`forward` directly.
    """

    def __init__(
        self,
        *,
        inner: _PhysicalResize | None = None,
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
        if inner is not None:
            self._inner = inner
        else:
            self._inner = _PhysicalResize(
                reference_index=reference_index,
                is_mask=is_mask,
                image_keys=image_keys,
                mask_keys=mask_keys,
                box_keys=box_keys,
                quad_keys=quad_keys,
                point_keys=point_keys,
                warp_keys=warp_keys,
                translation_keys=translation_keys,
                outputs=outputs,
            )

    @staticmethod
    def _extract_pixel_sizes(
        images: list[SpatialImage],
    ) -> list[tuple[float, float]]:
        sizes: list[tuple[float, float]] = []
        for img in images:
            ps = img.space.source.pixel_size_xy
            if ps is None:
                raise ValueError(
                    f"SpatialPhysicalResize requires pixel_size_xy metadata "
                    f"but source '{img.space.source.source_path}' has none."
                )
            sizes.append((float(ps[0]), float(ps[1])))
        return sizes

    def _handles_list(self) -> bool:
        return True

    def _apply(self, detail: ImageDetail) -> tuple[ImageDetail, np.ndarray]:
        raise NotImplementedError(
            "SpatialPhysicalResize requires SpatialImage metadata; "
            "use forward() instead of _apply()."
        )

    def forward(self, image: SpatialImage | list[SpatialImage]) -> SpatialImage | list[SpatialImage]:
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        pixel_sizes = self._extract_pixel_sizes(images)
        self._inner.set_pixel_sizes(pixel_sizes)
        scales = self._inner._compute_scale_factors([im.detail for im in images])

        results: list[SpatialImage] = []
        for spatial_img, (sx, sy) in zip(images, scales):
            H_in, W_in = int(spatial_img.detail.H), int(spatial_img.detail.W)
            new_detail = self._inner._forward_single_at_scale(spatial_img.detail, sx, sy)
            H_out, W_out = int(new_detail.H), int(new_detail.W)
            matrix = mat_resize(W_in, H_in, W_out, H_out)
            results.append(_compose(spatial_img, new_detail, matrix))

        if isinstance(image, list):
            return results
        return results[0]
