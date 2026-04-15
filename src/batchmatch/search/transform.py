"""Canonical registration output.

:class:`RegistrationTransform` is the single object flowing out of
registration.  It carries the search-time :class:`ImageSpace` for both
sides, the direct search-space point transform, and the full-resolution
matrix used by preview/export.

Coordinate frames and matrices
------------------------------

Every 3×3 matrix in this module maps **homogeneous pixel coordinates**
``(x, y, 1)`` from one frame to another.  The naming convention
``M_<dst>_from_<src>`` makes each matrix self-documenting:

* ``M_ref_search_from_mov_search`` — point-transform in the search canvas.
* ``M_ref_full_from_mov_full``     — point-transform in full-resolution
  (source-file) pixel coordinates.
* ``M_ref_phys_from_mov_phys``     — point-transform in physical units
  (if calibration metadata is present on both sides).

Full-resolution lifting
-----------------------

Each :class:`SpatialImage` that enters the search already carries an exact
``matrix_image_from_source`` that encodes every spatial operation applied
since loading (pyramid-level scaling, crop, resize, center-padding).
Composing through that matrix is therefore pixel-exact::

    M_ref_full = inv(M_ref_img_from_src) @ M_search @ M_mov_img_from_src

This is simpler and more accurate than independently re-deriving a
full-resolution warp canvas with scaled centres, which is sensitive to
rounding between canvas and content coordinates.

Search-space derivation
-----------------------

The search warp stage produces ``M_fwd`` — the *sampling matrix* fed to
``grid_sample(moving, M_fwd)`` with ``align_corners=False``.  It maps
output (reference-canvas) pixels to input (moving-canvas) sample
positions.  The translation stage then shifts by ``(-tx, -ty)`` (the
``negate=True`` convention).

Inverting the total sampling chain ``M_fwd @ T(-tx, -ty)``::

    p_ref = T(tx, ty) @ inv(M_fwd) @ p_mov
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from batchmatch.base.tensordicts import ImageDetail, TranslationResults, WarpParams
from batchmatch.helpers.affine import mat_inv, phys_to_pixel, pixel_to_phys
from batchmatch.io.space import ImageSpace, SpatialImage
from batchmatch.warp.specs import compute_warp_matrices

__all__ = ["RegistrationTransform"]


def _scalar(t: torch.Tensor, idx: int) -> float:
    if t.ndim == 0:
        return float(t.item())
    return float(t[idx].item())


def _shift_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, float(tx)],
            [0.0, 1.0, float(ty)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )




@dataclass(frozen=True)
class RegistrationTransform:
    """Result object produced by :meth:`from_search`.

    Attributes
    ----------
    moving, reference:
        The :class:`ImageSpace` snapshots taken at search-canvas resolution.
        Their ``matrix_image_from_source`` encodes every preprocessing step
        (unit-resize → target-resize → center-pad) so that the
        full-resolution lift is exact.
    matrix_ref_search_from_mov_search:
        3×3 point-transform in the **search canvas** pixel frame.
        Maps ``(x_mov, y_mov, 1)`` in the padded moving search image to
        the corresponding point in the padded reference search image.
    matrix_ref_full_from_mov_full:
        3×3 point-transform in **source file** pixel coordinates.
        Maps a full-resolution moving pixel to the corresponding
        full-resolution reference pixel.  Used by
        :func:`~batchmatch.warp.resample.warp_to_reference`,
        :func:`~batchmatch.io.export.export_registered`, and
        :func:`~batchmatch.view.preview.render_registration_preview`.
    matrix_ref_phys_from_mov_phys:
        Optional 3×3 transform in physical (calibrated) units.
        ``None`` when either source lacks ``pixel_size_xy`` / ``origin_xy``.
    search_summary:
        Human-readable dict of the raw search parameters and scores.
    """

    moving: ImageSpace
    reference: ImageSpace
    matrix_ref_search_from_mov_search: np.ndarray
    matrix_ref_full_from_mov_full: np.ndarray
    matrix_ref_phys_from_mov_phys: np.ndarray | None
    search_summary: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "matrix_ref_search_from_mov_search",
            "matrix_ref_full_from_mov_full",
        ):
            m = np.asarray(getattr(self, name), dtype=np.float64)
            if m.shape != (3, 3):
                raise ValueError(f"{name} must be 3x3, got {m.shape}.")
            object.__setattr__(self, name, m)
        if self.matrix_ref_phys_from_mov_phys is not None:
            m = np.asarray(self.matrix_ref_phys_from_mov_phys, dtype=np.float64)
            if m.shape != (3, 3):
                raise ValueError(
                    f"matrix_ref_phys_from_mov_phys must be 3x3, got {m.shape}."
                )
            object.__setattr__(self, "matrix_ref_phys_from_mov_phys", m)

    @classmethod
    def from_search(
        cls,
        *,
        moving: SpatialImage,
        reference: SpatialImage,
        search_result: ImageDetail,
        candidate_idx: int = 0,
    ) -> "RegistrationTransform":
        """Build a :class:`RegistrationTransform` from a search result.

        Parameters
        ----------
        moving, reference:
            The preprocessed :class:`SpatialImage` pair *at search-canvas
            resolution* (i.e. after unit-resize, target-resize, and
            center-padding).  Their ``space.matrix_image_from_source``
            must reflect every spatial operation applied since loading.
        search_result:
            An :class:`ImageDetail` carrying ``.warp`` (the affine warp
            parameters found by the exhaustive search) and
            ``.translation_results`` (the subpixel translation refinement,
            including ``SEARCH_H``/``SEARCH_W``).
        candidate_idx:
            Batch index into the search result (default ``0``).

        Transform derivation
        --------------------
        1. **Search-space transform** — The search warp stage evaluates
           ``grid_sample(moving, M_fwd)`` (``M_fwd``: output→input
           sampling matrix, ``align_corners=False``), then the
           translation stage shifts by ``(-tx, -ty)`` (``negate=True``).
           Inverting the sampling chain ``M_fwd @ T(-tx, -ty)``::

               M_search = T(tx, ty) @ inv(M_fwd)

        2. **Full-resolution transform** — Each ``SpatialImage``'s
           ``space.matrix_image_from_source`` (call it ``M_mov`` and
           ``M_ref``) encodes the full chain from source-file pixels to
           search-canvas pixels.  Composing through both matrices::

               M_full = inv(M_ref) @ M_search @ M_mov

           This is exact because ``M_mov``/``M_ref`` already account for
           per-image resize ratios, padding offsets, and any other
           spatial stages.

        3. **Physical transform** (optional) — When both sources carry
           ``pixel_size_xy`` and ``origin_xy`` calibration metadata::

               M_phys = phys_from_ref_pix @ M_full @ mov_pix_from_phys
        """
        warp = search_result.warp
        tr = search_result.translation_results
        if warp is None:
            raise ValueError("search_result.warp is required.")
        if tr is None:
            raise ValueError("search_result.translation_results is required.")
        search_h = tr.get(TranslationResults.Keys.SEARCH_H, default=None)
        search_w = tr.get(TranslationResults.Keys.SEARCH_W, default=None)
        if search_h is None or search_w is None:
            raise ValueError(
                "search_result.translation_results missing SEARCH_H/SEARCH_W."
            )

        search_canvas_h, search_canvas_w = moving.shape_hw
        M_fwd_search, _ = compute_warp_matrices(
            warp,
            search_canvas_h,
            search_canvas_w,
            inplace=False,
        )
        warp_fwd_search = (
            M_fwd_search[candidate_idx].detach().cpu().numpy().astype(np.float64)
        )
        tx = _scalar(tr.tx, candidate_idx)
        ty = _scalar(tr.ty, candidate_idx)
        # Search-space point transform from moving to reference.
        # The legacy product path warps moving via grid_sample(M_fwd)
        # (sampling matrix: output→input) then shifts by (-tx, -ty).
        # The total sampling matrix is M_fwd @ T(-tx,-ty).  Inverting:
        #   ref_point = T(tx, ty) @ inv(M_fwd) @ mov_point
        M_ref_search_from_mov_search = _shift_matrix(tx, ty) @ mat_inv(
            warp_fwd_search
        )

        # ---- Full-resolution transform via matrix_image_from_source ----
        # Each SpatialImage's space.matrix_image_from_source encodes the
        # full chain (pyramid → crop → downsample → spatial stages like
        # resize and center-padding) from source (full-res) pixels to the
        # search canvas pixels.  The correct full-res transform is:
        #
        #   p_ref_full = inv(M_ref_src2canvas) @ M_search @ M_mov_src2canvas @ p_mov_full
        #
        # This naturally handles different padding, scaling, and canvas
        # geometry without needing to rescale warp centres or invent a
        # separate full-resolution canvas.
        M_mov_src2canvas = moving.space.matrix_image_from_source.astype(np.float64)
        M_ref_src2canvas = reference.space.matrix_image_from_source.astype(np.float64)
        M_ref_full_from_mov_full = (
            mat_inv(M_ref_src2canvas)
            @ M_ref_search_from_mov_search
            @ M_mov_src2canvas
        )

        mov_src = moving.space.source
        ref_src = reference.space.source
        M_ref_phys_from_mov_phys: np.ndarray | None = None
        if mov_src.has_calibration() and ref_src.has_calibration():
            M_mov_pix_from_mov_phys = phys_to_pixel(
                mov_src.pixel_size_xy, mov_src.origin_xy  # type: ignore[arg-type]
            )
            M_ref_phys_from_ref_pix = pixel_to_phys(
                ref_src.pixel_size_xy, ref_src.origin_xy  # type: ignore[arg-type]
            )
            M_ref_phys_from_mov_phys = (
                M_ref_phys_from_ref_pix
                @ M_ref_full_from_mov_full
                @ M_mov_pix_from_mov_phys
            )

        score = tr.get(TranslationResults.Keys.SCORE, default=None)
        search_summary = {
            "candidate_idx": int(candidate_idx),
            "score": _scalar(score, candidate_idx) if score is not None else None,
            "warp": {
                "angle": _scalar(warp[WarpParams.Keys.ANGLE], candidate_idx),
                "scale_x": _scalar(warp[WarpParams.Keys.SCALE_X], candidate_idx),
                "scale_y": _scalar(warp[WarpParams.Keys.SCALE_Y], candidate_idx),
                "shear_x": _scalar(warp[WarpParams.Keys.SHEAR_X], candidate_idx),
                "shear_y": _scalar(warp[WarpParams.Keys.SHEAR_Y], candidate_idx),
                "tx": _scalar(warp[WarpParams.Keys.TX], candidate_idx),
                "ty": _scalar(warp[WarpParams.Keys.TY], candidate_idx),
            },
            "translation": {
                "tx": tx,
                "ty": ty,
                "search_h": int(round(_scalar(search_h, candidate_idx))),
                "search_w": int(round(_scalar(search_w, candidate_idx))),
                "canvas_h": int(search_canvas_h),
                "canvas_w": int(search_canvas_w),
            },
        }

        return cls(
            moving=moving.space,
            reference=reference.space,
            matrix_ref_search_from_mov_search=M_ref_search_from_mov_search,
            matrix_ref_full_from_mov_full=M_ref_full_from_mov_full,
            matrix_ref_phys_from_mov_phys=M_ref_phys_from_mov_phys,
            search_summary=search_summary,
        )

    @property
    def matrix_mov_full_from_ref_full(self) -> np.ndarray:
        return mat_inv(self.matrix_ref_full_from_mov_full)

    def to_dict(self) -> dict[str, Any]:
        """Compact serialization — matrices, search spaces, summary.

        Source metadata is **not** embedded here; it is stored once at
        the manifest top level by :class:`ProductIO`.
        """
        return {
            "matrices": {
                "ref_full_from_mov_full": self.matrix_ref_full_from_mov_full.tolist(),
                "ref_phys_from_mov_phys": (
                    self.matrix_ref_phys_from_mov_phys.tolist()
                    if self.matrix_ref_phys_from_mov_phys is not None
                    else None
                ),
                "ref_search_from_mov_search": self.matrix_ref_search_from_mov_search.tolist(),
            },
            "search_spaces": {
                "moving": self.moving.to_dict(),
                "reference": self.reference.to_dict(),
            },
            "search_summary": dict(self.search_summary),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        moving_source: "SourceInfo",
        reference_source: "SourceInfo",
    ) -> "RegistrationTransform":
        """Reconstruct from the compact dict + external source metadata."""
        mats = data["matrices"]
        spaces = data["search_spaces"]
        moving = ImageSpace.from_dict(spaces["moving"], source=moving_source)
        reference = ImageSpace.from_dict(spaces["reference"], source=reference_source)
        phys = mats.get("ref_phys_from_mov_phys")
        return cls(
            moving=moving,
            reference=reference,
            matrix_ref_search_from_mov_search=np.asarray(
                mats["ref_search_from_mov_search"], dtype=np.float64
            ),
            matrix_ref_full_from_mov_full=np.asarray(
                mats["ref_full_from_mov_full"], dtype=np.float64
            ),
            matrix_ref_phys_from_mov_phys=(
                np.asarray(phys, dtype=np.float64) if phys is not None else None
            ),
            search_summary=dict(data.get("search_summary", {})),
        )

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "RegistrationTransform":
        """Reconstruct from a full registration manifest dict."""
        from batchmatch.io.space import SourceInfo

        sources = manifest["sources"]
        mov_source = SourceInfo.from_dict(sources["moving"])
        ref_source = SourceInfo.from_dict(sources["reference"])
        return cls.from_dict(
            manifest["transform"],
            moving_source=mov_source,
            reference_source=ref_source,
        )
