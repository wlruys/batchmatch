from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "mat_identity",
    "mat_crop",
    "mat_downsample_int",
    "mat_resize",
    "mat_pad",
    "mat_compose",
    "mat_inv",
    "mat_from_search_result",
    "apply_to_points",
    "pixel_to_phys",
    "phys_to_pixel",
]


def _as_f64(mat: np.ndarray) -> np.ndarray:
    out = np.asarray(mat, dtype=np.float64)
    if out.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {out.shape}.")
    return out


def mat_identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def mat_crop(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Return M_dst_from_src for cropping to region (x, y, w, h)."""
    _ = (w, h)
    return np.array(
        [
            [1.0, 0.0, -float(x)],
            [0.0, 1.0, -float(y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mat_downsample_int(factor: int) -> np.ndarray:
    """Return M_dst_from_src for integer downsample (area-style)."""
    if factor <= 0:
        raise ValueError("downsample factor must be positive.")
    f = float(factor)
    offset = (f - 1.0) / (2.0 * f)
    return np.array(
        [
            [1.0 / f, 0.0, -offset],
            [0.0, 1.0 / f, -offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mat_resize(
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Return M_dst_from_src matching torch interpolate (align_corners=False).

    With ``align_corners=False`` the half-pixel mapping is:
    ``x_out = x_in * (out_w / in_w) + (out_w / in_w - 1) / 2``,
    giving ``sx = out_w / in_w``, ``tx = (sx - 1) / 2``.
    """
    if in_w <= 0 or in_h <= 0 or out_w <= 0 or out_h <= 0:
        raise ValueError("resize sizes must be positive.")
    sx = float(out_w) / float(in_w)
    sy = float(out_h) / float(in_h)
    tx = (sx - 1.0) * 0.5
    ty = (sy - 1.0) * 0.5
    return np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mat_pad(left: float, top: float, right: float, bottom: float) -> np.ndarray:
    _ = (right, bottom)
    return np.array(
        [
            [1.0, 0.0, float(left)],
            [0.0, 1.0, float(top)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mat_compose(mats: Iterable[np.ndarray]) -> np.ndarray:
    """Compose matrices in order: M = M_n @ ... @ M_1."""
    out = mat_identity()
    for mat in mats:
        out = _as_f64(mat) @ out
    return out


def mat_inv(mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_as_f64(mat)).astype(np.float64, copy=False)


def mat_from_search_result(
    warp_matrix_sample_from_ref: np.ndarray,
    tx: float,
    ty: float,
) -> np.ndarray:
    """Compose ``M_ref_search_from_mov_search`` from a registration search result.

    ``warp_matrix_sample_from_ref`` is the sampling matrix used by the
    search warp stage for the selected candidate: with
    ``PrepareWarpStage(inverse=True)`` it maps reference-search output
    pixels to moving-search sample pixels.  ``tx``/``ty`` are the pixel
    translation in search-space coordinates discovered by the translation
    stage and applied after the warp as a shift with ``negate=True`` in
    the legacy product path.

    The result is the direct point-transform matrix from moving-search
    pixels to reference-search pixels.  The legacy product path first
    warps moving via `grid_sample(moving, M_fwd)` (sampling output→input)
    then shifts by ``(-tx, -ty)``.  Inverting the sampling chain
    ``M_fwd @ T(-tx, -ty)``:

        p_ref = T(tx, ty) @ inv(M_fwd) @ p_mov

    so the shift is applied in the destination (reference) frame.
    """
    shift = np.array(
        [
            [1.0, 0.0, float(tx)],
            [0.0, 1.0, float(ty)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return shift @ mat_inv(_as_f64(warp_matrix_sample_from_ref))


def apply_to_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply 3x3 matrix to (N,2) or (B,N,2) points."""
    mat = _as_f64(mat)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] == 2:
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.concatenate([pts, ones], axis=1)
        warped = pts_h @ mat.T
        return warped[:, :2]
    if pts.ndim == 3 and pts.shape[2] == 2:
        ones = np.ones((pts.shape[0], pts.shape[1], 1), dtype=np.float64)
        pts_h = np.concatenate([pts, ones], axis=2)
        warped = pts_h @ mat.T
        return warped[:, :, :2]
    raise ValueError(f"Expected points shaped (N,2) or (B,N,2), got {pts.shape}.")


def pixel_to_phys(
    pixel_size_xy: Sequence[float],
    origin_xy: Sequence[float],
) -> np.ndarray:
    sx, sy = float(pixel_size_xy[0]), float(pixel_size_xy[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    return np.array(
        [
            [sx, 0.0, ox],
            [0.0, sy, oy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def phys_to_pixel(
    pixel_size_xy: Sequence[float],
    origin_xy: Sequence[float],
) -> np.ndarray:
    return mat_inv(pixel_to_phys(pixel_size_xy, origin_xy))
