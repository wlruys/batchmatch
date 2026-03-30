"""Numpy-level spatial operations for TIFF I/O: crop and area downsample."""

from __future__ import annotations

import numpy as np

__all__ = ["crop_region", "area_downsample", "validate_region"]


def validate_region(
    region: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Validate and clamp a region to image bounds.

    Returns the (possibly clamped) region.  Raises ``ValueError`` if the
    region is entirely outside the image.
    """
    y0, x0, h, w = region
    if y0 < 0 or x0 < 0:
        raise ValueError(f"Region origin must be non-negative, got ({y0}, {x0})")
    if y0 >= height or x0 >= width:
        raise ValueError(
            f"Region origin ({y0}, {x0}) is outside image bounds "
            f"({height}, {width})"
        )
    h = min(h, height - y0)
    w = min(w, width - x0)
    if h <= 0 or w <= 0:
        raise ValueError(
            f"Region (y0={y0}, x0={x0}, h={h}, w={w}) results in "
            f"zero-size crop for image ({height}, {width})"
        )
    return (y0, x0, h, w)


def crop_region(
    arr: np.ndarray,
    axes: str,
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop to (y_start, x_start, height, width)."""
    y0, x0, h, w = region
    y_ax = axes.find("Y")
    x_ax = axes.find("X")
    slices = [slice(None)] * arr.ndim
    slices[y_ax] = slice(y0, y0 + h)
    slices[x_ax] = slice(x0, x0 + w)
    return arr[tuple(slices)]


def area_downsample(arr: np.ndarray, axes: str, factor: int) -> np.ndarray:
    """Anti-aliased area-average downsampling along Y and X.

    Truncates to a size evenly divisible by *factor*, then reshapes and
    averages over the spatial blocks.
    """
    y_ax = axes.find("Y")
    x_ax = axes.find("X")
    h, w = arr.shape[y_ax], arr.shape[x_ax]
    new_h, new_w = h // factor, w // factor

    # Trim to exact multiple
    slices = [slice(None)] * arr.ndim
    slices[y_ax] = slice(0, new_h * factor)
    slices[x_ax] = slice(0, new_w * factor)
    arr = arr[tuple(slices)]

    # Reshape Y dim: (..., new_h, factor, ...) and average
    shape = list(arr.shape)
    shape[y_ax] = new_h
    shape.insert(y_ax + 1, factor)
    x_ax_adj = x_ax + 1 if x_ax > y_ax else x_ax
    shape[x_ax_adj] = new_w
    shape.insert(x_ax_adj + 1, factor)

    work = arr.reshape(shape).astype(np.float32, copy=False)
    ax1 = x_ax_adj + 1
    ax2 = y_ax + 1
    if ax1 > ax2:
        work = work.mean(axis=ax1).mean(axis=ax2)
    else:
        work = work.mean(axis=ax2).mean(axis=ax1)
    return work
