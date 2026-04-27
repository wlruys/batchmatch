"""Conversion utilities: numpy arrays to BCHW float32 tensors."""

from __future__ import annotations

import numpy as np
import torch

Tensor = torch.Tensor

__all__ = ["to_bchw_float"]

_CHUNKED_THRESHOLD_BYTES = 1 * 1024**3  # 1 GB


def to_bchw_float(arr: np.ndarray) -> Tensor:
    """Convert a numpy image array to a BCHW float32 tensor in [0, 1].

    For large integer arrays this uses a chunked conversion strategy to
    avoid holding both the full source and full destination arrays in
    memory simultaneously.
    """
    is_int = np.issubdtype(arr.dtype, np.integer)
    max_val = float(np.iinfo(arr.dtype).max) if is_int else 1.0

    if arr.nbytes > _CHUNKED_THRESHOLD_BYTES and is_int:
        t = _chunked_int_to_float(arr, max_val)
    elif is_int:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        arr /= max_val
        t = torch.from_numpy(arr)
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)

    while t.ndim < 4:
        t = t.unsqueeze(0)
    return t


def _chunked_int_to_float(arr: np.ndarray, max_val: float) -> Tensor:
    """Convert a large integer array to float32 tensor in row-chunks.

    Allocates the output tensor up front and fills it in slices so that
    at most one chunk (~256 MB) of the source array is duplicated in
    float32 at any time.
    """
    shape = arr.shape
    out = torch.empty(shape, dtype=torch.float32)
    chunk_rows = max(1, (256 * 1024**2) // (arr[0:1].nbytes))
    h = shape[-2]

    for start in range(0, h, chunk_rows):
        end = min(start + chunk_rows, h)
        src = arr[..., start:end, :]
        chunk = np.ascontiguousarray(src, dtype=np.float32)
        chunk /= max_val
        out[..., start:end, :] = torch.from_numpy(chunk)
        del chunk

    return out
