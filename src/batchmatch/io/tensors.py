from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Optional

import numpy as np
import torch

from batchmatch.io.utils import (
    _DEFAULT,
    PathLike,
    assert_overwrite,
    ensure_parent_dir,
    pathify,
)

Tensor = torch.Tensor

__all__ = [
    "TensorIO",
]


@dataclass(frozen=True)
class TensorIO:
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    def save(
        self,
        array: Tensor,
        path: PathLike,
        *,
        overwrite: bool = False,
    ) -> pathlib.Path:
        p = pathify(path)
        ensure_parent_dir(p)
        assert_overwrite(p, overwrite)

        t = torch.as_tensor(array)
        suffix = p.suffix.lower()

        if suffix in {".pt", ".pth"}:
            torch.save(t, str(p))
        elif suffix == ".npy":
            np.save(str(p), t.detach().cpu().numpy())
        elif suffix == ".npz":
            np.savez(str(p), arr=t.detach().cpu().numpy())
        else:
            raise ValueError(f"Unsupported extension: {suffix!r} (use .pt/.pth/.npy/.npz).")

        return p

    def load(
        self,
        path: PathLike,
        *,
        device: Optional[torch.device] | object = _DEFAULT,
        dtype: Optional[torch.dtype] | object = _DEFAULT,
    ) -> Tensor:
        device = self.device if device is _DEFAULT else device
        dtype = self.dtype if dtype is _DEFAULT else dtype

        p = pathify(path)
        suffix = p.suffix.lower()

        if suffix in {".pt", ".pth"}:
            t = torch.load(str(p), map_location=device, weights_only=True)
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Expected Tensor in {suffix} file; got {type(t).__name__}.")
        elif suffix == ".npy":
            t = torch.from_numpy(np.load(str(p), allow_pickle=False))
        elif suffix == ".npz":
            with np.load(str(p), allow_pickle=False) as z:
                arr = z["arr"] if "arr" in z.files else z[z.files[0]]
                t = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported extension: {suffix!r} (use .pt/.pth/.npy/.npz).")

        if device is not None and t.device != torch.device(device):
            t = t.to(device=device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype=dtype)

        return t

    def exists(self, path: PathLike) -> bool:
        return pathify(path).exists()
