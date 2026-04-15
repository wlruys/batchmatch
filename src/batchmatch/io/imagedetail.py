from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any, Optional

import torch

from batchmatch.base.tensordicts import ImageDetail, TranslationResults, WarpParams
from batchmatch.io.schema import IMAGEDETAIL_SCHEMA
from batchmatch.io.utils import (
    _DEFAULT,
    PathLike,
    assert_overwrite,
    ensure_parent_dir,
    pathify,
)

Tensor = torch.Tensor

__all__ = [
    "ImageDetailIO",
]


def _drop(d: dict[str, Any], key: str) -> None:
    if key in d:
        d.pop(key, None)


def _drop_nested(d: dict[str, Any], path: tuple[str, ...]) -> None:
    cur: Any = d
    for p in path[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


@dataclass(frozen=True)
class ImageDetailIO:
    include_grad: bool = False
    include_translation_surface: bool = False
    include_cache: bool = False
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    def _prune(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.include_cache:
            _drop(data, ImageDetail.Keys.CACHE.ROOT)
            for k in list(data.keys()):
                if isinstance(k, str) and k.startswith("cache"):
                    data.pop(k, None)

        if not self.include_grad:
            _drop(data, ImageDetail.Keys.GRAD.ROOT)

        if not self.include_translation_surface:
            _drop_nested(
                data,
                (ImageDetail.Keys.TRANSLATION.ROOT, TranslationResults.Keys.SURFACE),
            )

        for k in (
            WarpParams.Keys.GRID,
            WarpParams.Keys.M_FWD,
            WarpParams.Keys.M_INV,
            WarpParams.Keys.PIXEL_COORDS,
            WarpParams.Keys.COORD_XS,
            WarpParams.Keys.COORD_YS,
            WarpParams.Keys.SAMPLE,
        ):
            _drop_nested(data, (ImageDetail.Keys.WARP.ROOT, k))

        return data

    def save(
        self,
        detail: ImageDetail,
        path: PathLike,
        *,
        overwrite: bool = False,
    ) -> pathlib.Path:
        p = pathify(path)
        ensure_parent_dir(p)
        assert_overwrite(p, overwrite)

        data = detail.to_dict(convert_tensors=False)
        if not isinstance(data, dict) or "image" not in data:
            raise ValueError("ImageDetail.to_dict() must produce an 'image' entry.")

        data = self._prune(data)

        payload = IMAGEDETAIL_SCHEMA.envelope(
            {
                "batch_size": list(detail.batch_size),
                "data": data,
            }
        )
        torch.save(payload, str(p))
        return p

    def load(
        self,
        path: PathLike,
        *,
        device: Optional[torch.device] | object = _DEFAULT,
        dtype: Optional[torch.dtype] | object = _DEFAULT,
    ) -> ImageDetail:
        p = pathify(path)
        resolved_device = self._resolve_device(device)
        payload = torch.load(str(p), map_location=resolved_device, weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid ImageDetail payload type: {type(payload)!r}")

        IMAGEDETAIL_SCHEMA.validate(payload)
        raw_data = payload.get("data")
        batch_size = payload.get("batch_size")
        if not isinstance(raw_data, dict):
            raise TypeError("Invalid ImageDetail payload: missing 'data' dict.")
        if not (isinstance(batch_size, list) and batch_size and all(isinstance(x, int) for x in batch_size)):
            raise TypeError("Invalid ImageDetail payload: missing 'batch_size' list.")

        td = ImageDetail.from_dict(raw_data, batch_size=batch_size, device=resolved_device)

        resolved_dtype = self._resolve_dtype(dtype)
        if resolved_dtype is not None:
            td = td.apply(lambda t: t.to(dtype=resolved_dtype) if isinstance(t, torch.Tensor) else t)

        warp = td.get(ImageDetail.Keys.WARP.ROOT, default=None)
        if warp is not None and not isinstance(warp, WarpParams):
            try:
                warp_td = WarpParams(dict(warp.items()), batch_size=warp.batch_size)  # type: ignore[attr-defined]
            except Exception:
                warp_td = WarpParams(dict(warp.items()), batch_size=batch_size)  # type: ignore[arg-type]
            td.set(ImageDetail.Keys.WARP.ROOT, warp_td)

        translation = td.get(ImageDetail.Keys.TRANSLATION.ROOT, default=None)
        if translation is not None and not isinstance(translation, TranslationResults):
            try:
                tr_td = TranslationResults(
                    dict(translation.items()), batch_size=translation.batch_size  # type: ignore[attr-defined]
                )
            except Exception:
                tr_td = TranslationResults(dict(translation.items()), batch_size=batch_size)  # type: ignore[arg-type]
            td.set(ImageDetail.Keys.TRANSLATION.ROOT, tr_td)

        return td

    def exists(self, path: PathLike) -> bool:
        return pathify(path).exists()

    def _resolve_device(self, device: Optional[torch.device] | object) -> Optional[torch.device]:
        if device is _DEFAULT:
            return self.device
        return device

    def _resolve_dtype(self, dtype: Optional[torch.dtype] | object) -> Optional[torch.dtype]:
        if dtype is _DEFAULT:
            return self.dtype
        return dtype
