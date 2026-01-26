from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from batchmatch.base.detail import build_image_td
from batchmatch.base.tensordicts import ImageDetail, TranslationResults
from batchmatch.io.imagedetail import ImageDetailIO
from batchmatch.io.images import ImageIO
from batchmatch.io.schema import PRODUCT_SCHEMA
from batchmatch.io.utils import (
    PathLike,
    assert_overwrite,
    ensure_dir,
    pathify,
)
from batchmatch.view.config import CheckerboardSpec, OverlaySpec
from batchmatch.view.composite import render_checkerboard, render_overlay

Tensor = torch.Tensor

__all__ = [
    "RegistrationProduct",
    "ProductIO",
]


def _write_json(path: pathlib.Path, obj: Any, *, overwrite: bool) -> None:
    assert_overwrite(path, overwrite)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_detail(x: ImageDetail | Tensor) -> ImageDetail:
    if isinstance(x, ImageDetail):
        return x
    return build_image_td(torch.as_tensor(x))


def _as_xyxy_box(box: torch.Tensor) -> tuple[float, float, float, float]:
    if box.ndim == 3:
        b = box[0, 0]
    elif box.ndim == 2:
        b = box[0]
    elif box.ndim == 1 and box.numel() == 4:
        b = box
    else:
        raise ValueError(f"Unexpected box shape: {tuple(box.shape)}")
    x1, y1, x2, y2 = (float(v.item()) for v in b)
    return (x1, y1, x2, y2)


def _tensor_item(v: torch.Tensor, idx: int = 0) -> float:
    if v.ndim == 0:
        return float(v.item())
    return float(v[idx].item())


@dataclass(frozen=True)
class RegistrationProduct:
    moving: ImageDetail
    reference: ImageDetail
    registered: ImageDetail

    @classmethod
    def coerce(
        cls,
        *,
        moving: ImageDetail | Tensor,
        reference: ImageDetail | Tensor,
        registered: ImageDetail | Tensor,
    ) -> "RegistrationProduct":
        return cls(
            moving=_coerce_detail(moving),
            reference=_coerce_detail(reference),
            registered=_coerce_detail(registered),
        )


@dataclass(frozen=True)
class ProductIO:
    root: PathLike
    manifest_name: str = "product.json"
    detail_io: ImageDetailIO = field(default_factory=ImageDetailIO)
    preview_io: ImageIO = field(default_factory=ImageIO)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", pathify(self.root))
        if not self.manifest_name:
            raise ValueError("manifest_name must be non-empty.")

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.root / self.manifest_name

    def _detail_path(self, name: str) -> pathlib.Path:
        return self.root / f"{name}.pt"

    def _preview_path(self, name: str) -> pathlib.Path:
        return self.root / f"{name}.png"

    def save(
        self,
        *,
        moving: ImageDetail | Tensor,
        reference: ImageDetail | Tensor,
        registered: ImageDetail | Tensor,
        search_result: ImageDetail | None = None,
        search_idx: int = 0,
        overwrite: bool = False,
        save_previews: bool = True,
        save_checkerboard: bool = False,
        save_overlay: bool = False,
        checkerboard_spec: Optional[CheckerboardSpec] = None,
        overlay_spec: Optional[OverlaySpec] = None,
        include_grad: Optional[bool] = None,
        include_translation_surface: Optional[bool] = None,
        include_cache: Optional[bool] = None,
    ) -> pathlib.Path:
        ensure_dir(self.root)

        product = RegistrationProduct.coerce(
            moving=moving, reference=reference, registered=registered
        )

        io = self._detail_io(
            include_grad=include_grad,
            include_translation_surface=include_translation_surface,
            include_cache=include_cache,
        )

        paths: dict[str, Any] = {}
        for name, detail in (
            ("moving", product.moving),
            ("reference", product.reference),
            ("registered", product.registered),
        ):
            dpath = self._detail_path(name)
            io.save(detail, dpath, overwrite=overwrite)
            paths[name] = {"detail": dpath.name}

            if save_previews:
                ppath = self._preview_path(name)
                self.preview_io.save(detail.image, ppath, overwrite=overwrite)
                paths[name]["preview"] = ppath.name

        transform: dict[str, Any] | None = None
        if search_result is not None:
            rpath = self._detail_path("search_result")
            io.save(search_result, rpath, overwrite=overwrite)
            paths["search_result"] = {"detail": rpath.name}

            ref_box = product.reference.get(ImageDetail.Keys.DOMAIN.BOX, default=None)
            ref_box_xyxy = _as_xyxy_box(ref_box) if ref_box is not None else None

            warp = search_result.warp
            tr = search_result.translation_results
            if warp is not None and tr is not None:
                search_h = tr.get(TranslationResults.Keys.SEARCH_H, default=None)
                search_w = tr.get(TranslationResults.Keys.SEARCH_W, default=None)
                score = tr.get(TranslationResults.Keys.SCORE, default=None)
                transform = {
                    "idx": int(search_idx),
                    "reference_box_xyxy": list(ref_box_xyxy) if ref_box_xyxy is not None else None,
                    "raw": {
                        "warp": {
                            "angle": _tensor_item(warp.angle, search_idx),
                            "scale_x": _tensor_item(warp.scale_x, search_idx),
                            "scale_y": _tensor_item(warp.scale_y, search_idx),
                            "shear_x": _tensor_item(warp.shear_x, search_idx),
                            "shear_y": _tensor_item(warp.shear_y, search_idx),
                            "tx": _tensor_item(warp.tx, search_idx),
                            "ty": _tensor_item(warp.ty, search_idx),
                        },
                        "translation": {
                            "tx": _tensor_item(tr.tx, search_idx),
                            "ty": _tensor_item(tr.ty, search_idx),
                            "score": _tensor_item(score, search_idx) if score is not None else None,
                            "search_h": _tensor_item(search_h, search_idx) if search_h is not None else None,
                            "search_w": _tensor_item(search_w, search_idx) if search_w is not None else None,
                        },
                    },
                }

        # Save comparison previews (checkerboard and overlay)
        if save_checkerboard:
            cb_spec = checkerboard_spec if checkerboard_spec is not None else CheckerboardSpec()
            cb_image = render_checkerboard(product.reference, product.registered, cb_spec)
            cb_path = self.root / "checkerboard.png"
            self.preview_io.save(cb_image, cb_path, overwrite=overwrite)
            paths["checkerboard"] = {"preview": cb_path.name}

        if save_overlay:
            ov_spec = overlay_spec if overlay_spec is not None else OverlaySpec()
            ov_image = render_overlay(product.reference, product.registered, ov_spec)
            ov_path = self.root / "overlay.png"
            self.preview_io.save(ov_image, ov_path, overwrite=overwrite)
            paths["overlay"] = {"preview": ov_path.name}

        manifest = PRODUCT_SCHEMA.with_metadata(
            {
                "paths": paths,
                "transform": transform,
            }
        )

        _write_json(self.manifest_path, manifest, overwrite=overwrite)
        return self.manifest_path

    def load(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> RegistrationProduct:
        io = self.detail_io
        moving = io.load(self._detail_path("moving"), device=device, dtype=dtype)
        reference = io.load(self._detail_path("reference"), device=device, dtype=dtype)
        registered = io.load(self._detail_path("registered"), device=device, dtype=dtype)
        return RegistrationProduct(moving=moving, reference=reference, registered=registered)

    def load_search_result(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ImageDetail:
        return self.detail_io.load(self._detail_path("search_result"), device=device, dtype=dtype)

    def load_manifest(self) -> dict[str, Any]:
        raw = _read_json(self.manifest_path)
        if not isinstance(raw, dict):
            raise TypeError("Invalid product manifest.")
        PRODUCT_SCHEMA.validate(raw)
        return raw

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def _detail_io(
        self,
        *,
        include_grad: Optional[bool],
        include_translation_surface: Optional[bool],
        include_cache: Optional[bool],
    ) -> ImageDetailIO:
        if include_grad is None and include_translation_surface is None and include_cache is None:
            return self.detail_io
        return ImageDetailIO(
            include_grad=self.detail_io.include_grad if include_grad is None else include_grad,
            include_translation_surface=(
                self.detail_io.include_translation_surface
                if include_translation_surface is None
                else include_translation_surface
            ),
            include_cache=self.detail_io.include_cache if include_cache is None else include_cache,
            device=self.detail_io.device,
            dtype=self.detail_io.dtype,
        )
