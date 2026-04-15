"""Canonical registration manifest + slim :class:`ProductIO`.

``registration.json`` v3 is the single source of truth.  The manifest
stores each source **once**, the three characteristic matrices of
:class:`RegistrationTransform`, and a compact search summary with no
duplication.

Manifest layout (v3)::

    {
      "schema": "batchmatch.registration",
      "version": 3,
      "sources": {
        "moving":    { <SourceInfo — OME-first> },
        "reference": { <SourceInfo — OME-first> }
      },
      "transform": {
        "matrices": {
          "ref_full_from_mov_full":    [[3×3]],
          "ref_phys_from_mov_phys":    [[3×3]] | null,
          "ref_search_from_mov_search": [[3×3]]
        },
        "search_spaces": {
          "moving":    { pyramid_level, region_yxhw, … },
          "reference": { … }
        },
        "search_summary": { … }
      },
      "artifacts": { … }
    }

Export is handled by :func:`batchmatch.io.export.export_registered`.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from batchmatch.base.tensordicts import ImageDetail
from batchmatch.io.imagedetail import ImageDetailIO
from batchmatch.io.images import ImageIO
from batchmatch.io.schema import REGISTRATION_SCHEMA
from batchmatch.io.space import SpatialImage
from batchmatch.io.utils import (
    PathLike,
    assert_overwrite,
    ensure_dir,
    pathify,
)

if TYPE_CHECKING:
    from batchmatch.search.transform import RegistrationTransform

Tensor = torch.Tensor

__all__ = ["ProductIO", "export_registered"]


def export_registered(*args: Any, **kwargs: Any):
    from batchmatch.io.export import export_registered as _export_registered

    return _export_registered(*args, **kwargs)


def _write_json(path: pathlib.Path, obj: Any, *, overwrite: bool) -> None:
    assert_overwrite(path, overwrite)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ProductIO:
    """Read/write a registration manifest and its optional artifacts."""

    root: PathLike
    manifest_name: str = "registration.json"
    detail_io: ImageDetailIO = field(default_factory=ImageDetailIO)
    preview_io: ImageIO = field(default_factory=ImageIO)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", pathify(self.root))
        if not self.manifest_name:
            raise ValueError("manifest_name must be non-empty.")

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.root / self.manifest_name

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        transform: "RegistrationTransform",
        *,
        preview: Any = None,
        checkerboard: Optional[Tensor] = None,
        overlay: Optional[Tensor] = None,
        debug_details: Optional[dict[str, ImageDetail | SpatialImage]] = None,
        overwrite: bool = False,
    ) -> pathlib.Path:
        ensure_dir(self.root)

        artifacts: dict[str, Any] = {}
        if preview is not None:
            ref_path = self.root / "preview_reference.png"
            mov_path = self.root / "preview_moving_warped.png"
            self.preview_io.save(preview.reference.image, ref_path, overwrite=overwrite)
            self.preview_io.save(preview.moving_warped.image, mov_path, overwrite=overwrite)
            artifacts["preview"] = {
                "reference": ref_path.name,
                "moving_warped": mov_path.name,
                "crop_box_xyxy": (
                    list(preview.crop_box_xyxy) if preview.crop_box_xyxy is not None else None
                ),
                "output_hw": list(preview.output_hw),
            }

        if checkerboard is not None:
            cb_path = self.root / "checkerboard.png"
            self.preview_io.save(checkerboard, cb_path, overwrite=overwrite)
            artifacts["checkerboard"] = cb_path.name

        if overlay is not None:
            ov_path = self.root / "overlay.png"
            self.preview_io.save(overlay, ov_path, overwrite=overwrite)
            artifacts["overlay"] = ov_path.name

        if debug_details:
            det_paths: dict[str, str] = {}
            for name, detail in debug_details.items():
                d = detail.detail if isinstance(detail, SpatialImage) else detail
                dpath = self.root / f"{name}.pt"
                self.detail_io.save(d, dpath, overwrite=overwrite)
                det_paths[name] = dpath.name
            artifacts["debug_details"] = det_paths

        manifest = REGISTRATION_SCHEMA.envelope(
            {
                "sources": {
                    "moving": transform.moving.source.to_dict(),
                    "reference": transform.reference.source.to_dict(),
                },
                "transform": transform.to_dict(),
                "artifacts": artifacts,
            }
        )

        _write_json(self.manifest_path, manifest, overwrite=overwrite)
        return self.manifest_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_manifest(self) -> dict[str, Any]:
        raw = _read_json(self.manifest_path)
        if not isinstance(raw, dict):
            raise TypeError("Invalid registration manifest.")
        REGISTRATION_SCHEMA.validate(raw)
        return raw

    def load_transform(self) -> "RegistrationTransform":
        from batchmatch.search.transform import RegistrationTransform

        return RegistrationTransform.from_manifest(self.load_manifest())

    def exists(self) -> bool:
        return self.manifest_path.exists()
