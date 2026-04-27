"""Register two images of cells via exhaustive warp search.

- Pre-scales images based on estimated cell size (``CellUnitResize``).
- Runs an exhaustive warp search on a common low-res search canvas.
- Saves a :class:`RegistrationTransform` manifest + optional previews and
  exports a full-resolution aligned TIFF with reference and registered
  moving image stored as separate prefixed channels.

Run:
    uv run examples/registration/register_cells.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from batchmatch import auto_device
from batchmatch.gradient import (
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
)
from batchmatch.io import ImageIO, ProductIO, SourceInfo, load_image
from batchmatch.io.space import SpatialImage
from batchmatch.process.cells import CellSizeCfg
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import CellUnitResize, TargetResize
from batchmatch.process.spatial_stages import (
    SpatialCenterPad,
    SpatialTargetResize,
    SpatialUnitResize,
)
from batchmatch.search import (
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    SearchParams,
)
from batchmatch.search.config import ScaleRange
from batchmatch.search.transform import RegistrationTransform
from batchmatch.translate.config import (
    GPCTranslationConfig,
    NCCTranslationConfig,
    NGFTranslationConfig,
)
from batchmatch.view.config import CheckerboardSpec, EdgeOverlaySpec, OverlaySpec
from batchmatch.view.composite import render_checkerboard, render_overlay
from batchmatch.view.display import show_comparison
from batchmatch.view.preview import render_registration_preview
from batchmatch.warp.resample import warp_to_reference


def _parse_image_from_selection(selection: int) -> tuple[Path, Path]:
    base_path = Path("cells") / f"selection_{selection}"
    moving_path = base_path / "fish_dapi.jpg"
    reference_path = base_path / f"xenium_region_2-sel{selection}-highres_wide.png"
    return moving_path, reference_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "register_cells")
    parser.add_argument("--search-dim", type=int, default=1024)
    parser.add_argument("--metric", choices=("ncc", "ngf", "gpc"), default="gpc")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-prefix", type=str, default="reg_cells")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-export", action="store_true", help="Skip full-res export.")
    parser.add_argument("--export-name", type=str, default="registered_fullres.tif")
    return parser.parse_args()


def _str_to_translation_config(metric: str):
    if metric == "ncc":
        return NCCTranslationConfig()
    if metric == "ngf":
        return NGFTranslationConfig()
    if metric == "gpc":
        return GPCTranslationConfig()
    raise ValueError(f"Unsupported metric: {metric}")


def _prefixed_channel_names(source: SourceInfo, count: int, prefix: str) -> tuple[str, ...]:
    names = list(source.channel_names)
    if len(names) < count:
        names.extend(f"channel_{i}" for i in range(len(names), count))
    return tuple(f"{prefix}:{name or f'channel_{i}'}" for i, name in enumerate(names[:count]))


def _stacked_source(
    *,
    path: Path,
    reference: SourceInfo,
    moving: SourceInfo,
    ref_channels: int,
    mov_channels: int,
    out_hw: tuple[int, int],
) -> SourceInfo:
    return SourceInfo(
        source_path=str(path),
        series_index=0,
        level_count=1,
        level_shapes=((int(out_hw[0]), int(out_hw[1])),),
        axes="CYX",
        dtype="float32",
        channel_names=(
            _prefixed_channel_names(reference, ref_channels, "reference")
            + _prefixed_channel_names(moving, mov_channels, "moving")
        ),
        pixel_size_xy=reference.pixel_size_xy,
        unit=reference.unit,
        origin_xy=reference.origin_xy,
        physical_extent_xy=reference.physical_extent_xy,
        format="ome-tiff",
    )


def export_stacked_registered_tiff(
    transform: RegistrationTransform,
    *,
    moving: SpatialImage,
    reference: SpatialImage,
    output_path: Path,
    overwrite: bool = True,
) -> Path:
    """Save reference and registered moving image as prefixed TIFF channels."""
    warped_moving = warp_to_reference(
        moving.detail,
        transform.matrix_ref_full_from_mov_full,
        out_hw=reference.shape_hw,
    )
    stacked = torch.cat([reference.detail.image, warped_moving.image], dim=1)
    source = _stacked_source(
        path=output_path,
        reference=reference.space.source,
        moving=moving.space.source,
        ref_channels=int(reference.detail.image.shape[1]),
        mov_channels=int(warped_moving.image.shape[1]),
        out_hw=reference.shape_hw,
    )
    return ImageIO().save(
        stacked,
        output_path,
        overwrite=overwrite,
        source=source,
        channel_names=source.channel_names,
    )


def main() -> None:
    args = _parse_args()
    moving_path, reference_path = _parse_image_from_selection(args.selection)

    moving = load_image(moving_path, grayscale=True)
    reference = load_image(reference_path, grayscale=True)

    cell_cfg = CellSizeCfg(
        radius_hint_px=14.0,
        use_log=True,
        blob_weight=0.4,
        max_dim=1024,
        morph_radius_px=3,
        peak_rel_threshold=0.3,
    )

    unit = SpatialUnitResize(inner=CellUnitResize(cell_cfg=cell_cfg))
    resize = SpatialTargetResize(inner=TargetResize(target_width=args.search_dim))
    pad = SpatialCenterPad(
        inner=CenterPad(
            scale=2,
            window_alpha=0.05,
            pad_to_pow2=False,
            outputs=["image", "box", "mask", "quad", "window"],
        )
    )
    prepare = unit >> resize >> pad

    # clone: inner stages may mutate their input details; keep originals
    # intact so we can still use them for preview/export.
    moving_search, reference_search = prepare([moving.clone(), reference.clone()])

    print(f"search image shape: {tuple(reference_search.detail.image.shape)}")
    print(f"device: {args.device}")

    search_params = SearchParams(
        scale_x=ScaleRange(min_scale=0.9, max_scale=1.5, step=0.01),
        scale_y=ScaleRange(min_scale=0.9, max_scale=1.5, step=0.01),
    )
    gradient = (
        CDGradientConfig(
            eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
            normalize=NormalizeConfig(norm="l2", threshold=1e-3),
        )
        if args.metric == "ngf"
        else CDGradientConfig()
    )
    config = ExhaustiveSearchConfig(
        translation=_str_to_translation_config(args.metric),
        batch_size=4,
        progress_enabled=True,
        gradient=gradient,
    )
    search = ExhaustiveWarpSearch(search_params, config)

    device = auto_device(args.device)
    reference_search = reference_search.to(device)
    moving_search = moving_search.to(device)
    search = search.to(device)

    start_t = time.perf_counter()
    result = search(reference_search.detail, moving_search.detail, top_k=1, progress=True)
    print(f"Search took {time.perf_counter() - start_t:.2f} s.")

    warp = result.warp
    print(
        f"warp: angle={warp.angle.item():.2f}, "
        f"scale_x={warp.scale_x.item():.4f}, scale_y={warp.scale_y.item():.4f}"
    )

    moving_search_cpu = moving_search.to("cpu")
    reference_search_cpu = reference_search.to("cpu")
    result_cpu = result.to("cpu")

    transform = RegistrationTransform.from_search(
        moving=moving_search_cpu,
        reference=reference_search_cpu,
        search_result=result_cpu,
    )

    preview = render_registration_preview(
        transform,
        moving.to("cpu"),
        reference.to("cpu"),
        crop_mode="union",
    )

    checkerboard_spec = CheckerboardSpec(
        tiles=(64, 64),
        edge_overlay=EdgeOverlaySpec(edge_source="mov", edge_threshold=0.3),
    )
    overlay_spec = OverlaySpec(alpha=0.5)
    checkerboard = render_checkerboard(preview.reference, preview.moving_warped, checkerboard_spec)
    overlay = render_overlay(preview.reference, preview.moving_warped, overlay_spec)

    if args.show:
        show_comparison(preview.reference, preview.moving_warped, mode="overlay", spec=overlay_spec)

    out_dir = args.output_dir / args.output_prefix / f"metric_{args.metric}_searchdim_{args.search_dim}"
    store = ProductIO(out_dir)
    manifest_path = store.save(
        transform,
        preview=preview,
        checkerboard=checkerboard,
        overlay=overlay,
        debug_details={"search_result": result_cpu},
        overwrite=True,
    )
    print(f"manifest: {manifest_path}")

    if not args.no_export:
        export_path = export_stacked_registered_tiff(
            transform,
            moving=moving.to("cpu"),
            reference=reference.to("cpu"),
            output_path=out_dir / args.export_name,
            overwrite=True,
        )
        print(f"exported: {export_path}")


if __name__ == "__main__":
    main()
