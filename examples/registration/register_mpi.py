
"""Distributed MPI exhaustive warp search for cell registration.

Uses the SpatialImage → RegistrationTransform → ProductIO pipeline with
MPI-distributed exhaustive search.

Run (single process):
    uv run examples/registration/register_mpi.py

Run (4 MPI ranks):
    mpirun -n 4 uv run examples/registration/register_mpi.py
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
from batchmatch.io import ProductIO, load_image
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
    AngleRange,
    SearchParams,
    MPIExhaustiveSearchConfig,
    MPIExhaustiveWarpSearch,
    is_cuda_aware_mpi,
    InconsistentInputsError,
    MPISearchError,
)
from batchmatch.search.config import ScaleRange
from batchmatch.search.transform import RegistrationTransform
from batchmatch.translate.config import (
    GNGFTranslationConfig,
    GPCTranslationConfig,
    NCCTranslationConfig,
)
from batchmatch.view.config import CheckerboardSpec, EdgeOverlaySpec, OverlaySpec
from batchmatch.view.composite import render_checkerboard, render_overlay
from batchmatch.view.display import show_comparison
from batchmatch.view.preview import render_registration_preview


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed MPI exhaustive warp search for cell registration"
    )
    parser.add_argument(
        "--selection",
        type=int,
        default=7,
        help="Cell selection number",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "register_mpi",
        help="Directory to write output figures",
    )
    parser.add_argument(
        "--search-dim",
        type=int,
        default=1024,
        help="Dimension to which images are resized for search",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="gpc",
        choices=["ncc", "ngf", "gpc"],
        help="Similarity metric for registration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for processing (auto|cpu|cuda|mps)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="reg_cells_mpi",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Visualize the registration results (open matplotlib, rank 0 only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for search",
    )
    parser.add_argument(
        "--gpu-aware",
        action="store_true",
        help="Use GPU-aware MPI (requires CUDA-aware MPI implementation)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip input consistency validation across ranks",
    )
    return parser.parse_args()


def parse_image_from_selection(selection: int) -> tuple[Path, Path]:
    base_path = Path("cells") / f"selection_{selection}"
    moving_path = base_path / "fish_dapi.jpg"
    sel = f"sel{selection}"
    reference_path = base_path / f"xenium_region_2-{sel}-highres_wide.png"
    return moving_path, reference_path


def str_to_translation_config(metric: str):
    if metric == "ncc":
        return NCCTranslationConfig()
    elif metric == "ngf":
        return GNGFTranslationConfig()
    elif metric == "gpc":
        return GPCTranslationConfig()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def get_mpi_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


def main() -> None:
    args = parse_args()
    comm = get_mpi_comm()

    #TODO(wlr): If runnin multiple ranks on a single node, set different GPU devices per rank.

    if comm is not None:
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    else:
        rank = 0
        world_size = 1

    def log(msg: str):
        if rank == 0:
            print(msg)

    log(f"Running with {world_size} MPI process(es)")

    # Check for GPU-aware MPI
    if args.gpu_aware:
        if is_cuda_aware_mpi():
            log("GPU-aware MPI detected and enabled")
        else:
            log("WARNING: GPU-aware MPI requested but not detected. Falling back to CPU transfer.")
            log("  Set OMPI_MCA_opal_cuda_support=1 for OpenMPI with CUDA support")

    if not args.no_validate:
        log("Input consistency validation enabled")

    moving_path, reference_path = parse_image_from_selection(args.selection)

    moving = load_image(moving_path, grayscale=True)
    reference = load_image(reference_path, grayscale=True)

    log(f"Loaded selection {args.selection}")
    log(f"Reference image: {reference_path}")
    log(f"Moving image: {moving_path}")

    cfg = CellSizeCfg(
        radius_hint_px=14.0,
        use_log=True,
        blob_weight=0.4,
        max_dim=1024,
        morph_radius_px=3,
        peak_rel_threshold=0.3,
    )

    unit = SpatialUnitResize(inner=CellUnitResize(cell_cfg=cfg))
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

    moving_search, reference_search = prepare([moving.clone(), reference.clone()])

    log(f"Search image shape: {tuple(reference_search.detail.image.shape)}")

    device = auto_device(args.device)
    reference_search = reference_search.to(device)
    moving_search = moving_search.to(device)

    search_params = SearchParams(
        scale_x=ScaleRange(min_scale=1.0, max_scale=1.3, step=0.01),
        scale_y=ScaleRange(min_scale=1.0, max_scale=1.3, step=0.01),
    )

    config = ExhaustiveSearchConfig(
        translation=str_to_translation_config(args.metric),
        batch_size=args.batch_size,
        progress_enabled=True,
        gradient=CDGradientConfig(
            eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
            normalize=NormalizeConfig(norm="l2", threshold=1e-3),
        ),
    )

    if comm is not None:
        comm.Barrier()

    start_t = time.perf_counter()

    try:
        mpi_config = MPIExhaustiveSearchConfig(
            top_k=1,
            comm=comm,
            root=0,
            progress_rank=0,
            return_on_all_ranks=False,
            validate_inputs=not args.no_validate,
            gpu_aware_mpi=args.gpu_aware,
        )
        search = MPIExhaustiveWarpSearch(search_params, config, mpi_config)
        result = search(reference_search.detail, moving_search.detail)
    except InconsistentInputsError as e:
        log(f"ERROR: Ranks have inconsistent inputs!")
        log(f"  Mismatched ranks: {e.mismatched_ranks}")
        log("  Ensure all ranks load the same data files.")
        raise
    except MPISearchError as e:
        log(f"ERROR: Rank {e.failed_rank} failed during search")
        log(f"  {e.error_type}: {e.error_message}")
        raise

    if comm is not None:
        comm.Barrier()

    end_t = time.perf_counter()

    log(f"Search took {end_t - start_t:.2f} seconds with {world_size} process(es)")

    if rank == 0 and result is not None:
        print("\nEstimated transformation parameters:")
        warp = result.warp
        print(f"  angle: {warp.angle.item():.2f} degrees")
        print(f"  scale_x: {warp.scale_x.item():.4f}")
        print(f"  scale_y: {warp.scale_y.item():.4f}")

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
            edge_overlay=EdgeOverlaySpec(
                edge_source='mov',
                edge_threshold=0.3,
            )
        )
        overlay_spec = OverlaySpec(alpha=0.5)
        checkerboard = render_checkerboard(preview.reference, preview.moving_warped, checkerboard_spec)
        overlay = render_overlay(preview.reference, preview.moving_warped, overlay_spec)

        if args.show:
            show_comparison(
                preview.reference,
                preview.moving_warped,
                mode="overlay",
                spec=overlay_spec,
            )

        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.output_prefix

        out_dir = out_dir / prefix / f"metric_{args.metric}_searchdim_{args.search_dim}"

        store = ProductIO(out_dir)
        manifest_path = store.save(
            transform,
            preview=preview,
            checkerboard=checkerboard,
            overlay=overlay,
            debug_details={"search_result": result_cpu},
            overwrite=True,
        )
        print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
