from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from batchmatch.base.tensordicts import (
    ImageDetail,
    TranslationResults,
    WarpParams,
)
from batchmatch.search.config import ExhaustiveSearchConfig, SearchParams
from batchmatch.search.exhaustive import ExhaustiveWarpSearch

__all__ = [
    "InconsistentInputsError",
    "MPISearchError",
    "MPIExhaustiveSearchConfig",
    "validate_inputs_consistent",
    "serialize_search_result",
    "deserialize_search_result",
    "serialize_search_result_gpu",
    "deserialize_search_result_gpu",
    "merge_top_k_results",
    "gather_results",
    "allgather_results",
    "gather_results_gpu",
    "allgather_results_gpu",
    "reduce_top_k",
    "allreduce_top_k",
    "reduce_top_k_gpu",
    "allreduce_top_k_gpu",
    "is_cuda_aware_mpi",
    "MPIExhaustiveWarpSearch",
    "mpi_exhaustive_search",
]

class InconsistentInputsError(Exception):
    def __init__(
        self,
        mismatched_ranks: List[int],
        checksums: Dict[int, str],
        reference_rank: int = 0,
    ):
        self.mismatched_ranks = mismatched_ranks
        self.checksums = checksums
        self.reference_rank = reference_rank
        super().__init__(
            f"Ranks have inconsistent inputs. "
            f"Reference rank: {reference_rank}, "
            f"mismatched ranks: {mismatched_ranks}. "
            f"Ensure all ranks load identical data."
        )


class MPISearchError(Exception):
    def __init__(
        self,
        failed_rank: int,
        error_type: str,
        error_message: str,
    ):
        self.failed_rank = failed_rank
        self.error_type = error_type
        self.error_message = error_message
        super().__init__(
            f"Rank {failed_rank} failed with {error_type}: {error_message}"
        )

@dataclass(frozen=True)
class MPIExhaustiveSearchConfig:
    top_k: int = 1
    comm: Any = None
    root: int = 0
    progress_rank: int = 0
    return_on_all_ranks: bool = False
    validate_inputs: bool = True
    gpu_aware_mpi: bool = False

def _compute_input_checksum(
    search_params: SearchParams,
    config: ExhaustiveSearchConfig,
    reference: ImageDetail,
    moving: ImageDetail,
) -> str:
    hasher = hashlib.sha256()

    rotation_ap = search_params.rotation.to_ap()
    hasher.update(f"rotation:{rotation_ap.start},{rotation_ap.end},{rotation_ap.step}".encode())

    scale_x_ap = search_params.scale_x.to_ap()
    hasher.update(f"scale_x:{scale_x_ap.start},{scale_x_ap.end},{scale_x_ap.step}".encode())

    scale_y_ap = search_params.scale_y.to_ap()
    hasher.update(f"scale_y:{scale_y_ap.start},{scale_y_ap.end},{scale_y_ap.step}".encode())

    shear_x_ap = search_params.shear_x.to_ap()
    hasher.update(f"shear_x:{shear_x_ap.start},{shear_x_ap.end},{shear_x_ap.step}".encode())

    shear_y_ap = search_params.shear_y.to_ap()
    hasher.update(f"shear_y:{shear_y_ap.start},{shear_y_ap.end},{shear_y_ap.step}".encode())

    order_str = "_".join(g.value for g in search_params.order)
    hasher.update(f"order:{order_str}".encode())

    hasher.update(f"batch_size:{config.batch_size}".encode())
    hasher.update(f"translation_method:{config.translation_method}".encode())
    hasher.update(f"gradient_method:{config.gradient_method}".encode())

    ref_img = reference.image.detach().cpu().contiguous().numpy()
    mov_img = moving.image.detach().cpu().contiguous().numpy()

    hasher.update(f"ref_shape:{ref_img.shape}".encode())
    hasher.update(ref_img.tobytes())

    hasher.update(f"mov_shape:{mov_img.shape}".encode())
    hasher.update(mov_img.tobytes())

    ref_mask = reference.mask
    if ref_mask is not None:
        ref_mask_np = ref_mask.detach().cpu().contiguous().numpy()
        hasher.update(f"ref_mask_shape:{ref_mask_np.shape}".encode())
        hasher.update(ref_mask_np.tobytes())

    mov_mask = moving.mask
    if mov_mask is not None:
        mov_mask_np = mov_mask.detach().cpu().contiguous().numpy()
        hasher.update(f"mov_mask_shape:{mov_mask_np.shape}".encode())
        hasher.update(mov_mask_np.tobytes())

    return hasher.hexdigest()


def validate_inputs_consistent(
    search_params: SearchParams,
    config: ExhaustiveSearchConfig,
    reference: ImageDetail,
    moving: ImageDetail,
    comm: Any,
) -> None:
    rank = comm.Get_rank()

    local_checksum = _compute_input_checksum(
        search_params, config, reference, moving
    )

    all_checksums = comm.allgather(local_checksum)

    checksums = {i: cs for i, cs in enumerate(all_checksums)}

    reference_checksum = all_checksums[0]
    mismatched = [
        i for i, cs in enumerate(all_checksums)
        if cs != reference_checksum
    ]

    if mismatched:
        raise InconsistentInputsError(
            mismatched_ranks=mismatched,
            checksums=checksums,
            reference_rank=0,
        )

def _check_collective_error(
    local_error: Optional[Exception],
    comm: Any,
) -> None:
    rank = comm.Get_rank()

    if local_error is not None:
        error_info = {
            "rank": rank,
            "type": type(local_error).__name__,
            "message": str(local_error),
        }
    else:
        error_info = None

    all_errors = comm.allgather(error_info)

    for info in all_errors:
        if info is not None:
            raise MPISearchError(
                failed_rank=info["rank"],
                error_type=info["type"],
                error_message=info["message"],
            )


def serialize_search_result(detail: ImageDetail) -> Dict[str, Any]:
    warp = detail.warp
    if warp is None:
        raise ValueError("ImageDetail must contain warp parameters")

    translation = detail.translation_results
    if translation is None:
        raise ValueError("ImageDetail must contain translation results")

    warp_data: Dict[str, np.ndarray] = {}
    for key in WarpParams.Keys.PARAMS:
        tensor = warp.get(key, default=None)
        if tensor is not None:
            warp_data[key] = tensor.detach().cpu().numpy()

    translation_data: Dict[str, np.ndarray] = {
        TranslationResults.Keys.X: translation.tx.detach().cpu().numpy(),
        TranslationResults.Keys.Y: translation.ty.detach().cpu().numpy(),
        TranslationResults.Keys.SCORE: translation.score.detach().cpu().numpy(),
    }

    search_dims = translation.search_dimensions
    if search_dims is not None:
        search_h, search_w = search_dims
        translation_data[TranslationResults.Keys.SEARCH_H] = (
            search_h.detach().cpu().numpy()
        )
        translation_data[TranslationResults.Keys.SEARCH_W] = (
            search_w.detach().cpu().numpy()
        )

    batch_size = translation.tx.shape[0]

    return {
        "warp": warp_data,
        "translation": translation_data,
        "batch_size": batch_size,
    }


def deserialize_search_result(
    data: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> ImageDetail:
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    batch_size = data["batch_size"]

    warp_tensors: Dict[str, torch.Tensor] = {}
    for key, arr in data["warp"].items():
        warp_tensors[key] = torch.from_numpy(arr).to(device=device, dtype=dtype)

    warp = WarpParams(warp_tensors, batch_size=[batch_size])

    trans_data = data["translation"]
    trans_tensors: Dict[str, torch.Tensor] = {}
    for key, arr in trans_data.items():
        trans_tensors[key] = torch.from_numpy(arr).to(device=device, dtype=dtype)

    translation = TranslationResults(trans_tensors, batch_size=[batch_size])

    dummy_image = torch.zeros(
        (batch_size, 1, 1, 1), device=device, dtype=dtype
    )

    detail = ImageDetail(
        {ImageDetail.Keys.IMAGE: dummy_image},
        batch_size=[batch_size],
    )
    detail.set(ImageDetail.Keys.WARP.ROOT, warp)
    detail.set(ImageDetail.Keys.TRANSLATION.ROOT, translation)

    return detail


def serialize_search_result_gpu(detail: ImageDetail) -> Dict[str, Any]:
    warp = detail.warp
    if warp is None:
        raise ValueError("ImageDetail must contain warp parameters")

    translation = detail.translation_results
    if translation is None:
        raise ValueError("ImageDetail must contain translation results")

    warp_data: Dict[str, torch.Tensor] = {}
    for key in WarpParams.Keys.PARAMS:
        tensor = warp.get(key, default=None)
        if tensor is not None:
            warp_data[key] = tensor.detach().clone()

    translation_data: Dict[str, torch.Tensor] = {
        TranslationResults.Keys.X: translation.tx.detach().clone(),
        TranslationResults.Keys.Y: translation.ty.detach().clone(),
        TranslationResults.Keys.SCORE: translation.score.detach().clone(),
    }

    search_dims = translation.search_dimensions
    if search_dims is not None:
        search_h, search_w = search_dims
        translation_data[TranslationResults.Keys.SEARCH_H] = search_h.detach().clone()
        translation_data[TranslationResults.Keys.SEARCH_W] = search_w.detach().clone()

    batch_size = translation.tx.shape[0]
    device_str = str(translation.tx.device)

    return {
        "warp": warp_data,
        "translation": translation_data,
        "batch_size": batch_size,
        "device": device_str,
    }


def deserialize_search_result_gpu(
    data: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> ImageDetail:
    if device is None:
        device = data.get("device", "cpu")
    if dtype is None:
        dtype = torch.float32

    batch_size = data["batch_size"]

    warp_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in data["warp"].items():
        warp_tensors[key] = tensor.to(device=device, dtype=dtype)

    warp = WarpParams(warp_tensors, batch_size=[batch_size])

    trans_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in data["translation"].items():
        trans_tensors[key] = tensor.to(device=device, dtype=dtype)

    translation = TranslationResults(trans_tensors, batch_size=[batch_size])

    dummy_image = torch.zeros(
        (batch_size, 1, 1, 1), device=device, dtype=dtype
    )

    detail = ImageDetail(
        {ImageDetail.Keys.IMAGE: dummy_image},
        batch_size=[batch_size],
    )
    detail.set(ImageDetail.Keys.WARP.ROOT, warp)
    detail.set(ImageDetail.Keys.TRANSLATION.ROOT, translation)

    return detail


def merge_top_k_results(
    results: List[ImageDetail],
    k: int,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> ImageDetail:
    if not results:
        raise ValueError("results list cannot be empty")
    if k < 1:
        raise ValueError("k must be >= 1")

    non_empty = [r for r in results if r.batch_size[0] > 0]
    if not non_empty:
        raise ValueError("All results are empty")

    if len(non_empty) == 1:
        merged = non_empty[0]
    else:
        merged = torch.cat(non_empty, dim=0)

    if device is not None or dtype is not None:
        merged = merged.to(device=device, dtype=dtype)

    translation = merged.translation_results
    if translation is None:
        raise ValueError("Merged results missing translation data")

    scores = translation.score
    total = scores.shape[0]
    actual_k = min(k, total)

    _, indices = torch.topk(scores, k=actual_k, largest=True, sorted=True)

    return merged[indices].clone()


def _get_mpi():
    try:
        from mpi4py import MPI
        return MPI
    except ImportError as e:
        raise ImportError(
            "mpi4py is required for MPI communication. "
            "Install with: pip install mpi4py or uv sync --extras mpi"
        ) from e


def _detect_mpi_world_size() -> Optional[int]:
    env_vars = (
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",  # MPICH/PMI
        "PMIX_SIZE",  # PMIx
        "MV2_COMM_WORLD_SIZE",  # MVAPICH2
        "SLURM_NTASKS",  # SLURM
        "MPI_LOCALNRANKS",  # Intel MPI
    )
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            try:
                size = int(value)
            except ValueError:
                continue
            if size > 0:
                return size
    return None


def _warn_if_mpi_launcher_without_comm(*, stacklevel: int = 2) -> None:
    mpi_size = _detect_mpi_world_size()
    if mpi_size and mpi_size > 1:
        warnings.warn(
            "MPI launcher detected but no MPI communicator was provided. "
            "Running single-process search on this rank; if launched under "
            "mpirun/mpiexec, each rank will do the full search. "
            "Ensure mpi4py is installed and pass MPI.COMM_WORLD.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def is_cuda_aware_mpi() -> bool:
    try:
        from mpi4py import MPI

        if hasattr(MPI, "Query_cuda_support"):
            return MPI.Query_cuda_support()

        if os.environ.get("OMPI_MCA_opal_cuda_support", "0") == "1":
            return True

        if os.environ.get("MV2_USE_CUDA", "0") == "1":
            return True

        ompi_info = os.environ.get("OMPI_MCA_mpi_built_with_cuda_support", "")
        if ompi_info.lower() == "true":
            return True

        return False

    except ImportError:
        return False
    except Exception:
        return False

def gather_results(
    local_result: ImageDetail,
    comm: Any,
    root: int = 0,
) -> Optional[List[ImageDetail]]:
    _get_mpi()

    serialized = serialize_search_result(local_result)
    all_serialized = comm.gather(serialized, root=root)

    if comm.Get_rank() == root:
        device = local_result.device
        dtype = local_result.get(ImageDetail.Keys.IMAGE).dtype

        return [
            deserialize_search_result(s, device=device, dtype=dtype)
            for s in all_serialized
        ]
    return None


def allgather_results(
    local_result: ImageDetail,
    comm: Any,
) -> List[ImageDetail]:
    _get_mpi()

    serialized = serialize_search_result(local_result)
    all_serialized = comm.allgather(serialized)

    device = local_result.device
    dtype = local_result.get(ImageDetail.Keys.IMAGE).dtype

    return [
        deserialize_search_result(s, device=device, dtype=dtype)
        for s in all_serialized
    ]

def gather_results_gpu(
    local_result: ImageDetail,
    comm: Any,
    root: int = 0,
) -> Optional[List[ImageDetail]]:
    _get_mpi()

    serialized = serialize_search_result_gpu(local_result)
    all_serialized = comm.gather(serialized, root=root)

    if comm.Get_rank() == root:
        device = local_result.device
        dtype = local_result.get(ImageDetail.Keys.IMAGE).dtype

        return [
            deserialize_search_result_gpu(s, device=device, dtype=dtype)
            for s in all_serialized
        ]
    return None


def allgather_results_gpu(
    local_result: ImageDetail,
    comm: Any,
) -> List[ImageDetail]:
    _get_mpi()

    serialized = serialize_search_result_gpu(local_result)
    all_serialized = comm.allgather(serialized)

    device = local_result.device
    dtype = local_result.get(ImageDetail.Keys.IMAGE).dtype

    return [
        deserialize_search_result_gpu(s, device=device, dtype=dtype)
        for s in all_serialized
    ]

def reduce_top_k(
    local_result: ImageDetail,
    k: int,
    comm: Any,
    root: int = 0,
) -> Optional[ImageDetail]:
    all_results = gather_results(local_result, comm, root=root)

    if comm.Get_rank() == root:
        return merge_top_k_results(all_results, k)
    return None


def allreduce_top_k(
    local_result: ImageDetail,
    k: int,
    comm: Any,
) -> ImageDetail:
    all_results = allgather_results(local_result, comm)
    return merge_top_k_results(all_results, k)

def reduce_top_k_gpu(
    local_result: ImageDetail,
    k: int,
    comm: Any,
    root: int = 0,
) -> Optional[ImageDetail]:
    all_results = gather_results_gpu(local_result, comm, root=root)

    if comm.Get_rank() == root:
        return merge_top_k_results(all_results, k)
    return None


def allreduce_top_k_gpu(
    local_result: ImageDetail,
    k: int,
    comm: Any,
) -> ImageDetail:
    all_results = allgather_results_gpu(local_result, comm)
    return merge_top_k_results(all_results, k)

class MPIExhaustiveWarpSearch(ExhaustiveWarpSearch):
    """MPI-enabled exhaustive search with a unified forward interface."""

    def __init__(
        self,
        search_params: SearchParams,
        config: Optional[ExhaustiveSearchConfig] = None,
        mpi_config: Optional[MPIExhaustiveSearchConfig] = None,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        if config is None:
            config = ExhaustiveSearchConfig()
        if mpi_config is None:
            mpi_config = MPIExhaustiveSearchConfig()

        self.search_config = config
        self.mpi_config = mpi_config
        self.search_params = search_params

        comm = mpi_config.comm
        if comm is not None:
            rank = comm.Get_rank()
            world_size = comm.Get_size()
        else:
            rank = 0
            world_size = 1
            _warn_if_mpi_launcher_without_comm(stacklevel=3)

        progress_enabled = (
            config.progress_enabled
            and mpi_config.progress_rank >= 0
            and rank == mpi_config.progress_rank
        )
        rank_config = replace(config, progress_enabled=progress_enabled)

        super().__init__(
            search_params,
            rank_config,
            world_size=world_size,
            rank=rank,
            dtype=dtype,
        )

        self.comm = comm
        self.rank = rank
        self.world_size = world_size

    @torch.inference_mode()
    def search(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
        *,
        top_k: Optional[int] = None,
        progress: Union[bool, None] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        clone_inputs: bool = True,
    ) -> Optional[ImageDetail]:
        if top_k is None:
            top_k = self.mpi_config.top_k

        if self.comm is None:
            return super().search(
                reference,
                moving,
                top_k=top_k,
                progress=progress,
                callback=callback,
                clone_inputs=clone_inputs,
            )

        if self.mpi_config.validate_inputs:
            validate_inputs_consistent(
                self.search_params,
                self.search_config,
                reference,
                moving,
                self.comm,
            )

        local_error: Optional[Exception] = None
        local_result: Optional[ImageDetail] = None

        local_top_k = (
            min(top_k * 2, self.grid.total_count)
            if self.grid.total_count > 0
            else top_k
        )

        try:
            if self.grid.total_count == 0:
                local_result = _create_empty_result(self.device, self.dtype)
            else:
                local_result = super().search(
                    reference,
                    moving,
                    top_k=local_top_k,
                    progress=progress,
                    callback=callback,
                    clone_inputs=clone_inputs,
                )
        except Exception as e:
            local_error = e
            local_result = _create_empty_result(self.device, self.dtype)

        _check_collective_error(local_error, self.comm)

        if self.mpi_config.gpu_aware_mpi:
            if self.mpi_config.return_on_all_ranks:
                return allreduce_top_k_gpu(local_result, top_k, self.comm)
            return reduce_top_k_gpu(
                local_result, top_k, self.comm, root=self.mpi_config.root
            )

        if self.mpi_config.return_on_all_ranks:
            return allreduce_top_k(local_result, top_k, self.comm)
        return reduce_top_k(
            local_result, top_k, self.comm, root=self.mpi_config.root
        )

def mpi_exhaustive_search(
    search_params: SearchParams,
    config: ExhaustiveSearchConfig,
    reference: ImageDetail,
    moving: ImageDetail,
    *,
    top_k: int = 1,
    comm: Any = None,
    root: int = 0,
    progress_rank: int = 0,
    return_on_all_ranks: bool = False,
    validate_inputs: bool = True,
    gpu_aware_mpi: bool = False,
) -> Optional[ImageDetail]:
    mpi_config = MPIExhaustiveSearchConfig(
        top_k=top_k,
        comm=comm,
        root=root,
        progress_rank=progress_rank,
        return_on_all_ranks=return_on_all_ranks,
        validate_inputs=validate_inputs,
        gpu_aware_mpi=gpu_aware_mpi,
    )
    search = MPIExhaustiveWarpSearch(search_params, config, mpi_config)
    return search(reference, moving)


def _create_empty_result(
    device: torch.device,
    dtype: torch.dtype,
) -> ImageDetail:
    dummy_image = torch.zeros((0, 1, 1, 1), device=device, dtype=dtype)

    warp_data = {
        key: torch.zeros(0, device=device, dtype=dtype)
        for key in WarpParams.Keys.PARAMS
    }
    warp = WarpParams(warp_data, batch_size=[0])

    trans_data = {
        TranslationResults.Keys.X: torch.zeros(0, device=device, dtype=dtype),
        TranslationResults.Keys.Y: torch.zeros(0, device=device, dtype=dtype),
        TranslationResults.Keys.SCORE: torch.zeros(0, device=device, dtype=dtype),
    }
    translation = TranslationResults(trans_data, batch_size=[0])

    detail = ImageDetail({ImageDetail.Keys.IMAGE: dummy_image}, batch_size=[0])
    detail.set(ImageDetail.Keys.WARP.ROOT, warp)
    detail.set(ImageDetail.Keys.TRANSLATION.ROOT, translation)

    return detail
