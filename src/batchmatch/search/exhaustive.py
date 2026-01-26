from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from batchmatch.base.pipeline import Pipeline, Stage
from batchmatch.base.tensordicts import CacheTD, ImageDetail, TranslationResults, WarpParams
from batchmatch.search.config import (
    ExhaustiveSearchConfig,
    GridAP,
    SearchParams,
    grid_for_rank,
)
from batchmatch.search.iterator import (
    ProgressTracker,
    WarpParamIterator,
    make_search_progress,
)

__all__ = [
    "ExhaustiveSearchConfig",
    "ExhaustiveWarpSearch",
    "build_reference_pipeline",
    "build_moving_pipeline",
    "compile_search",
]



def _get_gradient_stage(
    method: Union[str, Mapping, GradientPipelineConfig],
    *,
    require_complex: bool = False,
    **kwargs,
) -> Stage:
    """Build a gradient computation stage."""
    from batchmatch.gradient.base import build_gradient_pipeline
    build_complex = kwargs.pop("build_complex", False)
    return build_gradient_pipeline(
        method,
        build_complex=require_complex or build_complex,
        **kwargs,
    )


def _get_translation_stage(method: str, **kwargs) -> Stage:
    """Build a translation search stage."""
    from batchmatch.translate import build_translation_stage

    return build_translation_stage(method, **kwargs)


def _get_warp_stages(*, inverse: bool = True) -> List[Stage]:
    """Build warp stages: prepare + image + mask + window."""
    from batchmatch.warp.stages import (
        PrepareWarpStage,
        WarpImageStage,
        WarpMaskStage,
        WarpWindowStage,
    )

    return [
        PrepareWarpStage(inverse=inverse),
        WarpImageStage(),
        WarpMaskStage(),
        WarpWindowStage(),
    ]

def build_reference_pipeline(
    config: ExhaustiveSearchConfig,
    translation_stage: Stage,
) -> Pipeline:
    stages: List[Stage] = []

    requires_gradients = getattr(translation_stage, "requires_gradients", False)
    requires_complex = getattr(translation_stage, "requires_complex_gradients", False)

    if requires_gradients:
        grad_stage = _get_gradient_stage(config.gradient_method, require_complex=requires_complex, **config.gradient_params)
        stages.append(grad_stage)

    if config.use_reference_cache:
        class PrecomputeCacheStage(Stage):
            def __init__(self, translation_stage: Stage):
                super().__init__()
                self.translation_stage = translation_stage

            def forward(self, image: ImageDetail) -> ImageDetail:
                if hasattr(self.translation_stage, "prepare_cache"):
                    self.translation_stage.prepare_cache(image)
                return image

        stages.append(PrecomputeCacheStage(translation_stage))

    return Pipeline(stages)


def build_moving_pipeline(
    config: ExhaustiveSearchConfig,
    translation_stage: Stage,
) -> Pipeline:
    stages: List[Stage] = []

    stages.extend(_get_warp_stages(inverse=True))

    requires_gradients = getattr(translation_stage, "requires_gradients", False)
    requires_complex = getattr(translation_stage, "requires_complex_gradients", False)

    if requires_gradients:
        grad_stage = _get_gradient_stage(config.gradient_method, require_complex=requires_complex, **config.gradient_params)
        stages.append(grad_stage)

    return Pipeline(stages)



class ExhaustiveWarpSearch(nn.Module):

    def __init__(
        self,
        search_params: SearchParams,
        config: Optional[ExhaustiveSearchConfig] = None,
        *,
        world_size: int = 1,
        rank: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if config is None:
            config = ExhaustiveSearchConfig()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = dtype

        self.grid: GridAP = grid_for_rank(search_params, world_size=world_size, rank=rank)

        self.translation_stage = _get_translation_stage(
            config.translation_method,
            **config.translation_params,
        )

        self.reference_pipeline = build_reference_pipeline(config, self.translation_stage)
        self.moving_pipeline = build_moving_pipeline(config, self.translation_stage)
        self.reference_pipeline = self.reference_pipeline.to(self.device)
        self.moving_pipeline = self.moving_pipeline.to(self.device)

        self._configured_batch_size = config.batch_size
        self._auto_batch_size = config.auto_batch_size
        self._max_auto_batch_size = config.max_auto_batch_size

        self._warp_buffer: Optional[Tensor] = None
        self._grid_buffer: Optional[Tensor] = None
        self._m_fwd_buffer: Optional[Tensor] = None
        self._m_inv_buffer: Optional[Tensor] = None

        self._moving_buffers: Dict[Any, Tensor] = {}

        self._moving_batch: Optional[ImageDetail] = None
        self._moving_batch_size: int = 0
        
        self._original_image_expanded: Optional[Tensor] = None
        self._original_mask_expanded: Optional[Tensor] = None
        self._original_window_expanded: Optional[Tensor] = None
        
        self._search_h: Optional[Tensor] = None
        self._search_w: Optional[Tensor] = None
        self._search_dims_batch_size: int = 0
        
        self._coord_xs: Optional[Tensor] = None
        self._coord_ys: Optional[Tensor] = None
        self._coord_shape: Tuple[int, int] = (0, 0)
        
        self._cached_group_tables: Optional[Dict] = None

    def clear_cache(self) -> None:
        self._warp_buffer = None
        self._grid_buffer = None
        self._m_fwd_buffer = None
        self._m_inv_buffer = None
        self._moving_buffers.clear()
        self._moving_batch = None
        self._moving_batch_size = 0
        self._original_image_expanded = None
        self._original_mask_expanded = None
        self._original_window_expanded = None
        self._search_h = None
        self._search_w = None
        self._search_dims_batch_size = 0
        self._coord_xs = None
        self._coord_ys = None
        self._coord_shape = (0, 0)
        self._cached_group_tables = None

    def _apply(self, fn):
        """Override _apply to update self.device when module is moved.

        This is called by .to(), .cuda(), .cpu(), etc. We use it to keep
        self.device in sync with the actual device of the module's parameters.
        """
        # Call parent's _apply to move all registered modules/buffers
        super()._apply(fn)

        # probe what device the fn maps to
        dummy = torch.tensor(0.0)
        try:
            new_dummy = fn(dummy)
            self.device = new_dummy.device
        except Exception:
            for module in self.modules():
                for param in module.parameters(recurse=False):
                    self.device = param.device
                    break
                for buffer in module.buffers(recurse=False):
                    self.device = buffer.device
                    break
                if self.device != torch.device("cpu"):
                    break
        self.clear_cache()

        return self

    def _validate_inputs(self, reference: ImageDetail, moving: ImageDetail) -> None:
        ref_img = reference.get(ImageDetail.Keys.IMAGE)
        mov_img = moving.get(ImageDetail.Keys.IMAGE)

        if ref_img.ndim != 4:
            raise ValueError(
                f"Reference image must be 4D BCHW tensor, got shape {tuple(ref_img.shape)} "
                f"with {ref_img.ndim} dimensions. Use image.unsqueeze(0) if needed."
            )
        if mov_img.ndim != 4:
            raise ValueError(
                f"Moving image must be 4D BCHW tensor, got shape {tuple(mov_img.shape)} "
                f"with {mov_img.ndim} dimensions. Use image.unsqueeze(0) if needed."
            )
        if ref_img.shape[0] != 1:
            raise ValueError(
                f"Reference must have batch size 1, got batch size {ref_img.shape[0]} "
                f"with shape {tuple(ref_img.shape)}. Use reference[0:1] to select first batch."
            )
        if mov_img.shape[0] != 1:
            raise ValueError(
                f"Moving must have batch size 1, got batch size {mov_img.shape[0]} "
                f"with shape {tuple(mov_img.shape)}. Use moving[0:1] to select first batch."
            )

    def _prepare_reference(self, reference: ImageDetail, clone: bool = True) -> ImageDetail:
        if clone:
            reference = reference.clone()
        reference = self.reference_pipeline(reference)
        return reference

    def _prepare_moving_base(self, moving: ImageDetail, clone: bool = True) -> ImageDetail:
        if clone:
            moving = moving.clone()
        return moving

    def _expand_container(
        self,
        td: CacheTD,
        batch_size: int,
        cls: type[CacheTD],
    ) -> CacheTD:
        data: Dict[str, Tensor] = {}
        for key, value in td.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                value = value.expand(batch_size, *value.shape[1:])
            data[key] = value
        return cls(data, batch_size=[batch_size])

    def _expand_to_batch(self, image: ImageDetail, batch_size: int) -> ImageDetail:
        skip_roots = set()
        if ImageDetail.Keys.WARP.ROOT in image.keys():
            skip_roots.add(ImageDetail.Keys.WARP.ROOT)
        if ImageDetail.Keys.TRANSLATION.ROOT in image.keys():
            skip_roots.add(ImageDetail.Keys.TRANSLATION.ROOT)

        expanded_data = {}
        for key in image.keys(include_nested=True, leaves_only=True):
            if (
                skip_roots
                and isinstance(key, tuple)
                and key
                and key[0] in skip_roots
            ):
                continue
            tensor = image.get(key)
            if tensor.shape[0] == 1:
                expanded = tensor.expand(batch_size, *tensor.shape[1:])
                expanded_data[key] = expanded
            else:
                expanded_data[key] = tensor

        expanded = ImageDetail(expanded_data, batch_size=[batch_size])

        warp = image.get(ImageDetail.Keys.WARP.ROOT, None)
        if isinstance(warp, WarpParams):
            expanded_warp = self._expand_container(warp, batch_size, WarpParams)
            expanded.set(ImageDetail.Keys.WARP.ROOT, expanded_warp)

        translation = image.get(ImageDetail.Keys.TRANSLATION.ROOT, None)
        if isinstance(translation, TranslationResults):
            expanded_translation = self._expand_container(
                translation, batch_size, TranslationResults
            )
            expanded.set(ImageDetail.Keys.TRANSLATION.ROOT, expanded_translation)

        return expanded

    def _add_warp_params_to_image(
        self, image: ImageDetail, warp_params: WarpParams
    ) -> ImageDetail:
        """Add warp parameters to ImageDetail under WARP namespace."""
        image.set(ImageDetail.Keys.WARP.ROOT, warp_params)
        return image

    def _init_moving_batch_container(
        self,
        moving_base: ImageDetail,
        batch_size: int,
    ) -> None:
        base_img = moving_base.image
        C, H, W = base_img.shape[-3:]
        
        self._moving_batch = self._expand_to_batch(moving_base, batch_size)
        self._moving_batch_size = batch_size
        
        self._original_image_expanded = self._moving_batch.get(ImageDetail.Keys.IMAGE)        
        self._original_mask_expanded = self._moving_batch.get(
            ImageDetail.Keys.DOMAIN.MASK, default=None
        )
        self._original_window_expanded = self._moving_batch.get(
            ImageDetail.Keys.DOMAIN.WINDOW, default=None
        )
        
        sample_shape = (batch_size, C + 1, H, W)
        grid_shape = (batch_size, H, W, 2)
        matrix_shape = (batch_size, 3, 3)
        
        self._warp_buffer = torch.empty(
            sample_shape, device=self.device, dtype=self.dtype
        )
        
        self._grid_buffer = torch.empty(
            grid_shape, device=self.device, dtype=self.dtype
        )
        
        self._m_fwd_buffer = torch.zeros(
            matrix_shape, device=self.device, dtype=self.dtype
        )
        self._m_fwd_buffer[:, 2, 2] = 1.0
        
        self._m_inv_buffer = torch.zeros(
            matrix_shape, device=self.device, dtype=self.dtype
        )
        self._m_inv_buffer[:, 2, 2] = 1.0
        
        self._moving_batch.set(ImageDetail.Keys.WARP.SAMPLE, self._warp_buffer)
        self._moving_batch.set(ImageDetail.Keys.WARP.GRID, self._grid_buffer)
        self._moving_batch.set(ImageDetail.Keys.WARP.M_FWD, self._m_fwd_buffer)
        self._moving_batch.set(ImageDetail.Keys.WARP.M_INV, self._m_inv_buffer)

        self._moving_buffers.clear()

        if hasattr(self.translation_stage, "moving_buffer_specs"):
            for spec in self.translation_stage.moving_buffer_specs:
                shape = spec.compute_shape(batch_size, C, H, W)
                dtype = spec.get_dtype(self.dtype)
                buffer = torch.empty(shape, device=self.device, dtype=dtype)
                self._moving_buffers[spec.key] = buffer
                self._moving_batch.set(spec.key, buffer)

        self._ensure_coord_vectors(H, W)

    def _create_moving_batch_no_cache(
        self,
        moving_base: ImageDetail,
        batch_size: int,
    ) -> ImageDetail:
        base_img = moving_base.image
        C, H, W = base_img.shape[-3:]

        # Expand the base image to batch size
        moving = self._expand_to_batch(moving_base, batch_size)

        sample_shape = (batch_size, C + 1, H, W)
        grid_shape = (batch_size, H, W, 2)
        matrix_shape = (batch_size, 3, 3)

        warp_buffer = torch.empty(sample_shape, device=self.device, dtype=self.dtype)
        grid_buffer = torch.empty(grid_shape, device=self.device, dtype=self.dtype)

        m_fwd_buffer = torch.zeros(matrix_shape, device=self.device, dtype=self.dtype)
        m_fwd_buffer[:, 2, 2] = 1.0

        m_inv_buffer = torch.zeros(matrix_shape, device=self.device, dtype=self.dtype)
        m_inv_buffer[:, 2, 2] = 1.0

        moving.set(ImageDetail.Keys.WARP.SAMPLE, warp_buffer)
        moving.set(ImageDetail.Keys.WARP.GRID, grid_buffer)
        moving.set(ImageDetail.Keys.WARP.M_FWD, m_fwd_buffer)
        moving.set(ImageDetail.Keys.WARP.M_INV, m_inv_buffer)

        if hasattr(self.translation_stage, "moving_buffer_specs"):
            for spec in self.translation_stage.moving_buffer_specs:
                shape = spec.compute_shape(batch_size, C, H, W)
                dtype = spec.get_dtype(self.dtype)
                buffer = torch.empty(shape, device=self.device, dtype=dtype)
                moving.set(spec.key, buffer)

        return moving

    def _ensure_coord_vectors(self, H: int, W: int) -> Tuple[Tensor, Tensor]:
        if self._coord_shape == (H, W) and self._coord_xs is not None:
            return self._coord_xs, self._coord_ys        
        self._coord_xs = torch.arange(W, device=self.device, dtype=self.dtype).view(1, 1, W)
        self._coord_ys = torch.arange(H, device=self.device, dtype=self.dtype).view(1, H, 1)
        self._coord_shape = (H, W)
        return self._coord_xs, self._coord_ys

    def _prepare_moving_batch(
        self,
        moving_base: ImageDetail,
        warp_params: WarpParams,
    ) -> ImageDetail:
        B = warp_params.batch_size[0]

        if not self.config.use_moving_cache:
            moving = self._create_moving_batch_no_cache(moving_base, B)
            moving = self._add_warp_params_to_image(moving, warp_params)
            moving = self.moving_pipeline(moving)
            return moving

        if self._moving_batch is None or self._moving_batch_size < B:
            self._init_moving_batch_container(moving_base, B)

        if B < self._moving_batch_size:
            moving = self._moving_batch[:B]
            moving.set(ImageDetail.Keys.WARP.SAMPLE, self._warp_buffer[:B])
            moving.set(ImageDetail.Keys.WARP.GRID, self._grid_buffer[:B])
            moving.set(ImageDetail.Keys.WARP.M_FWD, self._m_fwd_buffer[:B])
            moving.set(ImageDetail.Keys.WARP.M_INV, self._m_inv_buffer[:B])

            for key, buffer in self._moving_buffers.items():
                moving.set(key, buffer[:B])

            moving.set(ImageDetail.Keys.IMAGE, self._original_image_expanded[:B])
            if self._original_mask_expanded is not None:
                moving.set(ImageDetail.Keys.DOMAIN.MASK, self._original_mask_expanded[:B])
            if self._original_window_expanded is not None:
                moving.set(ImageDetail.Keys.DOMAIN.WINDOW, self._original_window_expanded[:B])
        else:
            moving = self._moving_batch

            moving.set(ImageDetail.Keys.IMAGE, self._original_image_expanded)
            if self._original_mask_expanded is not None:
                moving.set(ImageDetail.Keys.DOMAIN.MASK, self._original_mask_expanded)
            if self._original_window_expanded is not None:
                moving.set(ImageDetail.Keys.DOMAIN.WINDOW, self._original_window_expanded)

        moving = self._add_warp_params_to_image(moving, warp_params)

        #Warp -> gradient -> postprocess
        moving = self.moving_pipeline(moving)

        return moving

    def _get_search_dims(
        self, reference: ImageDetail, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        # Prefer box shape (content dimensions) over canvas H/W
        box = reference.box
        if box is not None and box.numel() >= 4:
            if box.ndim == 3:
                b = box[0, 0]
            else:
                b = box[0]
            
            W_t = b[2] - b[0]
            H_t = b[3] - b[1]
            
            search_h = H_t.expand(batch_size).to(dtype=torch.float32)
            search_w = W_t.expand(batch_size).to(dtype=torch.float32)
            return search_h, search_w

        H = reference.H
        W = reference.W
        
        if (
            self._search_h is not None
            and self._search_dims_batch_size >= batch_size
            and self._search_h.device == self.device
        ):
            return self._search_h[:batch_size], self._search_w[:batch_size]
        
        self._search_h = torch.full(
            (batch_size,), float(H), device=self.device, dtype=torch.float32
        )
        self._search_w = torch.full(
            (batch_size,), float(W), device=self.device, dtype=torch.float32
        )
        self._search_dims_batch_size = batch_size
        return self._search_h, self._search_w

    def _search_batch(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
        warp_params: WarpParams,
    ) -> ImageDetail:
        moving = self.translation_stage(reference, moving)
        return self._clean_detail(moving)

    def _clean_detail(self, detail: ImageDetail) -> ImageDetail:
        warp = detail.get(ImageDetail.Keys.WARP.ROOT, None)
        if isinstance(warp, WarpParams):
            for key in WarpParams.Keys.COMPUTED:
                if key in warp.keys():
                    warp.del_(key)
        return detail

    def _top_k_detail(self, detail: ImageDetail, k: int) -> ImageDetail:
        """Select the top-k results by translation score."""
        if k <= 0:
            raise ValueError("k must be positive")

        tr = detail.translation_results
        if tr is None:
            raise ValueError("TranslationResults not found in ImageDetail")
        scores = tr.score

        actual_k = min(k, scores.shape[0])
        _, indices = torch.topk(scores, k=actual_k, largest=True, sorted=True)
        sliced = detail[indices]

        return self._clean_detail(sliced.clone())

    def _merge_details(self, left: ImageDetail, right: ImageDetail) -> ImageDetail:
        merged = torch.cat([left, right], dim=0)
        return merged

    def _effective_batch_size(self, requested: int) -> int:
        total = max(1, self.grid.total_count)
        return max(1, min(requested, total))

    def _resolve_batch_size(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
    ) -> int:
        requested = self._effective_batch_size(self._configured_batch_size)

        if not self._auto_batch_size:
            return requested

        if self.device.type != "cuda":
            return requested

        #doubling until OOM, then binary search
        return self._auto_tune_batch_size(reference, moving)

    def _estimate_batch_size(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
    ) -> int:
        if self.device.type != "cuda":
            return self._configured_batch_size

        try:
            import torch.cuda as cuda
            available_mb = cuda.mem_get_info(self.device)[0] / (1024 ** 2) * 0.8
            img = reference.image
            C, H, W = img.shape[1:]
            bytes_per_element = img.element_size()

            # Rough estimate: image + gradients + FFT buffers + misc
            # original, gradients (2x), window, FFT (2x)
            est_mb_per_sample = (C * H * W * bytes_per_element * 6) / (1024 ** 2)

            if est_mb_per_sample > 0:
                estimated = int(available_mb / est_mb_per_sample)
                return max(1, min(estimated, 1024))
        except Exception:
            pass

        return self._configured_batch_size

    def _auto_tune_batch_size(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
    ) -> int:
        cap = self._effective_batch_size(self.grid.total_count)
        if self._max_auto_batch_size is not None:
            cap = min(cap, self._max_auto_batch_size)

        estimated = self._estimate_batch_size(reference, moving)
        start = min(max(1, estimated), cap)

        if self._try_batch_size(reference, moving, start):
            best = start
            current = min(cap, start * 2)
            attempts = 0
            while current <= cap and attempts < 3:
                if self._try_batch_size(reference, moving, current):
                    best = current
                    current = min(cap, current * 2)
                else:
                    break
                attempts += 1
            return self._effective_batch_size(best)

        if start <= 1:
            raise RuntimeError(
                f"Unable to fit batch size 1 on device. "
                f"Image size: {reference.image.shape}, "
                f"Available memory: {torch.cuda.mem_get_info(self.device)[0] / (1024**2):.0f} MB"
            )

        low, high = 1, start - 1
        best = None

        while low <= high:
            mid = (low + high) // 2
            if self._try_batch_size(reference, moving, mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best is None:
            raise RuntimeError("Unable to fit even batch size 1 on device")

        return self._effective_batch_size(best)

    def _try_batch_size(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
        batch_size: int,
    ) -> bool:
        try:
            with torch.inference_mode():
                dummy_params = WarpParams.identity(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                moving_batch = self._prepare_moving_batch(moving, dummy_params)
                self.translation_stage(reference, moving_batch)

                return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    @torch.compiler.disable
    def _create_iterator(self, batch_size: int) -> WarpParamIterator:
        iterator = WarpParamIterator(
            self.grid,
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
            group_tables=self._cached_group_tables,
        )
        if self._cached_group_tables is None:
            self._cached_group_tables = iterator.group_tables
        return iterator
    
    def _setup_progress(
        self,
        progress: Union[bool, None],
    ) -> Tuple[Any, Optional[ProgressTracker]]:
        """Setup progress tracking."""
        from contextlib import nullcontext

        try:
            from rich.progress import Progress as RichProgress
        except ImportError:
            RichProgress = None

        if progress is False or not self.config.progress_enabled:
            return nullcontext(), None

        if RichProgress is not None and isinstance(progress, RichProgress):
            tracker = ProgressTracker(progress, description="Warp search")
            return nullcontext(), tracker

        if progress is True or progress is None:
            progress_obj = make_search_progress(transient=self.config.progress_transient)
            tracker = ProgressTracker(progress_obj, description="Warp search")
            return progress_obj, tracker

        return nullcontext(), None

    @torch.inference_mode()
    def search(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
        *,
        top_k: int = 1,
        progress: Union[bool, None] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        clone_inputs: bool = True,
    ) -> ImageDetail:
        if torch.compiler.is_compiling():
            progress = False
            callback = None

        reference = reference.to(device=self.device, dtype=self.dtype)
        moving = moving.to(device=self.device, dtype=self.dtype)
        self._validate_inputs(reference, moving)

        if self.grid.total_count == 0:
            raise ValueError(
                "Search grid is empty - no parameter combinations to search. "
                "Check your SearchParams ranges and predicates."
            )

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if top_k > self.grid.total_count and not torch.compiler.is_compiling():
            import warnings
            warnings.warn(
                f"top_k={top_k} exceeds grid size {self.grid.total_count}. "
                f"Will return at most {self.grid.total_count} results.",
                UserWarning,
                stacklevel=2
            )

        reference = self._prepare_reference(reference, clone=clone_inputs)
        moving_base = self._prepare_moving_base(moving, clone=clone_inputs)

        batch_size = self._resolve_batch_size(reference, moving_base)
        if self.config.use_moving_cache:
            self._init_moving_batch_container(moving_base, batch_size)
        progress_ctx, tracker = self._setup_progress(progress)
        iterator = self._create_iterator(batch_size)
        global_best: Optional[ImageDetail] = None

        with progress_ctx:
            for warp_params in iterator:
                moving_batch = self._prepare_moving_batch(moving_base, warp_params)
                batch_result = self._search_batch(reference, moving_batch, warp_params)

                batch_top = self._top_k_detail(batch_result, top_k)
                if global_best is None:
                    global_best = batch_top
                else:
                    global_best = self._top_k_detail(
                        self._merge_details(global_best, batch_top),
                        top_k,
                    )

                if tracker is not None:
                    tracker.update(iterator.stats)

                if callback is not None:
                    callback(
                        {
                            "batch_result": batch_result,
                            "global_best": global_best,
                            "stats": iterator.stats,
                            "warp_params": warp_params,
                        }
                    )

            if tracker is not None:
                tracker.finish()

        if global_best is None:
            raise RuntimeError("No search results produced.")

        return global_best

    def forward(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
        **kwargs,
    ) -> ImageDetail:
        return self.search(reference, moving, **kwargs)

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: Optional[bool] = None,
        backend: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs,
    ) -> "ExhaustiveWarpSearch":
        return compile_search(
            self,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            **kwargs,
        )


def _get_compile_backend_for_device(device: torch.device) -> Optional[str]:
    #TODO(wlr): Currently the full pipeline is SLOWER with compilation. idk why. Stages (like warp) are faster in isolation.
    # Need to profile more to see where the bottlenecks are.
    
    if device.type == "mps":
        return "aot_eager" # Torch MPS backend is not stable yet for full graph compilation
    return None


def compile_search(
    search: "ExhaustiveWarpSearch",
    *,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    **kwargs,
) -> "ExhaustiveWarpSearch":
    if backend is None:
        backend = _get_compile_backend_for_device(search.device)

    compile_kwargs: Dict[str, Any] = {
        "fullgraph": fullgraph,
    }
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic
    if backend is not None:
        compile_kwargs["backend"] = backend
    if mode is not None:
        compile_kwargs["mode"] = mode
    compile_kwargs.update(kwargs)

    return torch.compile(search, **compile_kwargs)
