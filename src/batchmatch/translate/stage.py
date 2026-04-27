from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from batchmatch.base.pipeline import Stage, StageRegistry
from batchmatch.base.tensordicts import ImageDetail, TranslationResults, NestedKey
from batchmatch.helpers.tensor import expand_mask_to_image

from .utility import (
    _extract_peak_from_surface,
    _extract_peak_from_unshifted_surface,
    _fft2d,
    _fractional_overlap_threshold,
    _normalize_by_overlap,
    _rcross_correlation_surface,
    _rfft2d,
)
from .cc import _masked_ncc_surface
from .pc import (
    _phase_correlation_surface,
    _phase_correlation_surface_unshifted,
    _rphase_correlation_surface,
    _rphase_correlation_surface_unshifted,
)
from .ngf import _generalized_ngf_surface, _normalized_gradient_fields_surface

Tensor = torch.Tensor

translation_registry = StageRegistry("translation")

class BufferShapeType(Enum):
    # Real FFT output: [B, C, H, W // 2 + 1], complex64
    RFFT_IMAGE = "rfft_image"

    # Full FFT output: [B, C, H, W], complex64/complex128
    FFT_COMPLEX = "fft_complex"

    # Real surface output: [B, H, W], float
    SURFACE = "surface"

#TODO(wlr): Implement non-FFT shiffted versions of all metrics, this will avoid an extra copy / memory movement stage
#TODO(wlr): Implement subpixel peak localization / estimation

@dataclass(frozen=True)
class MovingBufferSpec:
    key: NestedKey
    shape_type: BufferShapeType
    channels: int | None = None
    complex_dtype: bool = False

    def compute_shape(
        self, batch_size: int, image_channels: int, height: int, width: int
    ) -> tuple[int, ...]:
        C = self.channels if self.channels is not None else image_channels

        if self.shape_type == BufferShapeType.RFFT_IMAGE:
            return (batch_size, C, height, width // 2 + 1)
        elif self.shape_type == BufferShapeType.FFT_COMPLEX:
            return (batch_size, C, height, width)
        elif self.shape_type == BufferShapeType.SURFACE:
            return (batch_size, height, width)
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")

    def get_dtype(self, base_dtype: torch.dtype) -> torch.dtype:
        if self.complex_dtype:
            if base_dtype.is_complex:
                return base_dtype
            if base_dtype == torch.float64:
                return torch.complex128
            return torch.complex64
        return base_dtype


class TranslationSearchStage(Stage, ABC):
    requires_gradients: bool = False
    requires_complex_gradients: bool = False
    requires_masks: bool = False
    requires_boxes: bool = False
    requires_quads: bool = False
    save_surface: bool = True
    skip_fftshift: bool = False

    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.TRANSLATION.ROOT})

    _auto_validate: bool = False
    _auto_invalidate: bool = False

    def __init__(
        self,
        *,
        overlap_fraction: float | None = None,
        save_surface: bool = True,
    ):
        super().__init__()
        self.overlap_fraction = overlap_fraction
        self.save_surface = bool(save_surface)

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        if image.get(ImageDetail.Keys.CACHE.ROOT, default=None) is None:
            pass
        return image

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return []

    @abstractmethod
    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        ...

    def forward(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
    ) -> ImageDetail:
        for key in self.requires:
            if key not in reference:
                raise KeyError(f"{type(self).__name__} missing required key in reference: {key}")
            if key not in moving:
                raise KeyError(f"{type(self).__name__} missing required key in moving: {key}")

        surface = self.compute_surface(reference, moving)
        B, H, W = surface.shape

        if getattr(self, "skip_fftshift", False):
            ty, tx, scores, peak_y, peak_x = _extract_peak_from_unshifted_surface(surface, H, W)
        else:
            ty, tx, scores, peak_y, peak_x = _extract_peak_from_surface(surface, H, W)

        device = surface.device
        box = reference.get(ImageDetail.Keys.DOMAIN.BOX, default=None)
        if box is not None and box.numel() >= 4:
            if box.ndim == 3:
                b = box[0, 0]
            else:
                b = box[0]
            content_w = float((b[2] - b[0]).item())
            content_h = float((b[3] - b[1]).item())
            search_h = torch.full((B,), content_h, device=device, dtype=torch.float32)
            search_w = torch.full((B,), content_w, device=device, dtype=torch.float32)
        else:
            ref_H, ref_W = reference.H, reference.W
            search_h = torch.full((B,), float(ref_H), device=device, dtype=torch.float32)
            search_w = torch.full((B,), float(ref_W), device=device, dtype=torch.float32)

        result = TranslationResults.from_components(
            x=tx,
            y=ty,
            score=scores,
            surface=surface if self.save_surface else None,
            search_h=search_h,
            search_w=search_w,
        )

        moving.set(ImageDetail.Keys.TRANSLATION.ROOT, result)
        return moving

    def __call__(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
    ) -> ImageDetail:
        return self.forward(reference, moving)


@translation_registry.register("cc", "cross_correlation")
class CrossCorrelationStage(TranslationSearchStage):
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.IMAGE,
        ImageDetail.Keys.DOMAIN.MASK,
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(
        self,
        *,
        overlap_fraction: float | None = None,
        mean_centered: bool = False,
        **kwargs,
    ):
        super().__init__(overlap_fraction=overlap_fraction, **kwargs)
        self.overlap_fraction = overlap_fraction
        self.mean_centered = bool(mean_centered)
        self.requires_masks = mean_centered or (overlap_fraction is not None)

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return [
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_IMAGE,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            ),
        ]

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        ref = image.get(ImageDetail.Keys.IMAGE)
        ref_mask = image.get(ImageDetail.Keys.DOMAIN.MASK)
        ref_window = image.get(ImageDetail.Keys.DOMAIN.WINDOW)

        if self.mean_centered:
            ref_sum = (ref * ref_mask).sum(dim=(-2, -1), keepdim=True)
            ref_denom = torch.clamp(ref_mask.sum(dim=(-2, -1), keepdim=True), min=1.0)
            ref = (ref - ref_sum / ref_denom) * ref_mask
        else:
            ref = ref * ref_mask

        image.set(ImageDetail.Keys.CACHE.FFT, _rfft2d(ref))
        if self.overlap_fraction is not None:
            image.set(ImageDetail.Keys.CACHE.FFT_MASK, _rfft2d(ref_window))

        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        ref = reference.get(ImageDetail.Keys.IMAGE)
        mov = moving.get(ImageDetail.Keys.IMAGE)
        ref_mask = reference.get(ImageDetail.Keys.DOMAIN.MASK)
        mov_mask = moving.get(ImageDetail.Keys.DOMAIN.MASK)
        ref_window = reference.get(ImageDetail.Keys.DOMAIN.WINDOW)
        mov_window = moving.get(ImageDetail.Keys.DOMAIN.WINDOW)

        if self.mean_centered:
            ref_sum = (ref * ref_mask).sum(dim=(-2, -1), keepdim=True)
            ref_denom = torch.clamp(ref_mask.sum(dim=(-2, -1), keepdim=True), min=1.0)
            ref = (ref - ref_sum / ref_denom) * ref_mask

            mov_sum = (mov * mov_mask).sum(dim=(-2, -1), keepdim=True)
            mov_denom = torch.clamp(mov_mask.sum(dim=(-2, -1), keepdim=True), min=1.0)
            mov = (mov - mov_sum / mov_denom) * mov_mask

        ref = ref * ref_window
        mov = mov * mov_window

        F_ref = reference.get(ImageDetail.Keys.CACHE.FFT, default=None)
        F_mov_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_IMAGE, default=None)
        cc = _rcross_correlation_surface(
            ref,
            mov,
            F_ref=F_ref,
            F_mov_out=F_mov_out,
        )
        if self.overlap_fraction is None:
            return cc

        min_area = _fractional_overlap_threshold(mov_mask, self.overlap_fraction)
        F_ref_mask = reference.get(ImageDetail.Keys.CACHE.FFT_MASK, default=None)
        return _normalize_by_overlap(
            cc, ref_window, mov_window, min_area=min_area, F_ref_mask=F_ref_mask
        )


@translation_registry.register("mean_cc", "mean_cross_correlation")
class MeanCrossCorrelationStage(CrossCorrelationStage):
    def __init__(self, *, mean_centered: bool = True, **kwargs):
        super().__init__(mean_centered=mean_centered, **kwargs)


@translation_registry.register("gcc", "gradient_cc", "gradient_cross_correlation")
class GradientCrossCorrelationStage(TranslationSearchStage):
    requires_gradients: bool = True
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.DOMAIN.MASK,
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(
        self,
        *,
        overlap_fraction: float | None = 0.1,
        **kwargs,
    ):
        super().__init__(overlap_fraction=overlap_fraction, **kwargs)
        self.overlap_fraction = overlap_fraction

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return [
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_IMAGE,
                shape_type=BufferShapeType.RFFT_IMAGE,
                channels=2,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            ),
        ]

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        ref_mask = image.get(ImageDetail.Keys.DOMAIN.MASK)
        ref_window = image.get(ImageDetail.Keys.DOMAIN.WINDOW)
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)

        ref = torch.cat([gx * ref_mask * ref_window, gy * ref_mask * ref_window], dim=1)
        F_ref = _rfft2d(ref)
        image.set(ImageDetail.Keys.CACHE.FFT, F_ref)
        
        if self.overlap_fraction is not None:
            F_mask = _rfft2d(ref_window)
            image.set(ImageDetail.Keys.CACHE.FFT_MASK, F_mask)
        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        ref_mask = reference.get(ImageDetail.Keys.DOMAIN.MASK)
        mov_mask = moving.get(ImageDetail.Keys.DOMAIN.MASK)
        ref_window = reference.get(ImageDetail.Keys.DOMAIN.WINDOW)
        mov_window = moving.get(ImageDetail.Keys.DOMAIN.WINDOW)

        ref = torch.cat([reference.gx * ref_window, reference.gy * ref_window], dim=1)
        mov = torch.cat([moving.gx * mov_window, moving.gy * mov_window], dim=1)
        
        F_ref = reference.get(ImageDetail.Keys.CACHE.FFT, default=None)
        F_mov_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_IMAGE, default=None)

        cc = _rcross_correlation_surface(
            ref,
            mov,
            F_ref=F_ref,
            F_mov_out=F_mov_out,
        )
        if self.overlap_fraction is None:
            return cc

        min_area = _fractional_overlap_threshold(mov_mask, self.overlap_fraction)
        F_ref_mask = reference.get(ImageDetail.Keys.CACHE.FFT_MASK, default=None)
        return _normalize_by_overlap(
            cc, ref_window, mov_window, min_area=min_area, F_ref_mask=F_ref_mask
        )


@translation_registry.register("ncc", "masked_ncc", "normalized_cc")
class MaskedNCCStage(TranslationSearchStage):
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.IMAGE,
        ImageDetail.Keys.DOMAIN.MASK,
    })

    def __init__(
        self,
        *,
        overlap_fraction: float | None = 0.5,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(overlap_fraction=overlap_fraction, **kwargs)
        self.overlap_fraction = overlap_fraction
        self.eps = eps

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return [
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_MASK,
                shape_type=BufferShapeType.RFFT_IMAGE,
                channels=None,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_MASKED,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_MASKED_SQ,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            ),
        ]

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        ref = image.get(ImageDetail.Keys.IMAGE)
        ref_mask = image.get(ImageDetail.Keys.DOMAIN.MASK)
        ref_weight = image.get(ImageDetail.Keys.DOMAIN.WINDOW, default=ref_mask)
        ref_weight = expand_mask_to_image(ref_weight, ref)

        Rm = ref * ref_weight
        Rm2 = ref * ref * ref_weight

        F_A = _rfft2d(ref_weight)
        F_Rm = _rfft2d(Rm)
        F_Rm2 = _rfft2d(Rm2)

        image.set(ImageDetail.Keys.CACHE.FFT_MASK, F_A)
        image.set(ImageDetail.Keys.CACHE.FFT_MASKED, F_Rm)
        image.set(ImageDetail.Keys.CACHE.FFT_MASKED_SQ, F_Rm2)

        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:

        ref = reference.get(ImageDetail.Keys.IMAGE)
        mov = moving.get(ImageDetail.Keys.IMAGE)
        ref_mask = reference.get(ImageDetail.Keys.DOMAIN.MASK)
        mov_mask = moving.get(ImageDetail.Keys.DOMAIN.MASK)
        # Use WINDOW for NCC weighting if available, otherwise fall back to MASK
        ref_weight = reference.get(ImageDetail.Keys.DOMAIN.WINDOW, default=ref_mask)
        mov_weight = moving.get(ImageDetail.Keys.DOMAIN.WINDOW, default=mov_mask)
        ref_weight = expand_mask_to_image(ref_weight, ref)
        mov_weight = expand_mask_to_image(mov_weight, mov)


        if self.overlap_fraction is None:
            min_count = 0.0
        else:
            min_count = _fractional_overlap_threshold(
                mov_weight, self.overlap_fraction
            )

        F_A = reference.get(ImageDetail.Keys.CACHE.FFT_MASK, default=None)
        F_Rm = reference.get(ImageDetail.Keys.CACHE.FFT_MASKED, default=None)
        F_Rm2 = reference.get(ImageDetail.Keys.CACHE.FFT_MASKED_SQ, default=None)

        F_B_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_MASK, default=None)
        F_Im_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_MASKED, default=None)
        F_Im2_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_MASKED_SQ, default=None)
        surface_out = moving.get(ImageDetail.Keys.CACHE.SURFACE_BUFFER, default=None)

        return _masked_ncc_surface(
            ref,
            mov,
            ref_weight,
            mov_weight,
            eps=self.eps,
            min_count=min_count,
            F_A=F_A,
            F_Rm=F_Rm,
            F_Rm2=F_Rm2,
            F_B_out=F_B_out,
            F_Im_out=F_Im_out,
            F_Im2_out=F_Im2_out,
            surface_out=surface_out,
        )

@translation_registry.register("pc", "phase_correlation")
class PhaseCorrelationStage(TranslationSearchStage):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.IMAGE, ImageDetail.Keys.DOMAIN.WINDOW})

    def __init__(
        self,
        *,
        eps: float = 1e-8,
        skip_fftshift: bool = True,
        p: float = 1.0,
        q: float = 1.0,
        overlap_fraction: float | None = 0.99, #only enabled for p/q != 1.0
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.skip_fftshift = skip_fftshift
        self.p = p
        self.q = q

        #TODO(wlr): Implement support for p,q != 1.0 with overlap mask scaling

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return [
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.CROSS_POWER,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            ),
        ]

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        ref = image.get(ImageDetail.Keys.IMAGE)
        ref_window = image.get(ImageDetail.Keys.DOMAIN.WINDOW, default=None)
        F_ref = _rfft2d(ref * ref_window)
        image.set(ImageDetail.Keys.CACHE.FFT, F_ref)

        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        F_ref = reference.get(ImageDetail.Keys.CACHE.FFT, default=None)

        ref = reference.get(ImageDetail.Keys.IMAGE)
        mov = moving.get(ImageDetail.Keys.IMAGE)
        ref_window = reference.get(ImageDetail.Keys.DOMAIN.WINDOW, default=None)
        mov_window = moving.get(ImageDetail.Keys.DOMAIN.WINDOW, default=None)
        if F_ref is None:
            ref = ref * ref_window
        mov = mov * mov_window

        #Load cached buffers for intermediates
        F_mov_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV, default=None)
        cross_power_out = moving.get(ImageDetail.Keys.CACHE.CROSS_POWER, default=None)
        surface_out = moving.get(ImageDetail.Keys.CACHE.SURFACE_BUFFER, default=None)

        if self.skip_fftshift:
            return _rphase_correlation_surface_unshifted(
                ref, 
                mov,
                eps=self.eps,
                F_ref=F_ref,
                F_mov_out=F_mov_out,
                cross_power_out=cross_power_out,
                surface_out=surface_out,
            )
        return _rphase_correlation_surface(
            ref,
            mov,
            eps=self.eps,
            F_ref=F_ref,
            F_mov_out=F_mov_out,
            cross_power_out=cross_power_out,
            surface_out=surface_out,
        )

@translation_registry.register("gpc", "gradient_phase_correlation")
class GradientPhaseCorrelationStage(TranslationSearchStage):
    requires_gradients: bool = True
    requires_complex_gradients: bool = True
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.I,
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(
        self,
        *,
        eps: float = 1e-8,
        skip_fftshift: bool = True,
        p: float = 1.0,
        q: float = 1.0,
        overlap_fraction: float | None = 0.99, #only enabled for p/q != 1.0
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.skip_fftshift = skip_fftshift
        self.p = p
        self.q = q

        #TODO(wlr): Implement support for p,q != 1.0 with overlap mask scaling

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return [
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV,
                shape_type=BufferShapeType.FFT_COMPLEX,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.CROSS_POWER,
                shape_type=BufferShapeType.FFT_COMPLEX,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            ),
        ]

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        ref_gi = image.get(ImageDetail.Keys.GRAD.I)
        ref_window = image.get(ImageDetail.Keys.DOMAIN.WINDOW)
        ref_gi = ref_gi * ref_window
        F_ref = _fft2d(ref_gi)
        image.set(ImageDetail.Keys.CACHE.FFT, F_ref)
        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        F_ref = reference.get(ImageDetail.Keys.CACHE.FFT, default=None)

        ref_gi = reference.get(ImageDetail.Keys.GRAD.I)
        mov_gi = moving.get(ImageDetail.Keys.GRAD.I)
        ref_window = reference.get(ImageDetail.Keys.DOMAIN.WINDOW)
        mov_window = moving.get(ImageDetail.Keys.DOMAIN.WINDOW)

        if F_ref is None:
            ref_gi = ref_gi * ref_window
        mov_gi = mov_gi * mov_window

        # Load cached buffers for intermediates
        F_mov_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV, default=None)
        cross_power_out = moving.get(ImageDetail.Keys.CACHE.CROSS_POWER, default=None)
        surface_out = moving.get(ImageDetail.Keys.CACHE.SURFACE_BUFFER, default=None)

        if self.skip_fftshift:
            return _phase_correlation_surface_unshifted(
                ref_gi,
                mov_gi,
                eps=self.eps,
                p=self.p,
                q=self.q,
                F_ref=F_ref,
                F_mov_out=F_mov_out,
                cross_power_out=cross_power_out,
                surface_out=surface_out,
            )
        return _phase_correlation_surface(
            ref_gi,
            mov_gi,
            eps=self.eps,
            F_ref=F_ref,
            p=self.p,
            q=self.q,
            F_mov_out=F_mov_out,
            cross_power_out=cross_power_out,
            surface_out=surface_out,
        )

@translation_registry.register("ngf", "normalized_gradient_fields")
class NormalizedGradientFieldsStage(TranslationSearchStage):
    requires_gradients: bool = True
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.DOMAIN.MASK,
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(
        self,
        *,
        overlap_fraction: float | None = 0.5,
        weight_by_gradient_norm: bool = False,
        gradient_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(overlap_fraction=overlap_fraction, **kwargs)
        self.overlap_fraction = overlap_fraction
        self.weight_by_gradient_norm = weight_by_gradient_norm
        self.gradient_norm_eps = gradient_norm_eps
        if overlap_fraction is not None:
            self.requires_masks = True

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        return [
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_GX2,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_GY2,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.FFT_MOV_CROSS,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            ),
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            ),
        ]

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        ref_gx = image.get(ImageDetail.Keys.GRAD.X)
        ref_gy = image.get(ImageDetail.Keys.GRAD.Y)
        ref_mask = image.get(ImageDetail.Keys.DOMAIN.MASK)
        ref_window = image.get(ImageDetail.Keys.DOMAIN.WINDOW)
        
        ref_gx = image.gx 
        ref_gy = image.gy 

        if self.weight_by_gradient_norm:
            ref_norm = torch.sqrt(ref_gx * ref_gx + ref_gy * ref_gy + self.gradient_norm_eps)
            ref_gx = ref_gx * ref_norm
            ref_gy = ref_gy * ref_norm

        ref_gx_2 = ref_gx * ref_gx * ref_window
        ref_gy_2 = ref_gy * ref_gy * ref_window
        ref_cross = ref_gy * ref_gx * ref_window
        
        F_ref_gx_2 = _rfft2d(ref_gx_2)
        F_ref_gy_2 = _rfft2d(ref_gy_2)
        F_ref_cross = _rfft2d(ref_cross)
        
        image.set(ImageDetail.Keys.CACHE.FFT_GX2, F_ref_gx_2)
        image.set(ImageDetail.Keys.CACHE.FFT_GY2, F_ref_gy_2)
        image.set(ImageDetail.Keys.CACHE.FFT_CROSS, F_ref_cross)
        
        if self.overlap_fraction is not None:
            F_mask = _rfft2d(ref_window)
            image.set(ImageDetail.Keys.CACHE.FFT_MASK, F_mask)
        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        ref_gx = reference.get(ImageDetail.Keys.GRAD.X)
        ref_gy = reference.get(ImageDetail.Keys.GRAD.Y)
        mov_gx = moving.get(ImageDetail.Keys.GRAD.X)
        mov_gy = moving.get(ImageDetail.Keys.GRAD.Y)

        ref_window = reference.get(ImageDetail.Keys.DOMAIN.WINDOW)
        mov_window = moving.get(ImageDetail.Keys.DOMAIN.WINDOW)

        if self.weight_by_gradient_norm:
            ref_norm = torch.sqrt(ref_gx * ref_gx + ref_gy * ref_gy + self.gradient_norm_eps)
            mov_norm = torch.sqrt(mov_gx * mov_gx + mov_gy * mov_gy + self.gradient_norm_eps)
            ref_gx = ref_gx * ref_norm
            ref_gy = ref_gy * ref_norm
            mov_gx = mov_gx * mov_norm
            mov_gy = mov_gy * mov_norm

        # Check for cached components
        F_ref_gx_2 = reference.get(ImageDetail.Keys.CACHE.FFT_GX2, default=None)
        F_ref_gy_2 = reference.get(ImageDetail.Keys.CACHE.FFT_GY2, default=None)
        F_ref_cross = reference.get(ImageDetail.Keys.CACHE.FFT_CROSS, default=None)
        F_mov_gx_2_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_GX2, default=None)
        F_mov_gy_2_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_GY2, default=None)
        F_mov_cross_out = moving.get(ImageDetail.Keys.CACHE.FFT_MOV_CROSS, default=None)

        cc = _normalized_gradient_fields_surface(
            ref_gx,
            ref_gy,
            ref_window,
            mov_gx,
            mov_gy,
            mov_window,
            F_ref_gx_2=F_ref_gx_2,
            F_ref_gy_2=F_ref_gy_2,
            F_ref_cross=F_ref_cross,
            F_mov_gx_2_out=F_mov_gx_2_out,
            F_mov_gy_2_out=F_mov_gy_2_out,
            F_mov_cross_out=F_mov_cross_out,
        )

        if self.overlap_fraction is None:
            return cc

        min_area = _fractional_overlap_threshold(mov_window, self.overlap_fraction)

        F_ref_mask = reference.get(ImageDetail.Keys.CACHE.FFT_MASK, default=None)
        return _normalize_by_overlap(
            cc,
            ref_window,
            mov_window,
            min_area=min_area,
            F_ref_mask=F_ref_mask,
        )


@translation_registry.register("gngf", "generalized_ngf")
class GeneralizedNGFStage(TranslationSearchStage):

    requires_gradients: bool = True
    requires: frozenset[NestedKey] = frozenset({
        ImageDetail.Keys.GRAD.X,
        ImageDetail.Keys.GRAD.Y,
        ImageDetail.Keys.DOMAIN.MASK,
        ImageDetail.Keys.DOMAIN.WINDOW,
    })

    def __init__(
        self,
        *,
        p: int = 2,
        overlap_fraction: float | None = 0.99,
        **kwargs,
    ):
        super().__init__(overlap_fraction=overlap_fraction, **kwargs)
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        self.p = int(p)
        self.overlap_fraction = overlap_fraction
        if overlap_fraction is not None:
            self.requires_masks = True

        self._ref_fft_keys = [
            f"cache_fft_gngf_p{self.p}_term{i}" for i in range(self.p + 1)
        ]
        self._mov_fft_keys = [
            f"cache_fft_mov_gngf_p{self.p}_term{i}" for i in range(self.p + 1)
        ]

    @property
    def moving_buffer_specs(self) -> list[MovingBufferSpec]:
        buffers = [
            MovingBufferSpec(
                key=key,
                shape_type=BufferShapeType.RFFT_IMAGE,
                complex_dtype=True,
            )
            for key in self._mov_fft_keys
        ]
        buffers.append(
            MovingBufferSpec(
                key=ImageDetail.Keys.CACHE.SURFACE_BUFFER,
                shape_type=BufferShapeType.SURFACE,
            )
        )
        return buffers

    def prepare_cache(self, image: ImageDetail) -> ImageDetail:
        """Cache FFT components for generalized NGF."""
        ref_gx = image.get(ImageDetail.Keys.GRAD.X)
        ref_gy = image.get(ImageDetail.Keys.GRAD.Y)
        ref_window = image.get(ImageDetail.Keys.DOMAIN.WINDOW)

        for i in range(self.p + 1):
            exp_x = self.p - i
            exp_y = i
            ref_term = (ref_gx ** exp_x) * (ref_gy ** exp_y) * ref_window
            image.set(self._ref_fft_keys[i], _rfft2d(ref_term))

        if self.overlap_fraction is not None:
            F_mask = _rfft2d(ref_window)
            image.set(ImageDetail.Keys.CACHE.FFT_MASK, F_mask)
        return image

    def compute_surface(self, reference: ImageDetail, moving: ImageDetail) -> Tensor:
        ref_gx = reference.get(ImageDetail.Keys.GRAD.X)
        ref_gy = reference.get(ImageDetail.Keys.GRAD.Y)
        mov_gx = moving.get(ImageDetail.Keys.GRAD.X)
        mov_gy = moving.get(ImageDetail.Keys.GRAD.Y)

        ref_window = reference.get(ImageDetail.Keys.DOMAIN.WINDOW)
        mov_window = moving.get(ImageDetail.Keys.DOMAIN.WINDOW)

        F_ref_terms = [
            reference.get(key, default=None) for key in self._ref_fft_keys
        ]
        F_mov_terms_out = [
            moving.get(key, default=None) for key in self._mov_fft_keys
        ]

        cc = _generalized_ngf_surface(
            ref_gx,
            ref_gy,
            ref_window,
            mov_gx,
            mov_gy,
            mov_window,
            p=self.p,
            F_ref_terms=F_ref_terms,
            F_mov_terms_out=F_mov_terms_out,
        )

        if self.overlap_fraction is None:
            return cc

        min_area = _fractional_overlap_threshold(mov_window, self.overlap_fraction)
        F_ref_mask = reference.get(ImageDetail.Keys.CACHE.FFT_MASK, default=None)
        return _normalize_by_overlap(
            cc,
            ref_window,
            mov_window,
            min_area=min_area,
            F_ref_mask=F_ref_mask,
        )
