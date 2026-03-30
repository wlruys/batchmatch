from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
from tensordict import TensorDict

from batchmatch.helpers.tensor import to_bchw

Tensor = torch.Tensor
NestedKey = Union[str, Tuple[str, ...]]


def _move_tensor(
    t: Optional[Tensor],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
) -> Optional[Tensor]:
    if t is None:
        return None
    if device is not None:
        t = t.to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def _normalize_keys(keys: Optional[Union[NestedKey, Sequence[NestedKey]]]) -> list[NestedKey]:
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, tuple) and all(isinstance(k, str) for k in keys):
        return [keys]
    return list(keys)


class CacheTD(TensorDict):
    _STALE_KEY = "_stale_keys"

    def mark_stale(self, key: NestedKey) -> None:
        stale = self._get_stale_set()
        stale.add(key if isinstance(key, str) else tuple(key))

    def mark_fresh(self, key: NestedKey) -> None:
        stale = self._get_stale_set()
        key_tuple = key if isinstance(key, str) else tuple(key)
        stale.discard(key_tuple)

    def is_stale(self, key: NestedKey) -> bool:
        key_tuple = key if isinstance(key, str) else tuple(key)
        return key_tuple in self._get_stale_set()

    def clear_stale(self, key: NestedKey) -> None:
        self.mark_fresh(key)

    def drop_stale(self) -> None:
        stale = self._get_stale_set()
        for key in list(stale):
            if key in self:
                self.del_(key)
        stale.clear()

    def _get_stale_set(self) -> set:
        if not hasattr(self, self._STALE_KEY):
            setattr(self, self._STALE_KEY, set())
        return getattr(self, self._STALE_KEY)


class WarpParams(CacheTD):
    """
    TensorDict containing affine transformation parameters.

    Keys are simple strings, available under WarpParams.Keys.
    Can be stored directly in ImageDetail under the "warp" key.

    Args:
        args: Positional arguments forwarded to TensorDict.
        **kwargs: Keyword arguments forwarded to TensorDict.
    """

    class Keys:
        ANGLE = "angle"
        SCALE_X = "scale_x"
        SCALE_Y = "scale_y"

        SHEAR_X = "shear_x"
        SHEAR_Y = "shear_y"
        TX = "tx"
        TY = "ty"
        CENTER_X = "center_x"
        CENTER_Y = "center_y"

        GRID = "grid"
        M_FWD = "M_fwd"
        M_INV = "M_inv"
        PIXEL_COORDS = "pixel_coords"  # Deprecated: use COORD_XS and COORD_YS
        COORD_XS = "coord_xs"  # Cached x coordinates (1, 1, W)
        COORD_YS = "coord_ys"  # Cached y coordinates (1, H, 1)
        SAMPLE = "sample"

        PARAMS = (ANGLE, SCALE_X, SCALE_Y, SHEAR_X, SHEAR_Y, TX, TY)
        CENTER = (CENTER_X, CENTER_Y)
        COMPUTED = (GRID, M_FWD, M_INV, PIXEL_COORDS, COORD_XS, COORD_YS, SAMPLE)

    @classmethod
    def identity(
        cls,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WarpParams":
        if dtype is None:
            dtype = torch.float32

        data = {
            cls.Keys.ANGLE: torch.zeros(batch_size, device=device, dtype=dtype),
            cls.Keys.SCALE_X: torch.ones(batch_size, device=device, dtype=dtype),
            cls.Keys.SCALE_Y: torch.ones(batch_size, device=device, dtype=dtype),
            cls.Keys.SHEAR_X: torch.zeros(batch_size, device=device, dtype=dtype),
            cls.Keys.SHEAR_Y: torch.zeros(batch_size, device=device, dtype=dtype),
            cls.Keys.TX: torch.zeros(batch_size, device=device, dtype=dtype),
            cls.Keys.TY: torch.zeros(batch_size, device=device, dtype=dtype),
        }
        return cls(data, batch_size=[batch_size])
    
    @classmethod
    def from_components(
        cls, 
        angle: Optional[Tensor] = None,
        scale_x: Optional[Tensor] = None,
        scale_y: Optional[Tensor] = None,
        shear_x: Optional[Tensor] = None,
        shear_y: Optional[Tensor] = None,
        tx: Optional[Tensor] = None,
        ty: Optional[Tensor] = None, 
        center_x: Optional[Tensor] = None,
        center_y: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "WarpParams":
        """
        Build warp parameters from component tensors.

        Args:
            angle: Rotation angles [B].
            scale_x: Scale in x [B].
            scale_y: Scale in y [B].
            shear_x: Shear in x [B].
            shear_y: Shear in y [B].
            tx: Translation in x [B].
            ty: Translation in y [B].
            center_x: Center of rotation x [B].
            center_y: Center of rotation y [B].
            device: Device for the output tensors.
            dtype: Data type for the output tensors.

        Returns:
            WarpParams containing the provided components.

        Raises:
            ValueError: If no component tensors are provided.
        """
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = torch.device("cpu")

        def prepare(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None:
                return None
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=device, dtype=dtype)
            if t.ndim == 0:
                t = t.unsqueeze(0)
            return _move_tensor(t, device, dtype)

        angle = prepare(angle)
        scale_x = prepare(scale_x)
        scale_y = prepare(scale_y)
        shear_x = prepare(shear_x)
        shear_y = prepare(shear_y)
        tx = prepare(tx)
        ty = prepare(ty)
        center_x = prepare(center_x)
        center_y = prepare(center_y)
        
        provided_tensors = [t for t in [angle, scale_x, scale_y, shear_x, shear_y, tx, ty, center_x, center_y] if t is not None]
        if not provided_tensors:
            raise ValueError("At least one parameter tensor must be provided.")
        
        B = provided_tensors[0].shape[0]

        data = {
            cls.Keys.ANGLE: angle if angle is not None else torch.zeros(B, device=device, dtype=dtype),
            cls.Keys.SCALE_X: scale_x if scale_x is not None else torch.ones(B, device=device, dtype=dtype),
            cls.Keys.SCALE_Y: scale_y if scale_y is not None else torch.ones(B, device=device, dtype=dtype),
            cls.Keys.SHEAR_X: shear_x if shear_x is not None else torch.zeros(B, device=device, dtype=dtype),
            cls.Keys.SHEAR_Y: shear_y if shear_y is not None else torch.zeros(B, device=device, dtype=dtype),
            cls.Keys.TX: tx if tx is not None else torch.zeros(B, device=device, dtype=dtype),
            cls.Keys.TY: ty if ty is not None else torch.zeros(B, device=device, dtype=dtype),
        }
        
        if center_x is not None:
            data[cls.Keys.CENTER_X] = center_x
        if center_y is not None:
            data[cls.Keys.CENTER_Y] = center_y
            
        z = cls(data, batch_size=[B])
        return z
    
    @property
    def B(self) -> int:
        return self.get(self.Keys.ANGLE).shape[0]
    
    def _populate_center(self, H:int, W:int) -> None:
        B = self.get(self.Keys.ANGLE).shape[0]
        center_x = torch.full((B,), W / 2.0, device=self.device, dtype=self.get(self.Keys.ANGLE).dtype)
        center_y = torch.full((B,), H / 2.0, device=self.device, dtype=self.get(self.Keys.ANGLE).dtype)
        self.set(self.Keys.CENTER_X, center_x)
        self.set(self.Keys.CENTER_Y, center_y)

    def populate_center(self, image: ImageDetail) -> None:
        H = image.H
        W = image.W
        self._populate_center(H=H, W=W)

    def has_transform_params(self) -> bool:
        return all(k in self.keys() for k in self.Keys.PARAMS)

    def has_grid(self) -> bool:
        return self.Keys.GRID in self.keys()

    def has_matrices(self) -> bool:
        return self.Keys.M_FWD in self.keys() and self.Keys.M_INV in self.keys()
    
    @property
    def angle(self) -> Tensor:
        return self.get(self.Keys.ANGLE)
    
    @property
    def scale_x(self) -> Tensor:
        return self.get(self.Keys.SCALE_X)
    
    @property
    def scale_y(self) -> Tensor:
        return self.get(self.Keys.SCALE_Y)
    
    @property
    def tx(self) -> Tensor:
        return self.get(self.Keys.TX)
    
    @property
    def ty(self) -> Tensor:
        return self.get(self.Keys.TY)
    
    @property
    def shear_x(self) -> Tensor:
        return self.get(self.Keys.SHEAR_X)
    
    @property
    def shear_y(self) -> Tensor:
        return self.get(self.Keys.SHEAR_Y)
    
    @property
    def center_x(self) -> Tensor:
        return self.get(self.Keys.CENTER_X)
    
    @property
    def center_y(self) -> Tensor:
        return self.get(self.Keys.CENTER_Y)
    

class TranslationResults(CacheTD):
    class Keys:
        X = "x"
        Y = "y"
        SURFACE = "surface"
        SCORE = "score"
        MEAN_NONZERO_SURFACE = "mean_nonzero_surface"
        SEARCH_H_CONTENT = "search_h_content"
        SEARCH_W_CONTENT = "search_w_content"
        SEARCH_H = "search_h"
        SEARCH_W = "search_w"

    def has_surface(self) -> bool:
        return self.Keys.SURFACE in self.keys()
    
    @property
    def tx(self) -> Tensor:
        return self.get(self.Keys.X)
    
    @property
    def ty(self) -> Tensor:
        return self.get(self.Keys.Y)
    
    @property
    def score(self) -> Tensor:
        return self.get(self.Keys.SCORE)
    
    @property
    def surface(self) -> Optional[Tensor]:
        return self.get(self.Keys.SURFACE, default=None)
    
    @property
    def search_dimensions(self) -> Optional[Tuple[Tensor, Tensor]]:
        search_h = self.get(self.Keys.SEARCH_H, default=None)
        search_w = self.get(self.Keys.SEARCH_W, default=None)
        if search_h is None or search_w is None:
            return None
        return (search_h, search_w)
    
    @property
    def content_dimensions(self) -> Optional[Tuple[Tensor, Tensor]]:
        search_h = self.get(self.Keys.SEARCH_H_CONTENT, default=None)
        search_w = self.get(self.Keys.SEARCH_W_CONTENT, default=None)
        if search_h is None or search_w is None:
            return None
        return (search_h, search_w)
    
    @classmethod 
    def from_components(
        cls,
        x: Tensor,
        y: Tensor,
        score: Tensor,
        surface: Optional[Tensor] = None,
        search_h: Optional[Tensor] = None,
        search_w: Optional[Tensor] = None,
        device : Optional[torch.device] = None,
        dtype : Optional[torch.dtype] = None,
    ) -> "TranslationResults":
        """
        Build translation results from component tensors.

        Args:
            x: Translation x tensor [B].
            y: Translation y tensor [B].
            score: Score tensor [B].
            surface: Optional surface tensor [B, H, W].
            search_h: Optional search height tensor [B].
            search_w: Optional search width tensor [B].
            device: Device for the output tensors.
            dtype: Data type for the output tensors.

        Returns:
            TranslationResults containing the provided components.
        """
        x = _move_tensor(x, device, dtype)
        y = _move_tensor(y, device, dtype)
        score = _move_tensor(score, device, dtype)
        surface = _move_tensor(surface, device, dtype)
        search_h = _move_tensor(search_h, device, dtype)
        search_w = _move_tensor(search_w, device, dtype)

        data = {
            cls.Keys.X: x,
            cls.Keys.Y: y,
            cls.Keys.SCORE: score,
        }
        if surface is not None:
            data[cls.Keys.SURFACE] = surface
        if search_h is not None:
            data[cls.Keys.SEARCH_H] = search_h
        if search_w is not None:
            data[cls.Keys.SEARCH_W] = search_w
        B = x.shape[0]
        return cls(data, batch_size=[B])

class ImageDetail(CacheTD):
    class Keys:
        IMAGE: NestedKey = "image"

        class GRAD:
            ROOT: NestedKey = "grad"
            X: NestedKey = ("grad", "x")
            Y: NestedKey = ("grad", "y")
            I: NestedKey = ("grad", "i")
            NORM: NestedKey = ("grad", "norm")
            ETA: NestedKey = ("grad", "eta")
        
        class CACHE:
            ROOT: NestedKey = "cache"

            # Reference FFT cache keys 
            FFT: NestedKey = "cache_fft"
            FFT_MASK: NestedKey = "cache_fft_mask"
            FFT_MASKED: NestedKey = "cache_fft_masked"
            FFT_MASKED_SQ: NestedKey = "cache_fft_masked_sq"
            FFT_GX2: NestedKey = "cache_fft_gx2"
            FFT_GY2: NestedKey = "cache_fft_gy2"
            FFT_CROSS: NestedKey = "cache_fft_cross"

            # Moving FFT buffer reuse keys (pre-allocated for search optimization)
            # Phase correlation (pc) buffers
            FFT_MOV: NestedKey = "cache_fft_mov"
            CROSS_POWER: NestedKey = "cache_cross_power"
            SURFACE_BUFFER: NestedKey = "cache_surface_buffer"

            # Cross-correlation (cc, gcc) buffers
            FFT_MOV_IMAGE: NestedKey = "cache_fft_mov_image"

            # NCC buffers
            FFT_MOV_MASK: NestedKey = "cache_fft_mov_mask"
            FFT_MOV_MASKED: NestedKey = "cache_fft_mov_masked"
            FFT_MOV_MASKED_SQ: NestedKey = "cache_fft_mov_masked_sq"

            # NGF buffers
            FFT_MOV_GX2: NestedKey = "cache_fft_mov_gx2"
            FFT_MOV_GY2: NestedKey = "cache_fft_mov_gy2"
            FFT_MOV_CROSS: NestedKey = "cache_fft_mov_cross"

        class DOMAIN:
            ROOT: NestedKey = "domain"
            MASK: NestedKey = ("domain", "mask")
            WINDOW: NestedKey = ("domain", "window")
            BOX: NestedKey = ("domain", "box")
            QUAD: NestedKey = ("domain", "quad")

        class AUX:
            AUX: NestedKey = "aux"
            BOXES: NestedKey = ("aux", "boxes")
            QUADS: NestedKey = ("aux", "quads")
            POINTS: NestedKey = ("aux", "points")

        class TRANSLATION:
            ROOT: NestedKey = "translation"
            X: NestedKey = ("translation", TranslationResults.Keys.X)
            Y: NestedKey = ("translation", TranslationResults.Keys.Y)
            SURFACE: NestedKey = ("translation", TranslationResults.Keys.SURFACE)
            SCORE: NestedKey = ("translation", TranslationResults.Keys.SCORE)
            SEARCH_H: NestedKey = ("translation", TranslationResults.Keys.SEARCH_H)
            SEARCH_W: NestedKey = ("translation", TranslationResults.Keys.SEARCH_W)

        class WARP:
            ROOT: NestedKey = "warp"
            ANGLE: NestedKey = ("warp", WarpParams.Keys.ANGLE)
            SCALE_X: NestedKey = ("warp", WarpParams.Keys.SCALE_X)
            SCALE_Y: NestedKey = ("warp", WarpParams.Keys.SCALE_Y)
            SHEAR_X: NestedKey = ("warp", WarpParams.Keys.SHEAR_X)
            SHEAR_Y: NestedKey = ("warp", WarpParams.Keys.SHEAR_Y)
            TX: NestedKey = ("warp", WarpParams.Keys.TX)
            TY: NestedKey = ("warp", WarpParams.Keys.TY)
            GRID: NestedKey = ("warp", WarpParams.Keys.GRID)
            M_FWD: NestedKey = ("warp", WarpParams.Keys.M_FWD)
            M_INV: NestedKey = ("warp", WarpParams.Keys.M_INV)
            PIXEL_COORDS: NestedKey = ("warp", WarpParams.Keys.PIXEL_COORDS)
            COORD_XS: NestedKey = ("warp", WarpParams.Keys.COORD_XS)
            COORD_YS: NestedKey = ("warp", WarpParams.Keys.COORD_YS)
            SAMPLE: NestedKey = ("warp", WarpParams.Keys.SAMPLE)
            CX: NestedKey = ("warp", WarpParams.Keys.CENTER_X)
            CY: NestedKey = ("warp", WarpParams.Keys.CENTER_Y)

    @property
    def image(self) -> Tensor:
        return self.get(self.Keys.IMAGE)
    
    @property
    def mask(self) -> Optional[Tensor]:
        return self.get(self.Keys.DOMAIN.MASK, default=None)
    
    @property
    def box(self) -> Optional[Tensor]:
        return self.get(self.Keys.DOMAIN.BOX, default=None)
    
    @property
    def quad(self) -> Optional[Tensor]:
        return self.get(self.Keys.DOMAIN.QUAD, default=None)
    
    @property
    def gx(self) -> Optional[Tensor]:
        return self.get(self.Keys.GRAD.X, default=None)
    
    @property
    def gy(self) -> Optional[Tensor]:
        return self.get(self.Keys.GRAD.Y, default=None)
    
    @property
    def gi(self) -> Optional[Tensor]:
        gi = self.get(self.Keys.GRAD.I, default=None)
        if gi is not None:
            return gi
        gx = self.gx
        gy = self.gy
        if gx is not None and gy is not None:
            return torch.complex(gx, gy)
        return None
    
    @property
    def warp(self) -> Optional[WarpParams]:
        return self.get(self.Keys.WARP.ROOT, default=None)
    
    @property
    def translation_results(self) -> Optional[TranslationResults]:
        return self.get(self.Keys.TRANSLATION.ROOT, default=None)
    
    @property
    def grad_i(self):
        return self.gi
    
    @property
    def H(self) -> int:
        return self.image.shape[-2]
    
    @property
    def W(self) -> int:
        return self.image.shape[-1]
    
    @property
    def C(self) -> int:
        return self.image.shape[1]
    
    @property
    def B(self) -> int:
        return self.image.shape[0]
    
    def apply_mask(self) -> None:
        mask = self.mask
        if mask is not None:
            img = self.image
            self.set(self.Keys.IMAGE, img * mask)

    @property
    def window(self) -> Optional[Tensor]:
        return self.get(self.Keys.DOMAIN.WINDOW, default=None)

    def window_image(self, set: bool = False) -> Tensor:
        window = self.get(self.Keys.DOMAIN.WINDOW)
        image = self.get(self.Keys.IMAGE)
        w_img = image * window
        if set:
            self.set(self.Keys.IMAGE, w_img)
        return w_img

    def window_gx(self, set: bool = False) -> Tensor:
        window = self.get(self.Keys.DOMAIN.WINDOW)
        gx = self.get(self.Keys.GRAD.X)
        w_gx = gx * window
        if set:
            self.set(self.Keys.GRAD.X, w_gx)
        return w_gx
    
    def window_gy(self, set: bool = False) -> Tensor:
        window = self.get(self.Keys.DOMAIN.WINDOW)
        gy = self.get(self.Keys.GRAD.Y)
        w_gy = gy * window
        if set:
            self.set(self.Keys.GRAD.Y, w_gy)
        return w_gy
    
    def window_gi(self, set: bool = False) -> Optional[Tensor]:
        window = self.get(self.Keys.DOMAIN.WINDOW)
        gi = self.gi
        if gi is None:
            return None
        w_gi = gi * window
        if set:
            self.set(self.Keys.GRAD.I, w_gi)
        return w_gi

    @property
    def tiff_meta(self) -> "object | None":
        """Optional TiffMeta attached during TIFF loading."""
        return getattr(self, "_tiff_meta", None)

    @tiff_meta.setter
    def tiff_meta(self, meta: "object | None") -> None:
        self._tiff_meta = meta

    @classmethod
    def from_image(cls, image: Tensor, *, tiff_meta: "object | None" = None) -> "ImageDetail":
        image = to_bchw(image)
        detail = cls({cls.Keys.IMAGE: image}, batch_size=image.shape[0:1])
        if tiff_meta is not None:
            detail.tiff_meta = tiff_meta
        return detail
    
    def add_warp_params(self,
        angle: Optional[Tensor] = None,
        scale_x: Optional[Tensor] = None,
        scale_y: Optional[Tensor] = None,
        shear_x: Optional[Tensor] = None,
        shear_y: Optional[Tensor] = None,
        tx: Optional[Tensor] = None,
        ty: Optional[Tensor] = None,
        ) -> None:
        """
        Add WarpParams to the ImageDetail under the WARP namespace.

        Centers are not populated here - they will be computed by PrepareWarpStage
        based on the image dimensions at warp time. This allows warp params to be
        added before resize/pad operations without the centers becoming stale.

        Args:
            angle: Rotation angles [B].
            scale_x: Scale in x [B].
            scale_y: Scale in y [B].
            shear_x: Shear in x [B].
            shear_y: Shear in y [B].
            tx: Translation in x [B].
            ty: Translation in y [B].
        """
        warp_params = WarpParams.from_components(
            angle=angle,
            scale_x=scale_x,
            scale_y=scale_y,
            shear_x=shear_x,
            shear_y=shear_y,
            tx=tx,
            ty=ty,
        )

        image = self.image
        warp_params = warp_params.to(device=image.device, dtype=image.dtype)

        if warp_params.B != image.shape[0]:
            if warp_params.B == 1:
                warp_params = warp_params.expand(batch_size=[image.shape[0]])
            else:
                raise ValueError(
                    f"WarpParams batch size {warp_params.B} does not match ImageDetail batch size {image.shape[0]}, nor can it be trivially expanded."
                )

        self.set(self.Keys.WARP.ROOT, warp_params)

    def clear_warp_params(self) -> None:
        self.del_(self.Keys.WARP.ROOT)

class ImagePairDetail(TensorDict):
    class Keys:
        REFERENCE: NestedKey = "reference"
        MOVING: NestedKey = "moving"

    @property
    def reference(self) -> ImageDetail:
        return self.get(self.Keys.REFERENCE)

    @property
    def moving(self) -> ImageDetail:
        return self.get(self.Keys.MOVING)
    

def top_k(td: ImagePairDetail | ImageDetail | TranslationResults, k: int) -> tuple[Tensor, Tensor]:
    if isinstance(td, ImagePairDetail):
        td = td.moving
    if isinstance(td, ImageDetail):
        td = td.get(ImageDetail.Keys.TRANSLATION)
        assert isinstance(td, TranslationResults), "TranslationResults not found in ImageDetail."

    scores = td.get(TranslationResults.Keys.SCORE)
    topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)
    return topk_values, topk_indices
