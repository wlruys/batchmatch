from __future__ import annotations

import torch
from tensordict import TensorDictBase

from batchmatch.helpers.tensor import to_bchw
from batchmatch.base.tensordicts import ImageDetail, NestedKey

Tensor = torch.Tensor

__all__ = ["NestedKey", "ImageDetail", "build_image_td", "validate_image_td_shape"]


def build_image_td(image: Tensor) -> ImageDetail:
    """Build an ImageDetail TensorDict from a [B, C, H, W] image tensor."""
    image = to_bchw(image)
    return ImageDetail({ImageDetail.Keys.IMAGE: image}, batch_size=image.shape[0:1])


def _check_bchw_like(val: Tensor, *, B: int, C: int, H: int, W: int, name: object) -> None:
    if val.ndim != 4:
        raise ValueError(f"{name!r} must be rank-4 [B,C,H,W]-like, got {tuple(val.shape)}.")
    if val.shape[0] != B or val.shape[2] != H or val.shape[3] != W:
        raise ValueError(f"{name!r} must have shape ({B},1|{C},{H},{W}), got {tuple(val.shape)}.")
    ch = val.shape[1]
    if ch != 1 and ch != C:
        raise ValueError(f"{name!r} channel dim must be 1 or {C}, got {ch}.")


def _check_bnxd(val: Tensor, *, B: int, D: int, name: object) -> None:
    if val.ndim != 3:
        raise ValueError(f"{name!r} must be rank-3 (B,N,{D}), got {tuple(val.shape)}.")
    if val.shape[0] != B or val.shape[2] != D:
        raise ValueError(f"{name!r} must be ({B},N,{D}), got {tuple(val.shape)}.")


def validate_image_td_shape(td: TensorDictBase) -> bool:
    """
    Validate the shapes and devices of an ImageDetail TensorDict.

    Args:
        td: TensorDict to validate.

    Returns:
        True when validation succeeds.

    Raises:
        KeyError: If the required image key is missing.
        TypeError: If a leaf value is not a tensor.
        ValueError: If a tensor has an unexpected shape.
        RuntimeError: If tensors reside on different devices.
    """
    image = td.get(ImageDetail.Keys.IMAGE, default=None)
    if image is None:
        raise KeyError(f"Missing required key {ImageDetail.Keys.IMAGE!r}.")
    if image.ndim != 4:
        raise ValueError(
            f"{ImageDetail.Keys.IMAGE!r} must have shape [B,C,H,W], got {tuple(image.shape)}."
        )

    ref_device = image.device
    B, C, H, W = image.shape

    keys = ImageDetail.Keys
    spatial_keys = {keys.GRAD.X, keys.GRAD.Y, keys.GRAD.I, keys.DOMAIN.MASK, keys.DOMAIN.WINDOW}
    box_keys = {keys.DOMAIN.BOX, keys.AUX.BOXES}
    quad_keys = {keys.DOMAIN.QUAD, keys.AUX.QUADS}
    point_keys = {keys.AUX.POINTS}

    for key, val in td.items(include_nested=True, leaves_only=True):
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"Leaf {key!r} must be a torch.Tensor, got {type(val)}.")

        if val.device != ref_device:
            raise RuntimeError(f"Device mismatch for {key!r}: expected {ref_device}, got {val.device}.")

        if key == ImageDetail.Keys.IMAGE:
            if val.shape != (B, C, H, W):
                raise ValueError(f"{key!r} shape mismatch: expected {(B, C, H, W)}, got {tuple(val.shape)}.")
        elif key in spatial_keys:
            _check_bchw_like(val, B=B, C=C, H=H, W=W, name=key)
        elif key in box_keys:
            _check_bnxd(val, B=B, D=4, name=key)
        elif key in quad_keys:
            _check_bnxd(val, B=B, D=8, name=key)
        elif key in point_keys:
            _check_bnxd(val, B=B, D=2, name=key)

    return True
