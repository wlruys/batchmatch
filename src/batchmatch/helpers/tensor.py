from __future__ import annotations

from typing import Literal, Optional
import torch

Tensor = torch.Tensor

__all__ = [
    "to_bchw",
    "to_bchw_flexible",
    "to_hw",
    "get_hw",
    "to_chw",
    "to_bhw",
    "check_same_shape",
    "mask_like_image",
    "expand_mask_to_image",
    "scale_points",
    "shift_points",
    "clip_points",
    "adjust_points_to_crop",
    "as_bool_mask",
    "reduce_batch",
    "to_common_device",
]

def to_bchw(x: Tensor, batch_size: Optional[int] = None) -> Tensor:
    """
    Return a tensor in BCHW layout without altering dtype/device.
    """
    if x.ndim == 2:  # H, W
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:  # C, H, W
        x = x.unsqueeze(0)
    elif x.ndim > 4 or x.ndim < 2:
        raise ValueError(f"Expected tensor with rank 2-4, got {tuple(x.shape)}")

    if batch_size is not None and x.shape[0] == 1:
        x = x.expand(batch_size, -1, -1, -1)
    return x

def to_hw(x: Tensor) -> Tensor:
    """
    Return a tensor in HW layout by removing batch and channel dimensions.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    if x.shape[0] != 1:
        raise ValueError(f"Cannot shrink batch dimension {x.shape[0]}")
    if x.shape[1] != 1:
        raise ValueError(f"Cannot shrink channel dimension {x.shape[1]}")
    return x[0, 0]

def get_hw(x: Tensor) -> tuple[int, int]:
    """
    Return the height and width of a BCHW tensor.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    return x.shape[-2], x.shape[-1]

def to_chw(x: Tensor) -> Tensor:
    """
    Return a tensor in CHW layout without altering dtype/device.
    """
    if x.ndim == 2: 
        return x.unsqueeze(0) 
    if x.ndim == 3:  
        return x
    if x.ndim == 4:
        #Convert to C, H, W*B
        B, C, H, W = x.shape
        return x.permute(1, 2, 0, 3).reshape(C, H, B * W)
    raise ValueError(f"Expected tensor with rank 2-4, got {tuple(x.shape)}")

def to_bhw(x: Tensor) -> Tensor:
    """
    Return a tensor in BHW layout without altering dtype/device.
    """
    if x.ndim == 2:  
        return x.unsqueeze(0)
    if x.ndim == 3: 
        return x
    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(f"Cannot convert BCHW with C={x.shape[1]} to BHW.")
        return x[:, 0]
    raise ValueError(f"Expected tensor with rank 2-4, got {tuple(x.shape)}")

def check_same_shape(a: Tensor, b: Tensor, what: str) -> None:
    """
    Validate that two tensors share the same shape.
    """
    if a.shape != b.shape:
        raise ValueError(f"{what} must share shape. Got {tuple(a.shape)} and {tuple(b.shape)}.")

def mask_like_image(image: Tensor, mask: Optional[Tensor]) -> Tensor:
    """
    Return a mask shaped like the image, expanding single-channel masks.
    """
    if image.ndim != 4:
        raise ValueError(f"mask requires BCHW image, got {tuple(image.shape)}")

    if mask is None:
        prepared = torch.ones_like(image[:, :1])
    else:
        prepared = to_bchw(mask)

    if prepared.shape[0] != image.shape[0]:
        raise ValueError(
            f"Mask batch dimension {prepared.shape[0]} must match image batch {image.shape[0]}."
        )
    if prepared.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"Mask spatial dims {prepared.shape[-2:]} must match image {image.shape[-2:]}."
        )
    if prepared.shape[1] not in (1, image.shape[1]):
        raise ValueError(
            f"Mask channels {prepared.shape[1]} must be 1 or {image.shape[1]} to match the image."
        )
    return prepared


def expand_mask_to_image(mask: Tensor, image: Tensor) -> Tensor:
    """
    Expand a single-channel mask to match multi-channel image.
    """
    if mask.shape[1] == 1 and image.shape[1] > 1:
        return mask.expand(image.shape[0], image.shape[1], *image.shape[-2:])
    return mask


def scale_points(points: Tensor, *, scale_x: float, scale_y: float) -> Tensor:
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points with last dim size 2, got {tuple(points.shape)}.")
    scale = torch.tensor([scale_x, scale_y], device=points.device, dtype=points.dtype)
    return points * scale


def shift_points(points: Tensor, *, dx: Tensor, dy: Tensor) -> Tensor:
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points with last dim size 2, got {tuple(points.shape)}.")
    offset = torch.stack([dx, dy], dim=-1)
    for _ in range(points.ndim - 2):
        offset = offset.unsqueeze(1)
    return points + offset


def clip_points(points: Tensor, *, H: int, W: int) -> Tensor:
    """
    Clip XY points to image bounds [0, W) x [0, H).
    """
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points with last dim size 2, got {tuple(points.shape)}.")
    out = points.clone()
    out[..., 0] = out[..., 0].clamp(min=0, max=W)
    out[..., 1] = out[..., 1].clamp(min=0, max=H)
    return out


def adjust_points_to_crop(
    points: Tensor,
    crop_box: Tensor,
    *,
    clip: bool = True,
) -> Tensor:
    """
    Adjust XY points to a new cropped domain.
    """
    if points.shape[-1] != 2:
        raise ValueError(f"Expected points with last dim size 2, got {tuple(points.shape)}.")

    if crop_box.ndim == 1:
        x0 = crop_box[0]
        y0 = crop_box[1]
        new_W = int((crop_box[2] - crop_box[0]).item())
        new_H = int((crop_box[3] - crop_box[1]).item())
        offset = torch.tensor([-x0, -y0], device=points.device, dtype=points.dtype)
    else:
        x0 = crop_box[:, 0]
        y0 = crop_box[:, 1]
        new_W = int((crop_box[0, 2] - crop_box[0, 0]).item())
        new_H = int((crop_box[0, 3] - crop_box[0, 1]).item())
        offset = torch.stack([-x0, -y0], dim=-1) 
        if points.ndim == 3:
            offset = offset.unsqueeze(1)

    out = points + offset

    if clip:
        out = clip_points(out, H=new_H, W=new_W)

    return out


def to_bchw_flexible(x: Tensor) -> Tensor:
    """
    Convert HW, BHW, or BCHW tensor to BCHW layout.
    """
    if x.ndim == 2: 
        return x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3: 
        return x.unsqueeze(1)
    elif x.ndim == 4:
        return x
    else:
        raise ValueError(f"Expected tensor with rank 2-4, got {tuple(x.shape)}")


def as_bool_mask(x: Tensor) -> Tensor:
    """
    Convert a tensor to a boolean mask.
    """
    if x.dtype == torch.bool:
        return x
    if x.is_floating_point():
        return x > 0
    return x != 0


def reduce_batch(x: Tensor, reduction: Literal["mean", "sum", "none", None] = None) -> Tensor:
    """
    Reduce a per-batch tensor according to the specified mode.
    """
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction in ("none", None):
        return x
    raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction}")


def to_common_device(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """
    Move tensors to a common device if they differ.
    """
    if x.device != y.device:
        y = y.to(device=x.device)
    return x, y
