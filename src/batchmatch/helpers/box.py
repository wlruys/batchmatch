from batchmatch.helpers.tensor import to_bchw, to_bhw
import torch 

Tensor = torch.Tensor
from torchvision.ops import masks_to_boxes
from torchvision import tv_tensors

def area_xyxy(box_bx4: Tensor) -> Tensor:
    """
    Compute areas of boxes in XYXY format.
    """
    if box_bx4.shape[-1] != 4:
        raise ValueError(f"area_xyxy expects (...,4), got {tuple(box_bx4.shape)}.")
    widths = (box_bx4[..., 2] - box_bx4[..., 0]).clamp(min=0)
    heights = (box_bx4[..., 3] - box_bx4[..., 1]).clamp(min=0)
    return widths * heights

def shift_xyxy(box_bx4: Tensor, *, dx: int, dy: int) -> Tensor:
    """
    Shift XYXY boxes by (dx, dy).
    """
    off = torch.tensor([dx, dy, dx, dy], device=box_bx4.device, dtype=box_bx4.dtype)
    return box_bx4 + off


def shift_xyxy_batch(box: Tensor, *, dx: Tensor, dy: Tensor) -> Tensor:
    """
    Shift XYXY boxes by (dx, dy) per batch element.
    """
    offset = torch.stack([dx, dy, dx, dy], dim=-1) 

    if box.ndim == 3:
        offset = offset.unsqueeze(1)

    return box + offset

def shrink_xyxy(box_bx4: Tensor, k: int) -> Tensor:
    """
    Shrink XYXY boxes by k pixels on all sides.
    """
    k = int(k)
    if k <= 0:
        return box_bx4.clone()
    out = box_bx4.clone()
    out[..., 0] += k
    out[..., 1] += k
    out[..., 2] -= k
    out[..., 3] -= k
    return out

def round_int_xyxy(box_bx4: Tensor) -> Tensor:
    """
    Round XYXY boxes to integer coordinates.
    """
    x = box_bx4
    if x.dtype.is_floating_point:
        x = x.round()
    return x.to(torch.int64)

def box_to_quad(box_bx4: Tensor) -> Tensor:
    """
    Convert XYXY boxes to XYXYXYXY quadrilaterals.
    """
    squeeze = False
    if box_bx4.ndim == 2:
        box_bx4 = box_bx4.unsqueeze(1)
        squeeze = True
    if box_bx4.ndim != 3 or box_bx4.shape[-1] != 4:
        raise ValueError(f"box_to_quad expects (...,4), got {tuple(box_bx4.shape)}.")

    x0 = box_bx4[..., 0:1]
    y0 = box_bx4[..., 1:2]
    x1 = box_bx4[..., 2:3]
    y1 = box_bx4[..., 3:4]
    quad = torch.cat([x0, y0, x1, y0, x1, y1, x0, y1], dim=-1)
    return quad.squeeze(1) if squeeze else quad

def shift_quad(box_bx8: Tensor, *, dx: int, dy: int) -> Tensor:
    """
    Shift XYXYXYXY quads by (dx, dy).
    """
    off = torch.tensor([dx, dy, dx, dy, dx, dy, dx, dy], device=box_bx8.device, dtype=box_bx8.dtype)
    return box_bx8 + off


def shift_quad_batch(quad: Tensor, *, dx: Tensor, dy: Tensor) -> Tensor:
    """
    Shift XYXYXYXY quads by (dx, dy) per batch element.
    """
    offset = torch.stack([dx, dy, dx, dy, dx, dy, dx, dy], dim=-1)  

    if quad.ndim == 3:
        offset = offset.unsqueeze(1)

    return quad + offset

def shrink_quad(box_bx8: Tensor, k: int) -> Tensor:
    k = int(k)
    if k <= 0:
        return box_bx8.clone()
    out = box_bx8.clone()
    out[..., 0] += k
    out[..., 1] += k
    out[..., 2] -= k
    out[..., 3] += k
    out[..., 4] -= k
    out[..., 5] -= k
    out[..., 6] += k
    out[..., 7] -= k
    return out

def area_quad(box_bx8: Tensor) -> Tensor:
    if box_bx8.shape[-1] != 8:
        raise ValueError(f"area_quad expects (...,8), got {tuple(box_bx8.shape)}.")
    x = box_bx8[..., 0::2]
    y = box_bx8[..., 1::2]
    sum1 = (x[..., 0] * y[..., 1] + x[..., 1] * y[..., 2] + x[..., 2] * y[..., 3] + x[..., 3] * y[..., 0])
    sum2 = (y[..., 0] * x[..., 1] + y[..., 1] * x[..., 2] + y[..., 2] * x[..., 3] + y[..., 3] * x[..., 0])
    area = 0.5 * torch.abs(sum1 - sum2)
    return area

def quad_to_box(box_bx8: Tensor) -> Tensor:
    """
    Convert XYXYXYXY quads to XYXY boxes.
    """
    squeeze = False
    if box_bx8.ndim == 2:
        box_bx8 = box_bx8.unsqueeze(1)
        squeeze = True
    if box_bx8.ndim != 3 or box_bx8.shape[-1] != 8:
        raise ValueError(f"quad_to_box expects (...,8), got {tuple(box_bx8.shape)}.")

    x_coords = box_bx8[..., 0::2]
    y_coords = box_bx8[..., 1::2]
    x_min, _ = x_coords.min(dim=2)
    x_max, _ = x_coords.max(dim=2)
    y_min, _ = y_coords.min(dim=2)
    y_max, _ = y_coords.max(dim=2)
    box = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    return box.squeeze(1) if squeeze else box


def quads_to_boxes(box_bx8: Tensor) -> Tensor:
    """
    Convert XYXYXYXY quads to XYXY boxes.
    """
    return quad_to_box(box_bx8)


def scale_xyxy(box_bx4: Tensor, *, scale_x: float, scale_y: float) -> Tensor:
    """
    Scale XYXY boxes by the given factors.
    """
    dtype = box_bx4.dtype
    scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=box_bx4.device, dtype=torch.float32)
    return (box_bx4.float() * scale).to(dtype)


def scale_quad(box_bx8: Tensor, *, scale_x: float, scale_y: float) -> Tensor:
    """
    Scale XYXYXYXY quads by the given factors.
    """
    dtype = box_bx8.dtype
    scale = torch.tensor(
        [scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x, scale_y],
        device=box_bx8.device,
        dtype=torch.float32,
    )
    return (box_bx8.float() * scale).to(dtype)

def mask_to_box(mask: Tensor) -> torch.Tensor:
    """
    Compute bounding boxes for binary masks. (uses torchvision function)
    """
    mask = to_bhw(mask)
    return masks_to_boxes(mask).unsqueeze(1)

def box_to_mask(H: int, W: int, box_bxn4: torch.Tensor) -> Tensor:
    """
    Rasterize XYXY boxes into masks.
    """
    if box_bxn4.ndim != 3 or box_bxn4.shape[-1] != 4:
        raise ValueError(f"box_to_mask expects (B,N,4), got {tuple(box_bxn4.shape)}.")
    B, N, _ = box_bxn4.shape
    masks = torch.zeros((B, H, W), dtype=torch.float32, device=box_bxn4.device)
    for b in range(B):
        for n in range(N):
            x0, y0, x1, y1 = box_bxn4[b, n].tolist()
            x0 = max(0, min(W, int(x0)))
            y0 = max(0, min(H, int(y0)))
            x1 = max(0, min(W, int(x1)))
            y1 = max(0, min(H, int(y1)))
            if x1 > x0 and y1 > y0:
                masks[b, y0:y1, x0:x1] = 1.0
    return to_bchw(masks)  # (B,1,H,W)

def quad_to_mask(H: int, W: int, box_bxn8: torch.Tensor) -> Tensor:
    """
    Rasterize XYXYXYXY quads into masks.
    """
    if box_bxn8.ndim != 3 or box_bxn8.shape[-1] != 8:
        raise ValueError(f"quad_to_mask expects (B,N,8), got {tuple(box_bxn8.shape)}.")
    B, N, _ = box_bxn8.shape
    masks = torch.zeros((B, H, W), dtype=torch.float32, device=box_bxn8.device)
    for b in range(B):
        img = tv_tensors.Image(masks[b:b+1])
        for n in range(N):
            quad = box_bxn8[b, n].view(4, 2).cpu().numpy()
            img = img.draw_polygon(quad, fill=1.0)
        masks[b] = img.tensor[0]
    return to_bchw(masks)  # (B,1,H,W)


def pad_to_box(B: int, H: int, W: int, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Create a padded bounding box for an image size.
    """
    left, top, right, bottom = pad
    box = torch.tensor([[left, top, left + W, top + H]], dtype=torch.int64)
    return box.repeat(B, 1, 1)

def pad_to_quad(B: int, H: int, W: int, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Create a padded quadrilateral for an image size.
    """
    left, top, right, bottom = pad
    x0 = left
    y0 = top
    x1 = left + W
    y1 = top
    x2 = left + W
    y2 = top + H
    x3 = left
    y3 = top + H
    box = torch.tensor([[x0, y0, x1, y1, x2, y2, x3, y3]], dtype=torch.int64)
    return box.repeat(B, 1, 1)

def clip_xyxy(box: Tensor, *, H: int, W: int) -> Tensor:
    """
    Clip XYXY boxes to image bounds [0, W) x [0, H).
    """
    out = box.clone()
    out[..., 0] = out[..., 0].clamp(min=0, max=W)  # x0
    out[..., 1] = out[..., 1].clamp(min=0, max=H)  # y0
    out[..., 2] = out[..., 2].clamp(min=0, max=W)  # x1
    out[..., 3] = out[..., 3].clamp(min=0, max=H)  # y1
    return out


def clip_quad(quad: Tensor, *, H: int, W: int) -> Tensor:
    """
    Clip XYXYXYXY quad vertices to image bounds [0, W) x [0, H).
    """
    out = quad.clone()
    out[..., 0::2] = out[..., 0::2].clamp(min=0, max=W)
    out[..., 1::2] = out[..., 1::2].clamp(min=0, max=H)
    return out


def adjust_xyxy_to_crop(
    box: Tensor,
    crop_box: Tensor,
    *,
    clip: bool = True,
) -> Tensor:
    """
    Adjust XYXY boxes to a new cropped domain.
    """
    if crop_box.ndim == 1:
        x0 = crop_box[0]
        y0 = crop_box[1]
        new_W = int((crop_box[2] - crop_box[0]).item())
        new_H = int((crop_box[3] - crop_box[1]).item())
        offset = torch.stack([-x0, -y0, -x0, -y0]).to(device=box.device, dtype=box.dtype)
    else:
        x0 = crop_box[:, 0]
        y0 = crop_box[:, 1]
        new_W = int((crop_box[0, 2] - crop_box[0, 0]).item())
        new_H = int((crop_box[0, 3] - crop_box[0, 1]).item())
        offset = torch.stack([-x0, -y0, -x0, -y0], dim=-1)  
        if box.ndim == 3:
            offset = offset.unsqueeze(1)

    out = box + offset

    if clip:
        out = clip_xyxy(out, H=new_H, W=new_W)

    return out


def adjust_quad_to_crop(
    quad: Tensor,
    crop_box: Tensor,
    *,
    clip: bool = True,
) -> Tensor:
    """
    Adjust XYXYXYXY quads to a new cropped domain.
    """
    # Get crop origin
    if crop_box.ndim == 1:
        x0 = crop_box[0]
        y0 = crop_box[1]
        new_W = int((crop_box[2] - crop_box[0]).item())
        new_H = int((crop_box[3] - crop_box[1]).item())
        offset = torch.stack(
            [-x0, -y0, -x0, -y0, -x0, -y0, -x0, -y0]
        ).to(device=quad.device, dtype=quad.dtype)
    else:
        x0 = crop_box[:, 0]
        y0 = crop_box[:, 1]
        new_W = int((crop_box[0, 2] - crop_box[0, 0]).item())
        new_H = int((crop_box[0, 3] - crop_box[0, 1]).item())
        offset = torch.stack(
            [-x0, -y0, -x0, -y0, -x0, -y0, -x0, -y0], dim=-1
        ) 
        if quad.ndim == 3:
            offset = offset.unsqueeze(1)  # (B, 1, 8)

    out = quad + offset

    if clip:
        out = clip_quad(out, H=new_H, W=new_W)

    return out


def is_box_valid(box: Tensor) -> Tensor:
    """
    Check if XYXY boxes are valid (non-empty, x1 > x0 and y1 > y0).
    """
    return (box[..., 2] > box[..., 0]) & (box[..., 3] > box[..., 1])
