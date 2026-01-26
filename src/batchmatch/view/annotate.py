from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from batchmatch.base.tensordicts import ImageDetail
from .config import QuadAnnotationSpec, BoxAnnotationSpec, PointAnnotationSpec
from . import render

__all__ = [
    "annotate_quads",
    "annotate_boxes",
    "annotate_points",
    "annotate_from_detail",
    "draw_line",
    "draw_polygon",
    "draw_circle",
]


def draw_line(
    image: Tensor,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[float, float, float],
    thickness: int = 1,
) -> Tensor:
    image = render.to_chw(image)
    C, H, W = image.shape

    color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(C if C == 3 else 1, 1)
    if C == 1:
        color_tensor = color_tensor.mean()

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    half_t = thickness // 2

    while True:
        for tx in range(-half_t, half_t + 1):
            for ty in range(-half_t, half_t + 1):
                px, py = x + tx, y + ty
                if 0 <= px < W and 0 <= py < H:
                    if C == 3:
                        image[:, py, px] = color_tensor.squeeze()
                    else:
                        image[0, py, px] = color_tensor

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return image


def draw_polygon(
    image: Tensor,
    points: Sequence[Tuple[int, int]],
    color: Tuple[float, float, float],
    thickness: int = 1,
    fill: bool = False,
    fill_alpha: float = 0.3,
) -> Tensor:
    image = render.to_chw(image)

    if fill:
        image = _fill_polygon(image, points, color, fill_alpha)

    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        image = draw_line(image, int(x0), int(y0), int(x1), int(y1), color, thickness)

    return image


def _fill_polygon(
    image: Tensor,
    points: Sequence[Tuple[int, int]],
    color: Tuple[float, float, float],
    alpha: float,
) -> Tensor:
    C, H, W = image.shape

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_y = max(0, int(min(ys)))
    max_y = min(H - 1, int(max(ys)))

    color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device)
    if C == 1:
        color_tensor = color_tensor.mean().unsqueeze(0)

    for y in range(min_y, max_y + 1):
        intersections = []
        n = len(points)
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    intersections.append(x)

        intersections.sort()

        for i in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(intersections[i]))
            x_end = min(W - 1, int(intersections[i + 1]))
            for x in range(x_start, x_end + 1):
                image[:, y, x] = (1 - alpha) * image[:, y, x] + alpha * color_tensor

    return image


def draw_circle(
    image: Tensor,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[float, float, float],
    fill: bool = False,
) -> Tensor:
    image = render.to_chw(image)
    C, H, W = image.shape

    color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device)
    if C == 1:
        color_tensor = color_tensor.mean().unsqueeze(0)

    for y in range(max(0, cy - radius), min(H, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(W, cx + radius + 1)):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if fill:
                if dist <= radius:
                    image[:, y, x] = color_tensor
            else:
                if abs(dist - radius) < 1.5:
                    image[:, y, x] = color_tensor

    return image


def annotate_quads(
    image: Tensor,
    quads: Tensor,
    spec: QuadAnnotationSpec = QuadAnnotationSpec(),
) -> Tensor:
    image = render.to_chw(image).clone()
    quads = quads.detach()

    if quads.ndim == 1:
        quads = quads.unsqueeze(0)

    for i in range(quads.shape[0]):
        q = quads[i]
        points = [
            (int(q[0].item()), int(q[1].item())),
            (int(q[2].item()), int(q[3].item())),
            (int(q[4].item()), int(q[5].item())),
            (int(q[6].item()), int(q[7].item())),
        ]
        image = draw_polygon(
            image, points, spec.color, spec.thickness, spec.fill, spec.fill_alpha
        )

    return image


def annotate_boxes(
    image: Tensor,
    boxes: Tensor,
    spec: BoxAnnotationSpec = BoxAnnotationSpec(),
) -> Tensor:
    image = render.to_chw(image).clone()
    boxes = boxes.detach()

    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)

    for i in range(boxes.shape[0]):
        b = boxes[i]
        x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        image = draw_polygon(
            image, points, spec.color, spec.thickness, spec.fill, spec.fill_alpha
        )

    return image


def annotate_points(
    image: Tensor,
    points: Tensor,
    spec: PointAnnotationSpec = PointAnnotationSpec(),
) -> Tensor:
    image = render.to_chw(image).clone()
    points = points.detach()

    if points.ndim == 1:
        points = points.unsqueeze(0)

    for i in range(points.shape[0]):
        pt = points[i]
        x, y = int(pt[0].item()), int(pt[1].item())

        if spec.marker == "circle":
            image = draw_circle(image, x, y, spec.radius, spec.color, fill=True)
        elif spec.marker == "dot":
            image = draw_circle(image, x, y, max(1, spec.radius // 2), spec.color, fill=True)
        elif spec.marker == "cross":
            r = spec.radius
            image = draw_line(image, x - r, y, x + r, y, spec.color, 1)
            image = draw_line(image, x, y - r, x, y + r, spec.color, 1)
        elif spec.marker == "x":
            r = spec.radius
            image = draw_line(image, x - r, y - r, x + r, y + r, spec.color, 1)
            image = draw_line(image, x - r, y + r, x + r, y - r, spec.color, 1)

    return image


def annotate_from_detail(
    detail: ImageDetail,
    quad_spec: Optional[QuadAnnotationSpec] = None,
    box_spec: Optional[BoxAnnotationSpec] = None,
    point_spec: Optional[PointAnnotationSpec] = None,
) -> Tensor:
    image = detail.image.detach()
    if image.ndim == 4:
        B, C, H, W = image.shape
    else:
        B = 1
        W = image.shape[-1] if image.ndim >= 2 else 0

    result = render.to_chw(image).clone()

    def offset_quads(quads: Tensor, b: int) -> Tensor:
        if b == 0:
            return quads
        offset = torch.zeros_like(quads)
        offset[..., 0::2] = b * W
        return quads + offset

    def offset_boxes(boxes: Tensor, b: int) -> Tensor:
        if b == 0:
            return boxes
        offset = torch.zeros_like(boxes)
        offset[..., 0] = b * W
        offset[..., 2] = b * W
        return boxes + offset

    def offset_points(points: Tensor, b: int) -> Tensor:
        if b == 0:
            return points
        offset = torch.zeros_like(points)
        offset[..., 0] = b * W 
        return points + offset

    #quads are [B, N, 8] where B=batch, N=num quads per image
    if quad_spec is not None:
        if ImageDetail.Keys.DOMAIN.QUAD in detail:
            quads = detail.get(ImageDetail.Keys.DOMAIN.QUAD)
            if quads.ndim == 3:
                for b in range(min(B, quads.shape[0])):
                    batch_quads = offset_quads(quads[b], b)
                    result = annotate_quads(result, batch_quads, quad_spec)
            else:
                result = annotate_quads(result, quads, quad_spec)
        if ImageDetail.Keys.AUX.QUADS in detail:
            quads = detail.get(ImageDetail.Keys.AUX.QUADS)
            if quads.ndim == 3:
                for b in range(min(B, quads.shape[0])):
                    batch_quads = offset_quads(quads[b], b)
                    result = annotate_quads(result, batch_quads, quad_spec)
            else:
                result = annotate_quads(result, quads, quad_spec)

    #boxes are [B, N, 4] where B=batch, N=num boxes per image
    if box_spec is not None:
        if ImageDetail.Keys.DOMAIN.BOX in detail:
            boxes = detail.get(ImageDetail.Keys.DOMAIN.BOX)
            if boxes.ndim == 3:
                for b in range(min(B, boxes.shape[0])):
                    batch_boxes = offset_boxes(boxes[b], b)
                    result = annotate_boxes(result, batch_boxes, box_spec)
            else:
                result = annotate_boxes(result, boxes, box_spec)
        if ImageDetail.Keys.AUX.BOXES in detail:
            boxes = detail.get(ImageDetail.Keys.AUX.BOXES)
            if boxes.ndim == 3:
                for b in range(min(B, boxes.shape[0])):
                    batch_boxes = offset_boxes(boxes[b], b)
                    result = annotate_boxes(result, batch_boxes, box_spec)
            else:
                result = annotate_boxes(result, boxes, box_spec)

    if point_spec is not None:
        if ImageDetail.Keys.AUX.POINTS in detail:
            points = detail.get(ImageDetail.Keys.AUX.POINTS)
            if points.ndim == 3:
                for b in range(min(B, points.shape[0])):
                    batch_points = offset_points(points[b], b)
                    result = annotate_points(result, batch_points, point_spec)
            else:
                result = annotate_points(result, points, point_spec)

    return result
