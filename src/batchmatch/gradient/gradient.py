from __future__ import annotations

import math
from typing import Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchmatch.base.tensordicts import ImageDetail
from batchmatch.base.pipeline import Stage, StageRegistry

Tensor = torch.Tensor
NestedKey = Union[str, tuple[str, ...]]

gradient_registry = StageRegistry("gradient")

try:
    from torch._higher_order_ops.associative_scan import associative_scan  # optional fast path
    _HAS_ASSOC_SCAN = True
except Exception:
    associative_scan = None
    _HAS_ASSOC_SCAN = False


@torch.jit.script
def _cum_add_scan(x: Tensor, a: Tensor) -> Tensor:
    L = x.size(-1)
    y = torch.empty_like(x)
    y[..., 0] = x[..., 0]
    for i in range(1, L):
        y[..., i] = a * y[..., i - 1] + x[..., i]
    return y


@torch.jit.script
def _ema_scan(x: Tensor, a: Tensor) -> Tensor:
    L = x.size(-1)
    y = torch.empty_like(x)
    y[..., 0] = x[..., 0]
    b = 1.0 - a
    for i in range(1, L):
        y[..., i] = a * y[..., i - 1] + b * x[..., i]
    return y

def _affine_combine(left, right):
    a1, b1 = left
    a2, b2 = right
    return a1 * a2, b1 * a2 + b2

def parallel_ema(x: Tensor, alpha: Tensor, chunk_size: int = 64) -> Tensor:
    N = x.size(-1)
    if N <= 1:
        return x

    orig_dtype = x.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        alpha = alpha.to(dtype=torch.float32, device=x.device)
    else:
        alpha = alpha.to(dtype=x.dtype, device=x.device)

    if _HAS_ASSOC_SCAN and x.is_cuda and N > 1:
        a = alpha.clamp(0.0, 1.0)
        b = (1.0 - a) * x
        b[..., 0] = x[..., 0]
        a_full = a.expand_as(b)
        y = associative_scan(_affine_combine, (a_full, b), dim=-1)[1]
        return y.to(orig_dtype)

    K = int(chunk_size)
    if N <= 2 * K:
        return _ema_scan(x, alpha).to(orig_dtype)

    pad = (-N) % K
    if pad:
        x = F.pad(x, (0, pad))
    M = x.size(-1) // K

    xk = x.view(*x.shape[:-1], M, K)
    yk = _ema_scan(xk, alpha)

    heads = xk[..., 0]
    tails = yk[..., -1]
    decayK = alpha ** K

    carry_in = tails.clone()
    carry_in[..., 1:] -= decayK * heads[..., 1:]
    carries = _cum_add_scan(carry_in, decayK)

    factors = torch.empty_like(carries)
    factors[..., 0] = 0
    factors[..., 1:] = carries[..., :-1] - heads[..., 1:]

    powers = alpha ** torch.arange(1, K + 1, device=x.device, dtype=x.dtype)
    y = (yk + factors.unsqueeze(-1) * powers).view(*x.shape)

    if pad:
        y = y[..., :N]
    return y.to(orig_dtype)



def integral_image(img: Tensor) -> Tensor:
    b, c, h, w = img.shape
    acc_dtype = torch.double if img.dtype == torch.double else torch.float32
    out = torch.zeros((b, c, h + 1, w + 1), device=img.device, dtype=acc_dtype)
    img_acc = img.to(dtype=acc_dtype)
    out[..., 1:, 1:] = img_acc.cumsum(dim=-1).cumsum(dim=-2)
    return out


def rect_sum(I: Tensor, y0: slice, y1: slice, x0: slice, x1: slice) -> Tensor:
    return I[..., y1, x1] - I[..., y1, x0] - I[..., y0, x1] + I[..., y0, x0]

class _GroupedConvKernels(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._w_cache: dict[tuple[str, int, int, torch.device, torch.dtype], Tensor] = {}

    def _conv2(self, x: Tensor, k: Tensor, *, padding: int, tag: str) -> Tensor:
        C = x.shape[1]
        K = 1
        key = (tag, C, K, x.device, x.dtype, getattr(k, "_version", 0))
        w = self._w_cache.get(key)
        if w is None:
            w = k.to(device=x.device, dtype=x.dtype).repeat_interleave(C, dim=0).contiguous()
            self._w_cache[key] = w
        return F.conv2d(x, w, padding=padding, groups=C)

    def _conv2_stacked(self, x: Tensor, k: Tensor, *, padding: int, tag: str) -> Tensor:
        C = x.shape[1]
        K = int(k.shape[0])
        key = (tag, C, K, x.device, x.dtype, getattr(k, "_version", 0))
        w = self._w_cache.get(key)
        if w is None:
            kk = k.to(device=x.device, dtype=x.dtype).contiguous()
            w = kk.repeat(C, 1, 1, 1).contiguous()
            self._w_cache[key] = w
        y = F.conv2d(x, w, padding=padding, groups=C)
        B, _, H, W = y.shape
        return y.view(B, C, K, H, W)

    def _conv2_xy(self, x: Tensor, kxy: Tensor, *, padding: int) -> tuple[Tensor, Tensor]:
        C = x.shape[1]
        if C == 1:
            y = F.conv2d(x, kxy, padding=padding, groups=1)
            gx = y[:, 0:1, :, :]
            gy = y[:, 1:2, :, :]
        else:
            # Multi-channel w/ grouped convolution
            w = kxy.repeat(C, 1, 1, 1)
            y = F.conv2d(x, w, padding=padding, groups=C)
            # Output is [B, 2*C, H, W], reshape to [B, C, 2, H, W]
            B, _, H, W = y.shape
            y = y.view(B, C, 2, H, W)
            gx = y[:, :, 0, :, :]
            gy = y[:, :, 1, :, :]
        return gx, gy


class GradientModuleBase(_GroupedConvKernels, Stage):
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.IMAGE})
    sets: Set[NestedKey] = {ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _compute(self, img: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, tensordict: ImageDetail) -> ImageDetail:
        img = tensordict.get(ImageDetail.Keys.IMAGE)
        gx, gy = self._compute(img)
        tensordict.set(ImageDetail.Keys.GRAD.X, gx)
        tensordict.set(ImageDetail.Keys.GRAD.Y, gy)
        return tensordict


@gradient_registry.register("centered_difference", "cd")
class CenteredDifferenceGradient(GradientModuleBase):
    """
    Compute gradients using a centered difference kernel.
    """
    def __init__(self, *, stencil_size: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        stencil_size = int(stencil_size)
        if stencil_size < 3 or (stencil_size % 2) != 1:
            raise ValueError(f"stencil_size must be odd and >= 3, got {stencil_size}")

        cd = self._compute_fd_coefficients(stencil_size)
        kx = torch.zeros((1, 1, stencil_size, stencil_size), dtype=torch.float32)
        ky = torch.zeros((1, 1, stencil_size, stencil_size), dtype=torch.float32)
        center = stencil_size // 2
        kx[0, 0, center, :] = cd
        ky[0, 0, :, center] = cd

        kxy = torch.cat([kx, ky], dim=0)
        self.register_buffer("kxy", kxy, persistent=True)
        self.padding = center

    @staticmethod
    def _compute_fd_coefficients(n: int) -> Tensor:
        half = n // 2
        points = torch.arange(-half, half + 1, dtype=torch.float64)
        A = torch.zeros((n, n), dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        for k in range(n):
            A[k, :] = points ** k
            if k == 1:
                b[k] = 1.0
        coeffs = torch.linalg.solve(A, b)
        return coeffs.to(torch.float32)

    def _compute(self, img: Tensor) -> tuple[Tensor, Tensor]:
        return self._conv2_xy(img, self.kxy, padding=self.padding)

@gradient_registry.register("sobel")
class SobelGradient(GradientModuleBase):
    """
    Compute gradients using a Sobel kernel.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        kxy = torch.cat([kx, ky], dim=0)
        self.register_buffer("kxy", kxy, persistent=True)

    def _compute(self, img: Tensor) -> tuple[Tensor, Tensor]:
        return self._conv2_xy(img, self.kxy, padding=1)

@gradient_registry.register("box_ratio")
class BoxRatioGradient(GradientModuleBase):
    """
    Compute gradients using box-averaged ratio differences.

    Args:
        width: Odd window size for the box filter.
        eps: Small value for numerical stability.
        mode: Ratio mode ("log", "sym", or "ratio").
        require_positive: Whether to clamp inputs to non-negative values for log mode.
        padding_mode: Padding mode for non-zero padding.
    """
    def __init__(
        self,
        *,
        width: int = 11,
        eps: float = 1e-6,
        mode: str = "sym",
        require_positive: bool = False,
        padding_mode: str = "zeros",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        width = int(width)
        if width < 3 or (width % 2) != 1:
            raise ValueError(f"width must be odd and >= 3, got {width}")
        if mode not in ("log", "sym", "ratio"):
            raise ValueError(f"mode must be one of {{'log','sym','ratio'}}, got {mode}")

        self.mode = mode
        self.require_positive = bool(require_positive)
        self.padding_mode = padding_mode

        R = width // 2
        area = float(R * (2 * R + 1))  # half-window
        inv_area = 1.0 / area

        kL = torch.zeros((1, 1, width, width), dtype=torch.float32)
        kR = torch.zeros((1, 1, width, width), dtype=torch.float32)
        kU = torch.zeros((1, 1, width, width), dtype=torch.float32)
        kD = torch.zeros((1, 1, width, width), dtype=torch.float32)

        kL[0, 0, :, :R] = inv_area
        kR[0, 0, :, R + 1 :] = inv_area
        kU[0, 0, :R, :] = inv_area
        kD[0, 0, R + 1 :, :] = inv_area

        k4 = torch.cat([kL, kR, kU, kD], dim=0)
        self.register_buffer("k4", k4, persistent=True)

        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32), persistent=True)
        self.padding = R

    def _ratio_to_grad(self, a: Tensor, b: Tensor) -> Tensor:
        eps = self.eps.to(dtype=a.dtype, device=a.device)
        if self.mode == "sym":
            diff = a - b
            denom = torch.maximum(a, b) + eps
            mag = diff.abs() / denom
            return diff.sign() * mag

        if self.mode == "ratio":
            return (a - b) / (b + eps)

        if self.require_positive:
            a = a.clamp_min(0)
            b = b.clamp_min(0)

        return torch.log(a + eps) - torch.log(b + eps)

    def _compute(self, img: Tensor) -> tuple[Tensor, Tensor]:
        pad = self.padding

        if self.padding_mode != "zeros":
            img = F.pad(img, (pad, pad, pad, pad), mode=self.padding_mode)
            g = self._conv2_stacked(img, self.k4, padding=0, tag="box_k4")
        else:
            g = self._conv2_stacked(img, self.k4, padding=pad, tag="box_k4")

        L = g[:, :, 0]
        R = g[:, :, 1]
        U = g[:, :, 2]
        D = g[:, :, 3]

        gx = self._ratio_to_grad(L, R)
        gy = self._ratio_to_grad(U, D)
        return gx, gy


def _roe_ratio(s1: Tensor, s2: Tensor, eps: Tensor, mode: str) -> Tensor:
    eps = eps.to(device=s1.device, dtype=s1.dtype)

    if mode == "log":
        s1 = s1.clamp_min(0)
        s2 = s2.clamp_min(0)
        return torch.log(s1 + eps) - torch.log(s2 + eps)

    r = (s1 + eps) / (s2 + eps)

    if mode == "ratio":
        return r - 1.0

    rinv = 1.0 / (r + eps)
    mag = 1.0 - torch.min(r, rinv)
    sgn = torch.sign(torch.log(r + eps))
    return sgn * mag


def _scan_ema(x: Tensor, alpha: Tensor, dim: int) -> Tensor:
    if dim == x.dim() - 1:
        return parallel_ema(x, alpha)
    y = parallel_ema(x.movedim(dim, -1).contiguous(), alpha)
    return y.movedim(-1, dim)


def _scan_ema_bidir(x: Tensor, alpha: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    fwd = _scan_ema(x, alpha, dim)
    bwd = _scan_ema(torch.flip(x, dims=[dim]), alpha, dim)
    bwd = torch.flip(bwd, dims=[dim])
    return fwd, bwd


def _isef_smooth(x: Tensor, alpha: Tensor, dim: int) -> Tensor:
    fwd, bwd = _scan_ema_bidir(x, alpha, dim)
    a = alpha.to(device=x.device, dtype=x.dtype)
    b = 1.0 - a
    return (fwd + bwd - b * x) / (1.0 + a)


@gradient_registry.register("roewa")
class ROEWAGradient(GradientModuleBase):
    """
    Compute gradients using the ROEWA operator.

    Args:
        alpha: EMA decay factor in (0, 1).
        eps: Small value for numerical stability.
        mode: Ratio mode ("log", "sym", or "ratio").
        orthogonal_smooth: Whether to use orthogonal smoothing.
    """
    def __init__(
        self,
        *,
        alpha: float = 0.9,
        eps: float = 1e-12,
        mode: str = "log",
        orthogonal_smooth: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if mode not in ("log", "sym", "ratio"):
            raise ValueError(f"mode must be one of {{'log','sym','ratio'}}, got {mode}")

        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32), persistent=True)
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32), persistent=True)
        self.mode = mode
        self.orthogonal_smooth = bool(orthogonal_smooth)

    @staticmethod
    def alpha_from_sigma(sigma: float) -> float:
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        return math.exp(-math.sqrt(2.0) / sigma)

    def _compute(self, img: Tensor) -> tuple[Tensor, Tensor]:
        a = self.alpha
        if img.dtype in (torch.float16, torch.bfloat16):
            a = a.to(dtype=torch.float32)

        if self.orthogonal_smooth:
            img_y = _isef_smooth(img, a, dim=-2)
            sL, sR = _scan_ema_bidir(img_y, a, dim=-1)
            gx = _roe_ratio(sL, sR, self.eps, self.mode)

            img_x = _isef_smooth(img, a, dim=-1)
            sU, sD = _scan_ema_bidir(img_x, a, dim=-2)
            gy = _roe_ratio(sU, sD, self.eps, self.mode)
        else:
            sL, sR = _scan_ema_bidir(img, a, dim=-1)
            gx = _roe_ratio(sL, sR, self.eps, self.mode)

            sU, sD = _scan_ema_bidir(img, a, dim=-2)
            gy = _roe_ratio(sU, sD, self.eps, self.mode)

        return gx, gy

def build_gradient_operator(name: str, **kwargs) -> GradientModuleBase:
    return gradient_registry.build(name, **kwargs)
