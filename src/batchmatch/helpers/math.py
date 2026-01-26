from __future__ import annotations
import math
import torch 
from typing import Any, Optional, Tuple, Union
from torch import nn

Tensor = torch.Tensor

__all__ = [
    "scalar_to_int",
    "next_power_2",
    "quantile",
    "batched_nanquantile",
    "batched_quantile_with_threshold",
    "batched_quantile_with_threshold_and_counts",
    "safe_divide_inplace",
    "safe_log_inplace",
    "safe_divide",
    "clamp_div",
    "masked_mean",
]

def scalar_to_int(value: Union[int, float, torch.Tensor], *, name: str) -> int:
    """
    Convert a Python or torch scalar to int.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
        if value.requires_grad:
            value = value.detach()
        if value.dtype.is_floating_point:
            value = value.round()
        value = value.to(torch.int64)
        return int(value.item())
    return int(value)

def next_power_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(int(x))))

def quantile(tensor, q, dim=None, keepdim=False):
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile value should be a float between 0 and 1.")

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    if tensor.size(dim) == 0:
        raise ValueError("Cannot compute quantile of an empty tensor")

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    n = sorted_tensor.shape[dim]

    q_tensor = torch.as_tensor(q, dtype=sorted_tensor.dtype, device=sorted_tensor.device)
    pos = q_tensor * (n - 1)
    lower_idx = pos.floor().to(torch.long)
    upper_idx = torch.clamp(lower_idx + 1, max=n - 1)
    weight = (pos - lower_idx.to(pos.dtype))

    gather_shape = list(sorted_tensor.shape)
    gather_shape[dim] = 1

    lower_idx = lower_idx.reshape([1] * sorted_tensor.dim()).expand(gather_shape)
    upper_idx = upper_idx.reshape([1] * sorted_tensor.dim()).expand(gather_shape)

    lower_value = torch.gather(sorted_tensor, dim, lower_idx)
    upper_value = torch.gather(sorted_tensor, dim, upper_idx)

    quantile_value = lower_value * (1 - weight) + upper_value * weight
    return quantile_value if keepdim else quantile_value.squeeze(dim)

def batched_nanquantile(tensor: Tensor, q: float) -> Tensor:
    result = torch.empty((tensor.shape[0],), dtype=tensor.dtype, device=tensor.device)
    for b in range(tensor.shape[0]):
        work = tensor[b]
        work = work[~torch.isnan(work)]
        if work.numel() == 0:
            qv = torch.tensor(float(0.0), dtype=tensor.dtype, device=tensor.device)
        else:
            qv = quantile(work, q)
        result[b] = qv

    return result


def batched_quantile_with_threshold(
    data: Tensor,
    q: float,
    threshold: float,
    default: float = 0.0,
    sample_size: int = 50000,
) -> Tensor:
    result, _ = batched_quantile_with_threshold_and_counts(data, q, threshold, default, sample_size)
    return result


def batched_quantile_with_threshold_and_counts(
    data: Tensor,
    q: float,
    threshold: float,
    default: float = 0.0,
    sample_size: int = 50000,
) -> tuple[Tensor, Tensor]:
    B, N = data.shape
    device = data.device

    valid_mask = data > threshold
    valid_counts = valid_mask.sum(dim=1)
    has_valid = valid_counts > 0
    max_valid = valid_counts.max().item()

    # If all valid counts fit in sample_size, use exact method (no sampling)
    if max_valid <= sample_size:
        masked_data = data.masked_fill(~valid_mask, float('inf'))
        sorted_data, _ = torch.sort(masked_data, dim=1)
        target_idx = (q * (valid_counts.float() - 1)).long().clamp(min=0, max=N - 1)
        result = sorted_data.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        result = torch.where(has_valid, result, torch.full_like(result, default))
        return result, valid_counts

    probs = valid_mask.float()
    probs[~has_valid] = 1.0  # Avoid NaN for empty batches
    probs = probs / probs.sum(dim=1, keepdim=True)

    sampled_indices = torch.multinomial(probs, sample_size, replacement=True)
    sampled_values = data.gather(1, sampled_indices)

    sorted_values, _ = torch.sort(sampled_values, dim=1)
    idx = int(q * (sample_size - 1))
    result = sorted_values[:, idx]

    result = torch.where(has_valid, result, torch.full_like(result, default))
    return result, valid_counts


def safe_divide_(a: Tensor, b: Tensor, threshold: float) -> Tensor:
    """
    Divide a by b in place, zeroing entries where |b| < threshold.
    """
    valid = b.abs() >= threshold
    safe_b = torch.where(valid, b, b.new_ones(()))
    a.div_(safe_b)
    a.masked_fill_(~valid, 0.0)
    return a


def safe_divide(a: Tensor, b: Tensor, threshold: float) -> Tensor:
    """
    Return a / b with zeros where |b| < threshold.
    """
    valid = b.abs() >= threshold
    safe_b = torch.where(valid, b, b.new_ones(()))
    return torch.where(valid, a / safe_b, a.new_zeros(()))


def clamp_div(num: Tensor, den: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Safe division using clamped denominator.
    """
    return num / den.clamp_min(eps)

def masked_mean(x: Tensor, mask: Tensor, dim: Tuple[int, ...], keepdim: bool = False, threshold: float = 1e-6) -> Tensor:
    """
    Compute the mean of x along dim, ignoring masked-out values.
    """
    masked = x * mask
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    summed = masked.sum(dim=dim, keepdim=keepdim)
    return summed / count
