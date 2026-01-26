from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from .tensor import to_bchw, to_chw

__all__ = [
    "image_to_uint8",
    "image_to_float",
    "rescale_to_unit",
    "round",
    "clip",
    "square",
    "whitten",
    "contrast_stretch",
    "scale_intensity",
    "remap_intensity",
    "binarize",
    "invert",
    "quantize_and_shuffle",
    "speckle_noise",
    "poisson_noise",
    "salt_and_pepper_noise",
    "additive_gaussian_noise",
    "median_filter",
    "gaussian_blur",
    "resize_bchw",
]

def image_to_uint8(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    dims = list(range(1, image.ndim))
    img_min = image.amin(dim=dims, keepdim=True)
    img_max = image.amax(dim=dims, keepdim=True)
    img_norm = (image - img_min) / (img_max - img_min + 1e-8)
    img_uint8 = (img_norm * 255.0).clamp(0, 255).to(dtype=torch.uint8)
    return img_uint8

def image_to_float(image: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    if image.dtype == torch.uint8:
        img_float = image.to(dtype=dtype) / 255.0
    elif image.dtype.is_floating_point:
        img_float = image.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype} (use uint8 or float).")
    return img_float


def rescale_to_unit(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")

    img_min = image.amin(dim=list(range(1, image.ndim)), keepdim=True)
    img_max = image.amax(dim=list(range(1, image.ndim)), keepdim=True)
    img_rescaled = (image - img_min) / (img_max - img_min + 1e-8)
    return img_rescaled

def round(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    return image.round()

def clip(image: Tensor, min_value: float = 0.0, max_value: float = 1.0) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    return image.clamp(min=min_value, max=max_value)

def square(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    return image * image

def whitten(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    mean = image.mean(dim=list(range(1, image.ndim)), keepdim=True)
    std = image.std(dim=list(range(1, image.ndim)), keepdim=True) + 1e-8
    return (image - mean) / std

def contrast_stretch(image: Tensor, low_perc: float = 2.0, high_perc: float = 98.0) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")

    B = image.shape[0]
    img_stretched = torch.empty_like(image)
    for b in range(B):
        img_b = image[b:b+1]
        low_val = torch.quantile(img_b, low_perc / 100.0, dim=list(range(1, img_b.ndim)), keepdim=True)
        high_val = torch.quantile(img_b, high_perc / 100.0, dim=list(range(1, img_b.ndim)), keepdim=True    )
        img_b_stretched = (img_b - low_val) / (high_val - low_val + 1e-8)
        img_b_stretched = img_b_stretched.clamp(0.0, 1.0)
        img_stretched[b:b+1] = img_b_stretched
    return img_stretched

def scale_intensity(image: Tensor, factor: float = 0.5) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    return image.sign() * (image.abs().pow(factor))

def remap_intensity(image: Tensor, in_min: float, in_max: float, out_min: float, out_max: float) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    img_clipped = image.clamp(min=in_min, max=in_max)
    img_normalized = (img_clipped - in_min) / (in_max - in_min + 1e-8)
    img_remapped = img_normalized * (out_max - out_min) + out_min
    return img_remapped

def binarize(image: Tensor, threshold: float = 0.5) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    return (image >= threshold).to(dtype=image.dtype)

def invert(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    return 1.0 - image

def quantize_and_shuffle(image: Tensor, num_bins: int = 8) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")

    B, C, H, W = image.shape
    img_flat = image.view(B, C, -1)

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=image.device)
    bin_indices = torch.bucketize(img_flat, bin_edges) - 1  # [0, num_bins-1]

    shuffled_bins = torch.randperm(num_bins, device=image.device)
    bin_map = torch.zeros(num_bins, device=image.device)
    for i in range(num_bins):
        bin_map[i] = shuffled_bins[i].float() / (num_bins - 1)

    img_shuffled = bin_map[bin_indices]
    img_shuffled = img_shuffled.view(B, C, H, W)
    return img_shuffled

def speckle_noise(image: Tensor, mean: float = 0.0, stddev: float = 0.1) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    noise = torch.randn_like(image) * stddev + mean
    return image + image * noise

def poisson_noise(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    vals = 2 ** torch.ceil(torch.log2(image.max() + 1e-8))
    noisy = torch.poisson(image * vals) / vals
    return noisy

def salt_and_pepper_noise(image: Tensor, amount: float = 0.05) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")

    B, C, H, W = image.shape
    num_salt = int(amount * H * W / 2)
    num_pepper = int(amount * H * W / 2)

    for b in range(B):
        for c in range(C):
            coords = [torch.randint(0, dim, (num_salt,), device=image.device) for dim in (H, W)]
            image[b, c, coords[0], coords[1]] = 1.0
            coords = [torch.randint(0, dim, (num_pepper,), device=image.device) for dim in (H, W)]
            image[b, c, coords[0], coords[1]] = 0.0

    return image

def additive_gaussian_noise(image: Tensor, mean: float = 0.0, stddev: float = 0.1) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")
    noise = torch.randn_like(image) * stddev + mean
    return image + noise

def median_filter(image: Tensor, kernel_size: int = 3) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")

    padding = kernel_size // 2
    padded = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode='reflect')
    unfolded = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # [B,C,H,W,kH,kW]
    median = unfolded.contiguous().view(*unfolded.shape[:4], -1).median(dim=-1).values
    return median

def gaussian_blur(image: Tensor, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
    if not image.dtype.is_floating_point:
        raise ValueError(f"Expected floating point image, got {image.dtype}.")

    channels = image.shape[1]
    kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
    kernel = kernel / kernel.sum()
    kernel_2d = kernel[:, None] @ kernel[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    blurred = torch.nn.functional.conv2d(image, kernel_2d, padding=padding, groups=channels)
    return blurred
