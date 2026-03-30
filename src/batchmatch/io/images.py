from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Iterable, Optional

import torch
from torchvision.io import read_image, write_jpeg, write_png
from torchvision.transforms.functional import rgb_to_grayscale

from batchmatch.helpers.image import image_to_float, image_to_uint8
from batchmatch.helpers.tensor import to_bchw, to_chw
from batchmatch.io.utils import (
    _DEFAULT,
    PathLike,
    assert_overwrite,
    coerce_extensions,
    ensure_parent_dir,
    pathify,
)

Tensor = torch.Tensor

_TIFF_EXTENSIONS = {".tif", ".tiff"}

__all__ = [
    "ImageIO",
]


def _coerce_image(image: Tensor | object) -> Tensor:
    if torch.is_tensor(image):
        return image
    if hasattr(image, "image"):
        return getattr(image, "image")
    if hasattr(image, "get") and "image" in image:  # type: ignore[operator]
        return image.get("image")  # type: ignore[call-arg]
    raise TypeError(f"Expected a Tensor or image container, got {type(image)!r}.")


@dataclass(frozen=True)
class ImageIO:
    grayscale: bool = True
    dtype: Optional[torch.dtype] = torch.float32
    device: Optional[torch.device] = None
    convert_uint8: bool = True
    jpeg_quality: int = 95

    def load(
        self,
        path: PathLike,
        *,
        grayscale: bool | object = _DEFAULT,
        dtype: Optional[torch.dtype] | object = _DEFAULT,
        device: Optional[torch.device] | object = _DEFAULT,
        channel: int | str | None = None,
        downsample: int = 1,
    ) -> Tensor:
        grayscale = self.grayscale if grayscale is _DEFAULT else bool(grayscale)
        dtype = self.dtype if dtype is _DEFAULT else dtype
        device = self.device if device is _DEFAULT else device

        p = pathify(path)

        # Dispatch TIFF files to the dedicated loader
        if p.suffix.lower() in _TIFF_EXTENSIONS:
            return self._load_tiff(p, grayscale=grayscale, dtype=dtype, device=device,
                                   channel=channel, downsample=downsample)

        img = read_image(str(p))

        if grayscale:
            c = img.shape[0]
            if c == 1:
                pass
            elif c >= 3:
                img = rgb_to_grayscale(img[:3])
            else:
                raise ValueError(f"Cannot grayscale-convert image with {c} channels.")

        img = to_bchw(img)

        if dtype is not None:
            if dtype.is_floating_point:
                img = image_to_float(img, dtype=dtype)
            else:
                img = image_to_uint8(img)
                if img.dtype != dtype:
                    img = img.to(dtype=dtype)

        if device is not None:
            img = img.to(device=device)

        return img

    def _load_tiff(
        self,
        path: pathlib.Path,
        *,
        grayscale: bool,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        channel: int | str | None,
        downsample: int,
    ) -> Tensor:
        from batchmatch.io.tiff import load_tiff

        tensor, _meta = load_tiff(path, channel=channel, downsample=downsample)

        if grayscale and tensor.shape[1] > 1:
            tensor = tensor.mean(dim=1, keepdim=True)

        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)

        if device is not None:
            tensor = tensor.to(device=device)

        return tensor

    def save(
        self,
        image: Tensor | object,
        path: PathLike,
        *,
        overwrite: bool = False,
        quality: int | object = _DEFAULT,
        convert_uint8: bool | object = _DEFAULT,
    ) -> pathlib.Path:
        quality = self.jpeg_quality if quality is _DEFAULT else int(quality)
        convert_uint8 = self.convert_uint8 if convert_uint8 is _DEFAULT else bool(convert_uint8)

        p = pathify(path)
        ensure_parent_dir(p)
        assert_overwrite(p, overwrite)

        img = _coerce_image(image)
        if img.dtype.is_floating_point and convert_uint8:
            img = image_to_uint8(img)

        img = to_chw(img).detach().cpu()

        suffix = p.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            write_jpeg(img, str(p), quality=quality)
        elif suffix == ".png":
            write_png(img, str(p))
        else:
            raise ValueError(f"Unsupported image extension: {suffix!r} (use .jpg/.jpeg/.png).")

        return p

    def list(
        self,
        directory: PathLike,
        extensions: Optional[Iterable[str]] = None,
    ) -> list[pathlib.Path]:
        exts = coerce_extensions(extensions)
        d = pathify(directory)
        return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)

    def exists(self, path: PathLike) -> bool:
        return pathify(path).exists()
