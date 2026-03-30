from __future__ import annotations

from batchmatch.io.config import IOConfig, OutputConfig
from batchmatch.io.convert import to_bchw_float
from batchmatch.io.downsample import area_downsample, crop_region, validate_region
from batchmatch.io.images import ImageIO
from batchmatch.io.imagedetail import ImageDetailIO
from batchmatch.io.product import ProductIO, RegistrationProduct
from batchmatch.io.tensors import TensorIO
from batchmatch.io.tiff import load_tiff, DownsampleMode, ReturnMode
from batchmatch.io.tiff_lazy import LazyTiffReader
from batchmatch.io.tiff_meta import TiffMeta, read_meta

__all__ = [
    # Config classes
    "IOConfig",
    "OutputConfig",
    # Image I/O
    "ImageIO",
    "TensorIO",
    # TIFF I/O
    "load_tiff",
    "read_meta",
    "TiffMeta",
    "LazyTiffReader",
    "DownsampleMode",
    "ReturnMode",
    # Conversion / spatial ops
    "to_bchw_float",
    "area_downsample",
    "crop_region",
    "validate_region",
    # ImageDetail I/O
    "ImageDetailIO",
    # Product I/O
    "RegistrationProduct",
    "ProductIO",
]
