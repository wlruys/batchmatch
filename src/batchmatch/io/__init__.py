"""Image and tensor I/O: TIFF / raster loading, saving, and registered exports."""
from __future__ import annotations

from batchmatch.io._tiff_meta import TiffMeta
from batchmatch.io.config import IOConfig, OutputConfig
from batchmatch.io.convert import to_bchw_float
from batchmatch.io.downsample import area_downsample, crop_region, validate_region
from batchmatch.io.imagedetail import ImageDetailIO
from batchmatch.io.images import (
    ImageIO,
    ImagePolicy,
    ImageSource,
    PreviewConfig,
    RasterSource,
    SaveOptions,
    TiffExportConfig,
    TiffSource,
    load_image,
    open_image,
    save_image,
    save_preview,
    save_tiff,
)
from batchmatch.io.space import (
    ImageSpace,
    PaddingLTRB,
    RegionYXHW,
    SourceFormat,
    SourceInfo,
    SpatialImage,
)
from batchmatch.io.product import ProductIO
from batchmatch.io.export import (
    RegisteredExport,
    export_moving_mask_tiff,
    export_registered,
    export_stacked_registered_tiff,
)
from batchmatch.io.tensors import TensorIO

__all__ = [
    # Config
    "IOConfig",
    "OutputConfig",
    # Image I/O
    "ImageIO",
    "ImagePolicy",
    "ImageSource",
    "RasterSource",
    "PreviewConfig",
    "SaveOptions",
    "TiffExportConfig",
    "TiffSource",
    "load_image",
    "open_image",
    "save_image",
    "save_preview",
    "save_tiff",
    # Tensor I/O
    "TensorIO",
    # TIFF metadata (re-exported for introspection)
    "TiffMeta",
    # Spatial ops
    "to_bchw_float",
    "area_downsample",
    "crop_region",
    "validate_region",
    # Detail / product I/O
    "ImageDetailIO",
    "ProductIO",
    "RegisteredExport",
    "export_moving_mask_tiff",
    "export_registered",
    "export_stacked_registered_tiff",
    # Spatial types
    "ImageSpace",
    "PaddingLTRB",
    "RegionYXHW",
    "SourceFormat",
    "SourceInfo",
    "SpatialImage",
]
