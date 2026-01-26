# BatchMatch

A PyTorch library for simple template matching and image registration.
Supports pixel-level translation estimation using FFT-based correlation methods, exhaustive affine parameter search through batched warping and grid search, and some (limited) support for differentiable optimization of affine warp parameters through autograd.

A while ago, as part of my PhD research, I implemented and compared various FFT-based image registration methods to evaluate their accuracy on aligning multimodal remote sensing images. That was a messy research code, with a fairly fragile and non-portable CUDA implementation of the core metrics. This exists as a more easily usable port of many common FFT-based template matching methods, along with some utilities for building registration pipelines. It is intended for research use and experimentation.

Many parts are still missing or incomplete compared to how I would ideally like this to be (see "Known TODOs and Development Notes" at the end), but it is generally usable for basic translation and **coarse** affine registration tasks. 

Local refinement and less-rigid registration methods are not currently supported.

## Features

- **FFT-Based Translation Matching**: Currently includes support for:
  - Cross-correlation (CC)
  - Normalized cross-correlation (NCC)
  - Phase correlation (PC) and gradient phase correlation (GPC)
  - Normalized gradient fields (NGF) and general dot-product powers (GNGF)

- **Exhaustive Affine Search**: Grid-based search over rotation, scale, shear, and translation parameters with configurable predicates to prune non-physical combinations. These comparisons are done by brute force warping for non-translation parameters, and reducing to the top peaks found on the translations surfaces for each warp.

- **MPI Support**: Optional distributed search using `mpi4py` for large-scale exhaustive searches.


## Installation & Environment Management

I recommend using [uv](https://github.com/astral-sh/uv), but a standard pip installation will also work.

```bash
pip install -e .[all] # Install with all extras (mpi, pytest, etc)
```

Examples can be run with `uv run <script>` to ensure the correct environment is used. MPI and other extras can be installed as needed, with `uv run --extras mpi <script>`.

## Quick Start

### Basic Translation Registration

```python
import torch
from batchmatch.base import build_image_td
from batchmatch.io import ImageIO
from batchmatch.process.pad import CenterPad
from batchmatch.translate import build_translation_stage
from batchmatch.process.shift import build_shift_pipeline

# Load images as grayscale
reference = build_image_td(ImageIO(grayscale=True).load("reference.png"))
moving = build_image_td(ImageIO(grayscale=True).load("moving.png"))

# Pad images for FFT
pad = CenterPad(scale=2, outputs=["image", "mask", "window"])
reference, moving = pad(reference, moving)

# Estimate translation using normalized gradient fields, using default gradient settings
search = build_translation_stage("ngf")
moving = search(reference, moving)

print(f"Translation: tx={moving.translation_results.tx:.2f}, ty={moving.translation_results.ty:.2f}")

# Apply shift to align
shift = build_shift_pipeline(source="translation")
aligned = shift(moving)
```

### Exhaustive Affine Search

```python
from batchmatch.search import (
    ExhaustiveWarpSearch,
    ExhaustiveSearchConfig,
    SearchParams,
    AngleRange,
    ScaleRange,
)
from batchmatch.gradient import CDGradientConfig, EtaConfig, L2NormConfig, NormalizeConfig

# Define search grid
search_params = SearchParams(
    angle=AngleRange(min_angle=-15.0, max_angle=15.0, step=1.0),
    scale_x=ScaleRange(min_scale=0.9, max_scale=1.1, step=0.02),
    scale_y=ScaleRange(min_scale=0.9, max_scale=1.1, step=0.02),
)

# Configure search with gradient-based translation metric
config = ExhaustiveSearchConfig(
    translation="gpc",  # Gradient phase correlation
    batch_size=16,
    progress_enabled=True,
    gradient=CDGradientConfig(
        eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
        normalize=NormalizeConfig(norm="l2", threshold=1e-3), 
        #Compute image gradients with centered differences, using regualrized normalization, and filtering out low-magnitude elements 
    ),
)

search = ExhaustiveWarpSearch(search_params, config).to("cuda")
result = search(reference, moving, top_k=1, progress=True)

print(f"Best angle: {result.warp.angle.item():.2f}°")
print(f"Best scale: ({result.warp.scale_x.item():.3f}, {result.warp.scale_y.item():.3f})")
```

### Differentiable Affine Optimization

```python
from batchmatch.optimize import AffineWarpOptimize, OptimizeConfig, MetricConfig, OptimizerConfig
from batchmatch.metric import ImageMetricSpec

config = OptimizeConfig(
    iterations=200,
    device="cuda",
    metric=MetricConfig(spec=ImageMetricSpec("ncc", {"reduction": "mean"})),
    optimizer=OptimizerConfig(
        params={"lr": 0.1},
        param_groups=[
            {"params": ["angle"], "lr": 0.05},
            {"params": ["tx", "ty"], "lr": 0.2},
        ],
    ),
)

result = AffineWarpOptimize(config).optimize(reference, moving)
registered = result.registered
```

## Project Structure

```
batchmatch/
├── base/           # Core types and workflow abstractions (Stage, Pipeline, ImageDetail)
├── gradient/       # Gradient computation (Sobel, Centered Differences, Multiplicative, ROEWA)
├── helpers/        # Utility functions for tensors, images, and math (custom quantile)
├── io/             # Image and tensor I/O utilities
├── metric/         # Intensity based, spatial, similarity metrics (MSE, NCC, NGF, MI)
├── optimize/       # Differentiable optimization pipelines
├── process/        # Image processing stages (crop, pad, resize, window)
├── search/         # Exhaustive grid search and MPI-distributed search
├── translate/      # FFT-based similarity metrics, and translation estimation
├── view/           # Visualization and rendering utilities
└── warp/           # Affine warp transformations
```

## Available Translation Methods

| Method | Name | Description |
|--------|------|-------------|
| `cc` | Cross-correlation | Standard FFT-based cross-correlation |
| `ncc` | Normalized cross-correlation | Intensity-normalized correlation |
| `pc` | Phase correlation | Phase-only correlation, robust to intensity changes |
| `gpc` | Gradient phase correlation | Phase correlation on image gradients |
| `ngf` | Normalized gradient fields | Gradient orientation matching (p=2) |
| `gngf` | Generalized NGF | NGF with configurable power parameter |


## Known TODOs and Development Notes

Some dev notes, and missing metrics, features, etc that I may add in the future as time permits:
- Quantile and local filtering & normalization for gradients is currently not implemented.
- Cell size estimation (for use in microscopy) is very fragile. A more robust segmentation-based approach would be better.
- Support for cross-power filtering via magnitude powers in phase correlation methods (generalizes the space between PC and standard CC by adjusting the power on the magnitude normalization).
- Multi-scale optimization pipelines and automatic refinement (exhaustive and differentiable)
- Subpixel peak estimation on translation search
- Hydra integration for all builders and configuration-based pipeline construction (to more easily dispatch experiments to different methods and parameters)
- Masking support in spatial metrics is limited. 
- Performance Optimizations:
    - Debug torch compile performance issues with full pipeline vs isolated stages
    - Non-shifted versions of translation search to avoid copy overheads
- Differentiable FFT-based translation estimation (with soft-argmax peak finding / thresholding) and pass-through gradients.

## AI Usage Notice

The core methods and implementations in this repository were manually ported by me from earlier NumPy/OpenCV/CUDA research code into PyTorch. AI tools (Claude Code) were used to assist with refactoring, interface experimentation, and generating integration-test scaffolding.
This repo was used as a testbed to gain experience and understand the usage of modern generative tooling.
As a result, you'll notice that there's like three different vestigial ways to build registration pipelines in here, as a result. Sorry about that, the one used and documented in "register_cells.py" is the one I recommend using, but there are no plans to deprecate the others at this time.