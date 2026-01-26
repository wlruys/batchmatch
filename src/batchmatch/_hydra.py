from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from batchmatch.search.config import (
    AngleRange,
    ExhaustiveSearchConfig,
    PredicateConfig,
    ScaleRange,
    SearchGridConfig,
    ShearRange,
)

from batchmatch.translate.config import (
    CCTranslationConfig,
    GCCTranslationConfig,
    GNGFTranslationConfig,
    GPCTranslationConfig,
    NCCTranslationConfig,
    NGFTranslationConfig,
    PCTranslationConfig,
)

from batchmatch.gradient.config import (
    BoxRatioGradientConfig,
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
    ROEWAGradientConfig,
    SobelGradientConfig,
)

from batchmatch.warp.config import WarpPipelineConfig

from batchmatch.process.config import (
    CropConfig,
    CropOutputConfig,
    MaskCropConfig,
    PadConfig,
    ProcessConfig,
    RandomCropConfig,
    ResizeConfig,
)

from batchmatch.io.config import IOConfig, OutputConfig

from batchmatch.view.config import CheckerboardSpec, DisplaySpec, OverlaySpec

__all__ = [
    "RegistrationConfig",
    "register_batchmatch_configs",
]


#TODO(wlr): Finish Hydra integration. Basically just adding this as (1) practice and (2) convenience for launching comparison experiments 

@dataclass
class RegistrationConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"search_grid": "rotation_only"},
            {"translation": "gngf"},
            {"gradient": "cd"},
            {"process": "default"},
        ]
    )

    io: IOConfig = field(default_factory=IOConfig)
    search_grid: SearchGridConfig = field(default_factory=SearchGridConfig)
    search: ExhaustiveSearchConfig = field(default_factory=ExhaustiveSearchConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    translation: Any = None
    gradient: Any = None

    device: str = "cpu"
    top_k: int = 1

def register_batchmatch_configs() -> None:
    try:
        from hydra.core.config_store import ConfigStore
    except ImportError:
        return

    cs = ConfigStore.instance()

    cs.store(name="config", node=RegistrationConfig)

    cs.store(
        group="search_grid",
        name="rotation_only",
        node=SearchGridConfig(
            rotation=AngleRange(min_angle=-15.0, max_angle=15.0, step=0.5),
        ),
    )

    cs.store(
        group="search_grid",
        name="rotation_fine",
        node=SearchGridConfig(
            rotation=AngleRange(min_angle=-15.0, max_angle=15.0, step=0.25),
        ),
    )

    cs.store(
        group="search_grid",
        name="rotation_scale",
        node=SearchGridConfig(
            rotation=AngleRange(min_angle=-15.0, max_angle=15.0, step=1.0),
            scale_x=ScaleRange(min_scale=0.9, max_scale=1.1, step=0.02),
            scale_y=ScaleRange(min_scale=0.9, max_scale=1.1, step=0.02),
            predicates=PredicateConfig(max_anisotropy=1.1, area_bounds=(0.8, 1.2)),
        ),
    )

    cs.store(
        group="search_grid",
        name="full_affine",
        node=SearchGridConfig(
            rotation=AngleRange(min_angle=-15.0, max_angle=15.0, step=1.0),
            scale_x=ScaleRange(min_scale=0.9, max_scale=1.1, step=0.02),
            scale_y=ScaleRange(min_scale=0.9, max_scale=1.1, step=0.02),
            shear_x=ShearRange(min_shear=-5.0, max_shear=5.0, step=1.0),
            shear_y=ShearRange(min_shear=-5.0, max_shear=5.0, step=1.0),
            predicates=PredicateConfig(
                max_anisotropy=1.15,
                max_shear_l2=7.0,
            ),
        ),
    )

    cs.store(
        group="translation",
        name="ngf",
        node=NGFTranslationConfig(overlap_fraction=0.99),
    )

    cs.store(
        group="translation",
        name="pc",
        node=PCTranslationConfig(),
    )

    cs.store(
        group="translation",
        name="gpc",
        node=GPCTranslationConfig(p=1.0),
    )

    cs.store(
        group="translation",
        name="ncc",
        node=NCCTranslationConfig(overlap_fraction=0.99),
    )

    cs.store(
        group="translation",
        name="gcc",
        node=GCCTranslationConfig(overlap_fraction=0.99),
    )

    cs.store(
        group="translation",
        name="cc",
        node=CCTranslationConfig(),
    )

    _default_eta = EtaConfig.from_mean(scale=0.2, norm=L2NormConfig())
    _default_normalize = NormalizeConfig(threshold=1e-3)

    cs.store(
        group="gradient",
        name="cd",
        node=CDGradientConfig(
            stencil_width=3,
            eta=_default_eta,
            normalize=_default_normalize,
        ),
    )

    cs.store(
        group="gradient",
        name="cd_wide",
        node=CDGradientConfig(
            stencil_width=5,
            eta=_default_eta,
            normalize=_default_normalize,
        ),
    )

    cs.store(
        group="gradient",
        name="sobel",
        node=SobelGradientConfig(
            eta=_default_eta,
            normalize=_default_normalize,
        ),
    )

    cs.store(
        group="gradient",
        name="roewa",
        node=ROEWAGradientConfig(
            alpha=0.9,
            eta=_default_eta,
            normalize=_default_normalize,
        ),
    )

    cs.store(
        group="gradient",
        name="box_ratio",
        node=BoxRatioGradientConfig(
            width=11,
            eta=_default_eta,
            normalize=_default_normalize,
        ),
    )

    cs.store(
        group="process",
        name="default",
        node=ProcessConfig(
            crop=None,
            resize=ResizeConfig(method="scale", scale=0.25),
            pad=PadConfig(scale=2.0, window_alpha=0.05),
        ),
    )

    cs.store(
        group="process",
        name="highres",
        node=ProcessConfig(
            crop=None,
            resize=ResizeConfig(method="scale", scale=0.5),
            pad=PadConfig(scale=1.5, window_alpha=0.05),
        ),
    )

    cs.store(
        group="process",
        name="target_256",
        node=ProcessConfig(
            crop=None,
            resize=ResizeConfig(method="target", target_width=256),
            pad=PadConfig(scale=2.0),
        ),
    )

    cs.store(
        group="process",
        name="target_512",
        node=ProcessConfig(
            crop=None,
            resize=ResizeConfig(method="target", target_width=512),
            pad=PadConfig(scale=2.0),
        ),
    )

    _random_crop_outputs = CropOutputConfig(
        crop_image=True,
        crop_mask=True,
        adjust_box=True,
    )

    cs.store(
        group="process",
        name="with_crop",
        node=ProcessConfig(
            crop=CropConfig(
                type="random",
                random=RandomCropConfig(
                    min_size=64,
                    max_size=256,
                    min_aspect=0.8,
                    max_aspect=1.2,
                    max_attempts=50,
                    outputs=_random_crop_outputs,
                ),
            ),
            resize=ResizeConfig(method="scale", scale=0.5),
            pad=PadConfig(scale=2.0, window_alpha=0.05, pad_to_pow2=False),
        ),
    )

    cs.store(
        group="process",
        name="with_random_crop",
        node=ProcessConfig(
            crop=CropConfig(
                type="random",
                random=RandomCropConfig(
                    min_size=64,
                    max_size=256,
                    min_aspect=0.8,
                    max_aspect=1.2,
                    max_attempts=50,
                    outputs=_random_crop_outputs,
                ),
            ),
            resize=ResizeConfig(method="scale", scale=0.5),
            pad=PadConfig(scale=2.0, window_alpha=0.05, pad_to_pow2=False),
        ),
    )

    cs.store(
        group="process",
        name="with_intersection_crop",
        node=ProcessConfig(
            crop=CropConfig(
                type="intersection",
                mask=MaskCropConfig(method="intersection"),
            ),
            resize=ResizeConfig(method="scale", scale=0.25),
            pad=PadConfig(scale=2.0),
        ),
    )


def _auto_register():
    import os

    if os.environ.get("BATCHMATCH_HYDRA_AUTO_REGISTER", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        register_batchmatch_configs()


_auto_register()
