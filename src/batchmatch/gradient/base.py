from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from batchmatch.base.tensordicts import ImageDetail
from batchmatch.base.pipeline import (
    Pipeline,
    Stage,
    StageRegistry,
    StageSpec,
    coerce_stage_spec,
    should_build_stage,
)

if TYPE_CHECKING:
    from batchmatch.base.tensordicts import NestedKey


def _get_registry(category: str) -> StageRegistry:
    if category == "gradient":
        from batchmatch.gradient.gradient import gradient_registry
        return gradient_registry
    elif category == "norm":
        from batchmatch.gradient.norm import norm_registry
        return norm_registry
    elif category == "eta":
        from batchmatch.gradient.normalize import eta_registry
        return eta_registry
    elif category == "normalize":
        from batchmatch.gradient.normalize import normalize_registry
        return normalize_registry
    elif category == "process":
        from batchmatch.gradient.process import process_registry
        return process_registry
    else:
        raise ValueError(f"Unknown stage category '{category}'")


@dataclass(frozen=True)
class GradientStageSpec:
    stage: StageSpec
    norm: Optional[StageSpec] = None


@dataclass(frozen=True)
class GradientPipelineSpec:
    """
    Define a gradient pipeline and its execution order.

    Stages run in order:
    1. gradient: Compute GRAD.X and GRAD.Y from the image
    2. preprocess: Optional preprocessing stages
    3. eta: Optional eta computation
    4. normalize: Optional gradient normalization
    5. postprocess: Optional postprocessing stages

    Args:
        gradient: Gradient stage specification.
        preprocess: Optional preprocessing stage specifications.
        eta: Optional eta stage specification.
        normalize: Optional normalization stage specification.
        postprocess: Optional postprocessing stage specifications.
    """
    gradient: StageSpec
    preprocess: Optional[Tuple[GradientStageSpec, ...]] = None
    eta: Optional[GradientStageSpec] = None
    normalize: Optional[GradientStageSpec] = None
    postprocess: Optional[Tuple[GradientStageSpec, ...]] = None


def _build_stage(category: str, spec: StageSpec) -> Stage:
    registry = _get_registry(category)
    return registry.build(spec.type, **spec.params)


def _stage_requires_norm(stage: Stage) -> bool:
    return ImageDetail.Keys.GRAD.NORM in stage.requires


def _build_gradient_stage_spec(
    grad_spec: GradientStageSpec,
    category: str,
) -> List[Stage]:
    stages: List[Stage] = []

    if not should_build_stage(grad_spec.stage):
        return stages

    main_stage = _build_stage(category, grad_spec.stage)

    # If stage requires norm and norm spec is provided, add norm stage first
    if _stage_requires_norm(main_stage) and grad_spec.norm is not None:
        if should_build_stage(grad_spec.norm):
            norm_stage = _build_stage("norm", grad_spec.norm)
            stages.append(norm_stage)

    stages.append(main_stage)
    return stages


def coerce_gradient_stage_spec(
    value: object,
    *,
    field_name: str,
    allow_none: bool = True,
) -> Optional[GradientStageSpec]:
    """
    Coerce input formats into a GradientStageSpec.

    Args:
        value: Stage specification input.
        field_name: Field name used in error messages.
        allow_none: Whether to allow None values.

    Returns:
        A GradientStageSpec or None when allowed.

    Raises:
        ValueError: If a required value is missing or unexpected keys exist.
        TypeError: If the input type is unsupported.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")

    if isinstance(value, GradientStageSpec):
        return value

    if isinstance(value, str):
        stage_spec = coerce_stage_spec(value, field_name=f"{field_name}.stage")
        return GradientStageSpec(stage=stage_spec, norm=None)

    if isinstance(value, Mapping):
        raw = dict(value)

        if "stage" in raw:
            stage_val = raw.pop("stage")
            norm_val = raw.pop("norm", None)

            if raw:
                raise ValueError(
                    f"{field_name} has unexpected keys: {sorted(raw.keys())}"
                )

            stage_spec = coerce_stage_spec(stage_val, field_name=f"{field_name}.stage")
            norm_spec = coerce_stage_spec(
                norm_val, field_name=f"{field_name}.norm", allow_none=True
            )
            return GradientStageSpec(stage=stage_spec, norm=norm_spec)

        # Handle flat dict with optional 'norm' key
        norm_val = raw.pop("norm", None)
        # Remainder is the stage spec
        stage_spec = coerce_stage_spec(raw, field_name=f"{field_name}.stage")
        norm_spec = coerce_stage_spec(
            norm_val, field_name=f"{field_name}.norm", allow_none=True
        )
        return GradientStageSpec(stage=stage_spec, norm=norm_spec)

    raise TypeError(
        f"{field_name} must be GradientStageSpec | Mapping | str | None; "
        f"got {type(value).__name__}"
    )


def coerce_gradient_stage_spec_list(
    value: object,
    *,
    field_name: str,
) -> Optional[Tuple[GradientStageSpec, ...]]:
    """
    Coerce input formats into a tuple of GradientStageSpecs.

    Args:
        value: Stage specification input.
        field_name: Field name used in error messages.

    Returns:
        A tuple of GradientStageSpecs or None.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if value is None:
        return None

    # Already a tuple of GradientStageSpecs
    if isinstance(value, tuple) and all(isinstance(v, GradientStageSpec) for v in value):
        return value

    # Single GradientStageSpec -> wrap in tuple
    if isinstance(value, GradientStageSpec):
        return (value,)

    # String or Mapping -> single stage spec
    if isinstance(value, (str, Mapping)):
        spec = coerce_gradient_stage_spec(value, field_name=field_name, allow_none=False)
        return (spec,) if spec is not None else None

    # Sequence (list) of specs
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        specs = []
        for i, item in enumerate(value):
            spec = coerce_gradient_stage_spec(
                item, field_name=f"{field_name}[{i}]", allow_none=False
            )
            if spec is not None:
                specs.append(spec)
        return tuple(specs) if specs else None

    raise TypeError(
        f"{field_name} must be GradientStageSpec | Mapping | str | Sequence | None; "
        f"got {type(value).__name__}"
    )


def coerce_gradient_pipeline_spec(
    value: object,
    *,
    field_name: str = "gradient_pipeline",
) -> GradientPipelineSpec:
    """
    Coerce input formats into a GradientPipelineSpec.

    Args:
        value: Pipeline specification input.
        field_name: Field name used in error messages.

    Returns:
        A GradientPipelineSpec instance.

    Raises:
        ValueError: If required keys are missing or unexpected keys exist.
        TypeError: If the input type is unsupported.
    """
    if isinstance(value, GradientPipelineSpec):
        return value

    if isinstance(value, str):
        gradient_spec = coerce_stage_spec(value, field_name=f"{field_name}.gradient")
        return GradientPipelineSpec(gradient=gradient_spec)

    if isinstance(value, Mapping):
        raw = dict(value)

        gradient_val = raw.pop("gradient", None)
        if gradient_val is None:
            raise ValueError(f"{field_name} must include 'gradient' key")

        gradient_spec = coerce_stage_spec(
            gradient_val, field_name=f"{field_name}.gradient"
        )

        preprocess = coerce_gradient_stage_spec_list(
            raw.pop("preprocess", None),
            field_name=f"{field_name}.preprocess",
        )
        eta = coerce_gradient_stage_spec(
            raw.pop("eta", None),
            field_name=f"{field_name}.eta",
        )
        normalize = coerce_gradient_stage_spec(
            raw.pop("normalize", None),
            field_name=f"{field_name}.normalize",
        )
        postprocess = coerce_gradient_stage_spec_list(
            raw.pop("postprocess", None),
            field_name=f"{field_name}.postprocess",
        )

        if raw:
            raise ValueError(
                f"{field_name} has unexpected keys: {sorted(raw.keys())}"
            )

        return GradientPipelineSpec(
            gradient=gradient_spec,
            preprocess=preprocess,
            eta=eta,
            normalize=normalize,
            postprocess=postprocess,
        )

    raise TypeError(
        f"{field_name} must be GradientPipelineSpec | Mapping | str; "
        f"got {type(value).__name__}"
    )


def build_gradient_pipeline(
    spec: Union[GradientPipelineSpec, Mapping, str, "GradientPipelineConfig"],
    *,
    build_complex: bool = False,
    **kwargs,
) -> Pipeline:
    """
    Build a gradient processing pipeline from a specification.

    Args:
        spec: Pipeline specification. Can be a GradientPipelineSpec object,
            a dict with "gradient" and optional stage specs, or a string
            for a gradient-only pipeline (e.g., "sobel").
        build_complex: If True, append a final stage that computes the complex
            gradient `GRAD.I = GRAD.X + 1j * GRAD.Y` after all postprocessing.
        **kwargs: Parameters for the gradient stage when spec is a string.

    Returns:
        Constructed gradient processing pipeline.

    Examples:
        >>> # Simple gradient-only pipeline
        >>> pipeline = build_gradient_pipeline("sobel")

        >>> # Gradient pipeline with params
        >>> pipeline = build_gradient_pipeline("sobel", kernel_size=5)

        >>> # Gradient-only pipeline that also outputs GRAD.I
        >>> pipeline = build_gradient_pipeline("sobel", build_complex=True)

        >>> # Full pipeline with normalization
        >>> pipeline = build_gradient_pipeline({
        ...     "gradient": {"type": "sobel"},
        ...     "eta": {
        ...         "stage": {"type": "mean", "scale": 0.2},
        ...         "norm": {"type": "l2"},  # Auto-inserted before eta
        ...     },
        ...     "normalize": {"type": "normalize"},
        ... })

        >>> # Pipeline with multiple preprocess and postprocess stages
        >>> pipeline = build_gradient_pipeline({
        ...     "gradient": {"type": "sobel"},
        ...     "preprocess": [
        ...         {"type": "flip", "horizontal": True},
        ...         {"type": "mean_whiten"},
        ...     ],
        ...     "postprocess": [
        ...         {"type": "mean_scale", "scale": 2.0},
        ...     ],
        ... })
    """
    from batchmatch.gradient.config import GradientMethodConfig, GradientPipelineConfig

    if isinstance(spec, GradientMethodConfig):
        spec = spec.to_pipeline_config()

    if isinstance(spec, GradientPipelineConfig):
        if not build_complex:
            build_complex = spec.build_complex
        spec = spec.to_spec_dict()

    if isinstance(spec, str) and kwargs:
        # Separate kwargs into stage params and pipeline components
        pipeline_components = {"preprocess", "eta", "normalize", "postprocess"}

        gradient_params = {}
        pipeline_args = {}

        for k, v in kwargs.items():
            if k in pipeline_components:
                pipeline_args[k] = v
            else:
                gradient_params[k] = v

        # Construct spec
        spec = {
            "gradient": {"type": spec, **gradient_params},
            **pipeline_args
        }

    # Coerce to GradientPipelineSpec
    if not isinstance(spec, GradientPipelineSpec):
        spec = coerce_gradient_pipeline_spec(spec)

    stages: List[Stage] = []

    # 1. Gradient operator (required)
    gradient_stage = _build_stage("gradient", spec.gradient)
    stages.append(gradient_stage)

    # 2. Preprocess (optional, can be a list of stages)
    if spec.preprocess is not None:
        for preprocess_spec in spec.preprocess:
            stages.extend(_build_gradient_stage_spec(preprocess_spec, "process"))

    # 3. Eta computation (optional)
    if spec.eta is not None:
        stages.extend(_build_gradient_stage_spec(spec.eta, "eta"))

    # 4. Normalization (optional)
    if spec.normalize is not None:
        stages.extend(_build_gradient_stage_spec(spec.normalize, "normalize"))

    # 5. Postprocess (optional, can be a list of stages)
    if spec.postprocess is not None:
        for postprocess_spec in spec.postprocess:
            stages.extend(_build_gradient_stage_spec(postprocess_spec, "process"))

    if build_complex:
        stages.append(_build_stage("process", StageSpec("complex")))

    return Pipeline(stages)
