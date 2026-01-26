from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Union

from batchmatch.base.pipeline import (
    Pipeline,
    Stage,
    StageSpec,
    coerce_stage_spec,
    should_build_stage,
)
from batchmatch.warp.config import WarpPipelineConfig


@dataclass(frozen=True)
class WarpPipelineSpec:
    prepare: Optional[StageSpec] = None
    image: Optional[StageSpec] = None
    window: Optional[StageSpec] = None
    mask: Optional[StageSpec] = None
    points: Optional[StageSpec] = None
    boxes: Optional[StageSpec] = None
    aux_boxes: Optional[StageSpec] = None
    quad: Optional[StageSpec] = None
    aux_quads: Optional[StageSpec] = None


def _get_warp_registry():
    from batchmatch.warp.stages import warp_registry
    return warp_registry


def _build_warp_stage(spec: StageSpec) -> Stage:
    registry = _get_warp_registry()
    return registry.build(spec.type, **spec.params)


_WARP_STAGE_FIELDS = (
    "prepare", "image", "window", "mask", "points",
    "boxes", "aux_boxes", "quad", "aux_quads",
)

_WARP_OUTPUT_FIELDS = tuple(field for field in _WARP_STAGE_FIELDS if field != "prepare")


def _normalize_warp_outputs(outputs: Sequence[str]) -> list[str]:
    normalized = [str(o) for o in outputs]
    lowered = {o.lower() for o in normalized}
    if "all" in lowered:
        return list(_WARP_OUTPUT_FIELDS)
    allowed = set(_WARP_OUTPUT_FIELDS)
    unknown = {o for o in lowered if o not in allowed}
    if unknown:
        raise ValueError(
            f"Unknown outputs: {sorted(unknown)}. Valid: {sorted(allowed)}"
        )
    ordered = []
    lowered_set = set(lowered)
    for field in _WARP_OUTPUT_FIELDS:
        if field in lowered_set:
            ordered.append(field)
    return ordered


def _build_outputs_spec(
    outputs: Sequence[str],
    prepare: Optional[object],
    stage_overrides: Mapping[str, object],
) -> dict:
    resolved_outputs = _normalize_warp_outputs(outputs)
    spec: dict = {}

    if prepare is not None:
        spec["prepare"] = prepare
    elif resolved_outputs:
        spec["prepare"] = "prepare"

    overrides = dict(stage_overrides)
    for output in resolved_outputs:
        if output in overrides:
            spec[output] = overrides.pop(output)
        else:
            spec[output] = output

    if overrides:
        allowed = set(_WARP_OUTPUT_FIELDS)
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                f"warp_pipeline has unexpected keys: {sorted(unknown)}"
            )
        for key, value in overrides.items():
            spec[key] = value
        if "prepare" not in spec:
            spec["prepare"] = "prepare"

    return spec


def coerce_warp_pipeline_spec(
    value: object,
    *,
    field_name: str = "warp_pipeline",
) -> WarpPipelineSpec:
    if isinstance(value, WarpPipelineSpec):
        return value

    if isinstance(value, str):
        prepare_spec = coerce_stage_spec(value, field_name=f"{field_name}.prepare")
        return WarpPipelineSpec(prepare=prepare_spec)

    if isinstance(value, Mapping):
        raw = dict(value)

        if "outputs" in raw:
            outputs = raw.pop("outputs")
            prepare = raw.pop("prepare", None)
            overrides = {k: raw.pop(k) for k in list(raw.keys()) if k in _WARP_OUTPUT_FIELDS}
            if raw:
                raise ValueError(
                    f"{field_name} has unexpected keys: {sorted(raw.keys())}"
                )
            spec = _build_outputs_spec(outputs, prepare, overrides)
            return coerce_warp_pipeline_spec(spec, field_name=field_name)

        specs = {}

        for stage_field in _WARP_STAGE_FIELDS:
            if stage_field in raw:
                specs[stage_field] = coerce_stage_spec(
                    raw.pop(stage_field),
                    field_name=f"{field_name}.{stage_field}",
                    allow_none=True,
                )

        if raw:
            raise ValueError(
                f"{field_name} has unexpected keys: {sorted(raw.keys())}"
            )

        return WarpPipelineSpec(**specs)

    raise TypeError(
        f"{field_name} must be WarpPipelineSpec | Mapping | str; "
        f"got {type(value).__name__}"
    )


def build_warp_pipeline(
    spec: Union[WarpPipelineSpec, Mapping, str, WarpPipelineConfig] | None = None,
    **kwargs,
) -> Pipeline:
    non_prepare_stages = set(_WARP_STAGE_FIELDS) - {"prepare"}

    if spec is None:
        spec = {
            "prepare": {"type": "prepare"},
            "image": {"type": "image"},
            "window": {"type": "window"},
            "mask": {"type": "mask"},
        }


    if isinstance(spec, WarpPipelineConfig):
        spec = spec.to_spec()

    if isinstance(spec, str) and kwargs:
        prepare_params = {}
        pipeline_args = {}

        for k, v in kwargs.items():
            if k in non_prepare_stages:
                pipeline_args[k] = v
            else:
                prepare_params[k] = v

        spec = {
            "prepare": {"type": spec, **prepare_params},
            **pipeline_args
        }

    if not isinstance(spec, WarpPipelineSpec):
        spec = coerce_warp_pipeline_spec(spec)

    stages: List[Stage] = []

    for stage_field in _WARP_STAGE_FIELDS:
        stage_spec = getattr(spec, stage_field)
        if stage_spec is not None and should_build_stage(stage_spec):
            stages.append(_build_warp_stage(stage_spec))

    return Pipeline(stages)
