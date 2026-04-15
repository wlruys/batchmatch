from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from frozendict import frozendict
except ImportError:
    frozendict = dict  # type: ignore

from batchmatch.base.pipeline import StageSpec, coerce_stage_spec

from .stage import TranslationSearchStage, translation_registry


def _freeze_params(params: Mapping[str, Any] | None) -> frozendict:
    if params is None:
        return frozendict()
    if isinstance(params, frozendict):
        return params
    return frozendict(params)


@dataclass(frozen=True)
class TranslationSearchSpec:

    type: str
    _params: frozendict = field(default_factory=frozendict)

    def __init__(self, type: str, params: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "_params", _freeze_params(params))

    @property
    def params(self) -> frozendict:
        return self._params

    def __repr__(self) -> str:
        if self.params:
            params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            return f"TranslationSearchSpec({self.type!r}, {{{params_str}}})"
        return f"TranslationSearchSpec({self.type!r})"


def coerce_translation_search_spec(
    value: object,
    *,
    field_name: str = "search",
) -> TranslationSearchSpec:
    if isinstance(value, TranslationSearchSpec):
        return value

    if isinstance(value, StageSpec):
        return TranslationSearchSpec(type=value.type, params=dict(value.params))

    if isinstance(value, str):
        return TranslationSearchSpec(type=value, params={})

    if isinstance(value, Mapping):
        raw = dict(value)
        type_name = raw.pop("type", None) or raw.pop("name", None)
        if type_name is None:
            raise ValueError(f"{field_name} mapping must include 'type' or 'name' key")

        params = raw.pop("params", None)
        if params is None:
            params = raw
        elif raw:
            raise ValueError(
                f"{field_name} has unexpected keys alongside 'params': {sorted(raw)}"
            )

        return TranslationSearchSpec(type=str(type_name), params=params)

    raise TypeError(
        f"{field_name} must be TranslationSearchSpec | Mapping | str; "
        f"got {type(value).__name__}"
    )


def build_translation_stage(
    spec: TranslationSearchSpec | StageSpec | Mapping | str | "TranslationConfig",
    **kwargs,
) -> TranslationSearchStage:
    """Build a translation search stage from specification.

        Examples:
            >>> stage = build_translation_stage("ncc")
            >>> stage = build_translation_stage({
            ...     "type": "ncc",
        ...     "overlap_fraction": 0.1,
            ...     "eps": 1e-6,
            ... })
        >>> spec = TranslationSearchSpec("ncc", {"overlap_fraction": 0.1})
        >>> stage = build_translation_stage(spec)
    """
    from batchmatch.translate.config import TranslationConfig

    if isinstance(spec, TranslationConfig):
        method, params = spec.to_method_and_params()
        if kwargs:
            params.update(kwargs)
        return translation_registry.build(method, **params)

    if isinstance(spec, str) and kwargs:
        return translation_registry.build(spec, **kwargs)

    parsed_spec = coerce_translation_search_spec(spec)

    if kwargs:
        merged_params = dict(parsed_spec.params)
        merged_params.update(kwargs)
    else:
        merged_params = dict(parsed_spec.params)

    return translation_registry.build(parsed_spec.type, **merged_params)


def get_translation_stage_class(name: str) -> type[TranslationSearchStage]:
    return translation_registry.get(name)


def list_translation_methods() -> list[str]:
    return sorted(translation_registry.keys())
