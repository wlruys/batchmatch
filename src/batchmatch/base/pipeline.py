from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import torch
import torch.nn as nn
import inspect
from tensordict.nn import TensorDictModuleBase
from batchmatch.base.tensordicts import ImageDetail, NestedKey
from tensordict import TensorDict
Tensor = torch.Tensor

T = TypeVar("T", bound="Stage")
S = TypeVar("S", bound=type)

from frozendict import frozendict


class StageRegistry:
    """
    Register and construct Stage subclasses by name.

    Use this to create category-specific registries and build stages from
    string identifiers.

    Args:
        name: Registry name used in error messages.

    Examples:
        >>> gradient_registry = StageRegistry("gradient")
        >>> @gradient_registry.register("sobel")
        ... class SobelGradient(Stage):
        ...     ...
        >>> stage = gradient_registry.build("sobel")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type[Stage]] = {}

    def register(self, *names: str) -> Callable[[S], S]:
        """
        Register a Stage class under one or more names.

        Args:
            names: One or more names to register (case-insensitive).

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If no names are provided or a name is duplicated.
        """
        if not names:
            raise ValueError("At least one name must be provided")

        def decorator(cls: S) -> S:
            for name in names:
                key = name.lower()
                if key in self._registry:
                    existing = self._registry[key]
                    raise ValueError(
                        f"Duplicate registration in '{self.name}' registry: "
                        f"'{key}' already registered to {existing.__name__}"
                    )
                self._registry[key] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs: Any) -> "Stage":
        """
        Build a stage instance from a registered name.

        Args:
            name: Registered name of the stage (case-insensitive).
            **kwargs: Arguments passed to the stage constructor.

        Returns:
            Instantiated stage.

        Raises:
            ValueError: If the name is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            raise ValueError(
                f"Unknown {self.name} type '{name}'. "
                f"Options: {sorted(self._registry.keys())}"
            )
        return self._registry[key](**kwargs)

    def get(self, name: str) -> type["Stage"]:
        """
        Get a registered class by name without instantiating.

        Args:
            name: Registered name of the stage (case-insensitive).

        Returns:
            Stage subclass registered under the given name.

        Raises:
            ValueError: If the name is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            raise ValueError(
                f"Unknown {self.name} type '{name}'. "
                f"Options: {sorted(self._registry.keys())}"
            )
        return self._registry[key]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._registry

    def __iter__(self):
        return iter(self._registry)

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()

    @property
    def options(self) -> set[str]:
        """
        Get the set of all registered names.

        Returns:
            Registered names.
        """
        return set(self._registry.keys())


def _register_scalar(
    module: nn.Module,
    name: str,
    value: float,
    *,
    persistent: bool = True,
) -> None:
    """
    Register a scalar buffer on a module.

    Args:
        module: Module to receive the buffer.
        name: Buffer name.
        value: Scalar value to store.
        persistent: Whether the buffer should be persistent.
    """
    module.register_buffer(
        name,
        torch.tensor(value, dtype=torch.float32),
        persistent=persistent,
    )


def _register_optional_scalar(
    module: nn.Module,
    name: str,
    value: float | None,
    *,
    persistent: bool = False,
) -> None:
    """
    Register an optional scalar buffer on a module.

    Args:
        module: Module to receive the buffer.
        name: Buffer name.
        value: Scalar value to store, or None.
        persistent: Whether the buffer should be persistent when set.
    """
    if value is None:
        module.register_buffer(name, None, persistent=False)
    else:
        module.register_buffer(
            name,
            torch.tensor(value, dtype=torch.float32),
            persistent=persistent,
        )


def _as_tuplekey(key: NestedKey) -> tuple[str, ...]:
    """
    Normalize a nested key to tuple form.

    Args:
        key: Nested key as a string or tuple.

    Returns:
        Tuple representation of the key.
    """
    return (key,) if isinstance(key, str) else tuple(key)


def _freeze_params(params: Mapping[str, Any] | None) -> frozendict:
    """
    Convert a mapping to a frozendict for immutability and hashability.

    Args:
        params: Mapping of parameters to freeze.

    Returns:
        Frozen mapping of parameters.
    """
    if params is None:
        return frozendict()
    if isinstance(params, frozendict):
        return params
    return frozendict(params)


@dataclass(frozen=True, slots=True)
class StageSpec:
    """
    Define an immutable stage specification.

    Use this to store a stage name and its constructor parameters in a
    hashable form.

    Args:
        type: Registered name or class identifier for the stage.
        params: Keyword arguments for the stage constructor.

    Examples:
        >>> StageSpec("gaussian_blur", {"kernel_size": 5, "sigma": 1.0})
    """

    type: str
    _params: frozendict = field(default_factory=frozendict)

    def __init__(self, type: str, params: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "_params", _freeze_params(params))

    @property
    def params(self) -> frozendict:
        """
        Get the frozen parameters mapping.

        Returns:
            Frozen parameter mapping.
        """
        return self._params

    def __repr__(self) -> str:
        if self.params:
            params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            return f"StageSpec({self.type!r}, {{{params_str}}})"
        return f"StageSpec({self.type!r})"


@dataclass(slots=True)
class StageSpecConf:
    """
    Store mutable configuration for Hydra/OmegaConf.

    Convert to StageSpec via `coerce_stage_spec()` before use.

    Args:
        type: Registered name or class identifier for the stage.
        params: Keyword arguments for the stage constructor.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)


def coerce_stage_spec(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> StageSpec | None:
    """
    Coerce various input formats into a StageSpec.

    Args:
        value: Input to coerce into a StageSpec.
        field_name: Name used in error messages for context.
        allow_none: Whether None is allowed and returned as None.

    Returns:
        Coerced stage specification, or None when allowed.

    Raises:
        TypeError: If the value is not a supported specification type.
        ValueError: If the value is None when not allowed or is malformed.

    Examples:
        >>> coerce_stage_spec("blur", field_name="preprocessor")
        StageSpec('blur')
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")

    if isinstance(value, StageSpec):
        return value

    if isinstance(value, str):
        return StageSpec(type=value, params={})

    if isinstance(value, StageSpecConf):
        return StageSpec(
            type=str(value.type),
            params=dict(value.params),
        )

    # Support other dataclass conf types with ``type`` and ``params`` fields.
    # This catches Hydra structured configs and any user-defined dataclass
    # that mirrors StageSpecConf's interface (must be a dataclass *instance*
    # with both ``type: str`` and ``params: dict`` attributes).
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "type") and hasattr(value, "params"):
            return StageSpec(
                type=str(getattr(value, "type")),
                params=dict(getattr(value, "params")),
            )

    if isinstance(value, Mapping):
        return _parse_stage_mapping(dict(value), field_name)

    raise TypeError(
        f"{field_name} must be StageSpec | Mapping | str; "
        f"got {type(value).__name__}"
    )


def _parse_stage_mapping(raw: dict[str, Any], field_name: str) -> StageSpec:
    """
    Parse a dictionary into a StageSpec.

    .. note:: This function mutates *raw* via ``.pop()``.  Callers that
       need to preserve the original mapping should pass a copy.

    Args:
        raw: Mapping containing a stage spec (will be mutated).
        field_name: Name used in error messages for context.

    Returns:
        Parsed stage specification.

    Raises:
        ValueError: If required keys are missing or unexpected keys appear.
    """
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

    return StageSpec(type=str(type_name), params=params)


def coerce_stage_list(
    value: object,
    *,
    field_name: str,
) -> list[StageSpec]:
    """
    Coerce a sequence of values into a list of StageSpecs.

    Args:
        value: Sequence of stage specifications, or None for an empty list.
        field_name: Name used in error messages.

    Returns:
        Coerced list of stage specifications.

    Raises:
        TypeError: If the value is not a sequence or None.
    """
    if value is None:
        return []

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            coerce_stage_spec(v, field_name=f"{field_name}[{i}]", allow_none=False)
            for i, v in enumerate(value)
        ]

    raise TypeError(f"{field_name} must be a sequence of stages or None")


def should_build_stage(spec: StageSpec | None) -> bool:
    """
    Check whether a stage spec should be instantiated.

    Args:
        spec: Stage specification or None.

    Returns:
        True when a stage should be built.
    """
    if spec is None:
        return False
    type_lower = spec.type.lower()
    return type_lower not in {"none", "null", "skip", "identity"}


def build_stage_list(
    specs: Sequence[StageSpec],
    builder: Callable[..., T],
) -> list[T]:
    """
    Build a list of stages from specifications.

    Args:
        specs: Stage specifications to build.
        builder: Factory function `builder(type_name, **params) -> Stage`.

    Returns:
        Instantiated stages, skipping identity specifications.
    """
    stages: list[T] = []
    for spec in specs:
        if should_build_stage(spec):
            stages.append(builder(spec.type, **spec.params))
    return stages


def register_structured_config(
    *,
    group: str,
    name: str,
    node: object,
) -> None:
    """
    Register a structured config with Hydra's ConfigStore.

    This is a no-op if Hydra is not installed.

    Args:
        group: Config group (e.g., "stage", "pipeline").
        name: Config name within the group.
        node: Structured config class or instance.
    """
    try:
        from hydra.core.config_store import ConfigStore

        cs = ConfigStore.instance()
        cs.store(group=group, name=name, node=node)
    except ImportError:
        pass


class Stage(TensorDictModuleBase, ABC):
    """
    Define a pipeline stage operating on TensorDict containers.

    Stages declare data dependencies via class attributes:
    `requires` (inputs), `sets` (outputs), and `invalidates` (stale keys).
    Subclasses implement `forward(image) -> image`.
    """

    # Class-level defaults
    requires: frozenset[NestedKey] = frozenset()
    sets: frozenset[NestedKey] = frozenset()
    invalidates: frozenset[NestedKey] = frozenset()

    _auto_validate: Final[bool] = True
    _auto_invalidate: Final[bool] = True

    _use_lazy_invalidate: Final[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._in_keys: list[NestedKey] = list(self.requires)
        self._out_keys: list[NestedKey] = list(self.sets)

        sig = inspect.signature(self.forward)
        self._has_varargs = any(
            p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()
        )

        self._requires_tuple: tuple[NestedKey, ...] = tuple(self.requires)
        self._last_validated_id: int | None = None

    @property
    def in_keys(self) -> list[NestedKey]:
        return self._in_keys

    @in_keys.setter
    def in_keys(self, keys: Sequence[NestedKey]) -> None:
        self._in_keys = list(keys)

    @property
    def out_keys(self) -> list[NestedKey]:
        return self._out_keys

    @out_keys.setter
    def out_keys(self, keys: Sequence[NestedKey]) -> None:
        self._out_keys = list(keys)

    def __call__(
        self, 
        image: TensorDict | list[TensorDict], 
        *args: TensorDict
    ) -> TensorDict | list[TensorDict]:
        """Execute the stage with validation and invalidation.

        For a single ``TensorDict``, validates, runs ``forward``, and
        invalidates.  For multiple inputs (a list or extra positional
        args) delegates to :meth:`call_batch`.

        Args:
            image: Input container or list of containers.
            *args: Additional input containers.

        Returns:
            Processed container or list of containers.
        """
        if not args and not isinstance(image, list):
            if self._auto_validate:
                img_id = id(image)
                if img_id != self._last_validated_id:
                    self._validate(image)
                    self._last_validated_id = img_id
            result = self.forward(image)
            if self._auto_invalidate:
                self._invalidate(result)
            return result

        if isinstance(image, list):
            inputs = image + list(args)
        else:
            inputs = [image] + list(args)

        return self.call_batch(inputs)

    def call_batch(
        self,
        images: list[TensorDict],
    ) -> list[TensorDict]:
        """Process multiple containers.

        Validates each input, then either calls ``forward`` once with the
        full list (when the subclass accepts ``*args``) or maps
        ``forward`` over each element individually.

        Args:
            images: List of input containers to process.

        Returns:
            List of processed containers.
        """
        if self._auto_validate:
            for img in images:
                self._validate(img)

        if self._has_varargs:
            result = self.forward(images)
        else:
            result = [self.forward(img) for img in images]

        if self._auto_invalidate:
            for res in result:
                self._invalidate(res)

        return result

    @abstractmethod
    def forward(self, image: TensorDict) -> TensorDict:
        """
        Process a single image container.

        Args:
            image: Input image container.

        Returns:
            Processed image container.
        """
        ...

    def _validate(self, image: TensorDict) -> None:
        """
        Verify required keys are present and not stale.

        Uses early-exit for faster validation when keys are missing.

        Args:
            image: Input container to validate.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If required keys are stale.
        """
        if not self._requires_tuple:
            return

        for k in self._requires_tuple:
            if k not in image:
                raise KeyError(
                    f"{type(self).__name__} missing required key: {k}"
                )

        if hasattr(image, "is_stale"):
            for k in self._requires_tuple:
                if image.is_stale(k):
                    raise ValueError(
                        f"{type(self).__name__} found stale required key: {k}"
                    )

    def _lazy_invalidate(self, image: TensorDict) -> None:
        """
        Mark invalidated keys as stale without immediate removal.

        Args:
            image: Container to update.
        """
        for key in self.invalidates:
            if key in image:
                image.mark_stale(key)

        for key in self.sets:
            image.mark_fresh(key)

    def _eager_invalidate(self, image: TensorDict) -> None:
        """
        Remove invalidated keys from the container immediately.

        Args:
            image: Container to update.
        """
        for key in self.invalidates:
            if key in image:
                image.del_(key)

    def _invalidate(self, image: TensorDict) -> None:
        """
        Invalidate keys according to the chosen strategy.

        Args:
            image: Container to update.
        """
        if self._use_lazy_invalidate:
            self._lazy_invalidate(image)
        else:
            self._eager_invalidate(image)

    def __rshift__(self, other: "Stage") -> "Pipeline":
        if not isinstance(other, Stage):
            return NotImplemented
        return _compose_stages(self, other)

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.requires:
            parts.append(f"requires={set(self.requires)}")
        if self.sets:
            parts.append(f"sets={set(self.sets)}")
        return f"<{', '.join(parts)}>"


class Pipeline(Stage):
    """
    Compose stages sequentially with dependency tracking.

    Aggregates `requires` and `sets` across child stages to represent the
    effective pipeline inputs and outputs.

    Args:
        stages: Sequence of stages to execute in order.
    """
    _auto_validate: Final[bool] = False
    _auto_invalidate: Final[bool] = False

    def __init__(self, stages: Sequence[Stage]) -> None:
        # Compute dependencies before calling super().__init__()
        # because Stage.__init__ accesses self.requires
        satisfied: set[NestedKey] = set()
        required: set[NestedKey] = set()
        produced: set[NestedKey] = set()

        for stage in stages:
            required.update(stage.requires - satisfied)
            satisfied.update(stage.sets)
            produced.update(stage.sets)

        self._requires = frozenset(required)
        self._sets = frozenset(produced)

        super().__init__()
        self.stages: nn.ModuleList = nn.ModuleList(list(stages))

        self._in_keys = list(required)
        self._out_keys = list(produced)

    @property
    def requires(self) -> frozenset[str]:  # type: ignore[override]
        """
        Get aggregated requirements not satisfied internally.

        Returns:
            Required keys for the pipeline.
        """
        return self._requires

    @property
    def sets(self) -> frozenset[str]:  # type: ignore[override]
        """
        Get aggregated outputs from all stages.

        Returns:
            Output keys produced by the pipeline.
        """
        return self._sets

    def forward(
        self, 
        image: TensorDict | list[TensorDict], 
        *args: TensorDict
    ) -> TensorDict | list[TensorDict]:
        """
        Execute all stages in sequence.

        Calls each stage's forward() directly to skip per-stage validation
        and invalidation overhead. Stale tracking should be handled after
        pipeline completion if needed.

        Args:
            image: Input container or list of containers.
            *args: Additional input containers.

        Returns:
            Processed container or list of containers.
        """
        if not args and not isinstance(image, list):
            current = image
            for stage in self.stages:
                current = stage.forward(current)
            return current

        if isinstance(image, list):
            current = image + list(args)
        else:
            current = [image] + list(args)

        for stage in self.stages:
            if stage._has_varargs:
                current = stage.forward(current)
            else:
                current = [stage.forward(img) for img in current]

        if len(current) == 1:
            return current[0]

        return current

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> Stage:
        return self.stages[index]

    def __iter__(self):
        return iter(self.stages)

    def __repr__(self) -> str:
        stage_names = [type(s).__name__ for s in self.stages]
        return f"Pipeline([{', '.join(stage_names)}])"
    

    def __rshift__(self: T, other: T) -> T:
        if not isinstance(other, Stage):
            return NotImplemented
        return _compose_stages(self, other)


def _compose_stages(left: Stage, right: Stage) -> Pipeline:
    """Flatten and compose two stages into a Pipeline."""
    left_stages = list(left.stages) if isinstance(left, Pipeline) else [left]
    right_stages = list(right.stages) if isinstance(right, Pipeline) else [right]
    return Pipeline(left_stages + right_stages)


class DropStaleStage(Stage):
    _auto_validate: Final[bool] = False  # No requirements
    _auto_invalidate: Final[bool] = False  # No invalidations

    def forward(self, image: TensorDict) -> TensorDict:
        image.drop_stale()
        return image


class RemoveStage(Stage):
    _auto_validate: Final[bool] = False  # No requirements

    def __init__(self, keys: Iterable[NestedKey]) -> None:
        super().__init__()
        self._keys: frozenset[NestedKey] = frozenset(keys)

    @property
    def invalidates(self) -> frozenset[NestedKey]:  # type: ignore[override]
        return self._keys

    def forward(self, image: TensorDict) -> TensorDict:
        for key in self._keys:
            if key in image:
                image.del_(key)
        return image

    def __repr__(self) -> str:
        return f"RemoveStage({set(self._keys)})"


class ZeroStage(Stage):
    def __init__(
        self,
        keys: Iterable[NestedKey],
        *,
        inplace: bool = True,
    ) -> None:
        self._keys: frozenset[NestedKey] = frozenset(keys)
        self._inplace: bool = inplace
        super().__init__()
        self._in_keys = list(self._keys)

    @property
    def requires(self) -> frozenset[NestedKey]:
        return self._keys

    def forward(self, image: TensorDict) -> TensorDict:
        for key in self._keys:
            if key in image:
                tensor = image.get(key)
                if self._inplace:
                    tensor.zero_()
                else:
                    image.set(key, torch.zeros_like(tensor))
        return image

    def __repr__(self) -> str:
        mode = "inplace" if self._inplace else "copy"
        return f"ZeroStage({set(self._keys)}, {mode})"
