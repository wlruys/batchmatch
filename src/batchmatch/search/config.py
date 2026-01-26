"""Configuration objects for grid-based warp parameter search."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "AP",
    "AngleRange",
    "ScaleRange",
    "ShearRange",
    "AxisGroup",
    "IterationOrder",
    "DEFAULT_ORDER",
    "SearchParams",
    "GridAP",
    "grid_for_rank",
    "make_angle_predicate",
    "make_scale_predicate",
    "make_shear_predicate",
    "ExhaustiveSearchConfig",
    "PredicateConfig",
    "SearchGridConfig",
]

@dataclass
class AP:
    start: float
    end: float
    step: float

    def __post_init__(self):
        if self.step <= 0:
            raise ValueError("step must be positive.")
        self.start = float(self.start)
        self.end = float(self.end)
        self.step = float(self.step)

    @property
    def count(self) -> int:
        return _inclusive_count(self.start, self.end, self.step)

    def value(self, i: int) -> float:
        """Get the i-th value (0-indexed)."""
        if i < 0 or i >= self.count:
            raise IndexError(f"Index {i} out of range [0, {self.count})")
        return self.start + i * self.step

    def slice(self, start_idx: int, length: int) -> "AP":
        if length <= 0:
            return AP(self.start, self.start - self.step, self.step)
        s = self.start + start_idx * self.step
        e = s + (length - 1) * self.step
        return AP(s, e, self.step)

    def is_empty(self) -> bool:
        return self.count == 0

    def __iter__(self):
        for i in range(self.count):
            yield self.start + i * self.step

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"AP(start={self.start}, end={self.end}, step={self.step}, count={self.count})"


def _inclusive_count(start: float, end: float, step: float) -> int:
    if start == end:
        return 1
    if step <= 0 or end < start:
        return 0
    eps = 1e-12 * max(1.0, abs(start), abs(end))
    return int(math.floor((end - start) / step + eps)) + 1


def _normalize_ap(start: float, end: float, step: float) -> AP:
    n = _inclusive_count(start, end, step)
    if n == 0:
        return AP(start, start - step, step)
    real_end = start + (n - 1) * step
    return AP(start, real_end, step)

@dataclass
class AngleRange:
    """
    Range of rotation angles (in degrees) to search over.
    
    Attributes:
        min_angle: Minimum rotation angle.
        max_angle: Maximum rotation angle.
        step: Angular increment between search points.
    """
    min_angle: float = 0.0
    max_angle: float = 0.0
    step: float = 1.0

    def __post_init__(self):
        self.min_angle = float(self.min_angle)
        self.max_angle = float(self.max_angle)
        self.step = float(self.step)
        if self.step <= 0:
            raise ValueError("step must be positive")

    def to_ap(self) -> AP:
        return _normalize_ap(self.min_angle, self.max_angle, self.step)


@dataclass
class ScaleRange:
    """
    Range of scale factors to search over.
    
    Attributes:
        min_scale: Minimum scale factor (should be > 0).
        max_scale: Maximum scale factor.
        step: Scale increment between search points.
    """
    min_scale: float = 1.0
    max_scale: float = 1.0
    step: float = 0.01

    def __post_init__(self):
        self.min_scale = float(self.min_scale)
        self.max_scale = float(self.max_scale)
        self.step = float(self.step)
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")

    def to_ap(self) -> AP:
        return _normalize_ap(self.min_scale, self.max_scale, self.step)


@dataclass
class ShearRange:
    """
    Range of shear angles (in degrees) to search over.
    
    Attributes:
        min_shear: Minimum shear angle.
        max_shear: Maximum shear angle.
        step: Shear angle increment between search points.
    """
    min_shear: float = 0.0
    max_shear: float = 0.0
    step: float = 1.0

    def __post_init__(self):
        self.min_shear = float(self.min_shear)
        self.max_shear = float(self.max_shear)
        self.step = float(self.step)
        if self.step <= 0:
            raise ValueError("step must be positive")

    def to_ap(self) -> AP:
        return _normalize_ap(self.min_shear, self.max_shear, self.step)


class AxisGroup(Enum):
    """
    Groups of related axes that share a predicate.
    
    Each group represents dimensions that must be iterated together
    because their predicate depends on all dimensions in the group.
    
    - ANGLE: Single dimension (angle), predicate: angle_ok(angle)
    - SCALE: Two dimensions (scale_x, scale_y), predicate: scale_ok(sx, sy)
    - SHEAR: Two dimensions (shear_x, shear_y), predicate: shear_ok(shx, shy)
    """
    ANGLE = "angle"
    SCALE = "scale"
    SHEAR = "shear"


IterationOrder = Tuple[AxisGroup, AxisGroup, AxisGroup]
DEFAULT_ORDER: IterationOrder = (AxisGroup.ANGLE, AxisGroup.SCALE, AxisGroup.SHEAR)

def _validate_order(order: IterationOrder) -> None:
    if len(order) != 3:
        raise ValueError(f"order must have exactly 3 groups, got {len(order)}")
    groups = set(order)
    if groups != {AxisGroup.ANGLE, AxisGroup.SCALE, AxisGroup.SHEAR}:
        missing = {AxisGroup.ANGLE, AxisGroup.SCALE, AxisGroup.SHEAR} - groups
        raise ValueError(f"order must contain all axis groups, missing: {missing}")


def _get_partition_axis(order: IterationOrder) -> str:
    outer_group = order[0]
    if outer_group == AxisGroup.ANGLE:
        return "angle"
    elif outer_group == AxisGroup.SCALE:
        return "scale_x"  # Partition first axis of scale group
    elif outer_group == AxisGroup.SHEAR:
        return "shear_x"  # Partition first axis of shear group
    else:
        raise ValueError(f"Unknown axis group: {outer_group}")


def make_angle_predicate(
    *,
    min_angle: Optional[float] = None,
    max_angle: Optional[float] = None,
) -> Callable[[float], bool]:
    """
    Create a predicate that filters rotation angles.
    
    Args:
        min_angle: Minimum allowed angle (inclusive).
        max_angle: Maximum allowed angle (inclusive).
        
    Returns:
        Predicate function: angle -> bool.
    """
    def pred(angle: float) -> bool:
        if min_angle is not None and angle < min_angle:
            return False
        if max_angle is not None and angle > max_angle:
            return False
        return True
    return pred


def make_scale_predicate(
    *,
    max_anisotropy: Optional[float] = 1.1,
    area_bounds: Optional[Tuple[float, float]] = None,
) -> Callable[[float, float], bool]:
    """
    Create a predicate that filters scale combinations.
    
    Args:
        max_anisotropy: Maximum ratio max(sx/sy, sy/sx). Must be >= 1.0.
            Controls how different X and Y scales can be.
        area_bounds: (min_area, max_area) bounds for sx * sy.
            Controls the combined zoom factor.
            
    Returns:
        Predicate function: (scale_x, scale_y) -> bool.
        
    Example:
        >>> pred = make_scale_predicate(max_anisotropy=1.2, area_bounds=(0.8, 1.2))
        >>> pred(1.0, 1.0)  # True - isotropic, area=1.0
        True
        >>> pred(1.5, 0.5)  # False - anisotropy=3.0 exceeds limit
        False
    """
    if max_anisotropy is not None and max_anisotropy < 1.0:
        raise ValueError("max_anisotropy must be >= 1.0")

    log_aniso = None if max_anisotropy is None else math.log(max_anisotropy)
    log_area_min = log_area_max = None
    
    if area_bounds is not None:
        area_min, area_max = area_bounds
        if area_min <= 0 or area_max <= 0 or area_min > area_max:
            raise ValueError("area_bounds must satisfy 0 < min <= max")
        log_area_min = math.log(area_min)
        log_area_max = math.log(area_max)

    def pred(sx: float, sy: float):
        if isinstance(sx, torch.Tensor) or isinstance(sy, torch.Tensor):
            sx_t = sx if isinstance(sx, torch.Tensor) else torch.as_tensor(sx)
            sy_t = sy if isinstance(sy, torch.Tensor) else torch.as_tensor(sy)

            valid = (sx_t > 0) & (sy_t > 0)
            if log_aniso is not None:
                if sx_t.dtype.is_floating_point:
                    eps = torch.finfo(sx_t.dtype).tiny
                    tol = torch.finfo(sx_t.dtype).eps * 8
                else:
                    eps = 1e-12
                    tol = 0.0
                sx_log = torch.log(sx_t.clamp_min(eps))
                sy_log = torch.log(sy_t.clamp_min(eps))
                valid = valid & (torch.abs(sx_log - sy_log) <= (log_aniso + tol))

            if log_area_min is not None and log_area_max is not None:
                if sx_t.dtype.is_floating_point:
                    eps = torch.finfo(sx_t.dtype).tiny
                    tol = torch.finfo(sx_t.dtype).eps * 8
                else:
                    eps = 1e-12
                    tol = 0.0
                log_area = torch.log(sx_t.clamp_min(eps)) + torch.log(sy_t.clamp_min(eps))
                valid = (
                    valid
                    & (log_area >= (log_area_min - tol))
                    & (log_area <= (log_area_max + tol))
                )

            return valid

        sx_f = float(sx)
        sy_f = float(sy)
        if sx_f <= 0 or sy_f <= 0:
            return False
        if log_aniso is not None:
            # |log(sx) - log(sy)| = |log(sx/sy)| <= log(max_anisotropy)
            if abs(math.log(sx_f) - math.log(sy_f)) > log_aniso:
                return False
        if log_area_min is not None and log_area_max is not None:
            log_area = math.log(sx_f) + math.log(sy_f)
            if log_area < log_area_min or log_area > log_area_max:
                return False
        return True

    return pred


def make_shear_predicate(
    *,
    max_abs: Optional[float] = None,
    max_l2: Optional[float] = None,
) -> Callable[[float, float], bool]:
    """
    Create a predicate that filters shear combinations.
    
    Args:
        max_abs: Maximum absolute value for each shear component.
            |shear_x| <= max_abs and |shear_y| <= max_abs.
        max_l2: Maximum L2 norm of shear vector.
            sqrt(shear_x^2 + shear_y^2) <= max_l2.
            
    Returns:
        Predicate function: (shear_x, shear_y) -> bool.
    """
    max_l2_sq = None if max_l2 is None else max_l2 * max_l2

    def pred(shx: float, shy: float) -> bool:
        if max_abs is not None:
            if abs(shx) > max_abs or abs(shy) > max_abs:
                return False
        if max_l2_sq is not None:
            if shx * shx + shy * shy > max_l2_sq:
                return False
        return True

    return pred


@dataclass
class SearchParams:
    rotation: AngleRange = field(default_factory=AngleRange)
    scale_x: ScaleRange = field(default_factory=ScaleRange)
    scale_y: ScaleRange = field(default_factory=ScaleRange)
    shear_x: ShearRange = field(default_factory=ShearRange)
    shear_y: ShearRange = field(default_factory=ShearRange)
    
    angle_ok: Callable[[float], bool] = field(
        default_factory=lambda: make_angle_predicate()
    )
    scale_ok: Callable[[float, float], bool] = field(
        default_factory=lambda: make_scale_predicate()
    )
    shear_ok: Callable[[float, float], bool] = field(
        default_factory=lambda: make_shear_predicate()
    )
    
    order: IterationOrder = field(default=DEFAULT_ORDER)

    def __post_init__(self):
        _validate_order(self.order)

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations (before predicate filtering)."""
        return (
            self.rotation.to_ap().count
            * self.scale_x.to_ap().count
            * self.scale_y.to_ap().count
            * self.shear_x.to_ap().count
            * self.shear_y.to_ap().count
        )
    
    @property
    def partition_axis(self) -> str:
        return _get_partition_axis(self.order)


@dataclass(frozen=True)
class GridAP:
    angles: AP
    scales_x: AP
    scales_y: AP
    shear_x: AP
    shear_y: AP
    
    angle_ok: Callable[[float], bool] = field(
        default_factory=lambda: make_angle_predicate()
    )
    scale_ok: Callable[[float, float], bool] = field(
        default_factory=lambda: make_scale_predicate()
    )
    shear_ok: Callable[[float, float], bool] = field(
        default_factory=lambda: make_shear_predicate()
    )
    
    order: IterationOrder = field(default=DEFAULT_ORDER)

    @property
    def total_count(self) -> int:
        """Total combinations in this grid partition (before predicate filtering)."""
        return (
            self.angles.count
            * self.scales_x.count
            * self.scales_y.count
            * self.shear_x.count
            * self.shear_y.count
        )

    def is_empty(self) -> bool:
        return self.total_count == 0
    
    def get_ap(self, axis: str) -> AP:
        mapping = {
            "angle": self.angles,
            "scale_x": self.scales_x,
            "scale_y": self.scales_y,
            "shear_x": self.shear_x,
            "shear_y": self.shear_y,
        }
        if axis not in mapping:
            raise ValueError(f"Unknown axis: {axis}")
        return mapping[axis]
    
    def get_group_axes(self, group: AxisGroup) -> Tuple[str, ...]:
        if group == AxisGroup.ANGLE:
            return ("angle",)
        elif group == AxisGroup.SCALE:
            return ("scale_x", "scale_y")
        elif group == AxisGroup.SHEAR:
            return ("shear_x", "shear_y")
        else:
            raise ValueError(f"Unknown group: {group}")
    
    def get_group_predicate(self, group: AxisGroup) -> Callable:
        if group == AxisGroup.ANGLE:
            return self.angle_ok
        elif group == AxisGroup.SCALE:
            return self.scale_ok
        elif group == AxisGroup.SHEAR:
            return self.shear_ok
        else:
            raise ValueError(f"Unknown group: {group}")



def _compute_block(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    """
    - Each process gets either floor(total/world_size) or ceil(total/world_size) items
    - The first (total % world_size) processes get the extra item
    - All items are covered exactly once
    
    Example:
        >>> # Distribute 10 items across 3 processes
        >>> _compute_block(10, 3, 0)  # (0, 4) - indices 0,1,2,3
        (0, 4)
        >>> _compute_block(10, 3, 1)  # (4, 3) - indices 4,5,6
        (4, 3)
        >>> _compute_block(10, 3, 2)  # (7, 3) - indices 7,8,9
        (7, 3)
    """
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if total <= 0:
        return (0, 0)

    base, remainder = divmod(total, world_size)
    
    if rank < remainder:
        start = rank * (base + 1)
        count = base + 1
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        count = base
    
    return (start, count)


def grid_for_rank(
    params: SearchParams,
    *,
    world_size: int = 1,
    rank: int = 0,
) -> GridAP:
    angles_ap = params.rotation.to_ap()
    scales_x_ap = params.scale_x.to_ap()
    scales_y_ap = params.scale_y.to_ap()
    shear_x_ap = params.shear_x.to_ap()
    shear_y_ap = params.shear_y.to_ap()
    
    # Determine which axis to partition based on order
    partition_axis = _get_partition_axis(params.order)
    
    ap_mapping = {
        "angle": angles_ap,
        "scale_x": scales_x_ap,
        "shear_x": shear_x_ap,
    }
    partition_ap = ap_mapping[partition_axis]
    
    # Compute partition
    start_idx, block_count = _compute_block(partition_ap.count, world_size, rank)
    partitioned_ap = partition_ap.slice(start_idx, block_count)
    
    # Build result with partitioned axis
    if partition_axis == "angle":
        return GridAP(
            angles=partitioned_ap,
            scales_x=scales_x_ap,
            scales_y=scales_y_ap,
            shear_x=shear_x_ap,
            shear_y=shear_y_ap,
            angle_ok=params.angle_ok,
            scale_ok=params.scale_ok,
            shear_ok=params.shear_ok,
            order=params.order,
        )
    elif partition_axis == "scale_x":
        return GridAP(
            angles=angles_ap,
            scales_x=partitioned_ap,
            scales_y=scales_y_ap,
            shear_x=shear_x_ap,
            shear_y=shear_y_ap,
            angle_ok=params.angle_ok,
            scale_ok=params.scale_ok,
            shear_ok=params.shear_ok,
            order=params.order,
        )
    elif partition_axis == "shear_x":
        return GridAP(
            angles=angles_ap,
            scales_x=scales_x_ap,
            scales_y=scales_y_ap,
            shear_x=partitioned_ap,
            shear_y=shear_y_ap,
            angle_ok=params.angle_ok,
            scale_ok=params.scale_ok,
            shear_ok=params.shear_ok,
            order=params.order,
        )
    else:
        raise ValueError(f"Unexpected partition axis: {partition_axis}")



@dataclass
class ExhaustiveSearchConfig:
    batch_size: int = 64
    auto_batch_size: bool = False
    max_auto_batch_size: Optional[int] = None

    translation_method: str = "pc"
    translation_params: Dict[str, Any] = field(default_factory=dict)

    gradient_method: str = "cd"
    gradient_params: Dict[str, Any] | None = None

    progress_enabled: bool = True
    progress_transient: bool = False
    device: Union[str, torch.device] = "cpu"

    use_reference_cache: bool = False
    use_moving_cache: bool = False

    translation: Optional[Any] = None  # TranslationConfig
    gradient: Optional[Any] = None  # GradientPipelineConfig | GradientMethodConfig

    def __post_init__(self):
        if self.translation is not None:
            self._resolve_translation_config()

        if self.gradient is not None:
            self._resolve_gradient_config()
        elif self.gradient_params is None:
            self.gradient_params = {
                "eta": {"type": "mean", "scale": 0.2, "norm": "l2"},
                "normalize": {"type": "normalize", "norm": "l2"},
            }

    def _resolve_translation_config(self) -> None:
        method, params = self.translation.to_method_and_params()
        self.translation_method = method
        self.translation_params = params

    def _resolve_gradient_config(self) -> None:
        method, params = self.gradient.to_method_and_params()
        self.gradient_method = method
        self.gradient_params = params


@dataclass
class PredicateConfig:
    # Angle constraints (None = no constraint)
    angle_min: Optional[float] = None
    angle_max: Optional[float] = None

    # Scale constraints
    max_anisotropy: float = 1.1
    area_bounds: Optional[Tuple[float, float]] = None

    # Shear constraints
    max_shear_abs: Optional[float] = None
    max_shear_l2: Optional[float] = None

    def build_angle_predicate(self) -> Callable[[float], bool]:
        """Build angle predicate from config."""
        return make_angle_predicate(
            min_angle=self.angle_min,
            max_angle=self.angle_max,
        )

    def build_scale_predicate(self) -> Callable[[float, float], bool]:
        """Build scale predicate from config."""
        return make_scale_predicate(
            max_anisotropy=self.max_anisotropy,
            area_bounds=self.area_bounds,
        )

    def build_shear_predicate(self) -> Callable[[float, float], bool]:
        """Build shear predicate from config."""
        return make_shear_predicate(
            max_abs=self.max_shear_abs,
            max_l2=self.max_shear_l2,
        )

    def build_all(
        self,
    ) -> Tuple[
        Callable[[float], bool],
        Callable[[float, float], bool],
        Callable[[float, float], bool],
    ]:
        """Build all three predicates."""
        return (
            self.build_angle_predicate(),
            self.build_scale_predicate(),
            self.build_shear_predicate(),
        )


@dataclass
class SearchGridConfig:
    rotation: AngleRange = field(default_factory=AngleRange)
    scale_x: ScaleRange = field(default_factory=ScaleRange)
    scale_y: ScaleRange = field(default_factory=ScaleRange)
    shear_x: ShearRange = field(default_factory=ShearRange)
    shear_y: ShearRange = field(default_factory=ShearRange)

    predicates: PredicateConfig = field(default_factory=PredicateConfig)

    iteration_order: str = "angle_scale_shear"

    def to_search_params(self) -> SearchParams:
        angle_ok, scale_ok, shear_ok = self.predicates.build_all()
        order = self._parse_iteration_order(self.iteration_order)

        return SearchParams(
            rotation=self.rotation,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            shear_x=self.shear_x,
            shear_y=self.shear_y,
            angle_ok=angle_ok,
            scale_ok=scale_ok,
            shear_ok=shear_ok,
            order=order,
        )

    def _parse_iteration_order(self, order_str: str) -> IterationOrder:
        """Parse string iteration order to AxisGroup tuple."""
        mapping = {
            "angle": AxisGroup.ANGLE,
            "scale": AxisGroup.SCALE,
            "shear": AxisGroup.SHEAR,
        }
        parts = order_str.lower().split("_")
        if len(parts) != 3:
            return DEFAULT_ORDER
        try:
            return tuple(mapping[p] for p in parts)  # type: ignore[return-value]
        except KeyError:
            return DEFAULT_ORDER
