from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

import torch
from torch import Tensor

from batchmatch.base.tensordicts import WarpParams
from batchmatch.search.config import AxisGroup, GridAP

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

__all__ = [
    "WarpParamIterator",
    "IteratorStats",
    "ProgressTracker",
    "make_search_progress",
    "iter_warp_params",
]

@dataclass
class IteratorStats:
    total: int = 0
    checked: int = 0
    skipped: int = 0
    
    best_score: float | None = None
    best_params: str | None = None

    @property
    def valid(self) -> int:
        return self.checked - self.skipped

    @property
    def skip_ratio(self) -> float:
        return self.skipped / self.total if self.total > 0 else 0.0

    @property
    def progress_ratio(self) -> float:
        return self.checked / self.total if self.total > 0 else 0.0
    
    @property 
    def is_complete(self) -> bool:
        return self.checked >= self.total and self.total > 0

    def reset(self, total: int = 0) -> None:
        self.total = total
        self.checked = 0
        self.skipped = 0
        self.best_score = None
        self.best_params = None


class ProgressTracker:
    
    def __init__(
        self,
        progress: "Progress",
        *,
        description: str = "Warp search",
        task_id: "TaskID | None" = None,
    ):
        self._progress = progress
        self._description = description
        self._task_id = task_id
        self._initialized = False
        self._last_checked = 0
    
    def _ensure_task(self, stats: IteratorStats) -> None:
        if self._initialized:
            return
        
        initial_fields = {
            "skip_ratio": 0.0,
            "checked": 0,
            "skipped": 0,
            "valid": 0,
            "total_expected": stats.total,
            "best": "",
        }
        
        if self._task_id is not None:
            self._progress.update(
                self._task_id,
                total=stats.total,
                description=self._description,
                **initial_fields,
            )
        else:
            self._task_id = self._progress.add_task(
                self._description,
                total=stats.total,
                **initial_fields,
            )
        self._initialized = True
    
    def update(self, stats: IteratorStats) -> None:
        self._ensure_task(stats)
        
        advance = stats.checked - self._last_checked
        self._last_checked = stats.checked
        
        self._progress.update(
            self._task_id,
            advance=advance,
            checked=stats.checked,
            skipped=stats.skipped,
            valid=stats.valid,
            skip_ratio=stats.skip_ratio,
            total_expected=stats.total,
            best=stats.best_params or "",
        )
    
    def finish(self) -> None:
        if self._task_id is not None and self._initialized:
            self._progress.update(
                self._task_id, 
                completed=self._progress.tasks[self._task_id].total
            )
    
    @property
    def task_id(self) -> "TaskID | None":
        return self._task_id


def _make_progress_columns():
    from rich.progress import ProgressColumn
    from rich.text import Text
    
    class CheckedColumn(ProgressColumn):
        """Display checked/total combinations."""
        
        def render(self, task):
            checked = task.fields.get("checked", 0)
            total = task.fields.get("total_expected", task.total or 0)
            return Text(f"checked {checked:,}/{total:,}", style="progress.data.speed")
    
    class ValidColumn(ProgressColumn):
        """Display valid combination count."""
        
        def render(self, task):
            valid = task.fields.get("valid", 0)
            return Text(f"valid {valid:,}", style="green")
    
    class SkipRatioColumn(ProgressColumn):
        """Display the fraction of combinations skipped."""
        
        def render(self, task):
            ratio = task.fields.get("skip_ratio", 0.0)
            pct = ratio * 100.0
            return Text(f"skipped {pct:5.1f}%", style="progress.data.speed")
    
    class BestParamsColumn(ProgressColumn):
        """Display best parameters found."""
        
        def render(self, task):
            summary = task.fields.get("best", "")
            if not summary:
                return Text("")
            return Text(summary, style="bold cyan", overflow="ellipsis")
    
    return CheckedColumn, ValidColumn, SkipRatioColumn, BestParamsColumn


def make_search_progress(*, transient: bool = False) -> "Progress":
    """ 
    Example:
        >>> from batchmatch.search import WarpParamIterator, ProgressTracker, make_search_progress
        >>> 
        >>> iterator = WarpParamIterator(grid, batch_size=64)
        >>> with make_search_progress() as progress:
        ...     tracker = ProgressTracker(progress)
        ...     for batch in iterator:
        ...         tracker.update(iterator.stats)
        ...     tracker.finish()
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    
    CheckedCol, ValidCol, SkipCol, BestCol = _make_progress_columns()
    
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        CheckedCol(),
        ValidCol(),
        SkipCol(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        BestCol(),
        transient=transient,
        expand=False,
    )



@dataclass
class _AxisSpec:
    name: str
    range_obj: range
    get_value: Callable[[int], float]


@dataclass
class _GroupSpec:
    group: AxisGroup
    axes: list[_AxisSpec]
    predicate: Callable[..., bool]
    inner_count: int


def _build_group_spec(
    grid: GridAP,
    group: AxisGroup,
    axis_ranges: dict[str, range],
    inner_count: int,
) -> _GroupSpec:
    axes = []
    for axis_name in grid.get_group_axes(group):
        ap = grid.get_ap(axis_name)
        axes.append(_AxisSpec(
            name=axis_name,
            range_obj=axis_ranges[axis_name],
            get_value=ap.value,
        ))
    
    return _GroupSpec(
        group=group,
        axes=axes,
        predicate=grid.get_group_predicate(group),
        inner_count=inner_count,
    )


def _iter_group_values(spec: _GroupSpec):
    if len(spec.axes) == 1:
        axis = spec.axes[0]
        for i in axis.range_obj:
            yield (axis.get_value(i),)
    elif len(spec.axes) == 2:
        axis0, axis1 = spec.axes
        for i in axis0.range_obj:
            v0 = axis0.get_value(i)
            for j in axis1.range_obj:
                v1 = axis1.get_value(j)
                yield (v0, v1)
    else:
        raise ValueError(f"Unexpected number of axes: {len(spec.axes)}")


def _group_combination_count(spec: _GroupSpec) -> int:
    count = 1
    for axis in spec.axes:
        count *= len(axis.range_obj)
    return count


class WarpParamIterator:
    def __init__(
        self,
        grid: GridAP,
        *,
        batch_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        angles_block: tuple[int, int] | None = None,
        scales_x_block: tuple[int, int] | None = None,
        scales_y_block: tuple[int, int] | None = None,
        shear_x_block: tuple[int, int] | None = None,
        shear_y_block: tuple[int, int] | None = None,
        group_tables: dict["AxisGroup", "_GroupTable"] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        self._grid = grid
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._dtype = dtype

        self._tensor_buffers: dict[str, Tensor] | None = None
        self._init_tensor_buffers()
        
        self._axis_ranges = self._compute_axis_ranges(
            angles_block=angles_block,
            scales_x_block=scales_x_block,
            scales_y_block=scales_y_block,
            shear_x_block=shear_x_block,
            shear_y_block=shear_y_block,
        )
        
        self._total = math.prod(len(r) for r in self._axis_ranges.values())
        
        self._stats = IteratorStats(total=self._total)

        if group_tables is not None:
            self._group_tables = group_tables
        else:
            self._group_tables = self._build_group_tables()
        self._valid_total = self._compute_valid_total()
    
    def _compute_axis_ranges(
        self,
        *,
        angles_block: tuple[int, int] | None,
        scales_x_block: tuple[int, int] | None,
        scales_y_block: tuple[int, int] | None,
        shear_x_block: tuple[int, int] | None,
        shear_y_block: tuple[int, int] | None,
    ) -> dict[str, range]:
        def resolve_range(block: tuple[int, int] | None, total: int) -> range:
            if block is None:
                return range(total)
            start, count = block
            if start < 0 or start >= total or count <= 0:
                return range(0)
            end = min(start + count, total)
            return range(start, end)
        
        return {
            "angle": resolve_range(angles_block, self._grid.angles.count),
            "scale_x": resolve_range(scales_x_block, self._grid.scales_x.count),
            "scale_y": resolve_range(scales_y_block, self._grid.scales_y.count),
            "shear_x": resolve_range(shear_x_block, self._grid.shear_x.count),
            "shear_y": resolve_range(shear_y_block, self._grid.shear_y.count),
        }

    def _init_tensor_buffers(self) -> None:
        self._tensor_buffers = {
            WarpParams.Keys.ANGLE: torch.empty(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
            WarpParams.Keys.SCALE_X: torch.empty(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
            WarpParams.Keys.SCALE_Y: torch.empty(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
            WarpParams.Keys.SHEAR_X: torch.empty(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
            WarpParams.Keys.SHEAR_Y: torch.empty(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
            WarpParams.Keys.TX: torch.zeros(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
            WarpParams.Keys.TY: torch.zeros(
                self._batch_size, device=self._device, dtype=self._dtype
            ),
        }

    @property
    def group_tables(self) -> dict["AxisGroup", "_GroupTable"]:
        return self._group_tables

    @dataclass(frozen=True, slots=True)
    class _GroupTable:
        group: AxisGroup
        unfiltered_size: int
        valid_unfiltered_idx: Tensor
        values: dict[str, Tensor]

        @property
        def valid_size(self) -> int:
            if not self.values:
                return 0
            first = next(iter(self.values.values()))
            return int(first.shape[0])

    def _ap_values_for_range(self, ap, idx_range: range) -> Tensor:
        count = len(idx_range)
        if count == 0:
            return torch.empty((0,), device=self._device, dtype=self._dtype)

        start_i = idx_range.start
        start_val = float(ap.start + start_i * ap.step)
        step = float(ap.step)
        return start_val + step * torch.arange(
            count, device=self._device, dtype=self._dtype
        )

    @staticmethod
    def _coerce_predicate_mask(result: object, shape: torch.Size, device: torch.device) -> Tensor:
        if isinstance(result, torch.Tensor):
            mask = result
            if mask.shape != shape:
                raise ValueError(f"Predicate returned shape {tuple(mask.shape)}, expected {tuple(shape)}.")
            if mask.dtype != torch.bool:
                mask = mask != 0
            return mask
        if isinstance(result, bool):
            return torch.full(shape, result, device=device, dtype=torch.bool)
        raise TypeError(
            "Predicate must return a torch.Tensor or bool when called with tensors; "
            f"got {type(result).__name__}."
        )

    @torch.compiler.disable
    def _predicate_mask_1d(self, predicate: Callable[..., bool], values: Tensor) -> Tensor:
        res = predicate(values)
        return self._coerce_predicate_mask(res, values.shape, values.device)

    @torch.compiler.disable
    def _predicate_mask_2d(
        self, predicate: Callable[..., bool], x_grid: Tensor, y_grid: Tensor
    ) -> Tensor:
        res = predicate(x_grid, y_grid)
        expected = torch.broadcast_shapes(x_grid.shape, y_grid.shape)
        return self._coerce_predicate_mask(res, expected, x_grid.device)

    def _build_2d_group_table(
        self,
        group: AxisGroup,
        ap_x,
        ap_y,
        range_x: range,
        range_y: range,
        predicate,
        key_x: str,
        key_y: str,
    ) -> "_GroupTable":
        x_all = self._ap_values_for_range(ap_x, range_x)
        y_all = self._ap_values_for_range(ap_y, range_y)
        x_grid = x_all[:, None]
        y_grid = y_all[None, :]
        mask = self._predicate_mask_2d(predicate, x_grid, y_grid)
        rc = torch.nonzero(mask, as_tuple=False)
        if rc.numel():
            r, c = rc[:, 0], rc[:, 1]
            x_valid = x_all.index_select(0, r)
            y_valid = y_all.index_select(0, c)
            idx = r * len(range_y) + c
        else:
            x_valid = x_all[:0]
            y_valid = y_all[:0]
            idx = torch.empty(0, dtype=torch.long, device=self._device)
        return self._GroupTable(
            group=group,
            unfiltered_size=len(range_x) * len(range_y),
            valid_unfiltered_idx=idx,
            values={key_x: x_valid, key_y: y_valid},
        )

    @torch.compiler.disable
    def _build_group_tables(self) -> dict[AxisGroup, "WarpParamIterator._GroupTable"]:
        grid = self._grid

        # (1D: angle)
        angle_range = self._axis_ranges["angle"]
        angles_all = self._ap_values_for_range(grid.angles, angle_range)
        angle_mask = self._predicate_mask_1d(grid.angle_ok, angles_all)
        angle_pos = torch.nonzero(angle_mask, as_tuple=False).flatten()
        angle_valid = angles_all.index_select(0, angle_pos) if angle_pos.numel() else angles_all[:0]
        angle_table = self._GroupTable(
            group=AxisGroup.ANGLE,
            unfiltered_size=len(angle_range),
            valid_unfiltered_idx=angle_pos,
            values={WarpParams.Keys.ANGLE: angle_valid},
        )

        #(2D: scale_x outer, scale_y inner)
        scale_table = self._build_2d_group_table(
            AxisGroup.SCALE,
            grid.scales_x, grid.scales_y,
            self._axis_ranges["scale_x"], self._axis_ranges["scale_y"],
            grid.scale_ok,
            WarpParams.Keys.SCALE_X, WarpParams.Keys.SCALE_Y,
        )

        # (2D: shear_x outer, shear_y inner)
        shear_table = self._build_2d_group_table(
            AxisGroup.SHEAR,
            grid.shear_x, grid.shear_y,
            self._axis_ranges["shear_x"], self._axis_ranges["shear_y"],
            grid.shear_ok,
            WarpParams.Keys.SHEAR_X, WarpParams.Keys.SHEAR_Y,
        )

        return {
            AxisGroup.ANGLE: angle_table,
            AxisGroup.SCALE: scale_table,
            AxisGroup.SHEAR: shear_table,
        }

    def _compute_valid_total(self) -> int:
        if self._total == 0:
            return 0
        v_angle = self._group_tables[AxisGroup.ANGLE].valid_size
        v_scale = self._group_tables[AxisGroup.SCALE].valid_size
        v_shear = self._group_tables[AxisGroup.SHEAR].valid_size
        return int(v_angle * v_scale * v_shear)

    @torch.compiler.disable
    def _checked_after_valid_linear(
        self,
        valid_linear: int,
        tables_in_order: list["_GroupTable"],
        valid_sizes: tuple[int, int, int],
        unfiltered_sizes: tuple[int, int, int],
    ) -> int:
        v0, v1, v2 = valid_sizes
        _, l1, l2 = unfiltered_sizes

        i0 = valid_linear // (v1 * v2)
        rem = valid_linear % (v1 * v2)
        i1 = rem // v2
        i2 = rem % v2

        idx0 = tables_in_order[0].valid_unfiltered_idx[i0].item()
        idx1 = tables_in_order[1].valid_unfiltered_idx[i1].item()
        idx2 = tables_in_order[2].valid_unfiltered_idx[i2].item()

        return ((idx0 * l1) + idx1) * l2 + idx2 + 1

    def _fill_param_buffer(
        self, dst: Tensor, src: Tensor, index: Tensor
    ) -> None:
        try:
            torch.index_select(src, 0, index, out=dst)
        except Exception:
            dst.copy_(src.index_select(0, index))

    def _iterate_device(self) -> Iterator[WarpParams]:
        stats = self._stats
        total = self._total

        if total == 0:
            return

        order = self._grid.order
        tables_in_order = [self._group_tables[g] for g in order]
        v_sizes = tuple(t.valid_size for t in tables_in_order)
        if any(v == 0 for v in v_sizes):
            stats.checked = total
            stats.skipped = total
            return

        v0, v1, v2 = v_sizes
        valid_total = int(v0 * v1 * v2)
        l_sizes = tuple(t.unfiltered_size for t in tables_in_order)

        buf = self._tensor_buffers
        if buf is None:
            raise RuntimeError("Tensor buffers were not initialized.")

        offset = 0
        device = self._device
        while offset < valid_total:
            B = min(self._batch_size, valid_total - offset)

            lin = offset + torch.arange(B, device=device, dtype=torch.long)
            i0 = torch.div(lin, v1 * v2, rounding_mode="floor")
            rem = lin - i0 * (v1 * v2)
            i1 = torch.div(rem, v2, rounding_mode="floor")
            i2 = rem - i1 * v2

            indices = (i0, i1, i2)
            for table, idx in zip(tables_in_order, indices):
                for key, src in table.values.items():
                    dst = buf[key][:B]
                    self._fill_param_buffer(dst, src, idx)

            buf[WarpParams.Keys.TX][:B].zero_()
            buf[WarpParams.Keys.TY][:B].zero_()

            data = {
                WarpParams.Keys.ANGLE: buf[WarpParams.Keys.ANGLE][:B],
                WarpParams.Keys.SCALE_X: buf[WarpParams.Keys.SCALE_X][:B],
                WarpParams.Keys.SCALE_Y: buf[WarpParams.Keys.SCALE_Y][:B],
                WarpParams.Keys.SHEAR_X: buf[WarpParams.Keys.SHEAR_X][:B],
                WarpParams.Keys.SHEAR_Y: buf[WarpParams.Keys.SHEAR_Y][:B],
                WarpParams.Keys.TX: buf[WarpParams.Keys.TX][:B],
                WarpParams.Keys.TY: buf[WarpParams.Keys.TY][:B],
            }
            params = WarpParams(data, batch_size=[B])

            produced_end = offset + B
            if produced_end >= valid_total:
                stats.checked = total
                stats.skipped = total - valid_total
            else:
                checked = self._checked_after_valid_linear(
                    produced_end - 1, tables_in_order, v_sizes, l_sizes
                )
                stats.checked = checked
                stats.skipped = checked - produced_end

            yield params
            offset = produced_end

    @property
    def stats(self) -> IteratorStats:
        return self._stats
    
    @property
    def total(self) -> int:
        return self._total
    
    def __iter__(self) -> Iterator[WarpParams]:
        self._stats.reset(self._total)
        yield from self._iterate_device()

def iter_warp_params(
    grid: GridAP,
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    angles_block: tuple[int, int] | None = None,
    scales_x_block: tuple[int, int] | None = None,
    scales_y_block: tuple[int, int] | None = None,
    shear_x_block: tuple[int, int] | None = None,
    shear_y_block: tuple[int, int] | None = None,
    stats_out: IteratorStats | None = None,
) -> Iterator[WarpParams]:
    iterator = WarpParamIterator(
        grid,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        angles_block=angles_block,
        scales_x_block=scales_x_block,
        scales_y_block=scales_y_block,
        shear_x_block=shear_x_block,
        shear_y_block=shear_y_block,
    )
    
    for batch in iterator:
        if stats_out is not None:
            stats_out.total = iterator.stats.total
            stats_out.checked = iterator.stats.checked
            stats_out.skipped = iterator.stats.skipped
        yield batch
    
    if stats_out is not None:
        stats_out.total = iterator.stats.total
        stats_out.checked = iterator.stats.checked
        stats_out.skipped = iterator.stats.skipped
        stats_out.best_score = iterator.stats.best_score
        stats_out.best_params = iterator.stats.best_params
