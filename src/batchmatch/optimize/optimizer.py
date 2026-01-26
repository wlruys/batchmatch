from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from batchmatch.base.tensordicts import ImageDetail, WarpParams
from batchmatch.metric import build_image_metric
from batchmatch.metric.metrics import NormalizedGradientFieldsMetric

from .config import OptimizeConfig
from .pipeline import build_reference_pipeline, build_moving_pipeline
from .result import OptimizeResult

Tensor = torch.Tensor

__all__ = [
    "AffineWarpOptimize",
    "OptimizationStats",
]


@dataclass
class OptimizationStats:
    step: int
    loss: float


class _ProgressTracker:
    def __init__(self, progress: Any, task_id: int) -> None:
        self._progress = progress
        self._task_id = task_id

    def update(self, step: int, loss: float) -> None:
        self._progress.update(self._task_id, advance=1, loss=loss, step=step)

    def finish(self) -> None:
        self._progress.update(self._task_id, completed=self._progress.tasks[self._task_id].total)


def _setup_progress(
    *,
    progress: Optional[bool],
    total: int,
    enabled: bool,
    transient: bool,
) -> Tuple[Any, Optional[_ProgressTracker]]:
    from contextlib import nullcontext

    try:
        from rich.progress import Progress as RichProgress
        from rich.progress import BarColumn, SpinnerColumn, TextColumn, TimeRemainingColumn
    except ImportError:
        RichProgress = None

    if not enabled or progress is False:
        return nullcontext(), None

    if RichProgress is not None and isinstance(progress, RichProgress):
        task_id = progress.add_task("Optimize", total=total, loss=float("nan"), step=0)
        return nullcontext(), _ProgressTracker(progress, task_id)

    if progress is True or progress is None:
        if RichProgress is None:
            return nullcontext(), None
        progress_obj = RichProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("step={task.fields[step]:>4}"),
            TextColumn("loss={task.fields[loss]:>9.4f}"),
            TimeRemainingColumn(),
            transient=transient,
        )
        task_id = progress_obj.add_task("Optimize", total=total, loss=float("nan"), step=0)
        return progress_obj, _ProgressTracker(progress_obj, task_id)

    return nullcontext(), None


class _DifferentiableWarp(nn.Module):
    """Learnable affine parameters wired to batchmatch warp pipeline."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

        self._angle = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self._log_scale_x = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self._log_scale_y = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self._shear_x = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self._shear_y = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self._tx = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self._ty = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

    @property
    def angle(self) -> Tensor:
        return self._angle

    @property
    def scale_x(self) -> Tensor:
        return torch.exp(self._log_scale_x)

    @property
    def scale_y(self) -> Tensor:
        return torch.exp(self._log_scale_y)

    @property
    def shear_x(self) -> Tensor:
        return self._shear_x

    @property
    def shear_y(self) -> Tensor:
        return self._shear_y

    @property
    def tx(self) -> Tensor:
        return self._tx

    @property
    def ty(self) -> Tensor:
        return self._ty

    def as_warp_params(self) -> WarpParams:
        return WarpParams.from_components(
            angle=self.angle.detach(),
            scale_x=self.scale_x.detach(),
            scale_y=self.scale_y.detach(),
            shear_x=self.shear_x.detach(),
            shear_y=self.shear_y.detach(),
            tx=self.tx.detach(),
            ty=self.ty.detach(),
            device=self.device,
            dtype=self.dtype,
        )

    def load_init(self, init: WarpParams) -> None:
        angle = float(init.angle[0].item())
        scale_x = float(init.scale_x[0].item())
        scale_y = float(init.scale_y[0].item())
        shear_x = float(init.shear_x[0].item())
        shear_y = float(init.shear_y[0].item())
        tx = float(init.tx[0].item())
        ty = float(init.ty[0].item())

        with torch.no_grad():
            self._angle.fill_(angle)
            self._log_scale_x.fill_(float(torch.log(torch.tensor(scale_x))))
            self._log_scale_y.fill_(float(torch.log(torch.tensor(scale_y))))
            self._shear_x.fill_(shear_x)
            self._shear_y.fill_(shear_y)
            self._tx.fill_(tx)
            self._ty.fill_(ty)


def _metric_requires_gradients(metric: nn.Module) -> bool:
    if isinstance(metric, NormalizedGradientFieldsMetric):
        return True
    return bool(getattr(metric, "requires_gradients", False))


def _metric_requires_complex_gradients(metric: nn.Module) -> bool:
    return bool(getattr(metric, "requires_complex_gradients", False))


def _build_optimizer(
    config: OptimizeConfig,
    warp_module: _DifferentiableWarp,
) -> torch.optim.Optimizer:
    params = dict(config.optimizer.params or {})
    base_lr = float(params.pop("lr", 1e-1))

    if config.optimizer.param_groups:
        groups = _resolve_param_groups(config.optimizer.param_groups, warp_module, base_lr)
        return _make_optimizer(config.optimizer.type, groups, params)

    groups = [
        {"params": [warp_module._angle], "lr": base_lr * 0.5},
        {"params": [warp_module._log_scale_x, warp_module._log_scale_y], "lr": base_lr * 0.1},
        {"params": [warp_module._tx, warp_module._ty], "lr": base_lr * 2.0},
        {"params": [warp_module._shear_x, warp_module._shear_y], "lr": base_lr * 0.1},
    ]
    return _make_optimizer(config.optimizer.type, groups, params)


def _resolve_param_groups(
    raw_groups: Iterable[dict[str, Any]],
    warp_module: _DifferentiableWarp,
    base_lr: float,
) -> list[dict[str, Any]]:
    mapping = {
        "angle": warp_module._angle,
        "log_scale_x": warp_module._log_scale_x,
        "log_scale_y": warp_module._log_scale_y,
        "scale_x": warp_module._log_scale_x,
        "scale_y": warp_module._log_scale_y,
        "shear_x": warp_module._shear_x,
        "shear_y": warp_module._shear_y,
        "tx": warp_module._tx,
        "ty": warp_module._ty,
    }
    groups: list[dict[str, Any]] = []
    for group in raw_groups:
        g = dict(group)
        params = g.get("params")
        if isinstance(params, list) and params and isinstance(params[0], str):
            resolved = []
            for name in params:
                if name not in mapping:
                    raise ValueError(f"Unknown parameter name in param_groups: {name}")
                resolved.append(mapping[name])
            g["params"] = resolved
        if "lr" not in g:
            g["lr"] = base_lr
        groups.append(g)
    return groups


def _make_optimizer(
    name: str,
    param_groups: list[dict[str, Any]],
    kwargs: Dict[str, Any],
) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(param_groups, **kwargs)
    if key == "adamw":
        return torch.optim.AdamW(param_groups, **kwargs)
    if key == "sgd":
        return torch.optim.SGD(param_groups, **kwargs)
    raise ValueError(f"Unknown optimizer type '{name}'")


def _make_scheduler(
    config: OptimizeConfig,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if not config.scheduler.type:
        return None
    name = config.scheduler.type.lower()
    params = dict(config.scheduler.params or {})
    if name in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    raise ValueError(f"Unknown scheduler type '{config.scheduler.type}'")


class AffineWarpOptimize(nn.Module):
    """
    Gradient-based optimizer for affine warp parameters.
    """

    def __init__(self, config: OptimizeConfig) -> None:
        super().__init__()
        self.config = config
        self.device = config.resolve_device()
        self.dtype = config.dtype

        self.metric = build_image_metric(config.metric.spec)
        if not getattr(self.metric, "differentiable", True):
            raise ValueError(
                f"Metric '{type(self.metric).__name__}' is not differentiable; "
                "use a differentiable metric for optimization."
            )
        self._requires_gradients = _metric_requires_gradients(self.metric)
        self._requires_complex_gradients = _metric_requires_complex_gradients(self.metric)
        self.metric_maximize = (
            config.metric.maximize
            if config.metric.maximize is not None
            else bool(getattr(self.metric, "maximize", False))
        )

        self.reference_pipeline = build_reference_pipeline(
            config,
            requires_gradients=self._requires_gradients,
            requires_complex_gradients=self._requires_complex_gradients,
        )
        self.moving_pipeline = build_moving_pipeline(
            config,
            requires_gradients=self._requires_gradients,
            requires_complex_gradients=self._requires_complex_gradients,
        )
        self.reference_pipeline = self.reference_pipeline.to(self.device)
        self.moving_pipeline = self.moving_pipeline.to(self.device)

    def optimize(
        self,
        reference: ImageDetail,
        moving: ImageDetail,
        *,
        init: Optional[WarpParams] = None,
        callback: Optional[Callable[[OptimizationStats], None]] = None,
        return_history: bool = False,
        progress: Optional[bool] = None,
        clone_inputs: bool = True,
    ) -> OptimizeResult:
        """Run optimization and return an OptimizeResult."""

        reference = reference.to(device=self.device, dtype=self.dtype)
        moving = moving.to(device=self.device, dtype=self.dtype)
        if clone_inputs:
            reference = reference.clone()
            moving = moving.clone()

        warp_module = _DifferentiableWarp(device=self.device, dtype=self.dtype)
        if init is not None:
            warp_module.load_init(init.to(device=self.device, dtype=self.dtype))

        optimizer = _build_optimizer(self.config, warp_module)
        scheduler = _make_scheduler(self.config, optimizer)

        reference_prepared = self.reference_pipeline(reference)

        loss_history = None
        if return_history:
            loss_history = torch.zeros(
                self.config.iterations,
                device=self.device,
                dtype=self.dtype,
            )

        best_loss = float("inf")
        best_state: Optional[Dict[str, Tensor]] = None

        progress_ctx, tracker = _setup_progress(
            progress=progress,
            total=self.config.iterations,
            enabled=self.config.progress_enabled,
            transient=self.config.progress_transient,
        )

        with progress_ctx:
            for step in range(self.config.iterations):
                optimizer.zero_grad()

                moving_step = moving.clone()
                moving_step.add_warp_params(
                    angle=warp_module.angle,
                    scale_x=warp_module.scale_x,
                    scale_y=warp_module.scale_y,
                    shear_x=warp_module.shear_x,
                    shear_y=warp_module.shear_y,
                    tx=warp_module.tx,
                    ty=warp_module.ty,
                )
                moving_prepared = self.moving_pipeline(moving_step)

                metric_value = self.metric(reference_prepared, moving_prepared)
                loss = metric_value
                if loss.ndim > 0:
                    loss = loss.mean()
                if self.metric_maximize:
                    loss = -loss

                loss.backward()

                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(warp_module.parameters(), max_norm=self.config.grad_clip)

                optimizer.step()

                loss_value = float(loss.detach().item())
                if return_history and loss_history is not None:
                    loss_history[step] = loss.detach()

                if loss_value < best_loss:
                    best_loss = loss_value
                    best_state = {k: v.detach().clone() for k, v in warp_module.state_dict().items()}

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(loss.detach())
                    else:
                        scheduler.step()

                if callback is not None:
                    callback(OptimizationStats(step=step, loss=loss_value))
                if tracker is not None:
                    tracker.update(step=step, loss=loss_value)

        if tracker is not None:
            tracker.finish()

        if best_state is not None:
            warp_module.load_state_dict(best_state)

        with torch.no_grad():
            moving_final = moving.clone()
            moving_final.add_warp_params(
                angle=warp_module.angle,
                scale_x=warp_module.scale_x,
                scale_y=warp_module.scale_y,
                shear_x=warp_module.shear_x,
                shear_y=warp_module.shear_y,
                tx=warp_module.tx,
                ty=warp_module.ty,
            )
            registered = self.moving_pipeline(moving_final)

        return OptimizeResult(
            reference=reference,
            moving=moving,
            registered=registered,
            warp=warp_module.as_warp_params(),
            loss_history=loss_history,
            best_loss=best_loss,
            steps=self.config.iterations,
        )

    def forward(self, reference: ImageDetail, moving: ImageDetail, **kwargs: Any) -> OptimizeResult:
        return self.optimize(reference, moving, **kwargs)
