from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from batchmatch.base.tensordicts import ImageDetail, WarpParams

Tensor = torch.Tensor

__all__ = ["OptimizeResult"]


@dataclass(frozen=True)
class OptimizeResult:
    reference: ImageDetail
    moving: ImageDetail
    registered: ImageDetail
    warp: WarpParams
    loss_history: Optional[Tensor]
    best_loss: float
    steps: int
