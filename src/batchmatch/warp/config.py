"""Configuration dataclasses for warp pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from batchmatch.warp.base import WarpPipelineSpec

__all__ = [
    "WarpPipelineConfig",
]


@dataclass(frozen=True)
class WarpPipelineConfig:
    outputs: Sequence[str] = ("image", "window", "mask")
    prepare: Optional[object] = "prepare"
    stages: Mapping[str, object] = field(default_factory=dict)

    def to_spec(self) -> "WarpPipelineSpec":
        from batchmatch.warp.base import _build_outputs_spec, coerce_warp_pipeline_spec

        spec = _build_outputs_spec(self.outputs, self.prepare, self.stages)
        return coerce_warp_pipeline_spec(spec)
