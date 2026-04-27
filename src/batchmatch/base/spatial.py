"""Peer pipeline hierarchy for :class:`SpatialImage`.

:class:`SpatialStage` takes and returns :class:`SpatialImage` (or a list of
them), delegating tensor work to an inner :class:`Stage` and composing
``M_next_from_current`` into :attr:`SpatialImage.space`.

It is a peer of :class:`Stage`, not a subclass: ``Stage`` is a
``TensorDictModuleBase`` with a ``forward(image: TensorDict) -> TensorDict``
contract, and mixing ``SpatialImage`` into that contract would break
validation / invalidation. ``SpatialStage`` has its own ``>>`` operator
producing a :class:`SpatialPipeline`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from batchmatch.base.tensordicts import ImageDetail
from batchmatch.io.space import SpatialImage

__all__ = ["SpatialStage", "SpatialPipeline"]


SpatialInput = SpatialImage | list[SpatialImage]


class SpatialStage(ABC):
    """A single spatial op on a :class:`SpatialImage`.

    Subclasses implement :meth:`_apply` returning
    ``(new_detail, matrix_next_from_current)``. The base ``__call__``
    composes the matrix into ``space`` and rebuilds the
    :class:`SpatialImage`.
    """

    @abstractmethod
    def _apply(
        self,
        detail: ImageDetail,
    ) -> tuple[ImageDetail, np.ndarray]:
        """Return the transformed detail and ``M_next_from_current`` (3x3)."""

    def _apply_many(
        self,
        details: list[ImageDetail],
    ) -> list[tuple[ImageDetail, np.ndarray]]:
        """Multi-input variant; default delegates per-image."""
        return [self._apply(d) for d in details]

    def _handles_list(self) -> bool:
        """Subclasses set True to force the list branch (e.g. for
        stages that compute a shared target size across inputs)."""
        return False

    def forward(self, image: SpatialInput) -> SpatialInput:
        if isinstance(image, list):
            return self._forward_list(image)
        if self._handles_list():
            return self._forward_list([image])[0]
        new_detail, mat = self._apply(image.detail)
        return _compose(image, new_detail, mat)

    def _forward_list(self, images: list[SpatialImage]) -> list[SpatialImage]:
        results = self._apply_many([im.detail for im in images])
        return [_compose(im, d, m) for im, (d, m) in zip(images, results)]

    def __call__(self, image: SpatialInput, *rest: SpatialImage) -> SpatialInput:
        if rest:
            if isinstance(image, list):
                raise TypeError("Pass either a list or varargs, not both.")
            return self._forward_list([image, *rest])
        return self.forward(image)

    def __rshift__(self, other: "SpatialStage") -> "SpatialPipeline":
        if not isinstance(other, SpatialStage):
            return NotImplemented
        left = list(self.stages) if isinstance(self, SpatialPipeline) else [self]
        right = list(other.stages) if isinstance(other, SpatialPipeline) else [other]
        return SpatialPipeline(left + right)


def _compose(
    spatial: SpatialImage,
    new_detail: ImageDetail,
    matrix_next_from_current: np.ndarray,
) -> SpatialImage:
    h = int(new_detail.image.shape[-2])
    w = int(new_detail.image.shape[-1])
    return spatial.with_detail_and_compose(
        new_detail, matrix_next_from_current, shape_hw=(h, w)
    )


class SpatialPipeline(SpatialStage):
    def __init__(self, stages: Sequence[SpatialStage]) -> None:
        self.stages: list[SpatialStage] = list(stages)

    def _apply(self, detail: ImageDetail) -> tuple[ImageDetail, np.ndarray]:
        raise NotImplementedError("SpatialPipeline composes via forward().")

    def forward(self, image: SpatialInput) -> SpatialInput:
        current = image
        for stage in self.stages:
            current = stage.forward(current)
        return current

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)

    def __repr__(self) -> str:
        names = [type(s).__name__ for s in self.stages]
        return f"SpatialPipeline([{', '.join(names)}])"
