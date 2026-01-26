from __future__ import annotations

from typing import Optional
import torch
from batchmatch.base.pipeline import Stage, StageRegistry, _register_scalar
from batchmatch.base.tensordicts import ImageDetail, NestedKey

Tensor = torch.Tensor

norm_registry = StageRegistry("norm")


@norm_registry.register("l2")
class L2GradientNormStage(Stage):
    """
    Compute L2 norm of gradient components.

    Computes `norm = sqrt(gx^2 + gy^2 + eps + eta^2)` when eta is present.

    Args:
        eps: Small constant for numerical stability.
        inplace: Whether to reuse an existing norm buffer when available.
        output_key: Key to store the norm result.
        squared: Whether to output the squared norm (skip sqrt).
    """
    requires: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.X, ImageDetail.Keys.GRAD.Y})
    sets: frozenset[NestedKey] = frozenset({ImageDetail.Keys.GRAD.NORM})

    def __init__(
        self,
        *,
        eps: float = 1e-8,
        inplace: bool = False,
        output_key: Optional[str] = None,
        squared: bool = False,
    ) -> None:
        super().__init__()
        self._inplace = inplace
        self._squared = squared
        self._output_key = output_key or ImageDetail.Keys.GRAD.NORM
        _register_scalar(self, "_eps", eps, persistent=True)

        if output_key is not None:
            self._out_keys = [output_key]

    @property
    def eps(self) -> float:
        """
        Return epsilon value used for numerical stability.

        Returns:
            Epsilon scalar value.
        """
        return float(self._eps.item())

    def forward(self, image: ImageDetail) -> ImageDetail:
        gx = image.get(ImageDetail.Keys.GRAD.X)
        gy = image.get(ImageDetail.Keys.GRAD.Y)
        eps = self._eps
        eta = image.get(ImageDetail.Keys.GRAD.ETA, default=None)

        existing_buffer: Optional[Tensor] = None
        if self._inplace and self._output_key in image:
            existing_buffer = image.get(self._output_key)

        if existing_buffer is not None:
            torch.mul(gx, gx, out=existing_buffer)
            existing_buffer.addcmul_(gy, gy).add_(eps)

            if eta is not None:
                existing_buffer.addcmul_(eta, eta)

            if not self._squared:
                existing_buffer.sqrt_()

            norm = existing_buffer
        else:
            norm_sq = gx.square() + gy.square() + eps

            if eta is not None:
                norm_sq = norm_sq + eta.square()

            if not self._squared:
                norm = norm_sq.sqrt()
            else:
                norm = norm_sq

        image.set(self._output_key, norm)
        return image

    def extra_repr(self) -> str:
        parts = [f"eps={self.eps:.0e}"]
        if self._inplace:
            parts.append("inplace=True")
        if self._squared:
            parts.append("squared=True")
        if self._output_key != ImageDetail.Keys.GRAD.NORM:
            parts.append(f"output_key={self._output_key!r}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"L2GradientNormStage({self.extra_repr()})"


def build_gradient_norm_operator(norm_type: str, **kwargs) -> Stage:
    """
    Build a gradient norm stage by registered name.

    Args:
        norm_type: Registered norm stage name.
        **kwargs: Parameters forwarded to the stage constructor.

    Returns:
        Instantiated norm stage.
    """
    return norm_registry.build(norm_type, **kwargs)
