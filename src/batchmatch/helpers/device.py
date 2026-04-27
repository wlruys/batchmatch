from __future__ import annotations

import torch

__all__ = ["auto_device"]


def auto_device(device: str | torch.device = "auto") -> torch.device:
    """Return a ``torch.device``, resolving ``"auto"`` to the best available backend.

    Priority order: CUDA > MPS > CPU.

    Args:
        device: A device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, …)
            or an existing :class:`torch.device`.  When *device* is not
            ``"auto"`` it is passed straight to :class:`torch.device`.

    Returns:
        A resolved :class:`torch.device`.
    """
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
