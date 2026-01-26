from __future__ import annotations

import os
import pathlib
from typing import Any, Iterable, Optional, Union

PathLike = Union[str, os.PathLike, pathlib.Path]
_DEFAULT = object()

__all__ = [
    "PathLike",
    "_DEFAULT",
    "pathify",
    "ensure_dir",
    "ensure_parent_dir",
    "coerce_extensions",
    "assert_overwrite",
]


def pathify(path: PathLike) -> pathlib.Path:
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def coerce_extensions(extensions: Optional[Iterable[str]]) -> set[str]:
    if extensions is None:
        return {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}


def assert_overwrite(path: pathlib.Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
