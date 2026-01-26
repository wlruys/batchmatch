from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Schema",
    "IMAGEDETAIL_SCHEMA",
    "PRODUCT_SCHEMA",
]


@dataclass(frozen=True)
class Schema:
    name: str
    version: int

    def with_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "schema" in payload or "schema_version" in payload:
            raise ValueError("Payload already contains schema metadata.")
        return {"schema": self.name, "schema_version": self.version, **payload}

    def validate(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != self.name:
            raise TypeError(f"Invalid schema: {payload.get('schema')!r} (expected {self.name!r}).")
        if payload.get("schema_version") != self.version:
            raise TypeError(
                f"Invalid schema_version: {payload.get('schema_version')!r} "
                f"(expected {self.version!r})."
            )


IMAGEDETAIL_SCHEMA = Schema("batchmatch.imagedetail", 1)
PRODUCT_SCHEMA = Schema("batchmatch.product", 1)
