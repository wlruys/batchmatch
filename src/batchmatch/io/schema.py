from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Schema",
    "IMAGEDETAIL_SCHEMA",
    "REGISTRATION_SCHEMA",
]


@dataclass(frozen=True)
class Schema:
    """Thin envelope for versioned JSON/binary payloads."""

    name: str
    version: int

    def envelope(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Wrap *payload* in a ``{"schema": …, "version": …}`` envelope."""
        if "schema" in payload or "version" in payload:
            raise ValueError("Payload already contains schema keys.")
        return {"schema": self.name, "version": self.version, **payload}

    def validate(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != self.name:
            raise TypeError(
                f"Invalid schema: {payload.get('schema')!r} (expected {self.name!r})."
            )
        if payload.get("version") != self.version:
            raise TypeError(
                f"Invalid version: {payload.get('version')!r} "
                f"(expected {self.version!r})."
            )


IMAGEDETAIL_SCHEMA = Schema("batchmatch.imagedetail", 1)
REGISTRATION_SCHEMA = Schema("batchmatch.registration", 3)
