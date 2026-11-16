"""Shared JSON mechanics; domain validation and error policies stay with callers."""

from __future__ import annotations

import json
import re
from typing import Any


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def strict_loads(raw: str | bytes | bytearray) -> Any:
    """Reject duplicate keys/constants without interpreting domain fields or roots."""
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def is_sha256_hex(value: object) -> bool:
    """Exactly a plain string of 64 lowercase hex digits; zero is valid syntax."""
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
