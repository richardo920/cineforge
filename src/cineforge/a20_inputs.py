"""Explicit private A20 inputs and whole controller/creative payload freeze."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.paired_media import image_metadata
from cineforge.private_store import read_private_input
from cineforge.workflows import canonical

INPUT_NAMES = ("en_line", "zh_line", "actor_png", "music", "ambience", "foley",
               "voice_design", "seed")


class A20Error(ValueError):
    """Fixed diagnostics only; input names are closed vocabulary."""


@dataclass(frozen=True)
class CreativeInputs:
    paths: dict[str, Path]

    def read(self) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for name in INPUT_NAMES:
            if name not in self.paths:
                raise A20Error("A20_MISSING_" + name.upper())
            try:
                result[name] = read_private_input(self.paths[name], max_bytes=16 * 1024 * 1024)
            except (OSError, ValueError):
                raise A20Error("A20_MISSING_" + name.upper()) from None
            if not result[name].strip():
                raise A20Error("A20_MISSING_" + name.upper())
        if set(self.paths) != set(INPUT_NAMES):
            raise A20Error("A20_INPUT_KEYS")
        for name in ("en_line", "zh_line", "voice_design"):
            text = result[name].decode("utf-8")
            if not text.strip() or len(text) > 4096:
                raise A20Error("A20_INVALID_" + name.upper())
        if result["seed"].strip() != b"15001":
            raise A20Error("A20_INVALID_SEED")
        if self.paths["actor_png"].name == "m52_actor_pending_approval.png":
            raise A20Error("A20_PLACEHOLDER_ACTOR")
        image_metadata(result["actor_png"])
        for name in ("music", "ambience", "foley"):
            inspect_pcm16_wav(result[name])
        return result


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(roots: tuple[Path, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, root in enumerate(roots):
        if not root.is_dir() or root.is_symlink():
            raise A20Error("A20_CONTROLLER_ROOT")
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in ("__pycache__", ".git") for part in relative.parts):
                continue
            if path.is_symlink():
                raise A20Error("A20_CONTROLLER_SYMLINK")
            if path.is_file():
                result[f"{index}/{relative.as_posix()}"] = _hash(
                    read_private_input(path, max_bytes=16 * 1024 * 1024))
            if len(result) > 10000:
                raise A20Error("A20_CONTROLLER_BOUND")
    if not result:
        raise A20Error("A20_CONTROLLER_EMPTY")
    return result


@dataclass(frozen=True)
class FrozenInputs:
    creative: CreativeInputs
    roots: tuple[Path, ...]
    source_json: bytes
    data_json: bytes

    @classmethod
    def capture(cls, creative: CreativeInputs, roots: tuple[Path, ...]) -> FrozenInputs:
        # Hash actual bytes, not path strings, metadata or a proposed payload.
        package = Path(__file__).parent
        roots = (package, *(root for root in roots if root != package))
        data = {name: _hash(value) for name, value in creative.read().items()}
        return cls(creative, roots, canonical(_source(roots)), canonical(data))

    @property
    def receipt(self) -> dict[str, Any]:
        return {"gate": "frozen-CODE+DATA", "code_sha256": _hash(self.source_json),
                "data_sha256": _hash(self.data_json),
                "scope": "cineforge package and explicit operator controller root; actual creative bytes"}

    def check(self) -> None:
        if (canonical(_source(self.roots)) != self.source_json
                or canonical({name: _hash(value) for name, value in self.creative.read().items()})
                != self.data_json):
            raise A20Error("A20_FROZEN_INPUT_CHANGED")
