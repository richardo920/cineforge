"""Pinned reference catalog; discovery and review never grant runtime authority."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

from cineforge.contracts import (
    Contract,
    ContractError,
    Shot,
    _digest,
    _identifier,
    _integer,
    _text,
    _tuple,
    _unique,
)
from cineforge.devices import Backend, GPUDevice

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_CATEGORIES = {"story", "camera", "audio", "editing", "model_card", "workflow", "recipe"}


class CatalogError(ContractError):
    """Invalid or unresolved catalog metadata; input content is not echoed."""


def _path(value: str) -> None:
    _text(value, "source path")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value or ".." in path.parts
            or any(char in value for char in "\\%?#:") or value == "."
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise CatalogError("invalid relative source path")


def _commit(value: str) -> None:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise CatalogError("source revision requires a full immutable commit")


class _Record(Contract):
    _strings: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict):
            raise CatalogError("catalog record must be an object")
        values = dict(data)
        for name in cls._strings:
            if name in values:
                if not isinstance(values[name], list):
                    raise CatalogError("catalog sequence must be an array")
                values[name] = tuple(values[name])
        # Optional nested records have None defaults; Contract's codec is strict.
        for name in cls._objects:
            if name in values and values[name] is None:
                values.pop(name)
        return super().from_dict(values)


@dataclass(frozen=True, slots=True)
class SourcePin(_Record):
    repository: str
    revision: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _text(self.repository, "repository")
        parsed = urlsplit(self.repository)
        if (parsed.scheme != "https" or parsed.netloc not in {"github.com", "huggingface.co"}
                or parsed.query or parsed.fragment or parsed.username or parsed.password
                or len(parsed.path.strip("/").split("/")) != 2
                or parsed.path.endswith("/")
                or any(ord(char) < 33 or ord(char) > 126 for char in self.repository)):
            raise CatalogError("source repository must be a public canonical HTTPS repository")
        _path(parsed.path.removeprefix("/"))
        _commit(self.revision)
        _path(self.path)
        _digest(self.sha256, "source digest")

    @property
    def uri(self) -> str:
        return f"{self.repository}/blob/{self.revision}/{self.path}"


@dataclass(frozen=True, slots=True)
class Permissions(_Record):
    """Declared ceiling for a later isolated runtime, never a grant to run tools."""

    tools: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    network_hosts: tuple[str, ...] = ()
    _strings: ClassVar[tuple[str, ...]] = ("tools", "read_paths", "write_paths", "network_hosts")

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        for name in self._strings:
            values = getattr(self, name)
            _tuple(values, str, name)
            _unique(values, name)
        for tool in self.tools:
            _identifier(tool, "tool")
        for path in (*self.read_paths, *self.write_paths):
            _path(path)
        for host in self.network_hosts:
            if re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", host) is None:
                raise CatalogError("invalid network host")


@dataclass(frozen=True, slots=True)
class Compatibility(_Record):
    """Either model-neutral guidance or one exact model/workflow/backend profile."""

    model_id: str | None = None
    model_revision: str | None = None
    workflow_digest: str | None = None
    backend: Backend | None = None
    minimum_memory_mib: int = 0
    _enums: ClassVar[dict[str, type[Enum]]] = {"backend": Backend}

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _integer(self.minimum_memory_mib, "minimum memory")
        values = (self.model_id, self.model_revision, self.workflow_digest)
        if any(value is not None for value in values):
            if any(value is None for value in values):
                raise CatalogError("compatibility requires exact model, revision and workflow")
            _text(self.model_id, "model id")
            _commit(self.model_revision)  # type: ignore[arg-type]
            _digest(self.workflow_digest, "workflow digest")
        if self.backend is not None and not isinstance(self.backend, Backend):
            raise CatalogError("invalid compatibility backend")
        if self.model_id is None and (self.backend is not None or self.minimum_memory_mib):
            raise CatalogError("model-neutral guidance cannot declare a hardware profile")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if isinstance(data, dict) and data.get("backend") is None:
            data = {key: value for key, value in data.items() if key != "backend"}
        return super(Compatibility, cls).from_dict(data)

    def reasons(self, shot: Shot, device: GPUDevice | None) -> tuple[str, ...]:
        reasons = []
        if self.model_id is not None:
            if (shot.model_id, shot.model_revision, shot.workflow_digest) != (
                    self.model_id, self.model_revision, self.workflow_digest):
                reasons.append("incompatible_model_workflow")
            if self.backend is not None:
                if device is None:
                    reasons.append("missing_device_profile")
                elif (device.backend != self.backend
                      or device.memory_mib < self.minimum_memory_mib):
                    reasons.append("incompatible_hardware")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ProfileEvidence(_Record):
    """Keep maintainer advice separate from a bounded local observation."""

    origin: str
    source: SourcePin
    summary: str
    _objects: ClassVar[dict[str, type[Contract]]] = {"source": SourcePin}

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        if self.origin not in {"upstream_recommendation", "local_observation"}:
            raise CatalogError("invalid profile provenance")
        if type(self.source) is not SourcePin:
            raise CatalogError("invalid profile source")
        _text(self.summary, "profile summary")


@dataclass(frozen=True, slots=True)
class SourceReview(_Record):
    declaration_sha256: str
    reviewer: str
    evidence_uri: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _digest(self.declaration_sha256, "review subject")
        _identifier(self.reviewer, "source reviewer")
        _text(self.evidence_uri, "review evidence")
        _digest(self.evidence_sha256, "review evidence digest")


@dataclass(frozen=True, slots=True)
class SkillPin(_Record):
    entry_id: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.entry_id, "catalog id")
        _identifier(self.revision, "catalog revision")
        if self.revision.lower() in {"main", "master", "latest", "head", "unknown"}:
            raise CatalogError("catalog revision must be pinned")
        _digest(self.sha256, "catalog digest")


@dataclass(frozen=True, slots=True)
class ShotSkills(_Record):
    """Sidecar binding preserves the unchanged version-one Shot wire contract."""

    shot_sha256: str
    skills: tuple[SkillPin, ...]
    _sequences: ClassVar[dict[str, type[Contract]]] = {"skills": SkillPin}

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _digest(self.shot_sha256, "shot digest")
        _tuple(self.skills, SkillPin, "skill pins")
        _unique(tuple(pin.entry_id for pin in self.skills), "shot skill id")
        if not self.skills:
            raise CatalogError("shot binding requires explicit skill pins")


def bind_shot(shot: Shot, skills: tuple[SkillPin, ...]) -> ShotSkills:
    if type(shot) is not Shot:
        raise CatalogError("invalid shot")
    return ShotSkills(hashlib.sha256(shot.to_json().encode()).hexdigest(), skills)


def _digest_record(record: Contract, *, exclude: tuple[str, ...] = ()) -> str:
    data: dict[str, Any] = record.to_dict()
    for key in exclude:
        data.pop(key)
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()
