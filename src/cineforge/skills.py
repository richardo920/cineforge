"""Pinned reference catalog; discovery and review never grant runtime authority."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import PurePosixPath
from typing import Any, Callable, ClassVar, Self
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
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.rights import Attribution, EvidenceRef, RightsLedger, RightsRef, SubjectRef, Use

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_CATEGORIES = {"story", "camera", "audio", "editing", "model_card", "workflow", "recipe"}


class CatalogError(ContractError):
    """Invalid or unresolved catalog metadata; input content is not echoed."""


def _safe_validation(validate: Callable[[Any], None]) -> Callable[[Any], None]:
    """Keep constructor failures, including chained parser errors, metadata-free."""
    @wraps(validate)
    def guarded(record: Any) -> None:
        try:
            validate(record)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise CatalogError("invalid catalog declaration") from None
    return guarded


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

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise CatalogError("unsupported catalog schema version")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        try:
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
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise CatalogError("invalid catalog record") from None

    @classmethod
    def from_json(cls, raw: str) -> Self:
        try:
            return super().from_json(raw)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise CatalogError("invalid catalog JSON") from None


@dataclass(frozen=True, slots=True)
class SourcePin(_Record):
    repository: str
    revision: str
    path: str
    sha256: str

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
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

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
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
    required_precision: Precision | None = None
    _enums: ClassVar[dict[str, type[Enum]]] = {"backend": Backend, "required_precision": Precision}

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
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
        if self.required_precision is not None and not isinstance(self.required_precision, Precision):
            raise CatalogError("invalid compatibility precision")
        if self.model_id is None and (self.backend is not None or self.minimum_memory_mib
                                      or self.required_precision is not None):
            raise CatalogError("model-neutral guidance cannot declare a hardware profile")
        if self.backend is None and (self.minimum_memory_mib or self.required_precision is not None):
            raise CatalogError("hardware requirements need an explicit backend")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if isinstance(data, dict):
            data = {key: value for key, value in data.items()
                    if key not in cls._enums or value is not None}
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
                      or device.memory_mib < self.minimum_memory_mib
                      or (self.required_precision is not None
                          and self.required_precision not in device.supported_precisions)):
                    reasons.append("incompatible_hardware")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ProfileEvidence(_Record):
    """Keep maintainer advice separate from a bounded local observation."""

    origin: str
    source: SourcePin
    summary: str
    _objects: ClassVar[dict[str, type[Contract]]] = {"source": SourcePin}

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
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

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        _digest(self.declaration_sha256, "review subject")
        _identifier(self.reviewer, "source reviewer")
        _text(self.evidence_uri, "review evidence")
        _digest(self.evidence_sha256, "review evidence digest")
        EvidenceRef(self.evidence_uri, self.evidence_sha256)


@dataclass(frozen=True, slots=True)
class SkillPin(_Record):
    entry_id: str
    revision: str
    sha256: str

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
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

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
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


@dataclass(frozen=True, slots=True)
class CatalogEntry(_Record):
    entry_id: str
    revision: str
    category: str
    summary: str
    source: SourcePin
    entrypoint: str
    permissions: Permissions
    compatibility: Compatibility
    rights: RightsRef
    mode: str = "reference_only"
    supplemental_sources: tuple[SourcePin, ...] = ()
    profiles: tuple[ProfileEvidence, ...] = ()
    limitations: tuple[str, ...] = ()
    review: SourceReview | None = None
    _strings: ClassVar[tuple[str, ...]] = ("limitations",)
    _objects: ClassVar[dict[str, type[Contract]]] = {
        "source": SourcePin, "permissions": Permissions, "compatibility": Compatibility,
        "rights": RightsRef, "review": SourceReview,
    }
    _sequences: ClassVar[dict[str, type[Contract]]] = {
        "supplemental_sources": SourcePin, "profiles": ProfileEvidence,
    }

    @_safe_validation
    def __post_init__(self) -> None:
        _Record.__post_init__(self)
        SkillPin(self.entry_id, self.revision, "0" * 64)
        if self.category not in _CATEGORIES:
            raise CatalogError("invalid catalog category")
        if self.mode not in {"reference_only", "isolated_skill"}:
            raise CatalogError("invalid catalog mode")
        _text(self.summary, "catalog summary")
        for value, kind in ((self.source, SourcePin), (self.permissions, Permissions),
                            (self.compatibility, Compatibility), (self.rights, RightsRef)):
            if type(value) is not kind:
                raise CatalogError("invalid catalog declaration")
        if self.entrypoint != self.source.path:
            raise CatalogError("entrypoint must be the pinned source file")
        _tuple(self.supplemental_sources, SourcePin, "supplemental sources")
        _tuple(self.profiles, ProfileEvidence, "profiles")
        _tuple(self.limitations, str, "limitations")
        _unique(self.supplemental_sources, "supplemental sources")
        for limitation in self.limitations:
            _text(limitation, "catalog limitation")
        if self.review is not None and type(self.review) is not SourceReview:
            raise CatalogError("invalid source review")

    @property
    def digest(self) -> str:
        """Review binds all source, permission, compatibility and rights-reference metadata."""
        return _digest_record(self, exclude=("review",))

    @property
    def pin(self) -> SkillPin:
        return SkillPin(self.entry_id, self.revision, self.digest)

    @property
    def subject(self) -> SubjectRef:
        kind = "workflow" if self.category in {"workflow", "recipe"} else "skill"
        if self.category == "model_card":
            kind = "model"
        return SubjectRef(kind, self.entry_id, self.revision, self.digest)


@dataclass(frozen=True, slots=True)
class ReferenceReadiness:
    """Reference-selection gate only; isolation, tools, GPU and release have later gates."""

    pin: SkillPin
    ready: bool
    reasons: tuple[str, ...]
    attributions: tuple[Attribution, ...] = ()

    @property
    def execution_authorized(self) -> bool:
        return False


class SkillCatalog:
    """Immutable entry history; callers must supply exact pins, never resolve 'latest'."""

    def __init__(self, entries: tuple[CatalogEntry, ...] = ()) -> None:
        _tuple(entries, CatalogEntry, "catalog entries")
        self._entries: dict[tuple[str, str], CatalogEntry] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: CatalogEntry) -> None:
        if type(entry) is not CatalogEntry:
            raise CatalogError("invalid catalog entry")
        key = (entry.entry_id, entry.revision)
        if key in self._entries:
            raise CatalogError("duplicate immutable catalog revision")
        self._entries[key] = entry

    @property
    def pins(self) -> tuple[SkillPin, ...]:
        """Enumerate every explicit revision, with no automatic upgrade selection."""
        return tuple(self._entries[key].pin for key in sorted(self._entries))

    def lookup(self, pin: SkillPin) -> CatalogEntry:
        if type(pin) is not SkillPin:
            raise CatalogError("invalid skill pin")
        entry = self._entries.get((pin.entry_id, pin.revision))
        if entry is None:
            raise CatalogError("missing catalog revision")
        if entry.digest != pin.sha256:
            raise CatalogError("catalog content differs from pinned declaration")
        return entry

    def reference_readiness(self, pin: SkillPin, *, shot: Shot, ledger: RightsLedger,
                            device: GPUDevice | None = None,
                            use: Use = Use("reference")) -> ReferenceReadiness:
        if type(shot) is not Shot or type(ledger) is not RightsLedger or type(use) is not Use:
            raise CatalogError("invalid reference context")
        if device is not None and type(device) is not GPUDevice:
            raise CatalogError("invalid device context")
        entry = self.lookup(pin)
        reasons = set(entry.compatibility.reasons(shot, device))
        if entry.review is None:
            reasons.add("unreviewed_source")
        elif entry.review.declaration_sha256 != entry.digest:
            reasons.add("stale_source_review")
        if use.action != "reference":
            reasons.add("reference_use_required")
        if entry.mode == "reference_only" and entry.permissions != Permissions():
            reasons.add("reference_permissions_must_be_empty")
        rights = ledger.preflight_use(entry.subject, entry.rights, use)
        reasons.update(rights.reasons)
        ordered = tuple(sorted(reasons))
        return ReferenceReadiness(pin, not ordered, ordered,
                                  rights.attributions if not ordered else ())

    def resolve_shot(self, binding: ShotSkills, *, shot: Shot, ledger: RightsLedger,
                     device: GPUDevice | None = None,
                     use: Use = Use("reference")) -> tuple[ReferenceReadiness, ...]:
        if type(binding) is not ShotSkills:
            raise CatalogError("invalid shot skill binding")
        if bind_shot(shot, binding.skills) != binding:
            raise CatalogError("shot differs from reviewed skill binding")
        return tuple(self.reference_readiness(pin, shot=shot, ledger=ledger, device=device, use=use)
                     for pin in binding.skills)
