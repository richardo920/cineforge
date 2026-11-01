"""Immutable character metadata; private artifact bytes and conditioning live elsewhere."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import ClassVar, Self
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
from cineforge.rights import EvidenceRef, RightsLedger, RightsRef, SubjectRef, Use


class CharacterError(ContractError):
    """Invalid character metadata; diagnostics never interpolate private inputs."""


def _instance(value: object, kind: type[object], label: str) -> None:
    if type(value) is not kind:
        raise CharacterError(f"invalid {label}")


class _CharacterContract(Contract):
    __slots__ = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise CharacterError("unsupported character schema version")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        try:
            return super().from_dict(data)
        except (ContractError, TypeError, UnicodeError, RecursionError):
            raise CharacterError("invalid character metadata") from None

    @classmethod
    def from_json(cls, raw: str) -> Self:
        try:
            return super().from_json(raw)
        except (ContractError, TypeError, UnicodeError, RecursionError):
            raise CharacterError("invalid character JSON") from None


@dataclass(frozen=True, slots=True)
class CharacterPin(_CharacterContract):
    character_id: str
    revision: int
    sha256: str

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        _identifier(self.character_id, "character_id")
        _integer(self.revision, "character revision", 1)
        _digest(self.sha256, "character digest")

    def to_key(self) -> str:
        return f"{self.character_id}@{self.revision}#{self.sha256}"

    @classmethod
    def from_key(cls, value: str) -> CharacterPin:
        if not isinstance(value, str):
            raise CharacterError("invalid character pin")
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]{0,127})@([1-9][0-9]{0,18})#"
                             r"([0-9a-f]{64})", value)
        if match is None:
            raise CharacterError("invalid character pin")
        try:
            return cls(match[1], int(match[2]), match[3])
        except ContractError:
            raise CharacterError("invalid character pin") from None


@dataclass(frozen=True, slots=True)
class RightsBinding(_CharacterContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"subject": SubjectRef, "rights": RightsRef}

    subject: SubjectRef
    rights: RightsRef

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        _instance(self.subject, SubjectRef, "subject")
        _instance(self.rights, RightsRef, "rights")


@dataclass(frozen=True, slots=True)
class ReferenceArtifact(_CharacterContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"artifact": EvidenceRef,
                                                 "permission": RightsBinding}

    reference_id: str
    role: str
    artifact: EvidenceRef = field(repr=False)
    size_bytes: int
    media_type: str
    permission: RightsBinding

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        _identifier(self.reference_id, "reference_id")
        _instance(self.artifact, EvidenceRef, "artifact")
        _instance(self.permission, RightsBinding, "permission")
        _integer(self.size_bytes, "reference size", 1)
        roles = {"identity", "wardrobe", "conditioning", "voice"}
        if not isinstance(self.role, str) or self.role not in roles:
            raise CharacterError("invalid reference role")
        images = {"image/png", "image/jpeg", "image/webp"}
        formats = ({"audio/wav", "audio/flac"} if self.role == "voice" else images)
        if self.role == "conditioning":
            formats = formats | {"application/octet-stream"}
        if not isinstance(self.media_type, str) or self.media_type not in formats:
            raise CharacterError("invalid reference media type")
        parsed = urlsplit(self.artifact.uri)
        if parsed.scheme not in {"private", "artifact"} or not parsed.netloc:
            raise CharacterError("reference requires a private logical artifact URI")
        expected = "voice" if self.role == "voice" else "character_reference"
        if (self.permission.subject.kind != expected
                or self.permission.subject.sha256 != self.artifact.sha256):
            raise CharacterError("reference permission must bind its exact kind and bytes")


@dataclass(frozen=True, slots=True)
class Wardrobe(_CharacterContract):
    wardrobe_id: str
    description: str = field(repr=False)
    reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        _identifier(self.wardrobe_id, "wardrobe_id")
        _text(self.description, "wardrobe description")
        _tuple(self.reference_ids, str, "wardrobe reference IDs")
        for value in self.reference_ids:
            _identifier(value, "wardrobe reference ID")
        _unique(self.reference_ids, "wardrobe reference IDs")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        # slots=True replaces the class; no-arg super captures the old class on 3.11/3.12.
        return super(Wardrobe, cls).from_dict(_string_tuple(data, "reference_ids"))


def _string_tuple(data: object, name: str) -> object:
    if not isinstance(data, dict):
        return data
    values = dict(data)
    if name in values:
        if not isinstance(values[name], list):
            raise CharacterError("character references must be arrays")
        values[name] = tuple(values[name])
    return values


@dataclass(frozen=True, slots=True)
class ConditioningHint(_CharacterContract):
    """Descriptive pins only; adapters must independently verify support."""

    mode: str
    model_id: str
    model_revision: str
    workflow_digest: str
    reference_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        for value, label in ((self.mode, "conditioning mode"), (self.model_id, "model ID"),
                             (self.model_revision, "model revision")):
            _text(value, label)
        if self.model_revision.lower() in {"main", "master", "head", "latest", "unknown"}:
            raise CharacterError("conditioning model revision must be pinned")
        _digest(self.workflow_digest, "conditioning workflow digest")
        _tuple(self.reference_ids, str, "conditioning reference IDs")
        if not self.reference_ids:
            raise CharacterError("conditioning reference IDs cannot be empty")
        for value in self.reference_ids:
            _identifier(value, "conditioning reference ID")
        _unique(self.reference_ids, "conditioning reference IDs")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        return super(ConditioningHint, cls).from_dict(_string_tuple(data, "reference_ids"))


@dataclass(frozen=True, slots=True)
class CharacterReview(_CharacterContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"evidence": EvidenceRef}

    declaration_sha256: str
    reviewer: str
    evidence: EvidenceRef = field(repr=False)

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        _digest(self.declaration_sha256, "character review digest")
        _identifier(self.reviewer, "character reviewer")
        _instance(self.evidence, EvidenceRef, "character review evidence")


@dataclass(frozen=True, slots=True)
class CharacterRevision(_CharacterContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"wardrobe": Wardrobe}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"references": ReferenceArtifact,
                                                   "conditioning": ConditioningHint}

    character_id: str
    revision: int
    identity_kind: str
    identity_description: str = field(repr=False)
    wardrobe: Wardrobe
    references: tuple[ReferenceArtifact, ...] = field(repr=False)
    likeness: RightsBinding | None = None
    voice_reference_id: str | None = None
    conditioning: tuple[ConditioningHint, ...] = ()
    review: CharacterReview | None = None

    def __post_init__(self) -> None:
        _CharacterContract.__post_init__(self)
        _identifier(self.character_id, "character_id")
        _integer(self.revision, "character revision", 1)
        if self.identity_kind not in ("fictional", "person"):
            raise CharacterError("invalid identity kind")
        _text(self.identity_description, "identity description")
        _instance(self.wardrobe, Wardrobe, "wardrobe")
        _tuple(self.references, ReferenceArtifact, "character references")
        _unique(tuple(r.reference_id for r in self.references), "character reference IDs")
        refs = {r.reference_id: r for r in self.references}
        if not any(r.role == "identity" for r in self.references):
            raise CharacterError("character requires an identity image reference")
        for key in self.wardrobe.reference_ids:
            if key not in refs or refs[key].role != "wardrobe":
                raise CharacterError("missing or incompatible wardrobe reference")
        voice_ids = tuple(r.reference_id for r in self.references if r.role == "voice")
        if self.voice_reference_id is not None:
            _identifier(self.voice_reference_id, "voice reference ID")
        if voice_ids != (() if self.voice_reference_id is None else (self.voice_reference_id,)):
            raise CharacterError("missing or ambiguous voice reference")
        if self.likeness is not None:
            _instance(self.likeness, RightsBinding, "likeness binding")
            if (self.likeness.subject.kind != "likeness"
                    or self.likeness.subject.sha256 not in {
                        r.artifact.sha256 for r in self.references if r.role == "identity"}):
                raise CharacterError("likeness binding must identify an exact identity reference")
        _tuple(self.conditioning, ConditioningHint, "conditioning hints")
        _unique(self.conditioning, "conditioning hints")
        for hint in self.conditioning:
            if any(key not in refs or refs[key].role not in {"identity", "conditioning"}
                   for key in hint.reference_ids):
                raise CharacterError("missing or incompatible conditioning reference")
        if self.review is not None:
            _instance(self.review, CharacterReview, "character review")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if isinstance(data, dict):
            data = dict(data)
            for name, kind in (("likeness", RightsBinding), ("review", CharacterReview)):
                if name in data and data[name] is not None:
                    data[name] = kind.from_dict(data[name])
        return super(CharacterRevision, cls).from_dict(data)

    def declaration_digest(self) -> str:
        return hashlib.sha256(replace(self, review=None).to_json().encode()).hexdigest()

    def pin(self) -> CharacterPin:
        return CharacterPin(self.character_id, self.revision, self.declaration_digest())


def with_character_review(record: CharacterRevision, *, reviewer: str,
                          evidence: EvidenceRef) -> CharacterRevision:
    """Record a review assertion, distinct from current rights or model capability."""
    _instance(record, CharacterRevision, "character revision")
    return replace(record, review=CharacterReview(record.declaration_digest(), reviewer, evidence))


@dataclass(frozen=True, slots=True)
class CharacterResolution:
    allowed: bool
    reasons: tuple[str, ...]
    character: CharacterRevision | None = field(default=None, repr=False)


class CharacterRegistry:
    """In-memory immutable history; callers manage private metadata persistence."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, int], CharacterRevision] = {}

    def register(self, record: CharacterRevision) -> None:
        _instance(record, CharacterRevision, "character revision")
        key = (record.character_id, record.revision)
        if key in self._records:
            raise CharacterError("character revision already registered")
        self._records[key] = record

    def lookup(self, pin: CharacterPin) -> CharacterRevision | None:
        _instance(pin, CharacterPin, "character pin")
        record = self._records.get((pin.character_id, pin.revision))
        return record if record is not None and record.pin() == pin else None

    def resolve(self, pin: CharacterPin, ledger: RightsLedger, *,
                use: Use = Use("reference")) -> CharacterResolution:
        _instance(ledger, RightsLedger, "rights ledger")
        _instance(use, Use, "requested use")
        record = self.lookup(pin)
        if record is None:
            return CharacterResolution(False, ("missing_or_changed_character",))
        reasons: set[str] = set()
        if record.review is None:
            reasons.add("unreviewed_character")
        elif record.review.declaration_sha256 != record.declaration_digest():
            reasons.add("stale_character_review")
        if record.identity_kind == "person" and record.likeness is None:
            reasons.add("missing_likeness_permission")
        bindings = tuple(r.permission for r in record.references)
        if record.likeness is not None:
            bindings += (record.likeness,)
        for binding in bindings:
            reasons.update(ledger.preflight_use(binding.subject, binding.rights, use).reasons)
        if reasons:
            return CharacterResolution(False, tuple(sorted(reasons)))
        return CharacterResolution(True, (), record)

    def resolve_shot(self, shot: Shot, ledger: RightsLedger, *,
                     use: Use = Use("reference")) -> CharacterResolution:
        _instance(shot, Shot, "shot")
        _instance(ledger, RightsLedger, "rights ledger")
        _instance(use, Use, "requested use")
        if shot.character_revision is None:
            return CharacterResolution(True, ())
        return self.resolve(CharacterPin.from_key(shot.character_revision), ledger, use=use)


def pin_shot(shot: Shot, pin: CharacterPin) -> Shot:
    _instance(shot, Shot, "shot")
    _instance(pin, CharacterPin, "character pin")
    return replace(shot, character_revision=pin.to_key())
