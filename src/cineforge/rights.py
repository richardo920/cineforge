"""Offline rights declarations and fail-closed operator-review gates.

Evidence references are metadata, never fetched. A passing declaration records an
operator's decision; it is neither legal certification nor execution authorization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from urllib.parse import urlsplit

from cineforge.contracts import Contract, ContractError, _digest, _identifier, _text, _tuple, _unique

_KINDS = frozenset({"code", "model", "skill", "workflow", "character_reference", "likeness",
                    "voice", "music", "asset"})
_ACTIONS = frozenset({"reference", "install", "execute", "condition", "generate", "modify",
                      "copy", "export", "distribute"})
_RESTRICTIONS = frozenset({"noncommercial", "internal_only", "no_modification",
                          "no_distribution"})
_UNKNOWN = frozenset({"", "unknown", "unreviewed", "noassertion", "none", "unlicensed"})
_MUTABLE = frozenset({"main", "master", "head", "latest", "unknown", "unversioned"})


class RightsError(ContractError):
    """Invalid declaration; diagnostics deliberately exclude supplied content."""


def _member(value: object, allowed: frozenset[str], label: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise RightsError(f"invalid {label}")


def _bool(value: object, label: str) -> None:
    if type(value) is not bool:
        raise RightsError(f"invalid {label}")


def _instance(value: object, kind: type[object], label: str) -> None:
    if type(value) is not kind:
        raise RightsError(f"invalid {label}")


@dataclass(frozen=True, slots=True)
class SubjectRef(Contract):
    kind: str
    subject_id: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RightsError("unsupported subject schema version")
        _member(self.kind, _KINDS, "subject kind")
        _identifier(self.subject_id, "subject_id")
        _identifier(self.revision, "subject revision")
        _digest(self.sha256, "subject sha256")
        if self.revision.lower() in _MUTABLE:
            raise RightsError("subject revision must be pinned")


@dataclass(frozen=True, slots=True)
class RightsRef(Contract):
    record_id: str
    revision: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RightsError("unsupported rights reference schema version")
        _identifier(self.record_id, "rights record_id")
        _identifier(self.revision, "rights revision")
        if self.revision.lower() in _MUTABLE:
            raise RightsError("rights revision must be pinned")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    uri: str = field(repr=False)
    sha256: str

    def __post_init__(self) -> None:
        _text(self.uri, "evidence URI")
        _digest(self.sha256, "evidence sha256")
        try:
            parsed = urlsplit(self.uri)
            if (not parsed.scheme or not parsed.path and not parsed.netloc or parsed.username
                    or parsed.password or parsed.query or parsed.fragment
                    or any(ord(char) < 32 for char in self.uri)):
                raise ValueError
        except ValueError:
            raise RightsError("invalid or credential-bearing evidence URI") from None


@dataclass(frozen=True, slots=True, order=True)
class Use:
    action: str
    audience: str = "internal"
    commercial: bool = False

    def __post_init__(self) -> None:
        _member(self.action, _ACTIONS, "use action")
        _member(self.audience, frozenset({"internal", "public"}), "use audience")
        _bool(self.commercial, "commercial context")


@dataclass(frozen=True, slots=True)
class RightsReview:
    declaration_sha256: str
    reviewer: str
    evidence: EvidenceRef = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.declaration_sha256, "review declaration digest")
        _identifier(self.reviewer, "reviewer")
        _instance(self.evidence, EvidenceRef, "review evidence")


@dataclass(frozen=True, slots=True)
class RightsRecord:
    ref: RightsRef
    subject: SubjectRef
    source: EvidenceRef | None = field(default=None, repr=False)
    source_revision: str | None = field(default=None, repr=False)
    license_id: str = field(default="unknown", repr=False)
    terms: EvidenceRef | None = field(default=None, repr=False)
    notices: tuple[str, ...] = field(default=(), repr=False)
    attribution: str = field(default="", repr=False)
    notices_required: bool = True
    attribution_required: bool = True
    allowed_uses: tuple[Use, ...] = ()
    restrictions: tuple[str, ...] = ()
    consent_required: bool = False
    consent: EvidenceRef | None = field(default=None, repr=False)
    review: RightsReview | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RightsError("unsupported rights schema version")
        _instance(self.ref, RightsRef, "rights reference")
        _instance(self.subject, SubjectRef, "subject reference")
        for value in (self.source, self.terms, self.consent):
            if value is not None:
                _instance(value, EvidenceRef, "evidence reference")
        if self.source_revision is not None:
            _identifier(self.source_revision, "source revision")
            if self.source_revision.lower() in _MUTABLE:
                raise RightsError("source revision must be pinned")
        if not isinstance(self.license_id, str) or "\x00" in self.license_id:
            raise RightsError("invalid license declaration")
        for value in (self.notices_required, self.attribution_required, self.consent_required):
            _bool(value, "rights requirement")
        _tuple(self.notices, str, "notices")
        for notice in self.notices:
            _text(notice, "notice")
        _unique(self.notices, "notices")
        if not isinstance(self.attribution, str) or "\x00" in self.attribution:
            raise RightsError("invalid attribution")
        _tuple(self.allowed_uses, Use, "allowed uses")
        _unique(self.allowed_uses, "allowed uses")
        _tuple(self.restrictions, str, "restrictions")
        _unique(self.restrictions, "restrictions")
        for restriction in self.restrictions:
            _member(restriction, _RESTRICTIONS, "restriction")
        if self.review is not None:
            _instance(self.review, RightsReview, "rights review")

    def declaration_digest(self) -> str:
        """Canonical content binding, excluding the review that signs this declaration."""
        data = asdict(self)
        data.pop("review")
        # Ordering of explicit sets is immaterial; notices retain their declared order.
        data["allowed_uses"] = [asdict(use) for use in sorted(self.allowed_uses)]
        data["restrictions"] = sorted(self.restrictions)
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class UseRequest:
    subject: SubjectRef
    rights: RightsRef
    use: Use

    def __post_init__(self) -> None:
        _instance(self.subject, SubjectRef, "requested subject")
        _instance(self.rights, RightsRef, "requested rights")
        _instance(self.use, Use, "requested use")


@dataclass(frozen=True, slots=True)
class Attribution:
    subject: SubjectRef
    rights: RightsRef
    declaration_sha256: str
    license_id: str
    notices: tuple[str, ...]
    attribution: str
    use: Use


@dataclass(frozen=True, slots=True)
class PreflightResult:
    allowed: bool
    reasons: tuple[str, ...]
    attributions: tuple[Attribution, ...] = ()


@dataclass(frozen=True, slots=True)
class Revocation:
    rights: RightsRef
    reason: str = field(repr=False)


class RightsLedger:
    """In-memory immutable history plus current revocation/supersession state.

    Registering a replacement rights record for an identical pinned subject makes
    the earlier grant inactive. Other subject revisions retain their own grants.
    """

    def __init__(self) -> None:
        self._records: dict[RightsRef, RightsRecord] = {}
        self._active: dict[SubjectRef, RightsRef] = {}
        self._revocations: dict[RightsRef, Revocation] = {}

    def register(self, record: RightsRecord) -> None:
        _instance(record, RightsRecord, "rights record")
        if record.ref in self._records:
            raise RightsError("duplicate immutable rights reference")
        self._records[record.ref] = record
        self._active[record.subject] = record.ref

    def lookup(self, ref: RightsRef) -> RightsRecord | None:
        _instance(ref, RightsRef, "rights reference")
        return self._records.get(ref)

    def revocation(self, ref: RightsRef) -> Revocation | None:
        _instance(ref, RightsRef, "rights reference")
        return self._revocations.get(ref)

    def is_revoked(self, ref: RightsRef) -> bool:
        return self.revocation(ref) is not None

    def revoke(self, ref: RightsRef, *, reason: str) -> None:
        _instance(ref, RightsRef, "rights reference")
        _text(reason, "revocation reason")
        if ref not in self._records:
            raise RightsError("cannot revoke missing rights record")
        if ref in self._revocations:
            raise RightsError("rights record already revoked")
        self._revocations[ref] = Revocation(ref, reason)

    def preflight_use(self, subject: SubjectRef, rights: RightsRef, use: Use) -> PreflightResult:
        request = UseRequest(subject, rights, use)
        record = self.lookup(request.rights)
        if record is None:
            return PreflightResult(False, ("missing_rights",))
        reasons: set[str] = set()
        if record.subject != subject:
            reasons.add("subject_mismatch")
        if self._active.get(subject) != rights:
            reasons.add("superseded_rights")
        if self.is_revoked(rights):
            reasons.add("revoked_rights")
        if record.review is None:
            reasons.add("unreviewed_rights")
        elif record.review.declaration_sha256 != record.declaration_digest():
            reasons.add("stale_review")
        if record.source is None or record.source_revision is None:
            reasons.add("missing_provenance")
        if record.terms is None or record.license_id.strip().lower() in _UNKNOWN:
            reasons.add("unknown_terms")
        if record.notices_required and not record.notices:
            reasons.add("missing_notices")
        if record.attribution_required and not record.attribution.strip():
            reasons.add("missing_attribution")
        if (record.consent_required or subject.kind in {"voice", "likeness", "character_reference"}
                ) and record.consent is None:
            reasons.add("missing_consent")
        if use not in record.allowed_uses:
            reasons.add("use_not_permitted")
        if (("noncommercial" in record.restrictions and use.commercial)
                or ("internal_only" in record.restrictions and use.audience != "internal")
                or ("no_modification" in record.restrictions and use.action == "modify")
                or ("no_distribution" in record.restrictions
                    and use.action in {"export", "distribute"})):
            reasons.add("restricted_use")
        if reasons:
            return PreflightResult(False, tuple(sorted(reasons)))
        attribution = Attribution(subject, rights, record.declaration_digest(), record.license_id,
                                  record.notices, record.attribution, use)
        return PreflightResult(True, (), (attribution,))

    def preflight_export(self, requests: tuple[UseRequest, ...]) -> PreflightResult:
        """All-or-nothing metadata preflight; no export, legal or spend authority."""
        _tuple(requests, UseRequest, "export requests")
        if not requests:
            return PreflightResult(False, ("empty_export",))
        reasons: set[str] = set()
        attributions: set[Attribution] = set()
        for request in requests:
            if request.use.action not in {"export", "distribute"}:
                reasons.add("invalid_export_action")
            result = self.preflight_use(request.subject, request.rights, request.use)
            reasons.update(result.reasons)
            attributions.update(result.attributions)
        if reasons:
            return PreflightResult(False, tuple(sorted(reasons)))
        ordered = tuple(sorted(attributions, key=lambda item: (
            item.subject.kind, item.subject.subject_id, item.subject.revision, item.subject.sha256,
            item.rights.record_id, item.rights.revision, item.use)))
        return PreflightResult(True, (), ordered)


def with_review(record: RightsRecord, *, reviewer: str, evidence: EvidenceRef) -> RightsRecord:
    """Attach an operator's review assertion; never infer a grant from a license label."""
    _instance(record, RightsRecord, "rights record")
    return replace(record, review=RightsReview(record.declaration_digest(), reviewer, evidence))
