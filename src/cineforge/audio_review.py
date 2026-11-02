"""Exact-media listening/visual review metadata; no human authentication or export.

Only trusted operator code supplies a technical verifier and owner decisions.
Checksums verify supplied bytes, not codecs, speech, native execution or human origin.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, Self

from cineforge.audio import CueSheet, MediaDependencies, RevisionPin, VoiceIdentity, _AudioContract
from cineforge.audio_mix import MixedArtifact
from cineforge.characters import CharacterPin
from cineforge.contracts import Contract, ContractError, _digest, _identifier, _integer
from cineforge.json_codec import canonical_json
from cineforge.rights import EvidenceRef
from cineforge.speech import SpeechArtifact

HUMAN_CHECKLIST = (
    "actor_identity", "artifacts", "balance", "en_intelligibility", "en_visible_lip_sync",
    "musical_fit", "shared_voice_identity", "transitions", "zh_intelligibility",
    "zh_visible_lip_sync",
)
TECHNICAL_CHECKLIST = (
    "mix_measurement", "native_speech_en", "native_speech_zh", "native_video_en",
    "native_video_zh", "timing",
)
_STATUSES = frozenset({"pass", "fail", "pending", "skipped", "unavailable"})
_VIDEO = frozenset({"video/mp4", "video/webm"})


class ReviewError(ContractError):
    """Fixed diagnostics excluding private media, evidence and verifier exceptions."""


def _instance(value: object, kind: type[object]) -> None:
    if type(value) is not kind:
        raise ReviewError("invalid media review record")


class _ReviewContract(_AudioContract):
    __slots__ = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ReviewError("unsupported media review schema")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        answer = None
        try:
            answer = super().from_dict(data)
        except (ContractError, TypeError, UnicodeError, RecursionError, OverflowError):
            pass
        if answer is None:
            raise ReviewError("invalid media review metadata")
        return answer

    @classmethod
    def from_json(cls, raw: str) -> Self:
        answer = None
        try:
            answer = super().from_json(raw)
        except (ContractError, TypeError, UnicodeError, RecursionError, OverflowError):
            pass
        if answer is None:
            raise ReviewError("invalid media review JSON")
        return answer


@dataclass(frozen=True, slots=True)
class ReviewAsset(_ReviewContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"pin": RevisionPin}
    pin: RevisionPin
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _ReviewContract.__post_init__(self)
        _instance(self.pin, RevisionPin)
        _integer(self.size_bytes, "review media size", 1, 128 * 1024 * 1024)
        if type(self.media_type) is not str or self.media_type not in _VIDEO | {"audio/wav"}:
            raise ReviewError("unsupported review media format")
        if self.media_type == "audio/wav" and self.size_bytes > 16 * 1024 * 1024:
            raise ReviewError("review audio exceeds supported byte bound")

    def check_bytes(self, payload: bytes) -> None:
        if (type(payload) is not bytes or len(payload) != self.size_bytes
                or hashlib.sha256(payload).hexdigest() != self.pin.sha256):
            raise ReviewError("review media bytes do not match")


@dataclass(frozen=True, slots=True)
class ReviewRecipe(_ReviewContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {
        name: RevisionPin for name in ("speech", "video_model", "video_workflow", "video_runtime")
    }
    speech: RevisionPin
    video_model: RevisionPin
    video_workflow: RevisionPin
    video_runtime: RevisionPin

    def __post_init__(self) -> None:
        _ReviewContract.__post_init__(self)
        for name in self._objects:
            _instance(getattr(self, name), RevisionPin)


@dataclass(frozen=True, slots=True)
class LanguageClip(_ReviewContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {
        "speech": ReviewAsset, "video": ReviewAsset, "request": RevisionPin,
        "utterance": RevisionPin, "character": CharacterPin, "voice": VoiceIdentity,
        "reference": RevisionPin, "recipe": ReviewRecipe,
    }
    language: str
    speech: ReviewAsset
    video: ReviewAsset
    request: RevisionPin
    utterance: RevisionPin
    character: CharacterPin
    voice: VoiceIdentity
    reference: RevisionPin
    recipe: ReviewRecipe
    speech_origin: str

    def __post_init__(self) -> None:
        _ReviewContract.__post_init__(self)
        for name, kind in self._objects.items():
            _instance(getattr(self, name), kind)
        if (type(self.language) is not str or self.language not in ("en", "zh")
                or self.speech.media_type != "audio/wav" or self.video.media_type not in _VIDEO
                or self.voice.reference_id != self.reference.input_id
                or self.speech_origin not in ("synthetic_fixture", "unverified_injected_runtime")):
            raise ReviewError("unsupported language clip declaration")


@dataclass(frozen=True, slots=True)
class ReviewSnapshot(_ReviewContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {
        "delivered": ReviewAsset, "mix": ReviewAsset, "mix_report": RevisionPin,
        "dependencies": MediaDependencies,
    }
    _sequences: ClassVar[dict[str, type[Contract]]] = {"clips": LanguageClip}
    snapshot_id: str
    revision: int
    delivered: ReviewAsset
    mix: ReviewAsset
    mix_report: RevisionPin
    mix_settings_sha256: str
    mix_status: str
    dependencies: MediaDependencies
    clips: tuple[LanguageClip, ...]

    def __post_init__(self) -> None:
        _ReviewContract.__post_init__(self)
        _identifier(self.snapshot_id, "review snapshot ID")
        _integer(self.revision, "review snapshot revision", 1)
        for name, kind in self._objects.items():
            _instance(getattr(self, name), kind)
        _digest(self.mix_settings_sha256, "review mix settings digest")
        if (self.delivered.media_type not in _VIDEO or self.mix.media_type != "audio/wav"
                or self.mix_status not in ("native_measured_pcm", "synthetic_fixture_only")
                or type(self.clips) is not tuple or len(self.clips) != 2
                or any(type(item) is not LanguageClip for item in self.clips)
                or tuple(item.language for item in self.clips) != ("en", "zh")):
            raise ReviewError("review requires one ordered English and Mandarin clip")
        english, chinese = self.clips
        if ((english.character, english.voice, english.reference)
                != (chinese.character, chinese.voice, chinese.reference)
                or english.utterance.input_id == chinese.utterance.input_id
                or english.speech.pin.input_id == chinese.speech.pin.input_id
                or english.video.pin.input_id == chinese.video.pin.input_id
                or self.dependencies.characters != (english.character,)
                or tuple(sorted((item.speech.pin for item in self.clips),
                                key=lambda pin: pin.input_id)) != self.dependencies.speech):
            raise ReviewError("review language or canonical identity dependencies mismatch")

    def pin(self) -> RevisionPin:
        return RevisionPin(self.snapshot_id, self.revision, self.digest())


@dataclass(frozen=True, slots=True)
class ReviewCheck(_ReviewContract):
    criterion: str
    status: str

    def __post_init__(self) -> None:
        _ReviewContract.__post_init__(self)
        if (type(self.criterion) is not str
                or self.criterion not in HUMAN_CHECKLIST + TECHNICAL_CHECKLIST
                or type(self.status) is not str or self.status not in _STATUSES):
            raise ReviewError("unsupported review criterion or decision")


def _checklist(checks: tuple[ReviewCheck, ...], required: tuple[str, ...]) -> None:
    if (type(checks) is not tuple or any(type(item) is not ReviewCheck for item in checks)
            or tuple(item.criterion for item in checks) != required):
        raise ReviewError("review checklist must be complete unique and ordered")


@dataclass(frozen=True, slots=True)
class OwnerReview(_ReviewContract):
    """Separate operator-supplied human assertion; fields do not authenticate a person."""

    _objects: ClassVar[dict[str, type[Contract]]] = {"snapshot": RevisionPin,
                                                   "evidence": EvidenceRef}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"checks": ReviewCheck}
    review_id: str
    revision: int
    snapshot: RevisionPin
    reviewer: str
    reviewed_at_ns: int
    evidence: EvidenceRef = field(repr=False)
    checks: tuple[ReviewCheck, ...]
    decision: str
    method: str = "human_listening_and_visual"

    def __post_init__(self) -> None:
        _ReviewContract.__post_init__(self)
        _identifier(self.review_id, "owner review ID")
        _integer(self.revision, "owner review revision", 1)
        _instance(self.snapshot, RevisionPin)
        _identifier(self.reviewer, "owner reviewer ID")
        _integer(self.reviewed_at_ns, "owner review timestamp", 1)
        _instance(self.evidence, EvidenceRef)
        _checklist(self.checks, HUMAN_CHECKLIST)
        if (type(self.decision) is not str
                or self.decision not in ("approved", "changes_requested", "pending")
                or self.method != "human_listening_and_visual"):
            raise ReviewError("unsupported human review decision or method")

    def pin(self) -> RevisionPin:
        return RevisionPin(self.review_id, self.revision, self.digest())


@dataclass(frozen=True, slots=True)
class TechnicalReview:
    """Trusted verifier response, never decoded from creative metadata by this module."""

    snapshot: RevisionPin
    checks: tuple[ReviewCheck, ...]
    evidence: EvidenceRef = field(repr=False)

    def __post_init__(self) -> None:
        _instance(self.snapshot, RevisionPin)
        _instance(self.evidence, EvidenceRef)
        _checklist(self.checks, TECHNICAL_CHECKLIST)


class TechnicalReviewVerifier(Protocol):
    """Trusted code must verify native receipts/media/metering for this exact snapshot.

    In particular, unverified_injected_runtime is not native proof: verify the
    NativeSpeechRuntime.receipt_for(request, actual_bytes), not a caller flag or JSON.
    """

    def verify(self, snapshot: ReviewSnapshot) -> TechnicalReview: ...


@dataclass(frozen=True, slots=True)
class ReviewAssessment:
    """Current metadata/verified evidence only; never rights, identity or export authority."""

    current: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MediaInput:
    pin: RevisionPin
    media_type: str
    payload: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ClipInput:
    language: str
    speech: SpeechArtifact = field(repr=False)
    video: MediaInput
    recipe: ReviewRecipe


def _asset(item: MediaInput) -> ReviewAsset:
    _instance(item, MediaInput)
    if type(item.payload) is not bytes:
        raise ReviewError("invalid review media bytes")
    asset = ReviewAsset(item.pin, len(item.payload), item.media_type)
    asset.check_bytes(item.payload)
    return asset


def snapshot_for_review(snapshot_id: str, revision: int, *, sheet: CueSheet,
                        mix: MixedArtifact, delivered: MediaInput,
                        clips: tuple[ClipInput, ...]) -> ReviewSnapshot:
    """Hash supplied media; video format/timing/quality and absent source bytes stay unverified.

    This baseline covers exactly one original canonical actor and one EN/ZH line
    each. It never reads a URI, authenticates provenance or performs media analysis.
    """
    failed = False
    answer = None
    try:
        _instance(sheet, CueSheet)
        _instance(mix, MixedArtifact)
        MixedArtifact(mix.payload, mix.report)
        if (type(clips) is not tuple or len(clips) != 2
                or any(type(item) is not ClipInput for item in clips)
                or tuple(item.language for item in clips) != ("en", "zh")
                or len(sheet.script.utterances) != 2):
            raise ReviewError("review requires paired language inputs")
        language_clips = []
        for item in clips:
            _instance(item.speech, SpeechArtifact)
            source = item.speech
            SpeechArtifact(source.pin, source.request, source.payload, source.runtime_kind,
                           source.sample_rate, source.frames)
            request = source.request
            utterance = next((line for line in sheet.script.utterances
                              if line.utterance_id == request.utterance.input_id), None)
            cue = next((cue for cue in sheet.cues
                        if cue.utterance_id == request.utterance.input_id), None)
            if (utterance is None or cue is None or item.language != utterance.language
                    or (request.script, request.sheet, request.utterance, request.character,
                        request.voice, request.text, request.language, request.timebase,
                        request.start_ticks, request.duration_ticks)
                    != (sheet.script.pin(), RevisionPin(sheet.sheet_id, sheet.revision,
                        sheet.digest()), RevisionPin(utterance.utterance_id,
                        utterance.text_revision, utterance.digest()), utterance.character,
                        utterance.voice, utterance.text, {"en": "English", "zh": "Chinese"}[
                            item.language], sheet.timebase, utterance.start_ticks,
                        utterance.duration_ticks)):
                raise ReviewError("speech review lineage does not match cue sheet")
            metadata = dict(request=request.to_dict(), runtime_kind=source.runtime_kind,
                            sample_rate=source.sample_rate, frames=source.frames)
            input_record = next((entry for entry in mix.report.inputs
                                 if entry.cue_id == cue.cue_id), None)
            if (input_record is None or input_record.source != source.pin
                    or input_record.frames != source.frames
                    or input_record.source_kind != source.runtime_kind
                    or input_record.metadata_sha256 != hashlib.sha256(
                        canonical_json(metadata).encode()).hexdigest()):
                raise ReviewError("speech does not match mixed source lineage")
            language_clips.append(LanguageClip(item.language,
                ReviewAsset(source.pin, len(source.payload), "audio/wav"), _asset(item.video),
                RevisionPin(utterance.utterance_id, utterance.text_revision, request.digest()),
                request.utterance, request.character, request.voice, request.reference,
                item.recipe, source.runtime_kind))
        speech_pins = tuple(sorted((item.speech.pin for item in clips),
                                   key=lambda pin: pin.input_id))
        if (sheet.dependencies(speech=speech_pins) != mix.report.dependencies
                or {item.cue_id for item in mix.report.inputs} != {item.cue_id for item in sheet.cues}):
            raise ReviewError("review mix dependencies do not match scene")
        answer = ReviewSnapshot(snapshot_id, revision, _asset(delivered),
            ReviewAsset(mix.pin, len(mix.payload), "audio/wav"),
            RevisionPin("mix-report", mix.pin.revision, mix.report.digest()),
            mix.report.settings.digest(), mix.report.delivery_status,
            mix.report.dependencies, tuple(language_clips))
    except (ContractError, TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        failed = True
    if failed or answer is None:
        raise ReviewError("review snapshot media or dependencies rejected")
    return answer


def assess_review(snapshot: ReviewSnapshot, review: OwnerReview | None, *,
                  current: ReviewSnapshot,
                  verifier: TechnicalReviewVerifier | None = None) -> ReviewAssessment:
    """Recheck separate owner and trusted technical inputs; never execute an export.

    A positive result relies on the caller's authenticated operator channel and
    verifier implementation. Public metadata, names and flags prove neither.
    """
    _instance(snapshot, ReviewSnapshot)
    _instance(current, ReviewSnapshot)
    invalid = False
    try:
        snapshot = ReviewSnapshot.from_dict(snapshot.to_dict())
        current = ReviewSnapshot.from_dict(current.to_dict())
        if review is not None:
            _instance(review, OwnerReview)
            review = OwnerReview.from_dict(review.to_dict())
    except (ContractError, TypeError, UnicodeError, RecursionError, OverflowError, AttributeError):
        invalid = True
    if invalid:
        raise ReviewError("invalid current media review metadata")
    if snapshot != current:
        return ReviewAssessment(False, ("stale_snapshot",))
    if review is None:
        return ReviewAssessment(False, ("missing_owner_review",))
    _instance(review, OwnerReview)
    if review.snapshot != current.pin():
        return ReviewAssessment(False, ("stale_owner_review",))
    if review.decision != "approved" or any(item.status != "pass" for item in review.checks):
        return ReviewAssessment(False, ("incomplete_owner_review",))
    if (current.mix_status != "native_measured_pcm"
            or any(item.speech_origin == "synthetic_fixture" for item in current.clips)):
        return ReviewAssessment(False, ("fixture_media_not_native_proof",))
    if verifier is None:
        return ReviewAssessment(False, ("missing_technical_verifier",))
    proof = None
    cancelled = False
    try:
        candidate = verifier.verify(current)
        if type(candidate) is TechnicalReview:
            proof = TechnicalReview(candidate.snapshot, candidate.checks, candidate.evidence)
    except KeyboardInterrupt:
        cancelled = True
    except BaseException:
        pass
    if cancelled:
        raise KeyboardInterrupt from None
    if type(proof) is not TechnicalReview:
        return ReviewAssessment(False, ("technical_verification_failed",))
    if proof.snapshot != current.pin():
        return ReviewAssessment(False, ("stale_technical_review",))
    if any(item.status != "pass" for item in proof.checks):
        return ReviewAssessment(False, ("incomplete_technical_review",))
    return ReviewAssessment(True, ())


def require_current_review(snapshot: ReviewSnapshot, review: OwnerReview | None, *,
                           current: ReviewSnapshot,
                           verifier: TechnicalReviewVerifier | None = None) -> ReviewAssessment:
    result = assess_review(snapshot, review, current=current, verifier=verifier)
    if not result.current:
        raise ReviewError("current owner and technical media review required")
    return result
