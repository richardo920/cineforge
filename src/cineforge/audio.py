"""Version-one sound planning and dependency identities; no media I/O or authority.

Times are integer ticks on one scene timebase (not samples or floating seconds).
Rights/source pins describe declarations, never verified bytes, consent or execution.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import ClassVar, Self

from cineforge.characters import CharacterPin, CharacterRegistry, RightsBinding
from cineforge.contracts import (
    Contract,
    ContractError,
    ProjectPlan,
    _digest,
    _identifier,
    _integer,
    _number,
    _text,
    _tuple,
    _unique,
)
from cineforge.rights import RightsLedger, Use

_MAX_TICKS = 2**63 - 1
_KINDS = frozenset({"dialogue", "ambience", "foley", "effects", "music"})


class AudioError(ContractError):
    """Invalid sound metadata; diagnostics exclude private supplied content."""


def _instance(value: object, kind: type[object], label: str) -> None:
    if type(value) is not kind:
        raise AudioError(f"invalid {label}")


def _member(value: object, choices: frozenset[str], label: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise AudioError(f"unsupported {label}")


def _private_text(value: object, label: str) -> None:
    try:
        _text(value, label)
    except ContractError:
        raise AudioError(f"invalid {label}") from None


def _interval(start: int, duration: int) -> None:
    _integer(start, "start ticks")
    _integer(duration, "duration ticks", 1)
    if start > _MAX_TICKS - duration:
        raise AudioError("timing end overflows")


class _AudioContract(Contract):
    __slots__ = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise AudioError("unsupported audio schema version")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        try:
            return super().from_dict(data)
        except (ContractError, TypeError, UnicodeError, RecursionError):
            raise AudioError("invalid audio metadata") from None

    @classmethod
    def from_json(cls, raw: str) -> Self:
        try:
            return super().from_json(raw)
        except (ContractError, TypeError, UnicodeError, RecursionError):
            raise AudioError("invalid audio JSON") from None

    def digest(self) -> str:
        """Canonical declaration identity, not an assertion about external bytes."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RevisionPin(_AudioContract):
    input_id: str
    revision: int
    sha256: str

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.input_id, "input ID")
        _integer(self.revision, "input revision", 1)
        _digest(self.sha256, "input digest")


@dataclass(frozen=True, slots=True)
class AudioTimebase(_AudioContract):
    ticks_per_second: int
    duration_ticks: int

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _integer(self.ticks_per_second, "ticks per second", 1)
        _integer(self.duration_ticks, "timeline duration ticks", 1)

    def check_interval(self, start: int, duration: int) -> None:
        _interval(start, duration)
        if start + duration > self.duration_ticks:
            raise AudioError("timing exceeds scene end")

    def ticks_for_seconds(self, seconds: float) -> int:
        """Bridge legacy positive seconds with <=1 float ULP, strictly <0.5 tick.

        Nonaligned durations fail; cue/scene/shot end bounds remain exact integers.
        """
        seconds = _number(seconds, "duration seconds", positive=True)
        exact = Fraction.from_float(seconds) * self.ticks_per_second
        ticks = round(exact)
        error = abs(exact - ticks)
        tolerance = Fraction.from_float(math.ulp(seconds)) * self.ticks_per_second
        if error >= Fraction(1, 2) or error > tolerance:
            raise AudioError("duration does not align with timebase")
        _integer(ticks, "converted duration ticks", 1)
        return ticks


@dataclass(frozen=True, slots=True)
class PronunciationHint(_AudioContract):
    """Exact substring plus descriptive pronunciation, never rewritten dialogue."""

    text: str = field(repr=False)
    pronunciation: str = field(repr=False)

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _private_text(self.text, "pronunciation text")
        _private_text(self.pronunciation, "pronunciation hint")


@dataclass(frozen=True, slots=True)
class VoiceIdentity(_AudioContract):
    """Logical voice subjects use matching IDs and vN revisions; no recording is required."""

    _objects: ClassVar[dict[str, type[Contract]]] = {"permission": RightsBinding}

    voice_id: str
    revision: int
    permission: RightsBinding
    reference_id: str | None = None

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.voice_id, "voice ID")
        _integer(self.revision, "voice revision", 1)
        _instance(self.permission, RightsBinding, "voice permission")
        subject = self.permission.subject
        if (subject.kind != "voice" or subject.subject_id != self.voice_id
                or subject.revision != f"v{self.revision}"):
            raise AudioError("voice permission must bind the exact voice identity and revision")
        if self.reference_id is not None:
            _identifier(self.reference_id, "voice reference ID")


@dataclass(frozen=True, slots=True)
class Utterance(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"character": CharacterPin,
                                                  "voice": VoiceIdentity}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"pronunciation": PronunciationHint}

    utterance_id: str
    text_revision: int
    text: str = field(repr=False)
    language: str
    speaker_id: str
    character: CharacterPin
    voice: VoiceIdentity
    emotion: str = field(repr=False)
    delivery: str = field(repr=False)
    pronunciation: tuple[PronunciationHint, ...] = field(repr=False)
    scene_id: str
    shot_id: str
    start_ticks: int
    duration_ticks: int

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for value in (self.utterance_id, self.speaker_id, self.scene_id, self.shot_id):
            _identifier(value, "utterance reference ID")
        _integer(self.text_revision, "text revision", 1)
        _private_text(self.text, "utterance text")
        # Canonical ISO 639 baseline: zh denotes Mandarin here, not code-switching.
        _member(self.language, frozenset({"en", "zh"}), "dialogue language")
        _instance(self.character, CharacterPin, "speaker character pin")
        _instance(self.voice, VoiceIdentity, "voice identity")
        _private_text(self.emotion, "emotion")
        _private_text(self.delivery, "delivery")
        _tuple(self.pronunciation, PronunciationHint, "pronunciation hints")
        _unique(self.pronunciation, "pronunciation hints")
        if any(hint.text not in self.text for hint in self.pronunciation):
            raise AudioError("pronunciation hint does not reference exact utterance text")
        _interval(self.start_ticks, self.duration_ticks)


@dataclass(frozen=True, slots=True)
class DialogueScript(_AudioContract):
    _sequences: ClassVar[dict[str, type[Contract]]] = {"utterances": Utterance}

    script_id: str
    revision: int
    utterances: tuple[Utterance, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.script_id, "script ID")
        _integer(self.revision, "script revision", 1)
        _tuple(self.utterances, Utterance, "utterances")
        _unique(tuple(item.utterance_id for item in self.utterances), "utterance IDs")
        speakers: dict[str, tuple[CharacterPin, VoiceIdentity]] = {}
        characters: dict[str, tuple[CharacterPin, VoiceIdentity]] = {}
        for item in self.utterances:
            identity = (item.character, item.voice)
            if item.speaker_id in speakers and speakers[item.speaker_id] != identity:
                raise AudioError("ambiguous speaker identity")
            character_id = item.character.character_id
            if character_id in characters and characters[character_id] != identity:
                raise AudioError("ambiguous character voice identity")
            speakers[item.speaker_id] = identity
            characters[character_id] = identity

    def pin(self) -> RevisionPin:
        return RevisionPin(self.script_id, self.revision, self.digest())


@dataclass(frozen=True, slots=True)
class Stem(_AudioContract):
    stem_id: str
    kind: str

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.stem_id, "stem ID")
        _member(self.kind, _KINDS, "stem kind")


@dataclass(frozen=True, slots=True)
class ShotTiming(_AudioContract):
    """Declared scene offset only; trimming/editorial operations belong to M6."""

    shot_id: str
    shot_revision: int
    start_ticks: int
    duration_ticks: int

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.shot_id, "shot ID")
        _integer(self.shot_revision, "shot revision", 1)
        _interval(self.start_ticks, self.duration_ticks)


@dataclass(frozen=True, slots=True)
class Cue(_AudioContract):
    """Non-dialogue cues may use shot_id=None for explicit scene scope."""

    _objects: ClassVar[dict[str, type[Contract]]] = {"source": RightsBinding}

    cue_id: str
    stem_id: str
    kind: str
    scene_id: str
    shot_id: str | None
    start_ticks: int
    duration_ticks: int
    source: RightsBinding
    utterance_id: str | None = None

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for value in (self.cue_id, self.stem_id, self.scene_id):
            _identifier(value, "cue reference ID")
        if self.shot_id is not None:
            _identifier(self.shot_id, "cue shot ID")
        _member(self.kind, _KINDS, "cue kind")
        _interval(self.start_ticks, self.duration_ticks)
        _instance(self.source, RightsBinding, "cue source")
        expected = "voice" if self.kind == "dialogue" else (
            "music" if self.kind == "music" else "asset")
        if self.source.subject.kind != expected:
            raise AudioError("cue source has incompatible subject kind")
        if self.kind == "dialogue":
            _identifier(self.shot_id, "dialogue shot ID")
            _identifier(self.utterance_id, "dialogue utterance ID")
        elif self.utterance_id is not None:
            raise AudioError("only dialogue cues reference utterances")


@dataclass(frozen=True, slots=True)
class CueSheet(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"timebase": AudioTimebase,
                                                  "script": DialogueScript,
                                                  "picture": RevisionPin}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"stems": Stem, "shots": ShotTiming,
                                                   "cues": Cue}

    sheet_id: str
    revision: int
    project_id: str
    scene_id: str
    timebase: AudioTimebase
    script: DialogueScript = field(repr=False)
    picture: RevisionPin
    stems: tuple[Stem, ...]
    shots: tuple[ShotTiming, ...]
    cues: tuple[Cue, ...]

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for value in (self.sheet_id, self.project_id, self.scene_id):
            _identifier(value, "cue sheet ID")
        _integer(self.revision, "cue sheet revision", 1)
        _instance(self.timebase, AudioTimebase, "timebase")
        _instance(self.script, DialogueScript, "dialogue script")
        _instance(self.picture, RevisionPin, "picture pin")
        _tuple(self.stems, Stem, "stems")
        _tuple(self.shots, ShotTiming, "shot timing")
        _tuple(self.cues, Cue, "cues")
        if not self.stems or not self.shots or not self.cues:
            raise AudioError("cue sheet requires stems, shot timing and cues")
        _unique(tuple(item.stem_id for item in self.stems), "stem IDs")
        _unique(tuple(item.shot_id for item in self.shots), "shot IDs")
        _unique(tuple(item.cue_id for item in self.cues), "cue IDs")
        stems = {item.stem_id: item for item in self.stems}
        shots = {item.shot_id: item for item in self.shots}
        utterances = {item.utterance_id: item for item in self.script.utterances}
        for shot in self.shots:
            self.timebase.check_interval(shot.start_ticks, shot.duration_ticks)
        produced: list[str] = []
        ordering = tuple((item.start_ticks, item.cue_id) for item in self.cues)
        if ordering != tuple(sorted(ordering)):
            raise AudioError("cues must be ordered by start ticks then cue ID")
        timed_items: tuple[Utterance | Cue, ...] = (*self.script.utterances, *self.cues)
        for item in timed_items:
            self.timebase.check_interval(item.start_ticks, item.duration_ticks)
            if item.scene_id != self.scene_id:
                raise AudioError("unknown scene reference")
            if item.shot_id is None:
                # Utterance and dialogue Cue constructors require a real shot ID.
                continue
            span = shots.get(item.shot_id)
            if span is None:
                raise AudioError("missing or unknown shot reference")
            if (item.start_ticks < span.start_ticks or item.start_ticks + item.duration_ticks
                    > span.start_ticks + span.duration_ticks):
                raise AudioError("cue or utterance exceeds shot bounds")
        for cue in self.cues:
            if cue.stem_id not in stems or stems[cue.stem_id].kind != cue.kind:
                raise AudioError("missing or incompatible stem reference")
            if cue.utterance_id is not None:
                utterance = utterances.get(cue.utterance_id)
                if utterance is None:
                    raise AudioError("unknown utterance reference")
                if ((cue.scene_id, cue.shot_id, cue.start_ticks, cue.duration_ticks, cue.source)
                        != (utterance.scene_id, utterance.shot_id, utterance.start_ticks,
                            utterance.duration_ticks, utterance.voice.permission)):
                    raise AudioError("dialogue cue does not match its utterance")
                produced.append(cue.utterance_id)
        _unique(tuple(produced), "dialogue cue utterances")
        if set(produced) != set(utterances):
            raise AudioError("every utterance requires exactly one dialogue cue")

    def preflight(self, plan: ProjectPlan, registry: CharacterRegistry, ledger: RightsLedger,
                  *, current_picture: RevisionPin, use: Use = Use("generate")) -> AudioPreflight:
        """Resolve current declarations/permissions; never grant execution or verify bytes."""
        _instance(plan, ProjectPlan, "production plan")
        _instance(registry, CharacterRegistry, "character registry")
        _instance(ledger, RightsLedger, "rights ledger")
        _instance(current_picture, RevisionPin, "current picture")
        _instance(use, Use, "requested use")
        reasons: set[str] = set()
        if self.picture != current_picture:
            reasons.add("stale_picture")
        if (plan.project.project_id != self.project_id
                or self.scene_id not in {item.scene_id for item in plan.scenes}):
            reasons.add("unknown_project_or_scene")
        plan_shots = {item.shot_id: item for item in plan.shots}
        for span in self.shots:
            shot = plan_shots.get(span.shot_id)
            if shot is None or shot.scene_id != self.scene_id:
                reasons.add("unknown_shot")
            else:
                try:
                    duration_ticks = self.timebase.ticks_for_seconds(shot.duration_seconds)
                except ContractError:
                    reasons.add("unaligned_shot_duration")
                else:
                    if span.shot_revision != shot.revision or span.duration_ticks != duration_ticks:
                        reasons.add("stale_shot_timing")
        for item in self.script.utterances:
            resolved = registry.resolve(item.character, ledger, use=use)
            reasons.update(resolved.reasons)
            record = registry.lookup(item.character)
            shot = plan_shots.get(item.shot_id)
            if shot is not None and shot.character_revision != item.character.to_key():
                reasons.add("shot_character_mismatch")
            if record is not None:
                if record.voice_reference_id != item.voice.reference_id:
                    reasons.add("missing_or_changed_voice_reference")
                if item.voice.reference_id is not None:
                    refs = {ref.reference_id: ref for ref in record.references}
                    ref = refs.get(item.voice.reference_id)
                    if ref is None or ref.role != "voice":
                        reasons.add("missing_or_changed_voice_reference")
                # registry.resolve checks recording permission; the logical voice profile
                # below is a separate subject and can have a different immutable revision.
            reasons.update(ledger.preflight_use(item.voice.permission.subject,
                                               item.voice.permission.rights, use).reasons)
        for cue in self.cues:
            reasons.update(ledger.preflight_use(cue.source.subject, cue.source.rights, use).reasons)
        return AudioPreflight(not reasons, tuple(sorted(reasons)))

    def dependencies(self, *, speech: tuple[RevisionPin, ...]) -> MediaDependencies:
        """Conservative scene identity: any changed input invalidates every bound result.

        Speech pins come from later adapters; this method never claims generated media.
        """
        characters = tuple(sorted({item.character for item in self.script.utterances},
                                  key=lambda item: item.to_key()))
        return MediaDependencies(self.script.pin(), speech, characters, self.picture,
                                 RevisionPin(self.sheet_id, self.revision, self.digest()))


@dataclass(frozen=True, slots=True)
class AudioPreflight:
    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MediaDependencies(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"script": RevisionPin,
                                                  "picture": RevisionPin,
                                                  "cue_sheet": RevisionPin}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"speech": RevisionPin,
                                                   "characters": CharacterPin}

    script: RevisionPin
    speech: tuple[RevisionPin, ...]
    characters: tuple[CharacterPin, ...]
    picture: RevisionPin
    cue_sheet: RevisionPin

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for item in (self.script, self.picture, self.cue_sheet):
            _instance(item, RevisionPin, "dependency pin")
        _tuple(self.speech, RevisionPin, "speech pins")
        _tuple(self.characters, CharacterPin, "character pins")
        _unique(tuple(item.input_id for item in self.speech), "speech dependency IDs")
        _unique(tuple(item.character_id for item in self.characters), "character dependency IDs")
        if tuple(sorted(self.speech, key=lambda item: item.input_id)) != self.speech:
            raise AudioError("speech dependencies must be ordered by ID")
        if tuple(sorted(self.characters, key=lambda item: item.to_key())) != self.characters:
            raise AudioError("character dependencies must be ordered by pin")


@dataclass(frozen=True, slots=True)
class DependencyBinding(_AudioContract):
    """Lineage for a later result/review; existence is never approval or export authority."""

    _objects: ClassVar[dict[str, type[Contract]]] = {"target": RevisionPin,
                                                  "inputs": MediaDependencies}
    target_kind: str
    target: RevisionPin
    inputs: MediaDependencies

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _member(self.target_kind, frozenset({"audio", "video", "sync_check", "listening_review",
                                           "visual_review"}), "dependency target kind")
        _instance(self.target, RevisionPin, "dependency target")
        _instance(self.inputs, MediaDependencies, "dependency inputs")

    def stale_reasons(self, current: MediaDependencies) -> tuple[str, ...]:
        _instance(current, MediaDependencies, "current dependency inputs")
        return tuple(name for name in ("script", "speech", "characters", "picture", "cue_sheet")
                     if getattr(self.inputs, name) != getattr(current, name))

    def is_current(self, current: MediaDependencies) -> bool:
        return not self.stale_reasons(current)
