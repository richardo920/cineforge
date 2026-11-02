"""Pinned Qwen Base argument translation and fixture runtime; no native SDK or I/O.

Reviewed rights/provenance are operator declarations, not proof of synthetic origin
or permission to install/run a model. Executable runtimes come only from trusted code.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from cineforge.audio import (
    AudioTimebase,
    CueSheet,
    RevisionPin,
    Utterance,
    VoiceIdentity,
    _AudioContract,
)
from cineforge.audio_pcm import PCM16Wave, PCMError, inspect_pcm16_wav
from cineforge.characters import CharacterPin, CharacterRegistry, RightsBinding
from cineforge.contracts import Contract, ContractError, ProjectPlan, _identifier, _integer, _text
from cineforge.rights import RightsLedger, Use

SDK_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"  # pragma: allowlist secret - public source revision
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BASE_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"  # pragma: allowlist secret - public model revision
_MAX_BYTES = 16 * 1024 * 1024


class SpeechError(ContractError):
    """Static diagnostics without private dialogue, audio, paths or runtime output."""


def _instance(value: object, kind: type[object]) -> None:
    if type(value) is not kind:
        raise SpeechError("invalid speech record")


def _private_text(value: object) -> None:
    failed = False
    try:
        _text(value, "speech text")
    except ContractError:
        failed = True
    if failed:
        raise SpeechError("invalid private speech text")


def _pcm(payload: bytes, *, mono: bool = False) -> PCM16Wave:
    failed = False
    try:
        wave = inspect_pcm16_wav(payload)
    except PCMError:
        failed = True
    if failed:
        raise SpeechError("unsupported or malformed speech PCM WAV")
    if mono and wave.channels != 1:
        raise SpeechError("speech requires mono PCM WAV")
    return wave


@dataclass(frozen=True, slots=True)
class CanonicalVoiceBinding(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {
        "character": CharacterPin, "voice": VoiceIdentity,
    }
    character: CharacterPin
    voice: VoiceIdentity

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _instance(self.character, CharacterPin)
        _instance(self.voice, VoiceIdentity)
        if self.voice.reference_id is None:
            raise SpeechError("canonical voice requires a reference")


@dataclass(frozen=True, slots=True)
class SyntheticReferenceInput(_AudioContract):
    """Operator-selected original synthetic recording; evidence is not machine proof."""

    _objects: ClassVar[dict[str, type[Contract]]] = {
        "source": RevisionPin, "permission": RightsBinding,
    }
    source: RevisionPin
    permission: RightsBinding
    transcript: str = field(repr=False)

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _instance(self.source, RevisionPin)
        _instance(self.permission, RightsBinding)
        if (self.permission.subject.kind != "voice"
                or self.permission.subject.sha256 != self.source.sha256):
            raise SpeechError("reference permission does not match recording")
        _private_text(self.transcript)
        if len(self.transcript) > 4096:
            raise SpeechError("reference transcript exceeds adapter limit")


@dataclass(frozen=True, slots=True)
class QwenBaseRequest(_AudioContract):
    """Serializable call data only; decoding this record never chooses executable code.

    Native follow-up maps reference PCM to ref_audio=(waveform, sample_rate), builds
    create_voice_clone_prompt(ref_text=ref_text, x_vector_only_mode=False), then calls
    generate_voice_clone(text=text, language=language, voice_clone_prompt=items,
    non_streaming_mode=True, max_new_tokens=512). No SDK/float conversion ships here.
    """

    _objects: ClassVar[dict[str, type[Contract]]] = {
        "script": RevisionPin, "sheet": RevisionPin, "utterance": RevisionPin,
        "character": CharacterPin, "voice": VoiceIdentity, "reference": RevisionPin,
        "recording_permission": RightsBinding, "timebase": AudioTimebase,
    }
    script: RevisionPin
    sheet: RevisionPin
    utterance: RevisionPin
    character: CharacterPin
    voice: VoiceIdentity
    reference: RevisionPin
    recording_permission: RightsBinding
    timebase: AudioTimebase
    start_ticks: int
    duration_ticks: int
    text: str = field(repr=False)
    language: str
    ref_text: str = field(repr=False)
    reference_sample_rate: int
    reference_frames: int
    sdk_revision: str = SDK_REVISION
    model_id: str = BASE_MODEL
    model_revision: str = BASE_REVISION
    x_vector_only_mode: bool = False
    non_streaming_mode: bool = True
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for name, kind in self._objects.items():
            _instance(getattr(self, name), kind)
        if (self.sdk_revision != SDK_REVISION or self.model_id != BASE_MODEL
                or self.model_revision != BASE_REVISION
                or self.x_vector_only_mode is not False or self.non_streaming_mode is not True
                or type(self.max_new_tokens) is not int or self.max_new_tokens != 512):
            raise SpeechError("unsupported Qwen Base translation target or controls")
        if self.language not in ("English", "Chinese"):
            raise SpeechError("unsupported speech language")
        _private_text(self.text)
        _private_text(self.ref_text)
        if len(self.text) > 4096 or len(self.ref_text) > 4096:
            raise SpeechError("speech text exceeds adapter limit")
        _integer(self.reference_frames, "reference frames", 1, 60 * 48000)
        if (type(self.reference_sample_rate) is not int
                or self.reference_sample_rate not in (16000, 24000, 44100, 48000)
                or self.reference_frames > 60 * self.reference_sample_rate):
            raise SpeechError("invalid reference PCM metadata")
        self.timebase.check_interval(self.start_ticks, self.duration_ticks)
        if (self.voice.reference_id != self.reference.input_id
                or self.recording_permission.subject.kind != "voice"
                or self.recording_permission.subject.sha256 != self.reference.sha256):
            raise SpeechError("speech reference identity mismatch")


class SpeechRuntime(Protocol):
    """Only trusted Python code injects an implementation; this PR ships no native one."""

    def synthesize(self, request: QwenBaseRequest, reference: PCM16Wave) -> bytes: ...


class FakeSpeechRuntime:
    """Deterministic silence fixture; does not synthesize words or evaluate quality."""

    def __init__(self) -> None:
        self.calls: list[QwenBaseRequest] = []

    def synthesize(self, request: QwenBaseRequest, reference: PCM16Wave) -> bytes:
        _instance(request, QwenBaseRequest)
        _instance(reference, PCM16Wave)
        self.calls.append(request)
        frames = min(480, request.duration_ticks * 24000 // request.timebase.ticks_per_second)
        if frames < 1:
            raise SpeechError("fixture cannot fit reserved speech duration")
        pcm = b"\x00\x00" * frames
        return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
                + b"data" + struct.pack("<I", len(pcm)) + pcm)


@dataclass(frozen=True, slots=True)
class SpeechArtifact:
    pin: RevisionPin
    request: QwenBaseRequest = field(repr=False)
    payload: bytes = field(repr=False)
    runtime_kind: str
    sample_rate: int
    frames: int

    def __post_init__(self) -> None:
        _instance(self.pin, RevisionPin)
        _instance(self.request, QwenBaseRequest)
        wave = _pcm(self.payload, mono=True)
        if (hashlib.sha256(self.payload).hexdigest() != self.pin.sha256
                or type(self.sample_rate) is not int or self.sample_rate != wave.sample_rate
                or type(self.frames) is not int or self.frames != wave.frames
                or self.runtime_kind not in ("synthetic_fixture", "unverified_injected_runtime")):
            raise SpeechError("speech artifact metadata mismatch")
        if (wave.frames * self.request.timebase.ticks_per_second
                > self.request.duration_ticks * wave.sample_rate):
            raise SpeechError("speech exceeds reserved duration")


class QwenBaseSpeechAdapter:
    """Exact Base requests, fresh declarations and verified in-memory results only.

    Current script/picture pins must be independently supplied by the caller. The
    adapter cannot authenticate that caller or prove synthetic-reference origin.
    """

    def __init__(self, voices: tuple[CanonicalVoiceBinding, ...], *,
                 runtime: SpeechRuntime | None = None) -> None:
        if (type(voices) is not tuple or not voices
                or any(type(item) is not CanonicalVoiceBinding for item in voices)):
            raise SpeechError("canonical voice bindings required")
        if len({item.character.character_id for item in voices}) != len(voices):
            raise SpeechError("ambiguous canonical character voice")
        self._voices = voices
        self._runtime: SpeechRuntime = FakeSpeechRuntime() if runtime is None else runtime

    def translate(self, sheet: CueSheet, utterance_id: str, reference: SyntheticReferenceInput,
                  reference_payload: bytes, *, plan: ProjectPlan, registry: CharacterRegistry,
                  ledger: RightsLedger, current_picture: RevisionPin,
                  current_script: RevisionPin) -> QwenBaseRequest:
        for value, kind in ((sheet, CueSheet), (reference, SyntheticReferenceInput),
                            (plan, ProjectPlan), (registry, CharacterRegistry),
                            (ledger, RightsLedger), (current_picture, RevisionPin),
                            (current_script, RevisionPin)):
            _instance(value, kind)
        _identifier(utterance_id, "utterance ID")
        if sheet.script.pin() != current_script:
            raise SpeechError("stale dialogue script")
        utterance = next((item for item in sheet.script.utterances
                          if item.utterance_id == utterance_id), None)
        if utterance is None:
            raise SpeechError("missing utterance")
        _instance(utterance, Utterance)
        if utterance.language not in ("en", "zh"):
            raise SpeechError("unsupported speech language")
        if (utterance.emotion != "neutral" or utterance.delivery != "default"
                or utterance.pronunciation):
            raise SpeechError("unsupported Base delivery or pronunciation controls")
        binding = next((item for item in self._voices
                        if item.character == utterance.character), None)
        if binding is None or binding.voice != utterance.voice:
            raise SpeechError("missing or changed canonical character voice")
        if not sheet.preflight(plan, registry, ledger, current_picture=current_picture).allowed:
            raise SpeechError("speech declarations or rights failed preflight")
        actor = registry.lookup(utterance.character)
        if actor is None or actor.identity_kind != "fictional":
            raise SpeechError("speech baseline requires a reviewed fictional character")
        recording = next((item for item in actor.references
                          if item.reference_id == utterance.voice.reference_id), None)
        if (recording is None or recording.role != "voice" or recording.media_type != "audio/wav"
                or actor.voice_reference_id != recording.reference_id
                or reference.source.input_id != recording.reference_id
                or reference.permission != recording.permission
                or reference.source.sha256 != recording.artifact.sha256
                or not isinstance(reference_payload, bytes)
                or len(reference_payload) > _MAX_BYTES
                or len(reference_payload) != recording.size_bytes
                or hashlib.sha256(reference_payload).hexdigest() != reference.source.sha256):
            raise SpeechError("missing or changed voice recording")
        for action in ("reference", "condition"):
            if not ledger.preflight_use(recording.permission.subject,
                                        recording.permission.rights, Use(action)).allowed:
                raise SpeechError("recording rights failed preflight")
        wave = _pcm(reference_payload, mono=True)
        return QwenBaseRequest(
            current_script, RevisionPin(sheet.sheet_id, sheet.revision, sheet.digest()),
            RevisionPin(utterance.utterance_id, utterance.text_revision, utterance.digest()),
            utterance.character, utterance.voice, reference.source, recording.permission,
            sheet.timebase, utterance.start_ticks, utterance.duration_ticks, utterance.text,
            {"en": "English", "zh": "Chinese"}[utterance.language], reference.transcript,
            wave.sample_rate, wave.frames,
        )

    def synthesize(self, sheet: CueSheet, utterance_id: str, reference: SyntheticReferenceInput,
                   reference_payload: bytes, *, plan: ProjectPlan, registry: CharacterRegistry,
                   ledger: RightsLedger, current_picture: RevisionPin, current_script: RevisionPin,
                   output_id: str, output_revision: int) -> SpeechArtifact:
        # Validate result IDs before any runtime call, too.
        RevisionPin(output_id, output_revision, "0" * 64)
        request = self.translate(sheet, utterance_id, reference, reference_payload, plan=plan,
                                 registry=registry, ledger=ledger, current_picture=current_picture,
                                 current_script=current_script)
        wave = _pcm(reference_payload, mono=True)
        failed = cancelled = False
        try:
            payload = self._runtime.synthesize(request, wave)
        except KeyboardInterrupt:
            cancelled = True
        except BaseException:
            failed = True
        if cancelled:
            raise KeyboardInterrupt from None
        if failed:
            raise SpeechError("speech runtime failed")
        fresh = self.translate(sheet, utterance_id, reference, reference_payload, plan=plan,
                               registry=registry, ledger=ledger, current_picture=current_picture,
                               current_script=current_script)
        if fresh != request:
            raise SpeechError("speech inputs changed during runtime")
        output = _pcm(payload, mono=True)
        kind = ("synthetic_fixture" if type(self._runtime) is FakeSpeechRuntime
                else "unverified_injected_runtime")
        return SpeechArtifact(RevisionPin(output_id, output_revision,
                                          hashlib.sha256(payload).hexdigest()),
                              request, payload, kind, output.sample_rate, output.frames)
