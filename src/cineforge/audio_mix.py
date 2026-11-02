"""Bounded PCM soundtrack mixing and measured linear presets; no export authority.

Sources are never resampled, trimmed or looped. Mono duplicates to explicit stereo;
unoccupied timeline/reserved speech tails are silence. Fake meters label fixtures only.
"""
from __future__ import annotations

import hashlib
import math
import struct
import time
from array import array
from dataclasses import dataclass, field
from typing import ClassVar

from cineforge.audio import CueSheet, MediaDependencies, RevisionPin, _AudioContract
from cineforge.audio_analysis import (
    AnalysisSettings,
    AudioAnalysisReport,
    DeclaredTiming,
    FFmpegTool,
    MeasurementTool,
    ToolIdentity,
    analyze_wav,
)
from cineforge.audio_pcm import PCM16Wave, PCMError, inspect_pcm16_wav
from cineforge.characters import CharacterRegistry, RightsBinding
from cineforge.contracts import Contract, ContractError, ProjectPlan, _digest, _identifier, _integer
from cineforge.json_codec import canonical_json
from cineforge.rights import RightsLedger, Use
from cineforge.speech import SpeechArtifact
from cineforge.stem_import import ImportedStemArtifact, import_original_stem

_MAX_BYTES = 16 * 1024 * 1024


class MixError(ContractError):
    """Fixed diagnostics, excluding tool, path, private text and media contents."""


def _instance(value: object, kind: type[object]) -> None:
    if type(value) is not kind:
        raise MixError("invalid soundtrack mix record")


def _finite(value: object, low: float, high: float) -> float:
    if type(value) not in (int, float):
        raise MixError("invalid finite mix setting")
    number = math.nan
    try:
        number = float(value)  # type: ignore[arg-type]
    except OverflowError:
        pass
    if not math.isfinite(number) or not low <= number <= high:
        raise MixError("invalid finite mix setting")
    return number


@dataclass(frozen=True, slots=True)
class StemMix(_AudioContract):
    stem_id: str
    gain_db: float = 0.0
    fade_in_ticks: int = 0
    fade_out_ticks: int = 0

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.stem_id, "mix stem ID")
        object.__setattr__(self, "gain_db", _finite(self.gain_db, -96, 24))
        _integer(self.fade_in_ticks, "fade in ticks")
        _integer(self.fade_out_ticks, "fade out ticks")


@dataclass(frozen=True, slots=True)
class MixSettings(_AudioContract):
    _sequences: ClassVar[dict[str, type[Contract]]] = {"stems": StemMix}
    stems: tuple[StemMix, ...]
    sample_rate: int = 24000
    channels: int = 2
    duck_db: float = -12.0
    duck_threshold_dbfs: float = -45.0
    attack_ticks: int = 50
    hold_ticks: int = 50
    release_ticks: int = 200
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        if (type(self.stems) is not tuple or not 1 <= len(self.stems) <= 16
                or any(type(item) is not StemMix for item in self.stems)
                or len({item.stem_id for item in self.stems}) != len(self.stems)
                or self.stems != tuple(sorted(self.stems, key=lambda item: item.stem_id))):
            raise MixError("mix stem controls must be unique and ordered")
        if type(self.sample_rate) is not int or self.sample_rate not in (16000, 24000, 44100, 48000):
            raise MixError("unsupported mix sample rate")
        _integer(self.channels, "mix channels", 1, 2)
        for name, low, high in (("duck_db", -60, 0), ("duck_threshold_dbfs", -96, -1),
                               ("timeout_seconds", .1, 30)):
            object.__setattr__(self, name, _finite(getattr(self, name), low, high))
        for name in ("attack_ticks", "hold_ticks", "release_ticks"):
            _integer(getattr(self, name), "duck timing ticks")


@dataclass(frozen=True, slots=True)
class DeliveryPreset(_AudioContract):
    preset_id: str
    target_lufs: float
    max_true_peak_dbtp: float
    loudness_tolerance_lu: float = .2
    peak_safety_db: float = .05

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        expected = {"broadcast-linear": (-23.0, -1.0), "internal-review-stereo": (-16.0, -2.0)}
        if (type(self.preset_id) is not str or self.preset_id not in expected
                or (self.target_lufs, self.max_true_peak_dbtp)
                != expected[self.preset_id] or type(self.target_lufs) not in (int, float)
                or type(self.max_true_peak_dbtp) not in (int, float)
                or type(self.loudness_tolerance_lu) not in (int, float)
                or self.loudness_tolerance_lu != .2 or type(self.peak_safety_db) not in (int, float)
                or self.peak_safety_db != .05):
            raise MixError("unsupported PCM delivery preset")


BROADCAST_LINEAR = DeliveryPreset("broadcast-linear", -23.0, -1.0)
INTERNAL_REVIEW_STEREO = DeliveryPreset("internal-review-stereo", -16.0, -2.0)


@dataclass(frozen=True, slots=True)
class MixInput(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"source": RevisionPin}
    cue_id: str
    source: RevisionPin
    metadata_sha256: str
    source_kind: str
    frames: int

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _identifier(self.cue_id, "mix cue ID")
        _instance(self.source, RevisionPin)
        _digest(self.metadata_sha256, "mix source metadata digest")
        _integer(self.frames, "mix source frames", 1, 60 * 48000)
        if self.source_kind not in ("original_import", "synthetic_fixture", "unverified_injected_runtime"):
            raise MixError("unsupported mix source")


@dataclass(frozen=True, slots=True)
class MixReport(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"output": RevisionPin,
        "dependencies": MediaDependencies, "settings": MixSettings, "preset": DeliveryPreset,
        "premix": AudioAnalysisReport, "delivered": AudioAnalysisReport}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"inputs": MixInput}
    output: RevisionPin
    dependencies: MediaDependencies
    settings: MixSettings
    preset: DeliveryPreset
    inputs: tuple[MixInput, ...]
    admission_sha256: str
    safety_gain: float
    preset_gain_db: float
    premix: AudioAnalysisReport
    delivered: AudioAnalysisReport
    delivery_status: str
    method: str = "pcm16-linear-stem-mix-v1"
    human_review: str = "not_assessed"
    export_authority: str = "none"

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for name, kind in self._objects.items():
            _instance(getattr(self, name), kind)
        if (type(self.inputs) is not tuple or not self.inputs
                or any(type(item) is not MixInput for item in self.inputs)
                or tuple(sorted(item.cue_id for item in self.inputs))
                != tuple(item.cue_id for item in self.inputs)
                or len({item.cue_id for item in self.inputs}) != len(self.inputs)):
            raise MixError("invalid mix input lineage")
        _digest(self.admission_sha256, "mix admission digest")
        object.__setattr__(self, "safety_gain", _finite(self.safety_gain, 0, 1))
        object.__setattr__(self, "preset_gain_db", _finite(self.preset_gain_db, -200, 100))
        if (self.output != self.delivered.input or self.premix.tool is None
                or self.premix.tool != self.delivered.tool
                or self.premix.measurement_status != "measured"
                or self.delivered.measurement_status != "measured"
                or self.delivery_status != ("native_measured_pcm" if self.delivered.tool.kind == "native"
                                             else "synthetic_fixture_only")
                or self.method != "pcm16-linear-stem-mix-v1" or self.human_review != "not_assessed"
                or self.export_authority != "none"):
            raise MixError("invalid mix measurement or authority claim")
        loudness, peak = self.delivered.integrated_lufs.value, self.delivered.true_peak_dbtp.value
        if (loudness is None or peak is None
                or (self.delivered.sample_rate, self.delivered.channels)
                != (self.settings.sample_rate, self.settings.channels)
                or self.delivered.settings.timebase != self.premix.settings.timebase
                or self.delivered.timing.picture != self.dependencies.picture
                or self.delivered.frames * self.delivered.settings.timebase.ticks_per_second
                != self.delivered.settings.timebase.duration_ticks * self.settings.sample_rate
                or abs(loudness - self.preset.target_lufs) > self.preset.loudness_tolerance_lu
                or peak > self.preset.max_true_peak_dbtp):
            raise MixError("delivered PCM misses preset limits")


@dataclass(frozen=True, slots=True)
class MixedArtifact:
    payload: bytes = field(repr=False)
    report: MixReport

    def __post_init__(self) -> None:
        _instance(self.report, MixReport)
        wave = None
        try:
            wave = inspect_pcm16_wav(self.payload)
        except PCMError:
            pass
        if wave is None:
            raise MixError("invalid mixed artifact PCM")
        if (hashlib.sha256(self.payload).hexdigest() != self.report.output.sha256
                or (wave.sample_rate, wave.channels, wave.frames)
                != (self.report.delivered.sample_rate, self.report.delivered.channels,
                    self.report.delivered.frames)):
            raise MixError("mixed artifact bytes mismatch")


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _frames(ticks: int, rate: int, timebase: int) -> int:
    frames, remainder = divmod(ticks * rate, timebase)
    if remainder:
        raise MixError("mix timing does not align exactly to sample frames")
    return frames


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining < .05:
        raise MixError("soundtrack mix deadline exceeded")
    return min(30, remaining)


def _admit(sheet: CueSheet, sources: tuple[SpeechArtifact | ImportedStemArtifact, ...],
           settings: MixSettings, plan: ProjectPlan, registry: CharacterRegistry,
           ledger: RightsLedger, picture: RevisionPin, script: RevisionPin) -> tuple[
               tuple[tuple[MixInput, PCM16Wave, int, str], ...], str]:
    for value, record_kind in ((sheet, CueSheet), (settings, MixSettings), (plan, ProjectPlan),
                        (registry, CharacterRegistry), (ledger, RightsLedger),
                        (picture, RevisionPin), (script, RevisionPin)):
        _instance(value, record_kind)
    if (type(sources) is not tuple or not 1 <= len(sources) <= 16
            or len(sources) != len(sheet.cues) or script != sheet.script.pin()
            or not sheet.preflight(plan, registry, ledger, current_picture=picture,
                                   use=Use("copy")).allowed
            or {item.stem_id for item in settings.stems} != {item.stem_id for item in sheet.stems}):
        raise MixError("stale or unadmitted soundtrack sources")
    if sheet.timebase.duration_ticks > sheet.timebase.ticks_per_second * 60:
        raise MixError("soundtrack exceeds supported duration")
    total = 0
    for source in sources:
        if type(source) not in (SpeechArtifact, ImportedStemArtifact):
            raise MixError("unsupported soundtrack source record")
        total += len(source.payload)
    if total > 32 * 1024 * 1024:
        raise MixError("soundtrack input byte bound exceeded")
    pin = RevisionPin(sheet.sheet_id, sheet.revision, sheet.digest())
    cues = {item.cue_id: item for item in sheet.cues}
    admitted = []
    bindings: set[RightsBinding] = {item.source for item in sheet.cues}
    for binding in bindings:
        if not ledger.preflight_use(binding.subject, binding.rights, Use("modify")).allowed:
            raise MixError("soundtrack modification rights failed preflight")
    for item in sheet.script.utterances:
        actor = registry.lookup(item.character)
        if actor is None:
            raise MixError("soundtrack character unavailable")
        bindings.update(ref.permission for ref in actor.references)
    for source in sources:
        wave = inspect_pcm16_wav(source.payload)
        if isinstance(source, ImportedStemArtifact):
            fresh = import_original_stem(sheet, source.cue.cue_id, source.payload, source.pin,
                ledger=ledger, original_source_review=source.original_source_review)
            if fresh != source:
                raise MixError("stale imported soundtrack source")
            cue = source.cue
            metadata = dict(sheet=pin.to_dict(), cue=cue.to_dict(), source=source.pin.to_dict(),
                permission=source.permission.to_dict(), review=source.original_source_review.to_dict(),
                sample_rate=source.sample_rate, channels=source.channels, frames=source.frames)
            source_kind = "original_import"
        else:
            request = source.request
            candidate = next((item for item in sheet.cues if item.utterance_id
                              == request.utterance.input_id), None)
            utterance = next((item for item in sheet.script.utterances
                              if item.utterance_id == request.utterance.input_id), None)
            if (candidate is None or utterance is None
                    or request.sheet != pin or request.script != script
                    or request.utterance != RevisionPin(utterance.utterance_id,
                        utterance.text_revision, utterance.digest())
                    or (request.character, request.voice, request.text, request.language,
                        request.start_ticks, request.duration_ticks, request.timebase)
                    != (utterance.character, utterance.voice, utterance.text,
                        {"en": "English", "zh": "Chinese"}[utterance.language],
                        candidate.start_ticks, candidate.duration_ticks, sheet.timebase)):
                raise MixError("stale speech soundtrack source")
            cue = candidate
            actor = registry.lookup(utterance.character)
            recording = None if actor is None else next((ref for ref in actor.references
                if ref.reference_id == request.reference.input_id), None)
            if (recording is None or recording.permission != request.recording_permission
                    or recording.artifact.sha256 != request.reference.sha256
                    or utterance.voice.reference_id != request.reference.input_id):
                raise MixError("stale speech recording lineage")
            bindings.add(request.recording_permission)
            for action in ("copy", "reference"):
                if not ledger.preflight_use(request.recording_permission.subject,
                        request.recording_permission.rights, Use(action)).allowed:
                    raise MixError("speech recording rights failed preflight")
            # Reconstruct to revalidate metadata even for trusted in-memory handles.
            SpeechArtifact(source.pin, request, source.payload, source.runtime_kind,
                           source.sample_rate, source.frames)
            metadata = dict(request=request.to_dict(), runtime_kind=source.runtime_kind,
                            sample_rate=source.sample_rate, frames=source.frames)
            source_kind = source.runtime_kind
        if (cue.cue_id not in cues or wave.sample_rate != settings.sample_rate
                or wave.channels > settings.channels
                or hashlib.sha256(source.payload).hexdigest() != source.pin.sha256):
            raise MixError("unsupported mix conversion or source bytes")
        offset = _frames(cue.start_ticks, wave.sample_rate, sheet.timebase.ticks_per_second)
        duration = _frames(cue.duration_ticks, wave.sample_rate, sheet.timebase.ticks_per_second)
        control = next(item for item in settings.stems if item.stem_id == cue.stem_id)
        fade_in = _frames(control.fade_in_ticks, wave.sample_rate, sheet.timebase.ticks_per_second)
        fade_out = _frames(control.fade_out_ticks, wave.sample_rate, sheet.timebase.ticks_per_second)
        if wave.frames > duration or fade_in + fade_out > wave.frames - 1:
            raise MixError("source or fades exceed supported cue duration")
        admitted.append((MixInput(cue.cue_id, source.pin, _hash(metadata), source_kind, wave.frames),
                         wave, offset, cue.stem_id))
    if {item[0].cue_id for item in admitted} != set(cues):
        raise MixError("missing or duplicate soundtrack cue source")
    records = [ledger.lookup(binding.rights) for binding in
               sorted(bindings, key=lambda item: item.to_json())]
    admission = _hash(dict(plan=plan.to_dict(), sheet=sheet.to_dict(),
        rights=[record.to_dict() if record is not None else None for record in records]))
    return tuple(sorted(admitted, key=lambda item: item[0].cue_id)), admission


def _wav(samples: array[float], rate: int, channels: int, gain: float, deadline: float) -> bytes:
    pcm = bytearray(len(samples) * 2)
    for index, value in enumerate(samples):
        if index % 4096 == 0:
            _remaining(deadline)
        number = round(value * gain)
        if not -32767 <= number <= 32767:
            raise MixError("mixed PCM would clip")
        struct.pack_into("<h", pcm, index * 2, number)
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * 2, channels * 2, 16)
        + b"data" + struct.pack("<I", len(pcm)) + pcm)


def _render(admitted: tuple[tuple[MixInput, PCM16Wave, int, str], ...], sheet: CueSheet,
            settings: MixSettings, deadline: float) -> tuple[array[float], float]:
    frames = _frames(sheet.timebase.duration_ticks, settings.sample_rate,
                     sheet.timebase.ticks_per_second)
    if not 1 <= frames <= settings.sample_rate * 60 or frames * settings.channels * 2 + 44 > _MAX_BYTES:
        raise MixError("soundtrack output bounds exceeded")
    foreground = array("d", [0.0]) * (frames * settings.channels)
    background = array("d", [0.0]) * len(foreground)
    detector = array("d", [0.0]) * frames
    kinds = {item.stem_id: item.kind for item in sheet.stems}
    for _, wave, offset, stem_id in admitted:
        control = next(item for item in settings.stems if item.stem_id == stem_id)
        gain = 10 ** (control.gain_db / 20)
        fade_in = _frames(control.fade_in_ticks, wave.sample_rate, sheet.timebase.ticks_per_second)
        fade_out = _frames(control.fade_out_ticks, wave.sample_rate, sheet.timebase.ticks_per_second)
        target = background if kinds[stem_id] in ("ambience", "music") else foreground
        for frame in range(wave.frames):
            if frame % 4096 == 0:
                _remaining(deadline)
            fade = min(1.0, frame / fade_in if fade_in else 1.0,
                       (wave.frames - 1 - frame) / fade_out if fade_out else 1.0)
            for channel in range(settings.channels):
                source_channel = min(channel, wave.channels - 1)
                value = struct.unpack_from("<h", wave.pcm, (frame * wave.channels + source_channel) * 2)[0]
                sample = value * gain * fade
                index = (offset + frame) * settings.channels + channel
                target[index] += sample
                if kinds[stem_id] == "dialogue":
                    detector[offset + frame] = max(detector[offset + frame], abs(sample))
    attack, hold, release = (_frames(value, settings.sample_rate, sheet.timebase.ticks_per_second)
                            for value in (settings.attack_ticks, settings.hold_ticks,
                                          settings.release_ticks))
    floor = 10 ** (settings.duck_db / 20)
    threshold = 32768 * 10 ** (settings.duck_threshold_dbfs / 20)
    envelope, last_active, peak = 1.0, -hold - 1, 0.0
    for frame in range(frames):
        if frame % 4096 == 0:
            _remaining(deadline)
        if detector[frame] >= threshold:
            last_active = frame
        active = frame - last_active <= hold
        if active:
            envelope = max(floor, envelope - (1 - floor) / attack) if attack else floor
        else:
            envelope = min(1.0, envelope + (1 - floor) / release) if release else 1.0
        for channel in range(settings.channels):
            index = frame * settings.channels + channel
            foreground[index] += background[index] * envelope
            peak = max(peak, abs(foreground[index]))
    return foreground, min(1.0, 32700 / peak) if peak else 1.0


def mix_scene(sheet: CueSheet, sources: tuple[SpeechArtifact | ImportedStemArtifact, ...],
              settings: MixSettings, preset: DeliveryPreset, *, plan: ProjectPlan,
              registry: CharacterRegistry, ledger: RightsLedger, current_picture: RevisionPin,
              current_script: RevisionPin, output_id: str, output_revision: int,
              tool: MeasurementTool | None = None,
              expected_tool: ToolIdentity | None = None) -> MixedArtifact:
    """Full-scene PCM16 mix, with explicit safety attenuation and linear preset gain.

    Ducking uses per-frame dialogue peaks, linear-gain attack/hold/release and affects
    ambience/music only. Fades reach zero at first/last frame; zero durations bypass.
    Target gain refuses excessive crest factors; it never inserts a limiter/compressor.
    Native results attest measured PCM limits only, never speech, human or codec review.
    """
    message = "soundtrack mix rejected"
    cancelled = False
    try:
        _instance(settings, MixSettings)
        _instance(preset, DeliveryPreset)
        RevisionPin(output_id, output_revision, "0" * 64)
        deadline = time.monotonic() + settings.timeout_seconds
        admitted, admission = _admit(sheet, sources, settings, plan, registry, ledger,
                                      current_picture, current_script)
        # All static bounds/alignment/rights checks precede native identification.
        for ticks in (sheet.timebase.duration_ticks, settings.attack_ticks,
                      settings.hold_ticks, settings.release_ticks):
            _frames(ticks, settings.sample_rate, sheet.timebase.ticks_per_second)
        if sheet.timebase.duration_ticks * 5 < sheet.timebase.ticks_per_second * 2:
            raise MixError("soundtrack shorter than supported meter window")
        speech = tuple(sorted((source.pin for source in sources if type(source) is SpeechArtifact),
                              key=lambda item: item.input_id))
        dependencies = sheet.dependencies(speech=speech)
        if preset.preset_id == "internal-review-stereo" and settings.channels != 2:
            raise MixError("internal review preset requires stereo delivery")
        if tool is None or type(expected_tool) is not ToolIdentity:
            raise MixError("mix measurement tooling unavailable")
        if expected_tool.kind == "native" and (type(tool) is not FFmpegTool
                                               or expected_tool.version != "9.0.2"):
            raise MixError("unsupported native mix meter")
        samples, safety = _render(admitted, sheet, settings, deadline)
        if not any(samples):
            raise MixError("silent soundtrack has undefined loudness")
        payload = _wav(samples, settings.sample_rate, settings.channels, safety, deadline)
        timing = DeclaredTiming(current_picture, 0, sheet.timebase.duration_ticks)
        window = sheet.timebase.ticks_per_second
        analysis = AnalysisSettings(sheet.timebase, window, timeout_seconds=_remaining(deadline))
        before = analyze_wav(payload, RevisionPin(output_id, output_revision,
                            hashlib.sha256(payload).hexdigest()), analysis, timing, tool=tool)
        if "tool_cleanup_uncertain" in before.issues:
            raise MixError("mix audio process cleanup unconfirmed")
        if before.tool != expected_tool or before.measurement_status != "measured":
            raise MixError("mix loudness or true peak unavailable or undefined")
        loudness, peak = before.integrated_lufs.value, before.true_peak_dbtp.value
        assert loudness is not None and peak is not None
        gain_db = preset.target_lufs - loudness
        if peak + gain_db > preset.max_true_peak_dbtp - preset.peak_safety_db:
            raise MixError("preset target unattainable by supported linear gain")
        payload = _wav(samples, settings.sample_rate, settings.channels,
                       safety * 10 ** (gain_db / 20), deadline)
        pin = RevisionPin(output_id, output_revision, hashlib.sha256(payload).hexdigest())
        analysis = AnalysisSettings(sheet.timebase, window, timeout_seconds=_remaining(deadline))
        after = analyze_wav(payload, pin, analysis, timing, tool=tool)
        if "tool_cleanup_uncertain" in after.issues:
            raise MixError("mix audio process cleanup unconfirmed")
        fresh, fresh_admission = _admit(sheet, sources, settings, plan, registry, ledger,
                                        current_picture, current_script)
        if fresh != admitted or fresh_admission != admission:
            raise MixError("soundtrack permissions or sources changed during mixing")
        _remaining(deadline)
        report = MixReport(pin, dependencies, settings, preset,
            tuple(item[0] for item in admitted), admission, safety, gain_db, before, after,
            "native_measured_pcm" if expected_tool.kind == "native" else "synthetic_fixture_only")
        return MixedArtifact(payload, report)
    except MixError as error:
        message = str(error)
    except KeyboardInterrupt:
        cancelled = True
    except BaseException:
        pass
    if cancelled:
        raise KeyboardInterrupt from None
    raise MixError(message)
