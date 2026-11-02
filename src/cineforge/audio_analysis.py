"""Objective PCM/FFmpeg measurements; no synthesis, mixing, ASR or review authority.

Supported input: RIFF PCM16 little-endian WAV, mono/stereo, 16/24/44.1/48 kHz,
fmt16 + data chunks only. Energy windows use channel-averaged mean square power;
LUFS and oversampled dBTP come exclusively from FFmpeg loudnorm input statistics.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import struct
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, ClassVar, Protocol

from cineforge.audio import AudioTimebase, RevisionPin, _AudioContract
from cineforge.audio_pcm import PCMError, inspect_pcm16_wav
from cineforge.audio_process import AudioProcessError, run_audio_process
from cineforge.contracts import Contract, ContractError, _digest, _integer, _tuple, _unique
from cineforge.json_codec import strict_loads

_MAX_BYTES = 16 * 1024 * 1024
_FILTER = "loudnorm=I=-24:LRA=7:TP=-2:print_format=json"
_LOUDNORM_KEYS = frozenset({"input_i", "input_tp", "input_lra", "input_thresh", "output_i",
    "output_tp", "output_lra", "output_thresh", "normalization_type", "target_offset"})
_ISSUES = frozenset({"clipping_risk", "true_peak_overload", "silent_audio", "duration_mismatch",
    "audio_exceeds_picture", "missing_expected_energy", "energy_outside_expected_windows",
    "integrated_loudness_undefined", "true_peak_undefined", "measurement_unavailable",
    "tool_cleanup_uncertain"})


class AudioAnalysisError(ContractError):
    """Only fixed local errors; no supplied audio, path or native output diagnostics."""


def _instance(value: object, kind: type[object]) -> None:
    if type(value) is not kind:
        raise AudioAnalysisError("invalid audio analysis record")


def _finite(value: object, low: float, high: float) -> float:
    if type(value) not in (float, int):
        raise AudioAnalysisError("invalid finite audio measurement")
    try:
        number = float(value)  # type: ignore[arg-type]
    except OverflowError:
        raise AudioAnalysisError("invalid finite audio measurement") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise AudioAnalysisError("invalid finite audio measurement")
    return number


@dataclass(frozen=True, slots=True)
class TickSpan(_AudioContract):
    start_ticks: int
    duration_ticks: int

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _integer(self.start_ticks, "analysis start ticks")
        _integer(self.duration_ticks, "analysis duration ticks", 1)
        if self.start_ticks > 2**63 - 1 - self.duration_ticks:
            raise AudioAnalysisError("audio analysis end overflows")


@dataclass(frozen=True, slots=True)
class AnalysisSettings(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"timebase": AudioTimebase}

    timebase: AudioTimebase
    window_ticks: int
    activity_threshold_dbfs: float = -45.0
    clipping_risk_dbfs: float = -0.1
    timing_tolerance_ticks: int = 0
    timeout_seconds: float = 10.0
    max_input_bytes: int = _MAX_BYTES
    max_output_bytes: int = 65536
    max_duration_seconds: int = 60

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _instance(self.timebase, AudioTimebase)
        _integer(self.timebase.ticks_per_second, "analysis ticks per second", 1, 192000)
        _integer(self.window_ticks, "energy window ticks",
                 (self.timebase.ticks_per_second + 99) // 100,
                 self.timebase.ticks_per_second)
        _integer(self.timing_tolerance_ticks, "timing tolerance ticks", 0,
                 self.timebase.duration_ticks)
        for name, low, high in (("activity_threshold_dbfs", -120, -1),
                               ("clipping_risk_dbfs", -6, 0), ("timeout_seconds", .05, 30)):
            object.__setattr__(self, name, _finite(getattr(self, name), low, high))
        _integer(self.max_input_bytes, "maximum audio bytes", 44, _MAX_BYTES)
        _integer(self.max_output_bytes, "maximum audio tool output", 4096, 1024 * 1024)
        _integer(self.max_duration_seconds, "maximum audio duration", 1, 60)


@dataclass(frozen=True, slots=True)
class DeclaredTiming(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"picture": RevisionPin}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"speech_windows": TickSpan}

    picture: RevisionPin
    start_ticks: int
    duration_ticks: int
    speech_windows: tuple[TickSpan, ...] = ()

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        _instance(self.picture, RevisionPin)
        TickSpan(self.start_ticks, self.duration_ticks)
        _tuple(self.speech_windows, TickSpan, "declared speech windows")
        _unique(self.speech_windows, "declared speech windows")
        if self.speech_windows != tuple(sorted(self.speech_windows,
                                               key=lambda item: item.start_ticks)):
            raise AudioAnalysisError("declared speech windows must be ordered")
        for item in self.speech_windows:
            if (item.start_ticks < self.start_ticks or item.start_ticks + item.duration_ticks
                    > self.start_ticks + self.duration_ticks):
                raise AudioAnalysisError("declared speech window exceeds declared audio")


@dataclass(frozen=True, slots=True)
class ToolIdentity(_AudioContract):
    version: str
    binary_sha256: str
    version_output_sha256: str
    kind: str = "native"

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        if (not isinstance(self.version, str)
                or re.fullmatch(r"[A-Za-z0-9._+\-]{1,80}", self.version) is None
                or self.kind not in ("native", "fake")):
            raise AudioAnalysisError("invalid audio tool identity")
        _digest(self.binary_sha256, "audio tool binary digest")
        _digest(self.version_output_sha256, "audio tool version output digest")


@dataclass(frozen=True, slots=True)
class Measurement(_AudioContract):
    value: float | None
    status: str
    reason: str = "none"

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        if self.status not in ("measured", "undefined", "unavailable"):
            raise AudioAnalysisError("invalid measurement status")
        if self.reason not in ("none", "silence", "insufficient_duration", "below_gate_or_undefined",
                                "tool_unavailable", "tool_failed", "invalid_tool_output",
                                "tool_cleanup_uncertain"):
            raise AudioAnalysisError("invalid measurement reason")
        if ((self.status == "undefined" and self.reason not in (
                "silence", "insufficient_duration", "below_gate_or_undefined"))
                or (self.status == "unavailable" and self.reason not in (
                    "tool_unavailable", "tool_failed", "invalid_tool_output",
                    "tool_cleanup_uncertain"))):
            raise AudioAnalysisError("measurement reason does not match its status")
        if self.status == "measured":
            object.__setattr__(self, "value", _finite(self.value, -200, 100))
            if self.reason != "none":
                raise AudioAnalysisError("measured value cannot have an undefined reason")
        elif self.value is not None or self.reason == "none":
            raise AudioAnalysisError("unavailable measurement cannot contain a value")


@dataclass(frozen=True, slots=True)
class EnergyWindow(_AudioContract):
    start_ticks: int
    duration_ticks: int
    rms_dbfs: float | None
    active: bool

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        TickSpan(self.start_ticks, self.duration_ticks)
        if self.rms_dbfs is not None:
            object.__setattr__(self, "rms_dbfs", _finite(self.rms_dbfs, -200, 0))
        if type(self.active) is not bool or self.active and self.rms_dbfs is None:
            raise AudioAnalysisError("invalid audio energy window")


@dataclass(frozen=True, slots=True)
class AudioAnalysisReport(_AudioContract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"input": RevisionPin,
        "settings": AnalysisSettings, "timing": DeclaredTiming, "integrated_lufs": Measurement,
        "true_peak_dbtp": Measurement}
    _optional_objects: ClassVar[dict[str, type[Contract]]] = {"tool": ToolIdentity}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"energy_windows": EnergyWindow}
    _tuple_fields: ClassVar[tuple[str, ...]] = ("issues",)

    input: RevisionPin
    settings: AnalysisSettings
    timing: DeclaredTiming
    tool: ToolIdentity | None
    sample_rate: int
    channels: int
    frames: int
    sample_peak_dbfs: float | None
    clipping_risk_samples: int
    digital_silence: bool
    energy_windows: tuple[EnergyWindow, ...]
    integrated_lufs: Measurement
    true_peak_dbtp: Measurement
    issues: tuple[str, ...]
    method: str = "pcm16-energy+loudnorm-input-v1"
    meter_filter: str = _FILTER
    sample_format: str = "pcm_s16le"
    qualitative_review: str = "unavailable"
    speech_recognition: str = "not_assessed"
    visible_lip_sync: str = "not_assessed"

    def __post_init__(self) -> None:
        _AudioContract.__post_init__(self)
        for item, kind in ((self.input, RevisionPin), (self.settings, AnalysisSettings),
                           (self.timing, DeclaredTiming), (self.integrated_lufs, Measurement),
                           (self.true_peak_dbtp, Measurement)):
            _instance(item, kind)
        if self.tool is not None:
            _instance(self.tool, ToolIdentity)
        if self.sample_rate not in (16000, 24000, 44100, 48000) or type(self.sample_rate) is not int:
            raise AudioAnalysisError("unsupported analysis sample rate")
        _integer(self.channels, "audio channels", 1, 2)
        _integer(self.frames, "audio frames", 1, self.sample_rate * 60)
        if self.frames > self.sample_rate * self.settings.max_duration_seconds:
            raise AudioAnalysisError("audio duration exceeds report bounds")
        self.settings.timebase.check_interval(self.timing.start_ticks, self.timing.duration_ticks)
        _integer(self.clipping_risk_samples, "clipping risk samples", 0, self.frames * self.channels)
        if self.sample_peak_dbfs is not None:
            object.__setattr__(self, "sample_peak_dbfs", _finite(self.sample_peak_dbfs, -200, 0))
        if type(self.digital_silence) is not bool:
            raise AudioAnalysisError("invalid digital silence marker")
        _tuple(self.energy_windows, EnergyWindow, "energy windows")
        if not 0 < len(self.energy_windows) <= 6000:
            raise AudioAnalysisError("invalid energy window count")
        step, remainder = divmod(self.settings.window_ticks * self.sample_rate,
                                 self.settings.timebase.ticks_per_second)
        if remainder or not step or len(self.energy_windows) != (self.frames + step - 1) // step:
            raise AudioAnalysisError("inconsistent energy windows")
        for index, window in enumerate(self.energy_windows):
            start = index * self.settings.window_ticks
            stop = -(-min(self.frames, (index + 1) * step)
                      * self.settings.timebase.ticks_per_second // self.sample_rate)
            if ((window.start_ticks, window.duration_ticks) != (start, stop - start)
                    or window.active != (window.rms_dbfs is not None
                                        and window.rms_dbfs >= self.settings.activity_threshold_dbfs)):
                raise AudioAnalysisError("inconsistent energy window metadata")
        if (self.digital_silence != (self.sample_peak_dbfs is None)
                or self.digital_silence != all(item.rms_dbfs is None for item in self.energy_windows)
                or self.digital_silence and self.clipping_risk_samples):
            raise AudioAnalysisError("inconsistent digital silence metadata")
        if (self.tool is None and any(metric.status != "unavailable"
                                     for metric in (self.integrated_lufs, self.true_peak_dbtp))):
            raise AudioAnalysisError("native metrics require a tool identity")
        if self.digital_silence and any(metric.status == "measured"
                                       for metric in (self.integrated_lufs, self.true_peak_dbtp)):
            raise AudioAnalysisError("digital silence cannot have finite loudness or true peak")
        _tuple(self.issues, str, "audio issues")
        if any(item not in _ISSUES for item in self.issues) or tuple(sorted(set(self.issues))) != self.issues:
            raise AudioAnalysisError("invalid audio analysis issues")
        if (self.method != "pcm16-energy+loudnorm-input-v1" or self.meter_filter != _FILTER
                or self.sample_format != "pcm_s16le" or self.qualitative_review != "unavailable"
                or self.speech_recognition != "not_assessed" or self.visible_lip_sync != "not_assessed"):
            raise AudioAnalysisError("unsupported analysis or review claim")
        if self.issues != tuple(sorted(_analysis_issues(self.settings, self.timing, self.frames,
                self.sample_rate, self.energy_windows, self.clipping_risk_samples,
                self.digital_silence, self.integrated_lufs, self.true_peak_dbtp))):
            raise AudioAnalysisError("inconsistent audio analysis issues")

    @property
    def measurement_status(self) -> str:
        statuses = (self.integrated_lufs.status, self.true_peak_dbtp.status)
        return "unavailable" if "unavailable" in statuses else (
            "undefined" if "undefined" in statuses else "measured")

    def matches_inputs(self, pin: RevisionPin, settings: AnalysisSettings, timing: DeclaredTiming,
                       tool: ToolIdentity | None) -> bool:
        return (self.input, self.settings, self.timing, self.tool) == (pin, settings, timing, tool)


class MeasurementTool(Protocol):
    def identify(self, *, timeout_seconds: float, max_output_bytes: int) -> ToolIdentity: ...

    def measure(self, payload: bytes, *, timeout_seconds: float,
                max_output_bytes: int) -> bytes: ...


ProcessExecutor = Callable[..., bytes]


class FFmpegTool:
    """Explicit operator-selected executable; sanitized, pipe-only, fixed measurement graph."""

    def __init__(self, executable: Path, *, executor: ProcessExecutor = run_audio_process) -> None:
        if not isinstance(executable, Path) or not executable.is_absolute():
            raise AudioAnalysisError("audio tool requires an absolute local executable")
        self._path, self._executor = executable, executor
        self._native = executor is run_audio_process
        self._identity: ToolIdentity | None = None

    def _binary_digest(self) -> str:
        descriptor = -1
        value = b""
        failed = False
        try:
            descriptor = os.open(self._path, os.O_RDONLY | os.O_NONBLOCK)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 64 * 1024 * 1024:
                raise AudioAnalysisError("unsupported audio tool executable")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                value = stream.read(64 * 1024 * 1024 + 1)
            if len(value) != info.st_size:
                raise AudioAnalysisError("audio tool executable changed")
        except Exception:
            failed = True
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except Exception:
                    failed = True
        if failed:
            raise AudioAnalysisError("audio tool identity unavailable") from None
        return hashlib.sha256(value).hexdigest()

    def identify(self, *, timeout_seconds: float, max_output_bytes: int) -> ToolIdentity:
        digest = self._binary_digest()
        output = self._executor((str(self._path), "-version"), payload=b"",
                               timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
        if not isinstance(output, bytes) or len(output) > max_output_bytes:
            raise AudioAnalysisError("invalid audio tool version response")
        match = re.match(rb"ffmpeg version ([A-Za-z0-9._+\-]{1,80})[ \r\n]", output)
        if match is None:
            raise AudioAnalysisError("unsupported audio tool version response")
        self._identity = ToolIdentity(match[1].decode("ascii"), digest,
            hashlib.sha256(output).hexdigest(), "native" if self._native else "fake")
        return self._identity

    def measure(self, payload: bytes, *, timeout_seconds: float, max_output_bytes: int) -> bytes:
        if self._identity is None or self._binary_digest() != self._identity.binary_sha256:
            raise AudioAnalysisError("audio tool identity changed")
        result = self._executor((str(self._path), "-hide_banner", "-nostdin", "-nostats", "-v",
            "info", "-protocol_whitelist", "pipe", "-f", "wav", "-i", "pipe:0", "-map", "0:a:0",
            "-vn", "-sn", "-dn", "-af", _FILTER, "-f", "null", "pipe:1"), payload=payload,
            timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
        if self._binary_digest() != self._identity.binary_sha256:
            raise AudioAnalysisError("audio tool identity changed")
        return result


def _meter(output: bytes, *, silence: bool, short: bool) -> tuple[Measurement, Measurement]:
    try:
        text = output.decode("utf-8")
        # Only the fixed loudnorm JSON block is parsed; other native diagnostics are discarded.
        blocks = re.findall(r"\{[^{}]*\}", text)
        if len(blocks) != 1:
            raise ValueError
        values = strict_loads(blocks[0])
        if not isinstance(values, dict) or set(values) != _LOUDNORM_KEYS:
            raise ValueError
        measurements = []
        for name in ("input_i", "input_tp"):
            value = values[name]
            if not isinstance(value, str):
                raise ValueError
            if value != "-inf":
                number = _finite(float(value), -200, 100)
            if value == "-inf" or silence:
                reason = "silence" if silence else (
                    "insufficient_duration" if short else "below_gate_or_undefined")
                measurements.append(Measurement(None, "undefined", reason))
            else:
                measurements.append(Measurement(number, "measured"))
        return measurements[0], measurements[1]
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise AudioAnalysisError("invalid audio meter output") from None


def _analysis_issues(settings: AnalysisSettings, timing: DeclaredTiming, frames: int, rate: int,
                     windows: tuple[EnergyWindow, ...], risk: int, silence: bool,
                     integrated: Measurement, true_peak: Measurement) -> set[str]:
    issues: set[str] = set()
    if risk:
        issues.add("clipping_risk")
    # Objective full-scale overload, separate from delivery-specific M5.4 targets.
    if true_peak.status == "measured" and true_peak.value is not None and true_peak.value > 0:
        issues.add("true_peak_overload")
    if silence:
        issues.add("silent_audio")
    exact_ticks = Fraction(frames * settings.timebase.ticks_per_second, rate)
    if abs(exact_ticks - timing.duration_ticks) > settings.timing_tolerance_ticks:
        issues.add("duration_mismatch")
    if exact_ticks + timing.start_ticks > settings.timebase.duration_ticks:
        issues.add("audio_exceeds_picture")
    active = tuple(item for item in windows if item.active)
    tolerance = settings.timing_tolerance_ticks
    if timing.speech_windows:
        def intersects(observed: EnergyWindow, expected: TickSpan) -> bool:
            start = observed.start_ticks + timing.start_ticks
            return (start < expected.start_ticks + expected.duration_ticks + tolerance
                    and start + observed.duration_ticks > expected.start_ticks - tolerance)
        if any(not any(intersects(item, expected) for item in active)
               for expected in timing.speech_windows):
            issues.add("missing_expected_energy")
        def covered(observed: EnergyWindow) -> bool:
            cursor = observed.start_ticks + timing.start_ticks
            end = cursor + observed.duration_ticks
            for expected in timing.speech_windows:
                if expected.start_ticks - tolerance > cursor:
                    break
                cursor = max(cursor, expected.start_ticks + expected.duration_ticks + tolerance)
                if cursor >= end:
                    return True
            return False
        if any(not covered(item) for item in active):
            issues.add("energy_outside_expected_windows")
    if "unavailable" in (integrated.status, true_peak.status):
        issues.add("measurement_unavailable")
    if integrated.status == "undefined":
        issues.add("integrated_loudness_undefined")
    if true_peak.status == "undefined":
        issues.add("true_peak_undefined")
    if "tool_cleanup_uncertain" in (integrated.reason, true_peak.reason):
        issues.add("tool_cleanup_uncertain")
    return issues


def analyze_wav(payload: bytes, pin: RevisionPin, settings: AnalysisSettings,
                timing: DeclaredTiming, *, tool: MeasurementTool | None = None) -> AudioAnalysisReport:
    """Measure exact WAV bytes; energy/declared timing are not ASR or visible lip sync.

    Window lengths must align to sample frames. Energy flags are conservative at
    window resolution, with configurable tick tolerance. The final partial window
    rounds its end outward by <1 tick; duration comparisons use exact fractions.
    No tool means unavailable native measurements, never a successful substitute.
    """
    _instance(pin, RevisionPin)
    _instance(settings, AnalysisSettings)
    _instance(timing, DeclaredTiming)
    settings.timebase.check_interval(timing.start_ticks, timing.duration_ticks)
    deadline = time.monotonic() + settings.timeout_seconds
    try:
        inspected = inspect_pcm16_wav(payload, max_input_bytes=settings.max_input_bytes,
                                      max_duration_seconds=settings.max_duration_seconds)
    except PCMError:
        raise AudioAnalysisError("unsupported or malformed PCM WAV") from None
    pcm, rate, channels = inspected.pcm, inspected.sample_rate, inspected.channels
    if hashlib.sha256(payload).hexdigest() != pin.sha256:
        raise AudioAnalysisError("audio input checksum differs")
    frame_step, remainder = divmod(settings.window_ticks * rate, settings.timebase.ticks_per_second)
    if remainder or frame_step < 1:
        raise AudioAnalysisError("energy window does not align to sample frames")
    frames = len(pcm) // (2 * channels)
    windows: list[EnergyWindow] = []
    peak, risk = 0, 0
    clip = 10 ** (settings.clipping_risk_dbfs / 20)
    for first in range(0, frames, frame_step):
        end = min(frames, first + frame_step)
        samples = (value[0] for value in struct.iter_unpack("<h", pcm[first * channels * 2:
                                                                      end * channels * 2]))
        square_sum = 0
        for value in samples:
            magnitude = abs(value)
            peak = max(peak, magnitude)
            risk += magnitude / 32768 >= clip
            square_sum += value * value
        rms = math.sqrt(square_sum / ((end - first) * channels)) / 32768
        dbfs = 20 * math.log10(rms) if rms else None
        start = first * settings.timebase.ticks_per_second // rate
        stop = -(-end * settings.timebase.ticks_per_second // rate)
        windows.append(EnergyWindow(start, stop - start, dbfs,
                                    dbfs is not None and dbfs >= settings.activity_threshold_dbfs))
    identity = None
    integrated = true_peak = Measurement(None, "unavailable", "tool_unavailable")
    if tool is not None:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AudioAnalysisError("audio analysis deadline exceeded")
            identified = tool.identify(timeout_seconds=remaining,
                                       max_output_bytes=settings.max_output_bytes)
            _instance(identified, ToolIdentity)
            if type(tool) is not FFmpegTool and identified.kind != "fake":
                raise AudioAnalysisError("unsupported native audio measurement adapter")
            identity = identified
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AudioAnalysisError("audio analysis deadline exceeded")
            output = tool.measure(payload, timeout_seconds=remaining,
                                  max_output_bytes=settings.max_output_bytes)
            if (not isinstance(output, bytes) or len(output) > settings.max_output_bytes
                    or time.monotonic() > deadline):
                raise AudioAnalysisError("audio analysis tool bounds exceeded")
        except AudioProcessError as error:
            reason = "tool_cleanup_uncertain" if error.cleanup_uncertain else (
                "tool_failed" if identity else "tool_unavailable")
            integrated = true_peak = Measurement(None, "unavailable", reason)
        except Exception:
            integrated = true_peak = Measurement(None, "unavailable",
                                                  "tool_failed" if identity else "tool_unavailable")
        else:
            try:
                integrated, true_peak = _meter(output, silence=peak == 0,
                                               short=Fraction(frames, rate) < Fraction(2, 5))
            except AudioAnalysisError:
                integrated = true_peak = Measurement(None, "unavailable", "invalid_tool_output")
    issues = _analysis_issues(settings, timing, frames, rate, tuple(windows), risk, peak == 0,
                             integrated, true_peak)
    return AudioAnalysisReport(pin, settings, timing, identity, rate, channels, frames,
        20 * math.log10(peak / 32768) if peak else None, risk, peak == 0, tuple(windows),
        integrated, true_peak, tuple(sorted(issues)))
