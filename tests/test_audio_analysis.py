"""Synthetic PCM and fake meters only; default discovery never invokes FFmpeg."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cineforge import audio_process
from cineforge.audio import AudioTimebase, RevisionPin
from cineforge.audio_analysis import (
    AnalysisSettings,
    AudioAnalysisError,
    AudioAnalysisReport,
    DeclaredTiming,
    FFmpegTool,
    Measurement,
    TickSpan,
    ToolIdentity,
    analyze_wav,
)
from cineforge.audio_pcm import PCMError, inspect_pcm16_wav
from cineforge.audio_process import AudioProcessError, run_audio_process
from cineforge.contracts import ContractError

DIGEST = "a" * 64


def wav_bytes(*, seconds: float = 1, rate: int = 48000, channels: int = 1,
              amplitude: float = .1, frequency: float = 1000, phase: float = 0,
              active: tuple[float, float] = (0, 100), saturated: bool = False) -> bytes:
    output = io.BytesIO()
    data = bytearray()
    for frame in range(round(seconds * rate)):
        sample = round(32767 * amplitude * math.sin(2 * math.pi * frequency * frame / rate + phase))
        if saturated:
            sample = 32767 if frame % 2 else -32768
        if not active[0] <= frame / rate < active[1]:
            sample = 0
        data.extend(struct.pack("<h", sample) * channels)
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(data)
    return output.getvalue()


def pin_for(payload: bytes) -> RevisionPin:
    return RevisionPin("synthetic-audio", 1, hashlib.sha256(payload).hexdigest())


def settings(duration: int = 1000) -> AnalysisSettings:
    return AnalysisSettings(AudioTimebase(1000, duration), 100)


def timing(duration: int = 1000, *, start: int = 0,
           speech: tuple[TickSpan, ...] = ()) -> DeclaredTiming:
    return DeclaredTiming(RevisionPin("synthetic-picture", 1, DIGEST), start, duration, speech)


def meter_output(integrated: str = "-23.05", true_peak: str = "-20.00") -> bytes:
    return json.dumps({"input_i": integrated, "input_tp": true_peak, "input_lra": "0.00",
        "input_thresh": "-33.05", "output_i": "-23.95", "output_tp": "-20.95",
        "output_lra": "0.00", "output_thresh": "-33.95", "normalization_type": "linear",
        "target_offset": "-0.05"}).encode()


class FakeTool:
    def __init__(self, output: bytes | None = None, *, failure: str = "") -> None:
        self.output = output if output is not None else meter_output()
        self.failure = failure
        self.calls = []

    def identify(self, **limits) -> ToolIdentity:
        self.calls.append(("identify", limits))
        if self.failure == "identify":
            raise RuntimeError("private-tool-canary")
        return ToolIdentity("synthetic-v1", DIGEST, DIGEST, "fake")

    def measure(self, payload: bytes, **limits) -> bytes:
        self.calls.append(("measure", limits))
        if self.failure == "measure":
            raise RuntimeError("private-tool-canary")
        return self.output


class AudioAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = wav_bytes(active=(.3, .7))
        self.pin = pin_for(self.payload)
        self.settings = settings()
        self.timing = timing(speech=(TickSpan(300, 400),))

    def analyze(self, **overrides) -> AudioAnalysisReport:
        values = dict(payload=self.payload, pin=self.pin, settings=self.settings,
                      timing=self.timing, tool=FakeTool())
        values.update(overrides)
        return analyze_wav(**values)

    def test_fake_meter_is_explicit_and_energy_is_not_lufs_or_lip_sync(self) -> None:
        report = self.analyze()
        self.assertEqual(report.tool.kind, "fake")
        self.assertEqual(report.measurement_status, "measured")
        self.assertEqual(report.integrated_lufs.value, -23.05)
        self.assertEqual(report.true_peak_dbtp.value, -20)
        self.assertAlmostEqual(report.sample_peak_dbfs, -20, delta=.002)
        self.assertEqual([item.start_ticks for item in report.energy_windows if item.active],
                         [300, 400, 500, 600])
        self.assertEqual(report.issues, ())
        self.assertNotEqual(report.energy_windows[3].rms_dbfs, report.integrated_lufs.value)
        self.assertEqual((report.qualitative_review, report.speech_recognition,
                          report.visible_lip_sync), ("unavailable", "not_assessed", "not_assessed"))

    def test_report_round_trip_identity_and_every_binding_change_is_stale(self) -> None:
        report = self.analyze()
        self.assertEqual(AudioAnalysisReport.from_json(report.to_json()), report)
        self.assertEqual(report.digest(), hashlib.sha256(report.to_json().encode()).hexdigest())
        self.assertTrue(report.matches_inputs(self.pin, self.settings, self.timing, report.tool))
        for pin, cfg, expected, tool in (
            (replace(self.pin, revision=2), self.settings, self.timing, report.tool),
            (replace(self.pin, sha256="b" * 64), self.settings, self.timing, report.tool),
            (self.pin, replace(self.settings, activity_threshold_dbfs=-40), self.timing, report.tool),
            (self.pin, self.settings, replace(self.timing, picture=replace(
                self.timing.picture, revision=2)), report.tool),
            (self.pin, self.settings, replace(self.timing, speech_windows=()), report.tool),
            (self.pin, self.settings, self.timing, replace(report.tool, binary_sha256="b" * 64)),
            (self.pin, self.settings, self.timing, None),
        ):
            self.assertFalse(report.matches_inputs(pin, cfg, expected, tool))

    def test_unavailable_missing_failed_and_malformed_tool_never_substitute_metrics(self) -> None:
        for tool, reason in ((None, "tool_unavailable"),
                             (FakeTool(failure="identify"), "tool_unavailable"),
                             (FakeTool(failure="measure"), "tool_failed"),
                             (FakeTool(b"private-tool-canary"), "invalid_tool_output"),
                             (FakeTool(b"x" * 65537), "tool_failed")):
            report = self.analyze(tool=tool)
            self.assertEqual(report.measurement_status, "unavailable")
            self.assertEqual(report.integrated_lufs.reason, reason)
            self.assertIsNone(report.integrated_lufs.value)
            self.assertIsNone(report.true_peak_dbtp.value)
            self.assertIn("measurement_unavailable", report.issues)
            self.assertNotIn("private-tool-canary", report.to_json())
        with tempfile.TemporaryDirectory() as folder:
            report = self.analyze(tool=FFmpegTool(Path(folder) / "missing-private-tool-canary"))
        self.assertEqual(report.measurement_status, "unavailable")
        self.assertIsNone(report.tool)

    def test_silence_short_and_gated_undefined_are_distinct_without_json_infinity(self) -> None:
        for payload, reason in ((wav_bytes(amplitude=0), "silence"),
                                (wav_bytes(seconds=.1), "insufficient_duration"),
                                (wav_bytes(amplitude=.00003), "below_gate_or_undefined")):
            cfg = settings(round((len(payload) - 44) / 96))
            expected = timing(cfg.timebase.duration_ticks)
            report = self.analyze(payload=payload, pin=pin_for(payload), settings=cfg,
                                  timing=expected, tool=FakeTool(meter_output("-inf", "-inf")))
            self.assertEqual(report.measurement_status, "undefined")
            self.assertEqual(report.integrated_lufs.reason, reason)
            self.assertIsNone(report.integrated_lufs.value)
            self.assertNotIn("Infinity", report.to_json())
            self.assertNotIn('"-inf"', report.to_json())
        silent = wav_bytes(amplitude=0)
        # Even a meter's finite floor cannot turn exact digital silence into finite LUFS.
        report = self.analyze(payload=silent, pin=pin_for(silent), tool=FakeTool())
        self.assertEqual(report.integrated_lufs, Measurement(None, "undefined", "silence"))
        self.assertTrue(report.digital_silence)
        self.assertTrue(all(item.rms_dbfs is None for item in report.energy_windows))

    def test_sample_clipping_risk_stereo_energy_and_true_peak_are_separate(self) -> None:
        data = wav_bytes(saturated=True, channels=2)
        report = self.analyze(payload=data, pin=pin_for(data), timing=timing(),
                              tool=FakeTool(meter_output("-1.0", "2.0")))
        self.assertEqual(report.sample_peak_dbfs, 0)
        self.assertEqual(report.clipping_risk_samples, 96000)
        self.assertEqual(report.true_peak_dbtp.value, 2)
        self.assertIn("clipping_risk", report.issues)
        self.assertNotEqual(report.sample_peak_dbfs, report.true_peak_dbtp.value)
        mono = self.analyze()
        stereo = wav_bytes(channels=2, active=(.3, .7))
        other = self.analyze(payload=stereo, pin=pin_for(stereo))
        self.assertEqual(mono.energy_windows, other.energy_windows)

    def test_declared_duration_picture_and_speech_window_issues(self) -> None:
        report = self.analyze(timing=timing(900, speech=(TickSpan(100, 100),)))
        self.assertIn("duration_mismatch", report.issues)
        self.assertIn("missing_expected_energy", report.issues)
        self.assertIn("energy_outside_expected_windows", report.issues)
        shifted = self.analyze(timing=timing(900, start=100))
        self.assertIn("audio_exceeds_picture", shifted.issues)
        tolerated = self.analyze(timing=timing(999), settings=replace(
            self.settings, timing_tolerance_ticks=1))
        self.assertNotIn("duration_mismatch", tolerated.issues)
        partial = self.analyze(timing=timing(speech=(TickSpan(350, 300),)))
        self.assertIn("energy_outside_expected_windows", partial.issues)
        with self.assertRaises(ContractError):
            self.analyze(timing=timing(1001))
        with self.assertRaises(ContractError):
            timing(speech=(TickSpan(999, 2),))

    def test_partial_frame_duration_uses_exact_fraction_not_rounded_end(self) -> None:
        data = wav_bytes(seconds=1 / 48000)
        report = self.analyze(payload=data, pin=pin_for(data), timing=timing(1),
                              tool=FakeTool(meter_output("-inf", "-inf")))
        self.assertEqual(report.frames, 1)
        self.assertEqual(report.energy_windows[0].duration_ticks, 1)
        self.assertIn("duration_mismatch", report.issues)
        with self.assertRaises(ContractError):
            self.analyze(settings=AnalysisSettings(AudioTimebase(44100, 44100), 1000))

    def test_bad_wav_and_checksum_reject_before_any_tool_call(self) -> None:
        fake = FakeTool()
        bad = [b"https://private-canary/input.wav", self.payload[:-1],
               b"RF64" + self.payload[4:], self.payload + b"x"]
        for offset, value in ((20, 3), (22, 3), (34, 24)):
            damaged = bytearray(self.payload)
            damaged[offset:offset + 2] = value.to_bytes(2, "little")
            bad.append(bytes(damaged))
        damaged = bytearray(self.payload)
        damaged[36:40] = b"LIST"
        bad.append(bytes(damaged))
        for data in bad:
            with self.assertRaises(AudioAnalysisError) as caught:
                self.analyze(payload=data, pin=pin_for(data), tool=fake)
            self.assertNotIn("private-canary", str(caught.exception))
        with self.assertRaises(AudioAnalysisError):
            self.analyze(pin=replace(self.pin, sha256=DIGEST), tool=fake)
        with self.assertRaises(ContractError):
            self.analyze(settings=replace(self.settings, max_input_bytes=44), tool=fake)
        longer = wav_bytes(seconds=2)
        with self.assertRaises(AudioAnalysisError):
            self.analyze(payload=longer, pin=pin_for(longer),
                         settings=replace(self.settings, max_duration_seconds=1), tool=fake)
        self.assertEqual(fake.calls, [])

    def test_settings_versions_unknown_fields_boolean_and_nonfinite_are_strict(self) -> None:
        report = self.analyze()
        records = (self.settings, self.timing, self.pin, report.tool, TickSpan(0, 1),
                   Measurement(-23, "measured"), report.energy_windows[0], report)
        for record in records:
            for data in (dict(record.to_dict(), schema_version=True),
                         dict(record.to_dict(), schema_version=2),
                         dict(record.to_dict(), **{"private-canary": True})):
                with self.assertRaises(ContractError) as caught:
                    type(record).from_dict(data)
                self.assertNotIn("private-canary", str(caught.exception))
        for name in ("activity_threshold_dbfs", "clipping_risk_dbfs", "timeout_seconds"):
            for value in (True, float("nan"), float("inf"), 10**1000):
                with self.assertRaises(ContractError):
                    replace(self.settings, **{name: value})
        for name in ("window_ticks", "timing_tolerance_ticks", "max_input_bytes",
                     "max_output_bytes", "max_duration_seconds"):
            with self.assertRaises(ContractError):
                replace(self.settings, **{name: True})
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ContractError):
                Measurement(value, "measured")
        with self.assertRaises(ContractError):
            TickSpan(2**63 - 1, 1)
        for fields in ({"qualitative_review": "approved"}, {"visible_lip_sync": "passed"},
                       {"sample_peak_dbfs": True}, {"digital_silence": True},
                       {"tool": None}, {"energy_windows": ()}, {"meter_filter": "amovie=secret"}):
            with self.assertRaises(ContractError):
                replace(report, **fields)
        unavailable = self.analyze(tool=None)
        with self.assertRaises(ContractError):
            replace(unavailable, issues=())

    def test_malformed_native_stats_fail_closed_and_never_echo_private_output(self) -> None:
        base = json.loads(meter_output())
        variants = [b'{"private-canary":NaN}', b'{"x":1,"x":2}', b'\xff',
                    json.dumps(dict(base, private_canary=1)).encode(),
                    json.dumps(dict(base, input_i=True)).encode(),
                    json.dumps(dict(base, input_tp="NaN")).encode(),
                    json.dumps(dict(base, input_i="private-canary")).encode()]
        for output in variants:
            report = self.analyze(tool=FakeTool(output))
            self.assertEqual(report.integrated_lufs.reason, "invalid_tool_output")
            self.assertNotIn("private-canary", report.to_json())

    def test_native_command_fixed_pipe_only_and_version_details_are_hashed(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "private-tool-canary"
            path.write_bytes(b"synthetic-executable-identity")
            version = b"ffmpeg version 9.0.2 Copyright\nprivate-tool-canary compiler path\n"
            def execute(command, **limits):
                calls.append((command, limits))
                return version if command[-1] == "-version" else meter_output()
            report = self.analyze(tool=FFmpegTool(path, executor=execute))
            self.assertEqual(report.tool.kind, "fake")
            self.assertEqual(report.tool.binary_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report.tool.version_output_sha256, hashlib.sha256(version).hexdigest())
        self.assertEqual(calls[1][0][calls[1][0].index("-i") + 1], "pipe:0")
        self.assertEqual(calls[1][0][calls[1][0].index("-protocol_whitelist") + 1], "pipe")
        self.assertEqual(calls[1][1]["payload"], self.payload)
        self.assertEqual(calls[1][0][calls[1][0].index("-af") + 1], report.meter_filter)
        self.assertLess(calls[1][1]["timeout_seconds"], self.settings.timeout_seconds)
        self.assertNotIn("private-tool-canary", report.to_json())
        self.assertNotIn("private-tool-canary", repr(report))

    def test_shared_pcm_inspector_has_no_analysis_dependency_and_rejects_special_tool_files(self):
        for rate in (16000, 24000, 44100, 48000):
            for channels in (1, 2):
                value = inspect_pcm16_wav(wav_bytes(seconds=.1, rate=rate, channels=channels))
                self.assertEqual((value.sample_rate, value.channels, value.frames),
                                 (rate, channels, rate // 10))
                self.assertNotIn("pcm=", repr(value))
        for bounds in ({"max_input_bytes": True}, {"max_duration_seconds": True},
                       {"max_input_bytes": 0}, {"max_duration_seconds": 61}):
            with self.assertRaises(PCMError):
                inspect_pcm16_wav(self.payload, **bounds)
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "private-canary-fifo"
            os.mkfifo(fifo)
            report = self.analyze(tool=FFmpegTool(fifo))
        self.assertEqual(report.measurement_status, "unavailable")
        self.assertNotIn("private-canary", report.to_json())


class AudioProcessTests(unittest.TestCase):
    def run_python(self, code: str, *, payload: bytes = b"", timeout: float = 2,
                   maximum: int = 4096) -> bytes:
        return run_audio_process((sys.executable, "-c", code), payload=payload,
                                 timeout_seconds=timeout, max_output_bytes=maximum)

    def test_binary_input_and_stderr_bound_no_inherited_ffreport(self) -> None:
        with patch.dict(os.environ, FFREPORT="file=private-canary-report", PRIVATE_CANARY="secret"):
            result = self.run_python("import os,sys; assert 'FFREPORT' not in os.environ; "
                "assert 'PRIVATE_CANARY' not in os.environ; sys.stdout.buffer.write(sys.stdin.buffer.read())",
                payload=b"\x00\xffsynthetic")
        self.assertEqual(result, b"\x00\xffsynthetic")
        with self.assertRaises(AudioProcessError):
            self.run_python("import sys; sys.stderr.write('x'*1000000)")

    def test_nonzero_timeout_and_invalid_bounds_are_static_and_bounded(self) -> None:
        for code in ("raise RuntimeError('private-tool-canary')", "import time; time.sleep(5)"):
            with self.assertRaises(AudioProcessError) as caught:
                self.run_python(code, timeout=.1)
            self.assertNotIn("private-tool-canary", str(caught.exception))
        for kwargs in ({"timeout_seconds": True}, {"timeout_seconds": float("nan")},
                       {"max_output_bytes": True}, {"payload": b"x" * (16 * 1024 * 1024 + 1)}):
            values = dict(payload=b"", timeout_seconds=1, max_output_bytes=4096)
            values.update(kwargs)
            with self.assertRaises(AudioProcessError):
                run_audio_process((sys.executable, "-c", "pass"), **values)

    def test_post_spawn_clock_failure_and_descendant_pipe_kill_the_group(self) -> None:
        spawned = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            spawned.append(process)
            return process
        with patch.object(audio_process.subprocess, "Popen", side_effect=launch):
            with patch.object(audio_process.time, "monotonic", side_effect=[0, RuntimeError("canary")]):
                with self.assertRaises(AudioProcessError):
                    self.run_python("import time;time.sleep(5)")
        self.assertIsNotNone(spawned[0].poll())
        original_kill = os.killpg
        with patch.object(audio_process.os, "killpg", wraps=original_kill) as kill:
            with self.assertRaises(AudioProcessError):
                self.run_python("import os,time; child=os.fork(); "
                                "time.sleep(5) if child==0 else None", timeout=.2)
        self.assertEqual(kill.call_args.args[1], signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
