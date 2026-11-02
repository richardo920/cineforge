"""Required opt-in native acceptance: generated PCM only, fail if FFmpeg is unavailable.

Run: PYTHONPATH=src python -m unittest discover -s tests_native -v
CINEFORGE_TEST_FFMPEG selects an existing binary; nothing is installed or downloaded.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_audio_analysis import pin_for, settings, timing, wav_bytes  # noqa: E402

from cineforge.audio_analysis import (  # noqa: E402
    AudioAnalysisReport,
    FFmpegTool,
    TickSpan,
    analyze_wav,
)
from cineforge.audio_process import AudioProcessError, run_audio_process  # noqa: E402


class NativeAudioAnalysisTests(unittest.TestCase):
    reports = []
    executable = Path(os.environ.get("CINEFORGE_TEST_FFMPEG", "/opt/homebrew/bin/ffmpeg"))

    def measure(self, name, payload, *, cfg=None, expected=None):
        cfg = cfg if cfg is not None else settings()
        expected = expected if expected is not None else timing()
        report = analyze_wav(payload, pin_for(payload), cfg, expected,
                             tool=FFmpegTool(self.executable))
        self.assertIsNotNone(report.tool, "native FFmpeg identification is REQUIRED")
        self.assertEqual(report.tool.kind, "native")
        self.assertNotEqual(report.measurement_status, "unavailable",
                            "required native measurements must run, never skip/fake")
        self.assertEqual(AudioAnalysisReport.from_json(report.to_json()), report)
        self.reports.append({"fixture": name, "report": report.to_dict()})
        return report

    def reference(self, payload):
        command = (str(self.executable), "-hide_banner", "-nostdin", "-nostats", "-v", "info",
            "-protocol_whitelist", "pipe", "-f", "wav", "-i", "pipe:0", "-af",
            "ebur128=peak=true:framelog=verbose", "-f", "null", "pipe:1")
        text = run_audio_process(command, payload=payload, timeout_seconds=10,
                                 max_output_bytes=65536).decode()
        integrated = re.findall(r"I:\s*(-?\d+\.\d+) LUFS", text)
        peak = re.findall(r"True peak:\s*Peak:\s*(-?\d+\.\d+) dBFS", text)
        self.assertTrue(integrated and peak, "independent ebur128 summary is REQUIRED")
        return float(integrated[-1]), float(peak[-1])

    def test_calibration_tone_lufs_true_peak_and_gain_change(self):
        data = wav_bytes()
        report = self.measure("1kHz-peak-minus20dBFS-mono", data)
        # Calibrated 1kHz mono tone; LUFS is from the upstream meter, never local RMS.
        self.assertAlmostEqual(report.integrated_lufs.value, -23.0, delta=.3)
        self.assertAlmostEqual(report.true_peak_dbtp.value, -20.0, delta=.1)
        self.assertAlmostEqual(report.sample_peak_dbfs, -20.0, delta=.002)
        integrated, peak = self.reference(data)
        self.assertAlmostEqual(report.integrated_lufs.value, integrated, delta=.15)
        self.assertAlmostEqual(report.true_peak_dbtp.value, peak, delta=.3)
        quiet = self.measure("same-tone-6dB-lower", wav_bytes(amplitude=.05))
        self.assertAlmostEqual(report.integrated_lufs.value - quiet.integrated_lufs.value,
                               6.02, delta=.1)

    def test_intersample_true_peak_exceeds_sample_peak(self):
        data = wav_bytes(amplitude=.7, frequency=12000, phase=.7853981633974483)
        report = self.measure("intersample-12kHz-phase45", data)
        self.assertGreater(report.true_peak_dbtp.value, report.sample_peak_dbfs + 2)
        _, reference_peak = self.reference(data)
        self.assertAlmostEqual(report.true_peak_dbtp.value, reference_peak, delta=.5)

    def test_positive_intersample_peak_is_reported_without_pcm_sample_risk(self):
        data = wav_bytes(amplitude=.99, frequency=12000, phase=.7853981633974483)
        report = self.measure("positive-intersample-overload", data)
        self.assertGreater(report.true_peak_dbtp.value, 0)
        self.assertLess(report.sample_peak_dbfs, -3)
        self.assertEqual(report.clipping_risk_samples, 0)
        self.assertNotIn("clipping_risk", report.issues)
        self.assertIn("true_peak_overload", report.issues)
        _, reference_peak = self.reference(data)
        self.assertGreater(reference_peak, 0)
        self.assertAlmostEqual(report.true_peak_dbtp.value, reference_peak, delta=.15)

    def test_saturated_samples_trigger_clipping_risk(self):
        report = self.measure("saturated-PCM-risk", wav_bytes(saturated=True))
        self.assertEqual(report.clipping_risk_samples, 48000)
        self.assertEqual(report.sample_peak_dbfs, 0)
        self.assertIn("clipping_risk", report.issues)

    def test_actual_silence_has_undefined_metrics(self):
        report = self.measure("digital-silence", wav_bytes(amplitude=0))
        self.assertTrue(report.digital_silence)
        self.assertEqual(report.measurement_status, "undefined")
        for metric in (report.integrated_lufs, report.true_peak_dbtp):
            self.assertIsNone(metric.value)
            self.assertEqual(metric.reason, "silence")
        self.assertTrue(all(not window.active for window in report.energy_windows))

    def test_short_and_gated_signals_do_not_fabricate_lufs(self):
        short = self.measure("100ms-nonsilent-undefined", wav_bytes(seconds=.1),
                             cfg=settings(100), expected=timing(100))
        self.assertFalse(short.digital_silence)
        self.assertIsNone(short.integrated_lufs.value)
        self.assertEqual(short.integrated_lufs.reason, "insufficient_duration")
        low = self.measure("below-gate-PCM-nonsilent", wav_bytes(amplitude=.00003))
        self.assertFalse(low.digital_silence)
        self.assertIsNone(low.integrated_lufs.value)
        self.assertEqual(low.integrated_lufs.reason, "below_gate_or_undefined")

    def test_energy_windows_and_declared_timeline_mismatch(self):
        payload = wav_bytes(active=(.3, .7))
        report = self.measure("speech-like-energy-window", payload,
                              expected=timing(speech=(TickSpan(300, 400),)))
        self.assertEqual([w.start_ticks for w in report.energy_windows if w.active],
                         [300, 400, 500, 600])
        self.assertNotIn("missing_expected_energy", report.issues)
        mismatch = self.measure("declared-window-offset-mismatch", payload,
            cfg=replace(settings(), timing_tolerance_ticks=0),
            expected=timing(900, start=100, speech=(TickSpan(100, 100),)))
        self.assertIn("duration_mismatch", mismatch.issues)
        self.assertIn("audio_exceeds_picture", mismatch.issues)
        self.assertIn("missing_expected_energy", mismatch.issues)
        self.assertIn("energy_outside_expected_windows", mismatch.issues)
        self.assertEqual(mismatch.visible_lip_sync, "not_assessed")
        self.assertEqual(mismatch.speech_recognition, "not_assessed")

    def test_available_native_tool_output_and_time_limits_are_enforced(self):
        with self.assertRaises(AudioProcessError):
            run_audio_process((str(self.executable), "-filters"), payload=b"",
                              timeout_seconds=10, max_output_bytes=4096)
        command = (str(self.executable), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
            "-protocol_whitelist", "pipe", "-re", "-f", "wav", "-i", "pipe:0",
            "-f", "null", "pipe:1")
        with self.assertRaises(AudioProcessError):
            run_audio_process(command, payload=wav_bytes(seconds=2), timeout_seconds=.1,
                              max_output_bytes=4096)

    @classmethod
    def tearDownClass(cls):
        value = {"method": "native-generated-fixtures-not-production-media",
                 "fixtures": cls.reports}
        destination = os.environ.get("CINEFORGE_NATIVE_EVIDENCE_FILE")
        if destination:
            Path(destination).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"native_fixtures": len(cls.reports), "tool": (
            cls.reports[0]["report"]["tool"] if cls.reports else None)}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
