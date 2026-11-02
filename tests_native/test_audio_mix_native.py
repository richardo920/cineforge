"""Required generated-PCM acceptance with existing FFmpeg9.0.2; never install tools."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_audio_mix import bed_scene, pcm, scene  # noqa: E402

from cineforge.audio_analysis import FFmpegTool  # noqa: E402
from cineforge.audio_mix import (  # noqa: E402
    BROADCAST_LINEAR,
    INTERNAL_REVIEW_STEREO,
    MixError,
    MixReport,
    mix_scene,
)
from cineforge.audio_pcm import inspect_pcm16_wav  # noqa: E402
from cineforge.audio_process import run_audio_process  # noqa: E402


class NativeMixTests(unittest.TestCase):
    reports = []
    executable = Path(os.environ.get("CINEFORGE_TEST_FFMPEG", "/opt/homebrew/bin/ffmpeg"))

    def mix(self, args, preset):
        tool = FFmpegTool(self.executable)
        identity = tool.identify(timeout_seconds=10, max_output_bytes=65536)
        self.assertEqual(identity.version, "9.0.2")
        self.assertEqual(identity.kind, "native")
        result = mix_scene(**args, preset=preset, output_id="native-mix", output_revision=1,
                           tool=tool, expected_tool=identity)
        self.assertEqual(result.report.delivery_status, "native_measured_pcm")
        self.assertEqual(result.report.delivered.tool, identity)
        self.assertEqual(result.report.output.sha256, hashlib.sha256(result.payload).hexdigest())
        self.assertAlmostEqual(result.report.delivered.integrated_lufs.value,
                               preset.target_lufs, delta=preset.loudness_tolerance_lu)
        self.assertLessEqual(result.report.delivered.true_peak_dbtp.value, preset.max_true_peak_dbtp)
        self.assertEqual(MixReport.from_json(result.report.to_json()), result.report)
        raw = run_audio_process((str(self.executable), "-hide_banner", "-nostdin", "-nostats",
            "-v", "info", "-protocol_whitelist", "pipe", "-f", "wav", "-i", "pipe:0",
            "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "pipe:1"),
            payload=result.payload, timeout_seconds=10, max_output_bytes=65536).decode()
        loudness = float(re.findall(r"I:\s*(-?\d+\.\d+) LUFS", raw)[-1])
        peak = float(re.findall(r"True peak:\s*Peak:\s*(-?\d+\.\d+) dBFS", raw)[-1])
        self.assertAlmostEqual(loudness, preset.target_lufs, delta=.2)
        self.assertLessEqual(peak, preset.max_true_peak_dbtp + .3)
        self.reports.append(dict(preset=preset.preset_id, sha256=result.report.output.sha256,
            loudness=result.report.delivered.integrated_lufs.value,
            true_peak=result.report.delivered.true_peak_dbtp.value,
            independent_ebur128_lufs=loudness, independent_ebur128_peak=peak,
            tool=identity.to_dict(), safety_gain=result.report.safety_gain))
        return result

    def test_actual_broadcast_and_project_review_stereo_presets(self):
        args = scene(background=2000, dialogue=5000, frequency=1000)
        for preset in (BROADCAST_LINEAR, INTERNAL_REVIEW_STEREO):
            with self.subTest(preset=preset.preset_id):
                result = self.mix(args, preset)
                self.assertEqual(inspect_pcm16_wav(result.payload).frames, 24000)
                self.assertEqual(result.report.human_review, "not_assessed")

    def test_actual_overlapping_summed_clipping_prevention_and_peak_verification(self):
        args = scene(background=30000, dialogue=30000, frequency=1000)
        args["settings"] = replace(args["settings"], duck_db=0)
        result = self.mix(args, BROADCAST_LINEAR)
        self.assertLess(result.report.safety_gain, 1)
        self.assertLess(max(abs(item[0]) for item in struct.iter_unpack(
            "<h", inspect_pcm16_wav(result.payload).pcm)), 32767)

    def test_native_silence_short_and_excessive_crest_factor_refuse_honestly(self):
        pulse = bytearray(pcm(value=0))
        struct.pack_into("<h", pulse, 44 + 12000 * 2, 32000)
        cases = ((bed_scene(pcm(value=0)), "silent"),
                 (bed_scene(pcm(frames=4800), duration_ticks=200), "shorter"),
                 (bed_scene(bytes(pulse)), "unattainable"))
        for args, reason in cases:
            with self.subTest(reason=reason):
                tool = FFmpegTool(self.executable)
                identity = tool.identify(timeout_seconds=10, max_output_bytes=65536)
                with self.assertRaises(MixError) as caught:
                    mix_scene(**args, preset=BROADCAST_LINEAR, output_id="rejected",
                        output_revision=1, tool=tool, expected_tool=identity)
                self.assertIn(reason, str(caught.exception))

    @classmethod
    def tearDownClass(cls):
        print(json.dumps({"native_mix_fixtures": cls.reports}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
