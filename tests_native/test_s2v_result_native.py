"""Generated colour WebM tests decoding only; no model/actor/lip-quality evidence."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_s2v_run as run_fixtures  # noqa: E402

from cineforge.audio_process import run_audio_process  # noqa: E402
from cineforge.s2v_run import inspect_s2v_media  # noqa: E402


class NativeS2VMediaContractTests(unittest.TestCase):
    def test_generated_77frame_vp9_is_decoded_and_paired_without_lip_quality_claim(self):
        executable = Path(os.environ.get("CINEFORGE_TEST_FFMPEG", "/opt/homebrew/bin/ffmpeg"))
        fixture = run_fixtures.S2VRunTests("test_success_pins_actual_fixture_bytes_and_all_lineage_without_human_authority")
        fixture.setUp()
        try:
            video = fixture.output.parent / "synthetic.webm"
            run_audio_process((str(executable), "-hide_banner", "-nostdin", "-nostats",
                "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=832x480:r=16:d=4.8125",
                "-frames:v", "77", "-an", "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
                "-deadline", "realtime", "-cpu-used", "8", "-f", "webm", str(video)),
                payload=b"", timeout_seconds=15, max_output_bytes=1024 * 1024)
            payload = video.read_bytes()
            self.assertTrue(payload.startswith(b"\x1a\x45\xdf\xa3"))
            fixture.wire.payload = payload
            result = fixture.run_scene(inspect_media=inspect_s2v_media)
            self.assertEqual(result["technical_result"]["video"]["decoded_frames"], 77)
            self.assertEqual(result["technical_result"]["video"]["fps"], 16)
            self.assertEqual(result["technical_result"]["video"]["codec"], "vp9")
            self.assertEqual(result["technical_result"]["visible_lip_sync"], "not_assessed")
            self.assertEqual(result["human_review"], "pending_human_review")
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
