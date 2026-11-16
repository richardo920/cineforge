"""Synthetic process outputs exercise mux wiring/refusals; no native binaries are run."""
from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cineforge.a20_delivery import mux_delivery
from cineforge.a20_inputs import A20Error
from cineforge.audio_analysis import ToolIdentity
from cineforge.private_store import PrivateRunDirectory

FACTS = {"streams": [
    {"codec_type": "video", "codec_name": "vp9", "width": 832, "height": 480,
     "pix_fmt": "yuv420p", "nb_read_frames": "77", "avg_frame_rate": "16/1"},
    {"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "24000", "channels": 2}],
    "format": {"duration": "4.813"}}
IDENTITY = ToolIdentity("9.0.2", "a" * 64, "b" * 64)


class DeliveryTests(unittest.TestCase):
    def exercise(self, facts=None, failure=None, identity=IDENTITY, deadline=None):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve()
            video = root / "video.webm"
            video.write_bytes(b"synthetic video")
            mix = NS(payload=b"synthetic PCM", pin=NS(to_dict=lambda: {"sha256": "c" * 64}),
                report=NS(delivery_status="native_measured_pcm", settings=NS(sample_rate=24000, channels=2),
                          to_dict=lambda: {"output": "synthetic mix"}))
            calls = []
            def process(command, **kwargs):
                calls.append((command, kwargs))
                self.assertGreater(kwargs["timeout_seconds"], 0)
                self.assertLessEqual(kwargs["timeout_seconds"], 30)
                self.assertNotIn("-shortest", command)
                if command[0].endswith("ffmpeg"):
                    if failure:
                        raise ValueError("private process error")
                    self.assertEqual(command[command.index("-c:v") + 1], "copy")
                    self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
                    Path(command[-1]).write_bytes(b"synthetic mux")
                    return b""
                return json.dumps(facts or FACTS).encode()
            stack.enter_context(patch("cineforge.a20_delivery.run_audio_process", process))
            tool = Mock()
            tool.identify.return_value = identity
            stack.enter_context(patch("cineforge.a20_delivery.FFmpegTool", return_value=tool))
            with PrivateRunDirectory(root / "delivery") as store:
                result = mux_delivery(video, mix, store, ffmpeg=root / "ffmpeg", ffprobe=root / "ffprobe",
                    expected_tool=IDENTITY, deadline=deadline or time.monotonic() + 30)
                self.assertEqual((store.path / result["clip"]).read_bytes(), b"synthetic mux")
                self.assertEqual(result["human_review"], "pending_human_review")
                self.assertEqual(result["export_authority"], "none")
            self.assertEqual(len(calls), 2)
            self.assertEqual(tool.identify.call_count, 2)
            return result

    def test_lossless_mux_persists_technical_and_lineage_evidence(self):
        result = self.exercise()
        self.assertEqual(result["technical"], FACTS)
        self.assertEqual(result["mix"], {"sha256": "c" * 64})

    def test_refuses_missing_extra_or_malformed_media_streams(self):
        for field, value in (("nb_read_frames", "76"), ("codec_name", "h264"),
                             ("width", 640), ("avg_frame_rate", "30/1")):
            facts = copy.deepcopy(FACTS)
            facts["streams"][0][field] = value
            with self.subTest(field=field), self.assertRaises(A20Error):
                self.exercise(facts)
        for streams in (FACTS["streams"][:1], FACTS["streams"] * 2):
            with self.assertRaises(A20Error):
                self.exercise(dict(FACTS, streams=streams))
        facts = copy.deepcopy(FACTS)
        facts["streams"][1]["sample_rate"] = "48000"
        with self.assertRaises(A20Error):
            self.exercise(facts)
        facts["format"]["duration"] = "nan"
        with self.assertRaises(A20Error):
            self.exercise(facts)

    def test_refuses_wrong_tool_expired_cap_and_mux_failure(self):
        with self.assertRaises(A20Error):
            self.exercise(identity=ToolIdentity("8.0", "a" * 64, "b" * 64))
        with self.assertRaises(A20Error):
            self.exercise(deadline=time.monotonic() - 1)
        with self.assertRaises(ValueError):
            self.exercise(failure=True)
