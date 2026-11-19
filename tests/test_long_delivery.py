"""Synthetic local VP9/PCM acceptance fixtures; no GPUs, credentials or paid services."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import time
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_audio_mix import pcm
from test_speech import EVIDENCE, Runtime, adapter, fixture, reviewed

from cineforge.a20_inputs import A20Error
from cineforge.audio import RevisionPin, ShotTiming
from cineforge.audio_analysis import FFmpegTool
from cineforge.audio_mix import MixSettings, StemMix, mix_scene
from cineforge.audio_process import run_audio_process
from cineforge.long_delivery import mux_long_delivery, stitch_shots
from cineforge.private_store import PrivateRunDirectory
from cineforge.rights import RightsLedger, Use, with_review
from cineforge.stem_import import import_original_stem


def long_scene(count=3):
    args, actor, _ = fixture()
    old = args["sheet"]
    ticks = 77000
    lines = tuple(replace(old.script.utterances[0], utterance_id=f"line-{i}",
        shot_id=f"shot-{i}", start_ticks=i * ticks, duration_ticks=ticks) for i in range(count))
    script = replace(old.script, utterances=lines)
    dialogue = next(cue for cue in old.cues if cue.kind == "dialogue")
    cues = [replace(dialogue, cue_id=f"dialogue-{i}", utterance_id=line.utterance_id,
        shot_id=line.shot_id, start_ticks=line.start_ticks, duration_ticks=ticks)
        for i, line in enumerate(lines)]
    payload = pcm(frames=count * 115500, frequency=220, value=800)
    digest = hashlib.sha256(payload).hexdigest()
    for cue in old.cues:
        if cue.kind != "dialogue":
            cues.append(replace(cue, duration_ticks=count * ticks, source=replace(cue.source,
                subject=replace(cue.source.subject, sha256=digest))))
    sheet = replace(old, script=script, timebase=replace(old.timebase,
        ticks_per_second=16000, duration_ticks=count * ticks),
        shots=tuple(ShotTiming(f"shot-{i}", 1, i * ticks, ticks) for i in range(count)),
        cues=tuple(sorted(cues, key=lambda cue: (cue.start_ticks, cue.cue_id))))
    plan = replace(args["plan"], shots=tuple(replace(args["plan"].shots[0],
        shot_id=f"shot-{i}", duration_seconds=77 / 16) for i in range(count)))
    ledger = RightsLedger()
    for binding in {*(cue.source for cue in sheet.cues), *(ref.permission for ref in actor.references)}:
        record = reviewed(binding)
        ledger.register(with_review(replace(record, allowed_uses=(*record.allowed_uses, Use("modify"))),
                                    reviewer="fixture-reviewer", evidence=EVIDENCE))
    args.update(sheet=sheet, plan=plan, ledger=ledger, current_script=script.pin())
    sources = [import_original_stem(sheet, cue.cue_id, payload,
        RevisionPin("source-" + cue.cue_id, 1, digest), ledger=ledger, original_source_review=EVIDENCE)
        for cue in sheet.cues if cue.kind != "dialogue"]
    speech_adapter = adapter(args, runtime=Runtime(payload=pcm(frames=24000, frequency=440, value=3000)))
    sources.extend(speech_adapter.synthesize(utterance_id=line.utterance_id,
        output_id=line.utterance_id, output_revision=1, **args) for line in lines)
    keys = ("sheet", "plan", "registry", "ledger", "current_picture", "current_script")
    return dict(**{key: args[key] for key in keys}, sources=tuple(sources),
        settings=MixSettings(tuple(sorted((StemMix(stem.stem_id) for stem in sheet.stems),
            key=lambda control: control.stem_id)), timeout_seconds=180))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "local ffmpeg required")
class LongDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cf-long-tests-")
        cls.root = Path(cls.temp.name).resolve()
        cls.ffmpeg, cls.ffprobe = Path(shutil.which("ffmpeg")), Path(shutil.which("ffprobe"))
        cls.shots = []
        for index, color in enumerate(("black", "white", "red")):
            path = cls.root / f"shot-{index}.webm"
            cls.command((str(cls.ffmpeg), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
                f"color={color}:s=832x480:r=16", "-frames:v", "77", "-c:v", "libvpx-vp9",
                "-lossless", "1", "-speed", "8", "-threads", "2", str(path)))
            cls.shots.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def command(command):
        return run_audio_process(command, payload=b"", timeout_seconds=30, max_output_bytes=65536)

    def options(self):
        return dict(ffmpeg=self.ffmpeg, ffprobe=self.ffprobe, deadline=time.monotonic() + 180)

    def test_three_segments_master_review_and_qc(self):
        identity = FFmpegTool(self.ffmpeg).identify(timeout_seconds=10, max_output_bytes=65536)
        if identity.version != "9.0.2":
            self.skipTest("native measured mix requires pinned ffmpeg 9.0.2")
        captured = []
        def capture_mix(*args, **kwargs):
            mix = mix_scene(*args, **kwargs)
            captured.append(mix)
            return mix
        with PrivateRunDirectory(self.root / "delivery") as store, patch(
                "cineforge.long_delivery.mix_scene", side_effect=capture_mix):
            result = mux_long_delivery(self.shots, store=store, **long_scene(),
                output_id="long-mix", output_revision=1, expected_tool=identity, **self.options())
            self.assertTrue((store.path / result["master"]).exists())
            self.assertTrue((store.path / result["review"]).exists())
            self.assertEqual(result["technical"]["streams"][0]["nb_read_frames"], "231")
            self.assertEqual([s["codec_name"] for s in result["review_technical"]["streams"]],
                             ["h264", "aac"])
            self.assertAlmostEqual(float(result["review_technical"]["format"]["duration"]), 231 / 16, delta=.063)
            qc = json.loads((store.path / result["qc_path"]).read_bytes())
            self.assertEqual(qc, result["qc"])
            self.assertEqual(qc["total_frames"], 231)
            self.assertEqual(len(qc["boundaries"]), 2)
            self.assertEqual(qc["duration_seconds"], 231 / 16)
            self.assertAlmostEqual(qc["integrated_lufs"], -16, delta=.2)
            self.assertEqual(len(qc["dialogue_rms"]), 3)
            self.assertTrue(all(value > 0 for value in qc["dialogue_rms"]))
            self.assertIn("hard_cut_outlier", qc["boundaries"][0]["flags"])
            self.assertTrue(all("frozen_suspect" in segment["flags"] for segment in qc["segments"]))
            # Decode PCM to prove the master contains exactly the final measured mix bytes.
            sound = self.root / "extracted.wav"
            self.command((str(self.ffmpeg), "-v", "error", "-nostdin", "-n", "-i",
                str(store.path / result["master"]), "-map", "0:a:0", "-c:a", "pcm_s16le", str(sound)))
            with wave.open(str(sound)) as delivered, wave.open(io.BytesIO(captured[0].payload)) as mixed:
                self.assertEqual(delivered.getnframes(), 3 * 115500)
                self.assertEqual(delivered.readframes(delivered.getnframes()),
                                 mixed.readframes(mixed.getnframes()))
            review = (store.path / result["review"]).read_bytes()
            self.assertLess(review.index(b"moov"), review.index(b"mdat"))

    def test_short_segment_rejected_before_concat(self):
        short = self.root / "short.webm"
        self.command((str(self.ffmpeg), "-v", "error", "-nostdin", "-n", "-i", str(self.shots[0]),
            "-frames:v", "76", "-c:v", "copy", str(short)))
        with self.assertRaises(ValueError):
            stitch_shots([self.shots[0], short], **self.options())

    def test_non_identity_hook_runs_in_order_and_changes_picture(self):
        calls = []
        def transform(frames, index):
            calls.append(index)
            # Replace every luma sample: an observable non-identity transform.
            changed = bytearray(frames)
            stride = 832 * 480 * 3 // 2
            for offset in range(0, len(changed), stride):
                changed[offset:offset + 832 * 480] = bytes([64 + index * 16]) * (832 * 480)
            return bytes(changed)
        payload = stitch_shots([self.shots[0].read_bytes(), *self.shots[1:]],
                               transform=transform, **self.options())
        self.assertEqual(calls, [0, 1, 2])
        output = self.root / "hook.webm"
        output.write_bytes(payload)
        raw = self.root / "hook.gray"
        self.command((str(self.ffmpeg), "-v", "error", "-nostdin", "-n", "-i", str(output),
            "-vf", "scale=1:1", "-pix_fmt", "gray", "-f", "rawvideo", str(raw)))
        data = raw.read_bytes()
        self.assertEqual(len(data), 231)
        self.assertGreater(data[77], data[0])
        self.assertGreater(data[154], data[77])

    def test_explicit_tools_work_without_path_discovery(self):
        with patch("cineforge.s2v_run.shutil.which", return_value=None):
            payload = stitch_shots(self.shots[:1], **self.options())
        self.assertTrue(payload)

    def test_invalid_hook_output_and_empty_input_rejected(self):
        with self.assertRaises(A20Error):
            stitch_shots([], **self.options())
        with self.assertRaises(A20Error):
            stitch_shots(self.shots[:1], transform=lambda frames, index: frames[:-1], **self.options())

    def test_timeline_rejected_before_native_work(self):
        args = long_scene()
        with patch("cineforge.long_delivery.FFmpegTool") as tool, self.assertRaises(A20Error):
            mux_long_delivery(self.shots[:2], store=None, **args, output_id="mix", output_revision=1,
                              expected_tool=None, **self.options())
        tool.assert_not_called()
