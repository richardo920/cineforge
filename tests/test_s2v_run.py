"""Truthful one-shot state, paired lineage and bounded fake Comfy failure acceptance."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_s2v_stage as stage_fixtures
from test_cloud_budget import Clock
from test_comfy_run import IDENTITY, PAYLOAD, WireFixture
from test_s2v import worker

from cineforge import s2v_profile as profile
from cineforge.comfy_run import execute_comfy
from cineforge.private_store import PrivateRunDirectory
from cineforge.s2v import S2VError
from cineforge.s2v_run import run_s2v
from cineforge.ssh_worker import SupervisedComfyTransport


def media(_):
    return {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
            "width": 832, "height": 480, "decoded_frames": 77, "fps": 16.0,
            "duration_seconds": profile.DURATION}


class S2VWire(WireFixture):
    def __init__(self, selection):
        super().__init__(selection)
        self.reject_audio = False
        self.fail_view = False
        self.payload = PAYLOAD

    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        if path.startswith("/object_info/"):
            self.calls.append((method, path, copy.deepcopy(body), timeout_seconds, max_bytes))
            kind = path.rsplit("/", 1)[1]
            fields = {}
            for node in self.template.values():
                if node["class_type"] == kind:
                    for field, value in node["inputs"].items():
                        if field in ("unet_name", "clip_name", "vae_name", "audio_encoder_name",
                                     "image", "audio"):
                            fields[field] = [[value]]
                        else:
                            fields[field] = ["SYNTHETIC"]
            if self.reject_audio and kind == "LoadAudio":
                fields["audio"] = [["unapproved.wav"]]
            count = 3 if kind == "WanSoundImageToVideo" else 1
            return json.dumps({kind: {"input": {"required": fields},
                "output": ["SYNTHETIC"] * count, "output_node": kind == "SaveWEBM"}}).encode()
        if path.startswith("/view?"):
            self.calls.append((method, path, copy.deepcopy(body), timeout_seconds, max_bytes))
            if self.fail_view:
                raise RuntimeError("CANARY transfer failed")
            return self.payload
        reply = super().request(method, path, body, timeout_seconds=timeout_seconds, max_bytes=max_bytes)
        if path.startswith("/history/"):
            value = json.loads(reply)
            if value:
                value[IDENTITY]["outputs"]["168"] = value[IDENTITY]["outputs"].pop("47")
                value[IDENTITY]["prompt"][4] = ["168"]
            return json.dumps(value).encode()
        return reply


class S2VRunTests(unittest.TestCase):
    def setUp(self):
        self.fixture = stage_fixtures.PairTransferTests(
            "test_complete_pair_exact_bytes_metadata_and_consumed_repeated_request")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.driver = self.fixture.driver
        proof = self.driver.stage_s2v(self.fixture.selection)
        self.verification = worker(self.fixture.gate)
        self.selection = self.fixture.selection.bind_worker(self.verification, proof)
        self.wire = S2VWire(self.selection)
        self.driver.endpoint = lambda: "http://127.0.0.1:8188"
        self.driver.execution_transport = lambda: self.wire
        self.output = self.fixture.root / "output"

    def run_scene(self, **changes):
        options = dict(allow_live=True, inspect_media=media)
        options.update(changes)
        return run_s2v(self.selection, self.verification, self.driver, self.output, **options)

    def test_success_pins_actual_fixture_bytes_and_all_lineage_without_human_authority(self):
        result = self.run_scene()
        self.assertEqual(result["outcome"], "pending_human_review")
        self.assertEqual(result["visible_lip_sync"], "not_assessed")
        self.assertEqual(result["export_authority"], "none")
        self.assertEqual(result["lineage"]["speech"], self.selection.gate.speech.pin.to_dict())
        measured = result["technical_result"]
        self.assertEqual(measured["video"]["decoded_frames"], 77)
        self.assertEqual(measured["speech_frames"], self.selection.gate.speech.frames)
        artifact = result["manifest"]["artifacts"][0]
        self.assertEqual((self.output / "shot" / artifact["uri"]).read_bytes(), PAYLOAD)
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)
        self.assertEqual(json.loads((self.output / "run.json").read_text()), result)
        new = self.fixture.root / "duplicate"
        with self.assertRaises(S2VError):
            run_s2v(self.selection, self.verification, self.driver, new,
                    allow_live=True, inspect_media=media)
        self.assertFalse(new.exists())
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)

    def test_missing_optin_stale_rights_foreign_worker_or_missing_stage_make_no_calls(self):
        with self.assertRaises(S2VError):
            self.run_scene(allow_live=False)
        self.assertEqual(self.wire.calls, [])
        self.assertFalse(self.output.exists())
        self.selection.gate.ledger.revoke(self.selection.gate.model_rights, reason="CANARY")
        with self.assertRaises(S2VError):
            self.run_scene()
        self.assertEqual(self.wire.calls, [])
        self.assertFalse(self.output.exists())

    def test_unavailable_audio_node_file_refuses_before_submit_and_keeps_honest_intent(self):
        self.wire.reject_audio = True
        with self.assertRaises(S2VError):
            self.run_scene()
        self.assertFalse(any(call[1] == "/prompt" for call in self.wire.calls))
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertIsNone(inner["known_run"])
        self.assertFalse(inner["cancellation"]["attempted"])

    def test_lost_submit_ack_keeps_uncertainty_and_does_not_retry_or_invent_cancelled(self):
        def fail():
            raise RuntimeError("CANARY acknowledgement lost")
        self.wire.on_submit = fail
        with self.assertRaises(S2VError):
            self.run_scene()
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertIsNone(inner["known_run"])
        self.assertFalse(inner["cancellation"]["attempted"])
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)
        self.assertNotIn("CANARY", (self.output / "run.json").read_text())
        with self.assertRaises(S2VError):
            self.run_scene()
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)

    def test_known_running_job_uses_only_owned_atomic_cancel(self):
        self.wire.mode = "running"
        def execute(*args, **options):
            return execute_comfy(*args, max_polls=1, poll_interval_seconds=0, **options)
        with self.assertRaises(S2VError):
            self.run_scene(execute=execute)
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["known_run"]["job_id"], IDENTITY)
        self.assertEqual(inner["known_run"]["status"], "cancelled")
        self.assertEqual(inner["cancellation"], {"attempted": True, "confirmed_terminal": True})
        self.assertEqual([call[1] for call in self.wire.calls if call[1].endswith("/cancel")],
                         [f"/api/jobs/{IDENTITY}/cancel"])
        self.assertFalse(any(call[1] == "/interrupt" for call in self.wire.calls))

    def test_transfer_or_decoded_contract_failure_preserves_paid_output_truthfully(self):
        self.wire.fail_view = True
        with self.assertRaises(S2VError):
            self.run_scene()
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["known_run"]["status"], "succeeded")
        self.assertIsNotNone(inner["server_output"])
        self.assertFalse(inner["cancellation"]["attempted"])
        self.assertNotIn("manifest", inner)
        self.assertNotIn("CANARY", (self.output / "run.json").read_text())

    def test_wrong_decoded_framecount_is_not_a_technical_pass_and_retains_manifest(self):
        with self.assertRaises(S2VError):
            self.run_scene(inspect_media=lambda _: dict(media(None), decoded_frames=80))
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual(outer["outcome"], "incomplete")
        self.assertIn("manifest", outer)
        self.assertNotIn("technical_result", outer)
        self.assertEqual(outer["manifest"]["shot_runs"][0]["status"], "succeeded")

    def _absolute_fixture(self):
        now = [100.0]
        self.driver.budget.remaining_seconds = lambda _: 180 - (now[0] - 100)
        self.driver.execution_transport = lambda: SupervisedComfyTransport(self.driver, self.wire)
        return now

    def test_slow_initial_native_proof_cannot_renew_run_cap_or_claim_storage(self):
        now = self._absolute_fixture()
        native, armed, slept = self.selection.gate.verify_speech, [False], [False]
        def proof(*args):
            if armed[0] and not slept[0]:
                now[0] = 161
                slept[0] = True
            return native(*args)
        self.selection = replace(self.selection, gate=replace(self.selection.gate, verify_speech=proof))
        armed[0] = True
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertTrue(slept[0])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.wire.calls, [])

    def test_slow_real_execute_preflight_keeps_original_cap_through_supervised_provider(self):
        now = self._absolute_fixture()
        native, entered, slept, caps = self.selection.gate.verify_speech, [False], [False], []
        def proof(*args):
            if entered[0] and not slept[0]:
                now[0] = 161
                slept[0] = True
            return native(*args)
        self.selection = replace(self.selection, gate=replace(self.selection.gate, verify_speech=proof))
        def execute(*args, **kwargs):
            entered[0] = True
            caps.append(kwargs["absolute_deadline"])
            return execute_comfy(*args, **kwargs)
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0], execute=execute)
        self.assertEqual(caps, [160])
        self.assertTrue(slept[0])
        self.assertEqual(self.wire.calls, [])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("incomplete", "unavailable"))
        self.assertFalse((self.output / "shot").exists())

    def test_slow_final_provider_proof_after_schema_checks_cannot_submit(self):
        now = self._absolute_fixture()
        native, delayed = self.selection.gate.verify_speech, []
        schemas = len(self.selection.workflow.allowed_classes)
        def proof(*args):
            if len(self.wire.calls) == schemas and not delayed:
                delayed.append(now[0])
                now[0] = 161
            return native(*args)
        self.selection = replace(self.selection, gate=replace(self.selection.gate, verify_speech=proof))
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(delayed, [100])
        self.assertEqual(len(self.wire.calls), schemas)
        self.assertFalse(any(call[1] == "/prompt" for call in self.wire.calls))

    def test_supervised_last_post_admission_cannot_send_after_original_cap(self):
        now = self._absolute_fixture()
        schemas = len(self.selection.workflow.allowed_classes)
        original, visits = self.driver.check, []
        def check():
            original()
            if len(self.wire.calls) == schemas:
                visits.append(now[0])
                if len(visits) == 2:  # After final schema return, then before POST.
                    now[0] = 161
        self.driver.check = check
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(len(visits), 2)
        self.assertFalse(any(call[1] == "/prompt" for call in self.wire.calls))
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertIsNone(inner["known_run"])
        self.assertFalse(inner["cancellation"]["attempted"])

    def test_late_technical_inspector_cannot_publish_readiness_and_keeps_generated_identity(self):
        now = self._absolute_fixture()
        def late(path):
            now[0] = 161
            return media(path)
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0], inspect_media=late)
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("incomplete", "unavailable"))
        self.assertNotIn("technical_result", outer)
        self.assertEqual(outer["manifest"]["shot_runs"][0]["status"], "succeeded")
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)

    def test_final_checkpoint_returning_late_is_corrected_to_incomplete_without_losing_manifest(self):
        now = self._absolute_fixture()
        original, late = PrivateRunDirectory.checkpoint, []
        def checkpoint(store, payload):
            original(store, payload)
            if json.loads(payload).get("outcome") == "pending_human_review":
                late.append(now[0])
                now[0] = 161
        with patch.object(PrivateRunDirectory, "checkpoint", checkpoint), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(late, [100])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("incomplete", "unavailable"))
        self.assertIn("manifest", outer)
        self.assertNotIn("technical_result", outer)

    def test_transport_typeerror_after_post_is_consumed_once_without_fallback(self):
        self._absolute_fixture()
        original, posts = self.wire.request, []
        def request(method, path, body=None, **limits):
            if path == "/prompt":
                posts.append(path)
                raise TypeError("CANARY transport body failure")
            return original(method, path, body, **limits)
        self.wire.request = request
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: 100)
        with self.assertRaises(S2VError):
            self.run_scene(clock=lambda: 100)
        self.assertEqual(posts, ["/prompt"])
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertIsNone(inner["known_run"])
        self.assertFalse(inner["cancellation"]["attempted"])

    def test_cancellation_cannot_extend_original_deadline_or_outer_lease_reserve(self):
        clock = Clock()
        started = clock.monotonic()
        self.driver.budget.remaining_seconds = lambda _: 180 - (clock.monotonic() - started)
        self.wire.mode = "running"
        original = self.wire.request
        histories = []
        def request(method, path, body=None, **limits):
            if path.startswith("/history/"):
                histories.append(limits["timeout_seconds"])
                clock.sleep(31 if len(histories) == 1 else limits["timeout_seconds"])
            return original(method, path, body, **limits)
        self.wire.request = request
        with self.assertRaises(S2VError):
            self.run_scene(clock=clock.monotonic)
        self.assertLessEqual(clock.monotonic() - started, 60)
        self.assertGreaterEqual(self.driver.budget.remaining_seconds("lease"), 120)
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertFalse(inner["cancellation"]["confirmed_terminal"])
        self.assertEqual(inner["known_run"]["job_id"], IDENTITY)


if __name__ == "__main__":
    unittest.main()
