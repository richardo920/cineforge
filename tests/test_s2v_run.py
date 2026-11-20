"""Truthful one-shot state, paired lineage and bounded fake Comfy failure acceptance."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_comfy_runtime as comfy_fixtures
import test_s2v_stage as stage_fixtures
from test_cloud_budget import Clock
from test_comfy_run import IDENTITY, PAYLOAD, WireFixture
from test_s2v import worker

from cineforge import comfy_precision as precision
from cineforge import s2v_profile as profile
from cineforge.comfy_run import execute_comfy
from cineforge.native_remote_plan import nr_digest
from cineforge.private_store import PrivateRunDirectory
from cineforge.s2v import S2VError
from cineforge.s2v_run import inspect_s2v_media, run_s2v
from cineforge.ssh_worker import SupervisedComfyTransport
from cineforge.ssh_worker import run_process as local_process


def media(_):
    return {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
            "width": 832, "height": 480, "decoded_frames": 77, "fps": 16.0,
            "duration_seconds": profile.DURATION}


class S2VWire(WireFixture):
    def __init__(self, selection):
        super().__init__(selection)
        self.reject_audio = False
        self.combo_audio_specs = False
        self.malformed_combo_audio_options = None
        self.fail_view = False
        self.payload = PAYLOAD
        self.execution_error = False

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
            if self.combo_audio_specs and kind in ("AudioEncoderLoader", "LoadAudio"):
                field = "audio" if kind == "LoadAudio" else "audio_encoder_name"
                options = {"multiselect": False, "options": fields[field][0]}
                if kind == "LoadAudio":
                    options["audio_upload"] = True
                    if self.malformed_combo_audio_options == "not_list":
                        options["options"] = "unapproved.wav"
                    elif self.malformed_combo_audio_options == "missing":
                        del options["options"]
                fields[field] = ["COMBO", options]
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
                if self.execution_error:
                    item = value[IDENTITY]
                    item["status"] = {"status_str": "error", "completed": False,
                        "messages": [["execution_error", {"node_id": "168",
                            "exception_type": "SyntheticExecutionError",
                            "exception_message": "synthetic failure"}]]}
                    item["node_errors"] = {"168": {"message": "synthetic node error"}}
                value[IDENTITY]["outputs"]["168"] = value[IDENTITY]["outputs"].pop("47")
                value[IDENTITY]["prompt"][4] = ["168"]
            return json.dumps(value).encode()
        return reply


class S2VInspectionBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "local ffmpeg required")
    def test_explicit_probe_inspects_real_vp9_without_path_lookup(self):
        ffmpeg, ffprobe = Path(shutil.which("ffmpeg")), Path(shutil.which("ffprobe"))
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.webm"
            local_process([str(ffmpeg), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
                "color=black:s=832x480:r=16", "-frames:v", "77", "-c:v", "libvpx-vp9",
                "-lossless", "1", "-speed", "8", "-threads", "2", str(clip)], timeout=30)
            with patch.dict(os.environ, {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}), \
                    patch("cineforge.s2v_run.shutil.which", return_value=None) as which:
                result = inspect_s2v_media(clip, ffprobe=ffprobe)
                which.assert_not_called()
                self.assertEqual(result, dict(media(clip), duration_seconds=4.813))
                with self.assertRaises(S2VError):
                    inspect_s2v_media(clip)
                which.assert_called_once_with("ffprobe")

    def test_direct_inspector_rejects_metadata_returned_after_actual_probe_finally(self):
        now, events = [100.0], []
        payload = json.dumps({"streams": [{"codec_type": "video", "codec_name": "vp9",
            "pix_fmt": "yuv420p", "width": 832, "height": 480, "nb_read_frames": "77",
            "avg_frame_rate": "16/1"}], "format": {"duration": "4.8125"}}).encode()
        def process(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 3)
            try:
                return payload
            finally:
                events.append("probe process cleanup completed")
                now[0] = 104
        with patch("cineforge.s2v_run.shutil.which", return_value="/synthetic/ffprobe"), \
                patch("cineforge.s2v_run.run_process", side_effect=process):
            with self.assertRaises(S2VError) as caught:
                inspect_s2v_media(Path("/synthetic/clip.webm"),
                                 absolute_deadline=103, clock=lambda: now[0])
        self.assertEqual(events, ["probe process cleanup completed"])
        self.assertIsNone(caught.exception.__context__)



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
        self.driver.endpoint = lambda **kwargs: "http://127.0.0.1:8188"
        self.driver.execution_transport = lambda **kwargs: self.wire
        self.output = self.fixture.root / "output"

    def run_scene(self, **changes):
        options = dict(allow_live=True, inspect_media=media)
        options.update(changes)
        if 'clock' in options:
            # A forwarded absolute deadline belongs to this same host/clock in
            # both the run and its real SSH/Comfy admission helpers.
            with patch('cineforge.ssh_worker.time.monotonic', options['clock']):
                return run_s2v(self.selection, self.verification, self.driver, self.output, **options)
        return run_s2v(self.selection, self.verification, self.driver, self.output, **options)

    def test_comfy_execution_details_reach_private_shot_record(self):
        self.wire.execution_error = True
        with self.assertRaises(S2VError):
            self.run_scene()
        record = json.loads((self.output / "shot" / "run.json").read_text())
        error = record["known_run"]["error"]
        self.assertIn("SyntheticExecutionError", error)
        self.assertIn('"node_id":"168"', error)
        self.assertIn('"node_errors"', error)

    def precision_runtime(self, mode):
        runtime, worker, _, _, state, bridge = comfy_fixtures.ComfyControllerTests().fixture()
        request = dict(version=1, policy=precision.POLICY, nonce="a" * 32,
            selection_sha256=self.selection.approved_subject_sha256,
            deadline=9999999999.0, worker_id="worker", session_id="session", profile_id="profile")
        for key in ("nonce_graph_sha256", "comfy_sha256", "runtime_sha256", "source_sha256",
                    "package_sha256", "enforcer_sha256", "binding_sha256"):
            request[key] = "a" * 64
        runtime._precision = request
        runtime._precision_posted = True
        runtime._expires_unix = bridge._expires_unix
        receipt = dict(version=1, policy=precision.POLICY, request_sha256=nr_digest(request),
            model_sha256=precision.MODEL_SHA256, header_sha256=precision.HEADER_SHA256,
            matrices_sha256=precision.MATRICES_SHA256, enforcer_sha256=request["enforcer_sha256"],
            model_nonce="b" * 32, patch_uuid=str(uuid.uuid4()), audit_count=467,
            manual_dtype="bf16", effective_dtype="bf16", step=1, matrix_calls=467,
            producer_pid=123, prompt_id=IDENTITY)
        if mode == "receipt_validation":
            receipt.update(audit_count=466, header_sha256="c" * 64)
        if mode == "unsafe_receipt":
            receipt.update(audit_count=float("nan"), model_sha256="x" * 10000,
                model_nonce="\ud800", header_sha256=["not scalar"], effective_dtype=2**1000)
        if mode == "nonobject_receipt":
            receipt = []
        if mode == "admit":
            state["revoked"] = True
        def rpc(operation, **kwargs):
            self.assertEqual(operation, "native_comfy_precision_finish")
            if mode == "finish_rpc":
                raise TimeoutError("synthetic RPC failure")
            if mode == "interrupt":
                raise KeyboardInterrupt
            return receipt
        bridge._call = rpc
        closed = self.driver._closed_comfy
        def finish(driver, selection, prompt_id, **kwargs):
            try:
                return runtime.finish_precision(worker, selection, prompt_id, **kwargs)
            finally:
                closed._precision_finish_diagnostic = runtime._precision_finish_diagnostic
        closed.finish_precision = finish
        return runtime, bridge, receipt

    def test_precision_finish_rpc_failure_is_checkpointed(self):
        runtime, bridge, _ = self.precision_runtime("finish_rpc")
        with self.assertRaises(S2VError):
            self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {"stage": "finish_rpc", "error_type": "TimeoutError"})
        self.assertEqual(record["outcome"], "incomplete")
        self.assertIsNone(runtime._precision)
        self.assertTrue(runtime._unresolved and bridge._unresolved)

    def test_precision_receipt_failure_retains_received_fields(self):
        _, _, receipt = self.precision_runtime("receipt_validation")
        receipt["extra_key"] = "synthetic"
        receipt["manual_dtype"] = None
        del receipt["effective_dtype"]
        with self.assertRaises(S2VError):
            self.run_scene()
        diagnostic = json.loads((self.output / "run.json").read_text())["precision_finish_failure"]
        self.assertEqual(diagnostic["stage"], "receipt_validation")
        self.assertEqual(diagnostic["receipt"]["audit_count"], 466)
        self.assertEqual(diagnostic["receipt"]["header_sha256"], "c" * 64)
        self.assertEqual(diagnostic["receipt_keys"], sorted(receipt))
        self.assertIsNone(diagnostic["receipt"]["manual_dtype"])
        self.assertNotIn("effective_dtype", diagnostic["receipt"])
        for key in ("version", "policy", "patch_uuid", "step", "matrix_calls", "producer_pid"):
            self.assertEqual(diagnostic["receipt"][key], receipt[key])
        for key, value in diagnostic["receipt"].items():
            self.assertEqual(value, receipt[key])

    def test_precision_receipt_diagnostic_is_bounded_and_canonical_safe(self):
        self.precision_runtime("unsafe_receipt")
        with self.assertRaises(S2VError):
            self.run_scene()
        diagnostic = json.loads((self.output / "run.json").read_text())["precision_finish_failure"]
        self.assertEqual(diagnostic["stage"], "receipt_validation")
        self.assertEqual(len(diagnostic["receipt"]["model_sha256"]), 256)
        self.assertNotIn("audit_count", diagnostic["receipt"])
        self.assertNotIn("header_sha256", diagnostic["receipt"])
        self.assertNotIn("effective_dtype", diagnostic["receipt"])

    def test_precision_nonobject_receipt_records_type(self):
        self.precision_runtime("nonobject_receipt")
        with self.assertRaises(S2VError):
            self.run_scene()
        diagnostic = json.loads((self.output / "run.json").read_text())["precision_finish_failure"]
        self.assertEqual(diagnostic, {"stage": "receipt_validation",
                                      "receipt_type": "list", "error_type": "ValueError"})

    def test_diagnostic_capture_failure_cannot_replace_rpc_interrupt(self):
        runtime, bridge, _ = self.precision_runtime("interrupt")
        with patch("cineforge.comfy_runtime._finish_diagnostic", side_effect=RuntimeError):
            with self.assertRaises(KeyboardInterrupt):
                self.run_scene()
        self.assertTrue(runtime._unresolved and bridge._unresolved)

    def test_interrupt_during_diagnostic_capture_is_preserved(self):
        runtime, bridge, _ = self.precision_runtime("finish_rpc")
        with patch("cineforge.comfy_runtime._finish_diagnostic", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_scene()
        self.assertTrue(runtime._unresolved and bridge._unresolved)

    def test_precision_internal_admit_failure_is_checkpointed(self):
        self.precision_runtime("admit")
        with self.assertRaises(S2VError):
            self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {"stage": "entry_admit", "error_type": "ValueError"})

    def test_precision_prefinish_admit_failure_is_checkpointed(self):
        original = self.driver.verify_comfy_current
        def verify(**kwargs):
            path = self.output / "run.json"
            if path.exists() and "manifest" in json.loads(path.read_text()):
                raise ValueError("synthetic revoked admission")
            return original(**kwargs)
        self.driver.verify_comfy_current = verify
        with self.assertRaises(S2VError):
            self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {"stage": "pre_finish_admit"})

    def test_precision_intent_guard_failure_is_checkpointed(self):
        runtime, _, _ = self.precision_runtime("success")
        runtime._precision_posted = False
        with self.assertRaises(S2VError):
            self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {"stage": "intent_guard"})

    def test_precision_job_identity_failure_is_checkpointed(self):
        def execute(*args, **kwargs):
            return replace(execute_comfy(*args, **kwargs), shot_runs=(), artifacts=())
        with self.assertRaises(S2VError):
            self.run_scene(execute=execute)
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {"stage": "job_identity"})

    def test_precision_prompt_identity_failure_is_checkpointed(self):
        _, _, receipt = self.precision_runtime("success")
        received = str(uuid.uuid4())
        receipt["prompt_id"] = received
        with self.assertRaises(S2VError):
            self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {
            "stage": "prompt_identity", "error_type": "ValueError",
            "received_prompt_id": received, "expected_prompt_id": IDENTITY})

    def test_precision_deadline_failure_is_checkpointed(self):
        runtime, _, _ = self.precision_runtime("success")
        original = runtime._admit
        calls = []
        def admit(worker, deadline):
            plan = original(worker, deadline)
            calls.append(deadline)
            if len(calls) == 2:
                expired.return_value = deadline
            return plan
        with patch.object(runtime, "_admit", side_effect=admit), \
                patch("cineforge.comfy_runtime.time.monotonic", return_value=100) as expired:
            with self.assertRaises(S2VError):
                self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(len(calls), 2)
        self.assertEqual(record["precision_finish_failure"], {
            "stage": "deadline", "error_type": "ValueError"})

    def test_real_precision_success_has_no_failure_key(self):
        self.precision_runtime("success")
        result = self.run_scene()
        self.assertNotIn("precision_finish_failure", result)
        durable = json.loads((self.output / "run.json").read_text())
        self.assertNotIn("precision_finish_failure", durable)
        self.assertEqual(result["model_precision"]["audit_count"], 467)

    def test_precision_rpc_interrupt_still_propagates_and_checkpoints(self):
        runtime, bridge, _ = self.precision_runtime("interrupt")
        with self.assertRaises(KeyboardInterrupt):
            self.run_scene()
        record = json.loads((self.output / "run.json").read_text())
        self.assertEqual(record["precision_finish_failure"], {"stage": "finish_rpc", "error_type": "KeyboardInterrupt"})
        self.assertTrue(runtime._unresolved and bridge._unresolved)

    def test_explicit_probe_flows_through_run_and_technical_result_with_deadline(self):
        binary = self.fixture.root / "explicit-ffprobe"
        payload = json.dumps({"streams": [{"codec_type": "video", "codec_name": "vp9",
            "pix_fmt": "yuv420p", "width": 832, "height": 480, "nb_read_frames": "77",
            "avg_frame_rate": "16/1"}], "format": {"duration": "4.813"}}).encode()
        with patch("cineforge.s2v_run.shutil.which", return_value=None) as which, \
                patch("cineforge.s2v_run.run_process", return_value=payload) as probe:
            result = self.run_scene(inspect_media=inspect_s2v_media, ffprobe=binary)
        which.assert_not_called()
        self.assertEqual(probe.call_args.args[0][0], str(binary))
        self.assertLessEqual(probe.call_args.kwargs["absolute_deadline"], result["controller_deadline"])
        self.assertEqual(result["technical_result"]["video"], dict(media(None), duration_seconds=4.813))

    def test_default_probe_still_uses_path_lookup(self):
        with patch("cineforge.s2v_run.shutil.which", return_value=None) as which:
            with self.assertRaises(S2VError):
                self.run_scene(inspect_media=inspect_s2v_media)
        which.assert_called_once_with("ffprobe")

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
        durable = json.loads((self.output / "run.json").read_text())
        self.assertEqual((durable["outcome"], durable["human_review"]), ("generated", "unavailable"))
        self.assertEqual(durable["technical_candidate"], result["technical_result"])
        self.assertNotIn("technical_result", durable)
        self.assertNotIn("technical_candidate", result)
        historical = dict(result, outcome="generated", human_review="unavailable")
        historical["technical_candidate"] = historical.pop("technical_result")
        self.assertEqual(durable, historical)
        new = self.fixture.root / "duplicate"
        with self.assertRaises(S2VError):
            run_s2v(self.selection, self.verification, self.driver, new,
                    allow_live=True, inspect_media=media)
        self.assertFalse(new.exists())
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)

    def test_combo_audio_specs_accept_approved_files_and_submit_once(self):
        self.wire.combo_audio_specs = True
        result = self.run_scene()
        self.assertEqual(result["outcome"], "pending_human_review")
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)

    def test_combo_audio_spec_without_approved_file_refuses_before_submit(self):
        self.wire.combo_audio_specs = True
        self.wire.reject_audio = True
        with self.assertRaises(S2VError):
            self.run_scene()
        self.assertFalse(any(call[1] == "/prompt" for call in self.wire.calls))
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertEqual(inner["failure_class"], "verify_nodes_file")

    def test_malformed_combo_audio_spec_fails_closed_before_submit(self):
        self.wire.combo_audio_specs = True
        self.wire.malformed_combo_audio_options = "not_list"
        with self.assertRaises(S2VError):
            self.run_scene()
        self.assertFalse(any(call[1] == "/prompt" for call in self.wire.calls))
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertEqual(inner["failure_class"], "verify_nodes_file")

    def test_combo_audio_spec_with_missing_options_fails_closed_before_submit(self):
        self.wire.combo_audio_specs = True
        self.wire.malformed_combo_audio_options = "missing"
        with self.assertRaises(S2VError):
            self.run_scene()
        self.assertFalse(any(call[1] == "/prompt" for call in self.wire.calls))
        inner = json.loads((self.output / "shot/run.json").read_text())
        self.assertEqual(inner["outcome"], "incomplete")
        self.assertEqual(inner["failure_class"], "verify_nodes_file")

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
            options['max_polls'] = 1
            return execute_comfy(*args, poll_interval_seconds=0, **options)
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
        self.driver.execution_transport = lambda **kwargs: SupervisedComfyTransport(self.driver, self.wire)
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
            value = json.loads(payload)
            if value.get("outcome") == "generated" and "technical_candidate" in value:
                late.append(now[0])
                now[0] = 161
        with patch.object(PrivateRunDirectory, "checkpoint", checkpoint), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(late, [100])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("incomplete", "unavailable"))
        self.assertIn("manifest", outer)
        self.assertNotIn("technical_result", outer)

    def test_real_outer_directory_close_then_expiry_keeps_durable_candidate_unavailable(self):
        now = self._absolute_fixture()
        original, closed = PrivateRunDirectory.__exit__, []
        def close(store, *args):
            original(store, *args)  # Real descriptor close, not a mocked cleanup.
            closed.append((store.path, store._directory))
            if store.path == self.output:
                now[0] = 161
        with patch.object(PrivateRunDirectory, "__exit__", close), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(closed, [(self.output / "shot", -1), (self.output, -1)])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("generated", "unavailable"))
        self.assertIn("technical_candidate", outer)
        self.assertNotIn("technical_result", outer)
        self.assertEqual(outer["manifest"]["shot_runs"][0]["status"], "succeeded")
        self.assertEqual(self.driver.budget.remaining_seconds("lease"), 119)
        self.assertEqual(sum(call[1] == "/prompt" for call in self.wire.calls), 1)

    def test_real_outer_close_failure_preserves_generated_evidence_without_readiness(self):
        now = self._absolute_fixture()
        original = PrivateRunDirectory.__exit__
        def close(store, *args):
            original(store, *args)
            if store.path == self.output:
                raise OSError("CANARY final close failure")
        with patch.object(PrivateRunDirectory, "__exit__", close), self.assertRaises(S2VError) as caught:
            self.run_scene(clock=lambda: now[0])
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("CANARY", str(caught.exception))
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("generated", "unavailable"))
        self.assertIn("manifest", outer)
        self.assertIn("technical_candidate", outer)
        self.assertNotIn("technical_result", outer)

    def test_fresh_post_close_proof_delay_cannot_finalize_cached_technical_candidate(self):
        now = self._absolute_fixture()
        original, closed, delayed = PrivateRunDirectory.__exit__, [False], []
        native = self.selection.gate.verify_speech
        def proof(*args):
            value = native(*args)
            if closed[0] and not delayed:
                delayed.append(now[0])
                now[0] = 161
            return value
        self.selection = replace(self.selection, gate=replace(self.selection.gate, verify_speech=proof))
        def close(store, *args):
            original(store, *args)
            if store.path == self.output:
                closed[0] = True
        with patch.object(PrivateRunDirectory, "__exit__", close), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(delayed, [100])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("generated", "unavailable"))
        self.assertIn("technical_candidate", outer)

    def test_fresh_post_close_rights_revocation_blocks_finalized_return(self):
        now = self._absolute_fixture()
        original, closed = PrivateRunDirectory.__exit__, [False]
        native = self.selection.gate.verify_speech
        def proof(*args):
            value = native(*args)
            if closed[0]:
                self.selection.gate.ledger.revoke(self.selection.gate.utterance.voice.permission.rights,
                                                 reason="CANARY after cleanup")
            return value
        self.selection = replace(self.selection, gate=replace(self.selection.gate, verify_speech=proof))
        def close(store, *args):
            original(store, *args)
            if store.path == self.output:
                closed[0] = True
        with patch.object(PrivateRunDirectory, "__exit__", close), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("generated", "unavailable"))
        self.assertIn("manifest", outer)

    def test_final_candidate_copy_latency_is_before_last_original_cap_checkout(self):
        from cineforge.workflows import strict_object
        now = self._absolute_fixture()
        original, closed = PrivateRunDirectory.__exit__, [False]
        def close(store, *args):
            original(store, *args)
            if store.path == self.output:
                closed[0] = True
        def copy(raw):
            value = strict_object(raw)
            if closed[0] and "technical_candidate" in value:
                now[0] = 161
            return value
        with patch.object(PrivateRunDirectory, "__exit__", close), \
                patch("cineforge.s2v_run.strict_object", side_effect=copy), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("generated", "unavailable"))

    def test_failed_late_checkpoint_correction_never_leaves_durable_readiness(self):
        now = self._absolute_fixture()
        original, candidate_written = PrivateRunDirectory.checkpoint, []
        def checkpoint(store, payload):
            value = json.loads(payload)
            if value.get("outcome") == "incomplete":
                raise OSError("CANARY correction unavailable")
            original(store, payload)
            if "technical_candidate" in value:
                candidate_written.append(now[0])
                now[0] = 161
        with patch.object(PrivateRunDirectory, "checkpoint", checkpoint), self.assertRaises(S2VError):
            self.run_scene(clock=lambda: now[0])
        self.assertEqual(candidate_written, [100])
        outer = json.loads((self.output / "run.json").read_text())
        self.assertEqual((outer["outcome"], outer["human_review"]), ("generated", "unavailable"))
        self.assertIn("technical_candidate", outer)
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
