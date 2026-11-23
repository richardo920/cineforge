"""Offline segment execution, durable resume and failure cleanup contracts."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_native_capabilities as native_capability_tests
from test_a20_driver import AdmissionCatalog, Harness
from test_native_remote import recipe
from test_speech import CanonicalVoiceBinding, fixture, wav_bytes

from cineforge import long_driver as driver
from cineforge.a20_inputs import A20Error, FrozenInputs
from cineforge.cloud_budget import CloudBudgetSession
from cineforge.cloud_journal import CloudJournal
from cineforge.rights import EvidenceRef
from cineforge.runpod import (
    AcquisitionIntent,
    AcquisitionUncertainError,
    CleanupUncertainError,
    LeaseSnapshot,
    ProvisioningError,
    RunpodPodSpec,
)
from cineforge.s2v_profile import IMAGE, L40S_GPU_ID
from cineforge.s2v_run import inspect_s2v_media
from cineforge.s2v_service import (
    LongSessionBudgetApproval,
    LongSessionServiceContext,
    prepare_long_session_journal,
)
from cineforge.workflows import strict_object


class LongHarness(Harness):
    def __init__(self, root, stack, lines=None):
        super().__init__(root, stack)
        paths = {k: v for k, v in self.inputs.paths.items()
                 if k in ("actor_png", "voice_design", "seed")}
        if lines is None:
            lines = ["Approved English.", "Second.", "Third."]
        for name, value in (("segments", json.dumps(lines)),
                            ("design_index", "0")):
            paths[name] = root / "data" / name
            paths[name].write_text(value)
        self.inputs = driver.LongCreativeInputs(paths)
        self.frozen = FrozenInputs.capture(self.inputs, (root / "code",))
        approval = LongSessionBudgetApproval.for_scope(str(root / "long-budget"), "EUR-IS-2",
            self.bindings.service.approval.operator_receipt, gpu_id=self.bindings.spec.gpu_id,
            phase_id="test")
        old = self.bindings
        self.bindings = driver.LongSessionBindings(LongSessionServiceContext(approval), old.spec,
            old.transport, old.ssh_private_key, old.acquisition_id, old.recipe, old.sox_wheel,
            old.design, old.select_native, old.check_comfy, self.segment, self.gate, self.checkpoint)
        for name in ("CloudBudgetSession", "SSHWorker", "RunpodPodProvisioner",
                     "RemoteNativeSpeechRuntime", "ClosedComfyRuntime", "QwenBaseSpeechAdapter", "select_s2v"):
            stack.enter_context(patch.object(driver, name, self.patches[name]))
        stack.enter_context(patch.object(driver, "prepare_long_session_journal", Mock()))
        stack.enter_context(patch.object(driver, "run_s2v", self.s2v))
        self.budget.remaining_seconds.return_value = 50000
        self.generated = []
        self.fail_segment = None
        self.adapter.synthesize.side_effect = self.speech

    def segment(self, index, reference):
        text = json.loads(self.inputs.read()["segments"])[index]
        context = NS(sheet=NS(index=index, script=NS(utterances=[NS(utterance_id="line", language="en", text=text)],
            pin=lambda: NS())), utterance_id="line", reference=NS(transcript=self.bindings.design.text),
            plan=NS(), registry=NS(), ledger=NS(), picture=NS())
        args, _, _ = fixture()
        line = args["sheet"].script.utterances[0]
        context.voices = (CanonicalVoiceBinding(line.character, line.voice),)
        return context

    def speech(self, sheet, *args, **kwargs):
        self.generated.append(sheet.index)
        return NS(payload=wav_bytes(frames=200 + sheet.index))

    def s2v(self, *args, **kwargs):
        index = int(Path(args[3]).name.split("-")[-1])
        if index == self.fail_segment:
            raise RuntimeError("synthetic S2V failure")
        payload = b"verified synthetic video " + str(index).encode()
        sha = hashlib.sha256(payload).hexdigest()
        name = "artifact-" + sha + ".webm"
        target = Path(args[3]) / "shot"
        target.mkdir(parents=True)
        (target / name).write_bytes(payload)
        return {"outcome": "pending_human_review", "technical_result": {"video_sha256": sha},
                "manifest": {"artifacts": [{"uri": name, "sha256": sha}]}}

    def run(self, deadline=None, planned_segments=None):
        return driver.run_long_session(self.bindings, self.frozen, self.output,
            absolute_deadline=deadline or driver.time.monotonic() + 50000, allow_live=True,
            planned_segments=planned_segments)


class LongDriverTests(unittest.TestCase):
    def test_external_speech_defaults_to_none(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            self.assertIsNone(h.bindings.external_speech)

    def test_explicit_probe_binding_verifies_without_ambient_path(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack, lines=["Approved English."])
            binary = Path(tmp).resolve() / "ffprobe"
            metadata = {"streams": [{"codec_type": "video", "codec_name": "vp9",
                "pix_fmt": "yuv420p", "width": 832, "height": 480, "nb_read_frames": "77",
                "avg_frame_rate": "16/1"}], "format": {"duration": "4.813"}}
            binary.write_text(f"#!{sys.executable}\nprint({json.dumps(metadata)!r})\n")
            binary.chmod(0o700)
            h.bindings = replace(h.bindings, ffprobe=binary)
            candidates = []
            def shot(*args, **kwargs):
                result = h.s2v(*args, **kwargs)
                artifact, = result["manifest"]["artifacts"]
                candidates.append(inspect_s2v_media(Path(args[3]) / "shot" / artifact["uri"],
                    ffprobe=kwargs["ffprobe"], absolute_deadline=kwargs["absolute_deadline"]))
                return result
            with patch.object(driver, "run_s2v", side_effect=shot) as run, \
                    patch.dict(os.environ, {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}), \
                    patch("cineforge.s2v_run.shutil.which", return_value=None) as which:
                record = h.run()
            self.assertEqual(record["outcome"], "complete", record)
            self.assertEqual(record["segments"][0]["status"], "verified")
            self.assertEqual(run.call_args.kwargs["ffprobe"], binary)
            self.assertEqual(candidates, [{"media_type": "video/webm", "codec": "vp9",
                "pixel_format": "yuv420p", "width": 832, "height": 480, "decoded_frames": 77,
                "fps": 16.0, "duration_seconds": 4.813}])
            which.assert_not_called()

    def test_default_probe_binding_is_none(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            with patch.object(driver, "run_s2v", side_effect=h.s2v) as run, \
                    patch.object(driver, "run_process") as probe:
                self.assertEqual(h.run()["outcome"], "complete")
            self.assertTrue(all(call.kwargs["ffprobe"] is None for call in run.call_args_list))
            probe.assert_not_called()

    def test_invalid_explicit_probe_rejected_before_reservation_or_acquisition(self):
        for kind in ("relative", "missing", "directory", "not-executable", "broken"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                binary = Path(tmp).resolve() / "ffprobe"
                if kind == "relative":
                    binary = Path("ffprobe")
                elif kind == "directory":
                    binary.mkdir()
                elif kind in ("not-executable", "broken"):
                    binary.write_text("#!/bin/sh\nexit 1\n")
                    if kind == "broken":
                        binary.chmod(0o700)
                h.bindings = replace(h.bindings, ffprobe=binary)
                with self.assertRaisesRegex(A20Error, "^LONG_FFPROBE_UNAVAILABLE:"):
                    h.run()
                h.patches["RunpodPodProvisioner"].assert_not_called()
                h.patches["CloudBudgetSession"].assert_not_called()
                self.assertFalse(h.output.exists())

    def test_planned_segments_sizes_lease(self):
        for planned, seconds in ((3, 6900), (1, 3900), (2, 5400), (100, 6900)):
            with self.subTest(planned=planned), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                record = h.run(planned_segments=planned)
                self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], seconds)
                self.assertEqual(record["budget"]["lease_seconds"], seconds)

    def test_planned_segments_rejects_invalid_values_before_acquisition(self):
        for planned in (True, False, 0, -1, 1.0, "3"):
            with self.subTest(planned=planned), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                with self.assertRaisesRegex(A20Error, "^LONG_PLANNED_SEGMENTS$"):
                    h.run(planned_segments=planned)
                h.patches["RunpodPodProvisioner"].assert_not_called()

    def test_planned_lease_stops_before_third_segment_speech(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack,
                lines=["Approved English.", "Second.", "Third.", "Fourth."])
            def checkpoint(name, record):
                if name == "segment-1":
                    h.budget.remaining_seconds.return_value = (
                        120 + driver.ATTEMPT_CAP_SECONDS + driver.SPEECH_MARGIN_SECONDS - 1)
            h.bindings = replace(h.bindings, checkpoint=checkpoint)
            record = h.run(planned_segments=3)
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 6900)
            self.assertEqual(record["budget"]["lease_seconds"], 6900)
            self.assertEqual(record["outcome"], "deadline_stopped", record)
            self.assertEqual(h.generated, [0, 1])
            self.assertEqual(record["failures"][-1]["message"], "LONG_SEGMENT_LEASE_EXHAUSTED")
            self.assertTrue(record["release_verified"])

    def test_planned_segments_stops_before_fourth_segment_speech(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack,
                lines=["Approved English.", "Second.", "Third.", "Fourth."])
            record = h.run(planned_segments=3)
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 6900)
            self.assertEqual(record["budget"]["lease_seconds"], 6900)
            self.assertEqual(h.generated, [0, 1, 2])
            self.assertEqual(record["outcome"], "deadline_stopped", record)
            self.assertEqual(record["failures"][-1]["message"], "LONG_PLANNED_SEGMENTS_REACHED")
            self.assertEqual(record["failures"][-1]["stage"], "speech")
            self.assertTrue(record["release_verified"])
            h.provisioner.release.assert_called_once_with("owned")

    def test_planned_resume_sizes_only_remaining_segments(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            h.fail_segment = 2
            first = h.run()
            self.assertEqual(first["outcome"], "failed", first)
            self.assertEqual([s["index"] for s in first["segments"]], [0, 1])
            h.generated.clear()
            h.fail_segment = None
            h.bindings = replace(h.bindings, acquisition_id="recovery")
            record = h.run(planned_segments=3)
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 3900)
            self.assertEqual(record["budget"]["lease_seconds"], 3900)
            self.assertEqual(h.generated, [2])
            self.assertEqual(record["outcome"], "complete", record)

    def test_happy_path_three_ordered_verified_segments(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            record = h.run()
            self.assertEqual(record["outcome"], "complete", record)
            self.assertEqual(h.generated, [0, 1, 2])
            self.assertEqual([s["status"] for s in record["segments"]], ["verified"] * 3)
            for kind in ("shot", "speech"):
                self.assertEqual(len({s[kind + "_sha256"] for s in record["segments"]}), 3)
            self.assertTrue(record["release_verified"])
            h.provisioner.release.assert_called_once_with("owned")
            h.bridge.preview.assert_called_once()
            h.budget.extend_retention.assert_not_called()
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 50400)
            self.assertEqual(h.provisioner.acquire.call_args.kwargs["max_cost_usd"], 17.78)
            self.assertEqual(record["human_review"], "pending_human_review")
            self.assertEqual(record["export_authority"], "none")

    def test_resume_skips_verified_payload_and_preserves_voice(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            h.fail_segment = 1
            first = h.run()
            self.assertEqual(first["outcome"], "failed")
            h.generated.clear()
            h.fail_segment = None
            h.bindings = replace(h.bindings, acquisition_id="recovery")
            second = h.run()
            self.assertEqual(second["outcome"], "complete", second)
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 5400)
            self.assertEqual(second["budget"], {"lease_seconds": 5400,
                "acquisition_max_cost_usd": 17.78, "aggregate_usd": 35, "max_hourly_usd": 1.25})
            self.assertEqual(h.generated, [1, 2])
            self.assertEqual(first["segments"][0], second["segments"][0])
            h.bridge.preview.assert_called_once()
            h.generated.clear()
            h.run()
            self.assertEqual(h.generated, [])

    def test_corrupt_or_missing_resume_payload_regenerates(self):
        for corrupt in (True, False):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                record = h.run()
                path = h.output / record["segments"][0]["shot_uri"]
                if corrupt:
                    path.write_bytes(b"corrupt")
                else:
                    path.unlink()
                h.generated.clear()
                h.bindings = replace(h.bindings, acquisition_id="recovery")
                record = h.run()
                self.assertEqual(record["outcome"], "complete", record)
                self.assertEqual(h.generated, [0])

    def test_failure_records_stage_and_releases_without_continuing(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            h.fail_segment = 1
            record = h.run()
            self.assertEqual(record["outcome"], "failed")
            self.assertEqual(record["failures"], [{"index": 1, "stage": "shot",
                "error_type": "RuntimeError", "message": "synthetic S2V failure"}])
            self.assertEqual([s["index"] for s in record["segments"]], [0])
            self.assertTrue(record["release_verified"])
            self.assertEqual(json.loads((h.output / "run.json").read_bytes()), record)

    def test_deadline_before_segment_two_preserves_completed_segments(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            now = [100.]
            def checkpoint(name, record):
                if name == "segment-1":
                    now[0] = 201.
            h.bindings = replace(h.bindings, checkpoint=checkpoint)
            with patch.object(driver.time, "monotonic", side_effect=lambda: now[0]):
                record = h.run(deadline=200.)
            self.assertEqual(record["outcome"], "deadline_stopped", record)
            self.assertEqual([s["index"] for s in record["segments"]], [0, 1])
            self.assertEqual(h.generated, [0, 1])
            self.assertTrue(record["release_verified"])

    def test_expired_entry_can_resume_without_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            result = h.run(deadline=driver.time.monotonic() - 1)
            self.assertEqual(result["outcome"], "deadline_stopped")
            h.provisioner.acquire.assert_not_called()
            self.assertEqual(h.run()["outcome"], "complete")

    def test_frozen_mutation_and_cleanup_failure_are_recorded(self):
        for fail in ("source", "release"):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                if fail == "source":
                    h.mutation = "segment-0"
                else:
                    h.provisioner.release.side_effect = RuntimeError("cleanup failed")
                record = h.run()
                self.assertEqual(record["outcome"], "failed")
                self.assertTrue(record["failures"])
                self.assertEqual(record["release_verified"], fail != "release")

    def test_failed_acquisition_records_snapshot_and_allows_resume(self):
        for state in ("released", "not_created"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                snapshot = LeaseSnapshot("acquire", state, "owned" if state == "released" else None)
                h.provisioner.acquire.side_effect = ProvisioningError("acquire failed", snapshot)
                record = h.run()
                self.assertEqual(record["outcome"], "failed")
                self.assertTrue(record["release_verified"])
                self.assertEqual(record["acquisition_snapshot"]["state"], state)
                self.assertEqual(json.loads((h.output / "run.json").read_bytes()), record)
                h.provisioner.release.assert_not_called()
                h.provisioner.acquire.side_effect = None
                h.bindings = replace(h.bindings, acquisition_id="recovery")
                self.assertEqual(h.run()["outcome"], "complete")

    def test_uncertain_acquisition_or_release_blocks_resume(self):
        for error in (AcquisitionUncertainError("uncertain"), CleanupUncertainError("uncertain")):
            with self.subTest(error=type(error)), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                h.provisioner.acquire.side_effect = error
                self.assertFalse(h.run()["release_verified"])
                h.provisioner.acquire.reset_mock()
                with self.assertRaisesRegex(A20Error, "LONG_RESUME_SCOPE_OR_RELEASE"):
                    h.run()
                h.provisioner.acquire.assert_not_called()

    def test_segment_stops_before_speech_when_lease_too_short(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            def checkpoint(name, record):
                if name == "segment-0":
                    h.budget.remaining_seconds.return_value = (
                        120 + driver.ATTEMPT_CAP_SECONDS + driver.SPEECH_MARGIN_SECONDS - 1)
            h.bindings = replace(h.bindings, checkpoint=checkpoint)
            record = h.run()
            self.assertEqual(record["outcome"], "deadline_stopped", record)
            self.assertEqual(h.generated, [0])
            self.assertEqual(record["failures"][-1]["message"], "LONG_SEGMENT_LEASE_EXHAUSTED")
            self.assertTrue(record["release_verified"])

    def test_voice_lock_within_run_and_across_resume(self):
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                if resume:
                    h.fail_segment = 1
                    h.run()
                    h.fail_segment = None
                    h.generated.clear()
                    h.bindings = replace(h.bindings, acquisition_id="recovery")
                def segment(index, reference, h=h):
                    context = h.segment(index, reference)
                    if index == 1:
                        context.voices = ()
                    return context
                h.bindings = replace(h.bindings, segment=segment)
                record = h.run()
                self.assertEqual(record["outcome"], "failed", record)
                self.assertEqual(record["failures"][-1]["message"], "A20_CANONICAL_VOICE_CHANGED")
                self.assertEqual(h.generated, [] if resume else [0])
                self.assertTrue(record["release_verified"])

    def test_checkpoint_failures_written_before_release(self):
        for source in ("binding", "store"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                events = []
                original = driver.PrivateRunDirectory.checkpoint
                injected = [False]
                def checkpoint(store, payload, source=source, h=h, injected=injected,
                               original=original, events=events):
                    record = json.loads(payload)
                    if (source == "store" and store.path == h.output and record["checkpoints"]
                            and not injected[0]):
                        injected[0] = True
                        raise OSError("checkpoint failed")
                    original(store, payload)
                    if store.path == h.output and record["failures"]:
                        events.append("failure_written")
                stack.enter_context(patch.object(driver.PrivateRunDirectory, "checkpoint", checkpoint))
                if source == "binding":
                    h.bindings = replace(h.bindings, checkpoint=Mock(side_effect=RuntimeError("checkpoint failed")))
                def release(device_id, h=h, events=events):
                    self.assertEqual(device_id, "owned")
                    self.assertEqual(json.loads((h.output / "run.json").read_bytes())["outcome"], "failed")
                    events.append("release")
                h.provisioner.release.side_effect = release
                record = h.run()
                self.assertEqual(record["outcome"], "failed", record)
                self.assertEqual(record["failures"][-1]["message"], "checkpoint failed")
                self.assertTrue(record["release_verified"])
                h.provisioner.release.assert_called_once()
                self.assertLess(events.index("failure_written"), events.index("release"))
                self.assertEqual(json.loads((h.output / "run.json").read_bytes()), record)

    def test_retain_node_rejected_before_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
            with self.assertRaisesRegex(A20Error, "LONG_RETAIN_NODE_UNSUPPORTED"):
                h.run()
            h.provisioner.acquire.assert_not_called()


class LongAdmissionTests(unittest.TestCase):
    def test_planned_lease_reserves_requested_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval = LongSessionBudgetApproval.for_scope(
                str(Path(tmp).resolve() / "journal"), "EUR-IS-2",
                EvidenceRef("private://synthetic/review", "a" * 64),
                gpu_id=L40S_GPU_ID, aggregate_usd=35, lifetime_seconds=50400, phase_id="test")
            service = LongSessionServiceContext(approval)
            spec = RunpodPodSpec("a20test", IMAGE, L40S_GPU_ID, "EUR-IS-2", "ssh-ed25519 AAAA",
                max_hourly_usd=1.25, disk_gb=150, min_ram_per_gpu_gb=64)
            maximum = driver.acquisition_max_cost_usd(spec, approval)
            self.assertEqual(maximum, 17.78)
            intent = AcquisitionIntent("acquisition", spec, "a" * 32,
                "cineforge-a20test-admission", service.model, maximum, 6900, 120, 10)
            prepare_long_session_journal(service)
            with CloudBudgetSession(approval.journal_directory, approval, AdmissionCatalog(),
                                    guard_deployment_configured=True) as session:
                session.event_sink(intent, LeaseSnapshot(intent.key, "reserved"))
                session.before_create(intent)
            with CloudJournal(approval.journal_directory) as journal:
                record = strict_object(journal.initial)["records"][intent.key]
                self.assertTrue(record["authorized"])
                self.assertEqual(record["reservation_usd"], 2.434167)


class StagingFailureRecordTests(unittest.TestCase):
    def test_stage_record_copies_diagnostic_and_handles_missing_bridge(self):
        for entry in ("stage", "measure_capabilities", "constructor"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                remote = native_capability_tests.CapabilityChannelTests("runTest")
                remote.setUp()
                stack.callback(remote.doCleanups)
                remote.lost = "native_setup" if entry == "stage" else "native_capabilities"
                h.bindings.sox_wheel.write_bytes(b"x")
                h.bindings = replace(h.bindings, recipe=recipe())
                factory = h.patches["RemoteNativeSpeechRuntime"]
                factory.return_value = remote.bridge
                if entry == "constructor":
                    factory.side_effect = RuntimeError("failure")
                record = h.run()
                failure = record["failures"][0]
                self.assertEqual(failure["stage"], "stage")
                if entry == "constructor":
                    self.assertNotIn("staging_failure", failure)
                else:
                    diagnostic = remote.bridge.last_staging_failure
                    self.assertEqual(diagnostic["entry"], entry)
                    self.assertEqual(diagnostic["sub_step"], "setup" if entry == "stage" else "call")
                    self.assertEqual(failure["staging_failure"], diagnostic)
                    original = diagnostic["sub_step"]
                    diagnostic["sub_step"] = "changed"
                    self.assertEqual(failure["staging_failure"]["sub_step"], original)
                saved = json.loads((h.output / "run.json").read_text())
                self.assertEqual(saved["failures"], record["failures"])

    def test_success_and_other_stage_failures_gain_no_diagnostic(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                h.bridge.last_staging_failure = {"entry": "stage", "sub_step": "setup"}
                if failed:
                    h.fail_segment = 0
                record = h.run()
                self.assertEqual(record["outcome"], "failed" if failed else "complete")
                self.assertNotIn('"staging_failure"', json.dumps(record))


class SpeechFailureRecordTests(unittest.TestCase):
    def test_speech_failure_record_deep_copies_diagnostic(self):
        for entry in ("invoke", "health_check"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                diagnostic = dict(entry=entry, sub_step="dispatch" if entry == "invoke" else "ping",
                                  error_type="NativeSpeechError", reply_keys=["binding"],
                                  cause_type="BootstrapError", reason="process_failed", returncode=1,
                                  supervision_message="worker supervision or lease deadline failed")
                h.bridge.last_speech_failure = diagnostic
                if entry == "invoke":
                    h.adapter.synthesize.side_effect = RuntimeError("failed")
                else:
                    h.bridge.speech_health_check.return_value = False
                record = h.run()
                failure = record["failures"][0]
                self.assertEqual(failure["stage"], "speech")
                self.assertEqual(failure["speech_failure"], diagnostic)
                diagnostic["reply_keys"].append("changed")
                diagnostic["returncode"] = 99
                self.assertEqual(failure["speech_failure"]["returncode"], 1)
                self.assertEqual(failure["speech_failure"]["reply_keys"], ["binding"])
                if entry == "health_check":
                    self.assertEqual(failure["message"], "LONG_SPEECH_HEALTH_CHECK")
                    h.adapter.synthesize.assert_not_called()
                    self.assertEqual(h.generated, [])
                saved = json.loads((h.output / "run.json").read_text())
                self.assertEqual(saved["failures"], record["failures"])

    def test_success_and_other_stage_failures_gain_no_speech_diagnostic(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                h.bridge.last_speech_failure = dict(entry="invoke", sub_step="dispatch")
                if failed:
                    h.fail_segment = 0
                record = h.run()
                self.assertEqual(record["outcome"], "failed" if failed else "complete")
                self.assertNotIn('"speech_failure"', json.dumps(record))
                if not failed:
                    self.assertEqual(h.bridge.speech_health_check.call_count, 3)


class SelectFailureRecordTests(unittest.TestCase):
    def test_select_failure_record_deep_copies_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack)
            diagnostic = dict(entry="select", sub_step="reply_validation",
                              error_type="ValueError", reply_keys=["selection_sha256"])
            h.bridge.last_select_failure = diagnostic
            h.bridge.select.side_effect = RuntimeError("failed")
            record = h.run()
            failure = record["failures"][0]
            self.assertEqual(failure["stage"], "select")
            self.assertEqual(failure["select_failure"], diagnostic)
            diagnostic["reply_keys"].append("changed")
            self.assertEqual(failure["select_failure"]["reply_keys"], ["selection_sha256"])
            saved = json.loads((h.output / "run.json").read_text())
            self.assertEqual(saved["failures"], record["failures"])

    def test_success_and_other_stage_failures_gain_no_select_diagnostic(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = LongHarness(Path(tmp).resolve(), stack)
                h.bridge.last_select_failure = dict(entry="select", sub_step="native_select")
                if failed:
                    h.fail_segment = 0
                record = h.run()
                self.assertEqual(record["outcome"], "failed" if failed else "complete")
                self.assertNotIn('"select_failure"', json.dumps(record))
