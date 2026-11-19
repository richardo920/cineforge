"""Offline segment execution, durable resume and failure cleanup contracts."""
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_a20_driver import Harness
from test_speech import CanonicalVoiceBinding, fixture, wav_bytes

from cineforge import long_driver as driver
from cineforge.a20_inputs import A20Error, FrozenInputs
from cineforge.runpod import (
    AcquisitionUncertainError,
    CleanupUncertainError,
    LeaseSnapshot,
    ProvisioningError,
)
from cineforge.s2v_service import LongSessionBudgetApproval, LongSessionServiceContext


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
    def test_planned_segments_sizes_lease(self):
        for planned, seconds in ((3, 6900), (1, 3900), (2, 5400), (100, 50400)):
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

    def test_planned_lease_stops_before_fourth_segment_speech(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = LongHarness(Path(tmp).resolve(), stack,
                lines=["Approved English.", "Second.", "Third.", "Fourth."])
            def checkpoint(name, record):
                if name == "segment-2":
                    h.budget.remaining_seconds.return_value = (
                        120 + driver.ATTEMPT_CAP_SECONDS + driver.SPEECH_MARGIN_SECONDS - 1)
            h.bindings = replace(h.bindings, checkpoint=checkpoint)
            record = h.run(planned_segments=3)
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 6900)
            self.assertEqual(record["budget"]["lease_seconds"], 6900)
            self.assertEqual(record["outcome"], "deadline_stopped", record)
            self.assertEqual(h.generated, [0, 1, 2])
            self.assertEqual(record["failures"][-1]["message"], "LONG_SEGMENT_LEASE_EXHAUSTED")
            self.assertTrue(record["release_verified"])

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
