"""Offline record-preparation privacy and claim-boundary regressions."""

import json
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

from test_comfy_run import WireFixture, plan_for, worker_record
from test_conditioning import context, runtime_fixture, selected

from cineforge import conditioning_profile as profile
from cineforge.comfy_policy import ApprovedComfySelection, WorkerVerification, load_comfy_selection
from cineforge.comfy_run import ComfyRunError, execute_comfy
from cineforge.conditioning import ConditioningError
from cineforge.contracts import Project, ProjectPlan, Scene
from cineforge.private_store import PrivateRunDirectory
from cineforge.workflows import WorkflowTemplate

ROOT = Path(__file__).resolve().parents[1]
CANARY = "synthetic-private-preparation-canary"


class ComfyPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cineforge-record-preparation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "run"
        self.selection = load_comfy_selection(ROOT / "config/comfy/wan21-t2v-1.3b.json")
        self.plan = plan_for(self.selection)
        self.worker = WorkerVerification.from_dict(worker_record(self.selection))
        self.wire = WireFixture(self.selection)

    def execute(self, **changes):
        options = dict(allow_live=True, transport=self.wire, poll_interval_seconds=0)
        options.update(changes)
        return execute_comfy(self.plan, self.selection, self.worker, self.output, **options)

    def rejected(self, **changes):
        error = None
        try:
            self.execute(**changes)
        except BaseException as caught:
            error = caught
        self.assertIsInstance(error, ComfyRunError)
        self.assertNotIn(CANARY, "".join(traceback.format_exception(error)))
        self.assertIsNone(error.__context__)
        self.assertIsNone(error.__cause__)
        return error

    def assert_preflight_rejected(self, **changes):
        error = self.rejected(**changes)
        self.assertEqual(str(error), "ComfyUI single-shot preflight rejected")
        self.assertEqual(self.wire.calls, [])
        self.assertFalse(self.output.exists())

    def test_second_translation_failures_are_static_and_do_not_claim_output(self):
        original = WorkflowTemplate.translate
        for index, failure in enumerate((KeyError, RuntimeError, KeyboardInterrupt)):
            with self.subTest(failure=failure):
                self.output = self.root / str(index)
                calls = 0

                def translate(template, request):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise failure(CANARY)
                    return original(template, request)

                with patch.object(WorkflowTemplate, "translate", translate):
                    self.assert_preflight_rejected()
                self.assertEqual(calls, 2)

    def test_late_approval_property_failure_is_static_before_claim(self):
        original = WorkflowTemplate.translate
        declarations = ApprovedComfySelection.declarations.fget
        calls = 0

        def translate(template, request):
            nonlocal calls
            calls += 1
            return original(template, request)

        def declaration(selection):
            if calls >= 2:
                raise RuntimeError(CANARY)
            return declarations(selection)

        with patch.object(WorkflowTemplate, "translate", translate), \
                patch.object(ApprovedComfySelection, "declarations", property(declaration)):
            self.assert_preflight_rejected()
        self.assertEqual(calls, 2)

    def test_unserializable_prepared_metadata_rejects_before_claim(self):
        original = WorkflowTemplate.translate
        declarations = ApprovedComfySelection.declarations.fget
        for index, invalid in enumerate((float("nan"), type(CANARY, (), {})())):
            with self.subTest(index=index):
                self.output = self.root / str(index)
                calls = 0

                def translate(template, request):
                    nonlocal calls
                    calls += 1
                    return original(template, request)

                def declaration(selection):
                    data = declarations(selection)
                    if calls >= 2:
                        data["approval"]["spend_limit_usd"] = invalid
                    return data

                with patch.object(WorkflowTemplate, "translate", translate), \
                        patch.object(ApprovedComfySelection, "declarations", property(declaration)):
                    self.assert_preflight_rejected()
                self.assertEqual(calls, 2)

    def test_real_reference_revocation_is_rechecked_at_third_translation(self):
        gate, records, _, _ = context(self.root)
        chosen = selected(gate)
        record = worker_record(chosen)
        record.update(image_digest=profile.IMAGE.split("@", 1)[1], torch_version="2.8.0+cu128")
        record["gpu"]["memory_mib"] = 32286
        self.worker = WorkerVerification.from_dict(record)
        self.selection = chosen.bind_worker(self.worker, gate.reference.worker_metadata,
                                           runtime_fixture())
        shot = self.selection.gate.requests[0]
        self.plan = ProjectPlan(Project("synthetic", 1, "Synthetic"),
                                (Scene(shot.scene_id, "synthetic", 1),), (shot,))
        self.wire = WireFixture(self.selection)
        original = WorkflowTemplate.translate
        calls = 0

        def revoke_after_request(template, request):
            nonlocal calls
            calls += 1
            value = original(template, request)
            if calls == 2:
                gate.ledger.revoke(records[0].ref, reason=CANARY)
            return value

        with patch.object(WorkflowTemplate, "translate", revoke_after_request):
            self.assert_preflight_rejected()
        self.assertEqual(calls, 3)
        with self.assertRaises(ConditioningError):
            gate.preflight()
        self.assert_preflight_rejected()

    def test_existing_unadmitted_inputs_leave_no_directory_or_lock(self):
        for options in ({"allow_live": False}, {"endpoint": "http://public.example"},
                        {"timeout_seconds": float("inf")}, {"result_timeout_seconds": True}):
            with self.subTest(options=options):
                self.assert_preflight_rejected(**options)

    def test_postclaim_node_preflight_failure_retains_honest_incomplete_record(self):
        checkpoint = PrivateRunDirectory.checkpoint
        snapshots = []

        def writing(store, payload):
            snapshots.append(json.loads(payload))
            checkpoint(store, payload)

        self.wire.failure = RuntimeError(CANARY)
        with patch.object(PrivateRunDirectory, "checkpoint", writing):
            self.rejected()
        self.assertEqual(snapshots[0]["outcome"], "prepared")
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["outcome"], "incomplete")
        self.assertIsNone(record["known_run"])
        self.assertIsNone(record["server_output"])
        self.assertFalse(record["cancellation"]["attempted"])
        self.assertNotIn("manifest", record)
        self.assertTrue((self.output / ".lock").exists())
        self.assertTrue(self.wire.calls)
        self.assertFalse(any(c[0] == "POST" for c in self.wire.calls))
        self.assertNotIn(CANARY, (self.output / "run.json").read_text())

    def test_postclaim_initial_checkpoint_failure_can_record_incomplete_without_fake_job(self):
        checkpoint = PrivateRunDirectory.checkpoint
        calls = 0

        def fail_once(store, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PermissionError(CANARY)
            checkpoint(store, payload)

        with patch.object(PrivateRunDirectory, "checkpoint", fail_once):
            self.rejected()
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["outcome"], "incomplete")
        self.assertIsNone(record["known_run"])
        self.assertIsNone(record["server_output"])
        self.assertEqual(record["request"], self.selection.request(self.plan.shots[0]).to_dict())
        self.assertEqual(record["cancellation"],
                         {"attempted": False, "confirmed_terminal": False})
        self.assertNotIn("manifest", record)
        self.assertEqual(self.wire.calls, [])
        self.assertTrue((self.output / ".lock").exists())
        self.assertNotIn(CANARY, (self.output / "run.json").read_text())


if __name__ == "__main__":
    unittest.main()
