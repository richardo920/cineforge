"""Offline composition probes; fixture bytes are deliberately not encoded video."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_cloud_budget import Clock, Cloud, selection
from test_comfy_run import WireFixture, worker_record
from test_runpod import spec

from cineforge.cloud_budget import BudgetApproval
from cineforge.cloud_smoke import (
    MODEL_WORKFLOW_SHA256,
    SCOPE_SHA256,
    CloudSmokeError,
    approved_plan,
    external_api_key,
    load_reference,
    normalize,
    probe_media,
    recover_managed,
    run_managed_smoke,
)
from cineforge.comfy_policy import WorkerVerification
from cineforge.comfy_run import execute_comfy
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.runpod import WorkerObservation
from cineforge.worker_guard import GuardReceipt, worker_guard_source_sha256
from cineforge.workflows import canonical


def technical(_path):
    return {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
            "width": 832, "height": 480, "decoded_frames": 33, "fps": 16.,
            "duration_seconds": 2.063}


class Worker:
    def __init__(self, budget, selected, pod_spec, mode="success"):
        self.budget, self.selected, self.spec = budget, selected, pod_spec
        self.mode, self.observation = mode, None
        self.closed = False
        self.prepared = False
        self.wire = WireFixture(selected)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def probe(self, snapshot, timeout):
        if self.mode == "arm_failed":
            raise RuntimeError("Bearer fixture-sensitive-value")
        plan = self.budget.guard_plan(snapshot.key)
        self.budget.arm_guard(snapshot.key, GuardReceipt(plan, worker_guard_source_sha256(),
            123, time.time(), True, True))
        self.observation = WorkerObservation(GPUDevice(snapshot.resource_id, Backend.CUDA,
            24082, frozenset({Precision.FP16})), self.spec.gpu_id)
        self.identity = snapshot.resource_id
        return self.observation

    def prepare(self):
        self.prepared = True
        if self.mode in ("setup_timeout", "download_timeout"):
            raise TimeoutError("https://user:fixture-sensitive-value@untrusted.invalid")  # pragma: allowlist secret - synthetic diagnostic fixture
        if self.mode == "interrupt":
            raise KeyboardInterrupt("fixture-sensitive-value")
        record = worker_record(self.selected)
        record.update(worker_id=self.identity, image_digest=self.spec.image.split("@", 1)[1])
        return WorkerVerification.from_dict(record)

    def endpoint(self):
        return "http://127.0.0.1:8188"

    def execution_transport(self):
        return self.wire

    def check(self):
        if self.mode == "heartbeat_lost" and self.prepared:
            raise ValueError("fixture-sensitive-value")
        if self.enforce():
            raise ValueError("fixture budget expired")

    def set_enforcer(self, enforce):
        self.enforce = enforce


class ManagedSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-m34-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection = selection()
        self.spec = replace(spec(), image="registry.invalid/fixture@sha256:"
                            + hashlib.sha256(b"synthetic image").hexdigest())
        self.approval = BudgetApproval(SCOPE_SHA256, MODEL_WORKFLOW_SHA256, self.selection.model,
            self.spec.image, self.spec.gpu_id, self.spec.data_center_id, "12.8", 3, 2, 1, 1800, 1, 80)
        self.cloud = Cloud(Clock())
        plan = approved_plan(self.selection)
        baseline = self.root / "baseline"
        self.manifest = execute_comfy(plan, self.selection,
            WorkerVerification.from_dict(worker_record(self.selection)), baseline,
            allow_live=True, transport=WireFixture(self.selection), poll_interval_seconds=0)
        self.reference = load_reference(baseline / "run.json", self.selection, inspect_media=technical)
        self.workers = []

    def run_smoke(self, name="success", *, mode="success", **options):
        def factory(budget, state):
            worker = Worker(budget, self.selection, self.spec, mode)
            self.workers.append(worker)
            return worker
        values = dict(allow_live=True, inspect_media=technical)
        values.update(options)
        return run_managed_smoke(self.selection, self.approval, self.spec, self.cloud,
            self.root / "budget", self.root / name, self.reference, factory, **values)

    def test_managed_success_and_second_injected_failure_release_owned_pods(self):
        result = self.run_smoke()
        self.assertEqual(result["outcome"], "succeeded")
        self.assertTrue(result["release_verified"])
        self.assertEqual(result["managed"]["contract"], result["reference"]["contract"])
        self.assertIsNone(result["costs"]["actual_total_usd"])
        self.assertEqual(result["costs"]["attempts"], 1)
        second = self.run_smoke("injected", case="post-acquire-failure")
        self.assertEqual(second["outcome"], "injected_post_acquire_failure")
        self.assertTrue(second["release_verified"])
        self.assertEqual(second["costs"]["attempts"], 2)
        self.assertFalse(self.workers[1].prepared)
        self.assertTrue(all(worker.closed for worker in self.workers))
        self.assertEqual(self.cloud.pods, {})
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 2)
        self.assertEqual(sum(c[0] == "DELETE" for c in self.cloud.calls), 2)
        before = copy.deepcopy(self.cloud.calls)
        with self.assertRaises(CloudSmokeError):
            self.run_smoke("third")
        self.assertEqual(self.cloud.calls, before)

    def test_setup_download_heartbeat_and_interruption_all_cleanup_without_secret_diagnostics(self):
        for mode in ("setup_timeout", "download_timeout", "heartbeat_lost", "interrupt", "arm_failed"):
            with self.subTest(mode=mode):
                # A fresh fixture has an independent synthetic scope. Live drivers never
                # reset a budget journal or retry paid creates after these failures.
                self.root = Path(self.temp.name) / mode
                self.root.mkdir(mode=0o700)
                self.cloud = Cloud(Clock())
                try:
                    self.run_smoke(mode=mode)
                except CloudSmokeError:
                    self.assertNotIn("fixture-sensitive-value", traceback.format_exc())
                else:
                    self.fail("managed failure was reported as success")
                self.assertFalse(self.cloud.pods)
                self.assertTrue(self.workers[-1].closed)
                report = json.loads((self.root / "success/run.json").read_text())
                self.assertTrue(report["release_verified"])
                self.assertNotIn("fixture-sensitive-value", json.dumps(report))
                self.assertIsNone(report["costs"]["actual_total_usd"])

    def test_execution_failure_and_timeout_always_release(self):
        def timeout(*args, **kwargs):
            raise TimeoutError("Bearer fixture-sensitive-value")
        with self.assertRaises(CloudSmokeError):
            self.run_smoke(execute=timeout)
        self.assertFalse(self.cloud.pods)
        report = json.loads((self.root / "success/run.json").read_text())
        self.assertEqual(report["outcome"], "incomplete")
        self.assertTrue(report["release_verified"])

    def test_missing_live_opt_in_does_not_touch_output_or_cloud(self):
        with self.assertRaises(CloudSmokeError):
            self.run_smoke(allow_live=False)
        self.assertFalse((self.root / "success").exists())
        self.assertFalse(self.cloud.calls)

    def test_unknown_create_is_retained_and_recovery_does_not_acquire(self):
        self.cloud.create_lost = True
        with self.assertRaises(CloudSmokeError):
            self.run_smoke()
        self.assertEqual(len(self.cloud.pods), 1)
        record = json.loads((self.root / "budget/budget.json").read_text())
        self.assertEqual(next(iter(record["records"].values()))["snapshot"]["state"],
                         "create_uncertain")
        self.cloud.create_lost = False
        result = recover_managed(self.spec, self.approval, self.cloud, self.root / "budget",
                                  allow_live=True)
        self.assertTrue(result["release_verified"])
        self.assertFalse(self.cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)

    def test_cleanup_uncertain_retains_identity_until_separate_recovery(self):
        self.cloud.delete_effect = False
        # Keep this offline timeout instantaneous without changing paid driver defaults.
        with patch("cineforge.cloud_smoke.RunpodPodProvisioner._clean", side_effect=ValueError):
            with self.assertRaises(CloudSmokeError):
                self.run_smoke(case="post-acquire-failure")
        report = json.loads((self.root / "success/run.json").read_text())
        self.assertFalse(report["release_verified"])
        self.assertEqual(report["leases"][0]["resource_id"], "pod_1")
        self.assertEqual(report["leases"][0]["state"], "cleanup_uncertain")
        self.cloud.delete_effect = True
        result = recover_managed(self.spec, self.approval, self.cloud, self.root / "budget",
                                  allow_live=True)
        self.assertTrue(result["release_verified"])
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)

    def test_recovery_refuses_foreign_metadata_and_never_deletes(self):
        self.cloud.create_lost = True
        with self.assertRaises(CloudSmokeError):
            self.run_smoke()
        self.cloud.pods["pod_1"]["env"]["CINEFORGE_LEASE"] = "foreign"
        with self.assertRaises(CloudSmokeError):
            recover_managed(self.spec, self.approval, self.cloud, self.root / "budget",
                            allow_live=True)
        self.assertFalse(any(call[0] == "DELETE" for call in self.cloud.calls))

    def test_used_budget_refuses_different_session_spec_without_spend(self):
        self.run_smoke(case="post-acquire-failure")
        self.spec = replace(self.spec, session_id="another-session")
        previous = copy.deepcopy(self.cloud.calls)
        with self.assertRaises(CloudSmokeError):
            self.run_smoke("changed")
        self.assertEqual(previous, self.cloud.calls)

    def test_changed_reference_artifact_plan_and_technical_contract_reject(self):
        source = self.root / "baseline/run.json"
        original = source.read_bytes()
        for change in (lambda d: d["manifest"]["plan"]["shots"][0].update(prompt="unapproved"),
                       lambda d: d["manifest"]["artifacts"][0].update(sha256="0"*64),
                       lambda d: d["translated_workflow"]["3"]["inputs"].update(seed=43)):
            record = json.loads(original)
            change(record)
            source.write_bytes(canonical(record))
            with self.assertRaises((CloudSmokeError, ValueError)):
                load_reference(source, self.selection, inspect_media=technical)
        source.write_bytes(original)
        with self.assertRaises(CloudSmokeError):
            normalize(self.manifest, self.selection, self.root / "baseline",
                      inspect_media=lambda p: dict(technical(p), decoded_frames=1))

    def test_ffprobe_contract_uses_decoded_frames_and_rejects_bad_media(self):
        probe = {"streams": [{"codec_name": "vp9", "width": 832, "height": 480,
            "pix_fmt": "yuv420p", "avg_frame_rate": "16/1", "nb_read_frames": "33"}],
            "format": {"duration": "2.063"}}
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(probe)) as run:
            self.assertEqual(probe_media(Path("fixture.webm"))["decoded_frames"], 33)
            self.assertIn("-count_frames", run.call_args.args[0])
            probe["streams"][0]["nb_read_frames"] = "0"
            run.return_value = canonical(probe)
            with self.assertRaises(CloudSmokeError):
                probe_media(Path("fixture.webm"))

    def test_credentials_only_private_file_or_env_never_in_cli_diagnostics(self):
        directory = self.root / "creds"
        directory.mkdir(mode=0o700)
        path = directory / "key"
        path.write_text("fixture-secret-value\n")
        path.chmod(0o600)
        self.assertEqual(external_api_key(str(path)), "fixture-secret-value")
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            external_api_key(str(path))
        path.chmod(0o600)
        link = directory / "link"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            external_api_key(str(link))
        hard = directory / "hard"
        os.link(path, hard)
        with self.assertRaises(ValueError):
            external_api_key(str(path))
        command = [sys.executable, "-m", "cineforge", "cloud-smoke", "--selection", "absent",
            "--proposal", "absent", "--budget-dir", "absent", "--session-id", "fixture",
            "--ssh-private-key", "fixture-secret-value", "--output-dir", "absent",
            "--reference-run", "absent"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("fixture-secret-value", result.stdout + result.stderr)
        self.assertIn("retain private journals", result.stderr)


if __name__ == "__main__":
    unittest.main()
