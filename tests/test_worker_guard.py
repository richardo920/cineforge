"""Standalone guard tests: fake clocks/transports/process, never a live GPU or credential."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cineforge.worker_guard import (
    GuardReceipt,
    WorkerGuardError,
    WorkerGuardPlan,
    read_heartbeat,
    run_guard,
    stop_own_pod,
    verify_own_pod,
    worker_credential,
    worker_guard_source_sha256,
)


class Clock:
    def __init__(self):
        self.wall, self.mono = 100., 10.

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, seconds):
        self.wall += seconds
        self.mono += seconds


def plan(**options):
    values = dict(pod_id="fixture-pod", session_id="fixture-session", lease_token="0"*32,
                  stop_at_unix=110., idle_seconds=3.)
    values.update(options)
    return WorkerGuardPlan(**values)


class Reply:
    def __init__(self, data):
        self.data = json.dumps(data).encode()

    def read(self, size):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class GuardTests(unittest.TestCase):
    def run_fake(self, heartbeat=None, stop=None, **options):
        clock = Clock()
        if heartbeat is None:
            heartbeat = clock.time
        stops = []
        if stop is None:
            def stop():
                stops.append(clock.time())
                return True
        result = run_guard(plan(), heartbeat, stop, clock=clock.time,
            monotonic=clock.monotonic, sleep=clock.sleep, **options)
        return clock, result, stops

    def test_absolute_deadline_survives_ongoing_inference_heartbeat(self):
        clock, result, stops = self.run_fake()
        self.assertEqual(clock.time(), 110.)
        self.assertEqual(stops, [110.])
        self.assertEqual(result["reason"], "deadline")
        self.assertEqual(result["status"], "compute_stop_confirmed")
        self.assertFalse(result["resource_deleted"])

    def test_idle_or_missing_bridge_triggers_stop(self):
        clock, result, stops = self.run_fake(heartbeat=lambda: 100.)
        self.assertEqual(clock.time(), 103.)
        self.assertEqual(result["reason"], "idle")
        self.assertEqual(stops, [103.])
        def invalid():
            raise ValueError("fixture-sensitive-value")
        self.assertEqual(self.run_fake(heartbeat=invalid)[1]["reason"], "invalid_heartbeat")
        for value in (True, float("nan"), float("inf"), 200.):
            self.assertEqual(self.run_fake(heartbeat=lambda: value)[1]["reason"],
                             "invalid_heartbeat")

    def test_frozen_wall_monotonic_deadline_and_both_clock_rollbacks(self):
        for mode in ("frozen", "wall_rollback", "mono_rollback"):
            clock = Clock()
            def sleep(seconds):
                if mode == "frozen":
                    clock.mono += seconds
                elif mode == "wall_rollback":
                    clock.wall -= 1
                else:
                    clock.mono -= 1
            result = run_guard(plan(idle_seconds=30), clock.time, lambda: True,
                clock=clock.time, monotonic=clock.monotonic, sleep=sleep)
            self.assertEqual(result["reason"], "deadline" if mode == "frozen" else "clock_rollback")

    def test_bounded_stop_retries_and_uncertainty_are_honest(self):
        attempts = []
        def stop():
            attempts.append(1)
            raise TimeoutError("fixture-sensitive-value")
        result = self.run_fake(stop=stop)[1]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(result["status"], "compute_stop_unconfirmed")
        self.assertFalse(result["resource_deleted"])

    def test_private_heartbeat_scope_links_duplicate_json_and_permissions(self):
        with tempfile.TemporaryDirectory(prefix="cineforge-heartbeat-", dir="/private/tmp") as tmp:
            path = Path(tmp) / "heartbeat"
            data = dict(schema_version=1, pod_id=plan().pod_id, lease_token=plan().lease_token,
                        heartbeat_unix=100)
            path.write_text(json.dumps(data))
            path.chmod(0o600)
            self.assertEqual(read_heartbeat(path, plan()), 100)
            path.chmod(0o644)
            with self.assertRaises(WorkerGuardError):
                read_heartbeat(path, plan())
            path.chmod(0o600)
            linked = Path(tmp) / "linked"
            linked.symlink_to(path)
            with self.assertRaises(WorkerGuardError):
                read_heartbeat(linked, plan())
            linked.unlink()
            os.link(path, linked)
            with self.assertRaises(WorkerGuardError):
                read_heartbeat(path, plan())
            linked.unlink()
            path.write_text('{"pod_id":"fixture-pod","pod_id":"other"}')
            with self.assertRaises(WorkerGuardError):
                read_heartbeat(path, plan())

    def test_receipt_requires_explicit_both_attestations_and_strict_bounds(self):
        for field, value in (("operator_verified_alive", False),
                             ("operator_verified_own_pod", False), ("pid", True),
                             ("observed_unix", float("nan"))):
            values = dict(plan=plan(), source_sha256=worker_guard_source_sha256(), pid=12,
                          observed_unix=100, operator_verified_alive=True,
                          operator_verified_own_pod=True)
            values[field] = value
            with self.assertRaises(WorkerGuardError):
                GuardReceipt(**values)
        receipt = GuardReceipt(plan(), worker_guard_source_sha256(), 12, 100, True, True)
        self.assertEqual(GuardReceipt.from_dict(receipt.to_dict()), receipt)

    def test_worker_credential_uses_only_same_pod_injected_pair(self):
        with patch.dict(os.environ, dict(RUNPOD_API_KEY="fixture-scoped-key",  # pragma: allowlist secret - fake
                                        RUNPOD_POD_ID="fixture-pod"), clear=True):
            self.assertEqual(worker_credential(plan()), "fixture-scoped-key")
            with self.assertRaises(WorkerGuardError):
                worker_credential(plan(pod_id="foreign"))
        # No shell sourcing, laptop store, key creation, or export; PID1 bytes stay local.
        with patch.dict(os.environ, {}, clear=True), patch("os.open", return_value=42), \
                patch("os.read", return_value=b"RUNPOD_POD_ID=fixture-pod\0"
                    b"RUNPOD_API_KEY=fixture-scoped-key\0"), patch("os.close"):  # pragma: allowlist secret - fake
            self.assertEqual(worker_credential(plan()), "fixture-scoped-key")
            self.assertNotIn("RUNPOD_API_KEY", os.environ)

    def test_fixed_origin_owned_get_stop_get_and_foreign_refusal(self):
        own = dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
            CINEFORGE_LEASE="0"*32), mounts={}, status="RUNNING", actions=["stop"], runtime={})
        calls = []
        class Opener:
            def open(self, request, timeout):
                calls.append((request.method, request.full_url, request.data))
                pod = dict(own)
                if len(calls) >= 2:
                    pod.update(status="EXITED", runtime=None)
                return Reply(pod)
        with patch.dict(os.environ, dict(RUNPOD_API_KEY="fixture-scoped-key",  # pragma: allowlist secret - fake
            RUNPOD_POD_ID="fixture-pod"), clear=True), patch(
                "urllib.request.build_opener", return_value=Opener()):
            with self.assertRaises(WorkerGuardError):
                stop_own_pod(plan())
            self.assertTrue(verify_own_pod(plan(), allow_live=True))
            self.assertEqual([c[0] for c in calls], ["GET"])
            calls.clear()
            self.assertTrue(stop_own_pod(plan(), allow_live=True))
            self.assertEqual([c[0] for c in calls], ["GET", "POST", "GET"])
            self.assertEqual(calls[1][1], "https://api.runpod.io/v2/pods/fixture-pod/action")
            self.assertEqual(json.loads(calls[1][2]), dict(action="stop"))
            calls.clear()
            own["env"]["CINEFORGE_LEASE"] = "1"*32
            self.assertFalse(stop_own_pod(plan(), allow_live=True))
            self.assertEqual([c[0] for c in calls], ["GET"])

    def test_standalone_fake_guard_process_proves_no_orchestrator_dependency(self):
        code = '''import json
from cineforge.worker_guard import WorkerGuardPlan,run_guard
plan=WorkerGuardPlan("fixture-pod","fixture-session","0"*32,1,3)
print(json.dumps(run_guard(plan,lambda:0,lambda:True,clock=lambda:2,monotonic=lambda:3)))
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                timeout=5, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "compute_stop_confirmed")
