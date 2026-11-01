"""Offline SSH/process safety and guardian handshake fixtures; no SSH daemon needed."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from test_cloud_budget import Clock, Cloud, approval, selection
from test_runpod import spec

from cineforge.cloud_budget import CloudBudgetSession
from cineforge.cloud_smoke import _provider
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.runpod import LeaseSnapshot, WorkerObservation
from cineforge.runpod_transport import CloudResponse
from cineforge.ssh_worker import SSHWorker, WorkerError, run_process, ssh_environment
from cineforge.worker_guard import GuardReceipt, WorkerGuardPlan, worker_guard_source_sha256
from cineforge.workflows import canonical


class Budget:
    def __init__(self):
        self.plan = WorkerGuardPlan("pod_1", spec().session_id, "a"*32, time.time()+100, 30)
        self.receipts = []
        self.beats = []
        self.remaining = 100

    def guard_plan(self, key):
        return self.plan

    def arm_guard(self, key, receipt):
        if receipt.plan != self.plan or receipt.source_sha256 != worker_guard_source_sha256():
            raise ValueError("mismatched fixture guard")
        self.receipts.append(receipt)

    def heartbeat(self, key):
        self.beats.append(key)

    def remaining_seconds(self, key):
        return self.remaining


class Transport:
    def __init__(self):
        self.pod = {"id": "pod_1", "status": "RUNNING", "dataCenterId": "US-IL-1",
            "image": spec().image, "mounts": {}, "env": {"CINEFORGE_SESSION": spec().session_id,
            "CINEFORGE_LEASE": "a"*32}, "ssh": {"direct": {"host": "8.8.8.8", "port": 22000,
            "username": "root", "command": "untrusted provider command; never execute"}}}

    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        assert method == "GET" and path == "/v2/pods/pod_1"
        return CloudResponse(200, canonical(self.pod))


class SSHTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-ssh-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.key = self.root / "synthetic-key"
        self.key.write_text("synthetic fixture, not a private key")
        self.key.chmod(0o600)
        self.budget, self.transport = Budget(), Transport()

    def worker(self, **options):
        return SSHWorker(spec(), selection(), self.budget, self.transport, self.key,
                         self.root / "state", **options)

    def test_explicit_opt_in_and_host_port_user_foreign_metadata_rejection(self):
        with self.assertRaises(WorkerError):
            self.worker()
        original = copy.deepcopy(self.transport.pod)
        changes = [lambda p: p["ssh"]["direct"].update(host="127.0.0.1"),
                   lambda p: p["ssh"]["direct"].update(host="169.254.169.254"),
                   lambda p: p["ssh"]["direct"].update(host="evil.invalid; touch /tmp/injected"),
                   lambda p: p["ssh"]["direct"].update(port=True),
                   lambda p: p["ssh"]["direct"].update(username="root;id"),
                   lambda p: p["env"].update(CINEFORGE_LEASE="foreign"),
                   lambda p: p.update(dataCenterId=None)]
        for change in changes:
            self.transport.pod = copy.deepcopy(original)
            change(self.transport.pod)
            with self.assertRaises((WorkerError, ValueError)):
                self.worker(allow_live=True)._connection(LeaseSnapshot("first", "created", "pod_1"), 5)

    def test_native_components_not_provider_command_and_private_ssh_options(self):
        worker = self.worker(allow_live=True)
        self.assertTrue(worker._connection(LeaseSnapshot("first", "created", "pod_1"), 5))
        command = worker._command()
        self.assertIn("StrictHostKeyChecking=accept-new", command)
        self.assertIn("IdentityAgent=none", command)
        self.assertIn("SendEnv=-*", command)
        self.assertIn("/dev/null", command)
        self.assertFalse(any("untrusted provider command" in value for value in command))
        self.assertNotIn("RUNPOD_API_KEY", ssh_environment())
        self.assertNotIn("SSH_AUTH_SOCK", ssh_environment())

    def test_guard_receipt_source_own_id_pid_verified_before_observation(self):
        worker = self.worker(allow_live=True)
        worker.set_enforcer(lambda: ())
        plan = self.budget.plan
        response = {"receipt": {"plan": plan.to_dict(), "source_sha256": worker_guard_source_sha256(),
            "pid": 123, "observed_unix": time.time(), "operator_verified_alive": True,
            "operator_verified_own_pod": True}, "runtime": {"gpu_memory_mib": 24082,
            "gpu_raw_name": spec().gpu_id}}
        with worker, patch.object(worker, "_request", side_effect=[{"status": "ssh_ready"}, response]):
            result = worker.probe(LeaseSnapshot("first", "created", "pod_1"), 30)
            self.assertEqual(result.device.memory_mib, 24082)
            self.assertEqual(len(self.budget.receipts), 1)
            self.assertEqual(self.budget.beats, ["first"])
            self.assertIsNotNone(worker._thread)
        self.assertFalse(worker._thread.is_alive())

    def test_arm_lost_ack_is_not_retried_and_unguarded_worker_cannot_prepare(self):
        worker = self.worker(allow_live=True)
        with worker, patch.object(worker, "_request", side_effect=[{}, WorkerError("fixture")]) as call:
            with self.assertRaises(WorkerError):
                worker.probe(LeaseSnapshot("first", "created", "pod_1"), 30)
            self.assertEqual(call.call_count, 2)  # one read-only ping, one ARM
            self.assertIsNone(worker.plan)
            with self.assertRaises(WorkerError):
                worker.prepare()

    def test_failed_connectivity_probe_is_read_only_and_retriable(self):
        worker = self.worker(allow_live=True)
        with worker, patch.object(worker, "_request", side_effect=WorkerError("unreachable")) as call:
            self.assertIsNone(worker.probe(LeaseSnapshot("first", "created", "pod_1"), 5))
            self.assertEqual(call.call_args.args[0]["operation"], "ping")
            self.assertFalse(self.budget.receipts)

    def test_failed_remote_heartbeat_blocks_work_and_does_not_extend_deadline(self):
        worker = self.worker(allow_live=True)
        worker.plan, worker._key = self.budget.plan, "first"
        worker._failed.set()
        with self.assertRaises(WorkerError):
            worker.check()
        worker._failed.clear()
        self.budget.remaining = 0
        with self.assertRaises(WorkerError):
            worker.check()
        self.assertFalse(self.budget.beats)

    def test_enforcer_runs_before_continued_setup_or_generation(self):
        worker = self.worker(allow_live=True)
        calls = []
        worker.set_enforcer(lambda: calls.append("enforce") or ("expired",))
        with self.assertRaises(WorkerError):
            worker.check()
        self.assertEqual(calls, ["enforce"])

    def test_remote_heartbeat_loss_keeps_local_supervision_and_enforcer_active(self):
        worker = self.worker(allow_live=True)
        worker.plan, worker._key = self.budget.plan, "first"
        enforcements = []
        worker.set_enforcer(lambda: enforcements.append("tick") or ())
        self.assertEqual(worker.suspend_remote_heartbeat(), self.budget.plan)
        with patch.object(worker._stop, "wait", side_effect=[False, True]), \
                patch.object(worker, "_request") as request:
            worker._supervise()
        request.assert_not_called()
        self.assertEqual(self.budget.beats, ["first"])
        self.assertTrue(enforcements)
        self.assertFalse(worker._failed.is_set())

    def test_remote_heartbeat_request_failure_blocks_subsequent_work(self):
        worker = self.worker(allow_live=True)
        worker.plan, worker._key = self.budget.plan, "first"
        with patch.object(worker._stop, "wait", return_value=False), \
                patch.object(worker, "_request", side_effect=WorkerError("fixture failed")):
            worker._supervise()
        self.assertTrue(worker._failed.is_set())
        self.assertFalse(self.budget.beats)
        with self.assertRaises(WorkerError):
            worker.check()

    def test_setup_reserves_inference_time_inside_the_same_lease_cutoff(self):
        worker = self.worker(allow_live=True)
        worker.plan, worker._key = self.budget.plan, "first"
        with patch.object(worker, "_request") as request:
            with self.assertRaises(WorkerError):
                worker.prepare()  # only100 seconds remain; no setup starts
        request.assert_not_called()
        self.budget.remaining = 500
        with patch.object(worker, "_request", side_effect=WorkerError("synthetic remote failure")) as call:
            with self.assertRaises(WorkerError):
                worker.prepare()
        self.assertEqual(call.call_args.args[1], 380)

    def test_real_driver_enforcer_and_acquire_probe_do_not_invert_locks_at_cutoff(self):
        clock = Clock()
        cloud = Cloud(clock)
        probe_entered, release_probe, release_entered = Event(), Event(), Event()
        errors = []
        with CloudBudgetSession(self.root / "journal", approval(), cloud,
                guard_deployment_configured=True, clock=clock.time,
                monotonic=clock.monotonic) as budget:
            worker = SSHWorker(spec(), selection(), budget, cloud, self.key,
                               self.root / "concurrent-state", allow_live=True)
            def probe(snapshot, timeout):
                plan = budget.guard_plan(snapshot.key)
                budget.arm_guard(snapshot.key, GuardReceipt(plan, worker_guard_source_sha256(),
                    123, clock.time(), True, True))
                worker.plan, worker._key = plan, snapshot.key
                probe_entered.set()
                if not release_probe.wait(3):
                    raise ValueError("test probe release timed out")
                return WorkerObservation(GPUDevice(snapshot.resource_id, Backend.CUDA, 24082,
                    frozenset({Precision.FP16})), spec().gpu_id)
            provider = _provider(spec(), cloud, budget, probe)
            worker.set_enforcer(lambda: budget.enforce_limits(provider))
            original_release = budget._release
            def marked_release(*args):
                release_entered.set()
                return original_release(*args)
            def acquire():
                try:
                    provider.acquire(selection().model, max_cost_usd=.51, idempotency_key="threaded")
                except Exception as error:
                    errors.append(type(error).__name__)
            def enforce():
                try:
                    clock.sleep(1741)
                    worker.check()
                except Exception as error:
                    errors.append(type(error).__name__)
            first = Thread(target=acquire, daemon=True)
            second = Thread(target=enforce, daemon=True)
            with patch.object(budget, "_release", side_effect=marked_release):
                first.start()
                self.assertTrue(probe_entered.wait(2))
                second.start()
                self.assertTrue(release_entered.wait(2))
                release_probe.set()
                first.join(3)
                second.join(3)
            self.assertFalse(first.is_alive(), "acquisition deadlocked on budget/provisioner locks")
            self.assertFalse(second.is_alive(), "driver supervision deadlocked on provider lock")
            self.assertEqual(budget.snapshots[0].state, "released")
            self.assertFalse(cloud.pods)
            self.assertEqual(sum(call[0] == "POST" for call in cloud.calls), 1)
            self.assertEqual(sum(call[0] == "DELETE" for call in cloud.calls), 1)
            self.assertIn("WorkerError", errors)

    def test_subprocess_timeout_output_and_local_environment_bounds(self):
        self.assertEqual(run_process([sys.executable, "-c", "print('fixture')"], timeout=2),
                         b"fixture\n")
        started = time.monotonic()
        with self.assertRaises(WorkerError):
            run_process([sys.executable, "-c", "import time;time.sleep(30)"], timeout=.1)
        self.assertLess(time.monotonic()-started, 3)
        with self.assertRaises(WorkerError):
            run_process([sys.executable, "-c", "print('x'*100000)"], timeout=2, max_output=100)
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "fixture-sensitive-value"}):  # pragma: allowlist secret - synthetic environment fixture
            output = run_process([sys.executable, "-c",
                "import os;print(os.environ.get('RUNPOD_API_KEY','absent'))"], timeout=2)
            self.assertEqual(output, b"absent\n")
        with self.assertRaises(WorkerError):
            run_process([sys.executable, "-c", "print('done')"], timeout=2,
                        check=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    def test_source_and_payload_are_separate_not_shell_interpolated(self):
        worker = self.worker(allow_live=True)
        worker._connection(LeaseSnapshot("first", "created", "pod_1"), 5)
        payload = {"operation": "ping", "synthetic": "$(touch /tmp/should-not-exist)"}
        with patch("cineforge.ssh_worker.run_process", return_value=b'{"status":"fixture"}') as call:
            worker._request(payload, 5, supervise=False)
            command, body = call.call_args.args[0], call.call_args.kwargs["payload"]
            self.assertNotIn(payload["synthetic"], " ".join(command))
            self.assertEqual(json.loads(body)["synthetic"], payload["synthetic"])
            self.assertEqual(hashlib.sha256(worker._source).digest(),
                             hashlib.sha256(Path("src/cineforge/worker_bootstrap.py").read_bytes()).digest())


if __name__ == "__main__":
    unittest.main()
