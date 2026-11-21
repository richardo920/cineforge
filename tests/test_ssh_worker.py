"""Offline SSH/process safety and guardian handshake fixtures; no SSH daemon needed."""

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from test_cloud_budget import Clock, Cloud, approval, selection
from test_runpod import spec
from test_worker_bootstrap import exercise_descendant_pipe, exercise_post_spawn_clock_failure

from cineforge.cloud_budget import CloudBudgetSession
from cineforge.cloud_smoke import _provider
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.runpod import LeaseSnapshot, WorkerObservation
from cineforge.runpod_transport import CloudResponse
from cineforge.ssh_worker import SSHWorker, WorkerError, run_process, ssh_environment
from cineforge.worker_bundle import worker_source
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

    def test_ping_validates_timeout_and_uses_fresh_unsupervised_requests(self):
        worker = self.worker(allow_live=True)
        self.addCleanup(worker.close)
        with patch.object(worker, "_request", return_value={"status": "ssh_ready"}) as request:
            for timeout in (0, -1, True, None, "1", float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(WorkerError):
                    worker.ping(timeout)
            request.assert_not_called()
            self.assertTrue(worker.ping(1))
            self.assertTrue(worker.ping(2.5))
            self.assertEqual(request.call_count, 2)
            request.assert_called_with({"operation": "ping"}, 2.5, supervise=False)
        for reply in ({}, {"status": "heartbeat"}, {"status": "ssh_ready", "extra": True}):
            with self.subTest(reply=reply), patch.object(worker, "_request", return_value=reply):
                self.assertFalse(worker.ping(1))
        with patch.object(worker, "_request", side_effect=WorkerError("private")):
            self.assertFalse(worker.ping(1))
        with patch.object(worker, "_request", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                worker.ping(1)

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

    def test_absent_connection_metadata_is_unready_without_ssh_or_arm(self):
        original = copy.deepcopy(self.transport.pod)
        changes = [lambda pod: pod.pop("ssh"), lambda pod: pod.update(ssh=None),
                   lambda pod: pod.update(ssh={}), lambda pod: pod["ssh"].update(direct=None)]
        for change in changes:
            with self.subTest(change=change):
                self.transport.pod = copy.deepcopy(original)
                change(self.transport.pod)
                worker = self.worker(allow_live=True)
                with patch.object(worker, "_request") as request:
                    self.assertIsNone(worker.probe(LeaseSnapshot("first", "created", "pod_1"), 5))
                request.assert_not_called()
                self.assertIsNone(worker.plan)
                self.assertEqual((worker._host, worker._port), ("", 0))

    def test_malformed_connection_metadata_is_static_and_clears_stale_endpoint(self):
        original = copy.deepcopy(self.transport.pod)
        canary = "private-api-host-canary.invalid;secret"
        changes = [lambda pod: pod["ssh"]["direct"].update(host=canary),
                   lambda pod: pod.update(ssh="private-api-ssh-canary"),
                   lambda pod: pod["ssh"].update(direct=[]),
                   lambda pod: pod["ssh"]["direct"].pop("port"),
                   lambda pod: pod["ssh"]["direct"].update(host=134744072),
                   lambda pod: pod.update(id="foreign"),
                   lambda pod: pod["env"].update(CINEFORGE_LEASE="foreign")]
        for change in changes:
            with self.subTest(change=change):
                self.transport.pod = copy.deepcopy(original)
                worker = self.worker(allow_live=True)
                self.assertTrue(worker._connection(LeaseSnapshot("first", "created", "pod_1"), 5))
                change(self.transport.pod)
                with patch("cineforge.ssh_worker.run_process") as run:
                    with self.assertRaises(WorkerError) as cm:
                        worker._connection(LeaseSnapshot("first", "created", "pod_1"), 5)
                diagnostic = "".join(traceback.format_exception(cm.exception))
                self.assertNotIn("private-api", diagnostic)
                self.assertIsNone(cm.exception.__context__)
                self.assertEqual((worker._host, worker._port), ("", 0))
                run.assert_not_called()

    def test_connection_timeout_is_lease_bounded_and_late_read_cannot_arm(self):
        worker = self.worker(allow_live=True)
        self.budget.remaining = 2
        original = self.transport.request
        clock = Clock()
        def late(*args, **kwargs):
            self.assertEqual(kwargs["timeout_seconds"], 2)
            reply = original(*args, **kwargs)
            clock.sleep(3)
            return reply
        with patch.object(self.transport, "request", side_effect=late), \
                patch("cineforge.ssh_worker.time.monotonic", side_effect=clock.monotonic), \
                patch.object(worker, "_request") as request:
            self.assertIsNone(worker.probe(LeaseSnapshot("first", "created", "pod_1"), 5))
        self.assertEqual((worker._host, worker._port), ("", 0))
        request.assert_not_called()

    def test_connection_cannot_use_a_snapshot_foreign_to_its_guard_plan(self):
        self.transport.pod["id"] = "other-pod"
        with patch("cineforge.ssh_worker.run_process") as run:
            with self.assertRaises(WorkerError):
                self.worker(allow_live=True)._connection(
                    LeaseSnapshot("first", "created", "other-pod"), 5)
        run.assert_not_called()

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

    def test_multicast_and_scoped_v4_v6_addresses_reject_before_ssh(self):
        addresses = ["224.0.0.1", "239.255.255.250", "ff02::1", "ff0e::1",
                     "::ffff:224.0.0.1", "::ffff:239.255.255.250",
                     "2606:4700:4700::1111%en0", "2606:4700:4700::1111%x;id",
                     "::ffff:8.8.8.8%en0"]
        for address in addresses:
            with self.subTest(address=address), patch("cineforge.ssh_worker.run_process") as run:
                self.transport.pod["ssh"]["direct"]["host"] = address
                worker = self.worker(allow_live=True)
                with self.assertRaises(WorkerError):
                    worker._connection(LeaseSnapshot("first", "created", "pod_1"), 5)
                self.assertEqual(worker._host, "")
                run.assert_not_called()

    def test_public_unicast_v4_v6_and_mapped_addresses_remain_supported(self):
        for address in ["8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"]:
            with self.subTest(address=address):
                self.transport.pod["ssh"]["direct"]["host"] = address
                self.assertTrue(self.worker(allow_live=True)._connection(
                    LeaseSnapshot("first", "created", "pod_1"), 5))

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
        with worker, patch.object(worker, "_request", side_effect=[{"status": "ssh_ready"}, WorkerError("fixture")]) as call:
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
        with self.assertRaises(KeyboardInterrupt):
            run_process([sys.executable, "-c", "print('done')"], timeout=2,
                        check=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    def test_exited_parent_detached_descendant_and_backpressure_cannot_hold_ssh_deadline(self):
        exercise_descendant_pipe(self, self.root, "cineforge.ssh_worker", "run_process")

    def test_post_spawn_clock_interruption_reaps_ssh_child_and_closes_pipes(self):
        exercise_post_spawn_clock_failure(self, run_process, KeyboardInterrupt)

    def test_only_native_fetch_has_a_larger_bounded_response(self):
        worker = self.worker(allow_live=True)
        worker._connection(LeaseSnapshot("first", "created", "pod_1"), 5)
        for operation in ("ping", "setup", "native_chunk", "native_setup", "native_fetch"):
            with self.subTest(operation=operation), patch("cineforge.ssh_worker.run_process",
                    return_value=b'{"status":"fixture"}') as call:
                deadline = time.monotonic() + 5
                worker._request({"operation": operation}, 5, supervise=False,
                                absolute_deadline=deadline)
                self.assertEqual(call.call_args.kwargs.get("max_output", 65536),
                                 256 * 1024 if operation == "native_fetch" else 65536)
                self.assertEqual(call.call_args.kwargs["absolute_deadline"], deadline)

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
                             hashlib.sha256(worker_source()).digest())
            self.assertIn("python3 -I -c ", command[-1])


if __name__ == "__main__":
    unittest.main()


class RetentionHeartbeatTests(unittest.TestCase):
    def test_detached_beats_keep_guard_live_past_idle_and_exit_at_expiry(self):
        from cineforge.retention_heartbeat import feed_until
        from cineforge.worker_guard import run_guard

        clock = Clock()
        clock.wall, clock.mono = 10000., 100.
        expiry = clock.wall + 600
        beats = []
        feed_until(expiry, lambda timeout: beats.append(clock.time()),
                   clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertEqual(beats, list(range(10000, 10600, 10)))
        self.assertEqual(clock.time(), expiry)
        clock.wall, clock.mono = 10000., 100.
        stops = []
        plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, expiry - 60, 120)
        def latest():
            return max(beat for beat in beats if beat <= clock.time())
        result = run_guard(plan, latest, lambda: stops.append(clock.time()) or True,
            clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertEqual(result["reason"], "deadline")
        self.assertEqual(stops, [expiry - 60])
        self.assertGreater(stops[0], 10000 + 120)

    def test_heartbeat_failure_exits_and_guard_stops_on_idle(self):
        from cineforge.retention_heartbeat import feed_until
        from cineforge.worker_guard import run_guard

        clock = Clock()
        beats = []
        def beat(timeout):
            if len(beats) == 2:
                raise RuntimeError("fake SSH failure")
            beats.append(clock.time())
        with self.assertRaises(RuntimeError):
            feed_until(11000, beat, clock=clock.time,
                       monotonic=clock.monotonic, sleep=clock.sleep)
        stops = []
        plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, 10940, 120)
        result = run_guard(plan, lambda: beats[-1],
            lambda: stops.append(clock.time()) or True, clock=clock.time,
            monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertEqual(result["reason"], "idle")
        self.assertEqual(stops, [beats[-1] + 120])

    def test_spawner_is_detached_private_and_injectable_without_running_a_child(self):
        import stat
        import subprocess
        from unittest.mock import Mock

        from cineforge.retention_heartbeat import spawn_heartbeat

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = Mock(pid=321)
            child.poll.return_value = None
            popen = Mock(return_value=child)
            pid = spawn_heartbeat(root, ["/usr/bin/ssh", "fake"], {"operation": "heartbeat"},
                                  14000, {"LANG": "C"}, popen=popen, settle_seconds=0)
            self.assertEqual(pid, 321)
            args, kwargs = popen.call_args
            self.assertEqual(args[0][:3], [sys.executable, "-I", "-c"])
            self.assertTrue(kwargs["start_new_session"])
            for stream in ("stdin", "stdout", "stderr"):
                self.assertEqual(kwargs[stream], subprocess.DEVNULL)
            path = root / "retention-heartbeat.json"
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            config = json.loads(path.read_bytes())
            self.assertEqual(config["expires_unix"], 14000)
            self.assertEqual(config["command"], ["/usr/bin/ssh", "fake"])
            self.assertEqual(config["retry_budget_seconds"], 60)
            self.assertEqual(config["stop_timeout_seconds"], 30)
            self.assertNotIn("budget", config)
            with self.assertRaises(ValueError):
                spawn_heartbeat(root, [], {}, float("inf"), {}, popen=popen)
            self.assertEqual(popen.call_count, 1)

    def test_clock_rollback_ends_feeding_without_another_beat(self):
        from cineforge.retention_heartbeat import feed_until

        clock, beats = Clock(), []
        def rollback(seconds):
            clock.sleep(seconds)
            clock.wall -= 100
        feed_until(11000, lambda timeout: beats.append(timeout), clock=clock.time,
                   monotonic=clock.monotonic, sleep=rollback)
        self.assertEqual(len(beats), 1)
        self.assertEqual(clock.mono, 110)

    def test_retry_transient_failure_continues_feed_and_persistent_failure_is_bounded(self):
        from types import SimpleNamespace

        from cineforge.retention_heartbeat import feed_until, make_beat

        for persistent in (False, True):
            clock, calls = Clock(), []
            config = {"command": ["fake"], "request": {"operation": "heartbeat"},
                "expires_unix": 10025., "retry_budget_seconds": 6}
            def run(command, *, calls=calls, clock=clock, persistent=persistent, **kwargs):
                calls.append(clock.time())
                if persistent or len(calls) <= 2:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                return SimpleNamespace(stdout=b'{"status":"heartbeat"}')
            beat = make_beat(config, run=run, clock=clock.time,
                             monotonic=clock.monotonic, sleep=clock.sleep)
            if persistent:
                with self.assertRaises(subprocess.TimeoutExpired):
                    beat(5)
                self.assertEqual(clock.time(), 10006)
                self.assertEqual(calls, [10000, 10002, 10004])
            else:
                feed_until(config["expires_unix"], beat, clock=clock.time,
                           monotonic=clock.monotonic, sleep=clock.sleep)
                self.assertEqual(calls, [10000, 10002, 10004, 10014, 10024])
                self.assertEqual(clock.time(), 10025)

    def test_child_main_retries_stop_until_confirmed_or_budget_exhausted(self):
        from types import SimpleNamespace

        from cineforge.retention_heartbeat import main

        plan = {"pod_id": "pod_1"}
        # Transient stop failure, then dispatch confirms the stop: two attempts.
        with tempfile.TemporaryDirectory() as tmp:
            clock, requests = Clock(), []
            path = Path(tmp) / "retention-heartbeat.json"
            path.write_text(json.dumps({"command": ["fake"], "request": {
                "operation": "heartbeat", "plan": plan, "guard_sha256": "a"*64},
                "expires_unix": 10025, "retry_budget_seconds": 3}))
            def run(command, *, requests=requests, **kwargs):
                request = json.loads(kwargs["input"])
                requests.append(request)
                if request["operation"] == "stop":
                    if len([r for r in requests if r["operation"] == "stop"]) == 1:
                        return SimpleNamespace(stdout=b'{"status":"stop_unconfirmed"}')
                    return SimpleNamespace(stdout=b'{"status":"stopped"}')
                return SimpleNamespace(stdout=b'{"status":"heartbeat"}')
            with patch.object(sys, "argv", ["heartbeat", str(path)]):
                main(run=run, clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep)
            stops = [request for request in requests if request["operation"] == "stop"]
            self.assertEqual(len(stops), 2)
            self.assertEqual(stops[0]["plan"], plan)
            self.assertEqual(stops[0]["guard_sha256"], "a"*64)
            self.assertEqual(stops[0]["timeout_seconds"], 30)
            self.assertLess(stops[1]["timeout_seconds"], 30)
            self.assertFalse(path.exists())
        # Persistent stop failure: retries until the stop budget is exhausted.
        for broken_heartbeat in (False, True):
            with self.subTest(broken_heartbeat=broken_heartbeat), tempfile.TemporaryDirectory() as tmp:
                clock, requests = Clock(), []
                path = Path(tmp) / "retention-heartbeat.json"
                path.write_text(json.dumps({"command": ["fake"], "request": {
                    "operation": "heartbeat", "plan": plan, "guard_sha256": "a"*64},
                    "expires_unix": 10025, "retry_budget_seconds": 3}))
                def failing_run(command, *, requests=requests,
                                broken_heartbeat=broken_heartbeat, **kwargs):
                    request = json.loads(kwargs["input"])
                    requests.append(request)
                    if broken_heartbeat or request["operation"] == "stop":
                        raise RuntimeError("synthetic failure")
                    return SimpleNamespace(stdout=b'{"status":"heartbeat"}')
                with patch.object(sys, "argv", ["heartbeat", str(path)]):
                    main(run=failing_run, clock=clock.time, monotonic=clock.monotonic,
                         sleep=clock.sleep)
                stops = [request for request in requests if request["operation"] == "stop"]
                self.assertGreater(len(stops), 1)
                self.assertEqual(stops[0]["plan"], plan)
                self.assertEqual(stops[0]["timeout_seconds"], 30)
                self.assertFalse(path.exists())

    def test_spawn_detects_early_exit_and_terminates_on_settle_interrupt(self):
        from unittest.mock import Mock

        from cineforge.retention_heartbeat import spawn_heartbeat

        with tempfile.TemporaryDirectory() as tmp:
            children = []
            def short_child(command, **kwargs):
                child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], **kwargs)
                children.append(child)
                return child
            with self.assertRaisesRegex(RuntimeError, "exited during handoff"):
                spawn_heartbeat(Path(tmp), [], {}, time.time()+100, {}, popen=short_child)
            self.assertEqual(children[0].wait(timeout=2), 3)
            self.assertFalse((Path(tmp) / "retention-heartbeat.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            child = Mock(pid=321)
            child.poll.return_value = None
            with patch("cineforge.retention_heartbeat.time.sleep", side_effect=KeyboardInterrupt), \
                    self.assertRaises(KeyboardInterrupt):
                spawn_heartbeat(Path(tmp), [], {}, time.time()+100, {}, popen=Mock(return_value=child))
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=2)
            self.assertFalse((Path(tmp) / "retention-heartbeat.json").exists())

    def test_spawn_failure_removes_config(self):
        from unittest.mock import Mock

        from cineforge.retention_heartbeat import spawn_heartbeat

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retention-heartbeat.json"
            with self.assertRaises(RuntimeError):
                spawn_heartbeat(Path(tmp), [], {}, time.time() + 100, {},
                                popen=Mock(side_effect=RuntimeError("synthetic spawn failure")))
            self.assertFalse(path.exists())

    def test_real_child_feeds_real_guard_then_stops_on_expiry_or_guard_death(self):
        """Real run_guard/heartbeat; /proc PID gate has separate faked-/proc unit coverage.

        CLI credential preflight cannot enter run_guard on macOS; invoke the real
        loop in a subprocess with its real credential-failing stop callback instead.
        """
        from cineforge import worker_guard
        from cineforge.retention_heartbeat import spawn_heartbeat
        from cineforge.worker_bootstrap import heartbeat, private_write

        environment = {"PATH": os.environ.get("PATH", ""),
                       "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        for killed in (False, True):
            with self.subTest(killed=killed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, time.time()+300, 12)
                private_write(root, "plan.json", json.dumps(plan.to_dict()).encode())
                private_write(root, "guard.py", Path(worker_guard.__file__).read_bytes())
                heartbeat(root, plan.to_dict(), first=True)
                marker, ready = root / "stop-marker.json", root / "guard-ready"
                guard_code = (
                    "import importlib.util,sys,json;from pathlib import Path;"
                    "root=Path(sys.argv[1]);"
                    "spec=importlib.util.spec_from_file_location('fixture_guard',root/'guard.py');"
                    "g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;"
                    "spec.loader.exec_module(g);"
                    "p=g.WorkerGuardPlan.from_dict(json.loads((root/'plan.json').read_text()));"
                    "assert not g.stop_own_pod(p,allow_live=True);"
                    "(root/'guard-ready').touch();"
                    "g.run_guard(p,lambda:g.read_heartbeat(str(root/'heartbeat.json'),p),"
                    "lambda:g.stop_own_pod(p,allow_live=True),poll_seconds=.1,max_stop_attempts=1)")
                guard = subprocess.Popen([sys.executable, "-c", guard_code, str(root)],
                    env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    start_new_session=True)
                children = []
                try:
                    deadline = time.monotonic()+5
                    while not ready.exists() and guard.poll() is None and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertTrue(ready.exists(), guard.stderr.read() if guard.poll() is not None else b'')
                    beat_code = (
                        "import json,os,sys;from pathlib import Path;"
                        "from cineforge.worker_bootstrap import heartbeat;"
                        "r=json.load(sys.stdin);root=Path(sys.argv[1])\n"
                        "if r['operation']=='stop':\n"
                        " (root/'stop-marker.json').write_text(json.dumps(r))\n"
                        " print(json.dumps({'status':'stopped'}))\n"
                        "else:\n"
                        " os.kill(int(sys.argv[2]),0)\n"
                        " heartbeat(root,r['plan'])\n"
                        " print(json.dumps({'status':'heartbeat'}))\n")
                    def popen(command, *, children=children, **kwargs):
                        child = subprocess.Popen(command, **kwargs)
                        children.append(child)
                        return child
                    request = {"operation": "heartbeat", "plan": plan.to_dict(),
                        "allow_live": True, "guard_sha256": worker_guard_source_sha256()}
                    spawn_heartbeat(root, [sys.executable, "-c", beat_code, str(root), str(guard.pid)],
                        request, time.time()+16 if not killed else time.time()+60, environment,
                        popen=popen, retry_budget_seconds=3, settle_seconds=.1)
                    child = children[0]
                    if killed:
                        os.killpg(guard.pid, signal.SIGKILL)
                        guard.wait(timeout=2)  # Reap: kill(pid, 0) must fail, not see a zombie.
                        child.wait(timeout=20)
                    else:
                        time.sleep(12.5)
                        self.assertIsNone(guard.poll())  # Beyond idle, sustained by real beats.
                        child.wait(timeout=8)
                        guard.wait(timeout=12)
                    self.assertEqual(child.returncode, 0)
                    self.assertTrue(marker.exists())
                    self.assertEqual(json.loads(marker.read_bytes())["plan"], plan.to_dict())
                    self.assertFalse((root / "retention-heartbeat.json").exists())
                finally:
                    for process in [*children, guard]:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=2)
                    guard.stderr.close()

    def test_real_child_stop_reaches_real_dispatch_and_stops_own_pod(self):
        """The stop seam: real worker_command_source -> real main/dispatch -> real stop_own_pod.

        Only the on-pod HTTP layer is stubbed (urllib opener + pod-scoped credential
        env). /workspace is not writable on the test host, so the bundle is loaded
        with a neutral __name__ and WORKSPACE_ROOT is redirected to a temp dir
        before main() runs; the rendered child command itself is unmodified.
        """
        from cineforge import worker_guard
        from cineforge.worker_bootstrap import encode

        driver = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, '__SRC__')\n"
            "import urllib.request\n"
            "_LOG = Path('__LOG__')\n"
            "_STATE = {'stopped': False}\n"
            "class _Response:\n"
            "    def __init__(self, payload): self._payload = payload\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, *args): return False\n"
            "    def read(self, size=-1): return self._payload\n"
            "class _Opener:\n"
            "    def open(self, request, timeout=None):\n"
            "        method = request.get_method()\n"
            "        with _LOG.open('a') as stream:\n"
            "            stream.write(json.dumps({'method': method, 'url': request.full_url,"
            " 'data': request.data.decode() if request.data else None}) + chr(10))\n"
            "        pod = {'id': 'pod_1', 'env': {'CINEFORGE_SESSION': 'fixture',"
            " 'CINEFORGE_LEASE': 'a'*32}, 'mounts': {}, 'actions': []}\n"
            "        if method == 'POST':\n"
            "            _STATE['stopped'] = True\n"
            "            pod['status'] = 'STOPPING'\n"
            "        elif _STATE['stopped']:\n"
            "            pod['status'] = 'EXITED'\n"
            "        else:\n"
            "            pod['status'] = 'RUNNING'\n"
            "            pod['actions'] = ['stop']\n"
            "        return _Response(json.dumps(pod).encode())\n"
            "urllib.request.build_opener = lambda *args, **kwargs: _Opener()\n"
            "from cineforge.worker_transport_source import worker_command_source as _s\n"
            "from cineforge.worker_bundle import worker_source as _b\n"
            "import types as _t\n"
            "_m = _t.ModuleType('cineforge_worker_bundle')\n"
            "sys.modules['cineforge_worker_bundle'] = _m\n"
            "_g = _m.__dict__\n"
            "_g['__name__'] = 'cineforge_worker_bundle'\n"
            "exec(_s(_b()), _g)\n"
            "_g['WORKSPACE_ROOT'] = Path('__WS__')\n"
            "raise SystemExit(_g['main']())\n"
        )
        for tampered in (False, True):
            with self.subTest(tampered=tampered), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp).resolve() / "workspace"
                root = workspace / "cineforge-m3-fixture"
                root.mkdir(parents=True, mode=0o700)
                plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, time.time() + 300, 120)
                (root / "guard.py").write_bytes(Path(worker_guard.__file__).read_bytes())
                (root / "guard.py").chmod(0o600)
                (root / "plan.json").write_bytes(encode(plan.to_dict()))
                (root / "plan.json").chmod(0o600)
                log = Path(tmp) / "http-log.jsonl"
                request = {"operation": "stop",
                           "plan": dict(plan.to_dict(), pod_id="foreign") if tampered
                           else plan.to_dict(),
                           "allow_live": True, "timeout_seconds": 30,
                           "guard_sha256": worker_guard_source_sha256()}
                environment = {"PATH": os.environ.get("PATH", ""), "LANG": "C",
                               "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                               "RUNPOD_API_KEY": "fixture-key-1", "RUNPOD_POD_ID": "pod_1"}  # pragma: allowlist secret - synthetic environment fixture
                child = subprocess.Popen(
                    [sys.executable, "-c", driver.replace("__SRC__", environment["PYTHONPATH"])
                     .replace("__WS__", str(workspace)).replace("__LOG__", str(log))],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=environment)
                try:
                    stdout, stderr = child.communicate(json.dumps(request).encode(), timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    stdout, stderr = child.communicate()
                    self.fail("real stop dispatch timed out: %r" % stderr)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(json.loads(stdout),
                                 {"status": "stop_unconfirmed" if tampered else "stopped"})
                calls = [json.loads(line) for line in log.read_text().splitlines()] \
                    if log.exists() else []
                if tampered:
                    self.assertEqual(calls, [])
                else:
                    self.assertEqual([call["method"] for call in calls], ["GET", "POST", "GET"])
                    self.assertEqual(calls[0]["url"], "https://api.runpod.io/v2/pods/pod_1")
                    self.assertEqual(calls[1]["url"], "https://api.runpod.io/v2/pods/pod_1/action")
                    self.assertEqual(calls[1]["data"], '{"action":"stop"}')

    def test_probe_retention_guard_reports_heartbeat_and_failures(self):
        from threading import Lock
        from unittest.mock import Mock

        worker = object.__new__(SSHWorker)
        worker._remote_lock = Lock()
        plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, time.time() + 300, 120)
        worker._request = Mock(return_value={"status": "heartbeat"})
        self.assertTrue(worker.probe_retention_guard(plan))
        worker._request.assert_called_once_with(
            {"operation": "heartbeat", "plan": plan.to_dict()}, 5, supervise=False)
        worker._request = Mock(side_effect=WorkerError("fake SSH failure"))
        self.assertFalse(worker.probe_retention_guard(plan))
        worker._request = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            worker.probe_retention_guard(plan)
        self.assertFalse(worker.probe_retention_guard("not-a-plan"))

    def test_failed_rearm_does_not_trust_prearm_old_guard_heartbeat(self):
        from threading import Lock
        from unittest.mock import Mock

        worker = object.__new__(SSHWorker)
        worker._remote_lock = Lock()
        worker.check = Mock()
        worker._key = "first"
        worker.plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, 11740, 120)
        updated = WorkerGuardPlan("pod_1", "fixture", "a"*32, 15340, 120)
        worker.budget = Mock()
        worker.budget.remaining_seconds.return_value = 1000
        worker.budget.guard_plan.return_value = updated
        worker._request = Mock(side_effect=[{"status": "heartbeat"},
            WorkerError("new guard died after retiring old guard")])
        with patch("cineforge.ssh_worker.time.time", return_value=10000):
            with self.assertRaisesRegex(WorkerError, "new guard died"):
                worker.retain_live({"retention_expires_unix": 15400.})
        self.assertEqual([call.args[0]["operation"] for call in worker._request.call_args_list],
                         ["heartbeat", "arm"])
        self.assertIsNone(worker.retention_guard_alive)
        self.assertIsNone(worker.retention_guard_stop_at_unix)
        worker.budget.arm_guard.assert_not_called()

    def test_retention_records_pid_and_alive_before_suspending_supervision(self):
        from threading import Event, Lock
        from unittest.mock import Mock

        worker = object.__new__(SSHWorker)
        worker._remote_lock = Lock()
        worker._remote_suspended = Event()
        worker.check = Mock()
        worker._failed, worker._stop, worker._tunnel = Event(), Event(), None
        worker._key = "first"
        worker.plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, 11740, 120)
        updated = WorkerGuardPlan("pod_1", "fixture", "a"*32, 15340, 120)
        worker.budget = Mock()
        worker.budget.remaining_seconds.return_value = 1000
        worker.budget.guard_plan.return_value = updated
        worker._profile = None
        worker.selection = None
        worker._source = worker_source()
        worker._host, worker._port = "8.8.8.8", 2222
        worker.private_key = Path("/synthetic-key")
        worker.state_dir = Path("/synthetic-private-worker")
        receipt = GuardReceipt(updated, worker_guard_source_sha256(), 123,
                               time.time(), True, True)
        worker._request = Mock(side_effect=[{"status": "heartbeat"},
                                            {"receipt": receipt.to_dict()},
                                            {"status": "heartbeat"}])
        worker._remote_suspended.set = Mock(side_effect=KeyboardInterrupt)
        facts = {"retention_expires_unix": 15400.}
        with patch("cineforge.ssh_worker.time.time", return_value=10000):
            with self.assertRaises(KeyboardInterrupt):
                worker.retain_live(facts, spawner=lambda *args: 777)
        self.assertTrue(worker.retention_guard_alive)
        self.assertEqual(worker.retention_heartbeat_pid, 777)

    def test_worker_rearms_before_immediate_beat_and_detached_handoff(self):
        self._exercise_retention_handoff(False)

    def test_failed_supervisor_can_rearm_with_a_live_lease(self):
        self._exercise_retention_handoff(True)

    def _exercise_retention_handoff(self, failed: bool) -> None:
        from threading import Lock
        from unittest.mock import Mock

        worker = object.__new__(SSHWorker)
        worker._remote_lock = Lock()
        worker._remote_suspended = Event()
        worker.check = Mock(side_effect=WorkerError("old guard died") if failed else None)
        worker._failed, worker._stop, worker._tunnel = Event(), Event(), None
        if failed:
            worker._failed.set()
        worker._key = "first"
        worker.plan = WorkerGuardPlan("pod_1", "fixture", "a"*32, 11740, 120)
        updated = WorkerGuardPlan("pod_1", "fixture", "a"*32, 15340, 120)
        worker.budget = Mock()
        worker.budget.remaining_seconds.return_value = 1000
        worker.budget.guard_plan.return_value = updated
        worker._profile = None
        worker.selection = None
        worker._source = worker_source()
        worker._host, worker._port = "8.8.8.8", 2222
        worker.private_key = Path("/synthetic-key")
        worker.state_dir = Path("/synthetic-private-worker")
        receipt = GuardReceipt(updated, worker_guard_source_sha256(), 123,
                               time.time(), True, True)
        worker._request = Mock(side_effect=[{"status": "heartbeat"}, {"receipt": receipt.to_dict()},
                                            {"status": "heartbeat"}])
        def spawn(directory, command, request, expiry, environment):
            worker.budget.arm_guard.assert_called_once_with("first", receipt)
            self.assertEqual(worker.plan, updated)
            self.assertEqual(worker._request.call_count, 3)
            self.assertEqual(request["plan"], updated.to_dict())
            self.assertEqual(expiry, 15400)
            self.assertEqual(command[0], "/usr/bin/ssh")
            self.assertIn("UserKnownHostsFile=/synthetic-private-worker/known-hosts", command)
            self.assertEqual(environment, ssh_environment())
            return 456
        facts = {"retention_expires_unix": 15400.}
        with patch("cineforge.ssh_worker.time.time", return_value=10000):
            result = worker.retain_live(facts, spawner=spawn)
        self.assertTrue(worker._remote_suspended.is_set())
        self.assertEqual(result["retention_heartbeat_pid"], 456)
        self.assertEqual(result["guard_stop_at_unix"], 15340)
        self.assertTrue(worker.retention_guard_alive)
        self.assertEqual(worker.retention_guard_stop_at_unix, 15340)
        worker.budget.extend_retention.assert_not_called()
