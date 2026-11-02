"""Standalone guard tests: fake clocks/transports/process, never a live GPU or credential."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cineforge import worker_guard
from cineforge.worker_guard import (
    GuardReceipt,
    WorkerGuardError,
    WorkerGuardPlan,
    pod_exited,
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
        with tempfile.TemporaryDirectory(prefix="cineforge-heartbeat-") as tmp:
            path = Path(tmp).resolve() / "heartbeat"
            data = dict(schema_version=1, pod_id=plan().pod_id, lease_token=plan().lease_token,
                        heartbeat_unix=100)
            path.write_text(json.dumps(data))
            path.chmod(0o600)
            self.assertEqual(read_heartbeat(path, plan()), 100)
            path.chmod(0o644)
            with self.assertRaises(WorkerGuardError):
                read_heartbeat(path, plan())
            path.chmod(0o600)
            linked = Path(tmp).resolve() / "linked"
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

    def test_exited_classifier_ignores_retained_runtime_but_never_guesses_other_states(self):
        self.assertTrue(pod_exited("EXITED"))
        for status in ("RUNNING", "STARTING", "PROVISIONING", "ERROR", "TERMINATED", None, True, {}):
            self.assertFalse(pod_exited(status))

    def test_owned_guard_stop_accepts_exited_with_runtime_present_and_keeps_ownership(self):
        own = dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
            CINEFORGE_LEASE="0"*32), mounts={}, status="RUNNING", actions=["stop"], runtime={})
        calls = []
        def request(method, stop=False, timeout_seconds=3):
            calls.append((method, stop))
            return dict(own, status="EXITED" if len(calls) >= 2 else own["status"])
        with patch("cineforge.worker_guard._worker_request", return_value=request):
            self.assertTrue(stop_own_pod(plan(), allow_live=True))
            self.assertEqual(calls, [("GET", False), ("POST", True), ("GET", False)])
            calls.clear()
            own["status"] = "EXITED"
            own["env"]["CINEFORGE_LEASE"] = "foreign"
            self.assertFalse(stop_own_pod(plan(), allow_live=True))
            self.assertEqual(calls, [("GET", False)])

    def test_delayed_exited_is_polled_and_stopping_does_not_repeat_stop(self):
        for initial in ("RUNNING", "STOPPING"):
            with self.subTest(initial=initial):
                calls = []
                reads = 0
                def request(method, stop=False, timeout_seconds=3):
                    nonlocal reads
                    calls.append((method, stop, timeout_seconds))
                    if method == "POST":
                        return {}
                    reads += 1
                    status = initial if reads == 1 else "STOPPING" if reads == 2 else "EXITED"
                    return dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                        CINEFORGE_LEASE="0"*32), mounts={}, status=status, actions=["stop"])
                with patch("cineforge.worker_guard._worker_request", return_value=request):
                    self.assertTrue(stop_own_pod(plan(), allow_live=True))
                self.assertEqual([c[0] for c in calls].count("POST"), int(initial == "RUNNING"))
                self.assertEqual(reads, 3)

    def test_already_exited_and_foreign_terminal_stop_without_post(self):
        for foreign in (False, True):
            calls = []
            def request(method, stop=False, timeout_seconds=3):
                calls.append(method)
                return dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                    CINEFORGE_LEASE="foreign" if foreign else "0"*32), mounts={},
                    status="EXITED", actions=[])
            with patch("cineforge.worker_guard._worker_request", return_value=request):
                self.assertEqual(stop_own_pod(plan(), allow_live=True), not foreign)
            self.assertEqual(calls, ["GET"])

    def test_stop_phase_clamps_every_request_to_shared_remaining_budget(self):
        clock = Clock()
        calls = []
        def request(method, stop=False, timeout_seconds=3):
            calls.append((method, timeout_seconds))
            clock.sleep(min(1.75, timeout_seconds))
            return dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                CINEFORGE_LEASE="0"*32), mounts={}, status="RUNNING", actions=["stop"])
        with patch("cineforge.worker_guard._worker_request", return_value=request):
            self.assertFalse(stop_own_pod(plan(), allow_live=True, timeout_seconds=5,
                monotonic=clock.monotonic, sleep=clock.sleep))
        self.assertLessEqual(clock.mono, 15)
        self.assertEqual([c[0] for c in calls], ["GET", "POST", "GET"])
        self.assertEqual(calls[:2], [("GET", 3), ("POST", 3)])
        self.assertLessEqual(calls[-1][1], 1.5)

    def test_unknown_terminated_and_stalled_states_stay_uncertain_with_bounded_polling(self):
        for status in ("TERMINATED", "UNKNOWN", "STOPPING"):
            with self.subTest(status=status):
                clock = Clock()
                calls = []
                def request(method, stop=False, timeout_seconds=3):
                    calls.append((method, timeout_seconds))
                    return dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                        CINEFORGE_LEASE="0"*32), mounts={}, status=status, actions=[])
                with patch("cineforge.worker_guard._worker_request", return_value=request):
                    self.assertFalse(stop_own_pod(plan(), allow_live=True, timeout_seconds=1,
                        monotonic=clock.monotonic, sleep=clock.sleep))
                self.assertLessEqual(clock.mono, 11)
                self.assertTrue(all(c[0] == "GET" and 0 < c[1] <= 1 for c in calls))
                self.assertLessEqual(len(calls), 6)
        # A frozen injected clock/sleep must still have a finite call limit.
        calls.clear()
        with patch("cineforge.worker_guard._worker_request", return_value=request):
            self.assertFalse(stop_own_pod(plan(), allow_live=True, monotonic=lambda: 0,
                                        sleep=lambda seconds: None))
        self.assertLessEqual(len(calls), 42)

    def test_stalled_or_late_owned_get_cannot_schedule_stop_after_deadline(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def stalled(method, stop=False, timeout_seconds=3):
            calls.append((method, timeout_seconds))
            entered.set()
            release.wait(2)
            return dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                CINEFORGE_LEASE="0"*32), mounts={}, status="RUNNING", actions=["stop"])
        class StalledOpener:
            def open(self, request, timeout):
                return Reply(stalled(request.method, False, timeout))
        try:
            with patch("cineforge.worker_guard.worker_credential", return_value="fixture-scoped-key"), \
                    patch("urllib.request.build_opener", return_value=StalledOpener()):
                started = time.monotonic()
                self.assertFalse(stop_own_pod(plan(), allow_live=True, timeout_seconds=.03))
                self.assertLess(time.monotonic() - started, .5)
                self.assertTrue(entered.is_set())
                self.assertEqual([c[0] for c in calls], ["GET"])
        finally:
            release.set()
        clock = Clock()
        def late(method, stop=False, timeout_seconds=3):
            calls.append((method, timeout_seconds))
            clock.sleep(6)
            return stalled(method, stop, timeout_seconds)
        calls.clear()
        with patch("cineforge.worker_guard._worker_request", return_value=late):
            self.assertFalse(stop_own_pod(plan(), allow_live=True, timeout_seconds=5,
                                        monotonic=clock.monotonic, sleep=clock.sleep))
        self.assertTrue(all(c[0] == "GET" for c in calls))

    def test_poll_revalidates_ownership_and_no_native_request_exceeds_remaining_time(self):
        clock = Clock()
        calls = []
        class Opener:
            def open(self, request, timeout):
                calls.append((request.method, timeout))
                self_status = "RUNNING" if len(calls) == 1 else "STOPPING"
                data = dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                    CINEFORGE_LEASE="0"*32), mounts={}, status=self_status, actions=["stop"])
                clock.sleep(1.5)
                if len(calls) == 3:
                    data["env"]["CINEFORGE_LEASE"] = "foreign"
                    data["status"] = "EXITED"
                return Reply(data)
        with patch("cineforge.worker_guard.worker_credential", return_value="fixture-scoped-key"), \
                patch("urllib.request.build_opener", return_value=Opener()):
            self.assertFalse(stop_own_pod(plan(), allow_live=True, timeout_seconds=5,
                monotonic=clock.monotonic, sleep=clock.sleep))
        self.assertEqual([c[0] for c in calls], ["GET", "POST", "GET"])
        self.assertEqual([c[1] for c in calls], [3, 3, 2])

    def test_run_guard_shares_attempt_and_overall_budget_with_native_stop(self):
        clock = Clock()
        budgets = []
        def bounded(remaining):
            budgets.append(remaining)
            clock.sleep(remaining)
            return False
        result = run_guard(plan(stop_at_unix=100), lambda: 100, lambda: False,
            clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep,
            max_stop_attempts=3, attempt_timeout_seconds=2, bounded_stop=bounded)
        self.assertEqual(budgets, [2, 2, 2])
        self.assertEqual(clock.mono, 18)
        self.assertEqual(result["status"], "compute_stop_unconfirmed")
        self.assertEqual(result["attempts"], 3)
        self.assertFalse(result["resource_deleted"])

    def test_native_stop_retries_share_overall_budget_without_repeating_stop(self):
        clock = Clock()
        calls = []
        def request(method, stop=False, timeout_seconds=3):
            calls.append((method, timeout_seconds))
            return dict(id="fixture-pod", env=dict(CINEFORGE_SESSION="fixture-session",
                CINEFORGE_LEASE="0"*32), mounts={}, status="STOPPING", actions=["stop"])
        with patch("cineforge.worker_guard._worker_request", return_value=request):
            result = run_guard(plan(stop_at_unix=100), lambda: 100, lambda: False,
                clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep,
                max_stop_attempts=2, attempt_timeout_seconds=1,
                bounded_stop=lambda remaining: stop_own_pod(plan(), allow_live=True,
                    timeout_seconds=remaining, monotonic=clock.monotonic, sleep=clock.sleep))
        self.assertEqual(clock.mono, 13)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["status"], "compute_stop_unconfirmed")
        self.assertFalse(result["resource_deleted"])
        self.assertTrue(all(method == "GET" and 0 < timeout <= 1 for method, timeout in calls))
        self.assertEqual(len(calls), 10)

    def test_native_stop_rejects_invalid_budget_and_late_success(self):
        for bound in (True, 0, -1, float("nan"), float("inf"), 11):
            with self.assertRaises(WorkerGuardError):
                stop_own_pod(plan(), allow_live=True, timeout_seconds=bound)
        clock = Clock()
        def late(remaining):
            clock.sleep(10)
            return True
        result = run_guard(plan(stop_at_unix=100), lambda: 100, lambda: False,
            clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep,
            max_stop_attempts=2, attempt_timeout_seconds=1, bounded_stop=late)
        self.assertEqual(result["status"], "compute_stop_unconfirmed")
        self.assertEqual(result["attempts"], 1)
        self.assertFalse(result["resource_deleted"])

    def test_native_https_slow_connect_cannot_send_late_stop(self):
        clock = Clock()
        class Socket:
            def __init__(self):
                self.sent, self.closed = [], False
            def settimeout(self, timeout):
                pass
            def sendall(self, data):
                self.sent.append((clock.mono, data))
            def close(self):
                self.closed = True
        socket = Socket()
        def connect(connection):
            clock.sleep(4)  # DNS/connect/TLS may ignore the caller's socket timeout.
            connection.sock = socket
        with patch("cineforge.worker_guard.worker_credential", return_value="fixture-scoped-key"), \
                patch("cineforge.worker_guard.time.monotonic", side_effect=clock.monotonic), \
                patch("http.client.HTTPSConnection.connect", autospec=True, side_effect=connect):
            request = worker_guard._worker_request(plan())
            with self.assertRaises(WorkerGuardError):
                request("POST", True, 3)
        self.assertEqual(socket.sent, [])
        self.assertTrue(socket.closed)

    def test_native_https_rechecks_deadline_between_stop_headers_and_body(self):
        clock = Clock()
        class Socket:
            def __init__(self):
                self.sent, self.closed = [], False
            def settimeout(self, timeout):
                pass
            def sendall(self, data):
                self.sent.append((clock.mono, data))
                clock.sleep(3)
            def close(self):
                self.closed = True
        socket = Socket()
        def connect(connection):
            clock.sleep(1)
            connection.sock = socket
        with patch("cineforge.worker_guard.worker_credential", return_value="fixture-scoped-key"), \
                patch("cineforge.worker_guard.time.monotonic", side_effect=clock.monotonic), \
                patch("http.client.HTTPSConnection.connect", autospec=True, side_effect=connect):
            request = worker_guard._worker_request(plan())
            with self.assertRaises(WorkerGuardError):
                request("POST", True, 3)
        self.assertEqual(len(socket.sent), 1)
        self.assertLess(socket.sent[0][0], 13)
        self.assertIn(b"POST /v2/pods/fixture-pod/action HTTP/1.1", socket.sent[0][1])
        self.assertNotIn(b'{"action":"stop"}', socket.sent[0][1])
        self.assertTrue(socket.closed)

    def test_standalone_fake_guard_process_proves_no_orchestrator_dependency(self):
        code = '''import json
from cineforge.worker_guard import WorkerGuardPlan,run_guard
plan=WorkerGuardPlan("fixture-pod","fixture-session","0"*32,1,3)
print(json.dumps(run_guard(plan,lambda:0,lambda:True,clock=lambda:2,monotonic=lambda:3)))
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                timeout=5, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "compute_stop_confirmed")
