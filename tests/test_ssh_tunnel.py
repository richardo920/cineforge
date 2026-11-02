"""Offline tunnel lifecycle and native local-bind fixtures; never launches SSH."""

import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from test_cloud_budget import selection
from test_runpod import spec
from test_ssh_worker import Budget, Transport

from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.workflows import canonical


class Tunnel:
    def __init__(self, pid, command, harness):
        self.pid, self.command, self.harness = pid, command, harness
        self.returncode = None
        self.waits = []
        self.wait_failure = None
        self.control = None
        self.listener = None
        self.path = Path(command[command.index("-S") + 1]) if "-S" in command else None
        if self.path is not None and harness.master_ready:
            self.control = socket.socket(socket.AF_UNIX)
            self.control.bind(str(self.path))
            self.control.listen(1)
        if harness.child_exited:
            self.returncode = 1
        # The old -L startup can still be connecting while an unrelated local
        # listener wins its chosen port. No fake API metadata is used as proof.
        if harness.foreign and "-L" in command:
            harness.foreign_listener(int(command[command.index("-L") + 1].split(":")[1]))

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.wait_failure is not None:
            raise self.wait_failure
        return self.returncode

    def killed(self):
        self.returncode = -signal.SIGKILL
        for listener in (self.control, self.listener):
            if listener is not None:
                listener.close()


class Harness:
    def __init__(self, worker, *, ready=True, foreign=False, exited=False):
        self.worker = worker
        self.master_ready, self.foreign, self.child_exited = ready, foreign, exited
        self.processes, self.kills, self.events, self.bounds = [], [], [], []
        self.foreign_socket = None
        self.now = 0.
        self.health = {"system": {}, "devices": []}
        self.health_error = None
        self.health_delay = 0.
        self.control_delay = 0.
        self.clock_fault = None

    def foreign_listener(self, port):
        self.foreign_socket = socket.socket()
        self.foreign_socket.bind(("127.0.0.1", port))
        self.foreign_socket.listen(1)

    def spawn(self, command, **options):
        self.events.append("spawn")
        self.worker_test.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.worker_test.assertEqual(options["stderr"], subprocess.DEVNULL)
        self.worker_test.assertTrue(options["start_new_session"])
        self.worker_test.assertNotIn("RUNPOD_API_KEY", options["env"])
        self.worker_test.assertNotIn("SSH_AUTH_SOCK", options["env"])
        process = Tunnel(987650000 + len(self.processes), command, self)
        self.processes.append(process)
        return process

    def kill(self, pid, sig):
        self.kills.append(pid)
        self.worker_test.assertEqual(sig, signal.SIGKILL)
        next(p for p in self.processes if p.pid == pid).killed()

    def monotonic(self):
        if self.clock_fault is not None and self.processes:
            fault, self.clock_fault = self.clock_fault, None
            raise fault
        return self.now

    def sleep(self, seconds):
        self.worker_test.assertGreater(seconds, 0)
        self.now += seconds

    def mux(self, command, **options):
        self.worker_test.assertIn("-O", command)
        self.worker_test.assertNotIn("-c", command)
        self.worker_test.assertNotIn("payload", options)
        self.bounds.append(options["timeout"])
        options["check"]()
        operation = command[command.index("-O") + 1]
        path = Path(command[command.index("-S") + 1])
        process = next(p for p in self.processes if p.path == path and p.poll() is None)
        self.events.append(operation)
        self.now += self.control_delay
        if operation == "forward":
            forward = command[command.index("-L") + 1]
            host, port, remote, target = forward.split(":")
            self.worker_test.assertEqual((host, remote, target), ("127.0.0.1", "127.0.0.1", "8188"))
            if self.foreign:
                self.foreign_listener(int(port))
            process.listener = socket.socket()
            try:
                process.listener.bind((host, int(port)))
                process.listener.listen(1)
            except OSError:
                process.listener.close()
                raise WorkerError("private-forward-canary") from None
        elif operation != "check":
            raise AssertionError("unapproved mux operation")
        return b""

    def http(self, endpoint, *, allow_live):
        self.worker_test.assertTrue(endpoint.startswith("http://127.0.0.1:"))
        self.worker_test.assertTrue(allow_live)
        harness = self
        class HTTP:
            def request(self, method, path, body=None, **options):
                harness.events.append("health")
                harness.worker_test.assertIn("forward", harness.events)
                harness.worker_test.assertEqual((method, path, body), ("GET", "/system_stats", None))
                harness.bounds.append(options["timeout_seconds"])
                harness.now += harness.health_delay
                if harness.health_error is not None:
                    raise harness.health_error
                return canonical(harness.health)
        return HTTP()

    def __enter__(self):
        self.stack = ExitStack()
        for name, kwargs in [
                ("subprocess.Popen", {"side_effect": self.spawn}),
                ("run_process", {"side_effect": self.mux}),
                ("os.killpg", {"side_effect": self.kill}),
                ("time.monotonic", {"side_effect": self.monotonic}),
                ("time.sleep", {"side_effect": self.sleep}),
                ("ComfyHTTPTransport", {"side_effect": self.http})]:
            self.stack.enter_context(patch("cineforge.ssh_worker." + name, **kwargs))
        return self

    def __exit__(self, *args):
        try:
            for process in self.processes:
                process.wait_failure = None
            self.worker.close()
        finally:
            if self.foreign_socket is not None:
                self.foreign_socket.close()
            self.stack.close()


class TunnelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-tunnel-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.key = self.root / "synthetic-key"
        self.key.write_text("synthetic fixture, not a private key")
        self.key.chmod(0o600)
        self.budget = Budget()
        self.worker = SSHWorker(spec(), selection(), self.budget, Transport(), self.key,
                                self.root / "state", allow_live=True)
        self.worker.plan, self.worker._key = self.budget.plan, "first"
        self.worker._host, self.worker._port = "8.8.8.8", 22000

    def harness(self, **options):
        fixture = Harness(self.worker, **options)
        fixture.worker_test = self
        return fixture

    def test_owned_forward_ack_precedes_health_and_endpoint_is_cached(self):
        with self.harness() as fixture:
            endpoint = self.worker.endpoint()
            self.assertEqual(self.worker.endpoint(), endpoint)
            self.assertEqual(len(fixture.processes), 1)
            self.assertLess(fixture.events.index("check"), fixture.events.index("forward"))
            self.assertLess(fixture.events.index("forward"), fixture.events.index("health"))
            command = fixture.processes[0].command
            self.assertIn("-M", command)
            self.assertIn("-N", command)
            self.assertNotIn("-L", command)
            self.assertIn("ControlPersist=no", command)
            self.assertIn("StrictHostKeyChecking=accept-new", command)
            control = fixture.processes[0].path
            self.assertEqual(control.parent.stat().st_mode & 0o777, 0o700)
            self.assertIn(spec().session_id, control.parent.name)
            self.assertTrue(all(0 < t <= 10 for t in fixture.bounds))
            self.worker.close()
            self.assertEqual(fixture.kills, [fixture.processes[0].pid])
            self.assertEqual(fixture.processes[0].waits, [2])
            self.assertFalse(control.parent.exists())
            self.worker.close()
            self.assertEqual(len(fixture.kills), 1)

    def test_foreign_listener_winning_bind_is_never_a_usable_endpoint(self):
        with self.harness(foreign=True) as fixture:
            with self.assertRaises(WorkerError) as cm:
                self.worker.execution_transport()
            self.assertNotIn("private-forward-canary", "".join(traceback.format_exception(cm.exception)))
            self.assertNotIn("health", fixture.events)
            self.assertEqual(self.worker._endpoint, "")
            self.assertIsNone(self.worker._tunnel)
            self.assertEqual(fixture.kills, [fixture.processes[0].pid])
            self.assertEqual(fixture.processes[0].waits, [2])

    def test_each_timed_out_attempt_is_killed_and_reaped_before_retry(self):
        self.budget.remaining = .35
        with self.harness(ready=False) as fixture:
            for _ in range(2):
                with self.assertRaises(WorkerError):
                    self.worker.endpoint()
                self.assertIsNone(self.worker._tunnel)
                self.assertEqual(self.worker._endpoint, "")
            self.assertEqual(fixture.kills, [p.pid for p in fixture.processes])
            self.assertTrue(all(p.waits == [2] for p in fixture.processes))
            self.assertTrue(all(not p.path.parent.exists() for p in fixture.processes))

    def test_exit_interrupt_and_check_failures_reap_owned_attempt(self):
        with self.harness(exited=True) as fixture:
            with self.assertRaises(WorkerError):
                self.worker.endpoint()
            self.assertEqual(fixture.kills, [fixture.processes[0].pid])
            self.assertEqual(fixture.processes[0].waits, [2])
        for fault in (KeyboardInterrupt(), RuntimeError("private-heartbeat-canary")):
            self.worker._stop.clear()
            def check():
                if self.worker._tunnel is not None:
                    raise fault
            with self.harness() as fixture, patch.object(self.worker, "check", side_effect=check):
                with self.assertRaises(WorkerError) as cm:
                    self.worker.endpoint()
                self.assertNotIn("private-heartbeat", "".join(traceback.format_exception(cm.exception)))
                self.assertIsNone(self.worker._tunnel)
                self.assertEqual(fixture.kills, [fixture.processes[0].pid])
                self.assertEqual(fixture.processes[0].waits, [2])

    def test_heartbeat_and_deadline_checks_close_an_already_ready_forward(self):
        for kind in ("heartbeat", "deadline"):
            self.worker._stop.clear()
            self.worker._failed.clear()
            self.budget.remaining = 100
            with self.harness() as fixture:
                self.worker.endpoint()
                if kind == "heartbeat":
                    self.worker._failed.set()
                else:
                    self.budget.remaining = 0
                with self.assertRaises(WorkerError):
                    self.worker.check()
                self.assertIsNone(self.worker._tunnel)
                self.assertEqual(self.worker._endpoint, "")
                self.assertEqual(fixture.kills, [fixture.processes[0].pid])
                self.assertEqual(fixture.processes[0].waits, [2])

    def test_health_timeout_and_late_success_cannot_publish_endpoint(self):
        for late in (False, True):
            self.worker._stop.clear()
            self.budget.remaining = .35
            with self.harness() as fixture:
                fixture.health_error = None if late else OSError("private-health-canary")
                fixture.health_delay = .5 if late else .1
                with self.assertRaises(WorkerError) as cm:
                    self.worker.endpoint()
                self.assertNotIn("private-health", "".join(traceback.format_exception(cm.exception)))
                self.assertEqual(self.worker._endpoint, "")
                self.assertIsNone(self.worker._tunnel)
                self.assertEqual(fixture.kills, [fixture.processes[0].pid])
                self.assertTrue(all(0 < t <= .35 for t in fixture.bounds))

    def test_unconfirmed_reap_retains_reference_and_blocks_another_spawn(self):
        with self.harness() as fixture:
            self.worker.endpoint()
            process = fixture.processes[0]
            process.wait_failure = subprocess.TimeoutExpired("private-wait-canary", 2)
            with self.assertRaises(WorkerError):
                self.worker.close()
            self.assertIs(self.worker._tunnel, process)
            self.worker._stop.clear()
            with self.assertRaises(WorkerError):
                self.worker.endpoint()
            self.assertEqual(len(fixture.processes), 1)
            self.assertIs(self.worker._tunnel, process)

    def test_claimed_state_is_released_even_when_tunnel_cleanup_fails(self):
        class Claim:
            exited = False
            def __exit__(self, *args):
                self.exited = True
        claim = Claim()
        with self.harness() as fixture:
            self.worker.endpoint()
            self.worker._claimed = claim
            fixture.processes[0].wait_failure = subprocess.TimeoutExpired("private-wait-canary", 2)
            with self.assertRaises(WorkerError):
                self.worker.__exit__(None, None, None)
            self.assertTrue(claim.exited)
            self.worker._claimed = None

    def test_control_timeout_and_post_spawn_clock_fault_cannot_leave_a_child(self):
        for kind in ("control", "clock"):
            self.worker._stop.clear()
            self.budget.remaining = .35
            with self.harness() as fixture:
                if kind == "control":
                    fixture.control_delay = .5
                else:
                    fixture.clock_fault = KeyboardInterrupt()
                with self.assertRaises(WorkerError):
                    self.worker.endpoint()
                self.assertNotIn("forward", fixture.events)
                self.assertNotIn("health", fixture.events)
                self.assertIsNone(self.worker._tunnel)
                self.assertEqual(fixture.kills, [fixture.processes[0].pid])
                self.assertEqual(fixture.processes[0].waits, [2])

    def test_remote_heartbeat_failure_reaps_an_owned_live_forward(self):
        with self.harness() as fixture:
            self.worker.endpoint()
            with patch.object(self.worker._stop, "wait", return_value=False), \
                    patch.object(self.worker, "_request", side_effect=RuntimeError("private-beat-canary")):
                self.worker._supervise()
            self.assertTrue(self.worker._failed.is_set())
            self.assertIsNone(self.worker._tunnel)
            self.assertEqual(fixture.kills, [fixture.processes[0].pid])
            self.assertEqual(fixture.processes[0].waits, [2])

    def test_replaced_control_socket_is_not_unlinked_or_reused(self):
        for replacement in ("file", "socket"):
            self.worker._stop.clear()
            with self.harness() as fixture:
                self.worker.endpoint()
                control = fixture.processes[0].path
                control.unlink()
                foreign = None
                if replacement == "file":
                    control.write_bytes(b"foreign-control-canary")
                else:
                    foreign = socket.socket(socket.AF_UNIX)
                    foreign.bind(str(control))
                try:
                    with self.assertRaises(WorkerError):
                        self.worker.close()
                    self.assertTrue(control.exists())
                    if replacement == "file":
                        self.assertEqual(control.read_bytes(), b"foreign-control-canary")
                    self.assertIsNone(self.worker._tunnel)
                    self.assertEqual(fixture.processes[0].waits, [2])
                finally:
                    if foreign is not None:
                        foreign.close()
                    control.unlink()
                    self.worker.close()
                self.assertFalse(control.parent.exists())

    def test_replaced_control_directory_is_never_recursively_removed(self):
        with self.harness() as fixture:
            self.worker.endpoint()
            control = fixture.processes[0].path
            original = control.parent
            moved = original.with_name(original.name + "-moved")
            original.rename(moved)
            original.mkdir(mode=0o700)
            foreign = original / "foreign"
            foreign.write_bytes(b"foreign-directory-canary")
            try:
                with self.assertRaises(WorkerError):
                    self.worker.close()
                self.assertEqual(foreign.read_bytes(), b"foreign-directory-canary")
                self.assertTrue(moved.exists())
                self.assertIsNone(self.worker._tunnel)
            finally:
                foreign.unlink()
                original.rmdir()
                moved.rename(original)
                self.worker.close()
            self.assertFalse(original.exists())

    def test_wrong_service_json_is_static_and_never_cached(self):
        with self.harness() as fixture:
            fixture.health = {"foreign": "private-service-canary"}
            with self.assertRaises(WorkerError) as cm:
                self.worker.endpoint()
            self.assertNotIn("private-service-canary", "".join(traceback.format_exception(cm.exception)))
            self.assertEqual(self.worker._endpoint, "")
            self.assertIsNone(self.worker._tunnel)

    def test_failed_spawn_cleans_its_private_namespace_and_is_static(self):
        with self.harness() as fixture:
            controls = []
            def fail(command, **options):
                controls.append(Path(command[command.index("-S") + 1]))
                raise RuntimeError("private-spawn-canary")
            with patch("cineforge.ssh_worker.subprocess.Popen", side_effect=fail):
                with self.assertRaises(WorkerError) as cm:
                    self.worker.endpoint()
            self.assertNotIn("private-spawn", "".join(traceback.format_exception(cm.exception)))
            self.assertFalse(controls[0].parent.exists())
            self.assertIsNone(self.worker._tunnel)
            self.assertEqual(fixture.processes, [])

    def test_cached_endpoint_cannot_survive_exited_owned_master(self):
        with self.harness() as fixture:
            self.worker.endpoint()
            fixture.processes[0].returncode = 0
            with self.assertRaises(WorkerError):
                self.worker.endpoint()
            self.assertEqual(self.worker._endpoint, "")
            self.assertIsNone(self.worker._tunnel)
            self.assertEqual(fixture.kills, [fixture.processes[0].pid])
            self.assertEqual(fixture.processes[0].waits, [2])

    def test_native_local_process_group_is_reaped_without_ssh(self):
        process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        self.worker._tunnel = process
        started = time.monotonic()
        self.worker.close()
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNotNone(process.returncode)
        self.assertIsNone(self.worker._tunnel)


if __name__ == "__main__":
    unittest.main()
