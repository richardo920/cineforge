"""RF-20 regressions: preserve interruption in M5 lower layers."""

import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from test_native_remote import RemoteFixture
from test_s2v import context

from cineforge.ssh_worker import SSHWorker


class M5InterruptionTests(unittest.TestCase):
    def test_closed_preparation_poison_then_interrupt(self):
        with tempfile.TemporaryDirectory(prefix="rf20-closed-") as root:
            gate, _ = context(Path(root).resolve())
            worker = SimpleNamespace(selection=gate.service)
            runtime = Mock()
            runtime.prepare.side_effect = lambda *a, **k: os.kill(os.getpid(), signal.SIGINT)
            with self.assertRaises(KeyboardInterrupt):
                SSHWorker._prepare_closed_comfy(worker, None, runtime, time.monotonic() + 5)
            runtime._poison.assert_called_once()
            import test_cli_error_contract as cli_fixtures
            runtime._poison.reset_mock()
            code, _, err = cli_fixtures.ErrorContractTests().invoke(
                ["conditioning-preflight", "--context", "x", "--reference", "x", "--output-dir", "x"],
                patch("cineforge.conditioning_cli.command", side_effect=lambda *a:
                    SSHWorker._prepare_closed_comfy(worker, None, runtime, time.monotonic() + 5)))
            self.assertEqual(code, 130)
            self.assertIn("CF_INTERRUPTED", err)
            runtime._poison.assert_called_once()

    def test_bound_remote_rpc_poison_then_interrupt(self):
        fixture = RemoteFixture()
        fixture.setUp()
        try:
            fixture.prepare()
            def interrupt(*args, **kwargs):
                os.kill(os.getpid(), signal.SIGINT)
            with patch.object(fixture.worker, "native_request", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    fixture.bridge._call("native_comfy_verify")
            self.assertTrue(fixture.bridge._unresolved)
            from cineforge.native_speech_integrity import NativeSpeechError
            with patch.object(fixture.worker, "native_request") as request:
                with self.assertRaises(NativeSpeechError):
                    fixture.bridge._call("native_comfy_verify")
                request.assert_not_called()
        finally:
            fixture.doCleanups()


class CleanupCancellationTests(unittest.TestCase):
    def test_owned_pod_cleanup_finishes_before_cli_130(self):
        import test_cli_error_contract as cli_fixtures
        import test_runpod as pod_fixtures

        for phase in ("cleanup_pending", "ownership", "delete", "reconcile", "released"):
            with self.subTest(phase=phase):
                fixture = pod_fixtures.RunpodLifecycleTests()
                fixture.setUp()
                provider = fixture.provider()
                fixture.acquire(provider)
                fixture.cloud.bounds.clear()
                original_sink, original_request = provider._sink, fixture.cloud.request
                calls = []
                interrupted = False

                def sink(intent, snapshot):
                    nonlocal interrupted
                    if snapshot.state == phase and not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt("private-cleanup-canary")
                    return original_sink(intent, snapshot)

                def request(method, path, body=None, **kwargs):
                    nonlocal interrupted
                    calls.append(method)
                    result = original_request(method, path, body, **kwargs)
                    target = (phase == "delete" and method == "DELETE"
                              or phase == "ownership" and calls == ["GET"]
                              or phase == "reconcile" and calls[-2:] == ["DELETE", "GET"])
                    if target and not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt("private-cleanup-canary")
                    return result

                provider._sink = sink
                with patch.object(fixture.cloud, "request", side_effect=request):
                    code, _, err = cli_fixtures.ErrorContractTests().invoke(
                        ["mock-run", "x"], patch("cineforge.cli.load_mock_plan",
                            side_effect=lambda *a: provider.release("pod_1")))
                self.assertEqual(code, 130)
                self.assertIn("CF_INTERRUPTED", err)
                self.assertNotIn("private-cleanup-canary", err)
                self.assertTrue(interrupted)
                self.assertEqual(fixture.cloud.pods, {})
                self.assertEqual(calls.count("DELETE"), 1)
                self.assertEqual(calls[-1], "GET")
                self.assertLessEqual(fixture.clock.now, provider._cleanup)
                self.assertTrue(all(bound[0] <= provider._cleanup for bound in fixture.cloud.bounds))

    def test_native_combined_interrupt_and_cleanup_fault_retains_owned_pid(self):
        import test_native_speech as speech_fixtures

        from cineforge.native_speech import OriginalVoiceDesign
        from cineforge.native_speech_integrity import NativeSpeechError
        from cineforge.native_speech_process import OwnedChildInterrupted, run_child

        for through_runtime in (False, True):
            for phase in ("wait", "body", "selector", "stdin", "stdout"):
                with self.subTest(runtime=through_runtime, phase=phase):
                    fixture = speech_fixtures.NativeSpeechTests()
                    fixture.setUp()
                    try:
                        process = Mock(pid=12345, returncode=0)
                        process.kill.side_effect = PermissionError("private-kill-canary")
                        selector = Mock()
                        selector.get_map.return_value = {}
                        if phase == "body":
                            selector.get_map.side_effect = KeyboardInterrupt("private-body-canary")
                            process.wait.side_effect = OSError("private-reap-canary")
                        elif phase == "wait":
                            process.wait.side_effect = KeyboardInterrupt("private-wait-canary")
                        elif phase == "selector":
                            selector.close.side_effect = KeyboardInterrupt("private-selector-canary")
                        else:
                            getattr(process, phase).close.side_effect = KeyboardInterrupt("private-fd-canary")

                        def child(*args, **kwargs):
                            return run_child(("synthetic",), payload=b"", env={}, cwd="/",
                                             deadline=time.monotonic() + 4)

                        with patch("cineforge.native_speech.run_child", side_effect=child), \
                             patch("cineforge.native_speech_process.subprocess.Popen", return_value=process), \
                             patch("cineforge.native_speech_process.selectors.DefaultSelector", return_value=selector), \
                             patch("cineforge.native_speech_process.os.set_blocking"), \
                             patch("cineforge.native_speech_process.os.killpg", side_effect=PermissionError("private-group-canary")):
                            with self.assertRaises(KeyboardInterrupt) as caught:
                                if through_runtime:
                                    fixture.runtime.synthesize(fixture.request, fixture.reference)
                                else:
                                    child()
                        self.assertEqual(str(caught.exception), "")
                        self.assertIsNone(caught.exception.__context__)
                        if through_runtime:
                            attempt = fixture.runtime.last_attempt
                            self.assertEqual(attempt.outcome, "interrupted")
                            self.assertTrue(attempt.cleanup_uncertain)
                            self.assertEqual(attempt.owned_process_id, 12345)
                            self.assertIs(fixture.runtime._unresolved, attempt)
                            with patch("cineforge.native_speech.run_child") as spawn:
                                with self.assertRaises(NativeSpeechError):
                                    fixture.runtime.synthesize(fixture.request, fixture.reference)
                                with self.assertRaises(NativeSpeechError):
                                    fixture.runtime.preview(OriginalVoiceDesign("text", "English", "fictional", speech_fixtures.HASH))
                                spawn.assert_not_called()
                        else:
                            self.assertIsInstance(caught.exception, OwnedChildInterrupted)
                            self.assertTrue(caught.exception.cleanup_uncertain)
                            self.assertEqual(caught.exception.owned_process_id, 12345)
                        process.kill.assert_called_once()
                        process.wait.assert_called_once()
                        selector.close.assert_called_once()
                        process.stdin.close.assert_called()
                        process.stdout.close.assert_called()
                    finally:
                        fixture.doCleanups()

    def test_closed_comfy_deeper_prepare_signal_poison_and_no_replay(self):
        import test_comfy_runtime as comfy_fixtures

        runtime, worker, selected, _, _, bridge = comfy_fixtures.ComfyControllerTests().fixture()
        with patch.object(bridge, "_upload", side_effect=lambda *a, **k: os.kill(os.getpid(), signal.SIGINT)):
            with self.assertRaises(KeyboardInterrupt):
                runtime.prepare(worker, selected, deadline=time.monotonic() + 5)
        self.assertTrue(runtime._unresolved)
        self.assertTrue(bridge._unresolved)
        with patch.object(bridge, "_upload") as upload:
            with self.assertRaises(ValueError):
                runtime.prepare(worker, selected, deadline=time.monotonic() + 5)
            upload.assert_not_called()

    def test_native_socket_privacy_handler_preserves_real_signal(self):
        from cineforge.native_speech_network import deny_network

        with patch("cineforge.native_speech_network.sys.platform", "linux"), \
             patch("cineforge.native_speech_network.platform.machine", return_value="x86_64"), \
             patch("cineforge.native_speech_network.verify_binary", side_effect=lambda *a: os.kill(os.getpid(), signal.SIGINT)):
            with self.assertRaises(KeyboardInterrupt) as caught:
                deny_network(None, time.monotonic() + 5)
        self.assertEqual(str(caught.exception), "")
        self.assertIsNone(caught.exception.__context__)

    def test_directory_failure_cannot_replace_owned_child_cancellation_facts(self):
        import test_native_speech as speech_fixtures

        from cineforge.native_speech_process import OwnedChildInterrupted

        fixture = speech_fixtures.NativeSpeechTests()
        fixture.setUp()
        original_directory = tempfile.TemporaryDirectory

        class Directory(original_directory):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise OSError("private-directory-canary")

        try:
            with patch("cineforge.native_speech.tempfile.TemporaryDirectory", Directory), \
                 patch("cineforge.native_speech.run_child", side_effect=OwnedChildInterrupted(12345, True)):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    fixture.runtime.synthesize(fixture.request, fixture.reference)
            self.assertEqual(str(caught.exception), "")
            self.assertIsNone(caught.exception.__context__)
            self.assertEqual(fixture.runtime.last_attempt.owned_process_id, 12345)
            self.assertTrue(fixture.runtime.last_attempt.cleanup_uncertain)
            self.assertIs(fixture.runtime._unresolved, fixture.runtime.last_attempt)
        finally:
            fixture.doCleanups()


class SSHContextCancellationTests(unittest.TestCase):
    def test_actual_context_close_failure_before_cli_boundary(self):
        import test_cli_error_contract as cli_fixtures

        from cineforge.ssh_worker import WorkerError

        worker = object.__new__(SSHWorker)
        worker._stop = Mock()
        worker._thread = None
        worker._claimed = MagicMock()
        claimed = worker._claimed
        worker._close_tunnel = Mock(side_effect=WorkerError("private-finalizer-canary"))

        def work(*args):
            with worker:
                os.kill(os.getpid(), signal.SIGINT)

        with patch.object(SSHWorker, "__enter__", return_value=worker):
            code, _, err = cli_fixtures.ErrorContractTests().invoke(
                ["mock-run", "x"], patch("cineforge.cli.load_mock_plan", side_effect=work))
        self.assertEqual(code, 130)
        self.assertIn("CF_INTERRUPTED", err)
        self.assertNotIn("private-finalizer-canary", err)
        worker._close_tunnel.assert_called_once()
        claimed.__exit__.assert_called_once()
        self.assertIsNone(worker._claimed)

    def test_real_sigint_with_ssh_finalizer_faults_through_owned_callers(self):
        import json
        from contextlib import ExitStack
        from threading import RLock

        import test_cli_error_contract as cli_fixtures
        import test_cloud_smoke as managed
        import test_conditioning as conditioning

        from cineforge.ssh_worker import WorkerError

        for module, fixture_type in ((managed, managed.ManagedSmokeTests),
                                     (conditioning, conditioning.ManagedTests)):
            for boundary in ("caller", "cli"):
                for phase in ("close", "tunnel", "kill", "reap", "control", "claimed", "both"):
                    with self.subTest(caller=module.__name__, boundary=boundary, phase=phase):
                        fixture = fixture_type()
                        fixture.setUp()
                        claimed = MagicMock()
                        if phase in ("claimed", "both"):
                            claimed.__exit__.side_effect = OSError("private-finalizer-canary")
                        process = Mock(pid=12345)
                        process.wait.return_value = 0
                        if phase == "reap":
                            process.wait.side_effect = OSError("private-finalizer-canary")

                        def enter(worker):
                            worker._stop = Mock()
                            worker._thread = Mock()
                            worker._claimed = claimed
                            worker._tunnel_lock = RLock()
                            worker._tunnel = process if phase in ("kill", "reap") else None
                            worker._control_dir = Path("/synthetic-owned-control")
                            worker._control_identity = (1, 2)
                            worker._control_socket_identity = (1, 3)
                            worker._retired_controls = []
                            worker._control_ready = Mock()
                            if phase == "control":
                                worker._control_ready.side_effect = WorkerError("private-finalizer-canary")
                            return worker

                        def interrupt(worker):
                            os.kill(os.getpid(), signal.SIGINT)

                        def run(*args, **kwargs):
                            if module is managed:
                                return fixture.run_smoke("context-interrupt")
                            return fixture.run_case("context-interrupt")

                        try:
                            with ExitStack() as stack:
                                for name, method in (("__enter__", enter), ("__exit__", SSHWorker.__exit__),
                                                     ("close", SSHWorker.close), ("_close_tunnel", SSHWorker._close_tunnel),
                                                     ("prepare", interrupt)):
                                    stack.enter_context(patch.object(module.Worker, name, method, create=True))
                                if phase == "close":
                                    stack.enter_context(patch.object(module.Worker, "close",
                                        side_effect=WorkerError("private-finalizer-canary")))
                                if phase in ("tunnel", "both"):
                                    stack.enter_context(patch.object(module.Worker, "_close_tunnel",
                                        side_effect=WorkerError("private-finalizer-canary")))
                                kill = stack.enter_context(patch("cineforge.ssh_worker.os.killpg",
                                    side_effect=PermissionError("private-finalizer-canary") if phase == "kill" else None))
                                if boundary == "caller":
                                    with self.assertRaises(KeyboardInterrupt) as caught:
                                        run()
                                    self.assertEqual(str(caught.exception), "")
                                    self.assertIsNone(caught.exception.__context__)
                                else:
                                    if module is managed:
                                        stack.enter_context(patch("cineforge.cli.recovery_inputs",
                                            return_value=(None, None, None)))
                                        argv = ["cloud-recover", "--selection", "x", "--proposal", "x", "--budget-dir", "x"]
                                        command = "cineforge.cli.recover_managed"
                                    else:
                                        argv = ["conditioning-preflight", "--context", "x", "--reference", "x", "--output-dir", "x"]
                                        command = "cineforge.conditioning_cli.command"
                                    code, _, err = cli_fixtures.ErrorContractTests().invoke(
                                        argv, patch(command, side_effect=run))
                                    self.assertEqual(code, 130)
                                    self.assertIn("CF_INTERRUPTED", err)
                                    self.assertNotIn("private-finalizer-canary", err)
                                worker = fixture.workers[0]
                                claimed.__exit__.assert_called_once()
                                self.assertIsNone(worker._claimed)
                                if phase != "close":
                                    worker._thread.join.assert_called_once_with(timeout=7)
                                if phase in ("kill", "reap"):
                                    kill.assert_called_once_with(12345, signal.SIGKILL)
                                    process.wait.assert_called_once_with(timeout=2)
                                    self.assertIs(worker._tunnel, process)
                                if phase == "control":
                                    self.assertIsNotNone(worker._control_dir)
                                    self.assertEqual(worker._retired_controls, [])
                            checkpoint = fixture.root / "context-interrupt" / "run.json"
                            report = json.loads(checkpoint.read_bytes())
                            self.assertEqual(report["outcome"], "incomplete")
                            self.assertTrue(report["release_verified"])
                            self.assertEqual(fixture.cloud.pods, {})
                            self.assertEqual(sum(c[0] == "DELETE" for c in fixture.cloud.calls), 1)
                            self.assertEqual(checkpoint.stat().st_mode & 0o777, 0o600)
                        finally:
                            fixture.doCleanups()
