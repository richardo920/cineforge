"""RF-20 regressions: preserve interruption in M5 lower layers."""

import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
