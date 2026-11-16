"""RF-20: offline invocation compatibility, safe diagnostics and owned cancellation."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import test_cloud_smoke as smoke_fixtures
import test_runpod_transport as transport_fixtures

from cineforge import audio_process, cli, native_speech_worker, worker_guard
from cineforge.interruptions import preserve_interruption, remember_interruption
from cineforge.native_speech_process import pack, run_child, unpack
from cineforge.runpod_transport import RunpodHTTPTransport

CANARY = "fixture-private-dialogue-and-credential"
ROOT = Path(__file__).resolve().parents[1]


class ErrorContractTests(unittest.TestCase):
    def invoke(self, args, replacement=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["cineforge", *args]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with contextlib.ExitStack() as stack:
                if replacement is not None:
                    stack.enter_context(replacement)
                try:
                    result = cli.main()
                    self.assertIsNone(result)  # Existing Python caller contract.
                    code = 0
                except SystemExit as exc:
                    code = exc.code
                    displayed = "".join(traceback.format_exception(exc))
                    self.assertNotIn(CANARY, displayed)
        self.assertNotIn(CANARY, out.getvalue() + err.getvalue())
        return code, out.getvalue(), err.getvalue()

    def test_success_bare_help_version_and_existing_python_callers(self):
        for args in ([], ["--help"], ["--version"],
                     ["validate-plan", str(ROOT / "examples/synthetic-plan.json")],
                     ["mock-run", str(ROOT / "examples/synthetic-plan.json")]):
            with self.subTest(args=args):
                code, out, err = self.invoke(args)
                self.assertEqual(code, 0)
                self.assertTrue(out)
                self.assertEqual(err, "")
        _, help_text, _ = self.invoke(["--help"])
        for item in ("Exit codes:", "130", "CF_APPLICATION", "historical"):
            self.assertIn(item.lower(), help_text.lower())

    def test_usage_and_application_keep_exit_two_with_fixed_codes(self):
        for args in ([CANARY], ["comfy-run", CANARY],
                     ["comfy-run", "x", "--selection", "x", "--worker-verification", "x",
                      "--output-dir", "x", "--max-polls", CANARY]):
            code, _, err = self.invoke(args)
            self.assertEqual(code, 2)
            self.assertIn("stage=usage code=CF_USAGE", err)
        for error in (ValueError(CANARY), OSError(CANARY),
                      type(CANARY, (Exception,), {})(CANARY)):
            code, _, err = self.invoke(["mock-run", "x"],
                patch.object(cli, "load_mock_plan", side_effect=error))
            self.assertEqual(code, 2)
            self.assertIn("stage=mock-run code=CF_APPLICATION", err)
            self.assertNotIn("Traceback", err)

    def test_known_failure_reasons_are_stable_without_echoing_messages(self):
        from cineforge.cloud_budget import CloudBudgetError
        from cineforge.cloud_smoke import CloudSmokeError
        from cineforge.comfy_run import ComfyRunError
        from cineforge.conditioning import ConditioningError
        from cineforge.private_store import ArtifactStoreError
        from cineforge.storage import StorageError
        for known, expected in ((CloudBudgetError, "cloud-budget"), (CloudSmokeError, "cloud-smoke"),
                                (ComfyRunError, "comfy-run"), (ConditioningError, "conditioning"),
                                (ArtifactStoreError, "private-store"), (StorageError, "storage"),
                                (TimeoutError, "timeout"), (OSError, "io"), (ValueError, "invalid-data"),
                                (RuntimeError, "unexpected")):
            code, _, err = self.invoke(["mock-run", "x"],
                patch.object(cli, "load_mock_plan", side_effect=known(CANARY)))
            self.assertEqual(code, 2)
            self.assertIn("reason=" + expected, err)

    def test_private_input_is_absent_in_actual_module_process(self):
        with tempfile.TemporaryDirectory(prefix="rf197-invalid-") as root:
            path = Path(root) / (CANARY + ".json")
            path.write_text(json.dumps({CANARY: CANARY}))
            result = subprocess.run([sys.executable, "-m", "cineforge", "mock-run", str(path)],
                                    capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(CANARY, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_interrupt_is_130_for_each_public_command_boundary(self):
        cases = [(["mock-run", "x"], "cineforge.cli.load_mock_plan"),
                 (["validate-plan", "x"], "cineforge.cli.load_mock_plan"),
                 (["comfy-run", "x", "--selection", "x", "--worker-verification", "x",
                   "--output-dir", "x"], "cineforge.cli.read_private_input"),
                 (["conditioning-preflight", "--context", "x", "--reference", "x",
                   "--output-dir", "x"], "cineforge.conditioning_cli.command"),
                 (["cloud-recover", "--selection", "x", "--proposal", "x",
                   "--budget-dir", "x"], "cineforge.cli.recovery_inputs")]
        for args, target in cases:
            with self.subTest(args=args):
                code, _, err = self.invoke(args,
                    patch(target, side_effect=KeyboardInterrupt(CANARY)))
                self.assertEqual(code, 130)
                self.assertIn("CF_INTERRUPTED", err)

    def test_real_sigint_module_exit_is_130_without_traceback(self):
        # Synchronous self-signal exercises Python's signal delivery without races.
        code = ("import os,signal; from unittest.mock import patch; "
                "from cineforge.cli import main; "
                "f=lambda *a: os.kill(os.getpid(),signal.SIGINT); "
                "p=patch('cineforge.cli.load_mock_plan',side_effect=f); p.start(); main()")
        result = subprocess.run([sys.executable, "-c", code, "mock-run", "x"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 130)
        self.assertIn("CF_INTERRUPTED", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_nested_privacy_handlers_finish_cleanup_and_do_not_lose_cancellation(self):
        calls = []
        @preserve_interruption
        def child():
            try:
                raise KeyboardInterrupt(CANARY)
            except (Exception, KeyboardInterrupt) as error:
                remember_interruption(error)
            finally:
                calls.append("child-cleanup")
            raise ValueError(CANARY)
        @preserve_interruption
        def parent():
            try:
                child()
            except (Exception, KeyboardInterrupt) as error:
                remember_interruption(error)
            finally:
                calls.append("parent-checkpoint")
            raise RuntimeError(CANARY)
        with self.assertRaises(KeyboardInterrupt) as caught:
            parent()
        self.assertEqual(calls, ["child-cleanup", "parent-checkpoint"])
        self.assertNotIn(CANARY, "".join(traceback.format_exception(caught.exception)))
        # State is local to one invocation; an ordinary later failure remains ordinary.
        with self.assertRaisesRegex(ValueError, "plain"):
            preserve_interruption(lambda: (_ for _ in ()).throw(ValueError("plain")))()

    def test_transport_interrupt_closes_socket_before_cli_exit(self):
        fake = transport_fixtures.FakeConnection()
        fake.getresponse = Mock(side_effect=KeyboardInterrupt(CANARY))
        transport = RunpodHTTPTransport("fixture-sensitive-value", allow_live=True)
        def load(*args):
            return transport.request("GET", "/v2/pods", timeout_seconds=1, max_bytes=100)
        with patch("cineforge.runpod_transport._open_connection", return_value=fake):
            code, _, _ = self.invoke(["mock-run", "x"],
                                     patch.object(cli, "load_mock_plan", side_effect=load))
        self.assertEqual(code, 130)
        self.assertTrue(fake.closed)

    def test_complete_create_response_survives_close_interrupt_for_owned_cleanup(self):
        fake = transport_fixtures.FakeConnection(
            transport_fixtures.FakeResponse(b'{"id":"pod_fixture"}'))
        fake.close = Mock(side_effect=KeyboardInterrupt(CANARY))
        transport = RunpodHTTPTransport("fixture-sensitive-value", allow_live=True)
        with patch("cineforge.runpod_transport._open_connection", return_value=fake):
            reply = transport.request("POST", "/v2/pods", {}, timeout_seconds=1, max_bytes=100)
        self.assertTrue(reply.interrupted)
        self.assertEqual(json.loads(reply.body)["id"], "pod_fixture")

    def test_audio_interrupt_and_cleanup_interrupt_attempt_every_owned_finalizer(self):
        calls = []
        process = Mock(pid=123)
        process.stdin.close.side_effect = lambda: calls.append("stdin")
        process.stdout.close.side_effect = lambda: calls.append("stdout")
        process.wait.side_effect = lambda **kw: calls.append(("wait", kw["timeout"]))
        selector = Mock()
        selector.close.side_effect = lambda: calls.append("selector")
        with patch.object(audio_process.os, "killpg", side_effect=KeyboardInterrupt(CANARY)):
            with self.assertRaises(KeyboardInterrupt) as caught:
                audio_process._finalize(process, selector)
        process.kill.assert_called_once()
        self.assertEqual(calls, [("wait", 2), "selector", "stdin", "stdout"])
        self.assertNotIn(CANARY, "".join(traceback.format_exception(caught.exception)))

    def test_guard_interrupt_runs_bounded_owned_shutdown_before_130(self):
        plan = worker_guard.WorkerGuardPlan("pod_fixture", "session", "a" * 32,
                                           time.time() + 60, 60)
        stop = Mock(return_value=True)
        real_guard = worker_guard.run_guard
        def run(*args, **kwargs):
            return real_guard(plan,
                Mock(side_effect=KeyboardInterrupt(CANARY)), stop,
                max_stop_attempts=1, attempt_timeout_seconds=.2)
        started = time.monotonic()
        with patch.object(sys, "argv", ["guard", "--plan", "x", "--heartbeat", "x",
                                       "--allow-live"]), \
                patch.object(worker_guard, "read_private_json", return_value=plan.to_dict()), \
                patch.object(worker_guard, "worker_credential", return_value="fixture-key"), \
                patch.object(worker_guard, "run_guard", side_effect=run):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = worker_guard.main()
        self.assertEqual(code, 130)
        self.assertLess(time.monotonic() - started, 1)
        stop.assert_called_once()
        self.assertNotIn(CANARY, out.getvalue())

    def test_native_child_exit_130_reaps_before_propagating_cancellation(self):
        children = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory(prefix="rf197-native-child-") as root, \
                patch("cineforge.native_speech_process.subprocess.Popen", side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):
                run_child((sys.executable, "-c",
                           "import sys;sys.stdin.buffer.read();raise SystemExit(130)"), payload=b"fixture",
                          env={}, cwd=root, deadline=time.monotonic() + 4)
        self.assertEqual(children[0].returncode, 130)
        self.assertTrue(children[0].stdin.closed)
        self.assertTrue(children[0].stdout.closed)

    def test_native_worker_keeps_framed_failure_and_interrupt_exit(self):
        output = bytearray()
        stdin = Mock()
        stdin.buffer.read.return_value = pack({}, b"")
        def write(fd, raw):
            output.extend(raw)
            return len(raw)
        with patch.object(native_speech_worker.os, "dup", return_value=10), \
                patch.object(native_speech_worker.os, "open", return_value=11), \
                patch.object(native_speech_worker.os, "dup2"), \
                patch.object(native_speech_worker.os, "close"), \
                patch.object(native_speech_worker.os, "write", side_effect=write), \
                patch.object(native_speech_worker.sys, "stdin", stdin), \
                patch.object(native_speech_worker, "handle", side_effect=KeyboardInterrupt(CANARY)):
            self.assertEqual(native_speech_worker.main(), 130)
        self.assertEqual(unpack(bytes(output)), ({"version": 1, "status": "failed"}, b""))
        self.assertNotIn(CANARY.encode(), output)


class ManagedCancellationContractTests(unittest.TestCase):
    def test_interrupted_complete_create_is_checkpointed_and_released_before_exit(self):
        fixture = smoke_fixtures.ManagedSmokeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        request = fixture.cloud.request
        def interrupted_create(method, *args, **kwargs):
            reply = request(method, *args, **kwargs)
            return replace(reply, interrupted=True) if method == "POST" else reply
        with patch.object(fixture.cloud, "request", side_effect=interrupted_create):
            with self.assertRaises(KeyboardInterrupt):
                fixture.run_smoke("create-interrupted")
        self.assertFalse(fixture.cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in fixture.cloud.calls), 1)
        self.assertEqual(sum(c[0] == "DELETE" for c in fixture.cloud.calls), 1)
        report = json.loads((fixture.root / "create-interrupted" / "run.json").read_bytes())
        self.assertTrue(report["release_verified"])
        self.assertEqual(report["leases"][0]["state"], "released")
        self.assertFalse(fixture.workers[0].prepared)

    def test_cli_interrupt_releases_owned_pod_and_retains_private_checkpoint(self):
        fixture = smoke_fixtures.ManagedSmokeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args = ["cineforge", "cloud-recover", "--selection", "x", "--proposal", "x",
                "--budget-dir", "x"]
        with patch.object(sys, "argv", args), \
                patch.object(cli, "recovery_inputs", return_value=(None, None, None)), \
                patch.object(cli, "recover_managed", side_effect=lambda *a, **k:
                             fixture.run_smoke("interrupted", mode="interrupt")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        self.assertEqual(caught.exception.code, 130)
        self.assertFalse(fixture.cloud.pods)
        self.assertTrue(fixture.workers[-1].closed)
        self.assertEqual(sum(c[0] == "DELETE" for c in fixture.cloud.calls), 1)
        path = fixture.root / "interrupted" / "run.json"
        report = json.loads(path.read_bytes())
        self.assertEqual(report["outcome"], "incomplete")
        self.assertTrue(report["release_verified"])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        before = path.read_bytes()
        with patch.object(sys, "argv", ["cineforge", "--version"]), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main()
        self.assertEqual(path.read_bytes(), before)  # Historical record is not rewritten.
        self.assertNotIn("fixture-sensitive-value", err.getvalue())
        self.assertIn("CF_INTERRUPTED", err.getvalue())
