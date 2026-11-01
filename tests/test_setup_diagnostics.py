"""Synthetic setup failures: no credentials, SSH, model bytes or paid services."""

import io
import json
import sys
import time
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import test_conditioning as conditioning_fixtures
from test_conditioning import Fixture, Worker, runtime_fixture, selected

from cineforge import conditioning_profile as profile
from cineforge import worker_bootstrap as bootstrap
from cineforge.conditioning import ConditioningError
from cineforge.private_store import PrivateRunDirectory
from cineforge.ssh_worker import SSHWorker, WorkerError, run_process

CANARY = "synthetic-sensitive-diagnostic-canary"


def diagnostic(**changes):
    value = dict(schema="m4-setup-failure-v1", stage="weights", asset="text_encoder",
                 reason="http_error", http_status=403)
    value.update(changes)
    return value


def envelope(value=None):
    return bootstrap.encode({"status": "worker_operation_failed",
        "setup_failure": diagnostic() if value is None else value}) + b"\n"


def child(raw, code=2):
    return [sys.executable, "-c", "import sys;sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]));"
            "sys.stderr.write(sys.argv[2]);sys.exit(int(sys.argv[3]))", raw.hex(), CANARY, str(code)]


class DiagnosticTests(Fixture):
    def test_only_exact_exit2_setup_optin_decodes_an_allowlisted_error(self):
        raw = envelope()
        with self.assertRaises(bootstrap.SetupFailure) as caught:
            bootstrap.bounded_process(child(raw), timeout=2, decode_setup_failure=True)
        self.assertEqual(caught.exception.diagnostic, diagnostic())
        self.assertNotIn(CANARY, str(caught.exception))
        for code, optin in ((2, False), (1, True), (3, True)):
            with self.subTest(code=code, optin=optin), self.assertRaises(bootstrap.BootstrapError) as error:
                bootstrap.bounded_process(child(raw, code), timeout=2, decode_setup_failure=optin)
            self.assertIs(type(error.exception), bootstrap.BootstrapError)
            self.assertNotIn(CANARY, str(error.exception))
        self.assertEqual(bootstrap.bounded_process(child(raw, 0), timeout=2,
            decode_setup_failure=True), raw)  # A zero-exit frame does not become a typed failure.

    def test_duplicate_extra_unknown_mismatched_or_oversize_frames_remain_generic(self):
        malformed = [envelope() + b"\n", envelope() + b"{}", b"x"*513,
            envelope().replace(b'"http_status":403', b'"http_status":403,"http_status":403'),
            bootstrap.encode({"verification": {"status": "success"}}) + b"\n"]
        variants = [dict(diagnostic(), unexpected=CANARY), dict(diagnostic(), reason=CANARY),
            dict(diagnostic(), stage=CANARY), dict(diagnostic(), asset=CANARY),
            dict(diagnostic(), http_status=True), dict(diagnostic(), http_status=399),
            dict(diagnostic(), http_status=600), dict(diagnostic(), http_status="403"),
            dict(diagnostic(), stage="source"), dict(diagnostic(), asset="none"),
            dict(diagnostic(), reason="timeout"), dict(diagnostic(), reason=[CANARY])]
        malformed += [envelope(v) for v in variants]
        for raw in malformed:
            with self.subTest(raw_size=len(raw)), self.assertRaises(bootstrap.BootstrapError) as caught:
                bootstrap.bounded_process(child(raw), timeout=2, decode_setup_failure=True)
            self.assertIs(type(caught.exception), bootstrap.BootstrapError)
            self.assertNotIn(CANARY, str(caught.exception))

    def setup_case(self, *, ordinal=2, error=None, process_stage=None, server_exited=False):
        root = self.root / ("worker-" + str(time.monotonic_ns()))
        root.mkdir()
        request = {"plan": {"session_id": "fixture", "pod_id": "pod_1"},
            "selection": selected(self.gate).declarations,
            "profile": selected(self.gate).worker_profile,
            "image_digest": profile.IMAGE.split("@", 1)[1]}
        def process(command, **options):
            if command[:2] == ["git", "init"]:
                Path(command[2]).mkdir()
                (Path(command[2]) / "requirements.txt").write_text("synthetic")
            if process_stage == "source" and command[0] == "git":
                raise bootstrap.BootstrapError(CANARY, reason="process_failed")
            if process_stage == "dependencies" and "install" in command:
                raise bootstrap.BootstrapError(CANARY, reason="process_failed")
            if process_stage == "metadata" and "freeze" in command:
                raise bootstrap.BootstrapError(CANARY, reason="process_failed")
            if command[-2:] == ["rev-parse", "HEAD"]:
                return profile.RUNNER_REVISION.encode()
            return b""
        calls = []
        def download(_root, asset, _deadline, **options):
            calls.append(asset["path"])
            if error is not None and len(calls) == ordinal:
                raise error
            return {key: asset[key] for key in ("path", "sha256", "size_bytes")}
        with patch.object(bootstrap, "worker_root", return_value=root), \
                patch.object(bootstrap, "runtime_probe", return_value=runtime_fixture()), \
                patch.object(bootstrap, "bounded_process", side_effect=process), \
                patch.object(bootstrap, "download", side_effect=download), \
                patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=50*1024**3)), \
                patch.object(bootstrap.subprocess, "Popen", return_value=Mock(pid=123, poll=Mock(return_value=1 if server_exited else None))) as server:
            try:
                return bootstrap.setup(request, time.monotonic()+1)
            finally:
                self.last_downloads, self.last_server = calls, server

    def test_each_weight_ordinal_retains_only_its_fixed_role_and_http_status(self):
        for ordinal, asset in enumerate(("diffusion_model", "text_encoder", "vae"), 1):
            for status in (403, 404, 429, 503):
                with self.subTest(ordinal=ordinal, status=status), self.assertRaises(bootstrap.SetupFailure) as caught:
                    self.setup_case(ordinal=ordinal, error=urllib.error.HTTPError(
                        "https://private.invalid/" + CANARY, status, CANARY, {}, None))
                self.assertEqual(caught.exception.diagnostic, dict(diagnostic(), asset=asset, http_status=status))
                self.assertNotIn(CANARY, str(caught.exception))
                self.assertEqual(len(self.last_downloads), ordinal)
                self.last_server.assert_not_called()

    def test_timeout_network_integrity_and_unknown_errors_do_not_retain_remote_strings(self):
        errors = [(TimeoutError(CANARY), "timeout"),
            (urllib.error.URLError(CANARY), "network_error"),
            (urllib.error.URLError(TimeoutError(CANARY)), "timeout"),
            (bootstrap.BootstrapError(CANARY, reason="integrity_failed"), "integrity_failed"),
            (ValueError(CANARY), "unknown")]
        for error, reason in errors:
            with self.subTest(reason=reason), self.assertRaises(bootstrap.SetupFailure) as caught:
                self.setup_case(error=error)
            expected = diagnostic()
            expected.pop("http_status")
            expected["reason"] = reason
            self.assertEqual(caught.exception.diagnostic, expected)
            self.assertNotIn(CANARY, bootstrap.encode(caught.exception.diagnostic).decode())

    def test_real_download_classifies_tiny_synthetic_integrity_failure_and_removes_partial(self):
        asset = profile.model_files()[1]
        response = MagicMock()
        response.__enter__.return_value = response
        response.read1.side_effect = [b"synthetic-fixture", b""]
        opener = Mock()
        opener.open.return_value = response
        with patch.object(bootstrap.urllib.request, "build_opener", return_value=opener), \
                self.assertRaises(bootstrap.BootstrapError) as caught:
            bootstrap.download(self.root, asset, time.monotonic()+1,
                               profile=selected(self.gate).worker_profile)
        self.assertEqual(caught.exception.reason, "integrity_failed")
        self.assertFalse(list(self.root.rglob("*.partial")))
        self.assertFalse(list(self.root.rglob("*.safetensors")))

    def test_process_and_server_stages_are_descriptive_without_subprocess_output(self):
        for stage in ("source", "dependencies", "metadata"):
            with self.subTest(stage=stage), self.assertRaises(bootstrap.SetupFailure) as caught:
                self.setup_case(process_stage=stage)
            self.assertEqual(caught.exception.diagnostic, {"schema": "m4-setup-failure-v1",
                "stage": stage, "asset": "none", "reason": "process_failed"})
        with self.assertRaises(bootstrap.SetupFailure) as caught:
            self.setup_case(server_exited=True)
        self.assertEqual(caught.exception.diagnostic, {"schema": "m4-setup-failure-v1",
            "stage": "server_health", "asset": "none", "reason": "server_exited"})

    def test_main_keeps_exit2_and_emits_only_the_sanitized_frame(self):
        request = io.BytesIO(bootstrap.encode({"timeout_seconds": 1}))
        output = io.StringIO()
        with patch.object(bootstrap.sys, "stdin", SimpleNamespace(buffer=request)), \
                patch.object(bootstrap, "dispatch", side_effect=bootstrap.SetupFailure(diagnostic())), \
                patch.object(bootstrap.signal, "signal"), patch.object(bootstrap.signal, "setitimer"), \
                redirect_stdout(output):
            self.assertEqual(bootstrap.main(), 2)
        self.assertEqual(output.getvalue().encode(), envelope())

    def test_only_m4_setup_ssh_opts_into_decode_and_retains_first_safe_failure(self):
        worker = object.__new__(SSHWorker)
        worker._source, worker._host = b"synthetic", "fixture.invalid"
        worker._command = lambda: ["synthetic"]
        worker.check = lambda: None
        worker.setup_failure = None
        original_run = run_process
        def invoke(_command, **options):
            return original_run(child(envelope()), timeout=2,
                decode_setup_failure=options.get("decode_setup_failure", False))
        for profile_value, operation in ((None, "setup"), ({}, "heartbeat"), ({}, "setup")):
            worker._profile = profile_value
            with self.subTest(profile=profile_value, operation=operation), \
                    patch("cineforge.ssh_worker.run_process", side_effect=invoke), \
                    self.assertRaises(WorkerError) as caught:
                worker._request({"operation": operation}, 2)
            self.assertNotIn(CANARY, str(caught.exception))
            self.assertEqual(worker.setup_failure, diagnostic() if profile_value is not None and operation == "setup" else None)
        other = dict(diagnostic(), http_status=503)
        with patch("cineforge.ssh_worker.run_process", side_effect=bootstrap.SetupFailure(other)), \
                self.assertRaises(WorkerError):
            worker._request({"operation": "setup"}, 2)
        self.assertEqual(worker.setup_failure, diagnostic())
        worker.setup_failure = None
        with patch("cineforge.ssh_worker.run_process", return_value=envelope()), \
                self.assertRaises(WorkerError):
            worker._request({"operation": "setup"}, 2)
        self.assertIsNone(worker.setup_failure)  # Zero-exit error JSON cannot be accepted as setup.


class DurableRunTests(Fixture):
    run_case = conditioning_fixtures.ManagedTests.run_case

    def fail_worker(self, *args, **options):
        worker = self
        worker.setup_failure = diagnostic()
        raise WorkerError(CANARY)

    def test_setup_failure_is_persisted_before_cleanup_failure_without_replacing_primary(self):
        def cleanup(_provider, _identity):
            self.workers[0].setup_failure = dict(diagnostic(), http_status=503)
            self.assertEqual(json.loads((self.root / "run/run.json").read_bytes())["setup_failure"], diagnostic())
            raise RuntimeError(CANARY)
        with patch.object(Worker, "prepare", new=DurableRunTests.fail_worker), \
                patch("cineforge.runpod.RunpodPodProvisioner.release", new=cleanup), \
                self.assertRaises(ConditioningError):
            self.run_case()
        report = json.loads((self.root / "run/run.json").read_bytes())
        self.assertEqual(report["setup_failure"], diagnostic())
        self.assertEqual(report["stage"], "worker_setup")
        self.assertFalse(report["release_verified"])
        self.assertEqual(report["shots"], [])
        self.assertNotIn(CANARY, json.dumps(report))
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)

    def test_diagnostic_checkpoint_failure_still_deletes_owned_pod_and_keeps_failure(self):
        checkpoint = PrivateRunDirectory.checkpoint
        failed = []
        def write(store, payload):
            if json.loads(payload).get("setup_failure") and not failed:
                failed.append(True)
                raise OSError(CANARY)
            return checkpoint(store, payload)
        with patch.object(Worker, "prepare", new=DurableRunTests.fail_worker), \
                patch.object(PrivateRunDirectory, "checkpoint", new=write), \
                self.assertRaises(ConditioningError):
            self.run_case()
        report = json.loads((self.root / "run/run.json").read_bytes())
        self.assertEqual(report["setup_failure"], diagnostic())
        self.assertTrue(report["release_verified"])
        self.assertFalse(self.cloud.pods)
        self.assertNotIn(CANARY, json.dumps(report))
