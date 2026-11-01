"""Offline remote setup policy: downloads/processes are synthetic or patched."""

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cineforge import worker_bootstrap, worker_guard
from cineforge.worker_bootstrap import (
    MODEL_REVISION,
    BootstrapError,
    _DownloadRedirect,
    arm,
    bounded_process,
    clean_environment,
    dispatch,
    download,
    private_write,
)


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def read1(self, limit):
        output, self.payload = self.payload[:limit], self.payload[limit:]
        return output

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-bootstrap-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"synthetic weights fixture; no tensor content"
        path = "split_files/vae/fixture.safetensors"
        self.asset = {"path": path, "revision": MODEL_REVISION,
            "url": "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/"+MODEL_REVISION+"/"+path,
            "size_bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}

    def test_no_operation_or_download_without_explicit_bounded_opt_in(self):
        with self.assertRaises(BootstrapError):
            dispatch({"operation": "ping", "allow_live": False, "timeout_seconds": 1})
        self.assertEqual(dispatch({"operation": "ping", "allow_live": True,
                                  "timeout_seconds": 1}), {"status": "ssh_ready"})
        for value in [True, 0, -1, float("nan"), float("inf"), 1800]:
            with self.assertRaises(BootstrapError):
                dispatch({"operation": "ping", "allow_live": True, "timeout_seconds": value})

    def test_verified_download_is_worker_local_and_checks_size_hash(self):
        opener = Mock()
        opener.open.return_value = Reply(self.payload)
        with patch("cineforge.worker_bootstrap.urllib.request.build_opener", return_value=opener):
            record = download(self.root, self.asset, time.monotonic()+5)
        self.assertEqual(record["sha256"], self.asset["sha256"])
        self.assertEqual((self.root/"ComfyUI/models/vae/fixture.safetensors").read_bytes(), self.payload)
        self.assertFalse(list(self.root.rglob("*.partial")))

    def test_download_mismatch_removes_partial_and_never_promotes_file(self):
        for data in [b"wrong", self.payload+b"overflow"]:
            opener = Mock()
            opener.open.return_value = Reply(data)
            with patch("cineforge.worker_bootstrap.urllib.request.build_opener", return_value=opener):
                with self.assertRaises(BootstrapError):
                    download(self.root, self.asset, time.monotonic()+5)
            self.assertFalse((self.root/"ComfyUI/models/vae/fixture.safetensors").exists())
            self.assertFalse(list(self.root.rglob("*.partial")))

    def test_unapproved_model_url_path_and_redirect_are_rejected_before_network(self):
        for asset in [dict(self.asset, url="https://user:fixture@evil.invalid"),  # pragma: allowlist secret - synthetic URL fixture
                      dict(self.asset, path="split_files/vae/../../escape.safetensors"),
                      dict(self.asset, revision="main")]:
            with patch("cineforge.worker_bootstrap.urllib.request.build_opener") as opener:
                with self.assertRaises(BootstrapError):
                    download(self.root, asset, time.monotonic()+5)
                opener.assert_not_called()
        redirect = _DownloadRedirect()
        for url in ["http://huggingface.co/file", "https://evil.invalid/file",
                    "https://user:fixture@huggingface.co/file"]:  # pragma: allowlist secret - synthetic redirect fixture
            with self.assertRaises(BootstrapError):
                redirect.redirect_request(None, None, 302, "", {}, url)

    def test_guard_own_pod_source_and_pid_verified_before_runtime_measurement(self):
        plan = worker_guard.WorkerGuardPlan("pod_1", "fixture", "a"*32, time.time()+100, 30)
        source = b"synthetic guard source; not executed by this fixture"
        request = {"plan": plan.to_dict(), "guard_source": base64.b64encode(source).decode(),
                   "guard_sha256": hashlib.sha256(source).hexdigest(), "controller_unix": time.time()}
        guard = Mock()
        guard.WorkerGuardPlan.from_dict.return_value = plan
        guard.verify_own_pod.return_value = True
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch("cineforge.worker_bootstrap.worker_root", return_value=self.root), \
                patch("cineforge.worker_bootstrap.load_guard", return_value=guard), \
                patch("cineforge.worker_bootstrap.subprocess.Popen", return_value=process), \
                patch("cineforge.worker_bootstrap.runtime_probe", return_value={"fixture": True}):
            result = arm(request, time.monotonic()+5)
        guard.verify_own_pod.assert_called_once_with(plan, allow_live=True)
        self.assertEqual(result["receipt"]["pid"], 123)
        self.assertTrue(result["receipt"]["operator_verified_own_pod"])
        self.assertEqual((self.root/"guard.py").read_bytes(), source)
        self.assertEqual(json.loads((self.root/"heartbeat.json").read_text())["pod_id"], "pod_1")

    def test_arm_rejects_source_tampering_before_worker_write(self):
        request = {"plan": {}, "guard_source": base64.b64encode(b"fixture").decode(),
                   "guard_sha256": "0"*64}
        with self.assertRaises(BootstrapError):
            arm(request, time.monotonic()+1)
        self.assertFalse(list(self.root.iterdir()))

    def test_private_heartbeat_updates_do_not_follow_symlinks(self):
        private_write(self.root, "heartbeat.json", b"first")
        private_write(self.root, "heartbeat.json", b"second", replace=True)
        self.assertEqual((self.root/"heartbeat.json").read_bytes(), b"second")
        self.assertEqual((self.root/"heartbeat.json").stat().st_mode & 0o777, 0o600)
        (self.root/"heartbeat.json").unlink()
        (self.root/"heartbeat.json").symlink_to(self.root/"foreign")
        with self.assertRaises(BootstrapError):
            private_write(self.root, "heartbeat.json", b"third", replace=True)
        self.assertFalse((self.root/"foreign").exists())

    def test_clean_child_environment_and_bounded_tools(self):
        self.assertNotIn("RUNPOD_API_KEY", clean_environment())
        self.assertNotIn("PIP_INDEX_URL", clean_environment())
        self.assertEqual(bounded_process([sys.executable, "-c", "print('fixture')"], timeout=2),
                         b"fixture\n")
        started = time.monotonic()
        with self.assertRaises(BootstrapError):
            bounded_process([sys.executable, "-c", "import time;time.sleep(30)"], timeout=.1)
        self.assertLess(time.monotonic()-started, 3)
        with self.assertRaises(BootstrapError):
            bounded_process([sys.executable, "-c", "print('x'*100000)"], timeout=2, max_output=100)

    def test_standalone_failure_does_not_echo_input_or_traceback(self):
        request = {"operation": "fixture-sensitive-value", "allow_live": False, "timeout_seconds": 1}
        result = subprocess.run([sys.executable, worker_bootstrap.__file__],
            input=json.dumps(request).encode(), capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b'{"status":"worker_operation_failed"}\n')
        self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main()
