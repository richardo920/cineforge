"""Offline remote setup policy: downloads/processes are synthetic or patched."""

import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from test_cloud_budget import selection

from cineforge import worker_bootstrap, worker_guard
from cineforge.worker_bootstrap import (
    MODEL_REVISION,
    REVISION,
    BootstrapError,
    _DownloadRedirect,
    arm,
    bounded_process,
    clean_environment,
    dispatch,
    download,
    guard_alive,
    private_write,
    setup,
)


def exercise_descendant_pipe(test, root, module, function):
    """The outer process has its own timeout; always remove only this fixture's child group."""
    marker = root / "pipe-holder.pid"
    parent = ("import subprocess,sys;from pathlib import Path;"
              "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],"
              "start_new_session=True);Path(" + repr(str(marker)) + ").write_text(str(p.pid));"
              "print('synthetic parent exited')")
    harness = ("import time;from " + module + " import " + function + ";"
               "started=time.monotonic()\n"
               "try:\n " + function + "([" + repr(sys.executable) + ",'-c',"
               + repr(parent) + "],timeout=.2,payload=b'x'*262144)\n"
               "except Exception:\n print(round(time.monotonic()-started,3))\n"
               "else:\n raise SystemExit('fixture unexpectedly succeeded')\n")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    try:
        result = subprocess.run([sys.executable, "-c", harness], capture_output=True,
                                text=True, timeout=2, env=environment)
        test.assertEqual(result.returncode, 0, result.stderr)
        test.assertLess(float(result.stdout), 1.5)
        test.assertTrue(marker.exists(), "fixture did not start its pipe-holder child")
    finally:
        if marker.exists():
            identity = int(marker.read_text())
            try:
                if os.getpgid(identity) == identity:
                    os.killpg(identity, signal.SIGKILL)
            except ProcessLookupError:
                pass


def exercise_post_spawn_clock_failure(test, function, expected_exception):
    """Interrupt the first clock read after a real spawn; fixture cleanup also covers old code."""
    original_spawn, original_clock = subprocess.Popen, time.monotonic
    children = []
    def spawn(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        children.append(child)
        return child
    def clock():
        if children:
            raise KeyboardInterrupt("synthetic post-spawn interruption")
        return original_clock()
    try:
        with patch("cineforge.worker_bootstrap.subprocess.Popen", side_effect=spawn), \
                patch("cineforge.worker_bootstrap.time.monotonic", side_effect=clock):
            with test.assertRaises(expected_exception):
                function([sys.executable, "-c", "import time;time.sleep(10)"], timeout=1)
        test.assertEqual(len(children), 1)
        test.assertIsNotNone(children[0].poll())
        test.assertTrue(children[0].stdin.closed)
        test.assertTrue(children[0].stdout.closed)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
            child.stdin.close()
            child.stdout.close()


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
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-bootstrap-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
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
                patch("cineforge.worker_bootstrap.guard_alive"), \
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

    def test_guard_pid_reuse_zombie_missing_or_different_command_rejected(self):
        command = b"\0".join(str(x).encode() for x in ["/usr/bin/python3",
            self.root / "guard.py", "--plan", self.root / "plan.json", "--heartbeat",
            self.root / "heartbeat.json", "--allow-live"]) + b"\0"
        for status, cmdline, valid in [(b"123 (python3) S 1 1", command, True),
                                      (b"123 (python3) Z 1 1", command, False),
                                      (b"123 (python3) S 1 1", b"python3\0foreign.py\0", False),
                                      (b"malformed", command, False)]:
            with patch("cineforge.worker_bootstrap.os.kill"), \
                    patch("cineforge.worker_bootstrap.Path.open", side_effect=[io.BytesIO(status),
                                                                            io.BytesIO(cmdline)]):
                if valid:
                    guard_alive(self.root, 123)
                else:
                    with self.assertRaises(BootstrapError):
                        guard_alive(self.root, 123)

    def setup_fixture(self, *, failure=None, changed_runtime=False):
        selected = selection().declarations
        runtime = {"python_version": "3.12.3", "torch_version": "2.8.0+cu128",
            "torchvision_version": "0.23.0+cu128", "cuda_version": "12.8",
            "gpu_memory_mib": 24082, "gpu_raw_name": "NVIDIA GeForce RTX 4090"}
        calls = []
        def process(command, **options):
            calls.append((command, options))
            if command[:2] == ["git", "init"]:
                comfy = Path(command[-1])
                comfy.mkdir()
                (comfy / "requirements.txt").write_text("synthetic-package==1\n")
            if command[-1] == "HEAD":
                return REVISION.encode()+b"\n"
            if "freeze" in command:
                return b"synthetic-package==1\n"
            if failure == "pip" and "install" in command:
                raise TimeoutError("synthetic tool error")
            if failure == "git" and "fetch" in command:
                raise TimeoutError("synthetic tool error")
            return b""
        record_files = [{k: asset[k] for k in ("path", "sha256", "size_bytes")}
                        for asset in selected["model"]["files"]]
        downloader = Mock(side_effect=record_files if failure != "download" else TimeoutError())
        server = Mock(pid=123)
        server.poll.return_value = None
        opener = Mock()
        reply = Reply(b"")
        reply.status = 200
        opener.open.return_value = reply
        request = {"plan": {"session_id": "fixture", "pod_id": "pod_1"}, "selection": selected,
                   "image_digest": "sha256:" + hashlib.sha256(b"fixture image").hexdigest()}
        installed = dict(runtime, torch_version="changed") if changed_runtime else runtime
        with patch("cineforge.worker_bootstrap.worker_root", return_value=self.root), \
                patch("cineforge.worker_bootstrap.runtime_probe", side_effect=[runtime, installed]), \
                patch("cineforge.worker_bootstrap.bounded_process", side_effect=process), \
                patch("cineforge.worker_bootstrap.download", downloader), \
                patch("cineforge.worker_bootstrap.subprocess.Popen", return_value=server) as popen, \
                patch("cineforge.worker_bootstrap.urllib.request.build_opener", return_value=opener):
            try:
                result = setup(request, time.monotonic()+1)
            except (BootstrapError, TimeoutError):
                if not failure and not changed_runtime:
                    raise
                self.assertFalse(popen.called)
                return calls, None
            return calls, (result, popen.call_args)

    def test_setup_pins_source_constrains_stack_and_exposes_only_loopback_core_nodes(self):
        calls, value = self.setup_fixture()
        result, invocation = value
        self.assertEqual(result["verification"]["server_revision"], REVISION)
        self.assertEqual(result["verification"]["gpu"]["memory_mib"], 24082)
        self.assertTrue(result["verification"]["custom_nodes_disabled"])
        self.assertTrue(any(cmd[-1] == REVISION and "fetch" in cmd for cmd, opts in calls))
        pip = next(cmd for cmd, opts in calls if "install" in cmd)
        self.assertIn("--constraint", pip)
        self.assertIn("--isolated", pip)
        self.assertIn("https://pypi.org/simple", pip)
        command = invocation.args[0]
        self.assertEqual(command[command.index("--listen")+1], "127.0.0.1")
        self.assertIn("--disable-all-custom-nodes", command)
        self.assertIn("--disable-api-nodes", command)
        self.assertTrue(all(0 < opts["timeout"] <= 1 for cmd, opts in calls))
        self.assertNotIn("RUNPOD_API_KEY", invocation.kwargs["env"])

    def test_setup_git_pip_download_timeout_never_starts_server_and_changed_stack_rejects(self):
        for mode in ["git", "pip", "download", "stack"]:
            with self.subTest(mode=mode):
                self.root = Path(self.temp.name).resolve() / mode
                self.root.mkdir(mode=0o700)
                calls, result = self.setup_fixture(failure=None if mode == "stack" else mode,
                                                   changed_runtime=mode == "stack")
                self.assertIsNone(result)

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

    def test_exited_parent_detached_descendant_and_backpressure_cannot_hold_worker_deadline(self):
        exercise_descendant_pipe(self, self.root, "cineforge.worker_bootstrap", "bounded_process")

    def test_selector_initialization_failure_still_reaps_child_and_closes_pipes(self):
        original = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        with patch("cineforge.worker_bootstrap.subprocess.Popen", side_effect=spawn), \
                patch("cineforge.worker_bootstrap.selectors.DefaultSelector", side_effect=OSError):
            with self.assertRaises(OSError):
                bounded_process([sys.executable, "-c", "import time;time.sleep(10)"], timeout=.2)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stdin.closed)
        self.assertTrue(children[0].stdout.closed)

    def test_post_spawn_clock_interruption_reaps_worker_child_and_closes_pipes(self):
        exercise_post_spawn_clock_failure(self, bounded_process, KeyboardInterrupt)

    def test_standalone_failure_does_not_echo_input_or_traceback(self):
        request = {"operation": "fixture-sensitive-value", "allow_live": False, "timeout_seconds": 1}
        result = subprocess.run([sys.executable, worker_bootstrap.__file__],
            input=json.dumps(request).encode(), capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b'{"status":"worker_operation_failed"}\n')
        self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main()
