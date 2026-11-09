"""Fixed M5 bootstrap selection; fake SSH and stdlib-only interpreter stand-ins."""
import json
import platform
import shlex
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import test_s2v_stage as stage_fixtures
from test_conditioning import context as m4_context
from test_conditioning import runtime_fixture, selected
from test_ssh_worker import Budget, Transport

from cineforge import native_remote_bootstrap
from cineforge import worker_bootstrap as bootstrap
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source
from cineforge.worker_transport_source import worker_command_source
from cineforge.workflows import canonical


class M5PythonTests(unittest.TestCase):
    def setUp(self):
        fixture = stage_fixtures.PairTransferTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.worker = SSHWorker(fixture.spec, fixture.gate.service, fixture.budget, None,
            fixture.key, fixture.root / "real-request", allow_live=True)
        self.addCleanup(self.worker.close)
        self.worker._host, self.worker._port = "8.8.8.8", 22000

    def test_all_m5_rpc_families_fix_interpreter_flags_and_immutable_source(self):
        operations = ("ping", "arm", "heartbeat", "setup", "stage_reference",
            "s2v_pair_begin", "s2v_pair_chunk", "s2v_pair_finish", "native_capabilities",
            "native_preflight", "native_begin", "native_chunk", "native_finish",
            "native_setup", "native_select", "native_fetch", "native_invoke",
            "native_comfy_install", "native_comfy_model", "native_comfy_setup", "native_comfy_verify")
        source = worker_command_source(worker_source())
        injection = "synthetic-not-a-command; $(false)"
        for operation in operations:
            with self.subTest(operation=operation):
                reply = {"verification": {}, "runtime": {}} if operation == "setup" else {}
                with patch("cineforge.ssh_worker.run_process", return_value=canonical(reply)) as call:
                    request = dict(operation=operation, executable=injection, flags=injection,
                                   source=injection)
                    cap = time.monotonic()+10
                    self.worker._request(request, 5, supervise=False, absolute_deadline=cap)
                command, limits = call.call_args.args[0], call.call_args.kwargs
                self.assertEqual(shlex.split(command[-1]),
                                 ["/usr/bin/python3.11", "-I", "-B", "-S", "-c", source])
                self.assertNotIn(injection, command[-1])
                body = json.loads(limits["payload"])
                self.assertEqual(body["source"], injection)
                self.assertEqual(body["profile"], self.fixture.gate.service.worker_profile)
                self.assertIs(body["allow_live"], True)
                self.assertEqual(limits["absolute_deadline"], cap)
                self.assertGreater(limits["timeout"], 0)
                self.assertLessEqual(limits["timeout"], 5)
                self.assertLess(len(command[-1].encode()), 131072)
        self.assertLessEqual(len(source.encode()), MAX_SOURCE_BYTES)

    def test_m3_m4_default_and_native_prefixes_remain_exact(self):
        from test_cloud_budget import selection
        from test_runpod import spec

        root = self.fixture.root / "legacy"
        root.mkdir()
        m4_gate, *_ = m4_context(root)
        for choice in (selection(), selected(m4_gate)):
            worker = SSHWorker(spec(), choice, Budget(), Transport(), self.fixture.key,
                               root / "state", allow_live=True)
            self.addCleanup(worker.close)
            worker._host, worker._port = "8.8.8.8", 22000
            for operation, prefix in (("ping", "python3 -I -c "),
                                      ("native_preflight", "python3 -I -B -S -c ")):
                with self.subTest(kind=type(choice).__name__, operation=operation):
                    with patch("cineforge.ssh_worker.run_process", return_value=b"{}") as call:
                        worker._request({"operation": operation}, 5, supervise=False)
                    command = call.call_args.args[0][-1]
                    self.assertEqual(command, prefix + shlex.quote(worker_command_source(worker_source())))

    def test_default_312_rejected_before_hardware_and_native_effects_311_admits_only_facts(self):
        deadline = time.monotonic()+10
        with patch.object(sys, "version_info", (3, 12, 3)), \
                patch.object(sys, "platform", "linux"), \
                patch.object(platform, "machine", return_value="x86_64"), \
                patch.object(platform, "libc_ver", return_value=("glibc", "2.39")), \
                patch.object(native_remote_bootstrap, "bounded_process") as native_child, \
                patch.object(bootstrap, "bounded_process") as child:
            with self.assertRaisesRegex(bootstrap.BootstrapError, "fixed interpreter ABI"):
                bootstrap.s2v_hardware_admission(self.fixture.root, deadline)
            with self.assertRaisesRegex(ValueError, "fixed native worker ABI"):
                native_remote_bootstrap.nr_preflight(deadline)
            child.assert_not_called()
            native_child.assert_not_called()
        runtime = runtime_fixture()
        resources = {key: runtime[key] for key in bootstrap.M4_RESOURCE_FIELDS}
        with patch.object(sys, "version_info", (3, 11, 14)), \
                patch.object(sys, "platform", "linux"), \
                patch.object(platform, "machine", return_value="x86_64"), \
                patch.object(platform, "python_version", return_value="3.11.14"), \
                patch.object(bootstrap, "resource_evidence", return_value=resources), \
                patch.object(bootstrap, "bounded_process", return_value=
                    b"NVIDIA GeForce RTX 5090,32768,64,12.0,580.95.05\n") as child:
            result = bootstrap.s2v_hardware_admission(self.fixture.root, deadline)
        self.assertEqual(result["python_version"], "3.11.14")
        self.assertEqual(result["admission_status"], "hardware_only_unverified_runtime")
        self.assertNotIn("torch_version", result)
        self.assertNotIn("compute_precision", result)
        self.assertEqual(child.call_args.kwargs["absolute_deadline"], deadline)
        for bad in (dict(python_version="3.12.3"), dict(gpu_memory_mib=1),
                    dict(capability=[8, 9]), dict(driver_version="unobserved")):
            with self.subTest(change=bad), self.assertRaises(bootstrap.BootstrapError):
                bootstrap.validate_s2v_admission(dict(result, **bad))

    def test_actual_stdlib_standin_retains_flags_and_ignores_site_hooks(self):
        marker = self.fixture.root / "startup-marker"
        (self.fixture.root / "sitecustomize.py").write_text(
            "from pathlib import Path\nPath("+repr(str(marker))+").touch()\n")
        (self.fixture.root / "unprocessed.pth").write_text("import sitecustomize\n")
        def local_standin(command, **limits):
            argv = shlex.split(command[-1])
            self.assertEqual(argv[:5], ["/usr/bin/python3.11", "-I", "-B", "-S", "-c"])
            # Explicit interpreter stand-in, not SSH, pinned image or hardware evidence.
            flags = subprocess.run([sys.executable, *argv[1:4], "-c",
                "import json,sys;print(json.dumps([sys.flags.isolated,"
                "sys.flags.dont_write_bytecode,sys.flags.no_site,'site' in sys.modules]))"],
                capture_output=True, timeout=5, env={"PATH": ""})
            self.assertEqual(flags.returncode, 0)
            self.assertEqual(json.loads(flags.stdout), [1, 1, 1, False])
            result = subprocess.run([sys.executable, *argv[1:]], input=limits["payload"],
                capture_output=True, cwd=self.fixture.root, timeout=5,
                env={"PATH": "", "PYTHONPATH": str(self.fixture.root),
                     "PYTHONUSERBASE": str(self.fixture.root)})
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout
        with patch("cineforge.ssh_worker.run_process", side_effect=local_standin):
            result = self.worker._request({"operation": "ping"}, 5, supervise=False)
        self.assertEqual(result, {"status": "ssh_ready"})
        self.assertFalse(marker.exists())
        self.assertEqual(list(self.fixture.root.glob("**/*.pyc")), [])

    def test_expired_original_cap_never_dispatches_pinned_command(self):
        with patch("cineforge.ssh_worker.run_process") as child:
            with self.assertRaises(WorkerError):
                self.worker._request({"operation": "ping"}, 5, supervise=False,
                                     absolute_deadline=time.monotonic()-1)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
