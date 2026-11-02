"""Actual Linux first-party process identity evidence; no Comfy/SDK/GPU readiness claim."""

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cineforge.comfy_runtime_launcher import launch_command
from cineforge.comfy_runtime_worker import _server_start, server_facts
from cineforge.native_remote_package import native_package
from cineforge.native_remote_plan import nr_encode
from cineforge.native_speech_integrity import NativeBinary, _directory, verify_binary
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source


class ClosedComfyProcessTests(unittest.TestCase):
    def test_actual_linux_quoted_worker_command_and_package_remain_bounded(self):
        self.assertEqual(sys.platform, "linux", "Linux command evidence cannot be skipped")
        source = worker_source()
        package, pins = native_package()
        command = "python3 -I -B -S -c " + shlex.quote(source.decode())
        m5_command = "/usr/bin/python3.11 -I -B -S -c " + shlex.quote(source.decode())
        self.assertLessEqual(len(source), MAX_SOURCE_BYTES)
        self.assertLess(len(command.encode()), 128 * 1024)
        self.assertLess(len(m5_command.encode()), 128 * 1024)
        self.assertLessEqual(len(package), 2 * 1024**2)
        # Legacy native SSH prefix through OS PATH; same no-site flags/source as
        # M5's fixed /usr/bin/python3.11 command. This CI interpreter is a stand-in,
        # not pinned-image CPython3.11 or hardware evidence. Zero setup effects.
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            input=b"{}",
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            timeout=10,
        )
        self.assertEqual(result.returncode, 2, result.stderr.decode())
        self.assertEqual(json.loads(result.stdout)["status"], "worker_operation_failed")
        print(
            "COMFY_FIRST_PARTY_COMMAND_EVIDENCE="
            + json.dumps(
                {
                    "source_bytes": len(source),
                    "ssh_shell_argument_bytes": len(command.encode()),
                    "m5_fixed_ssh_argument_bytes": len(m5_command.encode()),
                    "m5_fixed_prefix": "/usr/bin/python3.11 -I -B -S -c ",
                    "executed_interpreter_kind": "legacy native OS PATH stand-in",
                    "package_bytes": len(package),
                    "package_files": len(pins),
                    "actual_linux_shell_execution": True,
                    "invalid_admission_refused": True,
                    "sdk_gpu_model_execution": False,
                },
                sort_keys=True,
            )
        )

    def test_fixed_no_site_process_identity_and_loaded_file_allowlist(self):
        self.assertEqual(sys.platform, "linux", "Linux first-party evidence cannot be skipped")
        with tempfile.TemporaryDirectory(prefix="cf-comfy-linux-process-") as temporary:
            root = Path(temporary).resolve()
            native = root / "native"
            package = native / "firstparty/cineforge"
            package.mkdir(parents=True, mode=0o700)
            (package / "__init__.py").write_text("")
            marker = root / "ready"
            # Deliberately first-party stdlib stand-in, not actual Comfy or GPU code.
            (package / "comfy_runtime_launcher.py").write_text(
                "import time,socket,sys,json\nfrom pathlib import Path\n"
                "def main():\n sock=socket.socket();sock.bind(('127.0.0.1',8188));sock.listen()\n Path("
                + repr(str(marker))
                + ").write_text(json.dumps({'version':sys.version.split()[0],"
                "'executable':sys.executable}))\n time.sleep(20)\n return 0\n"
            )
            # setup-python's toolcache can have group-writable ancestors. Keep
            # those forbidden; use the explicit protected OS capability for this
            # stdlib-only stand-in, independently of the controller matrix.
            executable = Path("/usr/bin/python3").resolve(strict=True)

            def pin(path):
                return {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            plan = {"interpreter": pin(executable), "native": [], "installed": []}
            matrix_unprotected = []
            for line in Path("/proc/self/maps").read_text().splitlines():
                mapped = line.split(maxsplit=5)[-1]
                if not mapped.startswith("/") or mapped.startswith("/dev/"):
                    continue
                parent = Path(mapped).parent
                try:
                    descriptor = _directory(str(parent), private=False)
                except (ValueError, OSError):
                    ancestors = []
                    for ancestor in reversed((parent, *parent.parents)):
                        info = ancestor.lstat()
                        ancestors.append(
                            {
                                "path": str(ancestor),
                                "uid": info.st_uid,
                                "mode": oct(info.st_mode & 0o7777),
                            }
                        )
                    row = {"mapped": mapped, "ancestors": ancestors}
                    if row not in matrix_unprotected:
                        matrix_unprotected.append(row)
                else:
                    os.close(descriptor)
            unsafe = root / "unsafe-native-ancestor"
            unsafe.mkdir(mode=0o700)
            unsafe.chmod(0o770)
            try:
                with self.assertRaisesRegex(ValueError, "ancestor is not protected"):
                    _directory(str(unsafe), private=False)
            finally:
                unsafe.chmod(0o700)
            deadline = time.monotonic() + 15
            verify_binary(NativeBinary(**plan["interpreter"]), deadline)
            command, environment = launch_command(plan, root, native, "server", deadline)
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            try:
                for _ in range(500):
                    if marker.exists():
                        break
                    if process.poll() is not None:
                        self.fail("owned first-party stand-in exited")
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                paths = set()
                for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines():
                    path = line.split(maxsplit=5)[-1]
                    if path.startswith("/") and not path.startswith("/dev/"):
                        paths.add(path)
                plan["native"] = [
                    pin(Path(path)) for path in sorted(paths) if path != str(executable)
                ]
                (root / "server-pid.json").write_bytes(
                    nr_encode({"pid": process.pid, "start_ticks": _server_start(process.pid)})
                )
                (root / "server-pid.json").chmod(0o600)
                (native / "comfy-local-cap.json").write_bytes(nr_encode({"deadline": deadline}))
                (native / "comfy-local-cap.json").chmod(0o600)
                facts = server_facts(plan, root, native, deadline)
                self.assertEqual(facts["pid"], process.pid)
                self.assertEqual(set(facts["native_files"]), paths)
                missing = dict(plan, native=plan["native"][:-1])
                with self.assertRaises(ValueError):
                    server_facts(missing, root, native, deadline)
                record = {"pid": process.pid, "start_ticks": "changed"}
                (root / "server-pid.json").write_bytes(nr_encode(record))
                with self.assertRaises(ValueError):
                    server_facts(plan, root, native, deadline)
                print(
                    "COMFY_FIRST_PARTY_PROCESS_EVIDENCE="
                    + json.dumps(
                        {
                            "fixed_no_site_command": True,
                            "actual_pid_start_ticks": True,
                            "mapped_files_checked": len(paths),
                            "unknown_native_file_refused": True,
                            "pid_reuse_refused": True,
                            "unsafe_ancestor_refused": True,
                            "child_interpreter": {
                                **plan["interpreter"],
                                **json.loads(marker.read_text()),
                            },
                            "controller_interpreter": sys.executable,
                            "controller_matrix_unprotected_maps": matrix_unprotected,
                            "comfy_sdk_gpu_execution": False,
                        },
                        sort_keys=True,
                    )
                )
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
