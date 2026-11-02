"""Actual Linux first-party process identity evidence; no Comfy/SDK/GPU readiness claim."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cineforge.comfy_runtime_launcher import launch_command
from cineforge.comfy_runtime_worker import _server_start, server_facts
from cineforge.native_remote_plan import nr_encode


class ClosedComfyProcessTests(unittest.TestCase):
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
                "import time,socket\nfrom pathlib import Path\n"
                "def main():\n sock=socket.socket();sock.bind(('127.0.0.1',8188));sock.listen()\n Path("
                + repr(str(marker))
                + ").write_text('ready')\n time.sleep(20)\n return 0\n"
            )
            executable = Path(sys.executable).resolve()

            def pin(path):
                return {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            plan = {"interpreter": pin(executable), "native": [], "installed": []}
            deadline = time.monotonic() + 15
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
