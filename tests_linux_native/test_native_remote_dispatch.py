"""Required Linux synthetic remote-dispatch proof; no SDK, GPU, SSH or downloads."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cineforge.native_remote_bootstrap import nr_check
from cineforge.native_speech_integrity import NativeBinary, native_subject
from cineforge.native_speech_process import run_child

# Reuse only synthetic controller/RPC fixtures; package/source/transfer checks are real.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_native_remote import RemoteFixture  # noqa: E402


class NativeRemoteDispatchTests(unittest.TestCase):
    def test_fixed_dispatch_runs_owned_seccomp_child_and_retains_historical_proof(self):
        self.assertEqual(sys.platform, "linux", "required Linux dispatch cannot be skipped")
        case = RemoteFixture("runTest")
        case.setUp()
        try:
            library = next(
                (
                    p.resolve()
                    for p in (
                        Path("/lib/x86_64-linux-gnu/libseccomp.so.2"),
                        Path("/usr/lib/x86_64-linux-gnu/libseccomp.so.2"),
                    )
                    if p.exists()
                ),
                None,
            )
            self.assertIsNotNone(library)
            target = case.root / "libseccomp.so.2"
            shutil.copyfile(library, target)
            target.chmod(0o600)
            cap = NativeBinary(
                str(target), target.stat().st_size, hashlib.sha256(target.read_bytes()).hexdigest()
            )
            current = case.selected
            hardware = current.hardware
            subject = native_subject(
                current.interpreter,
                cap,
                current.runtime,
                current.sdk,
                current.base,
                current.design,
                hardware,
            )
            case.selected = replace(current, seccomp=cap, reviewed_subject_sha256=subject)
            # A trusted synthetic harness replaces only the SDK computation child.
            # It uses the real owned process, framing, descriptor closure and filter.
            # No production JSON switch or fake-runtime proof API is introduced.
            code = r"""
import errno,hashlib,json,os,socket,sys,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
from cineforge.native_speech_process import pack,unpack
from cineforge.native_speech_integrity import NativeBinary,digest
from cineforge.native_speech_network import deny_network
from cineforge.native_speech_pins import SDK_FILES,BASE_FILES,DESIGN_FILES
from test_native_speech import response
header,reference=unpack(sys.stdin.buffer.read())
cap=NativeBinary(**header['selection']['seccomp'])
deny_network(cap,header['deadline'])
for family in (socket.AF_INET,socket.AF_INET6,socket.AF_PACKET):
 try:socket.socket(family);raise AssertionError('network escape')
 except OSError as error:assert error.errno==errno.EPERM
assert 'HF_TOKEN' not in os.environ and 'PYTHONPATH' not in os.environ
assert 'LD_PRELOAD' not in os.environ and 'HTTP_PROXY' not in os.environ
assert not any(name.startswith(('torch','qwen_tts','numpy')) for name in sys.modules)
metadata,wave=response(header)
for name,pins in [('sdk_content_sha256',SDK_FILES),
 ('runtime_content_sha256',header['selection']['runtime']['files']),
 ('model_content_sha256',BASE_FILES if header['operation']=='base' else DESIGN_FILES)]:
 metadata[name]=digest(sorted((p[0],p[3]) for p in pins))
sys.stdout.buffer.write(pack(metadata,wave));sys.stdout.buffer.flush()
"""
            package = str(Path(__file__).resolve().parents[1] / "src")
            fixtures = str(Path(__file__).resolve().parents[1] / "tests")
            pids = []

            def actual_child(command, *, payload, env, cwd, deadline):
                nr_check(deadline)
                result = run_child(
                    (sys.executable, "-I", "-B", "-S", "-c", code, package, fixtures),
                    payload=payload,
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                )
                pids.append(result.process_id)
                return result

            with patch("cineforge.native_speech.run_child", actual_child):
                case.prepare()
                payload = case.bridge.preview(case.design)
            proof = case.bridge.proof_for(case.design, payload)
            self.assertEqual(pids, [proof.owned_process_id])
            with self.assertRaises(ProcessLookupError):
                os.kill(proof.owned_process_id, 0)
            case.worker.plan = None
            self.assertEqual(case.bridge.proof_for(case.design, payload), proof)
            self.assertEqual(len(case.bridge._proofs), 1)
            print(
                "NATIVE_REMOTE_SYNTHETIC_EVIDENCE="
                + str(
                    dict(
                        worker=proof.worker_id,
                        attempt=proof.attempt,
                        selection=proof.selection_sha256,
                        payload=proof.payload_sha256,
                        pid_reaped=True,
                        network_denied=True,
                        no_sdk_import=True,
                        firstparty_package=proof.package_sha256,
                    )
                )
            )
        finally:
            case.doCleanups()

    def test_absolute_transport_entry_refuses_before_owned_spawn(self):
        from cineforge.worker_bootstrap import BootstrapError, bounded_process

        with patch("cineforge.worker_bootstrap.subprocess.Popen") as spawn:
            with self.assertRaises(BootstrapError):
                bounded_process(
                    [sys.executable, "-c", "raise AssertionError()"],
                    timeout=10,
                    absolute_deadline=time.monotonic() - 1,
                )
        spawn.assert_not_called()
