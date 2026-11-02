"""Guarded remote protocol fixtures: no SSH daemon, network, SDK or model bodies."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_conditioning import runtime_fixture
from test_native_speech import response, selection
from test_runpod import spec as legacy_spec
from test_s2v import context
from test_speech import adapter, fixture

from cineforge import native_remote_bootstrap as bootstrap
from cineforge import native_remote_worker as remote
from cineforge import s2v_profile as profile
from cineforge import worker_bootstrap
from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_remote_package import native_package
from cineforge.native_remote_plan import (
    nr_encode,
    nr_json,
    nr_recipe,
    nr_recipe_subject,
)
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.native_speech_integrity import NativeSpeechError, digest, native_subject
from cineforge.native_speech_pins import BASE_FILES, DESIGN_FILES, RUNTIME_DISTRIBUTIONS, SDK_FILES
from cineforge.native_speech_process import OwnedChildResult, pack, unpack
from cineforge.runpod import RunpodPodSpec
from cineforge.speech import SpeechArtifact, SpeechError
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source
from cineforge.worker_guard import WorkerGuardPlan, worker_guard_source_sha256


def recipe():
    wheels = [
        dict(
            name=n,
            version=v,
            filename=n.replace("-", "_") + "-" + v + "-py3-none-any.whl",
            size_bytes=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
            url=None
            if n == "sox"
            else "https://files.pythonhosted.org/packages/a/"
            + n.replace("-", "_")
            + "-"
            + v
            + "-py3-none-any.whl",
        )
        for n, v in RUNTIME_DISTRIBUTIONS
    ]
    wheels.sort(key=lambda w: w["name"])
    return dict(
        version=1,
        wheels=wheels,
        reviewed_subject_sha256=nr_recipe_subject(wheels),
        review_evidence_sha256="a" * 64,
    )


class RemoteFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-remote-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.root.chmod(0o700)
        self.gate, _ = context(self.root)
        self.key = self.root / "key"
        self.key.write_bytes(b"not a credential")
        self.key.chmod(0o600)
        self.spec = RunpodPodSpec(
            "synthetic-remote",
            profile.IMAGE,
            profile.GPU_ID,
            "EU-RO-1",
            legacy_spec().ssh_public_key,
            max_hourly_usd=1.25,
            min_ram_per_gpu_gb=64,
            disk_gb=150,
        )
        self.plan = WorkerGuardPlan(
            "synthetic-worker", self.spec.session_id, "a" * 32, time.time() + 1000, 30
        )
        self.remaining = 900
        self.budget = SimpleNamespace(
            approval=self.gate.service.approval, remaining_seconds=lambda _: self.remaining
        )
        self.worker = SSHWorker(
            self.spec,
            self.gate.service,
            self.budget,
            None,
            self.key,
            self.root / "ssh",
            allow_live=True,
        )
        self.addCleanup(self.worker.close)
        self.worker.plan, self.worker._key = self.plan, "owned"
        self.worker.runtime_evidence = dict(runtime_fixture(), compute_precision="bf16")
        self.worker.check = lambda: None
        self.worker._request = self.rpc
        self.selected = selection(self.root)
        self.requests = []
        self.deadlines = []
        self.lost = None
        self.mutate = None
        self.child_count = 0
        self.remote = self.root / "remote"
        self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
        package, pins = native_package()
        patches = [
            patch.object(bootstrap, "NR_PACKAGE_SHA256", hashlib.sha256(package).hexdigest()),
            patch.object(bootstrap, "NR_PACKAGE_FILES", pins),
            patch.object(bootstrap, "nr_root", self.remote_root),
            patch.object(bootstrap, "nr_preflight", self.preflight),
            patch.object(remote, "_setup", self.setup),
            patch("cineforge.native_speech.run_child", self.child),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        args, _, _ = fixture()
        self.arguments = args
        self.request = adapter(args).translate(utterance_id="en-line", **args)
        self.reference = inspect_pcm16_wav(args["reference_payload"])
        self.design = OriginalVoiceDesign(
            "synthetic private text", "English", "original fictional timbre", "a" * 64
        )

    def remote_root(self, session, *, create=False):
        self.assertEqual(session, self.plan.session_id)
        if create:
            self.remote.mkdir(mode=0o700)
        bootstrap.nr_private(self.remote, directory=True)
        return self.remote

    def preflight(self, deadline):
        return dict(
            interpreter=asdict(self.selected.interpreter),
            seccomp=asdict(self.selected.seccomp),
            abi="cpython311-linux-x86_64-glibc2.28",
        )

    def setup(self, root, deadline):
        bootstrap.nr_write(root / "setup-consumed", b"consumed")
        result = {
            k: asdict(self.selected)[k]
            for k in ("interpreter", "seccomp", "runtime", "sdk", "base", "design")
        }
        result["recipe_sha256"] = nr_json(bootstrap.nr_read(root / "recipe.blob"))[
            "reviewed_subject_sha256"
        ]
        raw = nr_encode(result)
        bootstrap.nr_write(root / "installed.json", raw)
        return remote._descriptor(raw)

    def child(self, command, *, payload, env, cwd, deadline):
        self.child_count += 1
        header, _ = unpack(payload)
        meta, wave = response(header)
        for name, pins in [
            ("sdk_content_sha256", SDK_FILES),
            ("runtime_content_sha256", self.selected.runtime.files),
            ("model_content_sha256", BASE_FILES if header["operation"] == "base" else DESIGN_FILES),
        ]:
            meta[name] = digest(sorted((p[0], p[3]) for p in pins))
        return OwnedChildResult(pack(meta, wave), 12345)

    def rpc(self, request, timeout, **options):
        self.requests.append(copy.deepcopy(request))
        self.deadlines.append(options["absolute_deadline"])
        self.assertGreater(timeout, 0)
        reply = bootstrap.nr_dispatch(
            dict(request, profile=self.worker._profile, guard_sha256=worker_guard_source_sha256()),
            time.monotonic() + timeout,
        )
        if self.mutate:
            reply = self.mutate(request, reply)
        if self.lost == request["operation"]:
            raise TimeoutError("SYNTHETIC-PRIVATE-CANARY")
        return reply

    def prepare(self):
        installed = self.bridge.stage(recipe(), b"x")
        self.assertEqual(installed["recipe_sha256"], recipe()["reviewed_subject_sha256"])
        self.bridge.select(self.selected)

    def static_failure(self, call):
        with self.assertRaises(NativeSpeechError) as caught:
            call()
        self.assertNotIn(
            "SYNTHETIC-PRIVATE-CANARY", "".join(traceback.format_exception(caught.exception))
        )


class RemoteTests(RemoteFixture):
    def test_bootstrap_checks_owned_guard_and_profile_before_native_dispatch(self):
        guardian = self.root / "guardian"
        guardian.mkdir(mode=0o700)
        source = b"synthetic guard source, never executed"
        for name, raw in {
            "plan.json": nr_encode(self.plan.to_dict()),
            "profile.json": nr_encode(self.worker._profile),
            "guard-pid.json": nr_encode({"pid": 12345}),
            "guard.py": source,
        }.items():
            bootstrap.nr_write(guardian / name, raw)
        request = dict(
            operation="native_preflight",
            timeout_seconds=5,
            allow_live=True,
            plan=self.plan.to_dict(),
            profile=self.worker._profile,
            guard_sha256=hashlib.sha256(source).hexdigest(),
        )
        with (
            patch.object(worker_bootstrap, "worker_root", return_value=guardian),
            patch.object(worker_bootstrap, "guard_alive"),
            patch.object(
                bootstrap, "nr_dispatch", return_value={"status": "synthetic"}
            ) as dispatch,
        ):
            self.assertEqual(worker_bootstrap.dispatch(request), {"status": "synthetic"})
            self.assertEqual(dispatch.call_count, 1)
            with self.assertRaises(worker_bootstrap.BootstrapError):
                worker_bootstrap.dispatch(dict(request, guard_sha256="b" * 64))
            with self.assertRaises(worker_bootstrap.BootstrapError):
                worker_bootstrap.dispatch(dict(request, profile=None))
            self.assertEqual(dispatch.call_count, 1)

    def test_actual_transfer_binds_package_selection_attempt_and_history(self):
        self.prepare()
        designed = self.bridge.preview(self.design)
        first = self.bridge.proof_for(self.design, designed)
        payload = self.bridge.synthesize(self.request, self.reference)
        proof = self.bridge.proof_for(self.request, payload)
        self.assertNotEqual(first.attempt, proof.attempt)
        self.assertEqual(proof.worker_id, self.plan.pod_id)
        self.assertEqual(self.child_count, 2)
        self.assertEqual(len(set(self.deadlines)), 1)
        self.assertEqual(self.bridge.receipt_for(self.request, payload)["owned_process_id"], 12345)
        self.assertEqual(self.bridge._selection, self.selected)
        self.assertTrue((self.remote / (proof.attempt + ".consumed")).exists())
        self.worker.check = lambda: (_ for _ in ()).throw(WorkerError("worker deleted"))
        self.worker.plan = None
        self.assertEqual(self.bridge.proof_for(self.design, designed), first)
        self.assertEqual(self.bridge.proof_for(self.request, payload), proof)
        speech = SpeechArtifact(
            replace(self.gate.speech.pin, sha256=proof.payload_sha256),
            self.request,
            payload,
            "unverified_injected_runtime",
            24000,
            2400,
        )
        s2v = self.bridge.verify_speech(speech, self.gate.service)
        self.assertEqual(s2v.native_receipt_sha256, proof.native_receipt_sha256)
        self.static_failure(lambda: self.bridge.preview(self.design))
        self.assertEqual(self.child_count, 2)

    def test_lost_ack_consumed_no_second_setup_or_invoke(self):
        for operation in (
            "native_preflight",
            "native_begin",
            "native_chunk",
            "native_finish",
            "native_setup",
            "native_fetch",
            "native_select",
            "native_invoke",
        ):
            with self.subTest(operation=operation):
                # Each separate fixture owns its own isolated session/root.
                case = RemoteFixture("runTest")
                case.setUp()
                try:
                    if operation in ("native_select", "native_invoke"):
                        case.bridge.stage(recipe(), b"x")
                    if operation == "native_invoke":
                        case.bridge.select(case.selected)
                    case.lost = operation
                    action = (
                        (lambda: case.bridge.preview(case.design))
                        if operation == "native_invoke"
                        else (lambda: case.bridge.select(case.selected))
                        if operation == "native_select"
                        else (lambda: case.bridge.stage(recipe(), b"x"))
                    )
                    case.static_failure(action)
                    calls = len(case.requests)
                    case.static_failure(action)
                    self.assertEqual(len(case.requests), calls)
                    self.assertLessEqual(case.child_count, 1)
                finally:
                    case.doCleanups()

    def test_forged_serialized_receipt_or_artifact_never_enters_store(self):
        self.prepare()
        self.static_failure(lambda: self.bridge.proof_for(self.design, b"forged"))

        def tamper(request, reply):
            if request["operation"] == "native_invoke":
                token = request["token"]
                header, payload = unpack(bootstrap.nr_read(self.remote / (token + ".result")))
                header["binding"]["worker"] = "foreign"
                raw = pack(header, payload)
                (self.remote / (token + ".result")).write_bytes(raw)
                return remote._descriptor(raw)
            return reply

        self.mutate = tamper
        self.static_failure(lambda: self.bridge.preview(self.design))
        self.assertEqual(self.bridge._proofs, {})
        self.assertTrue(self.bridge._unresolved)
        self.assertEqual(self.child_count, 1)

    def test_source_guard_worker_scope_and_reserve_are_fresh_for_every_call(self):
        self.prepare()
        payload = self.bridge.preview(self.design)
        proof = self.bridge.proof_for(self.design, payload)
        self.remaining = 120
        self.static_failure(lambda: self.bridge.synthesize(self.request, self.reference))
        self.assertEqual(self.child_count, 1)
        self.assertEqual(self.bridge.proof_for(self.design, payload), proof)

    def test_late_entry_and_post_fetch_delay_publish_no_proof(self):
        self.prepare()
        self.bridge._deadline = time.monotonic() - 1
        before = len(self.requests)
        self.static_failure(lambda: self.bridge.preview(self.design))
        self.assertEqual(len(self.requests), before)
        self.assertEqual(self.child_count, 0)

    def test_remote_saved_cap_is_not_renewed_and_delayed_delivery_refuses(self):
        self.bridge.stage(recipe(), b"x")
        cap = self.remote / "local-cap.json"
        value = nr_json(bootstrap.nr_read(cap))
        value["deadline"] = time.monotonic() - 1
        cap.write_bytes(nr_encode(value))
        self.static_failure(lambda: self.bridge.select(self.selected))
        self.assertEqual(self.child_count, 0)

    def test_tampered_package_bytes_or_unreviewed_selection_refuse(self):
        self.bridge.stage(recipe(), b"x")
        bad = replace(self.selected.runtime, root="/untrusted")
        hardware = self.selected.hardware
        subject = native_subject(
            self.selected.interpreter,
            self.selected.seccomp,
            bad,
            self.selected.sdk,
            self.selected.base,
            self.selected.design,
            hardware,
        )
        changed = replace(self.selected, runtime=bad, reviewed_subject_sha256=subject)
        with self.assertRaises(NativeSpeechError):
            self.bridge.select(changed)
        self.assertEqual(self.child_count, 0)
        target = self.remote / "firstparty" / "cineforge" / "native_remote_worker.py"
        target.write_bytes(b'raise RuntimeError("SYNTHETIC-PRIVATE-CANARY")')
        self.static_failure(lambda: self.bridge.select(self.selected))
        self.assertEqual(self.child_count, 0)

    def test_original_parent_cap_includes_pre_stage_scheduler_delay(self):
        cap = time.monotonic() + 10
        bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True, absolute_deadline=cap)
        with patch("cineforge.native_remote.time.monotonic", return_value=cap + 1):
            self.static_failure(lambda: bridge.stage(recipe(), b"x"))
        self.assertEqual(self.requests, [])
        self.assertFalse(self.remote.exists())

    def test_post_fetch_lateness_cannot_publish_an_owned_proof(self):
        self.prepare()

        def delayed(request, reply):
            if request["operation"] == "native_fetch":
                self.bridge._deadline = time.monotonic() - 1
            return reply

        self.mutate = delayed
        self.static_failure(lambda: self.bridge.preview(self.design))
        self.assertEqual(self.child_count, 1)
        self.assertEqual(self.bridge._proofs, {})
        self.assertTrue(self.bridge._unresolved)

    def test_replaced_source_refuses_without_rpc_and_preserves_prior_facts(self):
        self.prepare()
        payload = self.bridge.preview(self.design)
        proof = self.bridge.proof_for(self.design, payload)
        count = len(self.requests)
        self.worker._source += b"\n# changed after admission\n"
        self.static_failure(lambda: self.bridge.synthesize(self.request, self.reference))
        self.assertEqual(len(self.requests), count)
        self.assertEqual(self.bridge.proof_for(self.design, payload), proof)

    def test_recipe_design_english_chinese_are_consumed_once_each(self):
        self.prepare()
        self.bridge.preview(self.design)
        self.bridge.synthesize(self.request, self.reference)
        chinese = replace(self.request, language="Chinese", text="\u5408\u6210\u6d4b\u8bd5")
        output = self.bridge.synthesize(chinese, self.reference)
        self.assertEqual(self.bridge.proof_for(chinese, output).operation, "base")
        before = len(self.requests)
        self.static_failure(lambda: self.bridge.synthesize(chinese, self.reference))
        self.assertEqual(len(self.requests), before)
        self.assertEqual(self.child_count, 3)

    def test_clock_delivery_bound_and_duplicate_chunk_cannot_expand_remote_cap(self):
        self.bridge.stage(recipe(), b"x")
        original = copy.deepcopy(self.requests[-1])
        original.update(profile=self.worker._profile, guard_sha256=worker_guard_source_sha256())
        original["native_sent_unix"] = time.time() - 6
        with self.assertRaises(ValueError):
            bootstrap.nr_dispatch(original, time.monotonic() + 1200)
        # A chunk ACK cannot be replayed at its previous offset.
        chunk = next(copy.deepcopy(r) for r in self.requests if r["operation"] == "native_chunk")
        chunk.update(
            profile=self.worker._profile,
            guard_sha256=worker_guard_source_sha256(),
            native_sent_unix=time.time(),
        )
        with self.assertRaises(ValueError):
            bootstrap.nr_dispatch(chunk, time.monotonic() + 1200)
        self.assertTrue((self.remote / "failed").exists())
        self.assertEqual(self.child_count, 0)

    def test_concurrent_controller_and_worker_operations_refuse_without_waiting(self):
        self.prepare()
        before = len(self.requests)
        self.bridge._lock.acquire()
        try:
            self.static_failure(lambda: self.bridge.preview(self.design))
        finally:
            self.bridge._lock.release()
        self.assertEqual(len(self.requests), before)
        self.assertEqual(self.bridge._attempts, 0)
        with bootstrap.nr_lock(self.remote):
            with self.assertRaises(BlockingIOError):
                with bootstrap.nr_lock(self.remote):
                    self.fail("overlapping native operation admitted")
        self.assertTrue(self.bridge.preview(self.design))
        self.assertEqual(self.child_count, 1)

    def test_malformed_oversize_truncated_and_wrong_digest_replies_never_prove_native(self):
        for fault in ("oversize", "base64", "truncated", "offset-bool", "digest"):
            with self.subTest(fault=fault):
                case = RemoteFixture("runTest")
                case.setUp()
                try:

                    def tamper(request, reply):
                        reply = copy.deepcopy(reply)
                        if request["operation"] == "native_setup":
                            if fault == "oversize":
                                reply["size_bytes"] = 16 * 1024**2 + 1
                            elif fault == "digest":
                                reply["sha256"] = "b" * 64
                        if request["operation"] == "native_fetch":
                            if fault == "base64":
                                reply["data"] = "not base64!"
                            elif fault == "truncated":
                                reply["data"] = base64.b64encode(b"truncated").decode()
                            elif fault == "offset-bool":
                                reply["offset"] = False
                        return reply

                    case.mutate = tamper
                    case.static_failure(lambda: case.bridge.stage(recipe(), b"x"))
                    self.assertTrue(case.bridge._unresolved)
                    self.assertEqual(case.bridge._proofs, {})
                    self.assertEqual(case.child_count, 0)
                finally:
                    case.doCleanups()

    def test_native_cleanup_uncertainty_is_sticky_and_owns_no_success_proof(self):
        self.prepare()
        error = NativeSpeechError(
            "SYNTHETIC-PRIVATE-CANARY", cleanup_uncertain=True, owned_process_id=12345
        )
        with patch("cineforge.native_speech.run_child", side_effect=error) as child:
            self.static_failure(lambda: self.bridge.preview(self.design))
            count = len(self.requests)
            self.static_failure(lambda: self.bridge.preview(self.design))
        self.assertEqual(child.call_count, 1)
        self.assertEqual(len(self.requests), count)
        self.assertTrue((self.remote / (self.bridge.last_attempt + ".consumed")).exists())
        self.assertTrue((self.remote / "failed").exists())
        self.assertEqual(self.bridge._proofs, {})

    def test_current_rights_revocation_during_remote_invoke_prevents_artifact(self):
        self.prepare()

        def revoke_after_owned_child(*args, **kwargs):
            result = self.child(*args, **kwargs)
            binding = self.arguments["reference"].permission
            self.arguments["ledger"].revoke(binding.rights, reason="synthetic current revocation")
            return result

        speech = adapter(self.arguments, runtime=self.bridge)
        with patch("cineforge.native_speech.run_child", revoke_after_owned_child):
            with self.assertRaises(SpeechError):
                speech.synthesize(
                    utterance_id="en-line", output_id="speech", output_revision=1, **self.arguments
                )
        self.assertEqual(self.child_count, 1)
        # A completed native fact is historical, never a cached permission grant.
        self.assertEqual(len(self.bridge._proofs), 1)


class RemoteDeclarations(unittest.TestCase):
    def test_standalone_bundle_imports_verified_staged_package_without_ambient_cineforge(self):
        package, _ = native_package()
        source = worker_source().rsplit(b'\nif __name__ == "__main__":', 1)[0]
        source += b"""
root=Path(sys.argv[1])
package=nr_package(root,install=True)
sys.path.insert(0,str(package))
implementation=importlib.import_module("cineforge.native_remote_worker")
assert Path(implementation.__file__).parent==package/"cineforge"
assert not any(n.startswith(("qwen_tts","torch","numpy","transformers")) for n in sys.modules)
nr_write(root/"installed.json",b"synthetic fixed-range fetch; not native evidence")
reply=implementation.remote_operation({"operation":"native_fetch","token":"installed","offset":0},
 root,time.monotonic()+10,{})
assert base64.b64decode(reply["data"])==nr_read(root/"installed.json")
print("verified first-party package imported; no ambient Cineforge or SDK")
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            bootstrap.nr_write(root / "package.blob", package)
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", source.decode(), str(root)],
                capture_output=True,
                timeout=15,
                cwd=root,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"no ambient Cineforge or SDK", result.stdout)

    def test_recipe_requires_complete_exact_runtime_closure_and_fixed_origins(self):
        original = recipe()
        self.assertEqual(len(nr_recipe(original)["wheels"]), 108)
        changes = [
            lambda v: v["wheels"].pop(),
            lambda v: v["wheels"][0].update(url="https://evil.invalid/x"),
            lambda v: v["wheels"][0].update(sha256="0" * 64),
            lambda v: v.update(command="arbitrary"),
            lambda v: v["wheels"][0].update(filename="../evil.whl"),
            lambda v: v["wheels"].append(copy.deepcopy(v["wheels"][0])),
            lambda v: v["wheels"][0].update(version="unreviewed"),
        ]
        for change in changes:
            value = copy.deepcopy(original)
            change(value)
            with self.assertRaises(ValueError):
                nr_recipe(value)
        with self.assertRaises(ValueError):
            nr_json(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            nr_json(b'{"a":NaN}')

    def test_bundle_closes_entire_package_and_stays_bounded(self):
        package, pins = native_package()
        files = nr_json(package)
        self.assertEqual(set(files), {p.name for p in Path("src/cineforge").glob("*.py")})
        for name, size, sha in pins:
            raw = base64.b64decode(files[name], validate=True)
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (size, sha))
        source = worker_source()
        self.assertLess(len(source), MAX_SOURCE_BYTES)
        self.assertIn(hashlib.sha256(package).hexdigest().encode(), source)
        self.assertEqual(source, worker_source())

    def test_fresh_import_and_opt_in_are_side_effect_free(self):
        script = """
import sys
sys.path.insert(0,sys.argv[1])
def guard(event,args):
 if event in ('subprocess.Popen','socket.connect','ctypes.dlopen'):raise AssertionError(event)
sys.addaudithook(guard)
from cineforge.native_remote import RemoteNativeSpeechRuntime
try:RemoteNativeSpeechRuntime(object(),allow_native=True);raise AssertionError('fake worker')
except ValueError:pass
assert not any(n.startswith(('qwen_tts','torch','numpy','transformers')) for n in sys.modules)
"""
        done = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", script, str(Path("src").resolve())],
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_chunk_hardlink_symlink_and_expired_cap_refusals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "private"
            bootstrap.nr_write(path, b"x")
            self.assertEqual(bootstrap.nr_read(path), b"x")
            os.link(path, root / "alias")
            with self.assertRaises(ValueError):
                bootstrap.nr_read(path)
            os.unlink(root / "alias")
            (root / "link").symlink_to(path)
            with self.assertRaises(ValueError):
                bootstrap.nr_read(root / "link")
            with self.assertRaises(ValueError):
                bootstrap.nr_check(time.monotonic() - 1)

    def test_fixed_wheel_record_extraction_checks_paths_hashes_and_closure(self):
        def wheel(path, *, unsafe=False, bad_hash=False):
            name = "../escape.py" if unsafe else "example/__init__.py"
            payload = b"not executed"
            metadata = b"Name: example\nVersion: 1\n"
            bodies = {name: payload, "example-1.dist-info/METADATA": metadata}
            rows = []
            for filename, raw in bodies.items():
                sha = hashlib.sha256(b"wrong" if bad_hash else raw).digest()
                rows.append(
                    [
                        filename,
                        "sha256=" + base64.urlsafe_b64encode(sha).decode().rstrip("="),
                        str(len(raw)),
                    ]
                )
            record = "example-1.dist-info/RECORD"
            rows.append([record, "", ""])
            output = io.StringIO()
            csv.writer(output).writerows(rows)
            bodies[record] = output.getvalue().encode()
            with zipfile.ZipFile(path, "w") as archive:
                for filename, raw in bodies.items():
                    archive.writestr(filename, raw)
            path.chmod(0o600)

        for fault in ("okay", "unsafe", "hash"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "example.whl"
                wheel(archive, unsafe=fault == "unsafe", bad_hash=fault == "hash")
                target = root / "runtime"
                target.mkdir(mode=0o700)
                manifest = {}

                def call():
                    remote._install_wheel(
                        archive,
                        target,
                        dict(name="example", version="1"),
                        manifest,
                        time.monotonic() + 10,
                    )

                if fault == "okay":
                    call()
                    self.assertEqual(len(manifest), 3)
                    self.assertEqual((target / "example/__init__.py").read_bytes(), b"not executed")
                else:
                    with self.assertRaises(ValueError):
                        call()
                    self.assertFalse((root / "escape.py").exists())
