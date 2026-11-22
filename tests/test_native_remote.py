"""Guarded remote protocol fixtures: no SSH daemon, network, SDK or model bodies."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import os
import stat
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
from test_speech import adapter, fixture, wav_bytes

from cineforge import native_remote_bootstrap as bootstrap
from cineforge import native_remote_worker as remote
from cineforge import s2v_profile as profile
from cineforge import worker_bootstrap
from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_remote_package import native_package
from cineforge.native_remote_plan import (
    NR_CHUNK,
    nr_encode,
    nr_json,
    nr_recipe,
    nr_recipe_subject,
)
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.native_speech_integrity import (
    NativeHardware,
    NativeSpeechError,
    NativeTree,
    digest,
    native_subject,
    verify_tree,
)
from cineforge.native_speech_pins import BASE_FILES, DESIGN_FILES, RUNTIME_DISTRIBUTIONS, SDK_FILES
from cineforge.native_speech_process import OwnedChildResult, pack, unpack
from cineforge.runpod import RunpodPodSpec
from cineforge.s2v_service import LongSessionBudgetApproval, LongSessionServiceContext
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
    gpu_id = profile.GPU_ID

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-remote-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.root.chmod(0o700)
        self.gate, _ = context(self.root, gpu_id=self.gpu_id)
        self.key = self.root / "key"
        self.key.write_bytes(b"not a credential")
        self.key.chmod(0o600)
        self.spec = RunpodPodSpec(
            "synthetic-remote",
            profile.IMAGE,
            self.gpu_id,
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
        if self.gpu_id != profile.GPU_ID:
            self.worker.runtime_evidence.update(python_version="3.11.14", gpu_raw_name=self.gpu_id,
                gpu_memory_mib=46000, nvml_total_mib=46068, nvml_reserved_mib=68,
                capability=list(profile.worker(self.gpu_id).reference_runtime.capability), arch_list=["sm_86"])
        expected = self.worker.runtime_evidence
        self.selected = selection(self.root, hardware=NativeHardware("synthetic-worker", self.gpu_id,
            expected["gpu_memory_mib"], tuple(expected["capability"])))
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

    def preflight(self, deadline, *, profile_id=profile.PROFILE_ID):
        self.assertEqual(profile_id, self.gate.service.worker_profile["id"])
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
    def test_long_session_bounded_repeated_english_and_one_design(self):
        short = self.gate.service.approval
        approval = LongSessionBudgetApproval.for_scope(short.journal_directory, short.region,
            short.operator_receipt, phase_id="long-native-test", gpu_id=short.gpu_id)
        self.worker.selection = LongSessionServiceContext(approval)
        self.worker._profile = self.worker.selection.worker_profile
        self.budget.approval = approval
        self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True,
            timeout_seconds=50400, max_operations=5)
        self.prepare()
        self.bridge.preview(self.design)
        for index in range(4):
            request = replace(self.request, text="Segment " + str(index))
            self.bridge.synthesize(request, self.reference)
        self.assertEqual(self.child_count, 5)
        self.static_failure(lambda: self.bridge.synthesize(self.request, self.reference))
        self.assertEqual(self.child_count, 5)
        binding = nr_json(bootstrap.nr_read(self.remote / "binding.json"))
        self.assertEqual(binding["operation_limit"], 5)
        # The worker also enforces input admission if the controller is bypassed.
        with self.assertRaisesRegex(ValueError, "attempt capacity"):
            self.rpc(dict(self.requests[-1], operation="native_begin", role="input",
                token="f" * 32, size_bytes=1, sha256=hashlib.sha256(b"x").hexdigest()),
                30, absolute_deadline=time.monotonic() + 30)

    def test_long_phase_caps_survive_short_rpc_without_renewal(self):
        short = self.gate.service.approval
        approval = LongSessionBudgetApproval.for_scope(short.journal_directory, short.region,
            short.operator_receipt, phase_id="long-phase-test", gpu_id=short.gpu_id)
        self.worker.selection = LongSessionServiceContext(approval)
        self.worker._profile = self.worker.selection.worker_profile
        self.budget.approval = approval
        self.remaining = 50000
        self.plan = replace(self.plan, stop_at_unix=time.time() + 50400)
        self.worker.plan = self.plan
        self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True,
            timeout_seconds=50400, max_operations=6)
        entered = time.monotonic()
        self.prepare()
        native_cap = nr_json(bootstrap.nr_read(self.remote / "local-cap.json"))["deadline"]
        self.assertGreater(native_cap - entered, 49000)
        self.bridge._upload("comfy", b"synthetic closure", comfy_expires_unix=self.bridge._expires_unix)
        comfy_cap = nr_json(bootstrap.nr_read(self.remote / "comfy-local-cap.json"))["deadline"]
        self.assertGreater(comfy_cap - entered, 49000)
        self.bridge.preview(self.design)
        self.assertEqual(nr_json(bootstrap.nr_read(self.remote / "local-cap.json"))["deadline"], native_cap)
        self.assertEqual(nr_json(bootstrap.nr_read(self.remote / "comfy-local-cap.json"))["deadline"], comfy_cap)

    def test_timeout_accepts_approved_lifetime_and_rejects_above_ceiling(self):
        bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True, timeout_seconds=7200)
        self.assertEqual(bridge._timeout, 7200)
        with self.assertRaises(NativeSpeechError):
            RemoteNativeSpeechRuntime(self.worker, allow_native=True, timeout_seconds=7201)

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

    def test_large_manifest_roundtrip_uses_33_bounded_data_frames_each_direction(self):
        self.bridge.stage(recipe(), b"x")
        # Synthetic canonical JSON matches the observed manifest size, without private data.
        raw = nr_encode({"files": "x" * (4211692 - len(nr_encode({"files": ""})))})
        self.assertEqual(len(raw), 4211692)
        (self.remote / "installed.json").unlink()
        bootstrap.nr_write(self.remote / "installed.json", raw)
        self.requests.clear()
        frames, replies = [], []
        original = self.worker._request

        def framed(request, timeout, **options):
            closed = dict(request, allow_live=True, timeout_seconds=timeout,
                          guard_sha256=worker_guard_source_sha256(), profile=self.worker._profile)
            encoded = nr_encode(closed)
            frames.append(encoded)
            self.assertLessEqual(len(encoded), 256 * 1024)
            reply = original(nr_json(encoded), timeout, **options)
            encoded_reply = nr_encode(reply)
            replies.append(encoded_reply)
            self.assertLessEqual(len(encoded_reply), 256 * 1024)
            return nr_json(encoded_reply)

        self.worker._request = framed
        descriptor = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.assertEqual(self.bridge._fetch("installed", descriptor), raw)
        self.bridge._upload("selection", raw)
        chunks = [r for r in self.requests if r["operation"] == "native_chunk"]
        fetches = [r for r in self.requests if r["operation"] == "native_fetch"]
        self.assertEqual(len(chunks), 33)
        self.assertEqual(len(fetches), 33)
        self.assertEqual([r["offset"] for r in chunks], list(range(0, len(raw), NR_CHUNK)))
        self.assertEqual([r["offset"] for r in fetches], list(range(0, len(raw), NR_CHUNK)))
        self.assertEqual(bootstrap.nr_read(self.remote / "selection.blob"), raw)
        self.assertEqual(b"".join(base64.b64decode(r["data"], validate=True) for r in chunks), raw)
        self.assertGreater(max(map(len, frames)), 174764)
        self.assertGreater(max(map(len, replies)), 174764)
        self.assertEqual(self.child_count, 0)
        self.assertEqual(len(set(self.deadlines)), 1)

    def test_larger_chunks_still_refuse_oversize_reorder_and_wrong_raw_hash(self):
        for fault in ("oversize", "reorder", "hash"):
            with self.subTest(fault=fault):
                case = RemoteFixture("runTest")
                case.setUp()
                try:
                    case.bridge.stage(recipe(), b"x")
                    data = b"x" * NR_CHUNK
                    common = {"role": "selection", "token": "a" * 32}
                    case.bridge._call("native_begin", **common, size_bytes=len(data)+1,
                                      sha256=hashlib.sha256(data+b"x").hexdigest())
                    if fault == "hash":
                        case.bridge._call("native_chunk", **common, offset=0,
                                          data=base64.b64encode(data).decode())
                        case.bridge._call("native_chunk", **common, offset=NR_CHUNK,
                                          data=base64.b64encode(b"y").decode())
                        case.static_failure(lambda *, case=case, common=common: case.bridge._call("native_finish", **common))
                    else:
                        payload = data+b"x" if fault == "oversize" else data
                        case.static_failure(lambda *, case=case, common=common, fault=fault, payload=payload: case.bridge._call("native_chunk", **common,
                            offset=0 if fault == "oversize" else NR_CHUNK,
                            data=base64.b64encode(payload).decode()))
                    self.assertTrue(case.bridge._unresolved)
                    self.assertFalse((case.remote / "selection.blob.done").exists())
                    self.assertEqual(case.child_count, 0)
                finally:
                    case.doCleanups()

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
                        (lambda *, case=case: case.bridge.preview(case.design))
                        if operation == "native_invoke"
                        else (lambda *, case=case: case.bridge.select(case.selected))
                        if operation == "native_select"
                        else (lambda *, case=case: case.bridge.stage(recipe(), b"x"))
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

                    def tamper(request, reply, *, fault=fault):
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
                    case.static_failure(lambda *, case=case: case.bridge.stage(recipe(), b"x"))
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
    def synthetic_distribution(self, path, bodies, *, name="sympy", version="1.14.0"):
        bodies = dict(bodies)
        prefix = name.replace("-", "_") + "-" + version + ".dist-info/"
        bodies[prefix + "METADATA"] = f"Name: {name}\nVersion: {version}\n".encode()
        record = prefix + "RECORD"
        rows = [
            [
                filename,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("="),
                str(len(raw)),
            ]
            for filename, raw in bodies.items()
        ]
        rows.append([record, "", ""])
        output = io.StringIO()
        csv.writer(output).writerows(rows)
        bodies[record] = output.getvalue().encode()
        with zipfile.ZipFile(path, "w") as archive:
            for filename, raw in bodies.items():
                archive.writestr(filename, raw)
        path.chmod(0o600)
        return bodies[record]

    def test_known_sympy_manpage_retains_original_record_path_and_full_subject_binding(self):
        member = "sympy-1.14.0.data/data/share/man/man1/isympy.1"
        body = b"synthetic inert manual; never imported or installed outside private tree"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "runtime"
            target.mkdir(mode=0o700)
            archive = root / "synthetic.whl"
            original_record = self.synthetic_distribution(
                archive, {"sympy/__init__.py": b"synthetic inert module", member: body}
            )
            manifest = {}
            remote._install_wheel(
                archive,
                target,
                {"name": "sympy", "version": "1.14.0"},
                manifest,
                time.monotonic() + 10,
            )
            self.assertEqual((target / member).read_bytes(), body)
            self.assertEqual(
                (target / "sympy-1.14.0.dist-info/RECORD").read_bytes(), original_record
            )
            self.assertEqual(manifest[member], (len(body), hashlib.sha256(body).hexdigest()))
            self.assertEqual((target / member).stat().st_mode & 0o777, 0o600)
            self.assertEqual((target / member).stat().st_nlink, 1)
            self.assertFalse((target / "share").exists())
            tree = NativeTree(
                str(target),
                tuple((p, size, "sha256", sha) for p, (size, sha) in sorted(manifest.items())),
            )
            self.assertEqual(
                verify_tree(tree, time.monotonic() + 10),
                digest(sorted((p, sha) for p, (_, sha) in manifest.items())),
            )
            picked = selection(root)
            subject = native_subject(
                picked.interpreter,
                picked.seccomp,
                tree,
                picked.sdk,
                picked.base,
                picked.design,
                picked.hardware,
            )
            changed = replace(
                tree,
                files=tuple(
                    (
                        p,
                        size,
                        alg,
                        hashlib.sha256(b"changed manual").hexdigest() if p == member else sha,
                    )
                    for p, size, alg, sha in tree.files
                ),
            )
            self.assertNotEqual(
                subject,
                native_subject(
                    picked.interpreter,
                    picked.seccomp,
                    changed,
                    picked.sdk,
                    picked.base,
                    picked.design,
                    picked.hardware,
                ),
            )
            (target / member).write_bytes(b"changed manual")
            with self.assertRaises(NativeSpeechError):
                verify_tree(tree, time.monotonic() + 10)

    def test_only_exact_pinned_script_bodies_remain_inert_at_original_hashed_paths(self):
        for name, version, member in (
            ("numba", "0.68.1", "numba-0.68.1.data/scripts/numba"),
            ("ruff", "0.17.3", "ruff-0.17.3.data/scripts/ruff"),
        ):
            with self.subTest(distribution=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                target = root / "runtime"
                target.mkdir(mode=0o700)
                archive = root / "synthetic.whl"
                body = b"synthetic inert script body; never executed"
                record = self.synthetic_distribution(
                    archive, {member: body}, name=name, version=version
                )
                # Match the actual ZIP permission shape while proving the
                # installer deliberately retains data with no executable bit.
                with zipfile.ZipFile(archive) as source:
                    bodies = {n: source.read(n) for n in source.namelist()}
                with zipfile.ZipFile(archive, "w") as output:
                    for path, raw in bodies.items():
                        info = zipfile.ZipInfo(path)
                        info.create_system = 3
                        info.external_attr = (stat.S_IFREG | 0o755) << 16
                        output.writestr(info, raw)
                manifest = {}
                remote._install_wheel(
                    archive,
                    target,
                    {"name": name, "version": version},
                    manifest,
                    time.monotonic() + 10,
                )
                self.assertEqual((target / member).read_bytes(), body)
                self.assertEqual((target / member).stat().st_mode & 0o777, 0o600)
                self.assertEqual((target / member).stat().st_nlink, 1)
                self.assertEqual(
                    (target / (name + "-" + version + ".dist-info/RECORD")).read_bytes(), record
                )
                self.assertFalse((target / "bin").exists())
                self.assertFalse((target / name).exists())
                declared = NativeTree(
                    str(target),
                    tuple((p, size, "sha256", sha) for p, (size, sha) in sorted(manifest.items())),
                )
                self.assertEqual(
                    verify_tree(declared, time.monotonic() + 10),
                    digest(sorted((p, sha) for p, (_, sha) in manifest.items())),
                )
                (target / member).write_bytes(b"tampered inert body")
                with self.assertRaises(NativeSpeechError):
                    verify_tree(declared, time.monotonic() + 10)
        for name, version, member in (
            ("numba", "0.68.1", "numba-0.68.1.data/scripts/other"),
            ("ruff", "0.17.3", "ruff-0.17.3.data/scripts/ruffx"),
            ("ruff", "0.17.2", "ruff-0.17.2.data/scripts/ruff"),
            ("sympy", "1.14.0", "sympy-1.14.0.data/scripts/isympy"),
        ):
            with self.assertRaises(ValueError):
                remote._destination(member, (name, version))

    def test_unreviewed_data_schemes_collisions_and_existing_paths_refuse(self):
        member = "sympy-1.14.0.data/data/share/man/man1/isympy.1"
        for invalid in (
            "sympy-1.14.0.data/scripts/isympy",
            "sympy-1.14.0.data/headers/anything.h",
            "sympy-1.14.0.data/data/etc/model.py",
            "sympy-1.14.0.data/data/share/man/man2/isympy.2",
            "sympy-1.14.0.data/data/share/man/man1/../isympy.1",
            "sympy-1.13.0.data/data/share/man/man1/isympy.1",
        ):
            with self.subTest(path=invalid):
                with self.assertRaises(ValueError):
                    remote._destination(invalid, ("sympy", "1.14.0"))
        with self.assertRaises(ValueError):
            remote._destination(member, ("example", "1"))
        for fault in ("collision", "existing"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                target = root / "runtime"
                target.mkdir(mode=0o700)
                archive = root / "synthetic.whl"
                bodies = {member: b"synthetic inert manual"}
                if fault == "collision":
                    bodies.update(
                        {
                            "sympy/__init__.py": b"same",
                            "sympy-1.14.0.data/purelib/sympy/__init__.py": b"same",
                        }
                    )
                else:
                    existing = remote._parent(target, member)
                    bootstrap.nr_write(existing, b"existing bytes must remain")
                self.synthetic_distribution(archive, bodies)
                with self.assertRaises((ValueError, FileExistsError)):
                    remote._install_wheel(
                        archive,
                        target,
                        {"name": "sympy", "version": "1.14.0"},
                        {},
                        time.monotonic() + 10,
                    )
                if fault == "existing":
                    self.assertEqual(existing.read_bytes(), b"existing bytes must remain")

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
            fixture_source = root / "standalone-fixture.py"
            fixture_source.write_bytes(source)
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", str(fixture_source), str(root)],
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

                def call(*, archive=archive, manifest=manifest, target=target):
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


class StagingDiagnosticTests(RemoteFixture):
    def diagnostic(self, sub_step):
        self.static_failure(lambda: self.bridge.stage(recipe(), b"x"))
        diagnostic = self.bridge.last_staging_failure
        self.assertEqual(diagnostic["entry"], "stage")
        self.assertEqual(diagnostic["sub_step"], sub_step)
        self.assertTrue(self.bridge._unresolved)
        nr_encode(diagnostic)
        self.assertNotIn("SYNTHETIC-PRIVATE-CANARY", str(diagnostic))
        return diagnostic

    def test_preflight_validation_retains_only_bounded_keys(self):
        for extra in (False, True):
            with self.subTest(extra=extra):
                self.remote = self.root / ("preflight-" + str(extra))
                self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
                def mutate(request, reply, *, extra=extra):
                    if request["operation"] == "native_preflight":
                        reply = dict(reply, binding_sha256="b" * 64)
                        if extra:
                            reply.update({str(i) + "x" * 100: "private" for i in range(40)})
                        return reply
                    return reply
                self.mutate = mutate
                diagnostic = self.diagnostic("preflight")
                self.assertEqual(diagnostic["error_type"], "ValueError")
                self.assertEqual(len(diagnostic["reply_keys"]), 32 if extra else 4)
                self.assertEqual(diagnostic["reply_keys"], sorted(diagnostic["reply_keys"]))
                self.assertTrue(all(len(k) <= 64 for k in diagnostic["reply_keys"]))
                self.assertNotIn("sent_unix", diagnostic)
                self.assertNotIn("operation", diagnostic)

    def test_upload_chunk_ack_failure_names_operation_without_send_time(self):
        self.mutate = lambda request, reply: ({} if request["operation"] == "native_chunk" else reply)
        diagnostic = self.diagnostic("upload_package")
        self.assertEqual(diagnostic["operation"], "native_chunk")
        self.assertNotIn("sent_unix", diagnostic)
        self.assertIs(diagnostic["reply_received"], True)

    def test_bound_worker_refusal_records_no_reply_or_send_time(self):
        with patch.object(RemoteNativeSpeechRuntime, "_bound_worker",
                          side_effect=NativeSpeechError("SYNTHETIC-PRIVATE-CANARY")):
            diagnostic = self.diagnostic("preflight")
        self.assertEqual(diagnostic["operation"], "native_preflight")
        self.assertEqual(diagnostic["error_type"], "NativeSpeechError")
        self.assertIs(diagnostic["reply_received"], False)
        self.assertNotIn("sent_unix", diagnostic)
        self.assertEqual(self.requests, [])

    def test_setup_no_reply_retains_exact_controller_send_time(self):
        for raised in (True, False):
            with self.subTest(raised=raised):
                self.remote = self.root / ("setup-" + str(raised))
                self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
                self.lost = "native_setup" if raised else None
                self.mutate = lambda request, reply, raised=raised: (None
                    if not raised and request["operation"] == "native_setup" else reply)
                diagnostic = self.diagnostic("setup")
                self.assertEqual(diagnostic["operation"], "native_setup")
                self.assertEqual(diagnostic["error_type"], "TimeoutError" if raised else "NativeSpeechError")
                self.assertEqual(diagnostic["sent_unix"], self.requests[-1]["native_sent_unix"])
                self.assertIs(type(diagnostic["sent_unix"]), float)
                self.assertIs(diagnostic["reply_received"], False)

    def test_fetch_framing_failure_is_distinct_from_dispatch_failure(self):
        self.mutate = lambda request, reply: (dict(reply, offset=-1)
            if request["operation"] == "native_fetch" else reply)
        diagnostic = self.diagnostic("fetch_installed")
        self.assertNotIn("sent_unix", diagnostic)
        self.assertNotIn("operation", diagnostic)

    def test_install_record_validation_retains_keys(self):
        def mutate(request, reply):
            if request["operation"] == "native_setup":
                path = self.remote / "installed.json"
                value = nr_json(path.read_bytes())
                value["recipe_sha256"] = "b" * 64
                raw = nr_encode(value)
                path.write_bytes(raw)
                return remote._descriptor(raw)
            return reply
        self.mutate = mutate
        diagnostic = self.diagnostic("result_validation")
        self.assertIn("recipe_sha256", diagnostic["reply_keys"])
        self.assertEqual(diagnostic["error_type"], "ValueError")
        self.assertNotIn("operation", diagnostic)

    def test_stage_wrapper_final_admission_is_final_admission(self):
        def mutate(request, reply):
            if request["operation"] == "native_fetch":
                checks = iter((None, WorkerError("SYNTHETIC-PRIVATE-CANARY")))
                def check():
                    error = next(checks)
                    if error:
                        raise error
                self.worker.check = check
            return reply
        self.mutate = mutate
        diagnostic = self.diagnostic("final_admission")
        self.assertEqual(diagnostic["error_type"], "WorkerError")
        self.assertNotIn("operation", diagnostic)

    def test_interrupt_keeps_original_propagation_and_operation(self):
        def interrupt(request, timeout, **options):
            raise KeyboardInterrupt("SYNTHETIC-PRIVATE-CANARY")
        self.worker._request = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.bridge.stage(recipe(), b"x")
        diagnostic = self.bridge.last_staging_failure
        self.assertEqual(diagnostic["sub_step"], "preflight")
        self.assertEqual(diagnostic["operation"], "native_preflight")
        self.assertEqual(diagnostic["error_type"], "KeyboardInterrupt")
        self.assertTrue(self.bridge._unresolved)


class SelectDiagnosticTests(RemoteFixture):
    def diagnostic(self, sub_step):
        with self.assertRaisesRegex(NativeSpeechError,
                "^remote native selection incomplete or unconfirmed$") as caught:
            self.bridge.select(self.selected)
        self.assertTrue(caught.exception.cleanup_uncertain)
        self.assertNotIn("SYNTHETIC-PRIVATE-CANARY",
                         "".join(traceback.format_exception(caught.exception)))
        diagnostic = self.bridge.last_select_failure
        self.assertEqual(diagnostic["entry"], "select")
        self.assertEqual(diagnostic["sub_step"], sub_step)
        self.assertTrue(self.bridge._unresolved)
        self.assertIsNone(self.bridge._selection)
        self.assertLessEqual(set(diagnostic), {"entry", "sub_step", "error_type", "operation",
            "reply_received", "sent_unix", "reply_keys", "reply_type"})
        nr_encode(diagnostic)
        self.assertNotIn("SYNTHETIC-PRIVATE-CANARY", str(diagnostic))
        return diagnostic

    def test_upload_ack_failure_names_operation(self):
        self.bridge.stage(recipe(), b"x")
        self.mutate = lambda request, reply: ({}
            if request["operation"] == "native_chunk" and request["role"] == "selection"
            else reply)
        diagnostic = self.diagnostic("upload")
        self.assertEqual(diagnostic["operation"], "native_chunk")
        self.assertIs(diagnostic["reply_received"], True)
        self.assertNotIn("sent_unix", diagnostic)
        self.assertNotIn("native_select", [request["operation"] for request in self.requests])

    def test_native_select_no_reply_retains_exact_send_time(self):
        for raised in (True, False):
            with self.subTest(raised=raised):
                self.remote = self.root / ("select-" + str(raised))
                self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
                self.bridge.stage(recipe(), b"x")
                self.requests.clear()
                self.lost = "native_select" if raised else None
                self.mutate = lambda request, reply, raised=raised: (None
                    if not raised and request["operation"] == "native_select" else reply)
                diagnostic = self.diagnostic("native_select")
                self.assertEqual(diagnostic["operation"], "native_select")
                self.assertEqual(diagnostic["error_type"],
                                 "TimeoutError" if raised else "NativeSpeechError")
                self.assertIs(diagnostic["reply_received"], False)
                self.assertEqual(diagnostic["sent_unix"], self.requests[-1]["native_sent_unix"])
                self.assertIs(type(diagnostic["sent_unix"]), float)
                self.assertTrue((self.remote / "selection-consumed").exists())
                self.assertEqual(sum(r["operation"] == "native_select" for r in self.requests), 1)

    def test_reply_validation_retains_only_keys_or_type(self):
        wrong_hash = "b" * 64
        for reply in ({"selection_sha256": wrong_hash}, ["SYNTHETIC-PRIVATE-CANARY"]):
            with self.subTest(reply_type=type(reply).__name__):
                self.remote = self.root / type(reply).__name__
                self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
                self.bridge.stage(recipe(), b"x")
                self.mutate = lambda request, original, reply=reply: (reply
                    if request["operation"] == "native_select" else original)
                diagnostic = self.diagnostic("reply_validation")
                self.assertEqual(diagnostic, dict(entry="select", sub_step="reply_validation",
                    error_type="ValueError", **({"reply_keys": ["selection_sha256"]}
                        if type(reply) is dict else {"reply_type": "list"})))
                self.assertNotIn(wrong_hash, str(diagnostic))
                self.assertNotIn(self.selected.subject_sha256(), str(diagnostic))

    def test_reply_validation_bounds_keys(self):
        self.bridge.stage(recipe(), b"x")
        self.mutate = lambda request, reply: ({str(i) + "x" * 100: "SYNTHETIC-PRIVATE-CANARY"
            for i in range(40)} if request["operation"] == "native_select" else reply)
        diagnostic = self.diagnostic("reply_validation")
        self.assertEqual(len(diagnostic["reply_keys"]), 32)
        self.assertEqual(diagnostic["reply_keys"], sorted(diagnostic["reply_keys"]))
        self.assertTrue(all(len(k) <= 64 for k in diagnostic["reply_keys"]))

    def test_installation_precheck_has_no_diagnostic(self):
        self.bridge.stage(recipe(), b"x")
        bad = replace(self.selected.runtime, root="/untrusted")
        subject = native_subject(self.selected.interpreter, self.selected.seccomp, bad,
            self.selected.sdk, self.selected.base, self.selected.design, self.selected.hardware)
        changed = replace(self.selected, runtime=bad, reviewed_subject_sha256=subject)
        calls = len(self.requests)
        with self.assertRaisesRegex(NativeSpeechError,
                "native selection differs from actual remote installation"):
            self.bridge.select(changed)
        self.assertIsNone(self.bridge.last_select_failure)
        self.assertFalse(self.bridge._unresolved)
        self.assertEqual(len(self.requests), calls)

    def test_success_and_second_selection_clear_stale_diagnostics(self):
        self.bridge.stage(recipe(), b"x")
        self.bridge.last_select_failure = {"entry": "stale"}
        self.bridge._last_call_failure = {"operation": "stale"}
        self.bridge.select(self.selected)
        self.assertIsNone(self.bridge.last_select_failure)
        self.assertIsNone(self.bridge._last_call_failure)
        self.assertIs(self.bridge._selection, self.selected)
        self.assertFalse(self.bridge._unresolved)
        self.bridge.last_select_failure = {"entry": "stale"}
        self.bridge._last_call_failure = {"operation": "stale"}
        calls = len(self.requests)
        with self.assertRaisesRegex(NativeSpeechError,
                "separately reviewed installed native selection required"):
            self.bridge.select(self.selected)
        self.assertIsNone(self.bridge.last_select_failure)
        self.assertIsNone(self.bridge._last_call_failure)
        self.assertEqual(len(self.requests), calls)
        self.assertIs(self.bridge._selection, self.selected)

    def test_failed_handshake_then_precheck_does_not_inherit_diagnostic(self):
        self.bridge.stage(recipe(), b"x")
        self.lost = "native_select"
        self.diagnostic("native_select")
        calls = len(self.requests)
        with self.assertRaisesRegex(NativeSpeechError,
                "separately reviewed installed native selection required"):
            self.bridge.select(None)
        self.assertIsNone(self.bridge.last_select_failure)
        self.assertIsNone(self.bridge._last_call_failure)
        self.assertEqual(len(self.requests), calls)
        self.assertTrue(self.bridge._unresolved)

    def test_interrupt_keeps_original_propagation_and_operation(self):
        self.bridge.stage(recipe(), b"x")
        def interrupt(request, reply):
            if request["operation"] == "native_select":
                raise KeyboardInterrupt("SYNTHETIC-PRIVATE-CANARY")
            return reply
        self.mutate = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.bridge.select(self.selected)
        diagnostic = self.bridge.last_select_failure
        self.assertEqual(diagnostic["entry"], "select")
        self.assertEqual(diagnostic["sub_step"], "native_select")
        self.assertEqual(diagnostic["operation"], "native_select")
        self.assertEqual(diagnostic["error_type"], "KeyboardInterrupt")
        self.assertTrue(self.bridge._unresolved)
        self.assertNotIn("SYNTHETIC-PRIVATE-CANARY", str(diagnostic))
        nr_encode(diagnostic)


class SpeechDiagnosticTests(RemoteFixture):
    def diagnostic(self, sub_step):
        self.static_failure(lambda: self.bridge.preview(self.design))
        diagnostic = self.bridge.last_speech_failure
        self.assertEqual(diagnostic["entry"], "invoke")
        self.assertEqual(diagnostic["sub_step"], sub_step)
        self.assertIn("error_type", diagnostic)
        self.assertNotIn("SYNTHETIC-PRIVATE-CANARY", str(diagnostic))
        self.assertTrue(self.bridge._unresolved)
        nr_encode(diagnostic)
        return diagnostic

    def test_speech_dispatch_failure_retains_call_metadata(self):
        self.prepare()
        original = self.worker.native_request
        for fault in ("raise", "no_reply"):
            def request(fields, *, fault=fault, **kwargs):
                if fields["operation"] == "native_invoke":
                    if fault == "raise":
                        raise TimeoutError("SYNTHETIC-PRIVATE-CANARY")
                    return None
                return original(fields, **kwargs)
            # Each case owns a fresh bridge and remote session.
            if fault == "no_reply":
                self.remote = self.root / "no-reply"
                self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
                self.prepare()
            with self.subTest(fault=fault), patch.object(self.worker, "native_request", request):
                diagnostic = self.diagnostic("dispatch")
                self.assertEqual(diagnostic["operation"], "native_invoke")
                self.assertFalse(diagnostic["reply_received"])
                self.assertEqual(set(diagnostic),
                    {"entry", "sub_step", "error_type", "operation", "reply_received", "sent_unix"})

    def test_capture_call_failure_retains_only_valid_child_output_tail(self):
        tail = "NativeSpeechError: native speech exceeds reserved duration"
        for value in (tail, "x" * 512, "x" * 513, 3, None, "", "bad\n", "é"):
            error = WorkerError("bounded SSH process failed")
            error.cause_diagnostic = {"child_output_tail": value}
            self.bridge._capture_call_failure("native_invoke", error, reply_received=False)
            expected = {"operation": "native_invoke", "reply_received": False,
                        "error_type": "WorkerError"}
            if value in (tail, "x" * 512):
                expected["child_output_tail"] = value
            with self.subTest(value=value):
                self.assertEqual(self.bridge._last_call_failure, expected)

    def test_speech_dispatch_retains_bounded_cause(self):
        cases = (
            ({"cause_type": "BootstrapError", "reason": "process_failed", "returncode": 1},
             {"cause_type": "BootstrapError", "reason": "process_failed", "returncode": 1}),
            ({"cause_type": "BootstrapError", "reason": "process_failed", "returncode": True},
             {"cause_type": "BootstrapError", "reason": "process_failed"}),
            ({"cause_type": "WorkerError",
              "supervision_message": "worker supervision or lease deadline failed"},
             {"cause_type": "WorkerError",
              "supervision_message": "worker supervision or lease deadline failed"}),
            ({"cause_type": "BootstrapError", "reason": "process_failed", "returncode": "1",
              "message": "SYNTHETIC-PRIVATE-CANARY", "extra": 1},
             {"cause_type": "BootstrapError", "reason": "process_failed"}),
        )
        for index, (cause, expected) in enumerate(cases):
            self.remote = self.root / f"cause-{index}"
            self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
            self.prepare()
            original = self.worker.native_request
            error = WorkerError("bounded SSH process failed")
            error.cause_diagnostic = cause
            def request(fields, *, error=error, original=original, **kwargs):
                if fields["operation"] == "native_invoke":
                    raise error
                return original(fields, **kwargs)
            with self.subTest(cause=cause), patch.object(self.worker, "native_request", request):
                diagnostic = self.diagnostic("dispatch")
                self.assertEqual(diagnostic["error_type"], "WorkerError")
                self.assertEqual(diagnostic["operation"], "native_invoke")
                self.assertFalse(diagnostic["reply_received"])
                self.assertEqual(set(diagnostic),
                    {"entry", "sub_step", "error_type", "operation", "reply_received", "sent_unix"}
                    | set(expected))
                for key, value in expected.items():
                    self.assertEqual(diagnostic[key], value)

    def test_speech_upload_failure_retains_sub_step(self):
        self.prepare()
        with patch.object(self.bridge, "_upload", side_effect=OSError("SYNTHETIC-PRIVATE-CANARY")):
            self.assertEqual(self.diagnostic("upload_input")["error_type"], "OSError")

    def test_speech_header_validation_retains_bounded_keys_only(self):
        self.prepare()
        def tamper(request, reply):
            if request["operation"] == "native_invoke":
                path = self.remote / (request["token"] + ".result")
                header, payload = unpack(bootstrap.nr_read(path))
                header["binding"] = "SYNTHETIC-PRIVATE-CANARY"
                header.update({str(i) + "x" * 100: "SYNTHETIC-PRIVATE-CANARY" for i in range(40)})
                raw = pack(header, payload)
                path.write_bytes(raw)
                return remote._descriptor(raw)
            return reply
        self.mutate = tamper
        diagnostic = self.diagnostic("header_validation")
        self.assertEqual(len(diagnostic["reply_keys"]), 32)
        self.assertTrue(all(len(k) <= 64 for k in diagnostic["reply_keys"]))
        self.assertEqual(diagnostic["reply_keys"], sorted(diagnostic["reply_keys"]))

    def test_speech_reserved_duration_headroom_returns_proof(self):
        self.prepare()
        slot_frames, remainder = divmod(self.request.duration_ticks * 24000,
                                        self.request.timebase.ticks_per_second)
        self.assertEqual(remainder, 0)
        self.assertEqual(slot_frames % 2, 0)
        payload = wav_bytes(frames=slot_frames * 3 // 2)
        with patch("test_native_remote.response",
                side_effect=lambda header, original=response: original(header, payload)):
            result = self.bridge.synthesize(self.request, self.reference)
        self.assertEqual(result, payload)
        self.assertEqual(self.bridge.proof_for(self.request, result).payload_sha256,
                         hashlib.sha256(payload).hexdigest())
        self.assertIsNone(self.bridge.last_speech_failure)

    def test_speech_above_headroom_fails_payload_validation(self):
        self.prepare()
        slot_frames, remainder = divmod(self.request.duration_ticks * 24000,
                                        self.request.timebase.ticks_per_second)
        self.assertEqual(remainder, 0)
        payload = wav_bytes(frames=slot_frames * 2 + 1)

        def tamper(request, reply):
            if request["operation"] == "native_invoke":
                path = self.remote / (request["token"] + ".result")
                header, _ = unpack(bootstrap.nr_read(path))
                header["receipt"].update(frames=slot_frames * 2 + 1,
                    payload_sha256=hashlib.sha256(payload).hexdigest())
                raw = pack(header, payload)
                path.write_bytes(raw)
                return remote._descriptor(raw)
            return reply

        self.mutate = tamper
        self.static_failure(lambda: self.bridge.synthesize(self.request, self.reference))
        self.assertEqual(self.bridge.last_speech_failure["entry"], "invoke")
        self.assertEqual(self.bridge.last_speech_failure["sub_step"], "payload_validation")
        self.assertEqual(self.bridge.last_speech_failure["error_type"], "ValueError")
        self.assertEqual(self.child_count, 1)

    def test_speech_synthesize_short_session_receipt_validation_attribution(self):
        self.prepare()
        def tamper(request, reply):
            if request["operation"] == "native_invoke":
                path = self.remote / (request["token"] + ".result")
                header, payload = unpack(bootstrap.nr_read(path))
                header["receipt"]["seed"] += 1
                raw = pack(header, payload)
                path.write_bytes(raw)
                return remote._descriptor(raw)
            return reply
        self.mutate = tamper
        # This fixture uses short-session base-English, without an attempts suffix.
        with self.assertRaisesRegex(NativeSpeechError,
                "remote native attempt incomplete or unconfirmed") as caught:
            self.bridge.synthesize(self.request, self.reference)
        self.assertTrue(caught.exception.cleanup_uncertain)
        self.assertEqual(self.bridge.last_speech_failure["entry"], "invoke")
        self.assertEqual(self.bridge.last_speech_failure["sub_step"], "receipt_validation")
        self.assertEqual(self.child_count, 1)
        self.assertIn("base-English", self.bridge._used)

    def test_speech_content_tree_validation_attribution(self):
        self.prepare()
        def tamper(request, reply):
            if request["operation"] == "native_invoke":
                path = self.remote / (request["token"] + ".result")
                header, payload = unpack(bootstrap.nr_read(path))
                header["receipt"]["sdk_content_sha256"] = "a" * 64
                raw = pack(header, payload)
                path.write_bytes(raw)
                return remote._descriptor(raw)
            return reply
        self.mutate = tamper
        self.diagnostic("content_tree_validation")
        self.assertEqual(self.child_count, 1)

    def test_speech_fetch_failure_attribution_after_dispatch(self):
        self.prepare()
        self.lost = "native_fetch"
        self.requests.clear()
        diagnostic = self.diagnostic("fetch_result")
        self.assertEqual(diagnostic["operation"], "native_fetch")
        self.assertFalse(diagnostic["reply_received"])
        self.assertEqual(self.child_count, 1)
        operations = [request["operation"] for request in self.requests]
        self.assertLess(operations.index("native_invoke"), operations.index("native_fetch"))

    def test_speech_preflight_preserves_error_and_success_clears_failure(self):
        with self.assertRaisesRegex(NativeSpeechError, "runtime unavailable or capacity exhausted") as caught:
            self.bridge.preview(self.design)
        self.assertFalse(caught.exception.cleanup_uncertain)
        self.assertEqual(self.bridge.last_speech_failure,
            dict(entry="invoke", sub_step="preflight", error_type="NativeSpeechError"))
        self.assertFalse(self.bridge._unresolved)
        self.prepare()
        self.bridge._last_call_failure = {"operation": "stale"}
        payload = self.bridge.preview(self.design)
        self.assertIsNone(self.bridge.last_speech_failure)
        self.assertIsNone(self.bridge._last_call_failure)
        self.assertEqual(self.bridge.proof_for(self.design, payload).payload_sha256,
                         hashlib.sha256(payload).hexdigest())

    def test_speech_health_check_dead_channel_does_not_consume_or_poison(self):
        self.prepare()
        attempts, used, calls = self.bridge._attempts, set(self.bridge._used), len(self.requests)
        for fault in (False, WorkerError("SYNTHETIC-PRIVATE-CANARY")):
            with self.subTest(fault=type(fault).__name__):
                if fault is False:
                    probe = patch.object(self.worker, "ping", return_value=False)
                else:
                    probe = patch.object(self.worker, "_request", side_effect=fault)
                with probe:
                    self.assertFalse(self.bridge.speech_health_check())
                self.assertEqual(self.bridge.last_speech_failure,
                    dict(entry="health_check", sub_step="ping", error_type="NativeSpeechError"))
                self.assertFalse(self.bridge._unresolved)
                self.assertEqual((self.bridge._attempts, self.bridge._used), (attempts, used))
                self.assertEqual(len(self.requests), calls)
        with patch.object(self.worker, "_request", return_value={"status": "ssh_ready"}) as ping:
            self.assertTrue(self.bridge.speech_health_check(2))
            ping.assert_called_once_with({"operation": "ping"}, 2, supervise=False)
        self.assertIsNone(self.bridge.last_speech_failure)
        self.bridge.preview(self.design)
        self.assertIsNone(self.bridge.last_speech_failure)

    def test_speech_health_check_requires_live_binding(self):
        self.assertFalse(self.bridge.speech_health_check())
        self.assertEqual(self.bridge.last_speech_failure,
            dict(entry="health_check", sub_step="binding", error_type="NativeSpeechError"))
        self.prepare()
        for field, value in (("_binding", None), ("_deadline", None), ("_unresolved", True)):
            with self.subTest(field=field), patch.object(self.bridge, field, value), patch.object(
                    self.worker, "ping") as ping:
                self.assertFalse(self.bridge.speech_health_check())
                ping.assert_not_called()
                self.assertEqual(self.bridge.last_speech_failure["sub_step"], "binding")

    def test_speech_health_check_expired_deadline_never_pings(self):
        self.prepare()
        with patch.object(self.bridge, "_deadline", time.monotonic() - 1), patch.object(
                self.worker, "ping") as ping:
            self.assertFalse(self.bridge.speech_health_check())
            ping.assert_not_called()
        self.assertEqual(self.bridge.last_speech_failure,
            dict(entry="health_check", sub_step="binding", error_type="NativeSpeechError"))
        self.assertFalse(self.bridge._unresolved)

    def test_speech_health_check_caps_ping_to_remaining_deadline(self):
        self.prepare()
        with patch.object(self.bridge, "_deadline", 102), patch(
                "cineforge.native_remote.time.monotonic", return_value=100), patch.object(
                self.worker, "ping", return_value=True) as ping:
            self.assertTrue(self.bridge.speech_health_check(30))
            ping.assert_called_once_with(2)
        self.assertIsNone(self.bridge.last_speech_failure)
