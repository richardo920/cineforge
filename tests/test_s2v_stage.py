"""Local fake owned SSH RPC plus real private files; never SSH/cloud/model execution."""
from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import time
import traceback
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_conditioning import runtime_fixture
from test_runpod import spec as legacy_spec
from test_s2v import context, worker
from test_s2v_profile import png

from cineforge import s2v_profile as profile
from cineforge import worker_bootstrap as bootstrap
from cineforge.comfy_runtime import ClosedComfyRuntime
from cineforge.runpod import RunpodPodSpec
from cineforge.s2v import S2VError, digest, select_s2v
from cineforge.s2v_stage import paired_transfer
from cineforge.skills import bind_shot
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.worker_guard import WorkerGuardPlan
from cineforge.workflows import canonical


def reapprove(gate, **changes):
    gate = replace(gate, authorize=lambda _: None, **changes)
    gate = replace(gate, binding=bind_shot(gate.shot, gate.binding.skills))
    approved = digest(gate.subject)
    def authorize(subject):
        if digest(subject) != approved:
            raise RuntimeError("CANARY stale authorization")
    return replace(gate, authorize=authorize)


class PairTransferTests(unittest.TestCase):
    gpu_id = profile.GPU_ID

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cf-s2v-stage-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.gate, self.calls = context(self.root, gpu_id=self.gpu_id)
        self.selection = select_s2v(self.gate)
        self.remote = self.root / "worker"
        (self.remote / "ComfyUI" / "input").mkdir(parents=True, mode=0o700)
        self.key = self.root / "synthetic-key"
        self.key.write_bytes(b"synthetic fixture, not real credential")
        self.key.chmod(0o600)
        self.spec = RunpodPodSpec("s2v-synthetic", profile.IMAGE, self.gpu_id, "EU-RO-1",
            legacy_spec().ssh_public_key, max_hourly_usd=1.25, min_ram_per_gpu_gb=64, disk_gb=150)
        self.remaining = 600
        self.plan = WorkerGuardPlan("owned-worker", self.spec.session_id, "a" * 32,
                                    time.time() + 1000, 30)
        self.budget = SimpleNamespace(approval=self.gate.service.approval,
            remaining_seconds=lambda _: self.remaining, guard_plan=lambda _: self.plan)
        self.runtime = dict(runtime_fixture(), compute_precision="bf16")
        if self.gpu_id != profile.GPU_ID:
            self.runtime.update(python_version="3.11.14", gpu_raw_name=self.gpu_id, gpu_memory_mib=46000,
                nvml_total_mib=46068, nvml_reserved_mib=68,
                capability=list(profile.worker(self.gpu_id).reference_runtime.capability), arch_list=["sm_86"])
        self.requests = []
        self.loss = None
        self.fault = None
        self.driver = self.new_driver()

    def new_driver(self):
        driver = SSHWorker(self.spec, self.gate.service, self.budget, None, self.key,
                           self.root / "ssh-state", allow_live=True)
        driver.plan, driver._key = self.plan, "lease"
        driver.runtime_evidence = self.runtime
        driver._prepared_s2v = worker(self.gate)
        # This suite's synthetic Comfy wire does not prove an installed runtime.
        closed = object.__new__(ClosedComfyRuntime)
        closed.verify = lambda driver, selected, **kw: selected.gate.preflight()
        # Synthetic protocol surrogates only; separate precision tests exercise actual contracts.
        closed.begin_precision = lambda driver, selected, **kw: {"synthetic_precision": "intent"}
        closed.finish_precision = lambda driver, selected, prompt_id, **kw: {"synthetic_precision": prompt_id}
        closed.abandon_precision = lambda: None
        closed.require_precision_post = lambda driver, body, **kw: None
        driver._closed_comfy, driver._comfy_selection = closed, self.selection
        driver._request = self.rpc
        self.addCleanup(driver.close)
        return driver

    def rpc(self, request, timeout, **options):
        self.requests.append(request)
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, min(10, self.remaining - 120))
        if self.fault is not None:
            self.fault(request)
        body = dict(request, profile=self.driver._profile)
        reply = paired_transfer(body, self.remote)
        if request["operation"] == self.loss:
            raise TimeoutError("CANARY lost acknowledgement")
        return reply

    def static_failure(self, operation):
        with self.assertRaises(WorkerError) as caught:
            operation()
        self.assertNotIn("CANARY", "".join(traceback.format_exception(caught.exception)))
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)

    def test_worker_constructor_requires_all_exact_scope_resources(self):
        for change in (dict(disk_gb=80), dict(min_cuda_version="12.9"),
                       dict(max_hourly_usd=2), dict(min_ram_per_gpu_gb=32)):
            with self.subTest(change=change), self.assertRaises(WorkerError):
                SSHWorker(replace(self.spec, **change), self.gate.service, self.budget,
                    None, self.key, self.root / "rejected-state", allow_live=True)
        self.assertFalse((self.root / "rejected-state").exists())

    def test_materialized_setup_requires_arm_runtime_rights_and_consumes_lost_ack(self):
        self.driver._prepared_s2v = None
        self.driver._request = lambda *args, **kw: self.fail("unapproved setup RPC")
        self.static_failure(lambda: self.driver.prepare())
        self.driver.runtime_evidence = None
        self.static_failure(lambda: self.driver.prepare(self.selection))
        self.driver.runtime_evidence = self.runtime
        self.assertFalse(self.driver._s2v_setup_attempted)
        requests = []
        def lost(request, timeout):
            requests.append(request)
            raise WorkerError("synthetic setup lost ACK")
        self.driver._request = lost
        runtime = object.__new__(ClosedComfyRuntime)  # Explicit trusted synthetic RPC stand-in.
        runtime._admit = lambda *args: None
        runtime._poison = lambda: None
        runtime.prepare = lambda driver, selection, **kw: driver._request({
            "operation": "native_comfy_setup", "selection": selection.declarations}, 10)
        self.static_failure(lambda: self.driver.prepare(self.selection, runtime=runtime))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["operation"], "native_comfy_setup")
        self.assertEqual(requests[0]["selection"], self.selection.declarations)
        self.assertIsNone(self.driver.s2v_verification)
        self.static_failure(lambda: self.driver.prepare(self.selection, runtime=runtime))
        self.assertEqual(len(requests), 1)

    def test_setup_rechecks_rights_after_reply_before_committing_prepared_worker(self):
        self.driver._prepared_s2v = None
        def reply(request, timeout):
            self.gate.ledger.revoke(self.gate.model_rights, reason="CANARY")
            return {"verification": worker(self.gate).record, "runtime": self.runtime}
        self.driver._request = reply
        runtime = object.__new__(ClosedComfyRuntime)  # Does not claim actual runtime proof.
        runtime._admit = lambda *args: None
        runtime._poison = lambda: None
        runtime.prepare = lambda driver, selection, **kw: driver._request({
            "operation": "native_comfy_setup", "selection": selection.declarations}, 10)
        self.static_failure(lambda: self.driver.prepare(self.selection, runtime=runtime))
        self.assertIsNone(self.driver.s2v_verification)
        self.assertTrue(self.driver._s2v_setup_attempted)

    def test_complete_pair_exact_bytes_metadata_and_consumed_repeated_request(self):
        proof = self.driver.stage_s2v(self.selection)
        bound = self.selection.bind_worker(worker(self.gate), proof)
        bound.require_execution_worker(worker(self.gate))
        for kind, payload in (("image", self.gate.actor.read()), ("audio", self.gate.speech.payload)):
            path = self.remote / "ComfyUI/input" / self.gate.media[kind]["basename"]
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(self.requests[0]["operation"], "s2v_pair_begin")
        self.assertEqual(self.requests[-1]["operation"], "s2v_pair_finish")
        calls = len(self.requests)
        self.static_failure(lambda: self.driver.stage_s2v(self.selection))
        self.assertEqual(len(self.requests), calls)
        fresh = self.new_driver()
        self.static_failure(lambda: fresh.stage_s2v(self.selection))
        self.assertEqual(self.requests[-1]["operation"], "s2v_pair_begin")

    def test_large_original_png_chunked_under_existing_request_limit(self):
        path = self.root / "large"
        path.mkdir()
        gate, _ = context(path, actor_png=png(compressed=False))
        gate = reapprove(gate, service=self.gate.service)
        selection = select_s2v(gate)
        proof = self.driver.stage_s2v(selection)
        self.assertGreater(proof.pair_sha256, "")
        chunks = [request for request in self.requests if request["operation"] == "s2v_pair_chunk"]
        self.assertGreater(len(chunks), 32)
        for request in chunks:
            self.assertLess(len(canonical(request)), 256 * 1024)
            self.assertLessEqual(len(base64.b64decode(request["chunk"])), 65536)
        path = self.remote / "ComfyUI/input" / gate.media["image"]["basename"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), gate.media["image"]["sha256"])

    def test_second_approved_language_reuses_only_complete_exact_owned_slots(self):
        self.driver.stage_s2v(self.selection)
        first_image = self.remote / "ComfyUI/input" / self.gate.media["image"]["basename"]
        inode = first_image.stat().st_ino
        path = self.root / "zh"
        path.mkdir()
        gate, _ = context(path, language="zh")
        gate = reapprove(gate, service=self.gate.service)
        selection = select_s2v(gate)
        start = len(self.requests)
        proof = self.driver.stage_s2v(selection)
        self.assertEqual(first_image.stat().st_ino, inode)
        self.assertEqual([request["operation"] for request in self.requests[start:]],
                         ["s2v_pair_begin", "s2v_pair_finish"])
        selection.bind_worker(worker(gate), proof).require_execution_worker(worker(gate))

    def test_lost_begin_chunk_or_finish_ack_is_consumed_without_blind_retry(self):
        for phase in ("s2v_pair_begin", "s2v_pair_chunk", "s2v_pair_finish"):
            with self.subTest(phase=phase):
                self.remote = self.root / phase
                (self.remote / "ComfyUI/input").mkdir(parents=True, mode=0o700)
                self.driver = self.new_driver()
                self.requests = []
                self.loss = phase
                self.static_failure(lambda: self.driver.stage_s2v(self.selection))
                self.assertEqual(self.requests[-1]["operation"], phase)
                old = len(self.requests)
                self.static_failure(lambda: self.driver.stage_s2v(self.selection))
                self.assertEqual(len(self.requests), old)
                with self.assertRaises(S2VError):
                    self.selection.require_execution_worker(worker(self.gate))
        self.loss = None

    def test_symlink_hardlink_foreign_or_partial_slots_are_never_overwritten(self):
        for kind in ("foreign", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                self.remote = self.root / kind
                (self.remote / "ComfyUI/input").mkdir(parents=True, mode=0o700)
                outside = self.remote / "foreign"
                outside.write_bytes(b"CANARY foreign bytes")
                outside.chmod(0o600)
                target = self.remote / "ComfyUI/input" / self.gate.media["image"]["basename"]
                if kind == "symlink":
                    target.symlink_to(outside)
                elif kind == "hardlink":
                    os.link(outside, target)
                else:
                    target.write_bytes(b"CANARY partial")
                    target.chmod(0o600)
                self.driver = self.new_driver()
                self.static_failure(lambda: self.driver.stage_s2v(self.selection))
                self.assertEqual(outside.read_bytes(), b"CANARY foreign bytes")
                self.assertEqual(target.read_bytes(), b"CANARY partial" if kind == "foreign"
                                 else b"CANARY foreign bytes")

    def test_revocation_mid_transfer_or_before_submission_and_foreign_worker_refuse(self):
        before = len(self.requests)
        self.plan = replace(self.plan, pod_id="foreign-worker")
        self.driver.plan = self.plan
        self.static_failure(lambda: self.driver.stage_s2v(self.selection))
        self.assertEqual(len(self.requests), before)
        self.plan = replace(self.plan, pod_id="owned-worker")
        self.driver.plan = self.plan
        def revoke(request):
            if request["operation"] == "s2v_pair_chunk":
                self.gate.ledger.revoke(self.gate.model_rights, reason="CANARY")
        self.fault = revoke
        self.static_failure(lambda: self.driver.stage_s2v(self.selection))
        self.assertNotEqual(self.requests[-1]["operation"], "s2v_pair_finish")
        self.assertEqual(list((self.remote / "ComfyUI/input").iterdir()), [])

    def test_native_callback_requires_arm_hardware_optin_and_remaining_reserve_without_graph(self):
        self.driver._prepared_s2v = None
        calls = []
        def callback(timeout, *, absolute_deadline):
            self.assertGreater(absolute_deadline, time.monotonic())
            calls.append(timeout)
            return "synthetic-result"
        self.assertFalse(hasattr(self.driver.selection, "workflow"))
        self.static_failure(lambda: self.driver.run_guarded_native(callback))
        self.assertEqual(calls, [])
        self.assertEqual(self.driver.run_guarded_native(callback, allow_live=True), "synthetic-result")
        self.assertEqual(calls, [480])
        self.driver.plan = None
        self.static_failure(lambda: self.driver.run_guarded_native(callback, allow_live=True))
        self.driver.plan = self.plan
        self.driver.runtime_evidence = None
        self.static_failure(lambda: self.driver.run_guarded_native(callback, allow_live=True))
        self.driver.runtime_evidence = self.runtime
        self.remaining = 120
        self.static_failure(lambda: self.driver.run_guarded_native(callback, allow_live=True))
        self.remaining = 600
        self.static_failure(lambda: self.driver.run_guarded_native(
            lambda _, **kw: (_ for _ in ()).throw(RuntimeError("CANARY")), allow_live=True))

    def test_native_original_absolute_cap_survives_delayed_callback_and_typeerror_is_once(self):
        import test_native_speech as native_fixtures

        from cineforge.native_speech import NativeSpeechRuntime
        configured = native_fixtures.selection(self.root)
        self.remaining = 123
        now = [100.0]
        calls = []
        def delayed(timeout, *, absolute_deadline):
            calls.append((timeout, absolute_deadline))
            now[0] += 4
            # A fresh relative timeout would grant until107; original cap103 refuses.
            return NativeSpeechRuntime(configured, allow_native=True, timeout_seconds=timeout,
                                       absolute_deadline=absolute_deadline)
        with patch("cineforge.ssh_worker.time.monotonic", side_effect=lambda: now[0]), \
                patch("cineforge.native_speech.run_child") as child:
            self.static_failure(lambda: self.driver.run_guarded_native(delayed, allow_live=True))
        self.assertEqual(calls, [(3, 103)])
        child.assert_not_called()
        calls = []
        def raises_typeerror(timeout, *, absolute_deadline):
            calls.append((timeout, absolute_deadline))
            raise TypeError("CANARY callback body")
        with patch("cineforge.ssh_worker.time.monotonic", return_value=100):
            self.static_failure(lambda: self.driver.run_guarded_native(raises_typeerror,
                                                                      allow_live=True))
        self.assertEqual(calls, [(3, 103)])

    def test_native_late_return_is_incomplete_even_when_callback_returns_a_value(self):
        self.remaining = 123
        now = [100.0]
        calls = []
        def late(timeout, *, absolute_deadline):
            calls.append((timeout, absolute_deadline))
            now[0] = absolute_deadline
            return "late result is not confirmed success"
        with patch("cineforge.ssh_worker.time.monotonic", side_effect=lambda: now[0]):
            self.static_failure(lambda: self.driver.run_guarded_native(late, allow_live=True))
        self.assertEqual(calls, [(3, 103)])

    def test_final_pair_ack_does_not_commit_proof_after_voice_revocation(self):
        def revoke(request):
            if request["operation"] == "s2v_pair_finish":
                self.gate.ledger.revoke(self.gate.utterance.voice.permission.rights, reason="CANARY")
        self.fault = revoke
        self.static_failure(lambda: self.driver.stage_s2v(self.selection))
        self.assertEqual(self.requests[-1]["operation"], "s2v_pair_finish")
        before = len(self.requests)
        self.static_failure(lambda: self.driver.stage_s2v(self.selection))
        self.assertEqual(len(self.requests), before)
        with self.assertRaises(S2VError):
            self.selection.require_execution_worker(worker(self.gate))

    def test_dispatch_checks_guard_and_bound_profile_before_transfer(self):
        (self.remote / "plan.json").write_bytes(canonical(self.plan.to_dict()))
        (self.remote / "profile.json").write_bytes(canonical(self.driver._profile))
        (self.remote / "guard-pid.json").write_bytes(b'{"pid":123}')
        (self.remote / "guard.py").write_bytes(b"synthetic guard fixture")
        request = {"operation": "s2v_pair_begin", "timeout_seconds": 10, "allow_live": True,
            "plan": self.plan.to_dict(), "profile": self.driver._profile,
            "guard_sha256": hashlib.sha256(b"synthetic guard fixture").hexdigest()}
        with patch.object(bootstrap, "worker_root", return_value=self.remote), \
                patch.object(bootstrap, "guard_alive", side_effect=bootstrap.BootstrapError("guard refused")), \
                patch("cineforge.s2v_stage.paired_transfer") as transfer:
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.dispatch(request)
        transfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
