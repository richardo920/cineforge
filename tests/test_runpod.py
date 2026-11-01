"""Offline owned-worker acceptance and uncertain-side-effect recovery probes."""

import copy
import json
import math
import traceback
import unittest
from dataclasses import replace

from cineforge.devices import Backend, GPUDevice, ModelSpec, Precision
from cineforge.provisioning import Provisioner
from cineforge.runpod import (
    AcquisitionIntent,
    AcquisitionUncertainError,
    CleanupUncertainError,
    LeaseSnapshot,
    ProvisioningError,
    RunpodPodProvisioner,
    RunpodPodSpec,
    WorkerObservation,
)
from cineforge.runpod_transport import CloudResponse


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def spec():
    return RunpodPodSpec("fixture-session", "registry.invalid/fixture@sha256:" + "0" * 64,
                         "NVIDIA GeForce RTX 4090", "US-IL-1", "ssh-ed25519 Zml4dHVyZQ==")


def model():
    return ModelSpec("fixture-model", "fixture-v1", "text-to-video",
                     frozenset({Backend.CUDA}), 24082, "fixture-license", Precision.FP16)


class FakeCloud:
    def __init__(self):
        self.calls = []
        self.pods = {}
        self.next_id = 0
        self.status = "RUNNING"
        self.create_status = 201
        self.create_lost = False
        self.get_change = lambda pod: pod
        self.delete_effect = True
        self.delete_lost = False
        self.delete_status = 204
        self.bounds = []

    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        self.calls.append((method, path, copy.deepcopy(body)))
        self.bounds.append((timeout_seconds, max_bytes))
        if method == "POST":
            if self.create_status != 201:
                return CloudResponse(self.create_status, b'{"detail":"private diagnostic"}')
            self.next_id += 1
            identity = "pod_" + str(self.next_id)
            pod = copy.deepcopy(body)
            pod.update(id=identity, status=self.status, locked=False, cost=.89,
                       actions=["stop", "restart", "terminate"],
                       dataCenterId=body["dataCenterIds"][0], runtime={}, cudaVersion="13.2")
            pod["gpu"].update(memory=41, vcpuCount=21)
            self.pods[identity] = pod
            if self.create_lost:
                raise TimeoutError("Bearer fixture-sensitive-value")
            return CloudResponse(201, json.dumps(pod).encode())
        identity = path.rsplit("/", 1)[-1]
        if method == "GET":
            if identity not in self.pods:
                return CloudResponse(404, b'{}')
            return CloudResponse(200, json.dumps(self.get_change(
                copy.deepcopy(self.pods[identity]))).encode())
        if method == "DELETE":
            if self.delete_effect:
                self.pods.pop(identity, None)
            if self.delete_lost:
                raise TimeoutError("Bearer fixture-sensitive-value")
            return CloudResponse(self.delete_status, b'')
        raise AssertionError("unexpected wire operation")


class RunpodLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.cloud, self.events = Clock(), FakeCloud(), []
        self.probes = []
        self.guards = []

    def probe(self, snapshot, timeout):
        self.probes.append((snapshot, timeout))
        return WorkerObservation(GPUDevice(snapshot.resource_id, Backend.CUDA, 24082,
                                          frozenset({Precision.FP16})), spec().gpu_id)

    def provider(self, **overrides):
        options = dict(probe=self.probe, before_create=self.guards.append,
                       event_sink=lambda intent, snapshot: self.events.append((intent, snapshot)),
                       startup_timeout_seconds=3, cleanup_timeout_seconds=2,
                       max_polls=3, clock=self.clock, sleep=self.clock.sleep)
        options.update(overrides)
        return RunpodPodProvisioner(spec(), self.cloud, **options)

    def acquire(self, provider, key="first"):
        return provider.acquire(model(), max_cost_usd=1, idempotency_key=key)

    def test_acquire_probe_inspect_duplicate_and_verified_idempotent_release(self):
        p = self.provider()
        self.assertIsInstance(p, Provisioner)
        device = self.acquire(p)
        self.assertEqual(device.memory_mib, 24082)  # API gpu.memory=41 is host RAM.
        self.assertEqual(len(self.guards), 1)
        self.assertEqual(self.guards[0].spec, spec())
        self.assertEqual(self.events[0][1].state, "reserved")
        self.assertEqual(self.probes[0][0].resource_id, "pod_1")
        body = self.cloud.calls[0][2]
        self.assertEqual(body["mounts"], {})
        self.assertEqual(body["ports"], ["22/tcp"])
        self.assertEqual(body["cloud"], "SECURE")
        self.assertEqual(body["disk"], 80)
        self.assertEqual(body["gpu"]["minCudaVersion"], "12.8")
        self.assertTrue(body["startSsh"])
        self.assertFalse(body["startJupyter"])
        before = len(self.cloud.calls)
        self.assertEqual(self.acquire(p), device)
        self.assertEqual(len(self.cloud.calls), before)
        self.assertEqual(p.inspect("pod_1"), device)
        self.assertEqual(p.inspect_lease("pod_1").hourly_usd, .89)
        p.release("pod_1")
        self.assertEqual(p.snapshots[0].state, "released")
        before = len(self.cloud.calls)
        p.release("pod_1")
        self.assertEqual(len(self.cloud.calls), before)
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)
        self.assertEqual(sum(c[0] == "DELETE" for c in self.cloud.calls), 1)
        self.assertTrue(all(0 < timeout <= 10 and size == 2 * 1024 * 1024
                            for timeout, size in self.cloud.bounds))

    def test_capacity_rejection_is_not_retried_and_key_conflicts_are_rejected(self):
        self.cloud.create_status = 400
        p = self.provider()
        with self.assertRaises(ProvisioningError):
            self.acquire(p)
        self.assertEqual(p.snapshots[0].state, "rejected")
        with self.assertRaises(AcquisitionUncertainError):
            self.acquire(p)
        with self.assertRaises(ProvisioningError):
            p.acquire(model(), max_cost_usd=2, idempotency_key="first")
        self.assertEqual(len(self.cloud.calls), 1)

    def test_uncertain_create_blocks_all_new_creation_until_operator_reconciliation(self):
        self.cloud.create_lost = True
        p = self.provider()
        with self.assertRaises(AcquisitionUncertainError) as cm:
            self.acquire(p)
        self.assertEqual(cm.exception.snapshot.state, "create_uncertain")
        self.assertIsNone(cm.exception.snapshot.resource_id)
        for key in ("first", "different"):
            with self.assertRaises(AcquisitionUncertainError):
                self.acquire(p, key)
        self.assertEqual(len(self.cloud.calls), 1)
        # M3.3 can attach a separately reconciled trusted journal identity in a new
        # process. Cloud ownership is still proved by GET before any deletion.
        restored = self.provider()
        intent = AcquisitionIntent.from_dict(p.intent("first").to_dict())
        snapshot = replace(p.snapshots[0], resource_id="pod_1")
        restored.restore_owned(intent, snapshot)
        restored.release("pod_1")
        self.assertFalse(self.cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)

    def test_restored_created_lease_inspection_emits_ready_after_measured_probe(self):
        p = self.provider()
        self.acquire(p)
        restored = self.provider()
        snapshot = replace(p.snapshots[0], state="created")
        restored.restore_owned(p.intent("first"), snapshot)
        measured = restored.inspect("pod_1")
        self.assertEqual(measured.memory_mib, 24082)
        self.assertEqual(restored.snapshots[0].state, "ready")
        self.assertEqual(self.acquire(restored).device_id, "pod_1")
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)

    def test_startup_timeout_releases_the_known_identity(self):
        self.cloud.status = "STARTING"
        p = self.provider()
        with self.assertRaises(ProvisioningError) as cm:
            self.acquire(p)
        self.assertEqual(cm.exception.snapshot.resource_id, "pod_1")
        self.assertEqual(cm.exception.snapshot.state, "released")
        self.assertEqual(cm.exception.snapshot.reason, "startup_timeout")
        self.assertFalse(self.cloud.pods)
        self.assertFalse(self.probes)

    def test_cancel_before_create_and_after_create_both_fail_safely(self):
        p = self.provider()
        with self.assertRaises(ProvisioningError):
            p.acquire(model(), max_cost_usd=1, idempotency_key="early", cancelled=lambda: True)
        self.assertFalse(self.cloud.calls)
        self.assertEqual(p.snapshots[0].state, "not_created")
        with self.assertRaises(ProvisioningError) as cm:
            p.acquire(model(), max_cost_usd=1, idempotency_key="later",
                      cancelled=lambda: bool(self.cloud.calls))
        self.assertEqual(cm.exception.snapshot.reason, "cancelled")
        self.assertEqual(cm.exception.snapshot.state, "released")
        self.assertFalse(self.cloud.pods)

    def test_probe_failures_and_incompatibilities_cleanup_without_leaking_diagnostics(self):
        def bad_probe(_snapshot, _timeout):
            raise RuntimeError("Bearer fixture-sensitive-value")

        probes = [bad_probe,
                  lambda s, t: WorkerObservation(GPUDevice(s.resource_id, Backend.CUDA, 24081,
                                             frozenset({Precision.FP16})), spec().gpu_id),
                  lambda s, t: WorkerObservation(GPUDevice(s.resource_id, Backend.CUDA, 24082),
                                                 spec().gpu_id),
                  lambda s, t: WorkerObservation(GPUDevice(s.resource_id, Backend.CUDA, 24082,
                                             frozenset({Precision.FP16})), "NVIDIA RTX 5090")]
        for probe in probes:
            with self.subTest(probe=probe):
                self.cloud = FakeCloud()
                p = self.provider(probe=probe)
                with self.assertRaises(ProvisioningError) as cm:
                    self.acquire(p)
                self.assertEqual(cm.exception.snapshot.state, "released")
                self.assertFalse(self.cloud.pods)
                self.assertNotIn("fixture-sensitive", "".join(traceback.format_exception(
                    cm.exception)))
                self.assertIsNone(cm.exception.__context__)

    def test_rate_readback_above_operator_ceiling_cleans_up_before_probe(self):
        def rate(pod):
            pod["cost"] = 1.19
            return pod
        self.cloud.get_change = rate
        p = self.provider()
        with self.assertRaises(ProvisioningError):
            self.acquire(p)
        self.assertFalse(self.cloud.pods)
        self.assertFalse(self.probes)

    def test_missing_or_old_host_cuda_fails_before_runtime_probe_and_cleans_up(self):
        for cuda in (None, "12.7", "bad-version"):
            self.cloud = FakeCloud()
            self.cloud.get_change = lambda p: {**p, "cudaVersion": cuda}
            p = self.provider()
            with self.assertRaises(ProvisioningError):
                self.acquire(p)
            self.assertFalse(self.cloud.pods)
        self.assertFalse(self.probes)

    def test_policy_and_precreate_persistence_failures_never_submit(self):
        def fail(*_args):
            raise RuntimeError("Bearer fixture-sensitive-value")
        for hooks in ({"before_create": fail}, {"event_sink": fail}):
            p = self.provider(**hooks)
            with self.assertRaises(ProvisioningError) as cm:
                self.acquire(p)
            self.assertFalse(self.cloud.calls)
            self.assertEqual(p.snapshots[0].state, "not_created")
            self.assertNotIn("fixture-sensitive", str(cm.exception))

    def test_policy_and_sink_must_complete_the_synchronous_none_contract(self):
        for hooks in ({"before_create": lambda i: False},
                      {"event_sink": lambda i, s: "not persisted"}):
            p = self.provider(**hooks)
            with self.assertRaises(ProvisioningError):
                self.acquire(p)
            self.assertFalse(self.cloud.calls)

    def test_postcreate_persistence_failure_retains_identity_and_attempts_cleanup(self):
        def sink(intent, snapshot):
            self.events.append((intent, snapshot))
            if snapshot.resource_id is not None:
                raise RuntimeError("Bearer fixture-sensitive-value")
        p = self.provider(event_sink=sink)
        with self.assertRaises(CleanupUncertainError) as cm:
            self.acquire(p)
        self.assertEqual(cm.exception.snapshot.resource_id, "pod_1")
        self.assertFalse(self.cloud.pods)  # GET/DELETE still ran despite sink failures.
        self.assertTrue(any(c[0] == "DELETE" for c in self.cloud.calls))
        self.assertFalse(self.probes)

    def test_ready_event_failure_cleans_up_after_a_successful_probe(self):
        def sink(intent, snapshot):
            if snapshot.state == "ready":
                raise RuntimeError("private persistence diagnostic")
            self.events.append((intent, snapshot))
        p = self.provider(event_sink=sink)
        with self.assertRaises(ProvisioningError) as cm:
            self.acquire(p)
        self.assertEqual(len(self.probes), 1)
        self.assertEqual(cm.exception.snapshot.state, "released")
        self.assertFalse(self.cloud.pods)

    def test_transient_startup_inspection_failure_cleans_up_using_fresh_read(self):
        count = 0

        def transient(pod):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError("private cloud diagnostic")
            return pod

        self.cloud.get_change = transient
        p = self.provider()
        with self.assertRaises(ProvisioningError) as cm:
            self.acquire(p)
        self.assertEqual(cm.exception.snapshot.resource_id, "pod_1")
        self.assertEqual(cm.exception.snapshot.state, "released")
        self.assertFalse(self.cloud.pods)

    def test_late_complete_create_response_retains_its_id_for_cleanup(self):
        original = self.cloud.request

        def late(method, path, body=None, **bounds):
            result = original(method, path, body, **bounds)
            if method == "POST":
                self.clock.sleep(4)
            return result

        self.cloud.request = late
        p = self.provider()
        with self.assertRaises(ProvisioningError) as cm:
            self.acquire(p)
        self.assertEqual(cm.exception.snapshot.resource_id, "pod_1")
        self.assertEqual(cm.exception.snapshot.state, "released")
        self.assertFalse(self.cloud.pods)
        self.assertFalse(self.probes)

    def test_non_cuda_requirements_fail_before_paid_create(self):
        p = self.provider()
        with self.assertRaises(ProvisioningError):
            p.acquire(replace(model(), allowed_backends=frozenset({Backend.METAL})),
                      max_cost_usd=1)
        self.assertFalse(self.cloud.calls)

    def test_foreign_or_configuration_drift_never_authorizes_deletion(self):
        for field, value in [("env", {}), ("mounts", {"persistent": {"size": 20}}),
                             ("image", "foreign-image"), ("dataCenterId", "EU-OTHER")]:
            with self.subTest(field=field):
                self.cloud = FakeCloud()
                self.cloud.get_change = lambda p: {**p, field: value}
                p = self.provider()
                with self.assertRaises(CleanupUncertainError) as cm:
                    self.acquire(p)
                self.assertEqual(cm.exception.snapshot.state, "cleanup_uncertain")
                self.assertEqual(sum(c[0] == "DELETE" for c in self.cloud.calls), 0)

    def test_unowned_identity_never_reaches_transport(self):
        p = self.provider()
        for operation in (p.inspect, p.inspect_lease, p.release):
            for identity in ("foreign", "../foreign", "https://credentials@host"):
                with self.assertRaises(ProvisioningError):
                    operation(identity)
        self.assertFalse(self.cloud.calls)

    def test_lost_delete_ack_is_reconciled_by_absence(self):
        p = self.provider()
        self.acquire(p)
        self.cloud.delete_lost = True
        p.release("pod_1")
        self.assertEqual(p.snapshots[0].state, "released")

    def test_unconfirmed_cleanup_retains_owned_id_and_can_be_reconciled_later(self):
        p = self.provider()
        self.acquire(p)
        self.cloud.delete_effect = False
        self.cloud.delete_lost = True
        with self.assertRaises(CleanupUncertainError) as cm:
            p.release("pod_1")
        self.assertEqual(cm.exception.snapshot.resource_id, "pod_1")
        self.assertEqual(cm.exception.snapshot.state, "cleanup_uncertain")
        with self.assertRaises(AcquisitionUncertainError):
            self.acquire(p, "second")
        self.cloud.delete_effect = True
        p.release("pod_1")
        self.assertEqual(p.snapshots[0].state, "released")

    def test_exited_is_not_a_confirmed_resource_deletion(self):
        p = self.provider()
        self.acquire(p)
        self.cloud.delete_effect = False
        self.cloud.pods["pod_1"]["status"] = "EXITED"
        with self.assertRaises(CleanupUncertainError):
            p.release("pod_1")
        self.assertEqual(p.snapshots[0].state, "cleanup_uncertain")

    def test_persisted_records_roundtrip_and_fail_closed_on_changed_spec(self):
        p = self.provider()
        self.acquire(p)
        intent = AcquisitionIntent.from_dict(p.intent("first").to_dict())
        snap = LeaseSnapshot.from_dict(p.snapshots[0].to_dict())
        self.assertEqual(snap, p.snapshots[0])
        changed = intent.to_dict()
        changed["spec"]["gpu_id"] = "NVIDIA GeForce RTX 5090"
        with self.assertRaises(ValueError):
            AcquisitionIntent.from_dict(changed)
        other = RunpodPodProvisioner(replace(spec(), session_id="other-session"), self.cloud,
                                    probe=self.probe, before_create=self.guards.append,
                                    event_sink=lambda i, s: None)
        with self.assertRaises(ProvisioningError):
            other.restore_owned(intent, snap)
        with self.assertRaises(ValueError):
            LeaseSnapshot.from_dict({**snap.to_dict(), "resource_id": "../bad"})

    def test_invalid_operator_bounds_and_nonfinite_costs_are_rejected_offline(self):
        for amount in (True, 0, -1, math.inf, math.nan):
            with self.assertRaises(ProvisioningError):
                self.provider().acquire(model(), max_cost_usd=amount)
        for hooks in ({"before_create": None}, {"event_sink": None}, {"probe": None},
                      {"max_polls": 0}, {"poll_interval_seconds": math.inf}):
            with self.assertRaises(ProvisioningError):
                self.provider(**hooks)
        self.assertFalse(self.cloud.calls)


if __name__ == "__main__":
    unittest.main()
