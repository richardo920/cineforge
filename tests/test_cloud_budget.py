"""Offline admission, durable recovery and crash-window acceptance probes."""

import copy
import hashlib
import json
import os
import tempfile
import traceback
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_runpod import FakeCloud, spec

from cineforge.cloud_budget import BudgetApproval, CloudBudgetError, CloudBudgetSession
from cineforge.cloud_journal import CloudJournal, CloudJournalError
from cineforge.comfy_policy import load_comfy_selection
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.runpod import (
    AcquisitionUncertainError,
    CleanupUncertainError,
    ProvisioningError,
    RunpodPodProvisioner,
    WorkerObservation,
)
from cineforge.runpod_transport import CloudResponse
from cineforge.worker_guard import GuardReceipt, worker_guard_source_sha256
from cineforge.workflows import canonical


def selection():
    return load_comfy_selection(Path(__file__).resolve().parents[1]
                                / "config/comfy/wan21-t2v-1.3b.json")


def model():
    return selection().model


def model_subject():
    return selection().declarations["approval"]["model_workflow"]["approved_subject_sha256"]


class Clock:
    def __init__(self):
        self.wall = 10000.
        self.mono = 100.

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, seconds):
        self.wall += seconds
        self.mono += seconds


def approval_fixture(**changes):
    subject = dict(provider="runpod", product="pod", cloud="SECURE", gpu_id=spec().gpu_id,
        gpu_count=1, region="US-IL-1", region_fallback=False, aggregate_limit_usd=3,
        max_concurrent_pods=1, max_create_attempts=2, worker_lifetime_limit_seconds=1800,
        compute_price_ceiling_usd_per_hour=1, container_disk_gb=80, persistent_volume_gb=0,
        network_volume_ids=[], image=spec().image, min_host_cuda="12.8", ports=["22/tcp"],
        startup_and_downloads_count_toward_lifetime=True, model_workflow_subject_sha256=model_subject(),
        synthetic_data_only=True)
    subject.update(changes)
    try:
        digest = hashlib.sha256(canonical(subject)).hexdigest()
    except ValueError:
        digest = "0"*64
    proposal = dict(state="owner_scope_approved",
        recommendation={k: subject[k] for k in ("provider", "product", "cloud")},
        approval=dict(m3_live_spend=dict(status="approved", aggregate_limit_usd=subject[
            "aggregate_limit_usd"], approved_subject=subject, approved_subject_sha256=digest,
            receipt="https://github.com/richardo920/cineforge/issues/48#issuecomment-1")),
        proposed_validation=dict(subject, actual_worker_verification_required=True))
    return proposal, digest


def approval(**changes):
    proposal, digest = approval_fixture(**changes)
    return BudgetApproval.from_proposal(proposal, expected_subject_sha256=digest,
        expected_model_workflow_sha256=model_subject(), model=model(), selection=selection())


class Cloud(FakeCloud):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.catalog = dict(id=spec().gpu_id, secure=True, price=dict(secure=.89),
            availability="HIGH", maxCount=dict(secure=8), memory=24,
            dataCenters=[dict(id="US-IL-1", availability="LOW")],
            cudaVersions=[dict(version="13.2", available=True)])
        self.quote_delay = 0
        self.pages = None
        self.list_error = False

    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        if path.startswith("/v2/catalog/"):
            self.calls.append((method, path, body))
            self.clock.sleep(self.quote_delay)
            return CloudResponse(200, canonical(self.catalog))
        if path.startswith("/v2/pods?"):
            self.calls.append((method, path, body))
            if self.list_error:
                raise TimeoutError("Bearer fixture-sensitive-value")
            page = self.pages.pop(0) if self.pages else dict(pods=list(self.pods.values()),
                pagination=dict(nextCursor=None, hasNextPage=False))
            return CloudResponse(200, canonical(page))
        return super().request(method, path, body, timeout_seconds=timeout_seconds,
                               max_bytes=max_bytes)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-budget-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "journal"
        self.clock = Clock()
        self.cloud = Cloud(self.clock)

    def session(self, **options):
        kwargs = dict(guard_deployment_configured=True, clock=self.clock.time,
                      monotonic=self.clock.monotonic)
        kwargs.update(options)
        return CloudBudgetSession(self.path, approval(), self.cloud, **kwargs)

    def provider(self, session, **options):
        def probe(snapshot, timeout):
            session.arm_guard(snapshot.key, GuardReceipt(session.guard_plan(snapshot.key),
                worker_guard_source_sha256(), 123, self.clock.time(), True, True))
            return WorkerObservation(GPUDevice(snapshot.resource_id, Backend.CUDA, 24082,
                frozenset({Precision.FP16})), spec().gpu_id)
        kwargs = dict(probe=probe, before_create=session.before_create,
            event_sink=session.event_sink, clock=self.clock.monotonic, sleep=self.clock.sleep,
            startup_timeout_seconds=3, cleanup_timeout_seconds=2, max_polls=3)
        kwargs.update(options)
        return RunpodPodProvisioner(spec(), self.cloud, **kwargs)

    def acquire(self, provider, key="first"):
        return provider.acquire(model(), max_cost_usd=1, idempotency_key=key)

    def test_reserves_before_post_identity_before_probe_and_unknown_invoice_is_retained(self):
        with self.session() as session:
            def probe(snapshot, timeout):
                committed = json.loads((self.path / "budget.json").read_bytes())
                record = committed["records"][snapshot.key]
                self.assertTrue(record["authorized"])
                self.assertEqual(record["snapshot"]["resource_id"], "pod_1")
                session.arm_guard(snapshot.key, GuardReceipt(session.guard_plan(snapshot.key),
                    worker_guard_source_sha256(), 123, self.clock.time(), True, True))
                return WorkerObservation(GPUDevice("pod_1", Backend.CUDA, 24082,
                    frozenset({Precision.FP16})), spec().gpu_id)
            provider = self.provider(session, probe=probe)
            self.acquire(provider)
            provider.release("pod_1")
            costs = session.costs()
            self.assertEqual(costs["attempts"], 1)
            self.assertEqual(costs["reserved_worst_case_usd"], .51)
            self.assertEqual(costs["invoice_unknown_keys"], ["first"])
            self.assertFalse(costs["hard_account_spend_cap"])
            self.assertIsNone(costs["actual_total_usd"])
            self.assertLess(costs["estimated_usd"], costs["reserved_worst_case_usd"])
            self.assertEqual(self.cloud.calls[0][0], "GET")
            self.assertEqual(self.cloud.calls[1][0], "POST")

    def test_scope_digest_and_operational_comparison_and_strict_types(self):
        proposal, digest = approval_fixture()
        proposal["proposed_validation"]["region"] = "foreign-region"
        with self.assertRaises(CloudBudgetError):
            BudgetApproval.from_proposal(proposal, expected_subject_sha256=digest,
                expected_model_workflow_sha256=model_subject(), model=model(), selection=selection())
        for value in (True, float("nan"), float("inf"), -1, 4):
            with self.subTest(value=value), self.assertRaises(CloudBudgetError):
                approval(aggregate_limit_usd=value)
        for field, value in (("gpu_count", True), ("max_create_attempts", True),
                             ("container_disk_gb", 80.), ("region_fallback", 0)):
            with self.subTest(field=field), self.assertRaises(CloudBudgetError):
                approval(**{field: value})

    def test_no_guard_integration_no_paid_post(self):
        with self.session(guard_deployment_configured=False) as session:
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session))
            self.assertFalse(any(c[0] == "POST" for c in self.cloud.calls))
            self.assertEqual(session.costs()["attempts"], 0)

    def test_stale_quote_incompatible_cuda_missing_region_and_excess_rate_are_declined(self):
        variants = [dict(price=dict(secure=1.01)), dict(price=dict(secure=True)),
                    dict(cudaVersions=[dict(version="12.7", available=True)]),
                    dict(cudaVersions=[dict(version="13.2", available=False)]),
                    dict(dataCenters=[]), dict(secure=False), dict(maxCount=dict(secure=True))]
        for index, variant in enumerate(variants):
            with self.subTest(variant=variant):
                self.path = Path(self.temp.name) / str(index)
                original = copy.deepcopy(self.cloud.catalog)
                self.cloud.catalog.update(variant)
                with self.session() as session:
                    with self.assertRaises(ProvisioningError):
                        self.acquire(self.provider(session))
                self.cloud.catalog = original
        self.path = Path(self.temp.name) / "stale"
        self.cloud.quote_delay = 31
        with self.session() as session:
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session))
        self.assertFalse(any(c[0] == "POST" for c in self.cloud.calls))

    def test_budget_decline_precedes_quote_and_post(self):
        with self.session() as session:
            provider = self.provider(session)
            with self.assertRaises(ProvisioningError):
                provider.acquire(model(), max_cost_usd=.1, idempotency_key="small")
            self.assertEqual(self.cloud.calls, [])

    def test_guard_missing_or_bad_receipt_releases_worker_before_ready(self):
        with self.session() as session:
            provider = self.provider(session, probe=lambda snapshot, timeout: WorkerObservation(
                GPUDevice(snapshot.resource_id, Backend.CUDA, 24082, frozenset({Precision.FP16})),
                spec().gpu_id))
            with self.assertRaises(ProvisioningError):
                self.acquire(provider)
            self.assertFalse(self.cloud.pods)
            self.assertEqual(provider.snapshots[0].state, "released")

    def test_wrong_guard_source_plan_receipt_or_expired_arming_are_declined(self):
        with self.session() as session:
            def probe(snapshot, timeout):
                good = GuardReceipt(session.guard_plan(snapshot.key),
                    worker_guard_source_sha256(), 123, self.clock.time(), True, True)
                for bad in (replace(good, source_sha256="f"*64),
                            replace(good, plan=replace(good.plan, pod_id="foreign")),
                            replace(good, observed_unix=self.clock.time()-31)):
                    with self.assertRaises(CloudBudgetError):
                        session.arm_guard(snapshot.key, bad)
                session.arm_guard(snapshot.key, good)
                return WorkerObservation(GPUDevice(snapshot.resource_id, Backend.CUDA, 24082,
                    frozenset({Precision.FP16})), spec().gpu_id)
            self.acquire(self.provider(session, probe=probe))

    def test_attempt_slots_never_reset_after_zero_final_invoice_or_restart(self):
        with self.session() as session:
            provider = self.provider(session)
            for key in ("one", "two"):
                self.acquire(provider, key)
                provider.release("pod_" + str(self.cloud.next_id))
                session.record_actual(key, 0, billing_reference="billing:fixture",
                                      authoritative_final=True)
            self.assertEqual(session.costs()["attempts"], 2)
        with self.session() as session:
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session), "third")
            self.assertEqual(session.costs()["attempts"], 2)
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 2)

    def test_nonfinal_or_estimated_costs_cannot_release_reservation(self):
        with self.session() as session:
            provider = self.provider(session)
            self.acquire(provider)
            with self.assertRaises(CloudBudgetError):
                session.record_actual("first", 0, billing_reference="billing:fixture",
                                      authoritative_final=True)
            provider.release("pod_1")
            for amount, reference, final in ((0, "billing:fixture", False),
                (float("nan"), "billing:fixture", True), (True, "billing:fixture", True),
                (0, "elapsed:120", True)):
                with self.assertRaises(CloudBudgetError):
                    session.record_actual("first", amount, billing_reference=reference,
                                          authoritative_final=final)
            self.assertEqual(session.costs()["invoice_unknown_keys"], ["first"])

    def test_new_spend_blocked_by_active_or_ambiguous_create_and_no_blind_recreation(self):
        self.cloud.create_lost = True
        with self.session() as session:
            with self.assertRaises(AcquisitionUncertainError):
                self.acquire(self.provider(session))
        with self.session() as session:
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session), "different")
            self.assertEqual(session.recover_owned(self.provider(session)), ("first",))
            self.assertFalse(self.cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)

    def test_empty_listing_and_partial_failure_remain_uncertain(self):
        self.cloud.create_lost = True
        with self.session() as session:
            with self.assertRaises(AcquisitionUncertainError):
                self.acquire(self.provider(session))
        owned = copy.deepcopy(self.cloud.pods["pod_1"])
        self.cloud.pods.clear()
        for pages in (None, [dict(pods=[owned], pagination=dict(nextCursor="more", hasNextPage=True)),
                             dict(pods=[], pagination=dict(nextCursor="more", hasNextPage=True))]):
            with self.session() as session:
                self.cloud.pages = pages
                self.assertEqual(session.recover_owned(self.provider(session), max_pages=2), ())
                self.assertEqual(session.costs()["invoice_unknown_keys"], ["first"])
                with self.assertRaises(ProvisioningError):
                    self.acquire(self.provider(session), "different")
        self.assertFalse(any(c[0] == "DELETE" for c in self.cloud.calls))

    def test_paginated_recovery_skips_foreign_resource_and_verifies_exact_owned_worker(self):
        self.cloud.create_lost = True
        with self.session() as session:
            with self.assertRaises(AcquisitionUncertainError):
                self.acquire(self.provider(session))
        own = self.cloud.pods["pod_1"]
        foreign = copy.deepcopy(own)
        foreign["id"] = "foreign"
        foreign["env"]["CINEFORGE_LEASE"] = "1"*32
        self.cloud.pods["foreign"] = foreign
        self.cloud.pages = [dict(pods=[foreign], pagination=dict(nextCursor="next", hasNextPage=True)),
            dict(pods=[own], pagination=dict(nextCursor=None, hasNextPage=False))]
        with self.session() as session:
            self.assertEqual(session.recover_owned(self.provider(session)), ("first",))
        self.assertEqual(set(self.cloud.pods), {"foreign"})

    def test_known_id_recovery_after_crash_uses_no_list_or_create(self):
        with self.session() as session:
            self.acquire(self.provider(session))
        initial_posts = sum(c[0] == "POST" for c in self.cloud.calls)
        with self.session() as session:
            self.assertEqual(session.recover_owned(self.provider(session)), ("first",))
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), initial_posts)
        self.assertFalse(any("?limit" in c[1] for c in self.cloud.calls))

    def test_cleanup_uncertainty_keeps_identity_reservation_and_blocks_other_spend(self):
        with self.session() as session:
            provider = self.provider(session)
            self.acquire(provider)
            self.cloud.delete_effect = False
            with self.assertRaises(CleanupUncertainError):
                provider.release("pod_1")
            self.assertEqual(provider.snapshots[0].resource_id, "pod_1")
        with self.session() as session:
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session), "new")
            self.assertEqual(session.costs()["attempts"], 1)
            self.cloud.delete_effect = True
            session.recover_owned(self.provider(session))

    def test_deadline_idle_frozen_wall_and_clock_rollback_trigger_owned_cleanup(self):
        for kind in ("deadline", "idle", "frozen", "rollback"):
            with self.subTest(kind=kind):
                self.path = Path(self.temp.name) / kind
                with self.session() as session:
                    provider = self.provider(session)
                    self.acquire(provider, kind)
                    if kind == "deadline":
                        self.clock.sleep(1801)
                        session.heartbeat(kind)
                    elif kind == "idle":
                        self.clock.sleep(121)
                    elif kind == "frozen":
                        self.clock.mono += 1801
                    else:
                        self.clock.wall -= 1
                    self.assertEqual(session.enforce_limits(provider), (kind,))
                    self.assertFalse(self.cloud.pods)

    def test_cleanup_starts_within_lifetime_and_remaining_time_never_extends(self):
        with self.session() as session:
            provider = self.provider(session)
            self.acquire(provider)
            self.assertEqual(session.remaining_seconds("first"), 1740)
            self.clock.sleep(1740)
            self.assertEqual(session.remaining_seconds("first"), 0)
            self.assertEqual(session.enforce_limits(provider), ("first",))
            self.assertLess(self.clock.time(), 10000 + 1800)

    def test_clock_rollback_after_restart_blocks_admission_but_allows_cleanup(self):
        with self.session() as session:
            self.acquire(self.provider(session))
            self.clock.sleep(5)
            session.heartbeat("first")
        self.clock.wall -= 10
        with self.session() as session:
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session), "new")
            session.recover_owned(self.provider(session))
        self.assertFalse(self.cloud.pods)

    def test_checkpoint_failure_before_post_poisoned_no_create(self):
        with self.session() as session:
            original = session._journal.checkpoint
            count = 0
            def fail_reservation(payload):
                nonlocal count
                count += 1
                if count >= 2:
                    raise OSError("Bearer fixture-sensitive-value")
                original(payload)
            with patch.object(session._journal, "checkpoint", side_effect=fail_reservation):
                with self.assertRaises(ProvisioningError) as error:
                    self.acquire(self.provider(session))
            self.assertNotIn("fixture-sensitive-value", "".join(traceback.format_exception(error.exception)))
            self.assertFalse(any(c[0] == "POST" for c in self.cloud.calls))

    def test_reservation_before_post_crash_stays_uncertain_without_atomic_create_proof(self):
        with self.session() as session:
            provider = self.provider(session)
            def crash(intent):
                session.before_create(intent)
                raise SystemExit
            provider._before = crash
            with self.assertRaises(SystemExit):
                self.acquire(provider)
        with self.session() as session:
            self.assertEqual(session.recover_owned(self.provider(session)), ())
            self.assertEqual(session.costs()["attempts"], 1)
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session), "new")
        self.assertFalse(any(c[0] == "POST" for c in self.cloud.calls))

    def test_aggregate_cap_rejects_second_lease_after_expensive_final_invoice(self):
        with self.session() as session:
            provider = self.provider(session)
            self.acquire(provider)
            provider.release("pod_1")
            session.record_actual("first", 2.8, billing_reference="billing:fixture",
                                  authoritative_final=True)
            calls = len(self.cloud.calls)
            with self.assertRaises(ProvisioningError):
                self.acquire(provider, "second")
            self.assertEqual(len(self.cloud.calls), calls)

    def test_cancellation_after_reservation_never_posts_but_keeps_attempt_and_reserve(self):
        with self.session() as session:
            provider = self.provider(session)
            with self.assertRaises(ProvisioningError):
                provider.acquire(model(), max_cost_usd=1, idempotency_key="cancelled",
                                 cancelled=lambda: True)
            self.assertFalse(any(c[0] == "POST" for c in self.cloud.calls))
            self.assertEqual(session.costs()["attempts"], 1)
            self.assertEqual(session.costs()["reserved_worst_case_usd"], .51)

    def test_identity_checkpoint_failure_causes_owned_cleanup(self):
        with self.session() as session:
            original = session._journal.checkpoint
            failed = False
            def fail_created_once(payload):
                nonlocal failed
                records = json.loads(payload)["records"]
                if not failed and any(r["snapshot"]["state"] == "created" for r in records.values()):
                    failed = True
                    raise OSError("fixture-sensitive-value")
                original(payload)
            with patch.object(session._journal, "checkpoint", side_effect=fail_created_once):
                with self.assertRaises(ProvisioningError):
                    self.acquire(self.provider(session))
            self.assertFalse(self.cloud.pods)
            self.assertEqual(json.loads((self.path / "budget.json").read_bytes())["records"][
                "first"]["snapshot"]["resource_id"], "pod_1")

    def test_duplicate_matching_discovery_preserves_all_ids_without_blind_delete(self):
        self.cloud.create_lost = True
        with self.session() as session:
            with self.assertRaises(AcquisitionUncertainError):
                self.acquire(self.provider(session))
        duplicate = copy.deepcopy(self.cloud.pods["pod_1"])
        duplicate["id"] = "pod_duplicate"
        self.cloud.pods["pod_duplicate"] = duplicate
        with self.session() as session:
            self.assertEqual(session.recover_owned(self.provider(session)), ())
            record = json.loads((self.path / "budget.json").read_bytes())["records"]["first"]
            self.assertEqual(record["discovered_ids"], ["pod_1", "pod_duplicate"])
            with self.assertRaises(ProvisioningError):
                self.acquire(self.provider(session), "new")
        self.assertFalse(any(c[0] == "DELETE" for c in self.cloud.calls))

    def test_known_foreign_configuration_is_never_deleted(self):
        with self.session() as session:
            self.acquire(self.provider(session))
        self.cloud.pods["pod_1"]["env"]["CINEFORGE_LEASE"] = "1"*32
        with self.session() as session:
            with self.assertRaises(CleanupUncertainError):
                session.recover_owned(self.provider(session))
            self.assertEqual(session.costs()["invoice_unknown_keys"], ["first"])
        self.assertFalse(any(c[0] == "DELETE" for c in self.cloud.calls))

    def test_corrupt_budget_records_and_approval_mismatch_refuse_resume(self):
        with self.session() as session:
            self.acquire(self.provider(session))
        original = (self.path / "budget.json").read_bytes()
        for mutation in (lambda d: d.update(approval_sha256="f"*64),
                lambda d: d["records"]["first"].update(reservation_usd=0),
                lambda d: d["records"]["first"].update(deadline=9999999999),
                lambda d: d["records"]["first"].update(authorized=1)):
            data = json.loads(original)
            mutation(data)
            (self.path / "budget.json").write_bytes(canonical(data))
            with self.assertRaises(CloudBudgetError):
                with self.session():
                    pass
        (self.path / "budget.json").write_bytes(original)
        with self.session() as session:
            session.recover_owned(self.provider(session))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-journal-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state"

    def test_lock_atomic_resume_private_permissions_and_duplicate_writer(self):
        with CloudJournal(self.path) as journal:
            journal.checkpoint(b'{"sequence":1}')
            with self.assertRaises(CloudJournalError):
                with CloudJournal(self.path):
                    pass
            self.assertEqual((self.path / "budget.json").stat().st_mode & 0o777, 0o600)
        with CloudJournal(self.path) as journal:
            self.assertEqual(journal.initial, b'{"sequence":1}')
            journal.checkpoint(b'{"sequence":2}')
        self.assertEqual((self.path / "budget.json").read_bytes(), b'{"sequence":2}')

    def test_symlink_hardlink_git_paths_and_interrupted_initialization_are_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir(mode=0o700)
        self.path.symlink_to(outside)
        with self.assertRaises(CloudJournalError):
            with CloudJournal(self.path):
                pass
        self.path.unlink()
        self.path.mkdir(mode=0o700)
        (self.path / ".lock").touch(mode=0o600)
        with self.assertRaises(CloudJournalError):
            with CloudJournal(self.path):
                pass
        (self.path / "budget.json").write_bytes(b'{}')
        (self.path / "budget.json").chmod(0o600)
        os.link(self.path / "budget.json", outside / "copy")
        with self.assertRaises(CloudJournalError):
            with CloudJournal(self.path):
                pass
        git = outside / ".git"
        git.mkdir()
        with self.assertRaises(CloudJournalError):
            with CloudJournal(outside / "journal"):
                pass

    def test_record_replacement_and_permissions_fail_closed(self):
        with CloudJournal(self.path) as journal:
            journal.checkpoint(b'{}')
            record = self.path / "budget.json"
            replacement = self.path / ".pending-fixture"
            replacement.write_bytes(b'{}')
            replacement.chmod(0o600)
            replacement.replace(record)
            with self.assertRaises(CloudJournalError):
                journal.checkpoint(b'{"new":true}')
