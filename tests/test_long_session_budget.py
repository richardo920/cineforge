"""Offline long-session scope bounds and retained phase admission ledger."""

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from test_cloud_budget import Clock, Cloud
from test_runpod import spec
from test_speech import EVIDENCE

from cineforge import s2v_profile as profile
from cineforge.cloud_budget import BudgetApproval, CloudBudgetError, CloudBudgetSession
from cineforge.runpod import AcquisitionIntent, LeaseSnapshot
from cineforge.s2v_service import (
    LongSessionBudgetApproval,
    LongSessionServiceContext,
    S2VBudgetApproval,
    S2VServiceContext,
    prepare_long_session_journal,
    prepare_service_journal,
)


class LongSessionBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cf-long-budget-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "journal"

    def approval(self, **changes):
        options = dict(phase_id="phase-1")
        options.update(changes)
        return LongSessionBudgetApproval.for_scope(str(self.path), "EU-RO-1", EVIDENCE, **options)

    def test_scope_and_profile_properties(self):
        approval = self.approval()
        self.assertIsInstance(approval, BudgetApproval)
        self.assertNotIsInstance(approval, S2VBudgetApproval)
        self.assertEqual((approval.aggregate_usd, approval.lifetime_seconds,
                          approval.max_attempts), (35, 50400, 3))
        for gpu in (profile.GPU_ID, profile.L40S_GPU_ID, profile.A40_GPU_ID):
            context = LongSessionServiceContext(self.approval(gpu_id=gpu))
            context.preflight()
            self.assertEqual(context.model, profile.model())
            self.assertEqual(context.declarations, profile.service_declarations(gpu))
            self.assertEqual(context.worker_profile, dict(id=profile.worker(gpu).profile_id,
                scope_sha256=context.approval.subject_sha256))
        self.assertEqual(self.approval(aggregate_usd=65).aggregate_usd, 65)
        for attempts in (1, 2, 3, 4):
            self.assertEqual(self.approval(max_attempts=attempts).max_attempts, attempts)

    def test_closed_bounds_and_tampered_scope(self):
        for changes in (dict(aggregate_usd=65.01), dict(aggregate_usd=0),
                dict(aggregate_usd=True), dict(aggregate_usd=float("nan")),
                dict(lifetime_seconds=50401), dict(lifetime_seconds=120),
                dict(max_attempts=0), dict(max_attempts=5), dict(max_attempts=True),
                dict(hourly_ceiling_usd=1.26), dict(hourly_ceiling_usd=0),
                dict(phase_id=""), dict(phase_id="../phase"), dict(phase_id="a\\b"),
                dict(phase_id="a:b"), dict(phase_id="a%b"), dict(phase_id="a\nb"),
                dict(phase_id=True)):
            with self.subTest(changes=changes), self.assertRaises(CloudBudgetError):
                self.approval(**changes)
        approval = self.approval()
        for changes in (dict(max_concurrent=2), dict(disk_gb=80), dict(image="mutable"),
                dict(min_cuda="12.9"), dict(model_workflow_sha256="a" * 64),
                dict(phase_id="phase-2"), dict(aggregate_usd=34), dict(max_attempts=2),
                dict(lifetime_seconds=3600), dict(region="EU-CZ-1"),
                dict(journal_directory=str(self.path.parent / "other"))):
            with self.subTest(changes=changes), self.assertRaises(CloudBudgetError):
                replace(approval, **changes)

    def test_exact_types_both_directions_and_subclasses(self):
        short = S2VBudgetApproval.for_scope(str(self.path), "EU-RO-1", EVIDENCE)
        long = self.approval()
        for context in (S2VServiceContext(long), LongSessionServiceContext(short)):
            with self.assertRaises(CloudBudgetError):
                context.preflight()
        class DerivedApproval(LongSessionBudgetApproval):
            pass
        derived = DerivedApproval.for_scope(str(self.path), "EU-RO-1", EVIDENCE,
                                            phase_id="phase-1")
        with self.assertRaises(CloudBudgetError):
            LongSessionServiceContext(derived).preflight()
        with self.assertRaises(CloudBudgetError):
            CloudBudgetSession(self.path, derived, None)
        with self.assertRaises(CloudBudgetError):
            prepare_long_session_journal(S2VServiceContext(short))

    def test_journal_initialization_permanently_binds_approval(self):
        context = LongSessionServiceContext(self.approval())
        prepare_long_session_journal(context)
        path = self.path / "budget.json"
        initial = path.read_bytes()
        data = json.loads(initial)
        self.assertEqual(set(data), {"schema_version", "approval_sha256", "last_wall", "records"})
        self.assertEqual(data["approval_sha256"], context.approval.subject_sha256)
        prepare_long_session_journal(context)
        self.assertEqual(path.read_bytes(), initial)
        for changes in (dict(phase_id="phase-2"), dict(aggregate_usd=1)):
            with self.assertRaises(CloudBudgetError):
                prepare_long_session_journal(LongSessionServiceContext(self.approval(**changes)))
            self.assertEqual(path.read_bytes(), initial)
        with self.assertRaises(CloudBudgetError):
            CloudBudgetSession(self.path.parent / "other", context.approval, None)

    def test_unprepared_journal_refused(self):
        with self.assertRaises(CloudBudgetError):
            with CloudBudgetSession(self.path, self.approval(), None):
                self.fail("offline initialization is required")

    def test_sequential_sessions_accumulate_until_phase_budget_stop(self):
        approval = self.approval(aggregate_usd=1)
        context = LongSessionServiceContext(approval)
        prepare_long_session_journal(context)
        clock = Clock()
        clock.wall = time.time() + 1
        cloud = Cloud(clock)
        cloud.catalog.update(id=approval.gpu_id, dataCenters=[dict(id=approval.region,
                             availability="HIGH")])
        worker = replace(spec(), image=approval.image, gpu_id=approval.gpu_id,
            data_center_id=approval.region, disk_gb=150, min_ram_per_gpu_gb=64,
            max_hourly_usd=1.25)
        def session():
            return CloudBudgetSession(self.path, approval, cloud, guard_deployment_configured=True,
                clock=clock.time, monotonic=clock.monotonic)
        def intent(key):
            return AcquisitionIntent(key, worker, "a" * 32, "fixture-" + key,
                                     approval.model, 1, 1200, 3, 1)
        # Two paid admissions consume .423334 each, even when creation never succeeds.
        with session() as first:
            one = intent("first")
            first.event_sink(one, LeaseSnapshot(one.key, "reserved"))
            first.before_create(one)
            first.event_sink(one, LeaseSnapshot(one.key, "not_created"))
        prepare_long_session_journal(context)
        with session() as second:
            two = intent("second")
            second.event_sink(two, LeaseSnapshot(two.key, "reserved"))
            second.before_create(two)
            second.event_sink(two, LeaseSnapshot(two.key, "not_created"))
            three = intent("third")
            second.event_sink(three, LeaseSnapshot(three.key, "reserved"))
            with self.assertRaisesRegex(CloudBudgetError, "aggregate estimated budget"):
                second.before_create(three)
        with session():
            data = json.loads((self.path / "budget.json").read_bytes())
            self.assertEqual(sum(r["authorized"] for r in data["records"].values()), 2)
            self.assertAlmostEqual(sum(r["reservation_usd"] or 0
                for r in data["records"].values()), .846668)
        self.assertEqual(len(cloud.calls), 2)  # Refusal happens before another quote or POST.


    def configured(self, approval):
        clock = Clock()
        clock.wall = time.time() + 1
        cloud = Cloud(clock)
        cloud.catalog.update(id=approval.gpu_id, dataCenters=[dict(id=approval.region,
                             availability="HIGH")])
        worker = replace(spec(), image=approval.image, gpu_id=approval.gpu_id,
            data_center_id=approval.region, disk_gb=150, min_ram_per_gpu_gb=64,
            max_hourly_usd=1.25, retain_node=True)
        return clock, cloud, worker

    def test_attempt_limit_is_journal_wide_across_sessions_not_dollars(self):
        for limit in (1, 2):
            with self.subTest(limit=limit):
                self.path = self.path.parent / ("attempts-" + str(limit))
                approval = self.approval(max_attempts=limit, lifetime_seconds=300)
                prepare_long_session_journal(LongSessionServiceContext(approval))
                clock, cloud, worker = self.configured(approval)
                for index in range(limit + 1):
                    with CloudBudgetSession(self.path, approval, cloud,
                            guard_deployment_configured=True, clock=clock.time,
                            monotonic=clock.monotonic) as session:
                        intent = AcquisitionIntent("attempt-" + str(index), worker, "a" * 32,
                            "fixture-" + str(index), approval.model, 1, 300, 3, 1)
                        session.event_sink(intent, LeaseSnapshot(intent.key, "reserved"))
                        if index == limit:
                            with self.assertRaisesRegex(CloudBudgetError, "durable admission policy"):
                                session.before_create(intent)
                        else:
                            session.before_create(intent)
                            session.event_sink(intent, LeaseSnapshot(intent.key, "not_created"))
                data = json.loads((self.path / "budget.json").read_bytes())
                self.assertEqual(sum(r["authorized"] for r in data["records"].values()), limit)
                self.assertLess(sum(r["reservation_usd"] or 0 for r in data["records"].values()), 1)
                self.assertEqual(len(cloud.calls), limit)

    def test_full_lease_reserve_and_retention_refusal_without_mutation(self):
        approval = self.approval()
        prepare_long_session_journal(LongSessionServiceContext(approval))
        clock, cloud, worker = self.configured(approval)
        with CloudBudgetSession(self.path, approval, cloud, guard_deployment_configured=True,
                clock=clock.time, monotonic=clock.monotonic) as session:
            intent = AcquisitionIntent("full", worker, "a" * 32, "fixture-full",
                approval.model, 17.78, 50400, 3, 1)
            session.event_sink(intent, LeaseSnapshot(intent.key, "reserved"))
            session.before_create(intent)
            session.event_sink(intent, LeaseSnapshot(intent.key, "created", "pod-full", .89))
            before = (self.path / "budget.json").read_bytes()
            self.assertEqual(json.loads(before)["records"]["full"]["reservation_usd"], 17.78)
            with self.assertRaisesRegex(CloudBudgetError, "sized at acquisition"):
                session.extend_retention("full")
            self.assertEqual((self.path / "budget.json").read_bytes(), before)

    def test_short_s2v_lease_still_extends_retention(self):
        approval = S2VBudgetApproval.for_scope(str(self.path), "EU-RO-1", EVIDENCE,
                                              lifetime_seconds=7200)
        prepare_service_journal(S2VServiceContext(approval))
        clock, cloud, worker = self.configured(approval)
        worker = replace(worker, retain_window_seconds=10000)
        with CloudBudgetSession(self.path, approval, cloud, guard_deployment_configured=True,
                clock=clock.time, monotonic=clock.monotonic) as session:
            intent = AcquisitionIntent("short", worker, "a" * 32, "fixture-short",
                approval.model, 3, 7200, 3, 1)
            session.event_sink(intent, LeaseSnapshot(intent.key, "reserved"))
            session.before_create(intent)
            session.event_sink(intent, LeaseSnapshot(intent.key, "created", "pod-short", .89))
            original = session.guard_plan("short").stop_at_unix
            session.extend_retention("short")
            self.assertGreater(session.guard_plan("short").stop_at_unix, original)

    def test_initializer_types_and_cross_scope_sha_both_directions(self):
        for long_first in (True, False):
            with self.subTest(long_first=long_first):
                self.path = self.path.parent / ("cross-" + str(long_first))
                long = LongSessionServiceContext(self.approval())
                short = S2VServiceContext(S2VBudgetApproval.for_scope(str(self.path),
                    "EU-RO-1", EVIDENCE))
                with self.assertRaises(CloudBudgetError):
                    prepare_service_journal(long)
                with self.assertRaises(CloudBudgetError):
                    prepare_long_session_journal(short)
                first, second = ((prepare_long_session_journal, long), (prepare_service_journal, short))
                if not long_first:
                    first, second = second, first
                first[0](first[1])
                before = (self.path / "budget.json").read_bytes()
                with self.assertRaisesRegex(CloudBudgetError, "different scope"):
                    second[0](second[1])
                self.assertEqual((self.path / "budget.json").read_bytes(), before)
