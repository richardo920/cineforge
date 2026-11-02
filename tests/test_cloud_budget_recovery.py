"""Offline cleanup-batch control flow, not multi-Pod admission evidence."""

import tempfile
import threading
import traceback
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_cloud_budget as fixtures
from test_runpod import spec

from cineforge.cloud_budget import CloudBudgetError, CloudBudgetSession
from cineforge.runpod import (
    AcquisitionIntent,
    CleanupUncertainError,
    LeaseSnapshot,
    ProvisioningError,
    RunpodPodProvisioner,
)


class ReleaseProbe:
    def __init__(self, session, failures):
        self.session = session
        self.failures = failures
        self.restored = {}
        self.attempts = []

    @property
    def snapshots(self):
        return tuple(snapshot for _, snapshot in self.restored.values())

    def restore_owned(self, intent, snapshot):
        self.restored[snapshot.resource_id] = (intent, snapshot)

    def release(self, identity):
        self.attempts.append(identity)
        intent, snapshot = self.restored[identity]
        if identity in self.failures:
            raise self.failures[identity]("private-release-canary")
        self.session.event_sink(intent, replace(snapshot, state="released"))


class CleanupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-cleanup-batch-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / "journal"
        self.clock = fixtures.Clock()
        self.cloud = fixtures.Cloud(self.clock)

    def session(self):
        return CloudBudgetSession(self.path, fixtures.approval(), self.cloud,
            guard_deployment_configured=True, clock=self.clock.time,
            monotonic=self.clock.monotonic)

    def seed(self, session, key, *, authorized=True, known=True):
        intent = AcquisitionIntent(key, spec(), key[-1] * 32, "fixture-" + key,
                                   fixtures.model(), 1, 1800, 3, 1)
        session.event_sink(intent, LeaseSnapshot(key, "reserved"))
        if authorized:
            session.before_create(intent)
        if known:
            session.event_sink(intent, LeaseSnapshot(key, "created", "pod_" + key))
        return intent

    def inject_three_claims(self, session):
        # Deliberate control-flow fixture: admission permits only one active Pod.
        # Copy one valid reservation to inject three cleanup records, never POST.
        first = self.seed(session, "claim1")
        for digit in ("2", "3"):
            intent = replace(first, key="claim" + digit, owner_token=digit * 32,
                             name="fixture-claim" + digit)
            record = dict(session._data["records"][first.key])
            record.update(intent=intent.to_dict(), snapshot=LeaseSnapshot(
                intent.key, "created", "pod_" + intent.key).to_dict())
            session._data["records"][intent.key] = record
        session._save()

    def test_first_failed_release_does_not_strand_later_claims_or_future_recovery(self):
        for operation in ("enforce_limits", "recover_owned"):
            with self.subTest(operation=operation):
                self.path = Path(self.temp.name).resolve() / operation
                with self.session() as session:
                    self.inject_three_claims(session)
                    self.clock.sleep(121)
                    provider = ReleaseProbe(session, {"pod_claim1": TimeoutError})
                    with self.assertRaises(CleanupUncertainError) as error:
                        getattr(session, operation)(provider)
                    self.assertEqual(provider.attempts,
                                     ["pod_claim1", "pod_claim2", "pod_claim3"])
                    self.assertEqual(error.exception.released_keys, ("claim2", "claim3"))
                    self.assertEqual(error.exception.uncertain_keys, ("claim1",))
                    self.assertEqual(session._cleanup_inflight, set())
                    self.assertEqual([s.state for s in session.snapshots],
                                     ["cleanup_uncertain", "released", "released"])
                    self.assertEqual(session.costs()["invoice_unknown_keys"],
                                     ["claim1", "claim2", "claim3"])
                    provider.failures.clear()
                    self.assertEqual(session.recover_owned(provider), ("claim1",))
                    self.assertTrue(all(s.state == "released" for s in session.snapshots))

    def test_multiple_failures_and_interruption_are_aggregated_without_private_chains(self):
        with self.session() as session:
            self.inject_three_claims(session)
            provider = ReleaseProbe(session, {"pod_claim1": TimeoutError,
                                              "pod_claim3": KeyboardInterrupt})
            with self.assertRaises(CleanupUncertainError) as error:
                session.recover_owned(provider)
            self.assertEqual(provider.attempts, ["pod_claim1", "pod_claim2", "pod_claim3"])
            self.assertEqual(error.exception.released_keys, ("claim2",))
            self.assertEqual(error.exception.uncertain_keys, ("claim1", "claim3"))
            self.assertEqual(session._cleanup_inflight, set())
            self.assertIsNone(error.exception.__context__)
            self.assertNotIn("private-release-canary",
                             "".join(traceback.format_exception(error.exception)))

    def test_unauthorized_reservation_is_terminal_after_restart_without_any_network(self):
        with self.session() as session:
            original = self.seed(session, "claim1", authorized=False, known=False)
        with self.session() as session:
            self.cloud.list_error = True
            self.assertEqual(session.recover_owned(ReleaseProbe(session, {})), ())
            self.assertEqual(session.snapshots[0].state, "not_created")
            self.assertEqual(self.cloud.calls, [])
            with self.assertRaises(CloudBudgetError):
                session.before_create(original)
            self.assertEqual(self.cloud.calls, [])
            self.assertEqual(session.costs()["attempts"], 0)
            self.assertEqual(session.costs()["reserved_worst_case_usd"], 0)
            self.cloud.list_error = False
            admitted = self.seed(session, "claim2", known=False)
            self.assertEqual(session.costs()["attempts"], 1)
            self.assertEqual(admitted.key, "claim2")
        with self.session() as session:
            self.assertEqual(session.snapshots[0].state, "not_created")
            self.assertEqual(session.costs()["attempts"], 1)

    def test_retirement_prevents_a_paused_acquirer_from_authorizing_or_posting_later(self):
        reserved, resume = threading.Event(), threading.Event()
        errors = []
        with self.session() as session:
            def delayed_before_create(intent):
                reserved.set()
                if not resume.wait(3):
                    raise AssertionError("acquirer was not resumed")
                session.before_create(intent)
            provider = RunpodPodProvisioner(spec(), self.cloud, probe=lambda *_: None,
                before_create=delayed_before_create, event_sink=session.event_sink,
                clock=self.clock.monotonic, sleep=self.clock.sleep)
            def acquire():
                try:
                    provider.acquire(fixtures.model(), max_cost_usd=1, idempotency_key="claim1")
                except ProvisioningError:
                    errors.append("denied")
                except BaseException as error:
                    errors.append(type(error).__name__)
            acquirer = threading.Thread(target=acquire, daemon=True)
            acquirer.start()
            try:
                self.assertTrue(reserved.wait(3))
                self.assertEqual(session.recover_owned(ReleaseProbe(session, {})), ())
                self.assertEqual(session.snapshots[0].state, "not_created")
            finally:
                resume.set()
                acquirer.join(3)
            self.assertFalse(acquirer.is_alive())
            self.assertEqual(errors, ["denied"])
            self.assertEqual(self.cloud.calls, [])
            self.assertEqual(session.costs()["attempts"], 0)

    def test_legacy_discovery_did_not_authorize_an_unposted_reservation(self):
        with self.session() as session:
            intent = self.seed(session, "claim1", authorized=False, known=False)
            # Prior recovery changed a never-authorized reservation to uncertain.
            session.event_sink(intent, LeaseSnapshot(intent.key, "create_uncertain",
                                                      reason="transport"))
        with self.session() as session:
            self.cloud.list_error = True
            self.assertEqual(session.recover_owned(ReleaseProbe(session, {})), ())
            self.assertEqual(session.snapshots[0].state, "not_created")
            self.assertEqual(self.cloud.calls, [])
            with self.assertRaises(CloudBudgetError):
                session.before_create(intent)

    def test_authorized_ambiguous_post_keeps_reservation_and_blocks_new_admission(self):
        with self.session() as session:
            self.seed(session, "claim1", known=False)
        with self.session() as session:
            self.assertEqual(session.recover_owned(ReleaseProbe(session, {})), ())
            self.assertEqual(session.snapshots[0].state, "create_uncertain")
            self.assertEqual(session.costs()["invoice_unknown_keys"], ["claim1"])
            self.assertEqual(session.costs()["attempts"], 1)
            self.assertTrue(any(path.startswith("/v2/pods?") for _, path, _ in self.cloud.calls))
            with self.assertRaises(CloudBudgetError):
                self.seed(session, "claim2", known=False)

    def test_later_discovery_checkpoint_failure_still_releases_prior_claim(self):
        with self.session() as session:
            self.seed(session, "claim1")
            first = session._data["records"]["claim1"]
            intent = replace(AcquisitionIntent.from_dict(first["intent"]), key="claim2",
                             owner_token="2" * 32, name="fixture-claim2")
            second = dict(first, intent=intent.to_dict(),
                          snapshot=LeaseSnapshot("claim2", "reserved").to_dict())
            session._data["records"]["claim2"] = second
            session._save()
            provider = ReleaseProbe(session, {})
            original = session._journal.checkpoint
            def fail_discovery(payload):
                if b'"state":"create_uncertain"' in payload:
                    raise OSError("private-checkpoint-canary")
                original(payload)
            with patch.object(session._journal, "checkpoint", side_effect=fail_discovery):
                with self.assertRaises(CleanupUncertainError):
                    session.recover_owned(provider)
            self.assertEqual(provider.attempts, ["pod_claim1"])
            self.assertEqual(session._cleanup_inflight, set())
            self.assertEqual(session.snapshots[0].state, "released")
            self.assertTrue(session._poisoned)


if __name__ == "__main__":
    unittest.main()
