"""Runnable fake-provider conformance suite; no credentials, hardware, media or transport."""

import dataclasses
import unittest

from cineforge.contracts import RunStatus, Shot
from cineforge.devices import Backend, ModelSpec
from cineforge.providers import (
    ExecutionProvider,
    ExecutionRequest,
    FakeExecutionProvider,
    IdempotencyConflictError,
    ProviderError,
    ResultNotReadyError,
    UnknownJobError,
)


def synthetic_request():
    return ExecutionRequest(
        Shot("shot", "scene", 1, "Synthetic geometry", 10, 1,
             "fixture", "fixture-v1", "1" * 64),
        ModelSpec("fixture", "fixture-v1", "metadata", frozenset({Backend.CPU}), 0,
                  "LicenseRef-SyntheticFixture"),
    )


class FakeProviderConformanceTests(unittest.TestCase):
    def setUp(self):
        self.provider: ExecutionProvider = FakeExecutionProvider()
        self.request = synthetic_request()

    def test_submit_poll_result_success_and_stable_terminal(self):
        queued = self.provider.submit(self.request, "key")
        self.assertEqual(queued.status, RunStatus.QUEUED)
        self.assertIsNone(queued.started_at)
        self.assertEqual(queued.artifact_ids, ())
        with self.assertRaises(ResultNotReadyError):
            self.provider.result(queued.job_id)
        running = self.provider.status(queued.job_id)
        self.assertEqual(running.status, RunStatus.RUNNING)
        self.assertIsNotNone(running.started_at)
        with self.assertRaises(ResultNotReadyError):
            self.provider.result(queued.job_id)
        succeeded = self.provider.status(queued.job_id)
        self.assertEqual(succeeded.status, RunStatus.SUCCEEDED)
        self.assertIsNotNone(succeeded.finished_at)
        result = self.provider.result(queued.job_id)
        self.assertEqual(result.run, succeeded)
        for _ in range(3):
            self.assertEqual(self.provider.status(queued.job_id), succeeded)
            self.assertEqual(self.provider.cancel(queued.job_id), succeeded)
            self.assertEqual(self.provider.submit(self.request, "key"), succeeded)
            self.assertEqual(self.provider.result(queued.job_id), result)
        self.assertEqual(queued.status, RunStatus.QUEUED)  # Previous snapshots remain immutable.
        self.assertEqual(running.status, RunStatus.RUNNING)

    def test_cancel_queued_and_running_never_publish_results(self):
        for start in [False, True]:
            with self.subTest(start=start):
                queued = self.provider.submit(self.request, f"cancel-{start}")
                if start:
                    self.provider.status(queued.job_id)
                cancelled = self.provider.cancel(queued.job_id)
                self.assertEqual(cancelled.status, RunStatus.CANCELLED)
                self.assertEqual(cancelled.artifact_ids, ())
                self.assertEqual(cancelled.cost_usd, 0)
                self.assertEqual(cancelled.started_at is not None, start)
                for _ in range(3):
                    self.assertEqual(self.provider.status(queued.job_id), cancelled)
                    self.assertEqual(self.provider.cancel(queued.job_id), cancelled)
                    self.assertEqual(self.provider.submit(self.request, f"cancel-{start}"), cancelled)
                    with self.assertRaises(ResultNotReadyError):
                        self.provider.result(queued.job_id)

    def test_failed_terminal_has_error_no_artifacts_and_no_result(self):
        provider = FakeExecutionProvider(fail=True)
        queued = provider.submit(self.request, "failure")
        self.assertEqual(provider.status(queued.job_id).status, RunStatus.RUNNING)
        failed = provider.status(queued.job_id)
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertTrue(failed.error)
        self.assertEqual(failed.artifact_ids, ())
        self.assertEqual(failed.cost_usd, 0)
        for _ in range(3):
            self.assertEqual(provider.status(queued.job_id), failed)
            self.assertEqual(provider.cancel(queued.job_id), failed)
            self.assertEqual(provider.submit(self.request, "failure"), failed)
            with self.assertRaises(ResultNotReadyError):
                provider.result(queued.job_id)

    def test_duplicate_keys_preserve_job_and_state_conflicts_do_not_corrupt_it(self):
        queued = self.provider.submit(self.request, "duplicate")
        identical = dataclasses.replace(self.request)
        self.assertEqual(self.provider.submit(identical, "duplicate"), queued)
        running = self.provider.status(queued.job_id)
        self.assertEqual(self.provider.submit(identical, "duplicate"), running)
        conflict = dataclasses.replace(self.request, attempt=2)
        with self.assertRaises(IdempotencyConflictError):
            self.provider.submit(conflict, "duplicate")
        succeeded = self.provider.status(queued.job_id)
        self.assertEqual(succeeded.status, RunStatus.SUCCEEDED)
        self.assertEqual(self.provider.result(queued.job_id).run.attempt, 1)
        different_key = self.provider.submit(self.request, "new-key")
        self.assertNotEqual(different_key.job_id, queued.job_id)
        self.assertEqual(different_key.status, RunStatus.QUEUED)

    def test_invalid_unknown_and_other_instance_jobs_fail_without_mutation(self):
        other_job = FakeExecutionProvider().submit(self.request, "foreign")
        for operation in [self.provider.status, self.provider.cancel, self.provider.result]:
            for job_id in ["unknown", other_job.job_id]:
                with self.subTest(operation=operation, job_id=job_id):
                    with self.assertRaises(UnknownJobError):
                        operation(job_id)
            for job_id in ["", None, 3, [], " padded"]:
                with self.subTest(operation=operation, job_id=job_id):
                    with self.assertRaises(ProviderError):
                        operation(job_id)
        self.assertEqual(self.provider.submit(self.request, "valid").status, RunStatus.QUEUED)


if __name__ == "__main__":
    unittest.main()
