"""Offline execution DTO validation, provenance and deterministic identity tests."""

import dataclasses
import hashlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from cineforge.contracts import Artifact, Project, ProjectPlan, RunManifest, RunStatus, Scene, Shot
from cineforge.devices import Backend, ModelSpec
from cineforge.providers import (
    ExecutionProvider,
    ExecutionRequest,
    ExecutionResult,
    FakeExecutionProvider,
    IdempotencyConflictError,
    ProviderError,
)
from cineforge.provisioning import Provisioner


def request():
    return ExecutionRequest(
        Shot("shot-1", "scene-1", 2, "Synthetic paper moon, 雪", 42, 1.5,
             "synthetic-model", "fixture-v1", "a" * 64, "character-v2"),
        ModelSpec("synthetic-model", "fixture-v1", "text-to-video", frozenset(Backend),
                  0, "LicenseRef-SyntheticFixture"),
    )


class ExecutionInputTests(unittest.TestCase):
    def test_model_identity_and_revision_must_match_shot(self):
        source = request()
        for change in [{"model_id": "different"}, {"revision": "fixture-v2"}]:
            with self.subTest(change=change), self.assertRaisesRegex(ProviderError, "match"):
                dataclasses.replace(source, model=dataclasses.replace(source.model, **change))

    def test_input_types_and_attempt_bounds_are_checked(self):
        source = request()
        for change in [{"shot": None}, {"model": None}, *(
            {"attempt": value} for value in [False, 0, -1, 1.0, "1", 2**63]
        )]:
            with self.subTest(change=change), self.assertRaises(ProviderError):
                dataclasses.replace(source, **change)
        self.assertEqual(dataclasses.replace(source, attempt=2**63 - 1).attempt, 2**63 - 1)

    def test_invalid_keys_requests_and_failure_mode_fail_before_submission(self):
        provider = FakeExecutionProvider()
        for key in [None, 1, "", " ", " key", "key ", "key\n", "key\0", "\ud800"]:
            with self.subTest(key=key), self.assertRaises(ProviderError):
                provider.submit(request(), key)
        with self.assertRaises(ProviderError):
            provider.submit({}, "key")
        for value in [None, 1, "false"]:
            with self.assertRaises(ProviderError):
                FakeExecutionProvider(fail=value)

    def test_protocol_boundaries_are_separate(self):
        self.assertIsInstance(FakeExecutionProvider(), ExecutionProvider)
        self.assertNotIsInstance(FakeExecutionProvider(), Provisioner)
        self.assertFalse(hasattr(FakeExecutionProvider(), "acquire"))


class FakeProvenanceTests(unittest.TestCase):
    def completed(self, provider, key="same-key", execution=None):
        queued = provider.submit(execution or request(), key)
        provider.status(queued.job_id)
        provider.status(queued.job_id)
        return provider.result(queued.job_id)

    def test_payload_checksums_full_provenance_and_manifest_integration(self):
        source = request()
        result = self.completed(FakeExecutionProvider(), execution=source)
        metadata = json.loads(result.payload)
        self.assertTrue(metadata["synthetic"])
        self.assertEqual(metadata["media_type"], "application/json")
        self.assertEqual(metadata["request"], source.to_dict())
        self.assertEqual(metadata["job_id"], result.run.job_id)
        self.assertEqual(metadata["provider_id"], result.run.provider_id)
        self.assertEqual(result.artifact.sha256, hashlib.sha256(result.payload).hexdigest())
        self.assertEqual(result.artifact.size_bytes, len(result.payload))
        self.assertEqual(result.run.cost_usd, 0)
        self.assertTrue(result.artifact.uri.startswith("memory://"))
        self.assertIn(result.run.job_id, result.artifact.origin)
        plan = ProjectPlan(Project("project", 1, "Synthetic"),
                           (Scene("scene-1", "project", 1),), (source.shot,))
        manifest = RunManifest("run", plan, (result.run,), (result.artifact,))
        self.assertEqual(RunManifest.from_json(manifest.to_json()), manifest)

    def test_ids_and_payload_are_deterministic_across_instances_and_unrelated_jobs(self):
        first = self.completed(FakeExecutionProvider())
        second_provider = FakeExecutionProvider()
        self.completed(second_provider, "unrelated-key")
        second = self.completed(second_provider)
        self.assertEqual(first, second)
        third = self.completed(FakeExecutionProvider(), "different-key")
        self.assertNotEqual(first.run.job_id, third.run.job_id)
        self.assertNotEqual(first.artifact.artifact_id, third.artifact.artifact_id)
        fourth = self.completed(FakeExecutionProvider(), execution=dataclasses.replace(
            request(), attempt=2,
        ))
        self.assertNotEqual(first.run.job_id, fourth.run.job_id)

    def test_every_request_provenance_field_participates_in_idempotency(self):
        source = request()
        provider = FakeExecutionProvider()
        provider.submit(source, "key")
        changes = [
            dataclasses.replace(source, attempt=2),
            dataclasses.replace(source, model=dataclasses.replace(source.model, license="other")),
            *[dataclasses.replace(source, shot=dataclasses.replace(source.shot, **change))
              for change in [{"prompt": "changed"}, {"seed": None}, {"revision": 3},
                             {"workflow_digest": "b" * 64}, {"character_revision": None},
                             {"duration_seconds": 2.0}, {"scene_id": "scene-2"}]],
        ]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(IdempotencyConflictError):
                provider.submit(changed, "key")

    def test_result_and_provenance_are_immutable_and_copies_do_not_change_job(self):
        source = request()
        snapshot = source.to_dict()
        snapshot["shot"]["prompt"] = "changed"
        result = self.completed(FakeExecutionProvider(), execution=source)
        self.assertEqual(json.loads(result.payload)["request"]["shot"]["prompt"], source.shot.prompt)
        for obj, field, value in [(source, "attempt", 4), (result, "payload", b"changed"),
                                  (result.run, "status", RunStatus.FAILED),
                                  (result.artifact, "sha256", "b" * 64)]:
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, field, value)
        self.assertIsInstance(result.payload, bytes)

    def test_result_dto_rejects_wrong_links_checksum_size_and_mutable_payload(self):
        result = self.completed(FakeExecutionProvider())
        for change in [
            {"payload": bytearray(result.payload)}, {"payload": b"wrong"}, {"run": None},
            {"artifact": None},
            *[{"artifact": dataclasses.replace(result.artifact, **fields)} for fields in [
                {"shot_id": "other"}, {"artifact_id": "other"},
                {"size_bytes": 0}, {"sha256": "0" * 64},
            ]],
        ]:
            with self.subTest(change=change), self.assertRaises(ProviderError):
                dataclasses.replace(result, **change)
        self.assertIsInstance(result.artifact, Artifact)
        self.assertIsInstance(result, ExecutionResult)

    def test_fake_runner_never_calls_network_filesystem_or_wall_clock(self):
        source = request()
        with (patch("socket.socket", side_effect=AssertionError("network")),
              patch("builtins.open", side_effect=AssertionError("filesystem")),
              patch("os.open", side_effect=AssertionError("filesystem")),
              patch("time.time", side_effect=AssertionError("clock")),
              patch("time.sleep", side_effect=AssertionError("sleep"))):
            result = self.completed(FakeExecutionProvider(), execution=source)
            self.assertEqual(result.run.cost_usd, 0)

    def test_concurrent_duplicate_submits_return_one_job(self):
        provider = FakeExecutionProvider()
        source = request()
        with ThreadPoolExecutor(max_workers=8) as pool:
            runs = list(pool.map(lambda _: provider.submit(source, "key"), range(32)))
        self.assertTrue(all(run == runs[0] for run in runs))


if __name__ == "__main__":
    unittest.main()
