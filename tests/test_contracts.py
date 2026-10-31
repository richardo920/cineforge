"""Synthetic v1 schema, lineage and malformed-input acceptance fixtures."""

import copy
import json
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from cineforge.contracts import (
    ContractError,
    Project,
    ProjectPlan,
    RunManifest,
    RunStatus,
    Scene,
    Shot,
    ShotRun,
)

FIXTURE = Path(__file__).parent / "fixtures" / "production-v1.json"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = RunManifest.from_json(FIXTURE.read_text())
        self.plan = self.manifest.plan
        self.shot = self.plan.shots[0]
        self.run = self.manifest.shot_runs[0]
        self.artifact = self.manifest.artifacts[0]

    def test_every_record_round_trips_and_canonical_json_is_stable(self):
        records = [self.plan.project, *self.plan.scenes, self.shot, self.plan,
                   self.run, self.artifact, self.manifest]
        for record in records:
            with self.subTest(record=type(record).__name__):
                self.assertEqual(type(record).from_dict(record.to_dict()), record)
                self.assertEqual(type(record).from_json(record.to_json()), record)
                self.assertEqual(type(record).from_json(record.to_json()).to_json(), record.to_json())
                self.assertEqual(record.to_dict()["schema_version"], 1)
        raw = json.dumps(self.manifest.to_dict(), indent=2)
        self.assertEqual(RunManifest.from_json(raw).to_json(), self.manifest.to_json())

    def test_fixture_contains_portable_reproduction_and_artifact_metadata(self):
        self.assertEqual(self.shot.seed, 7)
        self.assertEqual(self.shot.model_revision, "mock-v1")
        self.assertEqual(self.shot.workflow_digest, "0" * 64)
        self.assertEqual(self.run.provider_id, "mock")
        self.assertEqual(self.run.job_id, "job-001")
        self.assertEqual(self.run.cost_usd, 0)
        self.assertEqual(self.artifact.sha256, "1" * 64)
        self.assertEqual(self.artifact.license, "LicenseRef-Synthetic")
        self.assertEqual(self.artifact.retention_policy, "test-only")
        self.assertEqual(self.plan.get_shot("shot-001"), self.shot)

    def test_records_and_nested_collections_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.shot.prompt = "changed"
        document = self.manifest.to_dict()
        parsed = RunManifest.from_dict(document)
        document["plan"]["shots"][0]["prompt"] = "changed"
        self.assertEqual(parsed, self.manifest)
        with self.assertRaises(ContractError):
            ProjectPlan(self.plan.project, list(self.plan.scenes), self.plan.shots)
        with self.assertRaises(ContractError):
            replace(self.run, artifact_ids=["artifact-001"])

    def test_missing_ids_unknown_fields_and_missing_versions_fail(self):
        records = [(self.plan.project, "project_id"), (self.plan.scenes[0], "scene_id"),
                   (self.shot, "shot_id"), (self.artifact, "artifact_id"),
                   (self.run, "job_id"), (self.manifest, "run_id")]
        for record, identity in records:
            for missing in (identity, "schema_version"):
                document = record.to_dict()
                del document[missing]
                with self.subTest(record=type(record).__name__, missing=missing):
                    with self.assertRaises(ContractError):
                        type(record).from_dict(document)
            document = record.to_dict()
            document["unexpected"] = True
            with self.assertRaises(ContractError):
                type(record).from_dict(document)

    def test_all_records_reject_incompatible_or_wrong_typed_versions(self):
        records = [self.plan.project, self.plan.scenes[0], self.shot,
                   self.artifact, self.run, self.plan, self.manifest]
        for record in records:
            for version in (0, 2, True, 1.0, "1", None):
                with self.subTest(record=type(record).__name__, version=version):
                    with self.assertRaises(ContractError):
                        replace(record, schema_version=version)
                    document = record.to_dict()
                    document["schema_version"] = version
                    with self.assertRaises(ContractError):
                        type(record).from_dict(document)

    def test_nested_unknown_fields_and_versions_fail(self):
        for path in (("plan",), ("plan", "project"), ("plan", "shots", 0),
                     ("plan", "scenes", 0), ("artifacts", 0), ("shot_runs", 0)):
            document = self.manifest.to_dict()
            current = document
            for key in path:
                current = current[key]
            current["unreviewed_extension"] = "bad"
            with self.subTest(path=path), self.assertRaises(ContractError):
                RunManifest.from_dict(document)

    def test_ids_and_required_metadata_reject_empty_or_wrong_types(self):
        for invalid in ("", " ", None, True, 1, [], "../other", "id/child"):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                replace(self.shot, shot_id=invalid)
        for field in ("prompt", "model_id", "model_revision"):
            for invalid in ("", " ", None, 1, "bad\x00text", "bad\ud800text"):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ContractError):
                    replace(self.shot, **{field: invalid})
        for field in ("uri", "license", "origin", "retention_policy"):
            with self.subTest(field=field), self.assertRaises(ContractError):
                replace(self.artifact, **{field: ""})
        self.assertEqual(replace(self.shot, seed=None).seed, None)
        self.assertEqual(replace(self.shot, character_revision="character-r1").character_revision,
                         "character-r1")

    def test_numeric_validation_matches_direct_and_decoded_construction(self):
        for value in (True, False, None, "2", float("nan"), float("inf"), -1, 0, 10**1000):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ContractError):
                    replace(self.shot, duration_seconds=value)
                document = self.shot.to_dict()
                document["duration_seconds"] = value
                with self.assertRaises(ContractError):
                    Shot.from_dict(document)
        for value in (True, 1.5, -1, "1", 2**64):
            with self.subTest(seed=value), self.assertRaises(ContractError):
                replace(self.shot, seed=value)
        for value in (True, 1.5, -1, "1", 2**63):
            with self.subTest(size=value), self.assertRaises(ContractError):
                replace(self.artifact, size_bytes=value)
        for value in (True, -0.1, float("nan"), float("inf"), "0", None):
            with self.subTest(cost=value), self.assertRaises(ContractError):
                replace(self.run, cost_usd=value)
        self.assertEqual(replace(self.shot, duration_seconds=2).duration_seconds, 2.0)

    def test_checksums_and_workflow_digests_are_explicit_sha256(self):
        for value in ("", "a" * 63, "a" * 65, "g" * 64, "A" * 64, None):
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    replace(self.shot, workflow_digest=value)
                with self.assertRaises(ContractError):
                    replace(self.artifact, sha256=value)

    def test_unknown_or_duplicate_plan_references_fail(self):
        cases = [
            {"scenes": (replace(self.plan.scenes[0], project_id="unknown"),)},
            {"shots": (replace(self.shot, scene_id="unknown"),)},
            {"scenes": self.plan.scenes * 2},
            {"shots": self.plan.shots * 2},
            {"scenes": ()}, {"shots": ()},
            {"scenes": (*self.plan.scenes, Scene("empty-scene", "project-001", 1))},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ContractError):
                replace(self.plan, **values)
        with self.assertRaises(ContractError):
            self.plan.get_shot("missing")

    def test_run_and_artifact_references_and_revisions_are_checked(self):
        cases = [
            {"shot_runs": (replace(self.run, shot_id="unknown"),)},
            {"shot_runs": (replace(self.run, shot_revision=2),)},
            {"shot_runs": (replace(self.run, artifact_ids=("unknown",)),)},
            {"artifacts": (replace(self.artifact, shot_id="unknown"),)},
            {"shot_runs": self.manifest.shot_runs * 2},
            {"artifacts": self.manifest.artifacts * 2},
            {"shot_runs": ()},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ContractError):
                replace(self.manifest, **values)
        second_shot = replace(self.shot, shot_id="shot-002")
        plan = replace(self.plan, shots=(self.shot, second_shot))
        with self.assertRaises(ContractError):
            replace(self.manifest, plan=plan,
                    artifacts=(replace(self.artifact, shot_id="shot-002"),))
        second = replace(self.run, attempt=2, job_id="job-002", idempotency_key="retry-key")
        with self.assertRaises(ContractError):
            replace(self.manifest, shot_runs=(self.run, second))

    def test_partial_and_retried_runs_preserve_history_without_aliases(self):
        failed = replace(self.run, status=RunStatus.FAILED, error="synthetic failure", artifact_ids=())
        retry = replace(self.run, attempt=2, job_id="retry-job", idempotency_key="retry-key")
        manifest = replace(self.manifest, shot_runs=(failed, retry))
        self.assertEqual(RunManifest.from_json(manifest.to_json()), manifest)
        self.assertEqual(RunManifest("not-started", self.plan).shot_runs, ())
        with self.assertRaises(ContractError):
            replace(manifest, shot_runs=(failed, replace(retry, job_id=failed.job_id)))
        with self.assertRaises(ContractError):
            replace(manifest, shot_runs=(failed, replace(retry, idempotency_key=failed.idempotency_key)))

    def test_run_states_enforce_timestamps_error_and_artifact_invariants(self):
        queued = ShotRun("shot-001", 1, 1, "mock", "queued-job", "queued-key", RunStatus.QUEUED)
        running = replace(queued, status=RunStatus.RUNNING, started_at="2026-10-31T00:00:00Z")
        cancelled = replace(queued, status=RunStatus.CANCELLED,
                            finished_at="2026-10-31T00:00:00Z")
        for record in (queued, running, cancelled):
            self.assertEqual(ShotRun.from_dict(record.to_dict()), record)
        cases = [
            (queued, {"started_at": "2026-10-31T00:00:00Z"}),
            (running, {"started_at": None}),
            (running, {"finished_at": "2026-10-31T00:00:01Z"}),
            (self.run, {"finished_at": None}),
            (self.run, {"started_at": None}),
            (self.run, {"artifact_ids": ()}),
            (self.run, {"artifact_ids": ("artifact-001", "artifact-001")}),
            (self.run, {"error": "must not claim success"}),
            (cancelled, {"finished_at": None}),
            (cancelled, {"artifact_ids": ("artifact-001",)}),
            (queued, {"status": "queued"}),
            (queued, {"attempt": True}),
            (queued, {"status": RunStatus.FAILED, "finished_at": "2026-10-31T00:00:00Z"}),
        ]
        for record, values in cases:
            with self.subTest(values=values), self.assertRaises(ContractError):
                replace(record, **values)

    def test_timestamps_need_offsets_and_real_chronological_order(self):
        for value in ("bad", "2026-10-31T00:00:00", "2026-02-30T00:00:00Z", 1, ""):
            with self.subTest(value=value), self.assertRaises(ContractError):
                replace(self.run, started_at=value)
        with self.assertRaises(ContractError):
            replace(self.run, finished_at="2026-10-30T23:59:59Z")
        with self.assertRaises(ContractError):
            replace(self.run, started_at="2026-10-31T02:00:00+01:00")
        self.assertEqual(replace(self.run, started_at="2026-10-31T01:00:00+01:00").started_at,
                         "2026-10-31T01:00:00+01:00")

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_malformed_roots(self):
        for raw in ('{"schema_version":1,"schema_version":1}',
                    self.shot.to_json().replace('"duration_seconds":2.5', '"duration_seconds":NaN'),
                    self.shot.to_json().replace('"duration_seconds":2.5', '"duration_seconds":Infinity'),
                    self.shot.to_json().replace('"duration_seconds":2.5', '"duration_seconds":1e9999'),
                    '[]', 'null', '1', '{', '{"nested":{"a":1,"a":2}}'):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                Shot.from_json(raw)
        for data in ([], None, "{}", {1: "bad"}):
            with self.subTest(data=data), self.assertRaises(ContractError):
                Project.from_dict(data)
        document = copy.deepcopy(self.manifest.to_dict())
        document["shot_runs"][0]["status"] = "unrecognized"
        with self.assertRaises(ContractError):
            RunManifest.from_dict(document)


if __name__ == "__main__":
    unittest.main()
