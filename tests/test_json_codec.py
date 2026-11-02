"""Frozen v1 identities and registry-only decoding; synthetic metadata only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from codec_v1_baseline import BASELINE
from test_characters import character, permission, reference

from cineforge.characters import (
    CharacterError,
    CharacterPin,
    CharacterRevision,
    ConditioningHint,
    RightsBinding,
    Wardrobe,
    with_character_review,
)
from cineforge.contracts import (
    Contract,
    ContractError,
    Project,
    RunManifest,
    RunStatus,
    Scene,
    Shot,
    _tuple,
)
from cineforge.json_codec import canonical_json, strict_loads
from cineforge.providers import ProviderError
from cineforge.rights import EvidenceRef, RightsRef, SubjectRef
from cineforge.workflows import canonical, strict_object

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class StructuralRecord(Contract):
    _objects: ClassVar[dict[str, type[Contract]]] = {"project": Project}
    _optional_objects: ClassVar[dict[str, type[Contract]]] = {"optional": Project}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"children": Scene}
    _tuple_fields: ClassVar[tuple[str, ...]] = ("labels",)
    _enums: ClassVar[dict[str, type[RunStatus]]] = {"status": RunStatus}

    project: Project
    children: tuple[Scene, ...]
    labels: tuple[str, ...]
    optional: Project | None = None
    status: RunStatus = RunStatus.QUEUED

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _tuple(self.labels, str, "labels")


class JsonCodecTests(unittest.TestCase):
    def test_frozen_v1_manifest_character_rights_and_approved_workflow_hashes(self) -> None:
        records = {}
        manifest = RunManifest.from_json((ROOT / "tests/fixtures/production-v1.json").read_text())
        for item in (manifest, manifest.plan, manifest.plan.project, *manifest.plan.scenes,
                     *manifest.plan.shots, *manifest.shot_runs, *manifest.artifacts):
            records[type(item).__name__] = item.to_json()
        c = character()
        c = with_character_review(replace(c, references=c.references + (
            reference("voice", "voice"), reference("wardrobe", "wardrobe")),
            voice_reference_id="voice", wardrobe=Wardrobe("outfit", "Synthetic", ("wardrobe",)),
            conditioning=(ConditioningHint("test", "model", "v1", "a" * 64, ("front",)),)),
            reviewer="synthetic", evidence=EvidenceRef("private://synthetic/review", "a" * 64))
        for item in (c, c.pin(), c.review, c.wardrobe, *c.references, *c.conditioning,
                     permission(c.references[1].permission)):
            records[type(item).__name__ + str(len(records))] = item.to_json()
        records["character_pin"] = c.pin().to_key()
        records["character_declaration_digest"] = c.declaration_digest()
        for name in BASELINE["sha256"]:
            if name.startswith("config/"):
                records[name] = canonical(json.loads((ROOT / name).read_text())).decode()
        self.assertEqual({name: hashlib.sha256(value.encode()).hexdigest()
                          for name, value in records.items()}, BASELINE["sha256"])
        self.assertEqual(Shot.from_json(manifest.plan.shots[0].to_json()).to_json(),
                         manifest.plan.shots[0].to_json())
        self.assertIn('"character_revision":null', manifest.plan.shots[0].to_json())

    def test_registry_only_record_round_trip_absent_null_value_and_empty_tuples(self) -> None:
        record = StructuralRecord(Project("p", 1, "你好"), (Scene("s", "p", 1),), ("en", "zh"))
        for optional in (None, Project("other", 1, "English")):
            item = replace(record, optional=optional)
            self.assertEqual(StructuralRecord.from_json(item.to_json()), item)
        data = record.to_dict()
        del data["optional"]
        del data["status"]
        self.assertEqual(StructuralRecord.from_dict(data), record)
        empty = replace(record, labels=(), children=())
        self.assertEqual(StructuralRecord.from_json(empty.to_json()), empty)
        self.assertNotIn("from_dict", StructuralRecord.__dict__)

    def test_character_optional_likeness_and_review_preserve_absent_and_null_defaults(self) -> None:
        binding = RightsBinding(SubjectRef("likeness", "synthetic", "r1", "a" * 64),
                                RightsRef("synthetic", "r1"))
        for likeness in (None, binding):
            for review in (None, character().review):
                c = replace(character(), likeness=likeness, review=review)
                self.assertEqual(CharacterRevision.from_json(c.to_json()), c)
                self.assertEqual(CharacterRevision.from_json(c.to_json()).pin(), c.pin())
                data = c.to_dict()
                if likeness is None:
                    del data["likeness"]
                if review is None:
                    del data["review"]
                self.assertEqual(CharacterRevision.from_dict(data), c)

    def test_registry_strictness_unknown_fields_booleans_versions_and_containers(self) -> None:
        data = StructuralRecord(Project("p", 1, "Example"), (), ()).to_dict()
        for update in ({"unknown": True}, {"schema_version": True}, {"schema_version": 2},
                       {"children": ()}, {"labels": ()}, {"labels": [True]}, {"labels": "x"},
                       {"status": True}, {"optional": True},
                       {"optional": {"schema_version": 1, "project_id": "p", "revision": True,
                                     "title": "X"}},
                       {"children": [{"schema_version": 1, "scene_id": "s", "project_id": "p",
                                      "revision": True}]},
                       {"optional": {"schema_version": 1, "project_id": "p", "revision": 1,
                                     "title": "X", "unexpected": 1}}):
            with self.assertRaises(ContractError):
                StructuralRecord.from_dict(dict(data, **update))
        with self.assertRaises(ContractError):
            StructuralRecord.from_dict({k: v for k, v in data.items() if k != "schema_version"})

    def test_legacy_static_errors_and_private_workflow_diagnostics_are_unchanged(self) -> None:
        calls = {
            "character_json": lambda: CharacterRevision.from_json('{"private-canary":NaN}'),
            "character_dict": lambda: CharacterRevision.from_dict({"private-canary": 1}),
            "wardrobe_array": lambda: Wardrobe.from_dict({"schema_version": 1,
                "wardrobe_id": "x", "description": "x", "reference_ids": "private-canary"}),
            "workflow": lambda: strict_object(b'{"private-canary":NaN}'),
            "contract": lambda: Shot.from_json('{"x":NaN}'),
        }
        for name, call in calls.items():
            try:
                call()
            except (ContractError, ProviderError) as exc:
                self.assertEqual([type(exc).__name__, str(exc)], BASELINE["errors"][name])
            else:
                self.fail("invalid input accepted")
        for raw in (b'[]', b'{"private-canary":1,"private-canary":2}', b'\xff',
                    '{"private-canary":"\ud800"}', '{"x":1e9999}'):
            try:
                strict_object(raw)
            except ProviderError:
                self.assertNotIn("private-canary", traceback.format_exc())
            else:
                self.fail("invalid workflow accepted")
        with self.assertRaises(CharacterError):
            CharacterRevision.from_dict(dict(character().to_dict(), review=True))

    def test_leaf_preserves_canonical_numbers_and_raw_input_boundaries(self) -> None:
        value = {"zh": "你好", "float": 1.0, "negative_zero": -0.0, "none": None}
        self.assertEqual(canonical_json(value), json.dumps(value, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        for raw in (b'{"n":1}', bytearray(b'{"n":1}'), '{"n":1}'):
            self.assertEqual(strict_object(raw), {"n": 1})
        with self.assertRaisesRegex(ContractError, "JSON input must be text"):
            Project.from_json(b'{}')
        self.assertEqual(strict_loads('[]'), [])  # Object roots are a caller policy.
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError) as caught:
                strict_loads(raw)
            self.assertNotIn('"x"', str(caught.exception))
        shot = RunManifest.from_json((ROOT / "tests/fixtures/production-v1.json").read_text()
                                    ).plan.shots[0]
        legacy = replace(shot, character_revision="opaque-legacy-reference")
        self.assertEqual(Shot.from_json(legacy.to_json()), legacy)
        with self.assertRaises(CharacterError):
            CharacterPin.from_key("opaque-legacy-reference")

    def test_fresh_import_orders_do_not_cycle(self) -> None:
        for modules in (("contracts", "workflows"), ("workflows", "contracts")):
            result = subprocess.run([sys.executable, "-c", ";".join(
                f"import cineforge.{name}" for name in modules)], env=dict(os.environ),
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
