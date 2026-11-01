"""Synthetic metadata only; no fixtures assert actual consent or likeness rights."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from cineforge.characters import (
    CharacterError,
    CharacterPin,
    CharacterRegistry,
    CharacterRevision,
    ConditioningHint,
    ReferenceArtifact,
    RightsBinding,
    Wardrobe,
    pin_shot,
    with_character_review,
)
from cineforge.contracts import Shot
from cineforge.rights import (
    EvidenceRef,
    RightsLedger,
    RightsRecord,
    RightsRef,
    SubjectRef,
    Use,
    with_review,
)

DIGEST = "a" * 64
EVIDENCE = EvidenceRef("private://synthetic/evidence", DIGEST)


def reference(name: str = "front", role: str = "identity") -> ReferenceArtifact:
    subject = SubjectRef("voice" if role == "voice" else "character_reference", name, "v1",
                         DIGEST)
    return ReferenceArtifact(name, role, EvidenceRef(f"private://synthetic/{name}", DIGEST), 42,
                             "audio/wav" if role == "voice" else "image/png",
                             RightsBinding(subject, RightsRef(name, "v1")))


def character() -> CharacterRevision:
    pending = CharacterRevision("synthetic-robot", 1, "fictional", "A teal robot",
                                Wardrobe("scarf", "An orange scarf"), (reference(),))
    return with_character_review(pending, reviewer="synthetic-reviewer", evidence=EVIDENCE)


def permission(binding: RightsBinding) -> RightsRecord:
    pending = RightsRecord(binding.rights, binding.subject, source=EVIDENCE,
                           source_revision="v1", license_id="LicenseRef-Synthetic",
                           terms=EVIDENCE, notices=("Synthetic notice",),
                           attribution="Synthetic author", consent=EVIDENCE,
                           allowed_uses=(Use("reference"), Use("condition")))
    return with_review(pending, reviewer="synthetic-reviewer", evidence=EVIDENCE)


def ledger_for(record: CharacterRevision) -> RightsLedger:
    ledger = RightsLedger()
    for ref in record.references:
        ledger.register(permission(ref.permission))
    if record.likeness is not None:
        ledger.register(permission(record.likeness))
    return ledger


def shot() -> Shot:
    return Shot("shot", "scene", 1, "Robot waves", 7, 2, "synthetic-model", "v1", DIGEST)


class CharacterTests(unittest.TestCase):
    def test_pin_round_trip_preserves_schema_v1_and_noncharacter_shot(self) -> None:
        record = character()
        registry = CharacterRegistry()
        registry.register(record)
        ledger = ledger_for(record)
        self.assertEqual(CharacterPin.from_key(record.pin().to_key()), record.pin())
        bound = pin_shot(shot(), record.pin())
        self.assertEqual(Shot.from_json(bound.to_json()), bound)
        self.assertEqual(bound.schema_version, 1)
        self.assertEqual(registry.resolve_shot(bound, ledger).character, record)
        self.assertTrue(registry.resolve_shot(shot(), ledger).allowed)
        self.assertIsNone(registry.resolve_shot(shot(), ledger).character)
        self.assertIsNone(shot().character_revision)

    def test_stable_ids_new_revisions_and_immutable_history(self) -> None:
        first = character()
        second = with_character_review(replace(first, revision=2,
                                               wardrobe=Wardrobe("coat", "A blue coat")),
                                       reviewer="reviewer", evidence=EVIDENCE)
        registry = CharacterRegistry()
        registry.register(first)
        registry.register(second)
        self.assertEqual(registry.lookup(first.pin()), first)
        self.assertEqual(registry.lookup(second.pin()), second)
        self.assertNotEqual(first.pin(), second.pin())
        with self.assertRaises(FrozenInstanceError):
            first.revision = 2  # type: ignore[misc]
        for replacement in (first, replace(first, identity_description="Changed"),
                            replace(first, wardrobe=second.wardrobe)):
            with self.assertRaises(CharacterError):
                registry.register(replacement)
        self.assertEqual(registry.lookup(first.pin()), first)

    def test_changed_subject_content_cannot_resolve_same_pin(self) -> None:
        record = character()
        registry = CharacterRegistry()
        registry.register(record)
        ledger = ledger_for(record)
        for pin in (replace(record.pin(), sha256="b" * 64),
                    replace(record.pin(), revision=2), replace(record.pin(), character_id="absent")):
            result = registry.resolve(pin, ledger)
            self.assertFalse(result.allowed)
            self.assertEqual(result.reasons, ("missing_or_changed_character",))
            self.assertIsNone(result.character)
        changed = replace(record, identity_description="Different identity")
        other_registry = CharacterRegistry()
        other_registry.register(changed)
        self.assertFalse(other_registry.resolve(record.pin(), ledger).allowed)

    def test_missing_current_rights_and_revocation_after_success(self) -> None:
        record = character()
        registry = CharacterRegistry()
        registry.register(record)
        self.assertIn("missing_rights", registry.resolve(record.pin(), RightsLedger()).reasons)
        ledger = ledger_for(record)
        bound = pin_shot(shot(), record.pin())
        self.assertTrue(registry.resolve_shot(bound, ledger, use=Use("condition")).allowed)
        ledger.revoke(record.references[0].permission.rights, reason="Synthetic withdrawal")
        result = registry.resolve_shot(bound, ledger, use=Use("condition"))
        self.assertEqual(result.reasons, ("revoked_rights",))
        self.assertIsNone(result.character)
        self.assertEqual(registry.lookup(record.pin()), record)

    def test_superseded_unreviewed_changed_and_wrong_use_permissions(self) -> None:
        record = character()
        registry = CharacterRegistry()
        registry.register(record)
        grant = permission(record.references[0].permission)
        variants = ((replace(grant, review=None), "unreviewed_rights"),
                    (replace(grant, consent=None), "missing_consent"),
                    (replace(grant, subject=replace(grant.subject, sha256="b" * 64)),
                     "subject_mismatch"))
        for variant, reason in variants:
            ledger = RightsLedger()
            ledger.register(variant)
            self.assertIn(reason, registry.resolve(record.pin(), ledger).reasons)
        ledger = ledger_for(record)
        self.assertIn("use_not_permitted",
                      registry.resolve(record.pin(), ledger, use=Use("reference", "public")).reasons)
        ledger.register(replace(grant, ref=RightsRef(grant.ref.record_id, "v2"), review=None))
        self.assertIn("superseded_rights", registry.resolve(record.pin(), ledger).reasons)

    def test_character_review_binds_all_metadata_and_changes_require_fresh_review(self) -> None:
        record = character()
        mutations = ({"identity_description": "Other"}, {"identity_kind": "person"},
                     {"wardrobe": Wardrobe("same", "Other")}, {"revision": 2},
                     {"character_id": "other"},
                     {"references": (replace(record.references[0], size_bytes=100),)},
                     {"references": (replace(record.references[0],
                                             artifact=EvidenceRef("private://other", DIGEST)),)},
                     {"references": (replace(record.references[0], permission=RightsBinding(
                         record.references[0].permission.subject, RightsRef("other", "v1"))),)},
                     {"conditioning": (ConditioningHint("start_image", "model", "v1", DIGEST,
                                                        ("front",)),)})
        for fields in mutations:
            changed = replace(record, **fields)
            registry = CharacterRegistry()
            registry.register(changed)
            self.assertNotEqual(changed.pin(), record.pin())
            self.assertIn("stale_character_review",
                          registry.resolve(changed.pin(), ledger_for(changed)).reasons)
        pending = replace(record, review=None)
        registry = CharacterRegistry()
        registry.register(pending)
        self.assertEqual(registry.resolve(pending.pin(), ledger_for(pending)).reasons,
                         ("unreviewed_character",))

    def test_voice_and_person_likeness_use_current_central_consent(self) -> None:
        first = character()
        likeness = RightsBinding(SubjectRef("likeness", "synthetic-person", "v1", DIGEST),
                                 RightsRef("synthetic-likeness", "v1"))
        person = with_character_review(replace(first, identity_kind="person"),
                                       reviewer="reviewer", evidence=EVIDENCE)
        registry = CharacterRegistry()
        registry.register(person)
        self.assertIn("missing_likeness_permission",
                      registry.resolve(person.pin(), ledger_for(person)).reasons)
        person = with_character_review(replace(person, revision=2, likeness=likeness,
                                               references=person.references + (reference("voice", "voice"),),
                                               voice_reference_id="voice"),
                                       reviewer="reviewer", evidence=EVIDENCE)
        registry.register(person)
        with self.assertRaises(CharacterError):
            replace(person, likeness=replace(likeness,
                    subject=replace(likeness.subject, sha256="b" * 64)))
        for binding in (likeness, person.references[1].permission):
            ledger = ledger_for(person)
            self.assertTrue(registry.resolve(person.pin(), ledger).allowed)
            ledger.revoke(binding.rights, reason="Synthetic withdrawn consent")
            self.assertIn("revoked_rights", registry.resolve(person.pin(), ledger).reasons)

    def test_wardrobe_voice_conditioning_links_and_strict_round_trip(self) -> None:
        record = character()
        refs = record.references + (reference("outfit", "wardrobe"), reference("voice", "voice"))
        record = replace(record, references=refs, voice_reference_id="voice",
                         wardrobe=Wardrobe("scarf", "Orange scarf", ("outfit",)),
                         conditioning=(ConditioningHint("start_image", "synthetic/model", "v1",
                                                        DIGEST, ("front",)),))
        record = with_character_review(record, reviewer="reviewer", evidence=EVIDENCE)
        self.assertEqual(CharacterRevision.from_json(record.to_json()), record)
        self.assertEqual(CharacterPin.from_json(record.pin().to_json()), record.pin())
        registry = CharacterRegistry()
        registry.register(record)
        self.assertTrue(registry.resolve(record.pin(), ledger_for(record)).allowed)
        for fields in ({"wardrobe": Wardrobe("scarf", "Orange", ("missing",))},
                       {"wardrobe": Wardrobe("scarf", "Orange", ("front",))},
                       {"voice_reference_id": None}, {"voice_reference_id": "front"},
                       {"conditioning": (ConditioningHint("start_image", "model", "v1", DIGEST,
                                                          ("missing",)),)},
                       {"conditioning": (ConditioningHint("start_image", "model", "v1", DIGEST,
                                                          ("voice",)),)}):
            with self.assertRaises(CharacterError):
                replace(record, **fields)

    def test_missing_duplicate_wrong_format_or_digest_references_fail(self) -> None:
        record = character()
        bad = (lambda: replace(record, references=()),
               lambda: replace(record, references=record.references * 2),
               lambda: replace(record.references[0], role="unknown"),
               lambda: replace(record.references[0], size_bytes=True),
               lambda: replace(record.references[0], media_type="audio/wav"),
               lambda: replace(record.references[0], artifact=EvidenceRef("https://example.org/a",
                                                                          DIGEST)),
               lambda: replace(record.references[0], artifact=EvidenceRef("private://other", "b" * 64)),
               lambda: replace(record.references[0], permission=RightsBinding(
                   SubjectRef("asset", "front", "v1", DIGEST), RightsRef("front", "v1"))),
               lambda: ConditioningHint("start_image", "model", "latest", DIGEST, ("front",)))
        for constructor in bad:
            with self.assertRaises(ValueError):
                constructor()

    def test_malformed_json_and_ambiguous_pins_never_echo_private_input(self) -> None:
        data = character().to_dict()
        marker = "private://synthetic-person/private-sentinel"
        bad = (None, [], {}, dict(data, schema_version=marker), dict(data, unknown=marker),
               dict(data, references="string"), dict(data, wardrobe={"private": marker}),
               dict(data, references=[{}]), dict(data, review=marker),
               dict(data, references=[dict(data["references"][0], artifact={"uri": marker})]))
        for value in bad:
            with self.assertRaises(CharacterError) as caught:
                CharacterRevision.from_json(json.dumps(value))
            self.assertNotIn(marker, str(caught.exception))
        for raw in ('{"schema_version":1,"schema_version":1}', '{"private-sentinel":NaN}',
                    "not JSON"):
            with self.assertRaises(CharacterError):
                CharacterRevision.from_json(raw)
        registry = CharacterRegistry()
        for key in ("robot", "robot@1", "robot@latest#" + DIGEST,
                    "robot@01#" + DIGEST, "robot@1#" + DIGEST.upper(),
                    marker, "robot@9223372036854775808#" + DIGEST):
            with self.assertRaises(CharacterError) as caught:
                registry.resolve_shot(replace(shot(), character_revision=key), RightsLedger())
            self.assertNotIn("private-sentinel", str(caught.exception))

    def test_private_reference_details_absent_from_reprs_and_denied_results(self) -> None:
        first = character()
        ref = replace(first.references[0], artifact=EvidenceRef("private://private-sentinel", DIGEST))
        record = with_character_review(replace(first, references=(ref,),
                                               identity_description="private-sentinel",
                                               wardrobe=Wardrobe("scarf", "private-sentinel")),
                                       reviewer="reviewer", evidence=ref.artifact)
        registry = CharacterRegistry()
        registry.register(record)
        result = registry.resolve(record.pin(), RightsLedger())
        for value in (record, ref, record.review, record.wardrobe, result):
            self.assertNotIn("private-sentinel", repr(value))
        self.assertFalse(result.allowed)
        self.assertIsNone(result.character)


if __name__ == "__main__":
    unittest.main()
