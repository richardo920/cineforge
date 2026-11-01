"""Synthetic offline permissions; no fixture asserts real-world legal rights."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from cineforge.rights import (
    EvidenceRef,
    RightsError,
    RightsLedger,
    RightsRecord,
    RightsRef,
    RightsReview,
    SubjectRef,
    Use,
    UseRequest,
    with_review,
)

DIGEST = "a" * 64


def declaration(kind: str = "skill", name: str = "synthetic") -> RightsRecord:
    evidence = EvidenceRef("private://synthetic/evidence", DIGEST)
    return RightsRecord(
        RightsRef(name, "v1"), SubjectRef(kind, name, "v1", DIGEST),
        source=evidence, source_revision="v1", license_id="LicenseRef-Synthetic",
        terms=evidence, notices=("Synthetic notice",), attribution="Synthetic credit",
        allowed_uses=(Use("reference"), Use("export", "public")), consent=evidence)


def reviewed(kind: str = "skill", name: str = "synthetic") -> RightsRecord:
    return with_review(declaration(kind, name), reviewer="synthetic-reviewer",
                       evidence=EvidenceRef("private://synthetic/review", DIGEST))


class RightsTests(unittest.TestCase):
    def test_review_and_live_revocation_keep_immutable_history(self) -> None:
        record = reviewed()
        ledger = RightsLedger()
        ledger.register(record)
        self.assertTrue(ledger.preflight_use(record.subject, record.ref, Use("reference")).allowed)
        ledger.revoke(record.ref, reason="Consent withdrawn")
        result = ledger.preflight_use(record.subject, record.ref, Use("reference"))
        self.assertEqual(result.reasons, ("revoked_rights",))
        self.assertEqual(ledger.lookup(record.ref), record)
        self.assertTrue(ledger.is_revoked(record.ref))
        with self.assertRaises(FrozenInstanceError):
            record.subject = record.subject  # type: ignore[misc]
        with self.assertRaises(RightsError):
            ledger.register(record)

    def test_registration_is_not_permission(self) -> None:
        record = declaration()
        ledger = RightsLedger()
        ledger.register(record)
        self.assertEqual(ledger.preflight_use(record.subject, record.ref, Use("reference")).reasons,
                         ("unreviewed_rights",))

    def test_review_binds_all_declaration_content(self) -> None:
        record = replace(reviewed(), allowed_uses=(Use("execute"),))
        ledger = RightsLedger()
        ledger.register(record)
        self.assertIn("stale_review",
                      ledger.preflight_use(record.subject, record.ref, Use("execute")).reasons)

    def test_export_is_deterministic_and_all_or_nothing(self) -> None:
        ledger = RightsLedger()
        records = (reviewed("music", "second"), reviewed("model", "first"))
        for record in records:
            ledger.register(record)
        requests = tuple(UseRequest(r.subject, r.ref, Use("export", "public")) for r in records)
        a = ledger.preflight_export(requests)
        self.assertTrue(a.allowed)
        self.assertEqual(a, ledger.preflight_export(tuple(reversed(requests))))
        self.assertEqual(a, ledger.preflight_export(requests + requests))
        ledger.revoke(records[0].ref, reason="withdrawn")
        denied = ledger.preflight_export(requests)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.attributions, ())


class RightsAcceptanceTests(unittest.TestCase):
    def allowed(self, record: RightsRecord, use: Use | None = None):
        ledger = RightsLedger()
        ledger.register(record)
        return ledger.preflight_use(record.subject, record.ref, use or Use("reference"))

    def test_all_resource_kinds_require_real_reviewed_declarations(self) -> None:
        for kind in ("code", "model", "skill", "workflow", "character_reference", "likeness",
                     "voice", "music", "asset"):
            with self.subTest(kind=kind):
                self.assertTrue(self.allowed(reviewed(kind)).allowed)
                unknown = with_review(RightsRecord(RightsRef("unknown", "v1"),
                                          SubjectRef(kind, "unknown", "v1", DIGEST)),
                                      reviewer="reviewer",
                                      evidence=EvidenceRef("private://review", DIGEST))
                result = self.allowed(unknown)
                self.assertFalse(result.allowed)
                self.assertIn("unknown_terms", result.reasons)
                self.assertEqual(result.attributions, ())

    def test_each_missing_required_evidence_denies_even_with_review(self) -> None:
        base = declaration()
        for fields, code in (({"source": None}, "missing_provenance"),
                             ({"source_revision": None}, "missing_provenance"),
                             ({"terms": None}, "unknown_terms"),
                             ({"license_id": "NOASSERTION"}, "unknown_terms"),
                             ({"notices": ()}, "missing_notices"),
                             ({"attribution": ""}, "missing_attribution"),
                             ({"allowed_uses": ()}, "use_not_permitted")):
            with self.subTest(code=code, fields=fields):
                pending = replace(base, **fields)
                record = with_review(pending, reviewer="reviewer", evidence=base.source)
                self.assertIn(code, self.allowed(record).reasons)

    def test_required_likeness_and_voice_consent_cannot_be_disabled(self) -> None:
        for kind in ("voice", "likeness", "character_reference"):
            pending = replace(declaration(kind), consent=None, consent_required=False)
            record = with_review(pending, reviewer="reviewer", evidence=pending.source)
            self.assertEqual(self.allowed(record).reasons, ("missing_consent",))
        pending = replace(declaration("asset"), consent=None, consent_required=True)
        record = with_review(pending, reviewer="reviewer", evidence=pending.source)
        self.assertIn("missing_consent", self.allowed(record).reasons)

    def test_context_permissions_are_explicit_not_inferred(self) -> None:
        record = reviewed()
        self.assertTrue(self.allowed(record).allowed)
        for use in (Use("reference", "public"), Use("reference", "internal", True),
                    Use("execute"), Use("export", "public", True)):
            self.assertEqual(self.allowed(record, use).reasons, ("use_not_permitted",))

    def test_restrictions_override_explicit_grants(self) -> None:
        for restriction, use in (("noncommercial", Use("reference", "internal", True)),
                                 ("internal_only", Use("reference", "public")),
                                 ("no_modification", Use("modify")),
                                 ("no_distribution", Use("export", "public")),
                                 ("no_distribution", Use("distribute", "internal"))):
            pending = replace(declaration(), allowed_uses=(use,), restrictions=(restriction,))
            record = with_review(pending, reviewer="reviewer", evidence=pending.source)
            self.assertEqual(self.allowed(record, use).reasons, ("restricted_use",))

    def test_optional_notice_and_credit_requirements_are_explicit(self) -> None:
        pending = replace(declaration(), notices=(), attribution="", notices_required=False,
                          attribution_required=False)
        record = with_review(pending, reviewer="reviewer", evidence=pending.source)
        self.assertTrue(self.allowed(record).allowed)
        self.assertEqual(self.allowed(record).attributions[0].notices, ())

    def test_missing_replaced_and_superseded_subjects(self) -> None:
        record = reviewed()
        ledger = RightsLedger()
        self.assertEqual(ledger.preflight_use(record.subject, record.ref, Use("reference")).reasons,
                         ("missing_rights",))
        ledger.register(record)
        for fields in ({"sha256": "b" * 64}, {"revision": "v2"}, {"subject_id": "other"},
                       {"kind": "model"}):
            result = ledger.preflight_use(replace(record.subject, **fields), record.ref,
                                          Use("reference"))
            self.assertIn("subject_mismatch", result.reasons)
        replacement = replace(record, ref=RightsRef(record.ref.record_id, "v2"), review=None)
        ledger.register(replacement)
        self.assertIn("superseded_rights",
                      ledger.preflight_use(record.subject, record.ref, Use("reference")).reasons)
        self.assertFalse(ledger.preflight_use(replacement.subject, replacement.ref,
                                             Use("reference")).allowed)
        self.assertEqual(ledger.lookup(record.ref), record)

    def test_review_binds_every_permission_and_evidence_field(self) -> None:
        record = reviewed()
        mutations = ({"subject": replace(record.subject, sha256="b" * 64)},
                     {"source": EvidenceRef("private://other", DIGEST)},
                     {"terms": EvidenceRef("private://other", DIGEST)},
                     {"source_revision": "v2"}, {"license_id": "MIT"},
                     {"notices": ("Changed",)}, {"attribution": "Changed"},
                     {"notices_required": False}, {"attribution_required": False},
                     {"consent_required": True}, {"consent": None},
                     {"restrictions": ("internal_only",)},
                     {"ref": RightsRef("replacement", "v1")})
        for fields in mutations:
            self.assertIn("stale_review", self.allowed(replace(record, **fields)).reasons)

    def test_record_json_round_trip_strict_malformed_and_tampered_inputs(self) -> None:
        record = reviewed("voice")
        self.assertEqual(RightsRecord.from_json(record.to_json()), record)
        self.assertEqual(SubjectRef.from_dict(record.subject.to_dict()), record.subject)
        self.assertEqual(RightsRef.from_json(record.ref.to_json()), record.ref)
        data = record.to_dict()
        malformed = (None, [], {}, dict(data, unexpected=True), dict(data, schema_version=2), dict(data, schema_version=True),
                     dict(data, notices="string"), dict(data, allowed_uses="string"),
                     dict(data, consent={}), dict(data, notices_required=1),
                     dict(data, license_id="\ud800"), dict(data, attribution="\ud800"),
                     dict(data, restrictions=["ignore-unknown-restriction"]),
                     dict(data, allowed_uses=[{"action": "unknown", "schema_version": 1}]))
        for payload in malformed:
            with self.subTest(payload_type=type(payload)):
                with self.assertRaises(ValueError):
                    RightsRecord.from_json(json.dumps(payload))
        data["attribution"] = "Tampered"
        self.assertIn("stale_review", self.allowed(RightsRecord.from_dict(data)).reasons)
        for raw in ('{"schema_version":1,"schema_version":1}', '{"private-secret-key":NaN}',
                    'not JSON'):
            with self.assertRaises(RightsError) as caught:
                RightsRecord.from_json(raw)
            self.assertEqual(str(caught.exception), "invalid rights JSON")

    def test_malformed_unpinned_mutable_and_unsupported_inputs(self) -> None:
        constructors = (lambda: SubjectRef("alien", "synthetic", "v1", DIGEST),
                        lambda: SubjectRef("model", "synthetic", "main", DIGEST),
                        lambda: SubjectRef("model", "synthetic", "v1", "bad"),
                        lambda: RightsRef("synthetic", "latest"),
                        lambda: Use("reference", "unknown"),
                        lambda: Use("reference", "internal", 1),
                        lambda: replace(declaration(), allowed_uses=[Use("reference")]),
                        lambda: replace(declaration(), source_revision="main"),
                        lambda: replace(declaration(), allowed_uses=(Use("reference"),) * 2),
                        lambda: replace(declaration(), review="approved"),
                        lambda: RightsReview(DIGEST, "reviewer", "forged"))
        for constructor in constructors:
            with self.assertRaises(ValueError):
                constructor()

    def test_private_evidence_never_enters_results_or_error_diagnostics(self) -> None:
        private = "private://home/private-person/credential-sentinel"
        evidence = EvidenceRef(private, DIGEST)
        pending = replace(declaration(), source=evidence, terms=evidence, consent=evidence)
        record = with_review(pending, reviewer="reviewer", evidence=evidence)
        result = self.allowed(record)
        for diagnostic in (repr(evidence), repr(record), repr(record.review), repr(result),
                           result.to_json()):
            self.assertNotIn("credential-sentinel", diagnostic)
            self.assertNotIn("private-person", diagnostic)
        for uri in ("https://user:credential-sentinel@example.org/path",
                    "https://example.org/path?token=credential-sentinel", "https://[invalid",
                    "file:///tmp/private#credential-sentinel", "private://safe/\nsecret"):
            with self.assertRaises(ValueError) as caught:
                EvidenceRef(uri, DIGEST)
            self.assertNotIn("credential-sentinel", str(caught.exception))

    def test_invalid_export_empty_inventory_and_no_partial_attributions(self) -> None:
        record = reviewed()
        ledger = RightsLedger()
        ledger.register(record)
        self.assertEqual(ledger.preflight_export(()).reasons, ("empty_export",))
        request = UseRequest(record.subject, record.ref, Use("reference"))
        result = ledger.preflight_export((request,))
        self.assertEqual(result.reasons, ("invalid_export_action",))
        self.assertEqual(result.attributions, ())
        with self.assertRaises(ValueError):
            ledger.preflight_export([request])

    def test_revocation_history_cannot_be_overwritten_or_regrant_by_reregistering(self) -> None:
        record = reviewed("voice")
        ledger = RightsLedger()
        ledger.register(record)
        ledger.revoke(record.ref, reason="Private withdrawal reason")
        original = ledger.revocation(record.ref)
        with self.assertRaises(RightsError):
            ledger.revoke(record.ref, reason="Different reason")
        with self.assertRaises(RightsError):
            ledger.register(record)
        with self.assertRaises(RightsError):
            ledger.revoke(RightsRef("missing", "v1"), reason="Unknown")
        self.assertEqual(ledger.revocation(record.ref), original)
        self.assertNotIn("withdrawal", repr(original))


if __name__ == "__main__":
    unittest.main()
