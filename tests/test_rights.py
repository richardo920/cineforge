"""Synthetic offline permissions; no fixture asserts real-world legal rights."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

from cineforge.rights import (
    EvidenceRef,
    RightsError,
    RightsLedger,
    RightsRecord,
    RightsRef,
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


if __name__ == "__main__":
    unittest.main()
