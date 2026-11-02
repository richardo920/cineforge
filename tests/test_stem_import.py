"""Original-source selection and byte-identical import using synthetic fixtures."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_speech import EVIDENCE, fixture, reviewed, wav_bytes

from cineforge.audio import RevisionPin
from cineforge.contracts import ContractError
from cineforge.rights import EvidenceRef, RightsLedger, Use
from cineforge.stem_import import StemImportError, import_original_stem


class StemImportTests(unittest.TestCase):
    def setUp(self):
        self.arguments, self.actor, self.payload = fixture()
        self.sheet = self.arguments["sheet"]
        self.ledger = self.arguments["ledger"]

    def import_stem(self, kind="music", *, payload=None, sheet=None, ledger=None, review=EVIDENCE):
        payload = self.payload if payload is None else payload
        return import_original_stem(self.sheet if sheet is None else sheet, kind, payload,
            RevisionPin("original-source", 1, hashlib.sha256(payload).hexdigest()),
            ledger=self.ledger if ledger is None else ledger, original_source_review=review)

    def test_every_non_dialogue_kind_returns_identical_bytes_and_exact_lineage(self):
        for kind in ("ambience", "foley", "effects", "music"):
            result = self.import_stem(kind)
            cue = next(item for item in self.sheet.cues if item.cue_id == kind)
            self.assertIs(result.payload, self.payload)
            self.assertEqual(result.pin.sha256, hashlib.sha256(self.payload).hexdigest())
            self.assertEqual(result.cue, cue)
            self.assertEqual(result.stem.stem_id, cue.stem_id)
            self.assertEqual(result.stem.kind, kind)
            self.assertEqual(result.permission, cue.source)
            self.assertEqual(result.sheet, RevisionPin(self.sheet.sheet_id, self.sheet.revision,
                                                       self.sheet.digest()))
            self.assertEqual(result.original_source_review, EVIDENCE)
            self.assertEqual(result.timebase, self.sheet.timebase)
            self.assertEqual(result.frames * result.timebase.ticks_per_second,
                             result.cue.duration_ticks * result.sample_rate)
            self.assertEqual((result.sample_rate, result.channels, result.frames), (24000, 1, 24000))
            self.assertNotIn("payload=", repr(result))
            self.assertNotIn(EVIDENCE.uri, repr(result))
            self.assertFalse(hasattr(result, "approved"))

    def test_missing_or_other_original_source_review_selection_refuses(self):
        for review in (None, EvidenceRef("private://different/review", "b" * 64)):
            with self.assertRaises(StemImportError):
                self.import_stem(review=review)
        with self.assertRaises(StemImportError):
            import_original_stem(self.sheet, "music", self.payload,
                RevisionPin("source", 1, hashlib.sha256(self.payload).hexdigest()), ledger=self.ledger)

    def test_dialogue_missing_cue_and_incompatible_subject_refuse(self):
        for kind in ("en-cue", "missing"):
            with self.assertRaises(StemImportError):
                self.import_stem(kind)
        cue = self.sheet.cues[0]
        with self.assertRaises(ContractError):
            replace(cue, source=next(x.source for x in self.sheet.cues if x.kind == "music"))

    def test_hash_and_pcm_framing_must_match_before_import(self):
        source = RevisionPin("source", 1, hashlib.sha256(self.payload).hexdigest())
        with self.assertRaises(StemImportError):
            import_original_stem(self.sheet, "music", self.payload[:-1], source,
                ledger=self.ledger, original_source_review=EVIDENCE)
        for payload in (b"PRIVATE-PATH-CANARY", wav_bytes(sample_width=1), wav_bytes(frames=23999)):
            # Match the declaration hash so framing/duration must reject independently.
            cue = next(x for x in self.sheet.cues if x.cue_id == "music")
            binding = replace(cue.source, subject=replace(cue.source.subject,
                sha256=hashlib.sha256(payload).hexdigest()))
            changed = replace(self.sheet, cues=tuple(replace(item, source=binding)
                if item.cue_id == "music" else item for item in self.sheet.cues))
            ledger = RightsLedger()
            ledger.register(reviewed(binding))
            with self.assertRaises(StemImportError) as caught:
                self.import_stem(payload=payload, sheet=changed, ledger=ledger)
            self.assertNotIn("CANARY", str(caught.exception))

    def test_reviewed_stereo_original_is_preserved_without_downmix(self):
        payload = wav_bytes(frames=24000, channels=2)
        cue = next(x for x in self.sheet.cues if x.cue_id == "music")
        binding = replace(cue.source, subject=replace(cue.source.subject,
            sha256=hashlib.sha256(payload).hexdigest()))
        changed = replace(self.sheet, cues=tuple(replace(item, source=binding)
            if item.cue_id == "music" else item for item in self.sheet.cues))
        ledger = RightsLedger()
        ledger.register(reviewed(binding))
        result = self.import_stem(payload=payload, sheet=changed, ledger=ledger)
        self.assertIs(result.payload, payload)
        self.assertEqual(result.channels, 2)

    def test_fresh_rights_and_original_selection_required_on_every_operation(self):
        self.import_stem()
        cue = next(x for x in self.sheet.cues if x.cue_id == "music")
        self.ledger.revoke(cue.source.rights, reason="fixture revocation")
        with self.assertRaises(StemImportError):
            self.import_stem()
        ledger = RightsLedger()
        ledger.register(reviewed(cue.source, allowed_uses=(Use("reference"),)))
        with self.assertRaises(StemImportError):
            self.import_stem(ledger=ledger)
        ledger = RightsLedger()
        ledger.register(replace(reviewed(cue.source), review=None))
        with self.assertRaises(StemImportError):
            self.import_stem(ledger=ledger)

    def test_import_does_no_file_network_process_edit_or_mix(self):
        with patch("builtins.open", side_effect=AssertionError("file I/O")), \
             patch("os.open", side_effect=AssertionError("file I/O")), \
             patch("socket.socket", side_effect=AssertionError("network I/O")), \
             patch("subprocess.Popen", side_effect=AssertionError("process I/O")):
            result = self.import_stem()
        self.assertEqual(result.payload, self.payload)


if __name__ == "__main__":
    unittest.main()
