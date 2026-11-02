"""Synthetic contract/byte fixtures only; never actual human or native media approval."""
from __future__ import annotations

import hashlib
import json
import traceback
import unittest
from dataclasses import fields, replace
from unittest.mock import patch

from test_audio_mix import IDENTITY, Meter, scene
from test_speech import EVIDENCE

from cineforge.audio import RevisionPin
from cineforge.audio_mix import BROADCAST_LINEAR, mix_scene
from cineforge.audio_review import (
    HUMAN_CHECKLIST,
    TECHNICAL_CHECKLIST,
    ClipInput,
    LanguageClip,
    MediaInput,
    OwnerReview,
    ReviewCheck,
    ReviewError,
    ReviewRecipe,
    ReviewSnapshot,
    TechnicalReview,
    assess_review,
    require_current_review,
    snapshot_for_review,
)
from cineforge.contracts import ContractError
from cineforge.rights import EvidenceRef
from cineforge.speech import SpeechArtifact


def pin(name, payload=b"synthetic-declaration"):
    return RevisionPin(name, 1, hashlib.sha256(payload).hexdigest())


def fixture():
    args = scene()
    mixed = mix_scene(**args, preset=BROADCAST_LINEAR, output_id="mix", output_revision=1,
                      tool=Meter(), expected_tool=IDENTITY)
    recipe = ReviewRecipe(pin("speech-recipe"), pin("video-model"), pin("video-workflow"),
                          pin("video-runtime"))
    sources = {item.request.language: item for item in args["sources"]
               if type(item) is SpeechArtifact}
    clips = tuple(ClipInput(language, sources[native_language],
        MediaInput(pin("video-" + language, payload), "video/webm", payload), recipe)
        for language, native_language, payload in (("en", "English", b"synthetic-EN-container"),
                                                   ("zh", "Chinese", b"synthetic-ZH-container")))
    delivered = MediaInput(pin("delivered", b"synthetic-delivery"), "video/mp4",
                           b"synthetic-delivery")
    inputs = dict(sheet=args["sheet"], mix=mixed, delivered=delivered, clips=clips)
    return snapshot_for_review("review-scene", 1, **inputs), inputs


def owner(snapshot, **changes):
    options = dict(review_id="owner-review", revision=1, snapshot=snapshot.pin(),
        reviewer="synthetic-owner", reviewed_at_ns=1_800_000_000_000_000_000,
        evidence=EVIDENCE, checks=tuple(ReviewCheck(key, "pass") for key in HUMAN_CHECKLIST),
        decision="approved")
    options.update(changes)
    return OwnerReview(**options)


def native_shaped_contract(snapshot):
    # Explicit metadata-only stand-in. It is NOT native evidence or a production receipt.
    return replace(snapshot, mix_status="native_measured_pcm",
                   mix_report=pin("synthetic-native-shaped-report"))


class Verifier:
    """A synthetic trusted-code stand-in, not an implemented native receipt verifier."""

    def __init__(self, *, status="pass", snapshot=None, error=None):
        self.status, self.snapshot, self.error = status, snapshot, error
        self.calls = []

    def verify(self, snapshot):
        self.calls.append(snapshot)
        if self.error is not None:
            try:
                raise OSError("PRIVATE-CANARY/media.wav")
            except OSError:
                raise self.error("PRIVATE-CANARY transcript")
        return TechnicalReview(snapshot.pin() if self.snapshot is None else self.snapshot,
            tuple(ReviewCheck(key, self.status) for key in TECHNICAL_CHECKLIST), EVIDENCE)


class AudioReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot, cls.inputs = fixture()

    def test_snapshot_factory_binds_supplied_actual_bytes_and_existing_lineage(self):
        snapshot = self.snapshot
        self.assertEqual(snapshot.delivered.pin, self.inputs["delivered"].pin)
        self.assertEqual(snapshot.mix.pin, self.inputs["mix"].pin)
        self.assertEqual(snapshot.mix_report.sha256, self.inputs["mix"].report.digest())
        self.assertEqual(snapshot.mix_settings_sha256, self.inputs["mix"].report.settings.digest())
        self.assertEqual(snapshot.dependencies, self.inputs["mix"].report.dependencies)
        self.assertEqual(tuple(item.language for item in snapshot.clips), ("en", "zh"))
        for clip, supplied in zip(snapshot.clips, self.inputs["clips"]):
            self.assertEqual(clip.speech.pin, supplied.speech.pin)
            self.assertEqual(clip.request.sha256, supplied.speech.request.digest())
            self.assertEqual(clip.utterance, supplied.speech.request.utterance)
            self.assertEqual(clip.recipe, supplied.recipe)
            self.assertEqual(clip.speech_origin, "unverified_injected_runtime")
            clip.video.check_bytes(supplied.video.payload)
        self.assertEqual(snapshot.clips[0].voice, snapshot.clips[1].voice)
        self.assertEqual(ReviewSnapshot.from_json(snapshot.to_json()), snapshot)
        # These video bytes intentionally are not decoded or certified by the factory.
        self.assertEqual(snapshot.mix_status, "synthetic_fixture_only")

    def test_checksums_verify_supplied_payloads_and_sizes_without_io(self):
        payload = self.inputs["delivered"].payload
        for wrong in (payload + b"x", b"x" * len(payload), None, bytearray(payload)):
            with self.assertRaises(ReviewError):
                self.snapshot.delivered.check_bytes(wrong)
        for field in ("delivered", "video", "speech"):
            options = dict(self.inputs)
            if field == "delivered":
                options[field] = replace(options[field], payload=b"PRIVATE-CANARY")
            elif field == "video":
                options["clips"] = (replace(options["clips"][0],
                    video=replace(options["clips"][0].video, payload=b"PRIVATE-CANARY")),
                    options["clips"][1])
            else:
                forged = object.__new__(SpeechArtifact)
                original = options["clips"][0].speech
                for entry in fields(original):
                    object.__setattr__(forged, entry.name,
                        b"PRIVATE-CANARY" if entry.name == "payload"
                        else getattr(original, entry.name))
                options["clips"] = (replace(options["clips"][0], speech=forged),
                                     options["clips"][1])
            try:
                snapshot_for_review("snapshot", 1, **options)
            except ReviewError as error:
                self.assertIsNone(error.__context__)
                self.assertNotIn("CANARY", str(error))
            else:
                self.fail("wrong supplied media accepted")
        with patch("builtins.open", side_effect=AssertionError("file")), \
             patch("os.open", side_effect=AssertionError("file")), \
             patch("socket.socket", side_effect=AssertionError("network")), \
             patch("subprocess.Popen", side_effect=AssertionError("native")):
            self.assertEqual(snapshot_for_review("review-scene", 1, **self.inputs), self.snapshot)

    def test_missing_duplicate_unsupported_or_unordered_language_pairs_refuse(self):
        clips = self.inputs["clips"]
        for wrong in ((), clips[:1], (clips[0], clips[0]), tuple(reversed(clips)),
                      (replace(clips[0], language="fr"), clips[1])):
            with self.assertRaises(ReviewError):
                snapshot_for_review("snapshot", 1, **dict(self.inputs, clips=wrong))
        with self.assertRaises(ReviewError):
            replace(self.snapshot.clips[0], language="ja")
        with self.assertRaises(ReviewError):
            replace(self.snapshot, clips=self.snapshot.clips[:1])

    def test_factory_refuses_changed_sheet_request_mix_maps_and_canonical_identity(self):
        sheet, mixed = self.inputs["sheet"], self.inputs["mix"]
        for wrong_sheet in (replace(sheet, revision=2),
                            replace(sheet, picture=replace(sheet.picture, revision=2)),
                            replace(sheet, script=replace(sheet.script, revision=2))):
            with self.assertRaises(ReviewError):
                snapshot_for_review("snapshot", 1, **dict(self.inputs, sheet=wrong_sheet))
        changed_report = replace(mixed.report, inputs=tuple(replace(item,
            metadata_sha256="b" * 64) if item.source_kind == "unverified_injected_runtime"
            else item for item in mixed.report.inputs))
        with self.assertRaises(ReviewError):
            snapshot_for_review("snapshot", 1,
                **dict(self.inputs, mix=replace(mixed, report=changed_report)))
        for change in (dict(frames=1), dict(source_kind="original_import")):
            report = replace(mixed.report, inputs=tuple(replace(item, **change)
                if item.source_kind == "unverified_injected_runtime" else item
                for item in mixed.report.inputs))
            with self.assertRaises(ReviewError):
                snapshot_for_review("snapshot", 1,
                    **dict(self.inputs, mix=replace(mixed, report=report)))
        with self.assertRaises(ReviewError):
            replace(self.snapshot, dependencies=replace(self.snapshot.dependencies, speech=()))
        with self.assertRaises(ReviewError):
            replace(self.snapshot, clips=(self.snapshot.clips[0], replace(self.snapshot.clips[1],
                reference=replace(self.snapshot.clips[1].reference, revision=2))))

    def test_every_media_dependency_recipe_report_or_settings_change_invalidates_review(self):
        current = native_shaped_contract(self.snapshot)
        record = owner(current)
        first = current.clips[0]
        mutations = [replace(current, revision=2),
            replace(current, delivered=replace(current.delivered,
                pin=replace(current.delivered.pin, sha256="b" * 64))),
            replace(current, mix=replace(current.mix, pin=replace(current.mix.pin, revision=2))),
            replace(current, mix_report=replace(current.mix_report, sha256="c" * 64)),
            replace(current, mix_settings_sha256="d" * 64)]
        for name in ("script", "picture", "cue_sheet"):
            dependencies = replace(current.dependencies,
                **{name: replace(getattr(current.dependencies, name), revision=2)})
            mutations.append(replace(current, dependencies=dependencies))
        for name in ("speech", "video_model", "video_workflow", "video_runtime"):
            recipe = replace(first.recipe, **{name: replace(getattr(first.recipe, name), revision=2)})
            mutations.append(replace(current, clips=(replace(first, recipe=recipe), current.clips[1])))
        for mutated in mutations:
            verifier = Verifier()
            self.assertEqual(assess_review(current, record, current=mutated, verifier=verifier).reasons,
                             ("stale_snapshot",))
            self.assertFalse(verifier.calls)
            self.assertNotEqual(mutated.digest(), current.digest())

    def test_explicit_human_checklist_and_decisions_roundtrip_and_bind_all_evidence(self):
        record = owner(self.snapshot)
        self.assertEqual(len(record.checks), 10)
        self.assertEqual(OwnerReview.from_json(record.to_json()), record)
        self.assertEqual(record.method, "human_listening_and_visual")
        for changes in (dict(reviewer="other-owner"), dict(reviewed_at_ns=record.reviewed_at_ns + 1),
                        dict(evidence=EvidenceRef("private://different/review", "b" * 64)),
                        dict(snapshot=replace(record.snapshot, revision=2)),
                        dict(checks=(replace(record.checks[0], status="fail"), *record.checks[1:]))):
            self.assertNotEqual(replace(record, **changes).digest(), record.digest())
        for wrong in (record.checks[:-1], (record.checks[0], *record.checks[:-1]),
                      tuple(reversed(record.checks))):
            with self.assertRaises(ReviewError):
                replace(record, checks=wrong)
        for bad in ("asr_score", "energy_alignment", "model_review", "native_video_en"):
            with self.assertRaises(ContractError):
                checks = tuple(ReviewCheck(bad, "pass") if index == 0 else item
                               for index, item in enumerate(record.checks))
                replace(record, checks=checks)

    def test_missing_stale_pending_and_every_failed_human_item_refuse_before_verifier(self):
        current = native_shaped_contract(self.snapshot)
        valid = owner(current)
        records = [None, replace(valid, decision="pending"),
                   replace(valid, decision="changes_requested"),
                   replace(valid, snapshot=replace(valid.snapshot, sha256="b" * 64))]
        for index in range(len(valid.checks)):
            for status in ("pending", "fail", "skipped", "unavailable"):
                records.append(replace(valid, checks=tuple(replace(item, status=status)
                    if number == index else item for number, item in enumerate(valid.checks))))
        for record in records:
            verifier = Verifier()
            result = assess_review(current, record, current=current, verifier=verifier)
            self.assertFalse(result.current)
            self.assertFalse(verifier.calls)
            with self.assertRaises(ReviewError):
                require_current_review(current, record, current=current, verifier=verifier)
        verifier = Verifier()
        with self.assertRaises(ReviewError):
            assess_review(current, {"approved": True}, current=current, verifier=verifier)
        self.assertFalse(verifier.calls)

    def test_default_missing_fake_and_unverified_technical_evidence_never_passes(self):
        current = native_shaped_contract(self.snapshot)
        record = owner(current)
        self.assertFalse(assess_review(current, record, current=current).current)
        verifier = Verifier()
        self.assertEqual(assess_review(self.snapshot, owner(self.snapshot), current=self.snapshot,
            verifier=verifier).reasons, ("fixture_media_not_native_proof",))
        self.assertFalse(verifier.calls)
        fake = replace(current.clips[0], speech_origin="synthetic_fixture")
        fixture = replace(current, clips=(fake, current.clips[1]))
        self.assertFalse(assess_review(fixture, owner(fixture), current=fixture,
                                     verifier=verifier).current)
        self.assertFalse(verifier.calls)
        self.assertFalse(assess_review(current, record, current=current,
                                      verifier={"native_ok": True}).current)
        self.assertFalse(assess_review(current, record, current=current,
            verifier=Verifier(snapshot=replace(current.pin(), revision=2))).current)
        for status in ("fail", "pending", "skipped", "unavailable"):
            self.assertFalse(assess_review(current, record, current=current,
                                          verifier=Verifier(status=status)).current)

    def test_malformed_technical_responses_refuse_without_retry_or_private_context(self):
        current = native_shaped_contract(self.snapshot)
        record = owner(current)
        forged = object.__new__(TechnicalReview)
        object.__setattr__(forged, "snapshot", "PRIVATE-CANARY")
        object.__setattr__(forged, "checks", ())
        object.__setattr__(forged, "evidence", EVIDENCE)
        for response in (None, True, {"native_ok": True}, forged):
            class Malformed:
                calls = 0

                def verify(self, snapshot):
                    self.calls += 1
                    return response

            verifier = Malformed()
            try:
                require_current_review(current, record, current=current, verifier=verifier)
            except ReviewError as error:
                self.assertNotIn("CANARY", traceback.format_exc())
                self.assertIsNone(error.__context__)
            else:
                self.fail("malformed technical response accepted")
            self.assertEqual(verifier.calls, 1)

    def test_trusted_verifier_is_rechecked_and_positive_contract_fixture_grants_no_authority(self):
        # Both the owner and native-shaped evidence are synthetic assertions here.
        current = native_shaped_contract(self.snapshot)
        record = owner(current)
        verifier = Verifier()
        result = require_current_review(current, record, current=current, verifier=verifier)
        self.assertTrue(result.current)
        self.assertEqual(verifier.calls, [current])
        self.assertFalse({"export", "execute", "approved"} & {item.name for item in fields(result)})
        verifier.status = "fail"
        self.assertFalse(assess_review(current, record, current=current, verifier=verifier).current)
        self.assertEqual(verifier.calls, [current, current])
        with self.assertRaises(ReviewError):
            TechnicalReview(current.pin(), (), EVIDENCE)

    def test_verifier_failures_are_private_safe_single_call_and_never_typeerror_dispatch(self):
        current = native_shaped_contract(self.snapshot)
        record = owner(current)
        for exception in (TypeError, ValueError, RuntimeError, SystemExit):
            verifier = Verifier(error=exception)
            try:
                require_current_review(current, record, current=current, verifier=verifier)
            except ReviewError as error:
                self.assertNotIn("CANARY", traceback.format_exc())
                self.assertIsNone(error.__context__)
                self.assertIsNone(error.__cause__)
            else:
                self.fail("verifier error accepted")
            self.assertEqual(len(verifier.calls), 1)
        verifier = Verifier(error=KeyboardInterrupt)
        try:
            assess_review(current, record, current=current, verifier=verifier)
        except KeyboardInterrupt as error:
            self.assertIsNone(error.__context__)
            self.assertNotIn("CANARY", traceback.format_exc())
        else:
            self.fail("cancellation lost")
        self.assertEqual(len(verifier.calls), 1)

    def test_strict_json_unknown_duplicate_bool_nonfinite_overflow_and_privacy(self):
        snapshot = self.snapshot
        record = owner(snapshot)
        for cls, original in ((ReviewSnapshot, snapshot), (OwnerReview, record)):
            for field, value in (("schema_version", True), ("revision", True),
                                 ("revision", 2**63), ("PRIVATE-CANARY", "PRIVATE-CANARY")):
                data = original.to_dict()
                data[field] = value
                try:
                    cls.from_dict(data)
                except ReviewError as error:
                    self.assertIsNone(error.__context__)
                    self.assertNotIn("CANARY", traceback.format_exc())
                else:
                    self.fail("bad review metadata accepted")
            raw = original.to_json()
            for invalid in (raw[:-1] + ',"schema_version":1}',
                            raw.replace('"revision":1', '"revision":NaN', 1),
                            'PRIVATE-CANARY', '{"score":Infinity}'):
                with self.assertRaises(ReviewError):
                    cls.from_json(invalid)
        for bad in (True, float("nan"), float("inf"), 0, 2**63):
            with self.assertRaises(ContractError):
                replace(record, reviewed_at_ns=bad)
        for bad in (True, float("nan"), 0, 128 * 1024**2 + 1):
            with self.assertRaises(ContractError):
                replace(snapshot.delivered, size_bytes=bad)
        for bad in (True, "approved", "PRIVATE-CANARY"):
            with self.assertRaises(ReviewError):
                ReviewCheck("balance", bad)
        with self.assertRaises(ReviewError):
            replace(record, method="asr_score")
        self.assertNotIn(EVIDENCE.uri, repr(record))
        self.assertNotIn(self.inputs["delivered"].payload.decode(), repr(self.inputs["delivered"]))
        self.assertNotIn(self.inputs["clips"][0].speech.request.text, repr(self.snapshot))
        self.assertNotIn(self.inputs["clips"][0].speech.request.ref_text, repr(self.snapshot))
        self.assertNotIn("payload=", repr(self.inputs["clips"][0]))
        self.assertEqual(json.loads(record.to_json())["reviewer"], "synthetic-owner")
        with self.assertRaises(ReviewError):
            LanguageClip.from_dict(dict(self.snapshot.clips[0].to_dict(), native_ok=True))


if __name__ == "__main__":
    unittest.main()
