"""Synthetic declarations only: no recording, consent, approval or generated media."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
import unittest
from dataclasses import FrozenInstanceError, replace

from cineforge.audio import (
    AudioError,
    AudioTimebase,
    Cue,
    CueSheet,
    DependencyBinding,
    DialogueScript,
    MediaDependencies,
    PronunciationHint,
    RevisionPin,
    ShotTiming,
    Stem,
    Utterance,
    VoiceIdentity,
)
from cineforge.characters import (
    CharacterRegistry,
    CharacterRevision,
    ReferenceArtifact,
    RightsBinding,
    Wardrobe,
    pin_shot,
    with_character_review,
)
from cineforge.contracts import ContractError, Project, ProjectPlan, Scene, Shot
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
OTHER = "b" * 64
EVIDENCE = EvidenceRef("private://synthetic/evidence", DIGEST)


def permission(name: str, kind: str = "asset") -> RightsBinding:
    return RightsBinding(SubjectRef(kind, name, "v1", DIGEST), RightsRef(name, "v1"))


def character() -> CharacterRevision:
    refs = (ReferenceArtifact("face", "identity", EVIDENCE, 16, "image/png",
                              permission("face", "character_reference")),
            ReferenceArtifact("voice", "voice", EVIDENCE, 16, "audio/wav",
                              permission("voice", "voice")))
    return with_character_review(CharacterRevision(
        "synthetic-actor", 1, "fictional", "An original synthetic actor",
        Wardrobe("coat", "Blue coat"), refs, voice_reference_id="voice"),
        reviewer="synthetic-reviewer", evidence=EVIDENCE)


def sheet() -> CueSheet:
    actor = character()
    voice = VoiceIdentity("voice", 1, actor.references[1].permission, "voice")
    english = Utterance("line-en", 1, "Hello, Cineforge!", "en", "actor", actor.pin(), voice,
                        "warm", "clearly", (PronunciationHint("Cineforge", "sin-ee-forj"),),
                        "scene", "shot", 500, 1500)
    mandarin = replace(english, utterance_id="line-zh", text="你好，电影世界！", language="zh",
                       pronunciation=(PronunciationHint("你好", "nǐ hǎo"),), start_ticks=2500)
    cues = (Cue("ambience", "ambience", "ambience", "scene", "shot", 0, 5000,
                permission("room")),
            Cue("music", "music", "music", "scene", "shot", 0, 5000,
                permission("score", "music")),
            Cue("dialogue-en", "dialogue", "dialogue", "scene", "shot", 500, 1500,
                voice.permission, "line-en"),
            Cue("foley", "foley", "foley", "scene", "shot", 1000, 1000,
                permission("step")),
            Cue("effects", "effects", "effects", "scene", "shot", 1500, 1000,
                permission("door")),
            Cue("dialogue-zh", "dialogue", "dialogue", "scene", "shot", 2500, 1500,
                voice.permission, "line-zh"))
    return CueSheet("sound", 1, "project", "scene", AudioTimebase(1000, 5000),
                    DialogueScript("script", 1, (english, mandarin)),
                    RevisionPin("picture", 1, DIGEST),
                    tuple(Stem(kind, kind) for kind in
                          ("dialogue", "ambience", "foley", "effects", "music")),
                    (ShotTiming("shot", 1, 0, 5000),), cues)


def context(sound: CueSheet) -> tuple[ProjectPlan, CharacterRegistry, RightsLedger]:
    actor = character()
    registry = CharacterRegistry()
    registry.register(actor)
    shot = pin_shot(Shot("shot", "scene", 1, "Synthetic actor speaks", None, 5,
                         "synthetic-model", "v1", DIGEST), actor.pin())
    plan = ProjectPlan(Project("project", 1, "Synthetic"), (Scene("scene", "project", 1),),
                       (shot,))
    bindings = set(cue.source for cue in sound.cues)
    bindings.update(ref.permission for ref in actor.references)
    ledger = RightsLedger()
    for binding in bindings:
        ledger.register(with_review(RightsRecord(
            binding.rights, binding.subject, source=EVIDENCE, source_revision="v1",
            license_id="LicenseRef-Synthetic", terms=EVIDENCE, notices=("Synthetic",),
            attribution="Synthetic", consent=EVIDENCE,
            allowed_uses=(Use("generate"), Use("reference"))),
            reviewer="synthetic-reviewer", evidence=EVIDENCE))
    return plan, registry, ledger


class AudioTests(unittest.TestCase):
    def test_exact_bilingual_round_trip_hash_and_frozen_records(self) -> None:
        sound = sheet()
        raw = sound.to_json()
        self.assertIn("你好，电影世界！", raw)
        self.assertEqual(CueSheet.from_json(raw), sound)
        self.assertEqual(CueSheet.from_json(raw).to_json(), raw)
        self.assertEqual(sound.digest(), hashlib.sha256(raw.encode("utf-8")).hexdigest())
        self.assertEqual(sound.script.utterances[0].text, "Hello, Cineforge!")
        self.assertEqual(sound.script.utterances[1].text, "你好，电影世界！")
        self.assertNotEqual(replace(sound.script.utterances[0], text="Hello, Cineforge! ").digest(),
                            sound.script.utterances[0].digest())
        with self.assertRaises(FrozenInstanceError):
            sound.revision = 2  # type: ignore[misc]

    def test_every_record_codec_in_fresh_processes_has_stable_identity(self) -> None:
        sound = sheet()
        inputs = sound.dependencies(speech=(RevisionPin("speech-en", 1, DIGEST),
                                            RevisionPin("speech-zh", 1, OTHER)))
        binding = DependencyBinding("sync_check", RevisionPin("sync", 1, OTHER), inputs)
        records = (sound, sound.timebase, sound.script, sound.script.utterances[0],
                   sound.script.utterances[0].voice, sound.script.utterances[0].pronunciation[0],
                   sound.stems[0], sound.shots[0], sound.cues[0], sound.picture, inputs, binding)
        data = json.dumps([(type(item).__name__, item.to_dict(), item.digest()) for item in records])
        code = (
            "import json,sys,cineforge.audio as a; "
            "records=json.load(sys.stdin); "
            "parsed=[(getattr(a,name).from_dict(value),digest) for name,value,digest in records]; "
            "assert all(type(item).from_json(item.to_json())==item and item.digest()==digest "
            "for item,digest in parsed)"
        )
        for seed in ("0", "1", "random"):
            result = subprocess.run([sys.executable, "-c", code], input=data, text=True,
                                    capture_output=True, timeout=10,
                                    env=dict(os.environ, PYTHONHASHSEED=seed))
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_overlaps_are_intentional_and_end_boundary_is_inclusive(self) -> None:
        sound = sheet()
        self.assertEqual(sound.cues[0].start_ticks + sound.cues[0].duration_ticks, 5000)
        self.assertGreater(sound.cues[0].duration_ticks, sound.cues[2].start_ticks)
        plan, registry, ledger = context(sound)
        result = sound.preflight(plan, registry, ledger, current_picture=sound.picture)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reasons, ())

    def test_scene_cues_cross_two_real_shots_without_a_fake_shot(self) -> None:
        sound = sheet()
        # Each real shot is half the scene; both bilingual lines remain shot-bound.
        script = replace(sound.script, utterances=(sound.script.utterances[0],
            replace(sound.script.utterances[1], shot_id="shot-2")))
        cues = tuple(replace(cue, shot_id=None) if cue.kind in {"ambience", "music"}
                     else replace(cue, shot_id="shot-2") if cue.utterance_id == "line-zh"
                     else cue for cue in sound.cues)
        sound = replace(sound, script=script,
                        shots=(ShotTiming("shot", 1, 0, 2500),
                               ShotTiming("shot-2", 1, 2500, 2500)), cues=cues)
        plan, registry, ledger = context(sound)
        first = replace(plan.shots[0], duration_seconds=2.5)
        plan = replace(plan, shots=(first, replace(first, shot_id="shot-2")))
        self.assertEqual(len(plan.shots), 2)
        self.assertTrue(sound.preflight(plan, registry, ledger,
                                        current_picture=sound.picture).allowed)
        restored = CueSheet.from_json(sound.to_json())
        self.assertEqual(restored, sound)
        self.assertEqual(restored.digest(), sound.digest())
        for cue in restored.cues[:2]:
            self.assertIsNone(cue.shot_id)
            self.assertEqual(cue.start_ticks + cue.duration_ticks, sound.timebase.duration_ticks)
        for fields in ({"scene_id": "missing"}, {"shot_id": "missing"}, {"shot_id": "shot"},
                       {"start_ticks": 1}, {"duration_ticks": 5001}):
            with self.assertRaises(ContractError):
                replace(sound, cues=(replace(sound.cues[0], **fields), *sound.cues[1:]))
        with self.assertRaises(ContractError):
            replace(sound.cues[2], shot_id=None)
        with self.assertRaises(ContractError):
            replace(sound.script.utterances[0], shot_id=None)
        with self.assertRaises(AudioError):
            Cue.from_dict({key: value for key, value in sound.cues[0].to_dict().items()
                           if key != "shot_id"})
        other = ProjectPlan(plan.project, (Scene("other", "project", 1),),
                            tuple(replace(shot, scene_id="other") for shot in plan.shots))
        self.assertIn("unknown_project_or_scene", sound.preflight(
            other, registry, ledger, current_picture=sound.picture).reasons)

    def test_invalid_timing_bool_floats_nonfinite_bounds_and_overflow(self) -> None:
        sound = sheet()
        for invalid in (True, False, -1, 1.0, float("nan"), float("inf"), -float("inf"),
                        2**63, 10**1000, "1000", None):
            for field in ("start_ticks", "duration_ticks"):
                with self.subTest(field=field, value=type(invalid)):
                    with self.assertRaises(ContractError):
                        replace(sound.cues[0], **{field: invalid})
            for field in ("ticks_per_second", "duration_ticks"):
                with self.assertRaises(ContractError):
                    replace(sound.timebase, **{field: invalid})
        with self.assertRaises(ContractError):
            replace(sound.cues[0], duration_ticks=0)
        with self.assertRaises(ContractError):
            AudioTimebase(0, 1)
        with self.assertRaises(ContractError):
            AudioTimebase(1, 0)
        with self.assertRaises(ContractError):
            ShotTiming("shot", 1, 2**63 - 2, 2)
        with self.assertRaises(ContractError):
            replace(sound, timebase=AudioTimebase(1000, 4999))
        with self.assertRaises(ContractError):
            replace(sound, cues=(replace(sound.cues[0], duration_ticks=5001), *sound.cues[1:]))
        with self.assertRaises(ContractError):
            replace(sound, shots=(ShotTiming("shot", 1, 100, 4900),))

    def test_reject_unknown_missing_duplicate_ids_order_and_internal_links(self) -> None:
        sound = sheet()
        variants = ({"stems": sound.stems * 2}, {"shots": sound.shots * 2},
                    {"cues": sound.cues * 2}, {"stems": ()}, {"shots": ()}, {"cues": ()},
                    {"cues": tuple(reversed(sound.cues))},
                    {"cues": (replace(sound.cues[0], stem_id="missing"), *sound.cues[1:])},
                    {"cues": (replace(sound.cues[0], stem_id="music"), *sound.cues[1:])},
                    {"cues": (replace(sound.cues[0], scene_id="missing"), *sound.cues[1:])},
                    {"cues": (replace(sound.cues[0], shot_id="missing"), *sound.cues[1:])},
                    {"cues": tuple(replace(item, utterance_id="missing") if item.kind == "dialogue"
                                   else item for item in sound.cues)},
                    {"script": replace(sound.script, utterances=())},
                    {"cues": tuple(item for item in sound.cues if item.utterance_id != "line-en")})
        for fields in variants:
            with self.assertRaises(ContractError):
                replace(sound, **fields)
        with self.assertRaises(ContractError):
            replace(sound.script, utterances=sound.script.utterances * 2)
        with self.assertRaises(ContractError):
            replace(sound.cues[0], utterance_id="line-en")
        with self.assertRaises(ContractError):
            replace(sound.cues[2], utterance_id=None)
        with self.assertRaises(ContractError):
            replace(sound.cues[1], source=permission("wrong"))
        with self.assertRaises(ContractError):
            replace(sound, cues=tuple(replace(item, duration_ticks=1000) if item.kind == "dialogue"
                                      else item for item in sound.cues))

    def test_strict_baseline_language_text_pronunciation_and_identity(self) -> None:
        line = sheet().script.utterances[0]
        for value in ("fr", "zh-CN", "en-US", "mandarin", "EN", "", True, None, [], "en-zh"):
            with self.assertRaises(ContractError):
                replace(line, language=value)
        for value in ("", "  ", "hi\x00there", "private-canary\ud800", True, None):
            for name in ("text", "emotion", "delivery"):
                with self.assertRaises(ContractError):
                    replace(line, **{name: value})
        with self.assertRaises(ContractError):
            replace(line, pronunciation=(PronunciationHint("Missing", "missing"),))
        with self.assertRaises(ContractError):
            replace(line.voice, permission=permission("not-voice"))
        with self.assertRaises(ContractError):
            replace(line, character=None)
        with self.assertRaises(ContractError):
            replace(line, voice=None)
        with self.assertRaises(ContractError):
            replace(sheet().script, utterances=(line, replace(line, utterance_id="other",
                    character=replace(line.character, revision=2))))
        for name in ("utterance_id", "speaker_id", "scene_id", "shot_id"):
            with self.assertRaises(ContractError):
                replace(line, **{name: ""})
        for value in (True, 0, 2**63):
            with self.assertRaises(ContractError):
                replace(line, text_revision=value)

    def test_missing_current_project_scene_shot_and_picture_fail_closed(self) -> None:
        sound = sheet()
        plan, registry, ledger = context(sound)
        variants = ((replace(sound, project_id="unknown"), plan, "unknown_project_or_scene"),
                    (sound, replace(plan, shots=(replace(plan.shots[0], revision=2),)),
                     "stale_shot_timing"),
                    (sound, replace(plan, shots=(replace(plan.shots[0], duration_seconds=6),)),
                     "stale_shot_timing"),
                    (sound, replace(plan, shots=(replace(plan.shots[0], shot_id="other"),)),
                     "unknown_shot"),
                    (sound, replace(plan, shots=(replace(plan.shots[0], character_revision=None),)),
                     "shot_character_mismatch"))
        for candidate, current, reason in variants:
            result = candidate.preflight(current, registry, ledger, current_picture=sound.picture)
            self.assertFalse(result.allowed)
            self.assertIn(reason, result.reasons)
        for picture in (replace(sound.picture, input_id="other"),
                        replace(sound.picture, revision=2), replace(sound.picture, sha256=OTHER)):
            self.assertEqual(sound.preflight(plan, registry, ledger,
                                            current_picture=picture).reasons, ("stale_picture",))
        plan = ProjectPlan(plan.project, (Scene("other", "project", 1),),
                           (replace(plan.shots[0], scene_id="other"),))
        result = sound.preflight(plan, registry, ledger, current_picture=sound.picture)
        self.assertIn("unknown_project_or_scene", result.reasons)
        self.assertIn("unknown_shot", result.reasons)

    def test_missing_changed_unreviewed_characters_and_voice_references(self) -> None:
        sound = sheet()
        plan, registry, ledger = context(sound)
        result = sound.preflight(plan, CharacterRegistry(), ledger, current_picture=sound.picture)
        self.assertIn("missing_or_changed_character", result.reasons)
        for actor, reason in ((replace(character(), identity_description="Changed"),
                               "missing_or_changed_character"),
                              (replace(character(), review=None), "unreviewed_character")):
            current = CharacterRegistry()
            current.register(actor)
            result = sound.preflight(plan, current, ledger, current_picture=sound.picture)
            self.assertIn(reason, result.reasons)
        line = sound.script.utterances[0]
        for ref in ("missing", "face"):
            voice = replace(line.voice, reference_id=ref)
            script = replace(sound.script, utterances=tuple(replace(item, voice=voice)
                                                            for item in sound.script.utterances))
            changed = replace(sound, script=script)
            self.assertIn("missing_or_changed_voice_reference", changed.preflight(
                plan, registry, ledger, current_picture=sound.picture).reasons)

    def test_current_rights_use_consent_revocation_and_no_embedded_authority(self) -> None:
        sound = sheet()
        plan, registry, ledger = context(sound)
        self.assertIn("missing_rights", sound.preflight(plan, registry, RightsLedger(),
                                                       current_picture=sound.picture).reasons)
        self.assertIn("use_not_permitted", sound.preflight(plan, registry, ledger,
                      current_picture=sound.picture, use=Use("export", "public")).reasons)
        voice_rights = sound.script.utterances[0].voice.permission.rights
        grant = ledger.lookup(voice_rights)
        assert grant is not None
        for mutation, reason in (({"consent": None}, "missing_consent"),
                                 ({"review": None}, "unreviewed_rights")):
            _, _, current = context(sound)
            # A new central ledger, without replacing the immutable existing revision.
            replacement = RightsLedger()
            for binding in {cue.source for cue in sound.cues} | {
                    ref.permission for ref in character().references}:
                existing = current.lookup(binding.rights)
                assert existing is not None
                replacement.register(replace(existing, **mutation) if binding.rights == voice_rights
                                     else existing)
            self.assertIn(reason, sound.preflight(plan, registry, replacement,
                                                 current_picture=sound.picture).reasons)
        ledger.revoke(voice_rights, reason="Synthetic withdrawal")
        self.assertIn("revoked_rights", sound.preflight(plan, registry, ledger,
                                                      current_picture=sound.picture).reasons)
        for key in ("approved", "execution_ready", "consent", "spend_authorized", "review"):
            with self.assertRaises(AudioError):
                CueSheet.from_dict(dict(sound.to_dict(), **{key: True}))

    def test_synthetic_voice_needs_no_recording_but_still_requires_current_rights(self) -> None:
        sound = sheet()
        actor = character()
        actor = with_character_review(replace(actor, references=(actor.references[0],),
                                              voice_reference_id=None),
                                       reviewer="synthetic-reviewer", evidence=EVIDENCE)
        voice = replace(sound.script.utterances[0].voice, reference_id=None)
        script = replace(sound.script, utterances=tuple(replace(item, character=actor.pin(),
                                                               voice=voice)
                                                        for item in sound.script.utterances))
        sound = replace(sound, script=script)
        plan, _, ledger = context(sound)
        registry = CharacterRegistry()
        registry.register(actor)
        plan = replace(plan, shots=(pin_shot(plan.shots[0], actor.pin()),))
        self.assertTrue(sound.preflight(plan, registry, ledger,
                                        current_picture=sound.picture).allowed)
        self.assertEqual(CueSheet.from_json(sound.to_json()), sound)
        ledger.revoke(voice.permission.rights, reason="Synthetic withdrawn rights")
        self.assertIn("revoked_rights", sound.preflight(plan, registry, ledger,
                                                      current_picture=sound.picture).reasons)

    def test_voice_permission_cannot_authorize_unrelated_identity_or_revision(self) -> None:
        voice = sheet().script.utterances[0].voice
        for fields in ({"voice_id": "other-voice"}, {"revision": 2},
                       {"permission": permission("other-voice", "voice")},
                       {"permission": replace(voice.permission, subject=replace(
                           voice.permission.subject, revision="v2"))}):
            with self.assertRaises(ContractError):
                replace(voice, **fields)
        next_voice = replace(voice, revision=2, permission=replace(
            voice.permission, subject=replace(voice.permission.subject, revision="v2")))
        self.assertEqual(next_voice.revision, 2)

    def test_legacy_fractional_frame_seconds_bridge_without_relaxing_tick_bounds(self) -> None:
        timebase = AudioTimebase(48000, 98000)
        self.assertEqual(timebase.ticks_for_seconds(49 / 24), 98000)
        self.assertEqual(timebase.ticks_for_seconds(1 / 24), 2000)
        self.assertEqual(timebase.ticks_for_seconds(0.1), 4800)
        for value in (True, False, 0, -1, float("nan"), float("inf"), 1e308, 1 / 7):
            with self.assertRaises(ContractError):
                timebase.ticks_for_seconds(value)
        sound = sheet()
        # A non-dialogue sheet makes the scene/source duration bridge independently visible.
        sound = replace(sound, timebase=timebase, script=replace(sound.script, utterances=()),
                        shots=(ShotTiming("shot", 1, 0, 98000),),
                        stems=(Stem("ambience", "ambience"),),
                        cues=(replace(sound.cues[0], duration_ticks=98000),))
        plan, registry, ledger = context(sound)
        plan = replace(plan, shots=(replace(plan.shots[0], duration_seconds=49 / 24),))
        self.assertTrue(sound.preflight(plan, registry, ledger,
                                        current_picture=sound.picture).allowed)
        shortened = replace(sound, shots=(ShotTiming("shot", 1, 0, 97999),),
                            cues=(replace(sound.cues[0], duration_ticks=97999),))
        self.assertIn("stale_shot_timing", shortened.preflight(
            plan, registry, ledger, current_picture=sound.picture).reasons)
        with self.assertRaises(ContractError):
            replace(sound, cues=(replace(sound.cues[0], duration_ticks=98001),))
        with self.assertRaises(ContractError):
            replace(sound, shots=(ShotTiming("shot", 1, 0, 98001),))
        plan = replace(plan, shots=(replace(plan.shots[0], duration_seconds=1 / 7),))
        self.assertIn("unaligned_shot_duration", sound.preflight(
            plan, registry, ledger, current_picture=sound.picture).reasons)

    def test_every_input_revision_digest_invalidates_all_bound_result_kinds(self) -> None:
        sound = sheet()
        speech = (RevisionPin("speech-en", 1, DIGEST), RevisionPin("speech-zh", 1, OTHER))
        inputs = sound.dependencies(speech=speech)
        self.assertEqual(MediaDependencies.from_json(inputs.to_json()), inputs)
        changed = (
            replace(inputs, script=replace(inputs.script, revision=2)),
            replace(inputs, script=replace(inputs.script, sha256=OTHER)),
            replace(inputs, speech=(replace(speech[0], revision=2), speech[1])),
            replace(inputs, speech=(replace(speech[0], sha256=OTHER), speech[1])),
            replace(inputs, characters=(replace(inputs.characters[0], revision=2),)),
            replace(inputs, characters=(replace(inputs.characters[0], sha256=OTHER),)),
            replace(inputs, picture=replace(inputs.picture, revision=2)),
            replace(inputs, picture=replace(inputs.picture, sha256=OTHER)),
            replace(inputs, cue_sheet=replace(inputs.cue_sheet, revision=2)),
            replace(inputs, cue_sheet=replace(inputs.cue_sheet, sha256=OTHER)),
        )
        for kind in ("audio", "video", "sync_check", "listening_review", "visual_review"):
            binding = DependencyBinding(kind, RevisionPin("result", 1, DIGEST), inputs)
            self.assertTrue(binding.is_current(inputs))
            self.assertEqual(binding.stale_reasons(inputs), ())
            for current in changed:
                self.assertFalse(binding.is_current(current))
                self.assertEqual(len(binding.stale_reasons(current)), 1)
        with self.assertRaises(ContractError):
            replace(inputs, speech=speech * 2)
        with self.assertRaises(ContractError):
            replace(inputs, speech=tuple(reversed(speech)))
        with self.assertRaises(ContractError):
            replace(inputs, characters=inputs.characters * 2)

    def test_changed_words_delivery_language_timing_and_sources_change_dependencies(self) -> None:
        sound = sheet()
        baseline = sound.dependencies(speech=())
        for fields in ({"text": "Hello, Cineforge! "}, {"text_revision": 2}, {"language": "zh"},
                       {"emotion": "sad"}, {"delivery": "quietly"}, {"pronunciation": ()}):
            script = replace(sound.script, utterances=(replace(sound.script.utterances[0], **fields),
                                                       sound.script.utterances[1]))
            changed = replace(sound, script=script).dependencies(speech=())
            self.assertNotEqual(changed.script, baseline.script)
            self.assertNotEqual(changed.cue_sheet, baseline.cue_sheet)
        for changed in (replace(sound, script=replace(sound.script, revision=2)),
                        replace(sound, cues=(replace(sound.cues[0], duration_ticks=4999),
                                             *sound.cues[1:])),
                        replace(sound, cues=(replace(sound.cues[0], source=permission("new-room")),
                                             *sound.cues[1:]))):
            self.assertNotEqual(changed.dependencies(speech=()), baseline)

    def test_every_schema_rejects_unknown_versions_fields_and_missing_versions(self) -> None:
        sound = sheet()
        records = (sound, sound.timebase, sound.script, sound.script.utterances[0],
                   sound.script.utterances[0].voice, sound.script.utterances[0].pronunciation[0],
                   sound.stems[0], sound.shots[0], sound.cues[0], sound.picture,
                   sound.dependencies(speech=()))
        for record in records:
            data = record.to_dict()
            for value in (dict(data, schema_version=2), dict(data, schema_version=True),
                          dict(data, schema_version="private-canary"), dict(data, unknown=True),
                          {key: value for key, value in data.items() if key != "schema_version"}):
                with self.assertRaises(AudioError):
                    type(record).from_dict(value)
        for raw in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}',
                    '{"x":Infinity}', 'not JSON', '[]'):
            with self.assertRaises(AudioError):
                CueSheet.from_json(raw)

    def test_missing_ids_revisions_and_wrong_nested_types_fail_explicitly(self) -> None:
        sound = sheet()
        line = sound.script.utterances[0]
        records = ((sound, ("sheet_id", "project_id", "scene_id"), "revision"),
                   (sound.script, ("script_id",), "revision"),
                   (sound.picture, ("input_id",), "revision"),
                   (line.voice, ("voice_id",), "revision"),
                   (sound.shots[0], ("shot_id",), "shot_revision"))
        for record, names, revision in records:
            for name in names:
                for value in ("", None, True, "invalid ID"):
                    with self.assertRaises(ContractError):
                        replace(record, **{name: value})
            for value in (True, 0, 1.5, 2**63):
                with self.assertRaises(ContractError):
                    replace(record, **{revision: value})
        for fields in ({"script": None}, {"timebase": None}, {"picture": None},
                       {"stems": list(sound.stems)}, {"shots": list(sound.shots)},
                       {"cues": list(sound.cues)}):
            with self.assertRaises(ContractError):
                replace(sound, **fields)
        for name in ("schema_version", "script", "timebase", "picture", "cues"):
            with self.assertRaises(AudioError):
                CueSheet.from_dict({key: value for key, value in sound.to_dict().items()
                                    if key != name})

    def test_private_utterance_diagnostics_and_reprs_do_not_leak_canaries(self) -> None:
        marker = "private-utterance-canary"
        sound = sheet()
        data = sound.to_dict()
        invalid = (dict(data, schema_version=marker), dict(data, **{marker: marker}),
                   dict(data, script={marker: marker}), dict(data, timebase=marker),
                   dict(data, script=dict(data["script"], utterances=[dict(
                       data["script"]["utterances"][0], text=marker + "\ud800")])),
                   dict(data, script=dict(data["script"], utterances=[dict(
                       data["script"]["utterances"][0], language=marker)])))
        for value in invalid:
            with self.assertRaises(AudioError) as caught:
                CueSheet.from_json(json.dumps(value))
            self.assertNotIn(marker, str(caught.exception))
        for raw in ('{"' + marker + '":1,"' + marker + '":2}', marker):
            with self.assertRaises(AudioError) as caught:
                CueSheet.from_json(raw)
            self.assertNotIn(marker, str(caught.exception))
        hint = PronunciationHint(marker, marker)
        line = replace(sound.script.utterances[0], text=marker, emotion=marker,
                       delivery=marker, pronunciation=(hint,))
        script = replace(sound.script, utterances=(line, sound.script.utterances[1]))
        private = replace(sound, script=script)
        for record in (hint, line, script, private):
            self.assertNotIn(marker, repr(record))
        for fields in ({"text": marker + "\ud800"}, {"emotion": marker + "\ud800"}):
            try:
                replace(line, **fields)
            except AudioError:
                self.assertNotIn(marker, traceback.format_exc())
            else:
                self.fail("invalid private Unicode was accepted")
        plan, registry, _ = context(sound)
        result = private.preflight(plan, registry, RightsLedger(), current_picture=private.picture)
        self.assertNotIn(marker, repr(result))


if __name__ == "__main__":
    unittest.main()
