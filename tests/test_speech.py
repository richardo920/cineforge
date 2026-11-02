"""Synthetic declarations/PCM only; no SDK, real speech, reference or review."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import traceback
import unittest
import wave
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from cineforge.audio import (
    AudioTimebase,
    Cue,
    CueSheet,
    DialogueScript,
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
from cineforge.speech import (
    BASE_MODEL,
    BASE_REVISION,
    SDK_REVISION,
    CanonicalVoiceBinding,
    FakeSpeechRuntime,
    QwenBaseRequest,
    QwenBaseSpeechAdapter,
    SpeechArtifact,
    SpeechError,
    SyntheticReferenceInput,
)

DIGEST = "a" * 64
EVIDENCE = EvidenceRef("private://synthetic/review", DIGEST)


def wav_bytes(*, frames=2400, rate=24000, channels=1, sample_width=2):
    result = io.BytesIO()
    with wave.open(result, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(rate)
        output.writeframes(b"\x00" * frames * channels * sample_width)
    return result.getvalue()


def permission(name, kind="asset", *, digest=DIGEST, revision="v1"):
    return RightsBinding(SubjectRef(kind, name, revision, digest), RightsRef(name, "v1"))


def reviewed(binding, **changes):
    record = RightsRecord(binding.rights, binding.subject, source=EVIDENCE,
        source_revision="v1", license_id="LicenseRef-Synthetic", terms=EVIDENCE,
        notices=("Synthetic fixture",), attribution="Synthetic fixture", consent=EVIDENCE,
        allowed_uses=tuple(Use(action) for action in ("generate", "reference", "condition", "copy")))
    return with_review(replace(record, **changes), reviewer="fixture-reviewer", evidence=EVIDENCE)


def fixture(reference_payload=None):
    payload = wav_bytes() if reference_payload is None else reference_payload
    digest = hashlib.sha256(payload).hexdigest()
    recording = permission("recording", "voice", digest=digest, revision="r1")
    face = permission("face", "character_reference")
    actor = with_character_review(CharacterRevision("actor", 1, "fictional", "Original actor",
        Wardrobe("coat", "Blue coat"), (
            ReferenceArtifact("face", "identity", EVIDENCE, 16, "image/png", face),
            ReferenceArtifact("reference", "voice",
                EvidenceRef("private://synthetic/recording", digest), len(payload), "audio/wav",
                recording)), voice_reference_id="reference"),
        reviewer="fixture-reviewer", evidence=EVIDENCE)
    voice = VoiceIdentity("canonical-voice", 1, permission("canonical-voice", "voice"), "reference")
    english = Utterance("en-line", 1, "  Hello, Cineforge!\n", "en", "speaker", actor.pin(),
        voice, "neutral", "default", (), "scene", "shot", 100, 300)
    chinese = replace(english, utterance_id="zh-line", text="\t你好，电影世界！  ", language="zh",
                      start_ticks=600)
    original = wav_bytes(frames=24000)
    source_hash = hashlib.sha256(original).hexdigest()
    cues = tuple(Cue(kind, kind, kind, "scene", None, 0, 1000,
        permission(kind, "music" if kind == "music" else "asset", digest=source_hash))
        for kind in ("ambience", "effects", "foley", "music")) + (
            Cue("en-cue", "dialogue", "dialogue", "scene", "shot", 100, 300,
                voice.permission, "en-line"),
            Cue("zh-cue", "dialogue", "dialogue", "scene", "shot", 600, 300,
                voice.permission, "zh-line"))
    sheet = CueSheet("sheet", 1, "project", "scene", AudioTimebase(1000, 1000),
        DialogueScript("script", 1, (english, chinese)), RevisionPin("picture", 1, DIGEST),
        tuple(Stem(kind, kind) for kind in ("ambience", "effects", "foley", "music", "dialogue")),
        (ShotTiming("shot", 1, 0, 1000),), cues)
    plan = ProjectPlan(Project("project", 1, "Synthetic"), (Scene("scene", "project", 1),),
        (pin_shot(Shot("shot", "scene", 1, "Original actor speaks", None, 1,
            "synthetic-model", "v1", DIGEST), actor.pin()),))
    registry = CharacterRegistry()
    registry.register(actor)
    ledger = RightsLedger()
    for binding in {*(cue.source for cue in cues), face, recording}:
        ledger.register(reviewed(binding))
    return dict(sheet=sheet, reference=SyntheticReferenceInput(RevisionPin("reference", 1, digest),
        recording, "  An exact synthetic reference.\n"), reference_payload=payload, plan=plan,
        registry=registry, ledger=ledger, current_picture=sheet.picture,
        current_script=sheet.script.pin()), actor, original


def adapter(arguments, *, runtime=None):
    item = arguments["sheet"].script.utterances[0]
    return QwenBaseSpeechAdapter((CanonicalVoiceBinding(item.character, item.voice),), runtime=runtime)


def change_utterance(arguments, **changes):
    sheet = arguments["sheet"]
    utterance = replace(sheet.script.utterances[0], **changes)
    script = replace(sheet.script, utterances=(utterance, sheet.script.utterances[1]))
    cues = tuple(replace(cue, source=utterance.voice.permission) if cue.utterance_id == "en-line"
                 else cue for cue in sheet.cues)
    return dict(arguments, sheet=replace(sheet, script=script, cues=cues), current_script=script.pin())


class Runtime:
    def __init__(self, payload=None, action=None):
        self.payload = wav_bytes() if payload is None else payload
        self.action = action
        self.calls = []

    def synthesize(self, request, reference):
        self.calls.append((request, reference))
        if self.action:
            self.action()
        return self.payload


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.arguments, self.actor, self.original = fixture()
        self.runtime = Runtime()
        self.adapter = adapter(self.arguments, runtime=self.runtime)

    def synthesize(self, **changes):
        return self.adapter.synthesize(utterance_id="en-line", output_id="speech",
            output_revision=1, **dict(self.arguments, **changes))

    def test_exact_bilingual_base_requests_round_trip_and_digests(self):
        for identity, language in (("en-line", "English"), ("zh-line", "Chinese")):
            request = self.adapter.translate(utterance_id=identity, **self.arguments)
            item = next(x for x in self.arguments["sheet"].script.utterances
                        if x.utterance_id == identity)
            self.assertEqual(request.text, item.text)
            self.assertEqual(request.language, language)
            self.assertEqual(request.ref_text, self.arguments["reference"].transcript)
            self.assertEqual(request.script, self.arguments["current_script"])
            self.assertEqual(request.utterance, RevisionPin(identity, 1, item.digest()))
            self.assertEqual(request.character, self.actor.pin())
            self.assertEqual(request.voice, item.voice)
            self.assertEqual(request.reference, self.arguments["reference"].source)
            self.assertEqual(request.recording_permission, self.actor.references[1].permission)
            self.assertEqual((request.sdk_revision, request.model_id, request.model_revision),
                             (SDK_REVISION, BASE_MODEL, BASE_REVISION))
            self.assertEqual((request.x_vector_only_mode, request.non_streaming_mode,
                              request.max_new_tokens), (False, True, 512))
            self.assertEqual(QwenBaseRequest.from_json(request.to_json()), request)
            self.assertEqual(request.digest(), hashlib.sha256(request.to_json().encode()).hexdigest())
            self.assertNotIn(item.text, repr(request))
            self.assertNotIn(request.ref_text, repr(request))
            self.assertNotEqual(replace(request, text=request.text + " ").digest(), request.digest())
            self.assertNotEqual(replace(request, script=replace(request.script, revision=2)).digest(),
                                request.digest())
            self.assertNotEqual(replace(request, reference=replace(request.reference, revision=2)).digest(),
                                request.digest())
        self.assertEqual(self.runtime.calls, [])

    def test_fixture_result_is_explicit_not_speech_or_approval(self):
        result = adapter(self.arguments).synthesize(utterance_id="en-line", output_id="speech",
            output_revision=1, **self.arguments)
        self.assertEqual(result.runtime_kind, "synthetic_fixture")
        self.assertEqual(result.pin.sha256, hashlib.sha256(result.payload).hexdigest())
        self.assertEqual((result.sample_rate, result.frames), (24000, 480))
        self.assertEqual(result.payload[44:], b"\x00" * 960)
        self.assertNotIn("payload=", repr(result))
        self.assertNotIn("request=", repr(result))
        self.assertFalse(hasattr(result, "approved"))
        self.assertFalse(hasattr(result, "export"))
        again = adapter(self.arguments).synthesize(utterance_id="en-line", output_id="speech",
            output_revision=1, **self.arguments)
        self.assertEqual(result, again)

    def test_only_explicit_trusted_runtime_is_called_and_receives_validated_pcm(self):
        result = self.synthesize()
        self.assertEqual(len(self.runtime.calls), 1)
        request, pcm = self.runtime.calls[0]
        self.assertEqual((pcm.sample_rate, pcm.channels, pcm.frames), (24000, 1, 2400))
        self.assertEqual(pcm.pcm, self.arguments["reference_payload"][44:])
        self.assertEqual(request, result.request)
        self.assertEqual(result.runtime_kind, "unverified_injected_runtime")

    def test_unsupported_delivery_pronunciation_and_missing_line_refuse_before_runtime(self):
        for changes in ({"emotion": "happy"}, {"delivery": "whisper"},
                        {"pronunciation": (PronunciationHint("Cineforge", "hint"),)}):
            args = change_utterance(self.arguments, **changes)
            with self.assertRaises(SpeechError):
                self.adapter.synthesize(utterance_id="en-line", output_id="speech",
                    output_revision=1, **args)
        with self.assertRaises(SpeechError):
            self.adapter.translate(utterance_id="missing", **self.arguments)
        # M5.1 itself refuses unsupported language before an adapter can be called.
        with self.assertRaises(ContractError):
            change_utterance(self.arguments, language="ja")
        self.assertEqual(self.runtime.calls, [])

    def test_canonical_voice_cannot_switch_for_character_across_scripts(self):
        voice = replace(self.arguments["sheet"].script.utterances[0].voice,
                        voice_id="alternate", permission=permission("alternate", "voice"))
        sheet = self.arguments["sheet"]
        script = replace(sheet.script, utterances=tuple(replace(item, voice=voice)
            for item in sheet.script.utterances))
        sheet = replace(sheet, script=script, cues=tuple(replace(cue, source=voice.permission)
            if cue.kind == "dialogue" else cue for cue in sheet.cues))
        with self.assertRaises(SpeechError):
            self.synthesize(sheet=sheet, current_script=script.pin())
        with self.assertRaises(SpeechError):
            QwenBaseSpeechAdapter(())
        binding = CanonicalVoiceBinding(self.actor.pin(), voice)
        with self.assertRaises(SpeechError):
            QwenBaseSpeechAdapter((binding, binding))
        with self.assertRaises(SpeechError):
            CanonicalVoiceBinding(self.actor.pin(), replace(voice, reference_id=None))
        self.assertEqual(self.runtime.calls, [])

    def test_stale_script_picture_shot_and_character_refuse_before_runtime(self):
        for name in ("current_script", "current_picture"):
            with self.assertRaises(SpeechError):
                self.synthesize(**{name: replace(self.arguments[name], revision=2)})
        plan = self.arguments["plan"]
        with self.assertRaises(SpeechError):
            self.synthesize(plan=replace(plan, shots=(replace(plan.shots[0], revision=2),)))
        with self.assertRaises(SpeechError):
            self.synthesize(registry=CharacterRegistry())
        self.assertEqual(self.runtime.calls, [])

    def test_missing_changed_stereo_or_malformed_reference_refuses_before_runtime(self):
        for payload in (b"PRIVATE-reference-path", self.arguments["reference_payload"][:-1],
                        wav_bytes(channels=2)):
            with self.assertRaises(SpeechError):
                self.synthesize(reference_payload=payload)
        with self.assertRaises(SpeechError):
            self.synthesize(reference=replace(self.arguments["reference"],
                source=replace(self.arguments["reference"].source, input_id="other")))
        self.assertEqual(self.runtime.calls, [])

    def test_reference_pcm_parser_runs_on_exact_current_hash_and_size(self):
        for payload in (wav_bytes(channels=2), wav_bytes(sample_width=1),
                        b"PRIVATE-REFERENCE-CANARY", wav_bytes(rate=8000)):
            arguments, _, _ = fixture(reference_payload=payload)
            runtime = Runtime()
            instance = adapter(arguments, runtime=runtime)
            with self.assertRaises(SpeechError) as caught:
                instance.synthesize(utterance_id="en-line", output_id="speech",
                                    output_revision=1, **arguments)
            self.assertEqual(runtime.calls, [])
            self.assertNotIn("CANARY", str(caught.exception))

    def test_rights_missing_revoked_superseded_unknown_terms_consent_and_stale_review(self):
        original = self.arguments["ledger"]
        binding = self.arguments["reference"].permission
        record = original.lookup(binding.rights)
        for invalid in (None, replace(record, review=None), replace(record, license_id="unknown"),
                        replace(record, consent=None), replace(record, terms=None),
                        replace(record, attribution="changed after review")):
            ledger = RightsLedger()
            for current in {*(cue.source for cue in self.arguments["sheet"].cues),
                            *(ref.permission for ref in self.actor.references)}:
                if current == binding:
                    if invalid is not None:
                        ledger.register(invalid)
                else:
                    ledger.register(original.lookup(current.rights))
            with self.assertRaises(SpeechError):
                self.synthesize(ledger=ledger)
        for action in ("revoke", "supersede"):
            args, _, _ = fixture()
            if action == "revoke":
                args["ledger"].revoke(binding.rights, reason="synthetic revocation")
            else:
                replacement = replace(record, ref=RightsRef("replacement", "v2"))
                args["ledger"].register(with_review(replacement, reviewer="reviewer", evidence=EVIDENCE))
            with self.assertRaises(SpeechError):
                self.adapter.synthesize(utterance_id="en-line", output_id="speech",
                    output_revision=1, **args)
        self.assertEqual(self.runtime.calls, [])

    def test_logical_voice_and_recording_uses_checked_independently(self):
        for binding in (self.arguments["reference"].permission,
                        self.arguments["sheet"].script.utterances[0].voice.permission):
            args, _, _ = fixture()
            args["ledger"].revoke(binding.rights, reason="synthetic")
            with self.assertRaises(SpeechError):
                self.adapter.synthesize(utterance_id="en-line", output_id="speech",
                    output_revision=1, **args)
        self.assertEqual(self.runtime.calls, [])

    def test_revocation_during_runtime_and_after_translate_prevents_artifact(self):
        binding = self.arguments["reference"].permission
        self.runtime.action = lambda: self.arguments["ledger"].revoke(binding.rights, reason="during")
        with self.assertRaises(SpeechError):
            self.synthesize()
        self.assertEqual(len(self.runtime.calls), 1)
        self.runtime.calls.clear()
        with self.assertRaises(SpeechError):
            self.synthesize()
        self.assertEqual(self.runtime.calls, [])

    def test_runtime_cancellation_and_system_exit_do_not_expose_private_arguments(self):
        for exception in (KeyboardInterrupt, SystemExit):
            def fail():
                raise exception("PRIVATE-RUNTIME-CANARY")
            self.runtime.action = fail
            try:
                self.synthesize()
            except (KeyboardInterrupt, SpeechError) as error:
                self.assertIsNone(error.__context__)
                self.assertNotIn("CANARY", str(error))
                self.assertNotIn("CANARY", traceback.format_exc())
            else:
                self.fail("runtime cancellation/failure was lost")

    def test_runtime_output_and_errors_are_bounded_sanitized_and_never_rewritten(self):
        for payload in (b"PRIVATE-AUDIO-CANARY", wav_bytes(sample_width=1), wav_bytes(channels=2),
                        wav_bytes(frames=24000), b"x" * (16 * 1024 * 1024 + 1), None):
            # Assign directly so None is an invalid runtime response, not fixture default.
            self.runtime.payload = payload
            with self.assertRaises(SpeechError) as caught:
                self.synthesize()
            self.assertNotIn("CANARY", str(caught.exception))
        def fail():
            raise RuntimeError("PRIVATE-TEXT-PATH-TRANSCRIPT-CANARY")
        self.runtime.action = fail
        try:
            self.synthesize()
        except SpeechError as error:
            self.assertIsNone(error.__context__)
            self.assertNotIn("CANARY", traceback.format_exc())
        else:
            self.fail("private runtime exception escaped boundary")

    def test_private_text_unicode_errors_have_no_private_exception_context(self):
        invalid = "PRIVATE-CANARY\ud800"
        try:
            replace(self.arguments["reference"], transcript=invalid)
        except SpeechError as error:
            self.assertIsNone(error.__context__)
            self.assertNotIn("CANARY", traceback.format_exc())
        else:
            self.fail("invalid transcript accepted")

    def test_no_implicit_io_sdk_import_or_executable_metadata_selector(self):
        with patch("builtins.open", side_effect=AssertionError("file I/O")), \
             patch("os.open", side_effect=AssertionError("file I/O")), \
             patch("socket.socket", side_effect=AssertionError("network I/O")), \
             patch("subprocess.Popen", side_effect=AssertionError("process I/O")):
            instance = adapter(self.arguments)
            request = instance.translate(utterance_id="en-line", **self.arguments)
            instance.synthesize(utterance_id="en-line", output_id="speech", output_revision=1,
                                **self.arguments)
        self.assertNotIn("qwen_tts", sys.modules)
        self.assertNotIn("torch", sys.modules)
        for name, value in (("runtime", "qwen_tts"), ("model_path", "/private/path"),
                            ("sdk_revision", "main"), ("max_new_tokens", True)):
            data = request.to_dict()
            data[name] = value
            with self.assertRaises(ContractError):
                QwenBaseRequest.from_dict(data)

    def test_ids_validated_before_runtime_and_data_records_frozen(self):
        with self.assertRaises(ContractError):
            self.adapter.synthesize(utterance_id="en-line", output_id="private/invalid",
                output_revision=1, **self.arguments)
        self.assertEqual(self.runtime.calls, [])
        request = self.adapter.translate(utterance_id="en-line", **self.arguments)
        with self.assertRaises(FrozenInstanceError):
            request.text = "changed"
        with self.assertRaises(SpeechError):
            SpeechArtifact(RevisionPin("bad", 1, DIGEST), request, wav_bytes(),
                           "synthetic_fixture", 24000, 2400)

    def test_import_in_fresh_process_is_pure_and_default_runtime_fixture_only(self):
        command = "import sys,cineforge.speech;assert 'qwen_tts' not in sys.modules;" \
                  "assert 'torch' not in sys.modules;print('pure')"
        result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True,
                                timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "pure\n")
        runtime = FakeSpeechRuntime()
        instance = adapter(self.arguments, runtime=runtime)
        request = instance.translate(utterance_id="en-line", **self.arguments)
        self.assertEqual(json.loads(request.to_json())["text"], "  Hello, Cineforge!\n")
        self.assertEqual(runtime.calls, [])


if __name__ == "__main__":
    unittest.main()
