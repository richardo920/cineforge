"""Offline external render receipt, fitting and scope contracts; synthetic PCM only."""

import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_speech import adapter, fixture, wav_bytes

from cineforge.audio import AudioTimebase
from cineforge.audio_pcm import PCM16Wave, inspect_pcm16_wav
from cineforge.external_speech import (
    ExternalSpeechError,
    ExternalSpeechRender,
    ExternalSpeechRuntime,
)
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.rights import EvidenceRef
from cineforge.s2v import NativeSpeechProof
from cineforge.s2v_service import LongSessionBudgetApproval, LongSessionServiceContext
from cineforge.speech import SpeechArtifact
from cineforge.speech_fit import SpeechFitError
from cineforge.workflows import canonical


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class ExternalSpeechTests(unittest.TestCase):
    def setUp(self):
        args, _, _ = fixture()
        self.request = adapter(args).translate(utterance_id="en-line", **args)
        self.reference = inspect_pcm16_wav(args["reference_payload"])
        self.design = OriginalVoiceDesign("Synthetic line", "English", "Original timbre", "a" * 64)
        approval = LongSessionBudgetApproval.for_scope("/synthetic/external-speech-budget", "EUR-IS-2",
            EvidenceRef("private://synthetic/review", "a" * 64), phase_id="test")
        self.service = LongSessionServiceContext(approval)
        self.raw = wav_bytes()
        self.design_render = wav_bytes(frames=4800)

    def runtime(self, **changes):
        values = dict(provider="Muse", voice_id="avocado_v2:MAI_01", model_id="synthetic-model",
            worker_id="external-worker", scope_sha256=self.service.approval.subject_sha256,
            renders=(ExternalSpeechRender(self.request.text, self.raw),),
            design_render=self.design_render)
        values.update(changes)
        return ExternalSpeechRuntime(**values)

    def artifact(self, payload, request=None):
        request = self.request if request is None else request
        wave = inspect_pcm16_wav(payload)
        return SpeechArtifact(replace(request.reference, input_id="speech", sha256=sha(payload)),
            request, payload, "unverified_injected_runtime", wave.sample_rate, wave.frames)

    def test_construction_rejects_invalid_inputs(self):
        for changes in (
            {"worker_id": "bad worker"}, {"scope_sha256": "A" * 64},
            {"renders": (ExternalSpeechRender(self.request.text, self.raw),) * 2},
            {"renders": (ExternalSpeechRender(self.request.text, wav_bytes(channels=2)),)},
            {"renders": ()}, {"renders": []}, {"provider": ""}, {"voice_id": "x" * 129},
            {"model_id": None}, {"design_render": b"invalid"},
            {"renders": (ExternalSpeechRender(" ", self.raw),)},
            {"renders": (ExternalSpeechRender(self.request.text, b"invalid"),)},
        ):
            with self.subTest(fields=tuple(changes)), self.assertRaises(ExternalSpeechError):
                self.runtime(**changes)

    def test_under_slot_receipt_and_proof(self):
        runtime = self.runtime()
        payload = runtime.synthesize(self.request, self.reference)
        self.assertIs(payload, self.raw)
        self.assertEqual(runtime.synthesize(self.request, self.reference), payload)
        proof = runtime.verify_speech(self.artifact(payload), self.service)
        receipt = {"source": "external", "provider": "Muse", "voice_id": "avocado_v2:MAI_01",
            "model_id": "synthetic-model", "worker_id": "external-worker",
            "scope_sha256": self.service.approval.subject_sha256,
            "request_sha256": self.request.digest(),
            "reference_sha256": self.request.reference.sha256, "raw_sha256": sha(self.raw),
            "raw_frames": 2400, "raw_sample_rate": 24000, "fitted_sha256": sha(payload),
            "fitted_frames": 2400}
        self.assertEqual(proof, NativeSpeechProof(self.request.digest(), sha(payload),
            self.request.reference.sha256, sha(canonical(receipt)), "external-worker",
            self.service.approval.subject_sha256))

    def test_segment_two_fit_binds_fitted_payload(self):
        request = replace(self.request, timebase=AudioTimebase(10000, 100000),
                          start_ticks=0, duration_ticks=48125)
        raw = wav_bytes(frames=130560)
        runtime = self.runtime(renders=(ExternalSpeechRender(request.text, raw),))
        payload = runtime.synthesize(request, self.reference)
        self.assertEqual(inspect_pcm16_wav(payload).frames, 115500)
        proof = runtime.verify_speech(self.artifact(payload, request), self.service)
        self.assertEqual(proof.speech_sha256, sha(payload))
        self.assertNotEqual(proof.speech_sha256, sha(raw))
        with self.assertRaises(ExternalSpeechError):
            runtime.verify_speech(self.artifact(raw, request), self.service)

    def test_exact_slot_preserves_payload(self):
        raw = wav_bytes(frames=7200)
        runtime = self.runtime(renders=(ExternalSpeechRender(self.request.text, raw),))
        payload = runtime.synthesize(self.request, self.reference)
        self.assertIs(payload, raw)
        self.assertEqual(payload, raw)

    def test_exact_headroom_fits_to_slot(self):
        raw = wav_bytes(frames=14400)
        runtime = self.runtime(renders=(ExternalSpeechRender(self.request.text, raw),))
        payload = runtime.synthesize(self.request, self.reference)
        self.assertEqual(inspect_pcm16_wav(payload).frames, 7200)
        runtime.verify_speech(self.artifact(payload), self.service)

    def test_fitted_payload_cannot_be_verified_for_another_request(self):
        runtime = self.runtime(renders=(ExternalSpeechRender(self.request.text,
                                                            wav_bytes(frames=10000)),))
        payload = runtime.synthesize(self.request, self.reference)
        other = replace(self.request, duration_ticks=self.request.duration_ticks + 1)
        self.assertEqual(other.text, self.request.text)
        with self.assertRaises(ExternalSpeechError):
            runtime.verify_speech(self.artifact(payload, other), self.service)

    def test_over_headroom_rejected(self):
        runtime = self.runtime(renders=(ExternalSpeechRender(self.request.text,
                                                            wav_bytes(frames=14401)),))
        with self.assertRaises(ExternalSpeechError):
            runtime.synthesize(self.request, self.reference)

    def test_request_and_reference_rejected(self):
        runtime = self.runtime()
        for request, reference in (
            (replace(self.request, text="Unregistered line"), self.reference),
            (replace(self.request, language="Chinese"), self.reference),
            (self.request, inspect_pcm16_wav(wav_bytes(frames=2401))),
            (self.request, PCM16Wave(b"\x01\x00" * 2400, 24000, 1)),
            (self.request, replace(self.reference, channels=2)),
            (None, self.reference), (self.request, None),
        ):
            with self.subTest(request_type=type(request)), self.assertRaises(ExternalSpeechError):
                runtime.synthesize(request, reference)

    def test_other_scope_and_unregistered_payload_rejected(self):
        runtime = self.runtime()
        with self.assertRaises(ExternalSpeechError):
            runtime.verify_speech(self.artifact(self.raw), self.service)
        runtime.synthesize(self.request, self.reference)
        other = LongSessionServiceContext(LongSessionBudgetApproval.for_scope(
            "/synthetic/other-budget", "EUR-IS-2", self.service.approval.operator_receipt,
            phase_id="other"))
        with self.assertRaises(ExternalSpeechError):
            runtime.verify_speech(self.artifact(self.raw), other)
        with self.assertRaises(ExternalSpeechError):
            runtime.verify_speech(None, self.service)
        with self.assertRaises(ExternalSpeechError):
            runtime.verify_speech(self.artifact(self.raw), None)

    def test_preview_and_configuration(self):
        runtime = self.runtime()
        self.assertIs(runtime.preview(self.design), self.design_render)
        with self.assertRaises(ExternalSpeechError):
            runtime.preview(None)
        config = runtime.configuration()
        self.assertEqual(config, runtime.configuration())
        changed = self.runtime(renders=(ExternalSpeechRender(self.request.text,
                                                            wav_bytes(frames=2401)),))
        self.assertNotEqual(config, changed.configuration())
        renders = (ExternalSpeechRender("Second line", self.raw),
                   ExternalSpeechRender(self.request.text, self.raw))
        forward, reverse = self.runtime(renders=renders), self.runtime(renders=renders[::-1])
        self.assertEqual(forward.configuration(), reverse.configuration())
        hashes = [item["text_sha256"] for item in forward.configuration()["renders"]]
        self.assertEqual(hashes, sorted(hashes))
        self.assertNotIn(self.request.text, str(config))

    def test_fit_error_has_static_message_without_cause(self):
        with patch("cineforge.external_speech.fit_pcm16_wav_to_slot",
                   side_effect=SpeechFitError("private detail")):
            with self.assertRaises(ExternalSpeechError) as caught:
                self.runtime().synthesize(self.request, self.reference)
        self.assertNotIn("private detail", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
