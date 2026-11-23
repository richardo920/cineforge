"""Trusted controller-side source for lines rendered by an external TTS voice.

Lines are fitted to their slots at receipt under the native delivery policy.
Proofs bind a canonical external-render receipt and one approval scope: the
NativeSpeechProof receipt slot holds that digest, not a pod receipt. No I/O.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from cineforge.audio_pcm import PCM16Wave, PCMError, inspect_pcm16_wav
from cineforge.json_codec import is_sha256_hex
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.native_speech_integrity import NativeSpeechError
from cineforge.native_speech_pcm import pcm_wav
from cineforge.s2v import NativeSpeechProof
from cineforge.s2v_service import LongSessionServiceContext, S2VServiceContext, is_speech_service
from cineforge.speech import RESERVED_DURATION_HEADROOM_FACTOR, QwenBaseRequest, SpeechArtifact
from cineforge.speech_fit import SpeechFitError, fit_pcm16_wav_to_slot
from cineforge.workflows import canonical


class ExternalSpeechError(NativeSpeechError):
    """Static diagnostics without private text or payloads."""


@dataclass(frozen=True, slots=True)
class ExternalSpeechRender:
    text: str
    payload: bytes = field(repr=False)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _wave(payload: bytes) -> PCM16Wave:
    try:
        wave = inspect_pcm16_wav(payload)
    except PCMError:
        raise ExternalSpeechError("invalid external speech PCM WAV") from None
    if wave.channels != 1:
        raise ExternalSpeechError("external speech requires mono PCM WAV")
    return wave


class ExternalSpeechRuntime:
    def __init__(self, *, provider: str, voice_id: str, model_id: str, worker_id: str,
                 scope_sha256: str, renders: tuple[ExternalSpeechRender, ...],
                 design_render: bytes) -> None:
        if any(type(value) is not str or not 1 <= len(value) <= 128
               for value in (provider, voice_id, model_id)):
            raise ExternalSpeechError("invalid external speech provenance")
        if (type(worker_id) is not str
                or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", worker_id) is None):
            raise ExternalSpeechError("invalid external speech worker identity")
        if not is_sha256_hex(scope_sha256):
            raise ExternalSpeechError("invalid external speech scope identity")
        if type(renders) is not tuple or not renders:
            raise ExternalSpeechError("external speech renders required")
        self._renders: dict[str, tuple[bytes, str, int, int]] = {}
        for render in renders:
            if (type(render) is not ExternalSpeechRender or type(render.text) is not str
                    or not render.text.strip() or "\x00" in render.text or len(render.text) > 4096
                    or any(0xD800 <= ord(c) <= 0xDFFF for c in render.text)
                    or render.text in self._renders):
                raise ExternalSpeechError("invalid or duplicate external speech text")
            wave = _wave(render.payload)
            self._renders[render.text] = (render.payload, _sha(render.payload),
                                          wave.frames, wave.sample_rate)
        _wave(design_render)
        self._provider = provider
        self._voice_id = voice_id
        self._model_id = model_id
        self._worker_id = worker_id
        self._scope_sha256 = scope_sha256
        self._design_render = design_render
        self._facts: dict[tuple[str, str], tuple[str, str]] = {}

    def synthesize(self, request: QwenBaseRequest, reference: PCM16Wave) -> bytes:
        if type(request) is not QwenBaseRequest or type(reference) is not PCM16Wave:
            raise ExternalSpeechError("typed external speech request and reference required")
        try:
            raw_reference = pcm_wav(reference.pcm, reference.sample_rate)
        except NativeSpeechError:
            raise ExternalSpeechError("invalid external speech reference framing") from None
        if (reference.channels != 1 or reference.frames != request.reference_frames
                or reference.sample_rate != request.reference_sample_rate
                or _sha(raw_reference) != request.reference.sha256):
            raise ExternalSpeechError("external speech reference differs from request")
        if request.language != "English":
            raise ExternalSpeechError("external speech requires English")
        render = self._renders.get(request.text)
        if render is None:
            raise ExternalSpeechError("external speech text is not registered")
        raw, raw_sha256, raw_frames, raw_sample_rate = render
        if (raw_frames * request.timebase.ticks_per_second
                > RESERVED_DURATION_HEADROOM_FACTOR * request.duration_ticks * raw_sample_rate):
            raise ExternalSpeechError("external speech exceeds reserved duration headroom")
        try:
            fitted = fit_pcm16_wav_to_slot(raw,
                ticks_per_second=request.timebase.ticks_per_second,
                duration_ticks=request.duration_ticks)
        except SpeechFitError:
            raise ExternalSpeechError("external speech could not fit reserved duration") from None
        request_sha256, fitted_sha256 = request.digest(), _sha(fitted)
        receipt = {"source": "external", "provider": self._provider, "voice_id": self._voice_id,
            "model_id": self._model_id, "worker_id": self._worker_id,
            "scope_sha256": self._scope_sha256, "request_sha256": request_sha256,
            "reference_sha256": request.reference.sha256, "raw_sha256": raw_sha256,
            "raw_frames": raw_frames, "raw_sample_rate": raw_sample_rate,
            "fitted_sha256": fitted_sha256, "fitted_frames": _wave(fitted).frames}
        key = (request_sha256, fitted_sha256)
        facts = (_sha(canonical(receipt)), request.reference.sha256)
        if key in self._facts and self._facts[key] != facts:
            raise ExternalSpeechError("external speech proof registration differs")
        self._facts[key] = facts
        return fitted

    def preview(self, design: OriginalVoiceDesign) -> bytes:
        if type(design) is not OriginalVoiceDesign:
            raise ExternalSpeechError("typed original external voice design required")
        return self._design_render

    def verify_speech(
        self, speech: SpeechArtifact, service: S2VServiceContext | LongSessionServiceContext
    ) -> NativeSpeechProof:
        if type(speech) is not SpeechArtifact or not is_speech_service(service):
            raise ExternalSpeechError("typed S2V speech and service required")
        service.preflight()
        if service.approval.subject_sha256 != self._scope_sha256:
            raise ExternalSpeechError("owned speech belongs to another S2V scope")
        request_sha256, speech_sha256 = speech.request.digest(), _sha(speech.payload)
        facts = self._facts.get((request_sha256, speech_sha256))
        if facts is None:
            raise ExternalSpeechError("external speech proof is not registered")
        receipt_sha256, reference_sha256 = facts
        return NativeSpeechProof(request_sha256, speech_sha256, reference_sha256,
                                 receipt_sha256, self._worker_id, self._scope_sha256)

    def configuration(self) -> dict[str, Any]:
        renders = [{"text_sha256": _sha(text.encode()), "raw_sha256": raw_sha256,
                    "raw_frames": frames, "sample_rate": sample_rate}
                   for text, (_, raw_sha256, frames, sample_rate) in self._renders.items()]
        renders.sort(key=lambda render: str(render["text_sha256"]))
        return {"provider": self._provider, "voice_id": self._voice_id,
                "model_id": self._model_id, "worker_id": self._worker_id,
                "scope_sha256": self._scope_sha256, "design_sha256": _sha(self._design_render),
                "renders": renders}
