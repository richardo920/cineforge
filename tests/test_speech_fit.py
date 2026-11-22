"""Deterministic slot fitting and downstream admission with synthetic PCM only."""
from __future__ import annotations

import hashlib
import math
import struct
import unittest
from dataclasses import replace

from test_audio_mix import IDENTITY, Meter, scene

from cineforge.audio import DialogueScript, RevisionPin
from cineforge.audio_mix import BROADCAST_LINEAR, MixError, MixSettings, StemMix, mix_scene
from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.native_speech_pcm import pcm_wav
from cineforge.paired_media import PairedMediaError, speech_metadata
from cineforge.speech import SpeechArtifact
from cineforge.speech_fit import SpeechFitError, fit_pcm16_wav_to_slot

RATE = 24000
SLOT = 115500


def sine(frames: int, rate: int = RATE) -> bytes:
    return pcm_wav(struct.pack(f"<{frames}h", *(
        round(12000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))), rate)


def samples(payload: bytes) -> tuple[int, ...]:
    wave = inspect_pcm16_wav(payload)
    return struct.unpack(f"<{wave.frames}h", wave.pcm)


def fit(payload: bytes) -> bytes:
    return fit_pcm16_wav_to_slot(payload, ticks_per_second=10000, duration_ticks=48125)


def frequency(payload: bytes) -> float:
    values = samples(payload)
    crossings = sum(a <= 0 < b for a, b in zip(values, values[1:], strict=False))
    return crossings * RATE / len(values)


class SpeechFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = sine(130560)
        cls.fitted = fit(cls.source)

    def test_exact_canonical_output_and_pitch(self):
        wave = inspect_pcm16_wav(self.fitted)
        self.assertEqual((wave.frames, wave.sample_rate, wave.channels), (SLOT, RATE, 1))
        self.assertEqual(len(self.fitted), 44 + SLOT * 2)
        fitted_hz = frequency(self.fitted)
        self.assertLess(abs(fitted_hz / frequency(self.source) - 1), .03)
        self.assertLess(abs(fitted_hz - 440), abs(fitted_hz - 497))

    def test_determinism(self):
        self.assertEqual(fit(self.source), self.fitted)

    def test_at_and_under_slot_are_identical_objects(self):
        for frames in (SLOT, 100000):
            with self.subTest(frames=frames):
                payload = sine(frames)
                self.assertIs(fit(payload), payload)

    def test_twice_slot(self):
        self.assertEqual(inspect_pcm16_wav(fit(sine(231000))).frames, SLOT)

    def test_invalid_slots_and_stereo(self):
        with self.assertRaises(SpeechFitError):
            fit_pcm16_wav_to_slot(sine(130560), ticks_per_second=16000,
                                  duration_ticks=48125)
        mono = sine(1000)
        stereo = bytearray(mono[:44] + inspect_pcm16_wav(mono).pcm * 2)
        struct.pack_into("<I", stereo, 4, len(stereo) - 8)
        struct.pack_into("<H", stereo, 22, 2)
        struct.pack_into("<I", stereo, 28, RATE * 4)
        struct.pack_into("<H", stereo, 32, 4)
        struct.pack_into("<I", stereo, 40, len(stereo) - 44)
        with self.assertRaises(SpeechFitError):
            fit(bytes(stereo))
        for tps, ticks in ((True, 1), (10000, False), (0, 1), (10000, -1), (10000, 1)):
            with self.subTest(tps=tps, ticks=ticks), self.assertRaises(SpeechFitError):
                fit_pcm16_wav_to_slot(mono, ticks_per_second=tps, duration_ticks=ticks)
        with self.assertRaises(SpeechFitError):
            fit_pcm16_wav_to_slot(sine(32), ticks_per_second=RATE, duration_ticks=16)

    def test_under_slot_non_divisible_slot_is_unchanged(self):
        payload = sine(1000)
        self.assertIs(fit_pcm16_wav_to_slot(
            payload, ticks_per_second=16000, duration_ticks=48125), payload)

    def test_harmonic_stack_rms(self):
        count = 130560
        values = [round((.6 + .4 * math.sin(math.pi * i / count) ** 2) * sum(
            6000 / harmonic * math.sin(2 * math.pi * 120 * harmonic * i / RATE)
            for harmonic in range(1, 5))) for i in range(count)]
        source = pcm_wav(struct.pack(f"<{count}h", *values), RATE)
        fitted = fit(source)
        self.assertEqual(inspect_pcm16_wav(fitted).frames, SLOT)
        source_power = sum(value * value for value in values) / count
        fitted_power = sum(value * value for value in samples(fitted)) / SLOT
        self.assertTrue(.5 < math.sqrt(fitted_power / source_power) < 1.5)

    def test_paired_media_admission(self):
        self.assertEqual(speech_metadata(self.fitted)["frames"], SLOT)
        with self.assertRaises(PairedMediaError):
            speech_metadata(self.source)

    def test_mix_admission(self):
        args = scene()
        old = args["sheet"]
        utterance = replace(old.script.utterances[0], start_ticks=0, duration_ticks=48125)
        script = DialogueScript("script", 1, (utterance,))
        cue = replace(next(cue for cue in old.cues if cue.utterance_id == "en-line"),
                      start_ticks=0, duration_ticks=48125)
        sheet = replace(old, timebase=replace(old.timebase, ticks_per_second=10000,
            duration_ticks=48125), script=script, cues=(cue,),
            stems=tuple(stem for stem in old.stems if stem.kind == "dialogue"),
            shots=(replace(old.shots[0], duration_ticks=48125),))
        original = next(source for source in args["sources"]
                        if isinstance(source, SpeechArtifact))
        request = replace(original.request,
            sheet=RevisionPin(sheet.sheet_id, sheet.revision, sheet.digest()), script=script.pin(),
            utterance=RevisionPin(utterance.utterance_id, utterance.text_revision, utterance.digest()),
            start_ticks=0, duration_ticks=48125, timebase=sheet.timebase)
        args.update(sheet=sheet, current_script=script.pin(),
            plan=replace(args["plan"], shots=(replace(args["plan"].shots[0],
                                                     duration_seconds=4.8125),)),
            settings=MixSettings((StemMix("dialogue"),), sample_rate=RATE))
        for payload in (self.fitted, self.source):
            source = replace(original, request=request, payload=payload,
                pin=replace(original.pin, sha256=hashlib.sha256(payload).hexdigest()),
                frames=inspect_pcm16_wav(payload).frames)
            options = dict(args, sources=(source,), preset=BROADCAST_LINEAR, output_id="mixed",
                           output_revision=1, tool=Meter(), expected_tool=IDENTITY)
            if payload is self.fitted:
                self.assertEqual(inspect_pcm16_wav(mix_scene(**options).payload).frames, SLOT)
            else:
                with self.assertRaises(MixError):
                    mix_scene(**options)
