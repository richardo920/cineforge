"""Long-form bounds with synthetic PCM and the existing trusted fake meter."""
from __future__ import annotations

import hashlib
import struct
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_audio_mix import IDENTITY, Meter
from test_speech import EVIDENCE, Runtime, adapter, fixture, reviewed

from cineforge.audio import RevisionPin
from cineforge.audio_mix import (
    BROADCAST_LINEAR,
    MAX_DURATION_SECONDS,
    MixError,
    MixSettings,
    StemMix,
    _admit,
    _render,
    mix_scene,
)
from cineforge.audio_pcm import MAX_PCM_BYTES, PCMError, inspect_pcm16_wav
from cineforge.contracts import ContractError
from cineforge.rights import RightsLedger, Use
from cineforge.stem_import import import_original_stem


def constant_pcm(seconds, *, rate=24000, channels=1, value=100):
    samples = struct.pack('<h', value) * (int(seconds * rate) * channels)
    return (b'RIFF' + struct.pack('<I', len(samples) + 36) + b'WAVEfmt '
        + struct.pack('<IHHIIHH', 16, 1, channels, rate, 2 * rate * channels, channels * 2, 16)
        + b'data' + struct.pack('<I', len(samples)) + samples)


def long_scene(*, seconds=130, dialogue_count=27, rate=24000):
    args, actor, _ = fixture()
    old = args['sheet']
    lines = tuple(replace(old.script.utterances[0], utterance_id=f'line-{i:02}',
        start_ticks=100 + i * 1000) for i in range(dialogue_count))
    script = replace(old.script, utterances=lines)
    payloads = {kind: constant_pcm(seconds if kind != 'foley' else 1, rate=rate)
                for kind in ('music', 'ambience', 'foley')}
    beds = tuple(replace(cue, duration_ticks=seconds * 1000 if cue.kind != 'foley' else 1000,
        source=replace(cue.source, subject=replace(cue.source.subject,
            sha256=hashlib.sha256(payloads[cue.kind]).hexdigest())))
        for cue in old.cues if cue.kind in payloads)
    template = next(cue for cue in old.cues if cue.kind == 'dialogue')
    cues = beds + tuple(replace(template, cue_id=f'cue-{i:02}',
        utterance_id=line.utterance_id, start_ticks=line.start_ticks)
        for i, line in enumerate(lines))
    sheet = replace(old, script=script,
        timebase=replace(old.timebase, duration_ticks=seconds * 1000),
        shots=(replace(old.shots[0], duration_ticks=seconds * 1000),),
        stems=tuple(stem for stem in old.stems if stem.kind != 'effects'),
        cues=tuple(sorted(cues, key=lambda cue: (cue.start_ticks, cue.cue_id))))
    ledger = RightsLedger()
    for binding in {*(cue.source for cue in sheet.cues),
                    *(ref.permission for ref in actor.references)}:
        record = reviewed(binding)
        ledger.register(reviewed(binding, allowed_uses=(*record.allowed_uses, Use('modify'))))
    plan = replace(args['plan'], shots=(replace(args['plan'].shots[0],
                                              duration_seconds=seconds),))
    args.update(sheet=sheet, ledger=ledger, plan=plan, current_script=script.pin())
    sources = [import_original_stem(sheet, cue.cue_id, payloads[cue.kind],
        RevisionPin('original-' + cue.cue_id, 1, cue.source.subject.sha256),
        ledger=ledger, original_source_review=EVIDENCE) for cue in beds]
    speech = adapter(args, runtime=Runtime(payload=constant_pcm(.3, rate=rate, value=1000)))
    sources.extend(speech.synthesize(utterance_id=line.utterance_id,
        output_id=line.utterance_id, output_revision=1, **args) for line in lines)
    keys = ('plan', 'registry', 'ledger', 'current_picture', 'current_script')
    return dict(sheet=sheet, sources=tuple(sources), settings=MixSettings(tuple(sorted(
        (StemMix(stem.stem_id) for stem in sheet.stems), key=lambda stem: stem.stem_id)),
        sample_rate=rate, timeout_seconds=MAX_DURATION_SECONDS),
        **{key: args[key] for key in keys})


def admit(args):
    return _admit(*(args[key] for key in ('sheet', 'sources', 'settings', 'plan',
                                        'registry', 'ledger', 'current_picture', 'current_script')))


class LongFormMixTests(unittest.TestCase):
    def test_130_second_beds_27_dialogues_and_foley_mix_at_delivery_rate(self):
        with patch('subprocess.Popen', side_effect=AssertionError('native call')):
            args = long_scene()
            admitted, _ = admit(args)
            self.assertEqual(len(admitted), 30)
            self.assertEqual(sum(item[3] == 'dialogue' for item in admitted), 27)
            self.assertEqual([item[1].frames for item in admitted
                              if item[3] in ('music', 'ambience')], [130 * 24000] * 2)
            result = mix_scene(**args, preset=BROADCAST_LINEAR, output_id='long-mix',
                output_revision=1, tool=Meter(), expected_tool=IDENTITY)
        wave = inspect_pcm16_wav(result.payload, max_input_bytes=MAX_PCM_BYTES,
            max_duration_seconds=int(MAX_DURATION_SECONDS), long_form=True)
        self.assertEqual((wave.frames, wave.sample_rate, wave.channels), (130 * 24000, 24000, 2))
        self.assertEqual(len(result.payload), 130 * 24000 * 2 * 2 + 44)
        self.assertEqual(result.report.preset_gain_db, 0)
        self.assertEqual(result.report.safety_gain, 1)

    def test_duration_frame_timeout_and_source_count_boundaries(self):
        args = long_scene(seconds=180, dialogue_count=37, rate=48000)
        admitted, _ = admit(args)
        self.assertEqual(len(admitted), 40)
        self.assertEqual(max(item[0].frames for item in admitted), 180 * 48000)
        record = admitted[0][0]
        self.assertEqual(replace(record, frames=180 * 48000).frames, 180 * 48000)
        with self.assertRaises(ContractError):
            replace(record, frames=180 * 48000 + 1)
        with self.assertRaises(MixError):
            replace(args['settings'], timeout_seconds=180.001)
        with self.assertRaisesRegex(MixError, 'stale or unadmitted'):
            admit(long_scene(dialogue_count=38))
        # Extend only the timeline so all source metadata remains otherwise valid.
        sheet = replace(args['sheet'], timebase=replace(args['sheet'].timebase,
                                                       duration_ticks=180001))
        with self.assertRaisesRegex(MixError, 'supported duration'):
            admit(dict(args, sheet=sheet))
        with self.assertRaisesRegex(MixError, 'output bounds'):
            _render((), sheet, args['settings'], time.monotonic() + 180)

    def test_oversize_inputs_and_output_refuse_before_processing(self):
        args = long_scene()
        # An aggregate over 96 MiB must fail before inspecting any source bytes.
        source = replace(args['sources'][0], payload=b'X' * (96 * 1024 * 1024 + 1))
        with patch('cineforge.audio_mix.inspect_pcm16_wav', side_effect=AssertionError('inspect')):
            with self.assertRaisesRegex(MixError, 'input byte bound'):
                admit(dict(args, sources=(source, *args['sources'][1:])))
        # The byte guard is independent of the duration guard (48 MiB + WAV header).
        from cineforge import audio_mix
        with patch.object(audio_mix, '_MAX_BYTES', 44):
            with self.assertRaisesRegex(MixError, 'output bounds'):
                _render((), args['sheet'], args['settings'], time.monotonic() + 180)
        with self.assertRaises(PCMError):
            inspect_pcm16_wav(constant_pcm(181), max_input_bytes=MAX_PCM_BYTES,
                max_duration_seconds=180, long_form=True)
        with self.assertRaises(PCMError):
            inspect_pcm16_wav(b'X' * (MAX_PCM_BYTES + 1), max_input_bytes=MAX_PCM_BYTES,
                max_duration_seconds=180, long_form=True)
