"""Synthetic PCM and trusted fake meters; no native installation dependency."""
from __future__ import annotations

import hashlib
import math
import struct
import traceback
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_audio_analysis import meter_output
from test_speech import EVIDENCE, Runtime, adapter, fixture, reviewed

from cineforge.audio import DialogueScript, RevisionPin
from cineforge.audio_analysis import ToolIdentity
from cineforge.audio_mix import (
    BROADCAST_LINEAR,
    INTERNAL_REVIEW_STEREO,
    DeliveryPreset,
    MixedArtifact,
    MixError,
    MixReport,
    MixSettings,
    StemMix,
    _admit,
    _render,
    mix_scene,
)
from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.audio_process import AudioProcessError
from cineforge.rights import RightsLedger, Use, with_review
from cineforge.stem_import import import_original_stem

IDENTITY = ToolIdentity("fixture-v1", "a" * 64, "b" * 64, "fake")


def pcm(frames=24000, *, channels=1, value=1000, frequency=None, rate=24000):
    samples = bytearray()
    for frame in range(frames):
        sample = round(value * math.sin(2 * math.pi * frequency * frame / rate)) if frequency else value
        for channel in range(channels):
            samples.extend(struct.pack("<h", sample if channel == 0 else -sample))
    return (b"RIFF" + struct.pack("<I", len(samples) + 36) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, rate, 2 * rate * channels, channels * 2, 16)
        + b"data" + struct.pack("<I", len(samples)) + samples)


class Meter:
    def __init__(self, levels=None, action=None, identity=IDENTITY):
        self.levels = levels or [("-23", "-10"), ("-23", "-10")]
        self.action, self.identity = action, identity
        self.calls = []

    def identify(self, **limits):
        self.calls.append(("identify", limits))
        return self.identity

    def measure(self, payload, **limits):
        index = sum(item[0] == "measure" for item in self.calls)
        self.calls.append(("measure", payload, limits))
        if self.action:
            self.action()
        return meter_output(*self.levels[min(index, len(self.levels) - 1)])


def scene(*, background=1000, dialogue=2000, stereo=False, frequency=None, rate=24000):
    args, actor, _ = fixture()
    original_sheet = args["sheet"]
    payload = pcm(frames=rate, channels=2 if stereo else 1, value=background,
                  frequency=frequency, rate=rate)
    digest = hashlib.sha256(payload).hexdigest()
    cues = tuple(replace(cue, source=replace(cue.source,
        subject=replace(cue.source.subject, sha256=digest))) if cue.kind != "dialogue"
        else cue for cue in original_sheet.cues)
    sheet = replace(original_sheet, cues=cues)
    ledger = RightsLedger()
    bindings = {*(cue.source for cue in cues), *(ref.permission for ref in actor.references)}
    for binding in bindings:
        record = reviewed(binding)
        ledger.register(with_review(replace(record, allowed_uses=(*record.allowed_uses, Use("modify"))),
                                    reviewer="fixture-reviewer", evidence=EVIDENCE))
    args.update(sheet=sheet, ledger=ledger)
    sources = [import_original_stem(sheet, cue.cue_id, payload,
        RevisionPin("original-" + cue.cue_id, 1, digest), ledger=ledger,
        original_source_review=EVIDENCE) for cue in cues if cue.kind != "dialogue"]
    runtime = Runtime(payload=pcm(frames=round(rate * .3), value=dialogue,
                                 frequency=frequency, rate=rate))
    speech_adapter = adapter(args, runtime=runtime)
    for line in sheet.script.utterances:
        sources.append(speech_adapter.synthesize(utterance_id=line.utterance_id,
            output_id=line.utterance_id, output_revision=1, **args))
    settings = MixSettings(tuple(sorted((StemMix(item.stem_id) for item in sheet.stems),
                                        key=lambda item: item.stem_id)), sample_rate=rate)
    keys = ("plan", "registry", "ledger", "current_picture", "current_script")
    return dict(sheet=sheet, sources=tuple(sources), settings=settings,
                **{key: args[key] for key in keys})


def bed_scene(payload, *, duration_ticks=1000):
    args = scene()
    old = args["sheet"]
    digest = hashlib.sha256(payload).hexdigest()
    cue = next(item for item in old.cues if item.kind == "music")
    cue = replace(cue, duration_ticks=duration_ticks, source=replace(cue.source,
        subject=replace(cue.source.subject, sha256=digest)))
    sheet = replace(old, script=DialogueScript("script", 1, ()),
        timebase=replace(old.timebase, duration_ticks=duration_ticks), cues=(cue,),
        stems=tuple(item for item in old.stems if item.kind == "music"),
        shots=(replace(old.shots[0], duration_ticks=duration_ticks),))
    ledger = RightsLedger()
    record = reviewed(cue.source)
    ledger.register(with_review(replace(record, allowed_uses=(*record.allowed_uses, Use("modify"))),
        reviewer="fixture-reviewer", evidence=EVIDENCE))
    source = import_original_stem(sheet, cue.cue_id, payload, RevisionPin("bed", 1, digest),
        ledger=ledger, original_source_review=EVIDENCE)
    plan = replace(args["plan"], shots=(replace(args["plan"].shots[0],
                                               duration_seconds=duration_ticks / 1000),))
    return dict(args, sheet=sheet, sources=(source,), ledger=ledger, plan=plan,
                current_script=sheet.script.pin(), settings=MixSettings((StemMix("music"),)))


class MixTests(unittest.TestCase):
    def setUp(self):
        self.args = scene()
        self.meter = Meter()

    def mix(self, **changes):
        options = dict(self.args, preset=BROADCAST_LINEAR, output_id="mixed",
                       output_revision=1, tool=self.meter, expected_tool=IDENTITY)
        options.update(changes)
        return mix_scene(**options)

    def rejected(self, **changes):
        caught = None
        try:
            self.mix(**changes)
        except BaseException as error:
            caught = error
        self.assertIsInstance(caught, MixError)
        self.assertIsNone(caught.__context__)
        self.assertIsNone(caught.__cause__)
        self.assertNotIn("CANARY", "".join(traceback.format_exception(caught)))
        return caught

    def render(self, **changes):
        import time
        args = dict(self.args, **changes)
        admitted, _ = _admit(args["sheet"], args["sources"], args["settings"], args["plan"],
            args["registry"], args["ledger"], args["current_picture"], args["current_script"])
        return _render(admitted, args["sheet"], args["settings"], time.monotonic() + 20)

    def test_exact_scene_bytes_lineage_codec_determinism_and_no_authority(self):
        with patch("subprocess.Popen", side_effect=AssertionError("native call")):
            first = self.mix()
            second = self.mix()
        self.assertEqual(first.payload, second.payload)
        # Used meter timeout budgets record elapsed work; media/settings/lineage are stable.
        self.assertEqual(first.report.output, second.report.output)
        self.assertEqual(first.report.settings, second.report.settings)
        self.assertEqual(first.report.inputs, second.report.inputs)
        self.assertEqual(first.report.delivery_status, "synthetic_fixture_only")
        self.assertEqual(first.report.export_authority, "none")
        self.assertEqual(first.report.human_review, "not_assessed")
        self.assertEqual(MixReport.from_json(first.report.to_json()), first.report)
        self.assertEqual(first.report.dependencies, self.args["sheet"].dependencies(speech=tuple(
            sorted((source.pin for source in self.args["sources"] if hasattr(source, "request")),
                   key=lambda item: item.input_id))))
        self.assertEqual(first.report.output.sha256, hashlib.sha256(first.payload).hexdigest())
        self.assertEqual((inspect_pcm16_wav(first.payload).frames,
                          inspect_pcm16_wav(first.payload).channels), (24000, 2))
        self.assertNotIn("payload=", repr(first))
        self.assertFalse(hasattr(first, "approved"))
        self.assertEqual(MixSettings.from_json(self.args["settings"].to_json()), self.args["settings"])

    def test_offsets_gains_fades_exact_samples_and_source_immutability(self):
        before = tuple(source.payload for source in self.args["sources"])
        controls = tuple(replace(item, gain_db=-6.020599913279624,
            fade_in_ticks=10, fade_out_ticks=10) for item in self.args["settings"].stems)
        settings = replace(self.args["settings"], stems=controls, duck_db=0)
        samples, gain = self.render(settings=settings)
        self.assertEqual(gain, 1)
        # Four1000 backgrounds,0.5gain,10ms(240frame)fade.
        self.assertEqual(samples[0], 0)
        self.assertAlmostEqual(samples[120 * 2], 1000)
        self.assertAlmostEqual(samples[240 * 2], 2000)
        # English starts at2400; at2520speech has reached0.5fade.
        self.assertAlmostEqual(samples[2400 * 2], 2000)
        self.assertAlmostEqual(samples[2520 * 2], 2500)
        self.assertAlmostEqual(samples[2640 * 2], 3000)
        self.assertEqual(samples[-1], 0)
        self.assertEqual(before, tuple(source.payload for source in self.args["sources"]))

    def test_duck_attack_hold_release_and_only_background_kinds(self):
        self.args = scene(background=1000, dialogue=2000)
        settings = replace(self.args["settings"], duck_db=-6.020599913279624,
                           attack_ticks=10, hold_ticks=10, release_ticks=20)
        samples, _ = self.render(settings=settings)
        self.assertEqual(samples[0], 4000)
        self.assertAlmostEqual(samples[(2400 + 239) * 2], 5000)
        self.assertAlmostEqual(samples[(9600 + 100) * 2], 3000)
        self.assertAlmostEqual(samples[(9600 + 240 + 239) * 2], 3500)
        self.assertAlmostEqual(samples[(9600 + 240 + 479) * 2], 4000)

    def test_stereo_channels_preserved_mono_duplicate_no_downmix_or_resample(self):
        self.args = scene(stereo=True, dialogue=0)
        samples, _ = self.render(settings=replace(self.args["settings"], duck_db=0))
        self.assertEqual(tuple(samples[:4]), (4000, -4000, 4000, -4000))
        self.rejected(settings=replace(self.args["settings"], channels=1))
        self.assertEqual(self.meter.calls, [])
        self.rejected(settings=replace(self.args["settings"], sample_rate=48000))
        self.assertEqual(self.meter.calls, [])

    def test_summed_clipping_prevention_and_explicit_safety_gain(self):
        self.args = scene(background=30000, dialogue=30000)
        samples, safety = self.render(settings=replace(self.args["settings"], duck_db=0))
        self.assertGreater(max(samples), 32767)
        self.assertLess(safety, 1)
        result = self.mix(settings=replace(self.args["settings"], duck_db=0))
        self.assertEqual(result.report.safety_gain, safety)
        wave = inspect_pcm16_wav(result.payload)
        self.assertLessEqual(max(abs(item[0]) for item in struct.iter_unpack("<h", wave.pcm)), 32700)

    def test_unaligned_overflow_nonfinite_bool_missing_stale_and_tampered_refuse_pretool(self):
        for name, values in (("duck_db", (True, float("nan"), 10**400)),
                             ("channels", (True, 3)), ("sample_rate", (True, 32000)),
                             ("attack_ticks", (True, -1, 2**63))):
            for value in values:
                with self.subTest(name=name, value=type(value)):
                    with self.assertRaises(MixError if name == "duck_db" else ValueError):
                        replace(self.args["settings"], **{name: value})
        for changes in (dict(sources=self.args["sources"][:-1]),
                        dict(current_script=replace(self.args["current_script"], revision=2)),
                        dict(current_picture=replace(self.args["current_picture"], revision=2)),
                        dict(settings=replace(self.args["settings"], sample_rate=44100)),
                        dict(sources=(replace(self.args["sources"][0], payload=b"CANARY"),
                                      *self.args["sources"][1:]))):
            self.rejected(**changes)
            self.assertEqual(self.meter.calls, [])
        with self.assertRaises(MixError):
            DeliveryPreset("CANARY", -16, -2)

    def test_revoked_rights_before_or_during_meter_never_return_output(self):
        rights = self.args["sheet"].cues[0].source.rights
        self.args["ledger"].revoke(rights, reason="CANARY")
        self.rejected()
        self.assertEqual(self.meter.calls, [])
        self.args = scene()
        rights = self.args["sheet"].cues[0].source.rights
        self.meter = Meter(action=lambda: self.args["ledger"].revoke(rights, reason="CANARY"))
        self.rejected()
        self.assertGreater(len(self.meter.calls), 0)

    def test_copy_only_or_no_modification_permissions_do_not_authorize_mixing(self):
        binding = self.args["sheet"].cues[0].source
        for restrictions in ((), ("no_modification",)):
            self.args = scene()
            old = self.args["ledger"]
            ledger = RightsLedger()
            for cue in self.args["sheet"].cues:
                if ledger.lookup(cue.source.rights) is None:
                    record = old.lookup(cue.source.rights)
                    if cue.source.rights == binding.rights:
                        record = with_review(replace(record, restrictions=restrictions,
                            allowed_uses=(Use("copy"), Use("reference"), Use("generate"))),
                            reviewer="fixture-reviewer", evidence=EVIDENCE)
                    ledger.register(record)
            # Preserve character reference declarations as well.
            for actor in (self.args["registry"].lookup(self.args["sheet"].script.utterances[0].character),):
                for ref in actor.references:
                    if ledger.lookup(ref.permission.rights) is None:
                        ledger.register(old.lookup(ref.permission.rights))
            self.rejected(ledger=ledger)
            self.assertEqual(self.meter.calls, [])

    def test_unavailable_silence_short_invalid_meters_and_changed_identity_never_compliance(self):
        self.rejected(tool=None)
        self.rejected(expected_tool=replace(IDENTITY, binary_sha256="c" * 64))
        self.args = scene(background=0, dialogue=0)
        self.rejected()
        self.args = scene()
        self.meter = Meter(levels=[("-inf", "-10")])
        self.rejected()
        self.meter = Meter(levels=[("-23", "-10"), ("-23", "-0.5")])
        self.rejected()
        self.meter = Meter(levels=[("-23", "-10"), ("-22", "-10")])
        self.rejected()
        self.meter = Meter()
        with self.assertRaises(MixError):
            MixedArtifact(inspect_pcm16_wav(self.mix().payload).pcm, self.mix().report)

    def test_fractional_sample_fades_oversize_and_overlong_pretool_refusal(self):
        self.args = scene(rate=44100)
        settings = replace(self.args["settings"], stems=tuple(replace(item, fade_in_ticks=1)
            for item in self.args["settings"].stems))
        self.rejected(settings=settings)
        self.assertEqual(self.meter.calls, [])
        self.args = scene()
        sheet = replace(self.args["sheet"], timebase=replace(self.args["sheet"].timebase,
                                                          duration_ticks=61000))
        self.rejected(sheet=sheet)
        self.assertEqual(self.meter.calls, [])
        source = replace(self.args["sources"][0], payload=b"X" * (16 * 1024 * 1024 + 1))
        self.rejected(sources=(source, *self.args["sources"][1:]))
        self.assertEqual(self.meter.calls, [])

    def test_shorter_actual_speech_leaves_reserved_tail_silent_without_trimming(self):
        self.args = scene(background=0)
        sources = list(self.args["sources"])
        index = next(index for index, source in enumerate(sources)
                     if hasattr(source, "request") and source.request.utterance.input_id == "en-line")
        payload = pcm(frames=2400, value=2000)
        sources[index] = replace(sources[index], payload=payload, frames=2400,
            pin=replace(sources[index].pin, sha256=hashlib.sha256(payload).hexdigest()))
        samples, _ = self.render(sources=tuple(sources))
        self.assertEqual(samples[2400 * 2], 2000)
        self.assertEqual(samples[4799 * 2], 2000)
        self.assertEqual(samples[4800 * 2], 0)
        self.assertEqual(samples[9599 * 2], 0)
        self.assertEqual(samples[14400 * 2], 2000)

    def test_short_silent_tool_fault_cleanup_uncertainty_and_cancellation(self):
        self.args = bed_scene(pcm(frames=4800), duration_ticks=200)
        self.rejected()
        self.assertEqual(self.meter.calls, [])
        self.args = bed_scene(pcm(value=0))
        self.rejected()
        self.assertEqual(self.meter.calls, [])
        self.args = scene()
        for fault in (RuntimeError("CANARY"), AudioProcessError("CANARY", cleanup_uncertain=True)):
            def fail():
                raise fault
            self.meter = Meter(action=fail)
            self.rejected()
        self.meter = Meter(action=lambda: (_ for _ in ()).throw(KeyboardInterrupt("CANARY")))
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.mix()
        self.assertNotIn("CANARY", "".join(traceback.format_exception(caught.exception)))
        self.assertIsNone(caught.exception.__context__)
        self.meter = Meter()
        self.rejected(expected_tool=replace(IDENTITY, kind="native", version="9.0.2"))
        self.assertEqual(self.meter.calls, [])

    def test_preset_target_gain_unattainable_crest_and_project_stereo_limits(self):
        self.meter = Meter(levels=[("-30", "-20"), ("-23", "-13")])
        result = self.mix()
        self.assertEqual(result.report.preset_gain_db, 7)
        self.meter = Meter(levels=[("-30", "-2")])
        self.rejected()
        self.assertEqual(sum(call[0] == "measure" for call in self.meter.calls), 1)
        self.meter = Meter(levels=[("-23", "-10"), ("-16", "-3")])
        result = self.mix(preset=INTERNAL_REVIEW_STEREO)
        self.assertEqual(result.report.preset_gain_db, 7)
        self.assertEqual(result.report.preset.target_lufs, -16)
        self.rejected(preset=INTERNAL_REVIEW_STEREO,
                      settings=replace(self.args["settings"], channels=1))


if __name__ == "__main__":
    unittest.main()
