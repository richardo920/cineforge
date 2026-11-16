"""Offline orchestration/receipt tests. Every cloud/native component is replaced before use."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_s2v_profile import png
from test_speech import wav_bytes

from cineforge import a20_driver as driver
from cineforge.a20_inputs import INPUT_NAMES, A20Error, CreativeInputs, FrozenInputs
from cineforge.audio_analysis import ToolIdentity
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.private_store import PrivateRunDirectory
from cineforge.rights import EvidenceRef
from cineforge.runpod import RunpodPodSpec
from cineforge.s2v_profile import IMAGE, L40S_GPU_ID
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext


def creative(root):
    root.mkdir()
    paths = {}
    for name in INPUT_NAMES:
        value = {"actor_png": png(width=832, height=480), "seed": b"15001",
                 "en_line": b"Approved English.", "zh_line": "批准的中文。".encode(),
                 "voice_design": b"Approved fictional timbre"}.get(name, wav_bytes())
        path = root / name
        path.write_bytes(value)
        paths[name] = path
    return CreativeInputs(paths)


class InputTests(unittest.TestCase):
    def test_every_missing_input_names_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = creative(Path(tmp).resolve() / "data")
            for name in INPUT_NAMES:
                with self.subTest(name=name):
                    paths = dict(inputs.paths)
                    del paths[name]
                    with self.assertRaisesRegex(A20Error, "A20_MISSING_" + name.upper()):
                        CreativeInputs(paths).read()
                    value = inputs.paths[name].read_bytes()
                    inputs.paths[name].write_bytes(b"")
                    with self.assertRaisesRegex(A20Error, "A20_MISSING_" + name.upper()):
                        inputs.read()
                    inputs.paths[name].write_bytes(value)

    def test_seed_must_be_explicit_exact_15001_and_actor_not_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = creative(Path(tmp).resolve() / "data")
            for value in (b"0", b"15002", b"15001.0", b"True"):
                inputs.paths["seed"].write_bytes(value)
                with self.assertRaisesRegex(A20Error, "A20_INVALID_SEED"):
                    inputs.read()
            inputs.paths["seed"].write_bytes(b"15001")
            paths = dict(inputs.paths)
            paths["actor_png"] = Path(tmp).resolve() / "m52_actor_pending_approval.png"
            paths["actor_png"].write_bytes(inputs.paths["actor_png"].read_bytes())
            with self.assertRaisesRegex(A20Error, "PLACEHOLDER"):
                CreativeInputs(paths).read()

    def test_freeze_detects_code_data_add_delete_and_symlink(self):
        for mutation in ("code", "data", "add", "delete", "link"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                inputs = creative(root / "data")
                code = root / "controller"
                code.mkdir()
                source = code / "controller.py"
                source.write_text("approved = True\n")
                frozen = FrozenInputs.capture(inputs, (code,))
                frozen.check()
                self.assertEqual(frozen.receipt["gate"], "frozen-CODE+DATA")
                if mutation == "code":
                    source.write_text("approved = False\n")
                elif mutation == "data":
                    inputs.paths["en_line"].write_text("Unapproved change")
                elif mutation == "add":
                    (code / "helper.py").write_text("pass\n")
                elif mutation == "delete":
                    source.unlink()
                else:
                    (code / "helper.py").symlink_to(source)
                with self.assertRaises(ValueError):
                    frozen.check()

    def test_private_payload_is_atomic_and_outside_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            with PrivateRunDirectory(Path(tmp).resolve() / "out") as store:
                name = store.put_payload(b"evidence", "json")
                self.assertEqual((store.path / name).read_bytes(), b"evidence")
                with self.assertRaises(ValueError):
                    store.put_payload(b"overwrite", "../key")
                self.assertEqual(store.put_payload(b"evidence", "json"), name)


class CoreTests(unittest.TestCase):
    def probe(self, **changes):
        kwargs = dict(deadline=100, prep={"failure_code": "A20_COMFY_PREPARE_FAILED"},
                      staged=None, frozen={"code_sha256": "x"},
                      snapshot=Mock(return_value=NS(state="ready")), clock=lambda: 1)
        kwargs.update(changes)
        probe = driver.CoreProbe()
        return probe, kwargs, probe.run(**kwargs)

    def test_fixed_schema_allowlist_once_and_no_runtime_handle(self):
        probe, kwargs, result = self.probe()
        self.assertEqual(set(result), {"probe", "trigger", "invocation", "reads", "findings", "verdict"})
        self.assertEqual(set(result["findings"]), {"failed_step", "failure_code",
            "staged_selection_present", "frozen_receipt_present", "journal_state"})
        self.assertEqual(result["reads"], ["prep_attempt", "staged_selection_manifest",
            "frozen_code_data_receipt", "budget_lifecycle_snapshot"])
        self.assertEqual(result["findings"]["journal_state"], "ready")
        self.assertFalse(result["findings"]["staged_selection_present"])
        with self.assertRaisesRegex(A20Error, "ALREADY_CONSUMED"):
            probe.run(**kwargs)
        kwargs["snapshot"].assert_called_once_with()

    def test_exhausted_original_cap_performs_no_fact_reads(self):
        class Unreadable(dict):
            def __getitem__(self, key):
                raise AssertionError("fact read after cap")
            def __bool__(self):
                raise AssertionError("fact read after cap")
        _, kwargs, result = self.probe(deadline=1, prep=Unreadable(), frozen=Unreadable())
        self.assertEqual(result["reads"], [])
        self.assertEqual(result["findings"]["failure_code"], "A20_CORE_CAP_EXHAUSTED")
        kwargs["snapshot"].assert_not_called()

    def test_probe_failure_is_a_result_not_retried(self):
        _, kwargs, result = self.probe(snapshot=Mock(side_effect=RuntimeError("private text")))
        self.assertEqual(result["findings"]["failure_code"], "A20_CORE_PROBE_ERROR")
        self.assertNotIn("private text", json.dumps(result))
        kwargs["snapshot"].assert_called_once_with()

    def test_hold_is_minimum_of_600_original_parent_and_lease_reserve(self):
        self.assertEqual(driver.attempt_cap(2000, 1800, 10), 610)
        self.assertEqual(driver.attempt_cap(100, 1800, 10), 100)
        self.assertEqual(driver.attempt_cap(2000, 200, 10), 90)
        for parent, remaining in ((10, 1000), (2000, 150), (float("inf"), 1000)):
            with self.assertRaises(A20Error):
                driver.attempt_cap(parent, remaining, 10)


class Harness:
    """Replace effects, retain real driver flow, local freeze and durable run storage."""
    def __init__(self, root, stack, failure=None, mutation=None):
        self.root, self.events, self.failure = root, [], failure
        self.inputs = creative(root / "data")
        code = root / "code"
        code.mkdir()
        self.source = code / "controller.py"
        self.source.write_text("# synthetic controller\n")
        self.frozen = FrozenInputs.capture(self.inputs, (code,))
        self.output = root / "run"
        self.budget = Mock()
        self.budget.__enter__ = Mock(return_value=self.budget)
        self.budget.__exit__ = Mock(return_value=False)
        self.budget.remaining_seconds.return_value = 1700
        self.budget.costs.return_value = {"actual_total_usd": None}
        self.budget.snapshot_for.return_value = NS(state="ready")
        self.worker = Mock()
        self.worker.__enter__ = Mock(return_value=self.worker)
        self.worker.__exit__ = Mock(side_effect=lambda *a: self.events.append("worker-close"))
        self.worker.prepare.side_effect = lambda *a, **kw: self.event("prepare", NS(record={}))
        self.worker.stage_s2v.side_effect = lambda *a, **kw: self.event("pair", NS())
        self.provisioner = Mock()
        self.provisioner.acquire.side_effect = lambda *a, **kw: self.event("acquire", NS(device_id="owned"))
        self.provisioner.release.side_effect = lambda *a: self.event("release")
        self.bridge = Mock()
        self.bridge.measure_capabilities.side_effect = lambda **kw: self.event("capabilities", {})
        self.bridge.stage.side_effect = lambda *a: self.event("stage", {})
        self.bridge.select.side_effect = lambda *a: self.event("select")
        self.bridge.preview.side_effect = lambda *a: self.event("design", wav_bytes(frames=100))
        self.bridge.verify_speech = Mock()
        self.adapter = Mock()
        self.adapter.synthesize.side_effect = self.synthesize
        approval = S2VBudgetApproval.for_scope(str(root / "budget"), "EUR-IS-2",
            EvidenceRef("private://synthetic/review", "a" * 64), gpu_id=L40S_GPU_ID)
        service = S2VServiceContext(approval)
        spec = RunpodPodSpec("a20test", IMAGE, L40S_GPU_ID, "EUR-IS-2", "ssh-ed25519 AAAA",
            max_hourly_usd=1.25, disk_gb=150, min_ram_per_gpu_gb=64)
        wheel = root / "sox.whl"
        wheel.write_bytes(b"synthetic staging wheel")
        self.bindings = driver.A20Bindings(service, spec, Mock(), root / "key", "acquisition",
            {}, wheel, OriginalVoiceDesign("Approved English.", "English",
                "Approved fictional timbre", "a" * 64),
            lambda installed, worker: NS(), lambda bridge: NS(), self.language,
            self.gate, self.checkpoint, root / "ffmpeg", root / "ffprobe",
            ToolIdentity("9.0.2", "a" * 64, "b" * 64))
        self.mutation = mutation
        patches = {"prepare_service_journal": lambda *a: self.event("budget-preflight"),
            "CloudBudgetSession": Mock(return_value=self.budget), "SSHWorker": Mock(return_value=self.worker),
            "RunpodPodProvisioner": Mock(return_value=self.provisioner),
            "RemoteNativeSpeechRuntime": Mock(return_value=self.bridge),
            "ClosedComfyRuntime": Mock(return_value=NS()),
            "QwenBaseSpeechAdapter": Mock(return_value=self.adapter),
            "select_s2v": lambda gate: NS(bind_worker=lambda *a: NS()),
            "run_s2v": self.s2v, "import_original_stem": lambda *a, **kw: self.event("stem", NS()),
            "mix_scene": lambda *a, **kw: self.event("mix", NS(payload=wav_bytes(frames=300))),
            "FFmpegTool": Mock(), "mux_delivery": lambda *a, **kw: self.event("mux",
                {"clip": "synthetic.mkv", "video_sha256": "a" * 64})}
        for name, value in patches.items():
            stack.enter_context(patch.object(driver, name, value))
        self.patches = patches

    def event(self, name, result=None):
        self.events.append(name)
        if self.failure == name:
            raise RuntimeError("private error")
        if self.failure == "interrupt-" + name:
            raise KeyboardInterrupt
        return result

    def checkpoint(self, name, facts):
        self.event("checkpoint-" + name)
        if self.mutation == name:
            self.source.write_text("# changed source\n")

    def synthesize(self, sheet, *a, **kw):
        language = sheet.script.utterances[0].language
        return self.event("speech-" + language, NS(payload=wav_bytes(frames=200 if language == "en" else 250)))

    def language(self, language, reference):
        text = self.inputs.read()[language + "_line"].decode()
        return NS(sheet=NS(script=NS(utterances=[NS(utterance_id="line", language=language, text=text)],
            pin=lambda: NS())), utterance_id="line", voices=("same-voice",),
            reference=NS(transcript="Approved English."), plan=NS(), registry=NS(), ledger=NS(), picture=NS(),
            stems=tuple(NS(name=n, cue_id=n, pin=NS(), review=NS()) for n in ("music", "ambience", "foley")),
            mix_settings=NS(), preset=NS())

    def gate(self, context, speech, bridge):
        return NS(speech=speech, service=self.bindings.service, sheet=context.sheet,
                  verify_speech=bridge.verify_speech,
                  actor=NS(read=lambda: self.inputs.read()["actor_png"]))

    def s2v(self, *args, **kwargs):
        language = Path(args[3]).name
        self.event("s2v-" + language)
        return {"manifest": {"artifacts": [{"uri": "artifact-" + "a" * 64 + ".webm", "sha256": "a" * 64}]}}

    def run(self):
        return driver.run_a20(self.bindings, self.frozen, self.output,
                              absolute_deadline=driver.time.monotonic() + 1800, allow_live=True)


class DriverTests(unittest.TestCase):
    def test_sequence_once_only_prepare_two_languages_and_delivery(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            result = h.run()
            expected = ["budget-preflight", "acquire", "capabilities", "stage", "checkpoint-postinstall",
                "select", "checkpoint-select", "design", "checkpoint-design", "speech-en", "checkpoint-EN",
                "prepare", "pair", "s2v-en", "speech-zh", "checkpoint-ZH", "pair", "s2v-zh",
                "stem", "stem", "stem", "mix", "mux", "stem", "stem", "stem", "mix", "mux",
                "release", "worker-close"]
            self.assertEqual(h.events, expected)
            self.assertEqual(result["outcome"], "pending_human_review")
            self.assertTrue(result["release_verified"])
            self.assertIsNone(result["core_probe"])
            self.assertEqual([c["name"] for c in result["checkpoints"]], list(driver.CHECKPOINTS))
            h.worker.prepare.assert_called_once()
            h.provisioner.release.assert_called_once_with("owned")
            self.assertEqual(json.loads((h.output / "run.json").read_bytes()), result)
            h.budget.snapshot_for.assert_not_called()
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 1800)
            self.assertNotIn("cleanup_timeout_seconds", h.patches["RunpodPodProvisioner"].call_args.kwargs)

    def test_prepare_failure_core_only_then_cleanup_never_second_runtime_probe(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack, failure="prepare")
            with self.assertRaises(A20Error):
                h.run()
            self.assertEqual(h.events[-3:], ["prepare", "release", "worker-close"])
            h.bridge.measure_capabilities.assert_called_once()
            h.worker.stage_s2v.assert_not_called()
            h.budget.snapshot_for.assert_called_once_with("acquisition")
            record = json.loads((h.output / "run.json").read_bytes())
            self.assertEqual(record["core_probe"]["invocation"], 1)
            self.assertEqual(record["outcome"], "incomplete")
            self.assertTrue(record["release_verified"])

    def test_core_error_and_expired_cap_still_record_once_before_release(self):
        for expired in (False, True):
            with self.subTest(expired=expired), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack, failure="prepare")
                h.budget.snapshot_for.side_effect = RuntimeError("private diagnostic error")
                if expired:
                    # Core's default clock retains real monotonic; a historical
                    # parent cap must refuse all fact reads, even in this fake run.
                    stack.enter_context(patch.object(driver.time, "monotonic", return_value=1))
                def release(owned, h=h):
                    saved = json.loads((h.output / "run.json").read_bytes())
                    self.assertEqual(saved["core_probe"]["invocation"], 1)
                    h.event("release")
                h.provisioner.release.side_effect = release
                with self.assertRaises(A20Error):
                    h.run()
                saved = json.loads((h.output / "run.json").read_bytes())
                expected = "A20_CORE_CAP_EXHAUSTED" if expired else "A20_CORE_PROBE_ERROR"
                self.assertEqual(saved["core_probe"]["findings"]["failure_code"], expected)
                if expired:
                    self.assertEqual(saved["core_probe"]["reads"], [])
                    h.budget.snapshot_for.assert_not_called()
                else:
                    h.budget.snapshot_for.assert_called_once()
                h.provisioner.release.assert_called_once()

    def test_changed_data_wheel_or_recipe_at_checkpoint_blocks_prepare(self):
        for target in ("data", "wheel", "recipe"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack)
                def checkpoint(name, facts, h=h, target=target):
                    # Returned full-preview facts are copies, never live authority.
                    facts["outcome"] = "forged"
                    if name == "EN":
                        if target == "data":
                            h.inputs.paths["en_line"].write_text("Changed")
                        elif target == "wheel":
                            h.bindings.sox_wheel.write_bytes(b"changed wheel")
                        else:
                            h.bindings.recipe["extra"] = "changed recipe"
                h.bindings = replace(h.bindings, checkpoint=checkpoint)
                with self.assertRaises(ValueError):
                    h.run()
                h.worker.prepare.assert_not_called()
                h.provisioner.release.assert_called_once_with("owned")

    def test_changed_video_before_mux_cannot_publish_delivery(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            stack.enter_context(patch.object(driver, "mux_delivery",
                return_value={"clip": "changed.mkv", "video_sha256": "b" * 64}))
            with self.assertRaises(A20Error):
                h.run()
            h.provisioner.release.assert_called_once_with("owned")
            record = json.loads((h.output / "run.json").read_bytes())
            self.assertEqual(record["outcome"], "incomplete")
            self.assertNotIn("delivery", record["languages"]["en"])

    def test_failed_acquisition_and_worker_finalizer_never_publish_candidate_readiness(self):
        for boundary in ("acquire", "worker-close"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack, failure=boundary)
                if boundary == "worker-close":
                    h.worker.__exit__.side_effect = RuntimeError("private finalizer error")
                with self.assertRaises(ValueError if boundary == "acquire" else RuntimeError):
                    h.run()
                record = json.loads((h.output / "run.json").read_bytes())
                self.assertEqual(record["outcome"], "incomplete")
                if boundary == "worker-close":
                    h.provisioner.release.assert_called_once()
                else:
                    h.provisioner.release.assert_not_called()

    def test_all_effect_failure_boundaries_cleanup_no_replay_no_core(self):
        for failure in ("capabilities", "stage", "select", "design", "speech-en", "pair", "s2v-en",
                        "speech-zh", "s2v-zh", "mix", "mux", "checkpoint-EN", "release"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack, failure=failure)
                with self.assertRaises(A20Error):
                    h.run()
                h.provisioner.release.assert_called_once_with("owned")
                h.worker.__exit__.assert_called_once()
                record = json.loads((h.output / "run.json").read_bytes())
                self.assertEqual(record["outcome"], "incomplete")
                self.assertIsNone(record["core_probe"])
                if failure == "release":
                    self.assertFalse(record["release_verified"])
                    self.assertEqual(record["cleanup_error"], "A20_RELEASE_UNCERTAIN")

    def test_midrun_source_change_refuses_further_effect_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack, mutation="EN")
            with self.assertRaises(ValueError):
                h.run()
            h.worker.prepare.assert_not_called()
            h.provisioner.release.assert_called_once_with("owned")

    def test_ctrl_c_cleanup_and_no_further_work(self):
        for point in ("stage", "prepare", "s2v-en", "release"):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack, failure="interrupt-" + point)
                with self.assertRaises(KeyboardInterrupt):
                    h.run()
                h.provisioner.release.assert_called_once_with("owned")
                h.worker.__exit__.assert_called_once()

    def test_no_optin_no_acquisition_and_changed_frozen_data_before_acquire(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            with self.assertRaisesRegex(A20Error, "OPT_IN"):
                driver.run_a20(h.bindings, h.frozen, h.output, absolute_deadline=1)
            h.inputs.paths["en_line"].write_text("Changed")
            with self.assertRaises(A20Error):
                h.run()
            h.provisioner.acquire.assert_not_called()


class CLITests(unittest.TestCase):
    def test_help_missing_input_codes_and_no_private_diagnostics(self):
        help_result = subprocess.run([sys.executable, "-m", "cineforge", "a20-run", "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--controller-sha256", help_result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve() / "inputs.json"
            source.write_text("{}")
            result = subprocess.run([sys.executable, "-m", "cineforge", "a20-run", "--inputs", str(source),
                "--controller", "PRIVATE_PATH", "--controller-root", "PRIVATE_ROOT",
                "--controller-sha256", "PRIVATE_HASH", "--output-dir", "PRIVATE_OUTPUT"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("stage=a20-run", result.stderr)
            self.assertIn("reason=a20-missing-en_line", result.stderr)
            self.assertNotIn("PRIVATE_", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
