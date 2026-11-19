"""Offline orchestration/receipt tests. Cloud admission is real; native effects are replaced before use."""
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
from cineforge.cloud_budget import CloudBudgetError, CloudBudgetSession
from cineforge.cloud_journal import CloudJournal
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.private_store import PrivateRunDirectory
from cineforge.rights import EvidenceRef
from cineforge.runpod import AcquisitionIntent, LeaseSnapshot, RunpodPodSpec
from cineforge.runpod_transport import CloudResponse
from cineforge.s2v_profile import IMAGE, L40S_GPU_ID
from cineforge.s2v_run import ATTEMPT_CAP_SECONDS
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext, prepare_service_journal
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.workflows import canonical, strict_object


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

    def test_default_clock_is_resolved_at_call_time(self):
        snapshot = Mock()
        with patch.object(driver.time, "monotonic", return_value=100) as clock:
            result = driver.CoreProbe().run(deadline=100, prep={}, staged=None,
                                             frozen={}, snapshot=snapshot)
        clock.assert_called_once_with()
        self.assertEqual(result["reads"], [])
        self.assertEqual(result["findings"]["failure_code"], "A20_CORE_CAP_EXHAUSTED")
        snapshot.assert_not_called()

    def test_hold_is_minimum_of_attempt_cap_original_parent_and_lease_reserve(self):
        self.assertEqual(driver.attempt_cap(2000, 1800, 10), 1690)
        self.assertEqual(driver.attempt_cap(100, 1800, 10), 100)
        self.assertEqual(driver.attempt_cap(2000, 200, 10), 90)
        self.assertEqual(driver.attempt_cap(10000, 9000, 10), 10 + ATTEMPT_CAP_SECONDS)
        for parent, remaining in ((10, 1000), (2000, 150), (float("inf"), 1000)):
            with self.assertRaises(A20Error):
                driver.attempt_cap(parent, remaining, 10)

    def test_each_language_gets_a_fresh_attempt_window(self):
        # a20_driver.run calls attempt_cap inside its per-language EN/ZH loop.
        for now in (10, 2000):
            self.assertEqual(driver.attempt_cap(10000, 9000, now), now + ATTEMPT_CAP_SECONDS)


class AdmissionCatalog:
    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        if method != "GET" or not path.startswith("/v2/catalog/gpus/"):
            raise AssertionError("unexpected cloud request")
        return CloudResponse(200, canonical(dict(id=L40S_GPU_ID, secure=True,
            price={"secure": 0.89}, availability="HIGH", maxCount={"secure": 8},
            dataCenters=[{"id": "EUR-IS-2", "availability": "LOW"}],
            cudaVersions=[{"version": "13.2", "available": True}])))


class AcquisitionAdmissionTests(unittest.TestCase):
    def check_admission(self, lifetime_seconds, expected_reserve, aggregate_usd=5):
        with tempfile.TemporaryDirectory() as tmp:
            approval = S2VBudgetApproval.for_scope(str(Path(tmp).resolve() / "journal"),
                "EUR-IS-2", EvidenceRef("private://synthetic/review", "a" * 64),
                gpu_id=L40S_GPU_ID, lifetime_seconds=lifetime_seconds,
                aggregate_usd=aggregate_usd)
            service = S2VServiceContext(approval)
            spec = RunpodPodSpec("a20test", IMAGE, L40S_GPU_ID, "EUR-IS-2", "ssh-ed25519 AAAA",
                max_hourly_usd=1.25, disk_gb=150, min_ram_per_gpu_gb=64)
            derived = driver.acquisition_max_cost_usd(spec, approval)
            self.assertEqual(derived, min(expected_reserve, aggregate_usd))
            self.assertLessEqual(derived, approval.aggregate_usd)
            prepare_service_journal(service)
            intent = AcquisitionIntent("acquisition", spec, "a" * 32,
                "cineforge-a20test-admission", service.model, derived,
                approval.lifetime_seconds, 120, 10)
            with CloudBudgetSession(approval.journal_directory, approval, AdmissionCatalog(),
                                    guard_deployment_configured=True) as session:
                session.event_sink(intent, LeaseSnapshot(intent.key, "reserved"))
                if expected_reserve > aggregate_usd:
                    self.assertLess(derived, expected_reserve)
                    with self.assertRaisesRegex(CloudBudgetError, "aggregate estimated budget"):
                        session.before_create(intent)
                else:
                    self.assertGreaterEqual(derived, expected_reserve)
                    session.before_create(intent)
            # Reopen the journal to check durable authorization and reservation.
            with CloudJournal(approval.journal_directory) as journal:
                record = strict_object(journal.initial)["records"][intent.key]
                self.assertEqual(record["authorized"], expected_reserve <= aggregate_usd)
                self.assertEqual(record["reservation_usd"],
                                 expected_reserve if expected_reserve <= aggregate_usd else None)

    def test_7200_lifetime_admitted_with_254_reservation(self):
        self.check_admission(7200, 2.54)

    def test_1800_lifetime_preserves_exact_0635_reservation(self):
        self.check_admission(1800, 0.635)

    def test_7200_lifetime_over_aggregate_fails_closed(self):
        self.check_admission(7200, 2.54, aggregate_usd=2)


class Harness:
    """Replace effects, retain real driver flow, local freeze and durable run storage."""
    def __init__(self, root, stack, failure=None, mutation=None, lifetime_seconds=1800):
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
        self.budget.guard_plan.return_value = NS(stop_at_unix=15340., idle_seconds=120.)
        self.retention = {"retention_acquired_unix": 10000.,
            "retention_expires_unix": 15400., "lease_expires_unix": 15400.,
            "retention_window_seconds": 3600., "retention_window_seconds_effective": 3600.,
            "retention_bound_seconds": 14400.}
        self.budget.extend_retention.return_value = self.retention
        self.worker = Mock()
        self.worker.retain_live.return_value = dict(self.retention,
            guard_stop_at_unix=15340., guard_idle_seconds=120.,
            retention_heartbeat_pid=456, retention_heartbeat="active")
        self.worker.probe_retention_guard.return_value = False
        self.worker.connection_info = {"host": "8.8.8.8", "port": 2222, "username": "root"}
        self.worker.__enter__ = Mock(return_value=self.worker)
        self.worker.__exit__ = Mock(side_effect=lambda *a: self.events.append("worker-close"))
        self.worker.prepare.side_effect = lambda *a, **kw: self.event("prepare", NS(record={}))
        self.worker.stage_s2v.side_effect = lambda *a, **kw: self.event("pair", NS())
        self.provisioner = Mock()
        self.provisioner.inspect_lease.return_value = NS(state="ready", hourly_usd=1.09)
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
            EvidenceRef("private://synthetic/review", "a" * 64), gpu_id=L40S_GPU_ID,
            lifetime_seconds=lifetime_seconds)
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
        self.run_deadline = driver.time.monotonic() + 1800
        return driver.run_a20(self.bindings, self.frozen, self.output,
                              absolute_deadline=self.run_deadline, allow_live=True)


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
            self.assertEqual(h.provisioner.acquire.call_args.kwargs["max_cost_usd"], 0.635)
            self.assertEqual(h.events, expected)
            self.assertEqual(result["outcome"], "pending_human_review")
            self.assertTrue(result["release_verified"])
            self.assertNotIn("retained_node", result)
            h.provisioner.inspect_lease.assert_not_called()
            self.assertIsNone(result["core_probe"])
            self.assertEqual([c["name"] for c in result["checkpoints"]], list(driver.CHECKPOINTS))
            h.worker.prepare.assert_called_once()
            native = h.patches["RemoteNativeSpeechRuntime"]
            native.assert_called_once()
            self.assertEqual(native.call_args.kwargs["timeout_seconds"],
                             h.bindings.service.approval.lifetime_seconds)
            self.assertIsNotNone(native.call_args.kwargs["absolute_deadline"])
            self.assertLessEqual(native.call_args.kwargs["absolute_deadline"], h.run_deadline)
            self.assertGreater(native.call_args.kwargs["absolute_deadline"], h.run_deadline - 600)
            h.provisioner.release.assert_called_once_with("owned")
            self.assertEqual(json.loads((h.output / "run.json").read_bytes()), result)
            h.budget.snapshot_for.assert_not_called()
            self.assertEqual(h.patches["RunpodPodProvisioner"].call_args.kwargs["lease_seconds"], 1800)
            self.assertNotIn("cleanup_timeout_seconds", h.patches["RunpodPodProvisioner"].call_args.kwargs)

    def test_7200_lifetime_passes_derived_acquisition_cap(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack, lifetime_seconds=7200)
            h.run()
            self.assertEqual(h.provisioner.acquire.call_args.kwargs["max_cost_usd"], 2.54)

    def test_retains_node_on_success_and_each_failure_boundary(self):
        failures = (None, "capabilities", "stage", "select", "design", "speech-en",
                    "checkpoint-EN", "prepare", "pair", "s2v-en", "speech-zh", "s2v-zh",
                    "mix", "mux")
        for failure in failures:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack, failure=failure)
                h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
                mounts = [{"alias": "synthetic-driver-mount"}]
                h.bridge.measure_capabilities.side_effect = lambda h=h, mounts=mounts, **kw: h.event(
                    "capabilities", {"files": [{"driver_mount": mounts[0]}, {"status": "missing"}]})
                if failure is None:
                    result = h.run()
                    self.assertEqual(result["outcome"], "pending_human_review")
                else:
                    with self.assertRaisesRegex(A20Error, "A20_INCOMPLETE_RETAIN_PRIVATE_RUN"):
                        h.run()
                record = json.loads((h.output / "run.json").read_bytes())
                self.assertEqual(record["outcome"], "incomplete" if failure else "pending_human_review")
                self.assertFalse(record["release_verified"])
                self.assertEqual(record["retained_node"], {
                    "pod_id": "owned", "region": "EUR-IS-2", "gpu_id": L40S_GPU_ID,
                    "state": "ready", "hourly_usd": 1.09, "driver_version": None,
                    **h.retention, "guard_stop_at_unix": 15340., "guard_idle_seconds": 120.,
                    "retention_heartbeat_pid": 456, "retention_heartbeat": "active",
                    "max_retention_cost_usd": 1.09,
                    "driver_mounts": [] if failure == "capabilities" else mounts,
                    "ssh": h.worker.connection_info})
                h.provisioner.release.assert_not_called()
                h.provisioner.inspect_lease.assert_called_once_with("owned")
                h.worker.__exit__.assert_called_once()
                h.budget.extend_retention.assert_called_once_with("acquisition")

    def test_retention_handoff_precedes_close_and_failure_reports_only_known_guard(self):
        for broken in (False, True):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack)
                h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
                def handoff(facts, h=h, broken=broken):
                    h.worker.__exit__.assert_not_called()
                    self.assertEqual(facts, h.retention)
                    if broken:
                        raise WorkerError("fake spawn failure")
                    return h.worker.retain_live.return_value
                h.worker.retention_guard_alive = True
                h.worker.retention_guard_stop_at_unix = None
                h.worker.retain_live.side_effect = handoff
                if broken:
                    with self.assertRaises(A20Error):
                        h.run()
                else:
                    h.run()
                retained = json.loads((h.output / "run.json").read_bytes())["retained_node"]
                self.assertEqual(retained["lease_expires_unix"], None if broken else 15400.)
                if broken:
                    self.assertIsNone(retained["retention_expires_unix"])
                    self.assertNotIn("max_retention_cost_usd", retained)
                    self.assertNotIn("retention_window_seconds_effective", retained)
                    self.assertNotIn("retention_window_seconds", retained)
                    self.assertEqual(retained["guard_stop_at_unix"], 15340.)
                    self.assertEqual(retained["retention_acquired_unix"], 10000.)
                    self.assertEqual(retained["retention_bound_seconds"], 14400.)
                    h.provisioner.release.assert_not_called()
                self.assertEqual(retained["retention_heartbeat"], "failed" if broken else "active")

    def test_failed_handoff_probes_unknown_guard_and_releases_only_when_dead(self):
        for alive, probed, expect_release in ((False, False, True), (True, False, False),
                                              (None, False, True), (None, True, False)):
            with self.subTest(alive=alive, probed=probed), tempfile.TemporaryDirectory() as tmp, \
                    ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack)
                h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
                h.worker.retention_guard_alive = alive
                h.worker.retention_guard_stop_at_unix = None
                h.worker.probe_retention_guard.return_value = probed
                h.worker.retain_live.side_effect = WorkerError("fake handoff failure")
                with self.assertRaises(A20Error):
                    h.run()
                record = json.loads((h.output / "run.json").read_bytes())
                retained = record["retained_node"]
                self.assertEqual(retained["retention_heartbeat"], "failed")
                self.assertEqual(record["cleanup_error"], "A20_LIVE_RETENTION_FAILED")
                if alive is True:
                    h.worker.probe_retention_guard.assert_not_called()
                    h.provisioner.release.assert_not_called()
                    self.assertEqual(retained["guard_stop_at_unix"], 15340.)
                    self.assertEqual(retained["guard_idle_seconds"], 120.)
                else:
                    h.worker.probe_retention_guard.assert_called_once_with(
                        h.budget.guard_plan.return_value)
                    if expect_release:
                        h.provisioner.release.assert_called_once_with("owned")
                        self.assertTrue(record["release_verified"])
                        self.assertEqual(retained["state"], "released")
                        self.assertIsNone(retained["hourly_usd"])
                        self.assertIsNone(retained["guard_stop_at_unix"])
                    else:
                        h.provisioner.release.assert_not_called()
                        self.assertFalse(record["release_verified"])
                        self.assertEqual(retained["guard_stop_at_unix"], 15340.)
                        self.assertEqual(retained["guard_idle_seconds"], 120.)

    def test_extend_retention_failure_probes_old_guard_before_release(self):
        for probed in (False, True):
            with self.subTest(probed=probed), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack)
                h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
                h.budget.extend_retention.side_effect = ValueError("aggregate cannot admit")
                h.worker.retention_guard_alive = None
                h.worker.retention_guard_stop_at_unix = None
                h.worker.probe_retention_guard.return_value = probed
                with self.assertRaises(A20Error):
                    h.run()
                h.budget.extend_retention.assert_called_once_with("acquisition")
                h.worker.retain_live.assert_not_called()
                h.worker.probe_retention_guard.assert_called_once_with(
                    h.budget.guard_plan.return_value)
                record = json.loads((h.output / "run.json").read_bytes())
                retained = record["retained_node"]
                self.assertEqual(retained["retention_heartbeat"], "failed")
                self.assertEqual(record["cleanup_error"], "A20_LIVE_RETENTION_FAILED")
                self.assertIsNone(retained["retention_expires_unix"])
                if probed:
                    h.provisioner.release.assert_not_called()
                    self.assertFalse(record["release_verified"])
                    self.assertEqual(retained["guard_stop_at_unix"], 15340.)
                    self.assertEqual(retained["guard_idle_seconds"], 120.)
                else:
                    h.provisioner.release.assert_called_once_with("owned")
                    self.assertTrue(record["release_verified"])
                    self.assertEqual(retained["state"], "released")
                    self.assertIsNone(retained["hourly_usd"])
                    self.assertIsNone(retained["guard_stop_at_unix"])

    def test_interrupted_retention_probe_releases_before_reraising(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
            h.worker.retention_guard_alive = None
            h.worker.retention_guard_stop_at_unix = None
            h.worker.retain_live.side_effect = WorkerError("fake handoff failure")
            h.worker.probe_retention_guard.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                h.run()
            h.worker.probe_retention_guard.assert_called_once_with(
                h.budget.guard_plan.return_value)
            h.provisioner.release.assert_called_once_with("owned")
            record = json.loads((h.output / "run.json").read_bytes())
            self.assertTrue(record["release_verified"])
            self.assertEqual(record["retained_node"]["state"], "released")
            self.assertIsNone(record["retained_node"]["hourly_usd"])
            self.assertEqual(record["outcome"], "incomplete")
            self.assertEqual(record["cleanup_error"], "A20_LIVE_RETENTION_FAILED")

    def test_default_release_never_extends_or_spawns_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            result = h.run()
            self.assertNotIn("retained_node", result)
            h.budget.extend_retention.assert_not_called()
            h.worker.retain_live.assert_not_called()
            h.provisioner.release.assert_called_once_with("owned")

    def test_retention_survives_unavailable_rate_and_connection(self):
        for failure in (None, "stage"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                h = Harness(Path(tmp).resolve(), stack, failure=failure)
                h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
                h.provisioner.inspect_lease.side_effect = RuntimeError("private provider error")
                h.worker.connection_info = None
                if failure:
                    with self.assertRaises(A20Error):
                        h.run()
                else:
                    h.run()
                record = json.loads((h.output / "run.json").read_bytes())
                self.assertIsNone(record["retained_node"]["hourly_usd"])
                self.assertIsNone(record["retained_node"]["ssh"])
                self.assertNotIn("private provider error", json.dumps(record))
                h.provisioner.release.assert_not_called()

    def test_released_snapshot_does_not_report_a_billing_rate(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
            h.provisioner.inspect_lease.return_value = NS(state="released", hourly_usd=1.09)
            result = h.run()
            self.assertEqual(result["retained_node"]["state"], "released")
            self.assertIsNone(result["retained_node"]["hourly_usd"])
            h.provisioner.release.assert_not_called()

    def test_interrupt_during_retained_snapshot_lookup_persists_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
            h.provisioner.inspect_lease.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                h.run()
            record = json.loads((h.output / "run.json").read_bytes())
            self.assertEqual(record["outcome"], "incomplete")
            self.assertIsNone(record["retained_node"]["hourly_usd"])
            h.worker.__exit__.assert_called_once()
            h.budget.__exit__.assert_called_once()
            h.provisioner.release.assert_not_called()

    def test_retention_does_not_prevent_midrun_enforced_release(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack)
            h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
            # Simulate worker.check invoking the installed budget enforcer during staging.
            def enforce(provisioner):
                h.budget.extend_retention.assert_not_called()
                h.worker.__exit__.assert_not_called()
                provisioner.release("owned")
                provisioner.inspect_lease.return_value = NS(state="released", hourly_usd=1.09)
                # A released journal record can no longer supply a guard plan.
                h.budget.extend_retention.side_effect = ValueError("released lease")
                return ("acquisition",)
            h.budget.enforce_limits.side_effect = enforce
            def stage(*args):
                enforcer = h.worker.set_enforcer.call_args.args[0]
                self.assertEqual(enforcer(), ("acquisition",))
                raise WorkerError("worker supervision or lease deadline failed")
            h.bridge.stage.side_effect = stage
            with self.assertRaisesRegex(A20Error, "A20_INCOMPLETE_RETAIN_PRIVATE_RUN"):
                h.run()
            record = json.loads((h.output / "run.json").read_bytes())
            self.assertEqual(record["outcome"], "incomplete")
            self.assertEqual(record["retained_node"]["state"], "released")
            self.assertIsNone(record["retained_node"]["hourly_usd"])
            self.assertEqual(record["retained_node"]["retention_heartbeat"], "failed")
            h.worker.retain_live.assert_not_called()
            h.provisioner.release.assert_called_once_with("owned")
            h.budget.enforce_limits.assert_called_once_with(h.provisioner)

    def test_no_acquired_device_produces_no_retention_record(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            h = Harness(Path(tmp).resolve(), stack, failure="acquire")
            h.bindings = replace(h.bindings, spec=replace(h.bindings.spec, retain_node=True))
            with self.assertRaises(A20Error):
                h.run()
            record = json.loads((h.output / "run.json").read_bytes())
            self.assertNotIn("retained_node", record)
            h.provisioner.release.assert_not_called()
            h.provisioner.inspect_lease.assert_not_called()

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
                clock = stack.enter_context(patch.object(driver.time, "monotonic", return_value=1))
                if expired:
                    # Preparation consumes the original cap before failing.
                    def prepare(*args, clock=clock, h=h, **kwargs):
                        clock.return_value = kwargs["absolute_deadline"]
                        h.event("prepare")
                    h.worker.prepare.side_effect = prepare
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


class ConnectionInfoTests(unittest.TestCase):
    def test_cached_connection_projection_has_no_credentials_or_io(self):
        worker = object.__new__(SSHWorker)
        worker._host, worker._port = "", 0
        worker.private_key = Path("/never-read-private-key")
        self.assertIsNone(worker.connection_info)
        worker._host, worker._port = "8.8.8.8", 2222
        expected = {"host": "8.8.8.8", "port": 2222, "username": "root"}
        self.assertEqual(worker.connection_info, expected)
        worker.connection_info["host"] = "modified-copy"
        self.assertEqual(worker.connection_info, expected)

    def test_projection_tracks_last_successful_connection_and_clears_on_failure(self):
        from test_runpod import spec
        from test_ssh_worker import Budget, Transport

        worker = object.__new__(SSHWorker)
        worker.spec, worker.budget, worker.transport = spec(), Budget(), Transport()
        worker._host, worker._port = "", 0
        snapshot = LeaseSnapshot("first", "created", "pod_1")
        self.assertTrue(worker._connection(snapshot, 5))
        self.assertEqual(worker.connection_info,
            {"host": "8.8.8.8", "port": 22000, "username": "root"})
        worker.transport.pod["ssh"]["direct"].update(host="1.1.1.1", port=2222)
        self.assertTrue(worker._connection(snapshot, 5))
        self.assertEqual(worker.connection_info,
            {"host": "1.1.1.1", "port": 2222, "username": "root"})
        worker.transport.pod["ssh"]["direct"]["host"] = "127.0.0.1"
        with self.assertRaises(WorkerError):
            worker._connection(snapshot, 5)
        self.assertIsNone(worker.connection_info)
        worker.transport.pod["ssh"]["direct"]["host"] = "8.8.8.8"
        self.assertTrue(worker._connection(snapshot, 5))
        worker.transport.pod["ssh"] = None
        self.assertFalse(worker._connection(snapshot, 5))
        self.assertIsNone(worker.connection_info)


class CLITests(unittest.TestCase):
    def test_command_projects_retention_only_when_present(self):
        import hashlib

        from cineforge import a20_cli

        for retain in (False, True):
            with self.subTest(retain=retain), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp).resolve()
                h = Harness(root, stack)
                controller = root / "controller.py"
                controller.write_bytes(b"# synthetic controller")
                args = NS(inputs=root / "inputs.json", allow_live=True,
                    controller_root=root, controller=controller,
                    controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
                    output_dir=h.output)
                args.inputs.write_text(json.dumps({k: str(v) for k, v in h.inputs.paths.items()}))
                report = {"outcome": "pending_human_review", "release_verified": not retain,
                          "human_review": "pending_human_review", "export_authority": "none"}
                if retain:
                    report["retained_node"] = {"pod_id": "owned"}
                clock = stack.enter_context(patch.object(a20_cli.time, "monotonic", return_value=100))
                def load_controller(*args, clock=clock, h=h, **kwargs):
                    clock.return_value = 400  # Local review/import consumes the original cap.
                    return {"build_bindings": lambda creative, h=h: h.bindings}
                stack.enter_context(patch.object(a20_cli.runpy, "run_path", side_effect=load_controller))
                run = stack.enter_context(patch.object(a20_cli, "run_a20", return_value=report))
                self.assertEqual(a20_cli.command(args), report)
                self.assertEqual(run.call_args.kwargs["absolute_deadline"], 7300)

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
