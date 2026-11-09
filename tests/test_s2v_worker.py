"""Offline service/guard/profile conformance; no resolver, network or GPU execution."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_conditioning import runtime_fixture
from test_s2v import context
from test_speech import EVIDENCE

from cineforge import s2v_profile as profile
from cineforge import worker_bootstrap as bootstrap
from cineforge.cloud_budget import CloudBudgetError, CloudBudgetSession
from cineforge.cloud_journal import CloudJournalError
from cineforge.s2v import select_s2v
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext, prepare_service_journal
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source
from cineforge.worker_profiles import I2V_WORKER, S2V_WORKER
from cineforge.worker_transport_source import worker_command_source
from cineforge.workflows import canonical


class S2VWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cf-s2v-worker-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.service = S2VServiceContext(S2VBudgetApproval.for_scope(
            str(self.root / "budget"), "EU-RO-1", EVIDENCE))

    def test_offline_journal_initialization_preserves_existing_scope_and_records(self):
        unprepared = S2VServiceContext(S2VBudgetApproval.for_scope(
            str(self.root / "unprepared"), "EU-RO-1", EVIDENCE))
        with self.assertRaises(CloudBudgetError):
            with CloudBudgetSession(unprepared.approval.journal_directory,
                    unprepared.approval, None, guard_deployment_configured=True):
                self.fail("unprepared journal must not admit")
        with self.assertRaises(CloudJournalError):
            prepare_service_journal(unprepared)  # Never reset an interrupted/ambiguous lock-only path.
        prepare_service_journal(self.service)
        path = Path(self.service.approval.journal_directory) / "budget.json"
        original = json.loads(path.read_text())
        prepare_service_journal(self.service)
        self.assertEqual(json.loads(path.read_text()), original)
        with CloudBudgetSession(path.parent, self.service.approval, None,
                                guard_deployment_configured=True) as session:
            self.assertEqual(session.intents, ())
        other = S2VServiceContext(S2VBudgetApproval.for_scope(str(path.parent), "EU-CZ-1", EVIDENCE))
        with self.assertRaises(CloudBudgetError):
            prepare_service_journal(other)
        with self.assertRaises(CloudBudgetError):
            CloudBudgetSession(self.root / "different", self.service.approval, None)

    def test_exact_s2v_profile_audio_assets_do_not_relax_legacy_profiles(self):
        data = self.service.worker_profile
        self.assertEqual(bootstrap.checked_profile(data), data)
        self.assertEqual(bootstrap._configuration(data), S2V_WORKER)
        self.assertEqual(profile.FILES, S2V_WORKER.files)
        self.assertEqual(len(I2V_WORKER.files), 3)
        for malformed in (dict(data, reference={}), {"id": data["id"], "scope_sha256": True},
                          {"id": "CANARY", "scope_sha256": "a" * 64}):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.checked_profile(malformed)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.stage_reference({"profile": data}, self.root)
        audio = next(item for item in profile.model_files() if "audio_encoders/" in item["path"])
        with patch.object(bootstrap.urllib.request, "build_opener") as opener:
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.download(self.root, audio, time.monotonic() + 5)
            opener.assert_not_called()
        marker = bootstrap.BootstrapError("synthetic stop before network")
        with patch.object(bootstrap.urllib.request, "build_opener", side_effect=marker) as opener:
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.download(self.root, audio, time.monotonic() + 5, profile=data)
            self.assertEqual(opener.call_count, 1)

    def test_legacy_m5_setup_refuses_before_any_ambient_import_or_effect(self):
        gate, _ = context(self.root)
        request = {"profile": gate.service.worker_profile,
            "plan": {"session_id": "fixture"}, "selection": select_s2v(gate).declarations,
            "image_digest": profile.IMAGE.split("@", 1)[1]}
        with patch.object(bootstrap, "worker_root", return_value=self.root), \
                patch.object(bootstrap, "runtime_probe", side_effect=bootstrap.BootstrapError(
                    "synthetic incompatible GPU")) as probe, \
                patch.object(bootstrap, "bounded_process") as process, \
                patch.object(bootstrap, "download") as download:
            with self.assertRaises(bootstrap.SetupFailure):
                bootstrap.setup(request, time.monotonic() + 30)
            self.assertFalse((self.root / "s2v-setup-consumed.json").exists())
            self.assertEqual(probe.call_count, 0)
            with self.assertRaises(bootstrap.SetupFailure):
                bootstrap.setup(request, time.monotonic() + 30)
            self.assertEqual(probe.call_count, 0)
            process.assert_not_called()
            download.assert_not_called()

    def test_bf16_hardware_kernel_is_required_and_runtime_fields_remain_closed(self):
        runtime = dict(runtime_fixture(), compute_precision="bf16")
        bootstrap.validate_s2v_runtime(runtime)
        for change in (dict(compute_precision="fp16"), dict(cuda_kernel_ok=False),
                       dict(capability=[8, 9]), dict(gpu_raw_name="CANARY")):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.validate_s2v_runtime(dict(runtime, **change))
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.validate_m4_runtime(runtime)
        gpu = {key: value for key, value in runtime.items()
               if key not in bootstrap.M4_RESOURCE_FIELDS | {"nvml_total_mib", "nvml_reserved_mib",
                                                            "driver_version"}}
        commands = []
        def process(command, **limits):
            commands.append(command)
            return (canonical(gpu) if len(commands) == 1 else
                    f'{runtime["nvml_total_mib"]},{runtime["nvml_reserved_mib"]},{runtime["driver_version"]}'.encode())
        resources = {key: runtime[key] for key in bootstrap.M4_RESOURCE_FIELDS}
        with patch.object(bootstrap, "bounded_process", side_effect=process), \
                patch.object(bootstrap, "resource_evidence", return_value=resources):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.runtime_probe(10, profile=self.service.worker_profile)
        self.assertEqual(commands, [])  # M5 real kernel proof belongs to its closed launcher.

    def test_installed_stdlib_worker_bundle_handles_ping_without_package_imports(self):
        source = worker_source()
        self.assertLessEqual(len(source), MAX_SOURCE_BYTES)
        result = subprocess.run([sys.executable, "-I", "-c", worker_command_source(source)],
            input=b'{"operation":"ping","allow_live":true,"timeout_seconds":1}',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.root, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"status": "ssh_ready"})
        self.assertIn(b"paired_transfer", source)

    def test_m5_audio_failure_description_is_closed_without_widening_m4_assets(self):
        data = {"schema": "m5-setup-failure-v1", "stage": "weights",
                "asset": "audio_encoder", "reason": "integrity_failed"}
        self.assertEqual(bootstrap.checked_setup_failure(data), data)
        self.assertIsNone(bootstrap.checked_setup_failure(dict(data, schema="m4-setup-failure-v1")))
        self.assertIsNone(bootstrap.checked_setup_failure(dict(data, unknown_field="CANARY")))


if __name__ == "__main__":
    unittest.main()
