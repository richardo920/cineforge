"""Profile data cannot grant authority; fixed bundles run without an installed package."""

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import test_cloud_budget as budget_fixtures

from cineforge import conditioning_profile, worker_bootstrap
from cineforge.cloud_budget import BudgetApproval, CloudBudgetError
from cineforge.cloud_smoke import CloudSmokeError, approved_plan
from cineforge.comfy_policy import ApprovedComfySelection
from cineforge.selection import selection_subject
from cineforge.worker_bundle import MAX_SOURCE_BYTES, WorkerBundleError, worker_source
from cineforge.worker_profiles import I2V_WORKER, T2V_WORKER
from cineforge.worker_transport_source import worker_command_source
from cineforge.workflows import canonical

ROOT = Path(__file__).resolve().parents[1]


def reference_metadata():
    return {"id": I2V_WORKER.profile_id,
            "reference": {"sha256": "a" * 64, "size_bytes": 100,
                          "basename": "ref-" + "a" * 64 + ".png"}}


class ProfileTests(unittest.TestCase):
    def test_approved_configuration_bytes_and_subjects_are_unchanged(self):
        files = {
            "config/cloud/runpod-pod-proposal.json": "4ddde4e4b4a751a92146eb290e45ee0de1e2649f3be8e4468fc69edc8224aec9",  # pragma: allowlist secret - public approval declaration pin
            "config/comfy/wan21-t2v-1.3b.json": "f028fc75ba694605e327eb30ba5d411a8342e8841698f51cf9a6e399a0b1f60e",  # pragma: allowlist secret - public selection pin
            "config/comfy/wan21-t2v-1.3b-api.json": "8baab77ae0a01374dfad997d753eba2d7bad84945088dc1f90fdf7f9cf04d529",  # pragma: allowlist secret - public workflow pin
            "config/comfy/wan22-ti2v5b-proposal.json": "52fd25468cf8d3165c50977675c29e8c847768a64536679a30125f144b345913",  # pragma: allowlist secret - public proposal pin
        }
        for name, expected in files.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        selection = budget_fixtures.selection()
        self.assertEqual(selection.declarations["approval"]["model_workflow"][
            "approved_subject_sha256"],
            "1aaa7e024932fe008bf0c13304ce80be66991d03a2b7d94aa4bd0f37987b15d0")  # pragma: allowlist secret - public approval subject pin
        self.assertEqual(hashlib.sha256(canonical(selection.declarations)).hexdigest(),
            "e064376dcdde05e615d3864b86ef7dc1b272cf06e49fed02b00a2a9745322b9a")  # pragma: allowlist secret - public declaration pin
        values = [
            (conditioning_profile.declarations("a" * 64),
             "392e25a8b05c415bfd49efd453d0e20495c6de94eb8ae02b6d19026e44452eb6"),  # pragma: allowlist secret - public synthetic-reference declaration pin
            (conditioning_profile.graph("a" * 64),
             "9db81706d058a33226bc1bcf61fc1593cbfb74c9b4883baea7cefd6c15604f56"),  # pragma: allowlist secret - public synthetic-reference graph pin
            (conditioning_profile.model_files(),
             "c2a3cddaf8dc93bc83a061c44d879eef5f5b20e31fb9b192ad44cff2fe4e7d81"),  # pragma: allowlist secret - public manifest pin
        ]
        for value, expected in values:
            self.assertEqual(hashlib.sha256(canonical(value)).hexdigest(), expected)

    def test_two_immutable_capabilities_preserve_legacy_and_reference_limits(self):
        with self.assertRaises(AttributeError):
            T2V_WORKER.gpu = "arbitrary"
        with self.assertRaises(TypeError):
            I2V_WORKER.files[0] = ("arbitrary", 1, "f" * 64)
        self.assertEqual(T2V_WORKER.profile_id, "")
        self.assertEqual(T2V_WORKER.files, ())
        self.assertIsNone(T2V_WORKER.reference_runtime)
        self.assertEqual(T2V_WORKER.maximum_asset_bytes, 8 * 1024**3)
        self.assertEqual((T2V_WORKER.gpu, T2V_WORKER.minimum_memory_mib),
                         ("NVIDIA GeForce RTX 4090", 24082))
        self.assertEqual(I2V_WORKER.maximum_asset_bytes, 10_000_000_000)
        self.assertEqual(I2V_WORKER.files, conditioning_profile.FILES)
        self.assertEqual(sum(row[1] for row in I2V_WORKER.files), 18144966705)
        self.assertEqual((I2V_WORKER.gpu, I2V_WORKER.minimum_memory_mib),
                         ("NVIDIA GeForce RTX 5090", 30720))
        self.assertIs(worker_bootstrap._configuration(None), T2V_WORKER)
        self.assertIs(worker_bootstrap._configuration(reference_metadata()), I2V_WORKER)

    def test_json_cannot_supply_profile_overrides_or_new_adapter_capabilities(self):
        valid = reference_metadata()
        rejected = [
            {**valid, "gpu": T2V_WORKER.gpu}, {**valid, "files": []},
            {**valid, "approved": True}, {**valid, "module": "private-code-canary"},
            {**valid, "id": "m5-speech"}, {**valid, "id": "serverless"},
            {**valid, "id": T2V_WORKER.profile_id}, I2V_WORKER._replace(gpu="arbitrary"),
        ]
        with patch.object(worker_bootstrap, "bounded_process") as process, \
                patch.object(worker_bootstrap.urllib.request, "build_opener") as network:
            for value in rejected:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(worker_bootstrap.BootstrapError, "unknown worker"):
                        worker_bootstrap.runtime_probe(1, profile=value)
                    with self.assertRaisesRegex(worker_bootstrap.BootstrapError, "unknown worker"):
                        worker_bootstrap.download(Path("/unused"), {}, time.monotonic()+1,
                                                  profile=value)
            process.assert_not_called()
            network.assert_not_called()

    def test_rehashed_selection_is_still_rejected_by_independent_approval_boundary(self):
        original = budget_fixtures.selection()
        value = original.declarations
        value["hardware"]["gpu"] = I2V_WORKER.gpu
        value["hardware"]["vram_gib"] = 32
        claimed = hashlib.sha256(canonical(selection_subject(value))).hexdigest()
        value["approval"]["model_workflow"]["approved_subject_sha256"] = claimed
        # A self-consistent receipt declaration is not independently trusted approval.
        edited = ApprovedComfySelection(canonical(value).decode(), original.graph_json,
                                         original.license_bytes)
        with self.assertRaises(CloudSmokeError):
            approved_plan(edited)
        proposal, expected = budget_fixtures.approval_fixture()
        with self.assertRaises(CloudBudgetError):
            BudgetApproval.from_proposal(proposal, expected_subject_sha256=expected,
                expected_model_workflow_sha256=budget_fixtures.model_subject(),
                model=original.model, selection=edited)
        self.assertNotEqual(claimed, budget_fixtures.model_subject())


class BundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_bounded_and_contains_no_package_import_fallback(self):
        first = worker_source()
        self.assertEqual(first, worker_source())
        self.assertLessEqual(len(first), MAX_SOURCE_BYTES)
        self.assertNotIn(b"from cineforge", first)
        self.assertEqual(hashlib.sha256(first).digest(), hashlib.sha256(worker_source()).digest())

    def test_isolated_standalone_success_failure_and_unknown_profile_without_repo(self):
        with tempfile.TemporaryDirectory(prefix="cineforge-profile-bundle-") as path:
            for request, code, expected in [
                ({"operation": "ping", "allow_live": True, "timeout_seconds": 1},
                 0, b'{"status":"ssh_ready"}\n'),
                ({"operation": "private-payload-canary", "allow_live": False,
                  "timeout_seconds": 1}, 2, b'{"status":"worker_operation_failed"}\n'),
                ({"operation": "arm", "allow_live": True, "timeout_seconds": 1,
                  "profile": {"id": "private-module-canary"}},
                 2, b'{"status":"worker_operation_failed"}\n'),
            ]:
                result = subprocess.run([sys.executable, "-I", "-c", worker_command_source(worker_source())],
                    input=canonical(request), capture_output=True, cwd=path, timeout=3)
                self.assertEqual((result.returncode, result.stdout, result.stderr),
                                 (code, expected, b""))

    def test_standalone_and_package_resolve_identical_two_profiles(self):
        source = worker_source().rsplit(b'\nif __name__ == "__main__":', 1)[0]
        source += (b'\nprint(encode([_configuration(None)._asdict(), _configuration('
                   + repr(reference_metadata()).encode()
                   + b')._asdict()]).decode())\n')
        with tempfile.TemporaryDirectory(prefix="cineforge-profile-data-") as path:
            fixture_source = Path(path) / "bundle-fixture.py"
            fixture_source.write_bytes(source)
            result = subprocess.run([sys.executable, "-I", str(fixture_source)],
                                    capture_output=True, cwd=path, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = json.loads(canonical([T2V_WORKER._asdict(), I2V_WORKER._asdict()]))
        self.assertEqual(json.loads(result.stdout), expected)

    def test_missing_incomplete_oversized_or_nonstdlib_source_fails_closed(self):
        root = ROOT / "src/cineforge"
        profile = (root / "worker_profiles.py").read_bytes()
        bootstrap = (root / "worker_bootstrap.py").read_bytes()
        for parts in [
            [b"", bootstrap], [profile[:-10], bootstrap], [profile, bootstrap[:-20]],
            [b"\n# END_REVIEWED_WORKER_PROFILES\n", bootstrap],
            [profile.replace(b"I2V_WORKER =", b"OTHER_WORKER ="), bootstrap],
            [profile, bootstrap.replace(b"from cineforge.worker_profiles import", b"from other import")],
            [profile, bootstrap.replace(b"import base64", b"import private_module_canary")],
            [profile, b"#" * MAX_SOURCE_BYTES + bootstrap],
        ]:
            with self.subTest(lengths=[len(p) for p in parts]):
                fixtures = {"worker_profiles.py": parts[0], "worker_bootstrap.py": parts[1],
                            **{name: (root / name).read_bytes() for name in
                               ("audio_pcm.py", "paired_media.py", "s2v_stage.py", "native_speech_pins.py",
                                "native_driver.py", "native_remote_plan.py", "native_remote_bootstrap.py")}}
                with patch("cineforge.worker_bundle.read_private_input",
                           side_effect=lambda path, **kw: fixtures[path.name]):
                    with self.assertRaisesRegex(WorkerBundleError, "fixed reviewed worker"):
                        worker_source()
        with patch("cineforge.worker_bundle.read_private_input", side_effect=OSError("private-path")):
            with self.assertRaisesRegex(WorkerBundleError, "fixed reviewed worker"):
                worker_source()


if __name__ == "__main__":
    unittest.main()
