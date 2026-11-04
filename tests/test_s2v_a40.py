"""Explicit Ampere route with synthetic protocol/GPU facts, never native model proof."""

import hashlib
import json
import shlex
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_comfy_runtime as comfy_fixture
import test_native_remote as native_fixture
import test_s2v as s2v_fixture
import test_s2v_l40s as l40s_fixture
import test_s2v_stage as stage_fixture
from test_native_speech import selection as speech_selection
from test_speech import EVIDENCE

from cineforge import s2v_profile as profile
from cineforge import worker_bootstrap as bootstrap
from cineforge.comfy_runtime_plan import comfy_plan, comfy_subject, comfy_worker
from cineforge.comfy_runtime_worker import _require_profile
from cineforge.native_remote_package import native_package
from cineforge.native_speech_integrity import NativeHardware, NativeSpeechError
from cineforge.s2v import S2VError, select_s2v
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext, prepare_service_journal
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source
from cineforge.worker_profiles import S2V_A40_WORKER, S2V_L40S_WORKER, S2V_WORKER
from cineforge.workflows import canonical


def a40_runtime():
    return dict(l40s_fixture.l40s_runtime(), gpu_raw_name=profile.A40_GPU_ID, capability=[8, 6])


class A40ProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cf-a40-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def service(self, gpu_id=profile.A40_GPU_ID):
        return S2VServiceContext(
            S2VBudgetApproval.for_scope(
                str(self.root / "budget"), "EU-SE-1", EVIDENCE, gpu_id=gpu_id
            )
        )

    def test_closed_profile_keeps_l40s_identity_and_exact_assets(self):
        self.assertEqual(
            profile.recipe_sha256(profile.L40S_GPU_ID),
            "73fae7af8a6e2b059ef8c9987adb5f01a868c6f4740524f3cef067648bc7a4d9",  # pragma: allowlist secret - public baseline recipe
        )
        self.assertEqual(
            hashlib.sha256(
                canonical(profile.service_declarations(profile.L40S_GPU_ID))
            ).hexdigest(),
            "78720de28d64175f31722b8bfb9f040a9b71acd2493e401e40289d784d7fef69",  # pragma: allowlist secret - public baseline declarations
        )
        service = self.service()
        self.assertEqual(service.worker_profile["id"], profile.A40_PROFILE_ID)
        self.assertEqual(service.approval.gpu_id, "NVIDIA A40")
        self.assertEqual(service.approval.min_ram_per_gpu_gb, 64)
        self.assertEqual(S2V_A40_WORKER.files, S2V_WORKER.files)
        self.assertEqual(S2V_A40_WORKER.minimum_memory_mib, 30720)
        self.assertEqual(S2V_A40_WORKER.reference_runtime.maximum_memory_mib, 49152)
        self.assertEqual(S2V_A40_WORKER.reference_runtime.capability, (8, 6))
        self.assertEqual(S2V_A40_WORKER.reference_runtime.architecture, "sm_86")
        self.assertEqual(service.model, self.service(profile.GPU_ID).model)
        self.assertNotEqual(
            service.approval.subject_sha256,
            self.service(profile.L40S_GPU_ID).approval.subject_sha256,
        )
        for gpu in ("A40", "NVIDIA A100", "NVIDIA RTX A6000", None):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                self.service(gpu)
        with self.assertRaises(ValueError):
            replace(service.approval, gpu_id=profile.L40S_GPU_ID)
        prepare_service_journal(service)
        before = (self.root / "budget/budget.json").read_bytes()
        with self.assertRaises(ValueError):
            prepare_service_journal(self.service(profile.L40S_GPU_ID))
        self.assertEqual((self.root / "budget/budget.json").read_bytes(), before)

    def test_exact_arch_ram_precision_abi_and_physical_limits(self):
        value = a40_runtime()
        bootstrap.validate_s2v_runtime(value, profile_id=profile.A40_PROFILE_ID)
        for change in (
            dict(arch_list=["sm_89"]),
            dict(arch_list=["compute_86"]),
            dict(capability=[8, 9]),
            dict(capability=[8, True]),
            dict(gpu_raw_name=profile.L40S_GPU_ID),
            dict(gpu_memory_mib=30719),
            dict(gpu_memory_mib=49153),
            dict(python_version="3.12.3"),
            dict(memory_limit_bytes=63_999_999_999),
            dict(available_memory_bytes=32 * 1024**3 - 1),
            dict(cuda_version="12.7"),
            dict(compute_precision="fp16"),
            dict(cuda_kernel_ok=False),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bootstrap.validate_s2v_runtime(
                    dict(value, **change), profile_id=profile.A40_PROFILE_ID
                )
        for profile_id in (profile.PROFILE_ID, profile.L40S_PROFILE_ID):
            with self.subTest(profile_id=profile_id), self.assertRaises(ValueError):
                bootstrap.validate_s2v_runtime(value, profile_id=profile_id)
        admitted = dict(
            value, admission_status="hardware_only_unverified_runtime", gpu_memory_mib=46068
        )
        admitted = {
            k: v
            for k, v in admitted.items()
            if k
            in bootstrap.M4_RESOURCE_FIELDS
            | {
                "admission_status",
                "python_version",
                "gpu_raw_name",
                "gpu_memory_mib",
                "nvml_total_mib",
                "nvml_reserved_mib",
                "capability",
                "device_count",
                "driver_version",
            }
        }
        bootstrap.validate_s2v_admission(admitted, profile_id=profile.A40_PROFILE_ID)
        with self.assertRaises(ValueError):
            bootstrap.validate_s2v_admission(
                dict(admitted, memory_limit_bytes=50_000_000_000), profile_id=profile.A40_PROFILE_ID
            )

    def test_speech_selection_pair_and_fixed_ssh_handoffs_are_profile_bound(self):
        f = stage_fixture.PairTransferTests()
        f.gpu_id = profile.A40_GPU_ID
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.assertEqual(f.selection.declarations["profile_id"], profile.A40_PROFILE_ID)
        f.driver._host, f.driver._port = "8.8.8.8", 22000
        cap = time.monotonic() + 20
        with patch("cineforge.ssh_worker.run_process", return_value=b"{}") as process:
            SSHWorker._request(
                f.driver, dict(operation="arm"), 5, supervise=False, absolute_deadline=cap
            )
            self.assertEqual(process.call_args.kwargs["absolute_deadline"], cap)
            self.assertEqual(
                json.loads(process.call_args.kwargs["payload"])["profile"],
                f.gate.service.worker_profile,
            )
            self.assertEqual(
                shlex.split(process.call_args.args[0][-1])[:5],
                ["/usr/bin/python3.11", "-I", "-B", "-S", "-c"],
            )
            self.assertLess(len(process.call_args.args[0][-1].encode()), 131072)
        staged = f.driver.stage_s2v(f.selection)
        f.selection.bind_worker(s2v_fixture.worker(f.gate), staged)
        f.driver.runtime_evidence = l40s_fixture.l40s_runtime()
        before = len(f.requests)
        with self.assertRaises(WorkerError):
            f.driver.stage_s2v(f.selection)
        self.assertEqual(len(f.requests), before)
        wrong = replace(f.gate, service=self.service(profile.L40S_GPU_ID))
        with self.assertRaises(S2VError):
            select_s2v(wrong)

    def test_native_receipt_uses_actual_selected_a40_and_refuses_mixed_hardware(self):
        f = native_fixture.RemoteFixture()
        f.gpu_id = profile.A40_GPU_ID
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.prepare()
        wave = f.bridge.synthesize(f.request, f.reference)
        receipt = f.bridge.receipt_for(f.request, wave)
        self.assertEqual(receipt["hardware"]["gpu_name"], profile.A40_GPU_ID)
        self.assertEqual(receipt["hardware"]["capability"], [8, 6])
        self.assertTrue(all(d == f.bridge._deadline for d in f.deadlines))
        wrong = speech_selection(
            f.root, hardware=NativeHardware("synthetic-worker", profile.L40S_GPU_ID, 46000, (8, 9))
        )
        f.bridge._selection = None
        before = len(f.requests)
        with self.assertRaises(NativeSpeechError):
            f.bridge.select(wrong)
        self.assertEqual(len(f.requests), before)

    def test_comfy_exact_plan_subject_and_mixed_profile_refusals(self):
        plain = comfy_fixture.plan()
        a40 = dict(plain, profile_id=profile.A40_PROFILE_ID)
        self.assertEqual(comfy_worker(comfy_plan(a40)), S2V_A40_WORKER)
        self.assertNotEqual(comfy_subject(plain), comfy_subject(a40))
        self.assertNotEqual(
            comfy_subject(dict(plain, profile_id=profile.L40S_PROFILE_ID)), comfy_subject(a40)
        )
        request = dict(
            profile=dict(id=profile.A40_PROFILE_ID),
            selection=dict(
                profile_id=profile.A40_PROFILE_ID, hardware=dict(gpu=profile.A40_GPU_ID)
            ),
        )
        _require_profile(request, a40)
        for variant in (plain, dict(plain, profile_id=profile.L40S_PROFILE_ID)):
            with self.assertRaises(ValueError):
                _require_profile(request, variant)
        with self.assertRaises(ValueError):
            comfy_plan(dict(plain, profile_id="auto"))
        self.assertEqual(comfy_worker(comfy_plan(plain)), S2V_WORKER)
        self.assertEqual(
            comfy_worker(comfy_plan(dict(plain, profile_id=profile.L40S_PROFILE_ID))),
            S2V_L40S_WORKER,
        )

    def test_actual_quoted_command_and_native_package_are_bounded(self):
        source = worker_source()
        package, pins = native_package()
        self.assertLessEqual(len(source), MAX_SOURCE_BYTES)
        self.assertLess(
            len(("/usr/bin/python3.11 -I -B -S -c " + shlex.quote(source.decode())).encode()),
            131072,
        )
        self.assertLessEqual(len(package), 2 * 1024**2)
        self.assertTrue(all(size <= 256 * 1024 for _, size, _ in pins))

    def test_a40_observation_requires_actual_bf16_math_kernel(self):
        l40s_fixture.L40SProfileTests.exercise_math_observation(
            self, profile.A40_GPU_ID, profile.A40_PROFILE_ID, (8, 6)
        )
