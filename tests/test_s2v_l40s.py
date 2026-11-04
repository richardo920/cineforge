"""Closed L40S source/route/resource fixtures only; no model, network or GPU execution."""

import hashlib
import json
import platform
import shlex
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_comfy_runtime as comfy_fixture
import test_native_remote as remote_fixture
import test_s2v as s2v_fixture
import test_s2v_stage as stage_fixture
from test_conditioning import runtime_fixture
from test_native_speech import selection as speech_selection
from test_speech import EVIDENCE

from cineforge import native_remote_bootstrap as native_bootstrap
from cineforge import s2v_profile as profile
from cineforge import worker_bootstrap as bootstrap
from cineforge.cloud_budget import CloudBudgetError
from cineforge.comfy_runtime import CheckedComfyRuntime
from cineforge.comfy_runtime_plan import comfy_plan, comfy_subject, comfy_worker
from cineforge.comfy_runtime_worker import _require_profile
from cineforge.native_speech_integrity import NativeHardware, NativeSpeechError
from cineforge.s2v import S2VError, select_s2v
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext, prepare_service_journal
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source
from cineforge.worker_profiles import I2V_WORKER, S2V_L40S_WORKER, S2V_WORKER, s2v_worker
from cineforge.workflows import canonical


def l40s_runtime():
    return dict(
        runtime_fixture(),
        python_version="3.11.14",
        gpu_raw_name=profile.L40S_GPU_ID,
        gpu_memory_mib=46000,
        nvml_total_mib=46068,
        nvml_reserved_mib=68,
        capability=[8, 9],
        arch_list=["sm_86"],
        compute_precision="bf16",
    )


def admission():
    runtime = l40s_runtime()
    return dict(
        admission_status="hardware_only_unverified_runtime",
        python_version="3.11.14",
        gpu_raw_name=profile.L40S_GPU_ID,
        gpu_memory_mib=46068,
        nvml_total_mib=46068,
        nvml_reserved_mib=68,
        capability=[8, 9],
        device_count=1,
        driver_version="580.95.05",
        **{key: runtime[key] for key in bootstrap.M4_RESOURCE_FIELDS},
    )


class L40SProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cf-l40s-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def service(self, gpu=profile.L40S_GPU_ID):
        return S2VServiceContext(
            S2VBudgetApproval.for_scope(str(self.root / "budget"), "US-MO-1", EVIDENCE, gpu_id=gpu)
        )

    def test_default_bytes_closed_variant_and_same_assets_graph_model(self):
        # Public pre-change source-declaration digests, not credentials.
        self.assertEqual(
            profile.recipe_sha256(),
            "69be64e25b56bfe43a46e74664e4c47540f5bc47b077c54a33b518f0b21ba4a8",  # pragma: allowlist secret - public baseline recipe hash
        )
        self.assertEqual(
            hashlib.sha256(canonical(profile.service_declarations())).hexdigest(),
            "789cadad39efc0a6abe365a8956d42771c555f50d8d86d4638b87e000f4f28e2",  # pragma: allowlist secret - public baseline declaration hash
        )
        old, new = self.service(profile.GPU_ID), self.service()
        self.assertEqual(old.worker_profile["id"], profile.PROFILE_ID)
        self.assertEqual(new.worker_profile["id"], profile.L40S_PROFILE_ID)
        self.assertNotEqual(old.approval.subject_sha256, new.approval.subject_sha256)
        self.assertNotEqual(old.approval.model_workflow_sha256, new.approval.model_workflow_sha256)
        self.assertEqual(old.model, new.model)
        self.assertEqual(old.declarations["model"], new.declarations["model"])
        self.assertEqual(old.declarations["runner"], new.declarations["runner"])
        self.assertEqual(S2V_L40S_WORKER.files, S2V_WORKER.files)
        self.assertEqual(S2V_L40S_WORKER.minimum_memory_mib, 30720)
        self.assertEqual(S2V_L40S_WORKER.reference_runtime.capability, (8, 9))
        self.assertEqual(S2V_L40S_WORKER.reference_runtime.maximum_memory_mib, 49152)
        self.assertEqual(new.approval.min_ram_per_gpu_gb, 64)
        self.assertEqual(I2V_WORKER.reference_runtime.capability, (12, 0))
        for gpu in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5080", "NVIDIA L40", "", None):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                self.service(gpu)
        with self.assertRaises(ValueError):
            replace(old.approval, gpu_id=profile.L40S_GPU_ID)
        with self.assertRaises(ValueError):
            s2v_worker("auto-from-reported-name")

    def test_journal_and_materialized_speech_selection_bind_exact_profile(self):
        old = self.service(profile.GPU_ID)
        prepare_service_journal(old)
        before = (self.root / "budget/budget.json").read_bytes()
        with self.assertRaises(CloudBudgetError):
            prepare_service_journal(self.service())
        self.assertEqual((self.root / "budget/budget.json").read_bytes(), before)
        gate, _ = s2v_fixture.context(self.root, gpu_id=profile.L40S_GPU_ID)
        selected = select_s2v(gate)
        self.assertEqual(selected.declarations["hardware"]["gpu"], profile.L40S_GPU_ID)
        self.assertEqual(selected.declarations["profile_id"], profile.L40S_PROFILE_ID)
        self.assertEqual(gate.subject["recipe_sha256"], gate.service.approval.model_workflow_sha256)
        self.assertEqual(gate.subject["profile_id"], profile.L40S_PROFILE_ID)
        wrong = replace(gate, service=old)
        with self.assertRaises(S2VError):
            select_s2v(wrong)

    def test_hardware_and_runtime_do_not_infer_profile_from_observed_gpu(self):
        value = admission()
        bootstrap.validate_s2v_admission(value, profile_id=profile.L40S_PROFILE_ID)
        with self.assertRaises(ValueError):
            bootstrap.validate_s2v_admission(value)
        for patch_value in (
            dict(gpu_raw_name=profile.GPU_ID),
            dict(capability=[12, 0]),
            dict(capability=[8, True]),
            dict(gpu_memory_mib=30719, nvml_total_mib=30719),
            dict(gpu_memory_mib=49153, nvml_total_mib=49153),
            dict(device_count=2),
            dict(python_version="3.12.3"),
            dict(memory_limit_bytes=63_999_999_999),
            dict(available_memory_bytes=32 * 1024**3 - 1),
            dict(nvml_reserved_mib=46068),
        ):
            with self.subTest(change=patch_value), self.assertRaises(ValueError):
                bootstrap.validate_s2v_admission(
                    dict(value, **patch_value), profile_id=profile.L40S_PROFILE_ID
                )
        runtime = l40s_runtime()
        for arch in ("sm_86", "sm_89"):
            bootstrap.validate_s2v_runtime(
                dict(runtime, arch_list=[arch]), profile_id=profile.L40S_PROFILE_ID
            )
        for change in (
            dict(arch_list=["sm_120"]),
            dict(capability=[12, 0]),
            dict(gpu_raw_name=profile.GPU_ID),
            dict(cuda_version="12.7"),
            dict(compute_precision="fp16"),
            dict(cuda_kernel_ok=False),
            dict(python_version="3.12.3"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bootstrap.validate_s2v_runtime(
                    dict(runtime, **change), profile_id=profile.L40S_PROFILE_ID
                )
        with self.assertRaises(ValueError):
            bootstrap.validate_s2v_runtime(runtime)
        bootstrap.validate_s2v_runtime(dict(runtime_fixture(), compute_precision="bf16"))

    def test_m5_profile_whitelist_and_fixed_ssh_forward_selected_id(self):
        fixture = stage_fixture.PairTransferTests()
        fixture.gpu_id = profile.L40S_GPU_ID
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        worker = fixture.driver
        worker._host, worker._port = "8.8.8.8", 22000
        for operation in (
            "ping",
            "arm",
            "heartbeat",
            "native_preflight",
            "native_invoke",
            "native_comfy_setup",
            "s2v_pair_begin",
            "s2v_pair_chunk",
            "s2v_pair_finish",
        ):
            with (
                self.subTest(operation=operation),
                patch("cineforge.ssh_worker.run_process", return_value=b"{}") as process,
            ):
                SSHWorker._request(worker, dict(operation=operation), 5, supervise=False)
                self.assertEqual(
                    json.loads(process.call_args.kwargs["payload"])["profile"],
                    fixture.gate.service.worker_profile,
                )
                argv = shlex.split(process.call_args.args[0][-1])
                self.assertEqual(argv[:5], ["/usr/bin/python3.11", "-I", "-B", "-S", "-c"])
                self.assertLess(len(process.call_args.args[0][-1].encode()), 131072)
        self.assertEqual(bootstrap._configuration(worker._profile), S2V_L40S_WORKER)
        for invalid in (
            dict(id="auto", scope_sha256="a" * 64),
            dict(worker._profile, gpu=profile.L40S_GPU_ID),
        ):
            with self.assertRaises(ValueError):
                bootstrap.checked_profile(invalid)
        staged = worker.stage_s2v(fixture.selection)
        verification = s2v_fixture.worker(fixture.gate)
        fixture.selection.bind_worker(verification, staged)
        worker.runtime_evidence = dict(runtime_fixture(), compute_precision="bf16")
        before = len(fixture.requests)
        with self.assertRaises(WorkerError):
            worker.stage_s2v(fixture.selection)
        self.assertEqual(len(fixture.requests), before)
        source = worker_source()
        self.assertLessEqual(len(source), MAX_SOURCE_BYTES)
        self.assertLess(
            len(("/usr/bin/python3.11 -I -B -S -c " + shlex.quote(source.decode())).encode()),
            131072,
        )

    def test_comfy_variant_subject_and_cross_profile_refuse_before_transfer(self):
        plain = comfy_fixture.plan()
        l40s = dict(plain, profile_id=profile.L40S_PROFILE_ID)
        self.assertEqual(comfy_worker(comfy_plan(plain)), S2V_WORKER)
        self.assertEqual(comfy_worker(comfy_plan(l40s)), S2V_L40S_WORKER)
        self.assertNotEqual(comfy_subject(plain), comfy_subject(l40s))
        for wrong in (profile.PROFILE_ID, "auto", None):
            reviewer = comfy_fixture.Reviewer()
            with self.assertRaises(ValueError):
                CheckedComfyRuntime.reviewed(dict(plain, profile_id=wrong), reviewer)
            self.assertEqual(reviewer.calls, 0)
        request = dict(
            profile=dict(id=profile.L40S_PROFILE_ID),
            selection=dict(
                profile_id=profile.L40S_PROFILE_ID, hardware=dict(gpu=profile.L40S_GPU_ID)
            ),
        )
        _require_profile(request, l40s)
        for plan in (plain,):
            with self.assertRaises(ValueError):
                _require_profile(request, plan)
        runtime, worker, selected, reviewer, state, bridge = (
            comfy_fixture.ComfyControllerTests().fixture()
        )
        worker.selection = SimpleNamespace(worker_profile={"id": profile.L40S_PROFILE_ID})
        with self.assertRaises(ValueError):
            runtime.prepare(worker, selected, deadline=time.monotonic() + 20)
        self.assertEqual(state["calls"], [])

    def test_native_bridge_selection_bound_gpu_before_upload_and_original_deadline(self):
        fixture = remote_fixture.RemoteFixture()
        fixture.gpu_id = profile.L40S_GPU_ID
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        wave = fixture.bridge.synthesize(fixture.request, fixture.reference)
        receipt = fixture.bridge.receipt_for(fixture.request, wave)
        self.assertEqual(receipt["hardware"]["gpu_name"], profile.L40S_GPU_ID)
        self.assertEqual(receipt["hardware"]["capability"], [8, 9])
        self.assertEqual(receipt["hardware"]["memory_bytes"], 46000 * 1024**2)
        self.assertEqual(
            fixture.bridge._binding["scope"], fixture.gate.service.approval.subject_sha256
        )
        self.assertTrue(all(cap == fixture.bridge._deadline for cap in fixture.deadlines))
        wrong = speech_selection(
            fixture.root,
            hardware=NativeHardware("synthetic-worker", profile.GPU_ID, 32768, (12, 0)),
        )
        fixture.bridge._selection = None
        before = len(fixture.requests)
        with self.assertRaises(NativeSpeechError):
            fixture.bridge.select(wrong)
        self.assertEqual(len(fixture.requests), before)

    def test_nvml_hardware_producers_use_selected_profile_and_original_cap(self):
        deadline = time.monotonic() + 10
        resources = {key: l40s_runtime()[key] for key in bootstrap.M4_RESOURCE_FIELDS}
        with (
            patch.object(sys, "version_info", (3, 11, 14)),
            patch.object(sys, "platform", "linux"),
            patch.object(platform, "machine", return_value="x86_64"),
            patch.object(platform, "python_version", return_value="3.11.14"),
            patch.object(bootstrap, "resource_evidence", return_value=resources),
            patch.object(
                bootstrap, "bounded_process", return_value=b"NVIDIA L40S,46068,68,8.9,580.95.05\n"
            ) as child,
        ):
            result = bootstrap.s2v_hardware_admission(
                self.root, deadline, profile_id=profile.L40S_PROFILE_ID
            )
            self.assertEqual(result, admission())
            self.assertEqual(child.call_args.kwargs["absolute_deadline"], deadline)
            self.assertEqual(child.call_args.args[0][0], "/usr/bin/nvidia-smi")
            with self.assertRaises(ValueError):
                bootstrap.s2v_hardware_admission(self.root, deadline)
        binary = self.root / "python"
        library = self.root / "seccomp"
        binary.write_bytes(b"synthetic interpreter never executed")
        library.write_bytes(b"synthetic library never loaded")
        binary.chmod(0o600)
        library.chmod(0o600)
        exists, resolve = Path.exists, Path.resolve
        aliases = {
            "/lib/x86_64-linux-gnu/libseccomp.so.2",
            "/usr/lib/x86_64-linux-gnu/libseccomp.so.2",
        }

        def present(path):
            return str(path) in aliases or exists(path)

        def resolved(path, *args, **kwargs):
            if str(path) in aliases:
                return library
            if str(path) == sys.executable:
                return binary
            return resolve(path, *args, **kwargs)

        with (
            patch.object(sys, "version_info", (3, 11, 14)),
            patch.object(sys, "platform", "linux"),
            patch.object(platform, "machine", return_value="x86_64"),
            patch.object(platform, "libc_ver", return_value=("glibc", "2.39")),
            patch.object(Path, "exists", present),
            patch.object(Path, "resolve", resolved),
            patch.object(
                native_bootstrap, "bounded_process", return_value=b"NVIDIA L40S,46068,8.9"
            ) as child,
        ):
            result = native_bootstrap.nr_preflight(deadline, profile_id=profile.L40S_PROFILE_ID)
            self.assertEqual(result["interpreter"]["path"], str(binary))
            self.assertEqual(result["seccomp"]["path"], str(library))
            self.assertEqual(child.call_args.kwargs["absolute_deadline"], deadline)
            with self.assertRaises(ValueError):
                native_bootstrap.nr_preflight(deadline)
            for raw in (
                b"NVIDIA L40S,46068,12.0",
                b"NVIDIA GeForce RTX 5090,32768,8.9",
                b"NVIDIA L40S,49153,8.9",
            ):
                child.return_value = raw
                with self.assertRaises(ValueError):
                    native_bootstrap.nr_preflight(deadline, profile_id=profile.L40S_PROFILE_ID)

    def test_l40s_observation_requires_bf16_math_kernel_on_exact_gpu(self):
        import importlib
        from contextlib import contextmanager

        from cineforge import comfy_runtime_launcher as launcher

        events = []

        @contextmanager
        def math_backend(backend):
            self.assertEqual(backend, "synthetic-math")
            events.append("math-enter")
            try:
                yield
            finally:
                events.append("math-exit")

        dtype = object()

        def ones(*args, **kwargs):
            self.assertIs(kwargs["dtype"], dtype)
            self.assertEqual(kwargs["device"], "cuda")
            events.append("bf16-cuda-input")
            return object()

        def sdpa(*args):
            events.append("sdpa")
            return object()

        cuda = SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda i: SimpleNamespace(
                name=profile.L40S_GPU_ID, total_memory=46000 * 1024**2
            ),
            get_device_capability=lambda: (8, 9),
        )
        torch = SimpleNamespace(
            __version__="2.8.0+cu128",
            version=SimpleNamespace(cuda="12.8"),
            jit=SimpleNamespace(_state=SimpleNamespace(_enabled=False)),
            cuda=cuda,
            bfloat16=dtype,
            ones=ones,
            nn=SimpleNamespace(
                functional=SimpleNamespace(scaled_dot_product_attention=sdpa),
                attention=SimpleNamespace(
                    sdpa_kernel=math_backend, SDPBackend=SimpleNamespace(MATH="synthetic-math")
                ),
            ),
        )
        modules = {
            "torch": torch,
            "av": SimpleNamespace(),
            "PIL": SimpleNamespace(),
            "PIL.Image": SimpleNamespace(),
            "numpy": SimpleNamespace(__version__="2.2.6"),
            "torchaudio": SimpleNamespace(__version__="2.8.0+cu128"),
            "torchvision": SimpleNamespace(__version__="0.23.0+cu128"),
            "transformers": SimpleNamespace(__version__="4.57.3"),
        }
        with (
            patch.dict("os.environ", {"PYTORCH_JIT": "0"}),
            patch.object(launcher, "_require_abi"),
            patch.object(sys, "version_info", (3, 11, 14)),
            patch.object(sys, "platform", "linux"),
            patch.object(platform, "machine", return_value="x86_64"),
            patch.object(platform, "libc_ver", return_value=("glibc", "2.39")),
            patch.object(importlib, "import_module", side_effect=lambda name: modules[name]),
            patch.object(
                bootstrap, "bounded_process", side_effect=RuntimeError("stop before codecs")
            ) as child,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before codecs"):
                launcher._observe(
                    self.root, time.monotonic() + 5, profile_id=profile.L40S_PROFILE_ID
                )
            self.assertEqual(events, ["bf16-cuda-input", "math-enter", "sdpa", "math-exit"])
            self.assertEqual(child.call_count, 1)
            child.reset_mock()
            events.clear()
            cuda.is_bf16_supported = lambda: False
            with self.assertRaises(ValueError):
                launcher._observe(
                    self.root, time.monotonic() + 5, profile_id=profile.L40S_PROFILE_ID
                )
            child.assert_not_called()
            self.assertEqual(events, [])

    def test_expired_hardware_cap_never_starts_a_child(self):
        with (
            patch.object(sys, "version_info", (3, 11, 14)),
            patch.object(sys, "platform", "linux"),
            patch.object(platform, "machine", return_value="x86_64"),
            patch("subprocess.Popen") as child,
        ):
            with self.assertRaises(ValueError):
                bootstrap.s2v_hardware_admission(
                    self.root, time.monotonic() - 1, profile_id=profile.L40S_PROFILE_ID
                )
            child.assert_not_called()
