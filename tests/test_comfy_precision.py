"""467 synthetic quantized modules exercise actual guards; never Torch/model/GPU proof."""

from __future__ import annotations

import copy
import tempfile
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cineforge import comfy_model_precision as runtime
from cineforge import comfy_precision as policy
from cineforge.native_remote_plan import nr_digest, nr_encode, nr_json

H = "a" * 64


class Tensor:
    def __init__(self, shape=(2, 2), dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.device = SimpleNamespace(type="cuda")


class Quantized(Tensor):
    def __init__(self):
        super().__init__(dtype="fp16")  # Reproduces quant-config FP16-first construction.
        self._qdata = Tensor(dtype="fp8")
        self.params = SimpleNamespace(scale=Tensor(shape=(), dtype="fp32"))

    def dequantize(self):
        return Tensor(self.shape, self.dtype)


class Handle:
    counter = 0

    def __init__(self, mapping, function):
        Handle.counter += 1
        self.id, self.mapping = Handle.counter, mapping
        mapping[self.id] = function

    def remove(self):
        self.mapping.pop(self.id, None)


class Matrix:
    def __init__(self):
        self.weight = Quantized()
        self.quant_format = "float8_e4m3fn"
        self.weight_function, self.bias_function = [], []
        self._forward_pre_hooks, self._forward_hooks = {}, {}
        self.native_calls = 0
        self.output_dtype = "bf16"

    def register_forward_pre_hook(self, function):
        return Handle(self._forward_pre_hooks, function)

    def register_forward_hook(self, function):
        return Handle(self._forward_hooks, function)

    def __call__(self, value):
        for function in list(self._forward_pre_hooks.values()):
            function(self, (value,))
        self.native_calls += 1
        result = Tensor(dtype=self.output_dtype)
        for function in list(self._forward_hooks.values()):
            function(self, (value,), result)
        return result


class Patcher:
    def __init__(self):
        self.model = SimpleNamespace(manual_cast_dtype="fp16", current_patcher=None)
        self.model.get_dtype_inference = lambda: self.model.manual_cast_dtype
        self.matrices = [("layer." + str(i).zfill(3), Matrix()) for i in range(policy.COUNT)]

        class Diffusion(Matrix):
            def named_modules(inner):
                return self.matrices

            def __call__(inner, value):
                for callback in list(inner._forward_pre_hooks.values()):
                    callback(inner, (value,))
                return self.matrices[0][1](value)

        self.model.diffusion_model = Diffusion()
        self.callbacks, self.wrappers, self.object_patches = {}, {}, {}
        self.patches_uuid = uuid.uuid4()
        self.setter_count = 0
        self.patches, self.weight_wrapper_patches, self.forced_hooks = {}, {}, None

    def clone(self):
        other = copy.copy(self)
        other.object_patches = dict(self.object_patches)
        other.callbacks = {
            k: {key: list(v) for key, v in data.items()} for k, data in self.callbacks.items()
        }
        other.wrappers = {
            k: {key: list(v) for key, v in data.items()} for k, data in self.wrappers.items()
        }
        return other

    def set_model_compute_dtype(self, dtype):
        self.object_patches["manual_cast_dtype"] = dtype
        self.patches_uuid = uuid.uuid4()
        self.setter_count += 1

    def add_callback_with_key(self, kind, key, callback):
        self.callbacks.setdefault(kind, {}).setdefault(key, []).append(callback)

    def add_wrapper_with_key(self, kind, key, callback):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(callback)

    def get_callbacks(self, kind, key):
        return self.callbacks.get(kind, {}).get(key, [])

    def get_all_callbacks(self, kind):
        return [item for group in self.callbacks.get(kind, {}).values() for item in group]

    def get_all_wrappers(self, kind):
        return [item for group in self.wrappers.get(kind, {}).values() for item in group]

    def get_wrappers(self, kind, key):
        return self.wrappers.get(kind, {}).get(key, [])


class PrecisionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cf-precision-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.directory = self.root / "comfy-precision"
        self.directory.mkdir(mode=0o700)
        self.now = 100.0
        clock = patch("cineforge.native_remote_bootstrap.time.monotonic", lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.rows = [["layer." + str(i).zfill(3), [2, 2]] for i in range(policy.COUNT)]
        self.synthetic_sha = nr_digest(self.rows)
        for module in (runtime, policy):
            change = patch.object(module, "MATRICES_SHA256", self.synthetic_sha)
            change.start()
            self.addCleanup(change.stop)
        self.exit_action = lambda: None
        self.cast_dtype = "bf16"
        self.active_casts = self.maximum_casts = self.casts = 0
        fixture = self

        class Context:
            def __init__(self, module, value, **kwargs):
                fixture.active_casts += 1
                fixture.casts += 1
                fixture.maximum_casts = max(fixture.maximum_casts, fixture.active_casts)
                self.weight = Tensor(module.weight.shape, fixture.cast_dtype)

            def __enter__(self):
                return self.weight, None

            def __exit__(self, *args):
                fixture.active_casts -= 1
                fixture.exit_action()

        self.ops = SimpleNamespace(QuantizedTensor=Quantized, CastBiasWeightContext=Context)
        self.torch = SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            float8_e4m3fn="fp8",
            empty=lambda shape, **kw: Tensor(shape, kw["dtype"]),
            no_grad=nullcontext,
            cuda=SimpleNamespace(mem_get_info=lambda: (1024**3, 1024**3), synchronize=lambda: None),
        )
        self.ctx = SimpleNamespace(node_id="152", list_index=0, prompt_id=str(uuid.uuid4()))
        self.callbacks = SimpleNamespace(ON_LOAD="load", ON_PRE_RUN="pre", ON_CLEANUP="cleanup")
        self.wrappers = SimpleNamespace(APPLY_MODEL="apply")
        self.expected = dict(
            worker_id="owned-worker",
            session_id="owned-session",
            profile_id="m5-wan22-s2v14b-a40-v1",
            binding_sha256=H,
            package_sha256=H,
            comfy_sha256=H,
            source_sha256=H,
            runtime_sha256=H,
            enforcer_sha256=H,
        )
        self.guard = runtime.ModelGuard(
            self.root,
            200,
            self.torch,
            self.ops,
            self.callbacks,
            self.wrappers,
            self.expected,
            lambda: self.ctx,
            lambda: None,
            lambda module: None,
        )
        self.base = Patcher()
        self.patcher = self.guard.attach(self.base)
        # Faithful fake patch_model: object patch is applied before load callbacks.
        self.patcher.model.manual_cast_dtype = self.patcher.object_patches["manual_cast_dtype"]
        self.patcher.object_patches["model_sampling"] = object()
        self.patcher.model.model_sampling = self.patcher.object_patches["model_sampling"]
        self.patcher.model.current_patcher = self.patcher
        self.request = self.new_request()

    def save(self, name, value):
        path = self.directory / name
        path.write_bytes(nr_encode(value))
        path.chmod(0o600)

    def new_request(self):
        value = dict(
            self.expected,
            version=1,
            policy=policy.POLICY,
            nonce=uuid.uuid4().hex,
            nonce_graph_sha256=H,
            selection_sha256=H,
            deadline=190.0,
        )
        self.save(value["nonce"] + ".request.json", value)
        return value

    def apply(self, input_dtype="bf16"):
        fixture = self

        class Executor:
            class_obj = fixture.patcher.model

            def __call__(self):
                fixture.patcher.model.diffusion_model(Tensor(dtype=input_dtype))
                return Tensor(dtype="fp32")  # Intentional final denoising float remains allowed.

        return self.guard.apply(Executor())

    def test_fp16_quant_config_changes_only_actual_manual_and_effective_compute(self):
        self.assertEqual(self.base.matrices[0][1].weight.dtype, "fp16")
        self.guard.on_pre_run(self.patcher)  # Does not rely on ON_LOAD firing.
        self.assertEqual(self.casts, 467)
        self.assertEqual(self.maximum_casts, 1)
        self.assertEqual(self.active_casts, 0)
        self.assertEqual(self.apply().dtype, "fp32")
        self.guard.on_cleanup(self.patcher)
        receipt = nr_json((self.directory / (self.request["nonce"] + ".closed.json")).read_bytes())
        policy.precision_receipt(receipt, self.request)
        self.assertEqual(receipt["prompt_id"], self.ctx.prompt_id)
        self.assertEqual(receipt["matrix_calls"], 1)
        self.assertFalse(self.guard.armed)
        self.assertTrue(
            all(not m._forward_pre_hooks and not m._forward_hooks for _, m in self.base.matrices)
        )
        self.assertTrue(all(m.weight._qdata.dtype == "fp8" for _, m in self.base.matrices))

    def test_enzh_cached_model_and_sampling_clone_reaudit_fresh_nonce_without_reload(self):
        model = self.patcher.model
        patch_uuid = self.patcher.patches_uuid
        for index in range(2):
            if index:
                self.save(self.request["nonce"] + ".finished", {"historical": True})
                self.request = self.new_request()
                self.ctx.prompt_id = str(uuid.uuid4())
                self.patcher = self.guard.base.clone().clone()  # ModelSamplingSD3 clone.
                self.patcher.object_patches["model_sampling"] = object()
                self.patcher.model.model_sampling = self.patcher.object_patches["model_sampling"]
                self.patcher.model.current_patcher = self.patcher
            self.guard.on_pre_run(self.patcher)
            self.apply()
            self.guard.on_cleanup(self.patcher)
            self.assertIs(self.patcher.model, model)
            self.assertEqual(self.patcher.patches_uuid, patch_uuid)
        self.assertEqual(self.casts, 934)
        self.assertEqual(self.guard.base.setter_count, 1)

    def test_wrong_effective_cast_fails_before_inference_and_disarms_without_sampler_finally(self):
        self.cast_dtype = "fp16"
        with self.assertRaises(ValueError):
            self.guard.on_load(self.patcher)
        self.assertEqual(self.casts, 1)
        self.assertEqual(self.base.matrices[0][1].native_calls, 0)
        self.assertFalse(self.guard.armed)
        self.assertTrue(self.guard.failed)
        self.assertTrue(all(not m._forward_pre_hooks for _, m in self.base.matrices))

    def test_cast_exit_expiry_or_failure_cannot_leave_audit_or_observers(self):
        def late():
            self.now = 191

        self.exit_action = late
        with self.assertRaises(ValueError):
            self.guard.on_load(self.patcher)
        self.assertEqual(self.active_casts, 0)
        self.assertFalse((self.directory / (self.request["nonce"] + ".audit.json")).exists())
        self.assertTrue(all(not m._forward_hooks for _, m in self.base.matrices))

    def test_missing_callback_refuses_before_any_model_effect(self):
        self.patcher.callbacks["pre"].clear()
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)

    def test_replaced_callback_refuses_before_any_model_effect(self):
        self.patcher.callbacks["pre"][policy.POLICY] = [lambda _: None]
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)

    def test_wrong_manual_dtype_refuses_before_any_model_effect(self):
        self.patcher.model.manual_cast_dtype = "fp16"
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)

    def test_stale_patch_or_replaced_module_hooks_refuse(self):
        self.patcher.patches_uuid = uuid.uuid4()
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)

    def test_replaced_matrix_guard_refuses_before_any_model_effect(self):
        self.base.matrices[0][1]._forward_hooks.clear()
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)

    def test_unknown_backend_refuses_before_cast_or_inference(self):
        def unsupported():
            raise ValueError("synthetic forbidden compiler/backend")

        self.guard.backend_check = unsupported
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)
        self.assertEqual(self.base.matrices[0][1].native_calls, 0)

    def test_unknown_operation_source_refuses_before_cast(self):
        def unknown(module):
            raise ValueError("synthetic wrong module origin")

        self.guard.module_check = unknown
        with self.assertRaises(ValueError):
            self.guard.on_pre_run(self.patcher)
        self.assertEqual(self.casts, 0)

    def test_another_actual_context_cannot_use_audited_model(self):
        self.guard.on_pre_run(self.patcher)
        self.ctx.prompt_id = str(uuid.uuid4())
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(self.base.matrices[0][1].native_calls, 0)

    def test_wrong_input_refuses_before_matrix_native_and_wrong_output_is_truthful_after(self):
        self.guard.on_pre_run(self.patcher)
        with self.assertRaises(ValueError):
            self.apply("fp16")
        self.assertEqual(self.base.matrices[0][1].native_calls, 0)
        self.assertFalse((self.directory / (self.request["nonce"] + ".step-001.json")).exists())

    def test_wrong_actual_output_fails_after_native_and_never_records_success(self):
        self.guard.on_pre_run(self.patcher)
        self.base.matrices[0][1].output_dtype = "fp16"
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(self.base.matrices[0][1].native_calls, 1)
        self.assertTrue((self.directory / (self.request["nonce"] + ".failed")).exists())
        self.assertFalse((self.directory / (self.request["nonce"] + ".step-001.json")).exists())

    def test_stale_other_prompt_context_or_missing_audit_never_delegates(self):
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(self.base.matrices[0][1].native_calls, 0)

    def test_expiry_during_owned_cleanup_removes_hooks_but_never_publishes_closed(self):
        self.guard.on_pre_run(self.patcher)
        self.apply()
        self.now = 191
        with self.assertRaises(ValueError):
            self.guard.on_cleanup(self.patcher)
        self.assertFalse((self.directory / (self.request["nonce"] + ".closed.json")).exists())
        self.assertTrue(all(not m._forward_hooks for _, m in self.base.matrices))

    def test_duplicate_concurrent_nonce_or_expired_entry_fails_currentness(self):
        self.new_request()
        with self.assertRaises(ValueError):
            self.guard.fingerprint("161")

    def test_receipt_strict_types_and_stale_bindings(self):
        self.guard.on_pre_run(self.patcher)
        self.apply()
        self.guard.on_cleanup(self.patcher)
        good = nr_json((self.directory / (self.request["nonce"] + ".closed.json")).read_bytes())
        for key, value in [
            ("audit_count", True),
            ("step", True),
            ("prompt_id", str(uuid.uuid4())),
            ("request_sha256", "b" * 64),
            ("manual_dtype", "fp16"),
            ("model_sha256", H),
        ]:
            altered = dict(good, **{key: value})
            if key == "prompt_id":
                # Identity equality is checked by current finish protocol, not UUID syntax alone.
                self.assertNotEqual(altered["prompt_id"], good["prompt_id"])
                continue
            with self.assertRaises(ValueError):
                policy.precision_receipt(altered, self.request)


class PrecisionWorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        import os

        from test_comfy_runtime import plan

        from cineforge import comfy_precision_worker as worker

        self.fixture = PrecisionGuardTests(
            "test_fp16_quant_config_changes_only_actual_manual_and_effective_compute"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f, self.worker, self.plan = self.fixture, worker, plan()
        (self.f.directory / (self.f.request["nonce"] + ".request.json")).unlink()
        self.native = self.f.root / "native"
        (self.native / "firstparty/cineforge").mkdir(parents=True, mode=0o700)
        body = b"inert synthetic enforcer fixture; never imported"
        path = self.native / "firstparty/cineforge/comfy_model_precision.py"
        path.write_bytes(body)
        path.chmod(0o600)
        import hashlib

        from cineforge.comfy_runtime_plan import comfy_subject

        self.binding = dict(package=H, worker="owned-worker", session="owned-session")
        self.f.expected.update(
            comfy_sha256=comfy_subject(self.plan),
            runtime_sha256=nr_digest(self.plan["installed"]),
            source_sha256=nr_digest(self.plan["source"]),
            binding_sha256=nr_digest(self.binding),
            enforcer_sha256=hashlib.sha256(body).hexdigest(),
        )
        path = self.f.root / "comfy-selection.json"
        path.write_bytes(nr_encode(self.plan))
        path.chmod(0o600)
        path = self.f.root / "server-pid.json"
        path.write_bytes(nr_encode({"pid": os.getpid()}))
        path.chmod(0o600)
        self.base = dict(
            plan={"session_id": "owned-session"},
            profile={"id": self.f.expected["profile_id"]},
            comfy_subject_sha256=comfy_subject(self.plan),
            precision_nonce=uuid.uuid4().hex,
        )
        for target, replacement in [
            ("cineforge.comfy_runtime_worker.current_comfy", lambda *a: None),
            ("cineforge.worker_bootstrap.worker_root", lambda _: self.f.root),
        ]:
            patched = patch(target, replacement)
            patched.start()
            self.addCleanup(patched.stop)

    def begin(self):
        return self.worker.precision_operation(
            dict(
                self.base,
                operation="native_comfy_precision_begin",
                precision_binding=dict(
                    version=1, policy=policy.POLICY, nonce_graph_sha256=H, selection_sha256=H
                ),
            ),
            self.native,
            180,
            self.binding,
        )

    def finish(self, prompt_id=None):
        return self.worker.precision_operation(
            dict(
                self.base,
                operation="native_comfy_precision_finish",
                precision_prompt_id=self.f.ctx.prompt_id if prompt_id is None else prompt_id,
            ),
            self.native,
            180,
            self.binding,
        )

    def generate(self):
        self.f.request = self.begin()
        self.f.guard.on_pre_run(self.f.patcher)
        self.f.apply()
        self.f.guard.on_cleanup(self.f.patcher)

    def test_actual_fixed_begin_audit_guarded_forward_close_finish_and_no_replay(self):
        self.generate()
        result = self.finish()
        self.assertEqual(result["prompt_id"], self.f.ctx.prompt_id)
        self.assertEqual(result["matrix_calls"], 1)
        self.assertEqual(self.f.request["deadline"], 180)  # Worker-local original dispatcher cap.
        with self.assertRaises((ValueError, FileExistsError)):
            self.finish()
        with self.assertRaises((ValueError, FileExistsError)):
            self.begin()

    def test_finish_without_actual_model_audit_or_forward_refuses(self):
        self.begin()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.finish()

    def test_foreign_job_cannot_consume_historical_model_facts_as_current(self):
        self.generate()
        with self.assertRaises(ValueError):
            self.finish(str(uuid.uuid4()))
        with self.assertRaises(FileExistsError):
            self.finish()

    def test_missing_real_sampler_cleanup_refuses_even_after_valid_forward(self):
        self.f.request = self.begin()
        self.f.guard.on_pre_run(self.f.patcher)
        self.f.apply()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.finish()

    def test_late_final_currentness_cannot_publish_finished_receipt(self):
        self.generate()

        def late(*args):
            self.f.now = 181

        with patch("cineforge.comfy_runtime_worker.current_comfy", late):
            with self.assertRaises(ValueError):
                self.finish()
        self.assertFalse((self.f.directory / (self.base["precision_nonce"] + ".finished")).exists())

    def test_other_active_request_and_more_than_two_are_refused(self):
        self.begin()
        self.base["precision_nonce"] = uuid.uuid4().hex
        with self.assertRaises(ValueError):
            self.begin()


class PrecisionPostBoundaryTests(unittest.TestCase):
    def test_exact_current_graph_body_stale_intent_and_preseeded_cache_fields(self):
        import time

        from test_comfy_runtime import ComfyControllerTests

        runtime, worker, _, _, _, bridge = ComfyControllerTests().fixture()
        runtime._prepared = True
        graph = {"161": {"class_type": policy.CLASS, "inputs": {"weight_dtype": "default"}}}
        selected = SimpleNamespace(approved_subject_sha256=H)
        worker._comfy_selection = selected
        runtime._precision = dict(nonce_graph_sha256=nr_digest(graph), selection_sha256=H)
        body = {"prompt": graph}
        runtime.require_precision_post(worker, body, deadline=time.monotonic() + 30)
        altered = copy.deepcopy(body)
        altered["prompt"]["161"]["is_changed"] = ["stale-preseed"]
        with self.assertRaises(ValueError):
            runtime.require_precision_post(worker, altered, deadline=time.monotonic() + 30)
        runtime._precision["selection_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            runtime.require_precision_post(worker, body, deadline=time.monotonic() + 30)
        runtime._precision = None
        with self.assertRaises(ValueError):
            runtime.require_precision_post(worker, body, deadline=time.monotonic() + 30)
        self.assertFalse(bridge._unresolved)  # Pure pre-send refusals do not invent remote effects.


class PrecisionLoaderFactoryTests(unittest.TestCase):
    """Actual factory and guards with explicitly fake modules/origins; no SDK import."""

    def setUp(self):
        import hashlib
        import inspect

        self.f = PrecisionGuardTests(
            "test_fp16_quant_config_changes_only_actual_manual_and_effective_compute"
        )
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.loads = 0
        self.graph = {
            "161": {
                "class_type": policy.CLASS,
                "inputs": {"unet_name": policy.MODEL_NAME, "weight_dtype": "default"},
            },
            "152": {"class_type": "KSampler", "inputs": {"seed": 5201}},
        }
        self.f.ctx.node_id = "161"
        fixture = self

        class Original:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {"unet_name": [[policy.MODEL_NAME]], "weight_dtype": [["default"]]}
                }

            def load_unet(inner, name, dtype):
                fixture.loads += 1
                base = Patcher()
                base.model.model_config = object()
                return (base,)

        def quantize(*args):
            raise AssertionError("synthetic backend must not execute")

        def dequantize(*args):
            raise AssertionError("synthetic backend must not execute")

        eager = SimpleNamespace(
            quantize_per_tensor_fp8=quantize, dequantize_per_tensor_fp8=dequantize
        )
        layout = object()
        fake_modules = {
            "nodes": SimpleNamespace(
                UNETLoader=Original, NODE_CLASS_MAPPINGS={"UNETLoader": Original}
            ),
            "comfy.ops": self.f.ops,
            "comfy.patcher_extension": SimpleNamespace(
                CallbacksMP=self.f.callbacks, WrappersMP=self.f.wrappers
            ),
            "comfy.model_base": SimpleNamespace(WAN22_S2V=SimpleNamespace),
            "comfy.supported_models": SimpleNamespace(WAN22_S2V=object),
            "comfy.quant_ops": SimpleNamespace(TensorCoreFP8E4M3Layout=layout),
            "torch": self.f.torch,
            "comfy_execution.utils": SimpleNamespace(get_executing_context=lambda: self.f.ctx),
            "comfy_kitchen": SimpleNamespace(
                list_backends=lambda: {"eager": {"available": True, "disabled": False}},
                registry=SimpleNamespace(
                    get_implementation=lambda operation, **kw: getattr(eager, operation)
                ),
            ),
            "comfy_kitchen.backends.eager.quantization": eager,
        }
        self.f.torch.float16 = "fp16"
        self.f.torch.jit = SimpleNamespace(_state=SimpleNamespace(_enabled=False))
        enforcer = self.f.root / "firstparty-enforcer.py"
        enforcer.write_bytes(Path(runtime.__file__).read_bytes())
        enforcer.chmod(0o600)
        previous_init = Matrix.__init__

        def initialize(module):
            previous_init(module)
            module.layout_type = module.weight._layout_cls = "TensorCoreFP8E4M3Layout"
            module.weight.layout_cls = layout
            module.weight.params.orig_dtype = "fp16"
            module._full_precision_mm = True

        original_file = inspect.getfile

        def origin(value):
            if value is Original:
                return str(self.f.root / "ComfyUI/nodes.py")
            if value is Matrix:
                return str(self.f.root / "ComfyUI/comfy/ops.py")
            if value in (quantize, dequantize):
                return str(
                    self.f.root / "comfy-runtime/site/comfy_kitchen/backends/eager/quantization.py"
                )
            return original_file(value)

        for change in [
            patch.object(runtime, "_INSTALLED", None),
            patch.object(runtime, "__file__", str(enforcer)),
            patch.object(runtime.importlib, "import_module", side_effect=lambda n: fake_modules[n]),
            patch.object(runtime.inspect, "getfile", side_effect=origin),
            patch.object(Matrix, "__init__", initialize),
            patch.object(Matrix, "__module__", "comfy.ops"),
            patch.dict(runtime.os.environ, {"PYTORCH_JIT": "0", "TORCH_COMPILE_DISABLE": "1"}),
        ]:
            change.start()
            self.addCleanup(change.stop)
        self.request = self.f.request
        self.request.update(
            nonce_graph_sha256=nr_digest(self.graph),
            enforcer_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        )
        self.f.save(self.request["nonce"] + ".request.json", self.request)
        self.cls = runtime.install(
            self.f.root, 200, {k: v for k, v in self.f.expected.items() if k != "enforcer_sha256"}
        )
        self.loader = self.cls()
        self.nodes = fake_modules["nodes"]

    def dynamic(self, graph):
        return SimpleNamespace(
            node_overrides={},
            ephemeral_prompt={},
            get_original_prompt=lambda: graph,
            get_node=lambda key: graph[key],
        )

    def load(self, graph=None, dynamic=None):
        graph = self.graph if graph is None else graph
        return self.loader.load_unet(
            policy.MODEL_NAME,
            "default",
            graph,
            "161",
            self.dynamic(graph) if dynamic is None else dynamic,
        )[0]

    def test_real_factory_fingerprint_runtime_projection_origin_and_no_upstream_override(self):
        self.assertEqual(
            self.cls.IS_CHANGED(policy.MODEL_NAME, "default", {}, "161", None),
            self.request["nonce"],
        )
        self.graph["161"]["is_changed"] = [self.request["nonce"]]  # Engine-added only.
        patcher = self.load()
        self.assertEqual(patcher.object_patches["manual_cast_dtype"], "bf16")
        self.assertEqual(runtime.installed_nodes(), policy.FIRST_PARTY_NODES)
        self.assertIsNot(self.nodes.NODE_CLASS_MAPPINGS["UNETLoader"], self.cls)
        self.assertEqual(len(self.graph), 2)  # Tiny fake semantic graph only.
        with self.assertRaises(ValueError):
            runtime.install(self.f.root, 200, self.f.expected)

    def test_unknown_semantic_prompt_override_or_role_refuses_before_upstream_loader(self):
        graph = copy.deepcopy(self.graph)
        graph["161"]["unreviewed_semantics"] = True
        with self.assertRaises(ValueError):
            self.load(graph)
        dynamic = self.dynamic(self.graph)
        dynamic.node_overrides["161"] = self.graph["161"]
        with self.assertRaises(ValueError):
            self.load(dynamic=dynamic)
        with self.assertRaises(ValueError):
            self.cls.IS_CHANGED(policy.MODEL_NAME, "default", {}, "foreign", None)
        self.assertEqual(self.loads, 0)

    def test_actual_factory_two_language_cache_uses_one_load_and_fresh_audit_context(self):
        model, patch_uuid = None, None
        for index in range(2):
            if index:
                self.f.save(self.request["nonce"] + ".finished", {"historical": True})
                self.graph["152"]["inputs"]["seed"] = 5202
                self.request = dict(
                    self.request, nonce=uuid.uuid4().hex, nonce_graph_sha256=nr_digest(self.graph)
                )
                self.f.save(self.request["nonce"] + ".request.json", self.request)
                self.f.ctx.prompt_id = str(uuid.uuid4())
            self.f.ctx.node_id = "161"
            patcher = self.load().clone()
            patcher.object_patches["model_sampling"] = object()
            patcher.model.model_sampling = patcher.object_patches["model_sampling"]
            patcher.model.manual_cast_dtype = "bf16"
            patcher.model.current_patcher = patcher
            self.f.ctx.node_id = "152"
            guard = patcher.get_callbacks("pre", policy.POLICY)[0].__self__
            self.f.guard, self.f.patcher, self.f.base = guard, patcher, patcher
            guard.on_pre_run(patcher)
            self.f.apply()
            guard.on_cleanup(patcher)
            if index == 0:
                model, patch_uuid = patcher.model, patcher.patches_uuid
            else:
                self.assertIs(patcher.model, model)
                self.assertEqual(patcher.patches_uuid, patch_uuid)
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.f.casts, 934)

    def test_removed_or_replaced_registered_methods_are_not_startup_precision(self):
        self.cls.load_unet = lambda *a: None
        with self.assertRaises(ValueError):
            runtime.installed_nodes()


class PrecisionSupervisedTransportTests(unittest.TestCase):
    def test_real_supervised_post_is_once_and_strict_body_refusal_has_zero_http_calls(self):
        import time

        from test_comfy_runtime import ComfyControllerTests
        from test_s2v import context

        from cineforge.ssh_worker import SupervisedComfyTransport

        with tempfile.TemporaryDirectory(prefix="cf-precision-post-") as directory:
            gate, _ = context(Path(directory).resolve())
            for invalid in (False, True):
                closed, worker, _, _, _, bridge = ComfyControllerTests().fixture()
                closed._prepared = True
                worker.selection = gate.service
                worker.check = lambda: None
                worker.verify_comfy_current = lambda **kw: None
                worker._key = "owned"
                worker.budget = SimpleNamespace(remaining_seconds=lambda _: 500)
                worker._closed_comfy = closed
                worker._comfy_selection = SimpleNamespace(approved_subject_sha256=H)
                graph = {"161": {"class_type": policy.CLASS, "inputs": {}}}
                closed._precision = dict(nonce_graph_sha256=nr_digest(graph), selection_sha256=H)
                calls = []

                class Wire:
                    def request(self, method, path, body=None, **kw):
                        calls.append((method, path, body, kw))
                        return b"{}"

                transport = SupervisedComfyTransport(worker, Wire())
                body = {"prompt": copy.deepcopy(graph)}
                if invalid:
                    body["prompt"]["161"]["is_changed"] = ["attacker-stale-cache"]
                    with self.assertRaises(ValueError):
                        transport.request(
                            "POST",
                            "/prompt",
                            body,
                            timeout_seconds=3,
                            max_bytes=4096,
                            absolute_deadline=time.monotonic() + 2,
                        )
                    self.assertEqual(calls, [])
                else:
                    self.assertEqual(
                        transport.request(
                            "POST",
                            "/prompt",
                            body,
                            timeout_seconds=3,
                            max_bytes=4096,
                            absolute_deadline=time.monotonic() + 2,
                        ),
                        b"{}",
                    )
                    with self.assertRaises(ValueError):
                        transport.request(
                            "POST",
                            "/prompt",
                            body,
                            timeout_seconds=3,
                            max_bytes=4096,
                            absolute_deadline=time.monotonic() + 2,
                        )
                    self.assertEqual(len(calls), 1)
