"""Sealed first-party Wan loader/instance guards, imported only by the remote launcher.

BF16 logical matrix computation, FP8 storage, intentional upstream FP32 work retained.
A cast audit is actual model materialization, not inference or a hardware toy probe.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import uuid
from pathlib import Path
from typing import Any, Callable

from cineforge.comfy_precision import (
    CLASS,
    COUNT,
    FIRST_PARTY_NODES,
    HEADER_SHA256,
    MATRICES_SHA256,
    MAX_MATRIX_BYTES,
    MAX_STEPS,
    MODEL_NAME,
    MODEL_SHA256,
    POLICY,
    precision_receipt,
    precision_request,
)
from cineforge.native_remote_bootstrap import nr_check, nr_read, nr_write
from cineforge.native_remote_plan import nr_digest, nr_encode, nr_json

_INSTALLED: tuple[Any, dict[str, Any]] | None = None

def active_request(root: Path, deadline: float) -> dict[str, Any]:
    nr_check(deadline)
    files = sorted((root / "comfy-precision").glob("*.request.json"))
    if not 1 <= len(files) <= 2:
        raise ValueError("one owned current precision request required")
    active = [p for p in files if not p.with_name(p.name[:-13] + ".finished").exists()]
    if len(active) != 1:
        raise ValueError("precision request unavailable or concurrent")
    value = precision_request(nr_json(nr_read(active[0], maximum=8192)))
    if active[0].name != value["nonce"] + ".request.json":
        raise ValueError("precision request identity changed")
    nr_check(min(deadline, value["deadline"]))
    return value


class ModelGuard:
    """Process-local instance state; persisted receipts never restore this authority."""
    def __init__(self, root: Path, deadline: float, torch: Any, ops: Any,
                 callbacks: Any, wrappers: Any, expected: dict[str, Any], context: Any, backend_check: Callable[[], None],
                 module_check: Callable[[Any], None]) -> None:
        self.root, self.deadline, self.torch, self.ops = root, deadline, torch, ops
        self.callbacks, self.wrappers, self.expected = callbacks, wrappers, expected
        self.context = context
        self.backend_check = backend_check
        self.module_check = module_check
        self.prompt_id: str | None = None
        self.cast_context = ops.CastBiasWeightContext
        self.dynamic_prompt: Any = None
        self.base: Any = None
        self.modules: list[tuple[str, Any, Any, Any]] = []
        self.nonce: str | None = None
        self.request: dict[str, Any] | None = None
        self.model_nonce = uuid.uuid4().hex
        self.step = self.calls = 0
        self.inside = self.failed = False
        self.armed = False
        # Keep exact bound callable identities across clones and cache hits.
        self.on_load, self.on_pre_run, self.apply = self._on_load, self._on_pre_run, self._apply
        self.on_cleanup = self._on_cleanup
        self.before, self.after = self._before, self._after
        self.before_model = self._before_model
        self.model_handle: Any = None

    def _time(self) -> None:
        nr_check(self.deadline)
        if self.request is not None:
            nr_check(self.request["deadline"])
        if self.failed:
            raise ValueError("model precision request failed; never replay")

    def _current(self) -> dict[str, Any]:
        self._time()
        request = active_request(self.root, self.deadline)
        if any(request[key] != value for key, value in self.expected.items()):
            raise ValueError("precision request differs from actual sealed producer")
        if (self.root / "comfy-precision" / (request["nonce"] + ".failed")).exists():
            raise ValueError("model precision request is unresolved")
        self._time()
        return request

    def _context(self, request: dict[str, Any], roles: tuple[str, ...]) -> str:
        context = self.context()
        if (context is None or context.node_id not in roles or context.list_index not in (None, 0)
                or type(context.prompt_id) is not str
                or str(uuid.UUID(context.prompt_id)) != context.prompt_id):
            raise ValueError("actual Comfy execution context missing")
        if self.nonce == request["nonce"] and self.prompt_id != context.prompt_id:
            raise ValueError("precision intent cannot admit another prompt")
        return str(context.prompt_id)

    def fingerprint(self, unique_id: str) -> str:
        if unique_id != "161":
            raise ValueError("precision loader has a different graph role")
        return str(self._current()["nonce"])

    def attach(self, patcher: Any) -> Any:
        self.patcher_methods = {name: getattr(type(patcher), name) for name in (
            "clone", "set_model_compute_dtype", "get_callbacks", "get_wrappers",
            "get_all_callbacks", "get_all_wrappers")}
        self.base = patcher.clone()
        self.base.set_model_compute_dtype(self.torch.bfloat16)
        self.base.add_callback_with_key(self.callbacks.ON_LOAD, POLICY, self.on_load)
        self.base.add_callback_with_key(self.callbacks.ON_PRE_RUN, POLICY, self.on_pre_run)
        self.base.add_callback_with_key(self.callbacks.ON_CLEANUP, POLICY, self.on_cleanup)
        self.base.add_wrapper_with_key(self.wrappers.APPLY_MODEL, POLICY, self.apply)
        rows = []
        for name, module in self.base.model.diffusion_model.named_modules():
            if getattr(module, "quant_format", None) is None:
                continue
            self.module_check(module)
            weight = module.weight
            if (module.quant_format != "float8_e4m3fn"
                    or not isinstance(weight, self.ops.QuantizedTensor)
                    or weight._qdata.dtype != self.torch.float8_e4m3fn
                    or weight.params.scale.dtype != self.torch.float32
                    or tuple(weight.params.scale.shape) != ()
                    or module._forward_pre_hooks or module._forward_hooks
                    or module.weight_function or module.bias_function):
                raise ValueError("unsupported selected-model quantized module")
            rows.append([name, list(weight.shape)])
            pre = module.register_forward_pre_hook(self.before)
            try:
                post = module.register_forward_hook(self.after)
            except BaseException:
                pre.remove()
                raise
            self.modules.append((name, module, pre, post))
        if len(rows) != COUNT or nr_digest(sorted(rows)) != MATRICES_SHA256:
            raise ValueError("actual quantized model differs from checked header")
        self._arm_model()
        self.armed = True
        self._time()
        return self.base.clone()

    @staticmethod
    def _one(values: Any, expected: Any) -> bool:
        return type(values) is list and len(values) == 1 and values[0] is expected

    def _validate(self, patcher: Any) -> None:
        self._time()
        self.backend_check()
        self._time()
        if self.dynamic_prompt is not None:
            prompt = self.dynamic_prompt
            request = self._current()
            if (prompt.node_overrides or prompt.ephemeral_prompt
                    or nr_digest({k: {f:prompt.get_node(k)[f] for f in ("class_type", "inputs")}
                        for k in prompt.get_original_prompt()}) != request["nonce_graph_sha256"]):
                raise ValueError("effective runtime graph differs from precision intent")
        if (self.ops.CastBiasWeightContext is not self.cast_context
                or any(getattr(type(patcher), name) is not method
                       for name, method in self.patcher_methods.items())
                or self.base is None or patcher.model is not self.base.model
                or patcher.patches_uuid != self.base.patches_uuid
                or set(patcher.object_patches) != {"manual_cast_dtype", "model_sampling"}
                or patcher.model.model_sampling is not patcher.object_patches["model_sampling"]
                or patcher.patches or patcher.weight_wrapper_patches or patcher.forced_hooks is not None
                or patcher.object_patches.get("manual_cast_dtype") != self.torch.bfloat16
                or patcher.model.manual_cast_dtype != self.torch.bfloat16
                or patcher.model.get_dtype_inference() != self.torch.bfloat16
                or not self._one(patcher.get_all_callbacks(self.callbacks.ON_LOAD), self.on_load)
                or not self._one(patcher.get_all_callbacks(self.callbacks.ON_PRE_RUN), self.on_pre_run)
                or not self._one(patcher.get_all_wrappers(self.wrappers.APPLY_MODEL), self.apply)
                or not self._one(patcher.get_all_callbacks(self.callbacks.ON_CLEANUP), self.on_cleanup)
                or self.model_handle is None
                or (set(patcher.model.diffusion_model._forward_pre_hooks) != {self.model_handle.id}
                    or patcher.model.diffusion_model._forward_pre_hooks.get(self.model_handle.id) is not self.before_model)
                or len(self.modules) != COUNT):
            raise ValueError("actual BF16 instance or callback enforcement is stale")
        rows = []
        for name, module, pre, post in self.modules:
            self.module_check(module)
            if ((set(module._forward_pre_hooks) != {pre.id}
                        or module._forward_pre_hooks.get(pre.id) is not self.before)
                    or (set(module._forward_hooks) != {post.id}
                        or module._forward_hooks.get(post.id) is not self.after)
                    or module.quant_format != "float8_e4m3fn"
                    or not isinstance(module.weight, self.ops.QuantizedTensor)
                    or module.weight_function or module.bias_function):
                raise ValueError("model precision guard replaced or quantization changed")
            rows.append([name, list(module.weight.shape)])
        if nr_digest(sorted(rows)) != MATRICES_SHA256:
            raise ValueError("actual model matrix coverage changed")
        self._time()

    def _audit(self, patcher: Any) -> None:
        request = self._current()
        self._validate(patcher)
        prompt_id = self._context(request, ("161", "152"))
        if self.nonce == request["nonce"]:
            return
        self.request, self.nonce = request, request["nonce"]
        self.prompt_id = prompt_id
        self.step = self.calls = 0
        target = self.root / "comfy-precision"
        nr_write(target / (self.nonce + ".started"), nr_encode({"pid": os.getpid()}))
        try:
            for _, module, _, _ in self.modules:
                self._time()
                self.backend_check()
                self._time()
                shape = tuple(module.weight.shape)
                size = shape[0] * shape[1] * 2
                if (len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
                        or size > MAX_MATRIX_BYTES or shape[1] > 30720
                        or self.torch.cuda.mem_get_info()[0] < size + 32 * 1024**2):
                    raise ValueError("bounded actual BF16 matrix materialization unavailable")
                input_tensor = self.torch.empty((1, shape[1]), device="cuda:0",
                                                dtype=self.torch.bfloat16)
                with self.torch.no_grad(), self.ops.CastBiasWeightContext(
                        module, input_tensor, offloadable=True) as (weight, bias):
                    plain = (weight.dequantize() if isinstance(weight, self.ops.QuantizedTensor)
                             else weight)
                    if (plain.dtype != self.torch.bfloat16 or tuple(plain.shape) != shape
                            or plain.device.type != "cuda"
                            or bias is not None and bias.dtype != self.torch.bfloat16):
                        raise ValueError("actual quantized operation cast is not BF16")
                    self.torch.cuda.synchronize()
                    self._time()
                    del plain
                del weight, bias, input_tensor
                self.torch.cuda.synchronize()  # Account for offload/teardown work too.
                self._time()  # After controlled cast/offload context cleanup too.
            self._validate(patcher)
            nr_write(target / (self.nonce + ".audit.json"), nr_encode(self._receipt(patcher)))
            self._time()
        except BaseException:
            self._fail()
            raise

    def _arm_model(self) -> None:
        model = self.base.model.diffusion_model
        if model._forward_pre_hooks:
            raise ValueError("foreign selected-model input observer")
        self.model_handle = model.register_forward_pre_hook(self.before_model)

    def _disarm(self) -> None:
        if self.base is not None and self.model_handle is not None:
            model = self.base.model.diffusion_model
            if model._forward_pre_hooks.get(self.model_handle.id) is self.before_model:
                self.model_handle.remove()
        for _, module, pre, post in self.modules:
            if module._forward_pre_hooks.get(pre.id) is self.before:
                pre.remove()
            if module._forward_hooks.get(post.id) is self.after:
                post.remove()
        self.armed = False

    def _fail(self) -> None:
        self.failed = True
        self._disarm()
        if self.nonce is not None:
            try:
                nr_write(self.root / "comfy-precision" / (self.nonce + ".failed"), b"failed")
            except BaseException:
                pass

    def _arm(self) -> None:
        self._time()
        if self.armed:
            return
        self._arm_model()
        for index, (name, module, _, _) in enumerate(self.modules):
            if module._forward_pre_hooks or module._forward_hooks:
                raise ValueError("foreign precision hooks on reused model")
            pre = module.register_forward_pre_hook(self.before)
            try:
                post = module.register_forward_hook(self.after)
            except BaseException:
                pre.remove()
                raise
            self.modules[index] = (name, module, pre, post)
        self.armed = True
        self._time()

    def _on_load(self, patcher: Any, *_args: Any) -> None:
        try:
            self._arm()
            self._audit(patcher)
        except BaseException:
            self._fail()
            raise

    def _on_pre_run(self, patcher: Any) -> None:
        self._on_load(patcher)

    def _on_cleanup(self, patcher: Any) -> None:
        valid = False
        try:
            request = self._current()
            self._context(request, ("152",))
            self._validate(patcher)
            if request["nonce"] != self.nonce or self.step < 1 or self.inside:
                raise ValueError("actual precision sampler completion missing")
            valid = True
        finally:
            # Remove only owned entries even on failure; never remove replacements.
            self._disarm()
        if not valid:
            self._fail()
            raise ValueError("actual precision cleanup incomplete")
        self._time()
        assert self.nonce is not None
        nr_write(self.root / "comfy-precision" / (self.nonce + ".closed.json"),
                 nr_encode(self._receipt(patcher)))
        self._time()

    def _before_model(self, module: Any, args: Any) -> None:
        self._time()
        if not self.inside or not args or args[0].dtype != self.torch.bfloat16:
            self._fail()
            raise ValueError("selected model forward bypassed precision wrapper")

    def _before(self, module: Any, args: Any) -> None:
        self._time()
        self.backend_check()
        self._time()
        if not self.inside or len(args) != 1 or args[0].dtype != self.torch.bfloat16:
            self._fail()
            raise ValueError("quantized matrix input bypassed actual BF16 enforcement")

    def _after(self, module: Any, args: Any, result: Any) -> None:
        self._time()
        if not self.inside or result.dtype != self.torch.bfloat16:
            self._fail()
            raise ValueError("actual quantized matrix output rejected after execution")
        self.calls += 1
        self._time()

    def _receipt(self, patcher: Any) -> dict[str, Any]:
        assert self.request is not None
        return {"version": 1, "policy": POLICY, "request_sha256": nr_digest(self.request),
            "model_sha256": MODEL_SHA256, "header_sha256": HEADER_SHA256,
            "matrices_sha256": MATRICES_SHA256, "enforcer_sha256": self.expected["enforcer_sha256"],
            "model_nonce": self.model_nonce, "patch_uuid": str(patcher.patches_uuid),
            "audit_count": COUNT, "manual_dtype": "bf16", "effective_dtype": "bf16",
            "step": self.step, "matrix_calls": self.calls, "producer_pid": os.getpid(), "prompt_id": self.prompt_id}

    def _apply(self, executor: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            request = self._current()
            self._context(request, ("152",))
            patcher = executor.class_obj.current_patcher
            self._validate(patcher)
            if self.nonce != request["nonce"] or self.inside or self.step >= MAX_STEPS:
                raise ValueError("current model precision audit required before inference")
            self.inside = True
            before = self.calls
            try:
                result = executor(*args, **kwargs)
            finally:
                self.inside = False
            self.torch.cuda.synchronize()
            self._validate(patcher)
            if self._current() != request or self.calls <= before:
                raise ValueError("actual matrix observations missing")
            self.step += 1
            receipt = precision_receipt(self._receipt(patcher), request)
            nr_write(self.root / "comfy-precision" / (
                request["nonce"] + ".step-" + str(self.step).zfill(3) + ".json"), nr_encode(receipt))
            self._time()
            return result
        except BaseException:
            self._fail()
            raise


def install(root: Path, deadline: float, expected: dict[str, Any]) -> type[Any]:
    # Called only after current package/runtime/source verification by fixed launcher.
    global _INSTALLED
    if _INSTALLED is not None:
        raise ValueError("precision installer is once-only")
    model_base = importlib.import_module("comfy.model_base")
    ops = importlib.import_module("comfy.ops")
    quant_ops = importlib.import_module("comfy.quant_ops")
    extension = importlib.import_module("comfy.patcher_extension")
    supported = importlib.import_module("comfy.supported_models")
    nodes = importlib.import_module("nodes")
    torch = importlib.import_module("torch")
    get_executing_context = importlib.import_module("comfy_execution.utils").get_executing_context

    original = nodes.UNETLoader
    if CLASS in nodes.NODE_CLASS_MAPPINGS or nodes.NODE_CLASS_MAPPINGS.get("UNETLoader") is not original:
        raise ValueError("first-party precision node collision or changed upstream loader")
    if Path(inspect.getfile(original)).resolve() != root / "ComfyUI/nodes.py":
        raise ValueError("upstream precision loader origin differs")
    expected = dict(expected, enforcer_sha256=hashlib.sha256(nr_read(Path(__file__))).hexdigest())
    kitchen = importlib.import_module("comfy_kitchen")
    def backend_check() -> None:
        if (os.environ.get("PYTORCH_JIT") != "0" or os.environ.get("TORCH_COMPILE_DISABLE") != "1"
                or bool(torch.jit._state._enabled)):
            raise ValueError("quantized audit requires existing no-JIT/no-compile policy")
        backends = kitchen.list_backends()
        enabled = {name for name, facts in backends.items()
                   if facts["available"] and not facts["disabled"]}
        if not enabled or not enabled <= {"cuda", "eager"}:
            raise ValueError("quantized audit selected an unsupported backend")
        for backend in enabled:
            origin = ("comfy_kitchen.backends.cuda" if backend == "cuda"
                      else "comfy_kitchen.backends.eager.quantization")
            module = importlib.import_module(origin)
            for operation in ("quantize_per_tensor_fp8", "dequantize_per_tensor_fp8"):
                function = kitchen.registry.get_implementation(operation, backend=backend)
                if (function is not getattr(module, operation)
                        or Path(inspect.getfile(function)).resolve() != root / "comfy-runtime/site" /
                        (origin.replace(".", "/") + ("/__init__.py" if backend == "cuda" else ".py"))):
                    raise ValueError("quantized caster backend origin changed")
        nr_check(deadline)
    def module_check(module: Any) -> None:
        if (type(module).__module__ != "comfy.ops"
                or module.layout_type != "TensorCoreFP8E4M3Layout"
                or module.weight._layout_cls != module.layout_type
                or module.weight._qdata.dtype != torch.float8_e4m3fn
                or module.weight.params.scale.dtype != torch.float32
                or tuple(module.weight.params.scale.shape) != ()
                or module._full_precision_mm_config is not False
                or module.weight.layout_cls is not quant_ops.TensorCoreFP8E4M3Layout
                or module.weight.params.orig_dtype not in (torch.float16, torch.bfloat16)
                or type(module._full_precision_mm) is not bool
                or expected["profile_id"] == "m5-wan22-s2v14b-a40-v1" and not module._full_precision_mm
                or Path(inspect.getfile(type(module))).resolve() != root / "ComfyUI/comfy/ops.py"):
            raise ValueError("quantized operation has an unsupported source role")
    backend_check()
    guard = ModelGuard(root, deadline, torch, ops, extension.CallbacksMP,
                       extension.WrappersMP, expected, get_executing_context, backend_check, module_check)

    class CineforgeBF16UNETLoader(original):  # type: ignore[misc, valid-type]
        @classmethod
        def INPUT_TYPES(cls) -> dict[str, Any]:
            data = original.INPUT_TYPES()
            return {**data, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID", "dynprompt": "DYNPROMPT"}}

        @classmethod
        def IS_CHANGED(cls, unet_name: str, weight_dtype: str, prompt: Any,
                       unique_id: str, dynprompt: Any) -> str:
            # Pinned Comfy fingerprints call get_input_data without dynprompt:
            # hidden PROMPT is {} here. Actual graph is checked in load_unet below.
            return guard.fingerprint(unique_id)

        def load_unet(self, unet_name: str, weight_dtype: str, prompt: Any,
                      unique_id: str, dynprompt: Any) -> tuple[Any]:
            request = guard._current()
            guard._context(request, ("161",))
            if (type(prompt) is not dict or any(type(n) is not dict
                    or not {"class_type", "inputs"} <= set(n)
                    or set(n) - {"class_type", "inputs", "is_changed"} for n in prompt.values())):
                raise ValueError("unsupported actual prompt fields")
            projected = {k: {f:n[f] for f in ("class_type", "inputs")} for k,n in prompt.items()}
            if (unique_id != "161" or unet_name != MODEL_NAME or weight_dtype != "default"
                    or nr_digest(projected) != request["nonce_graph_sha256"]):
                raise ValueError("precision loader requires exact current submitted graph")
            if (dynprompt is None or dynprompt.get_original_prompt() is not prompt
                    or dynprompt.node_overrides or dynprompt.ephemeral_prompt):
                raise ValueError("effective precision graph is not the original graph")
            guard.dynamic_prompt = dynprompt
            if guard.base is None:
                base, = original.load_unet(self, unet_name, weight_dtype)
                if (type(base.model.model_config) is not supported.WAN22_S2V
                        or type(base.model) is not model_base.WAN22_S2V):
                    raise ValueError("precision adapter only supports exact Wan S2V model")
                try:
                    return (guard.attach(base),)
                except BaseException:
                    guard._fail()
                    raise
            guard._time()
            return (guard.base.clone(),)

    _INSTALLED = (CineforgeBF16UNETLoader, {n: CineforgeBF16UNETLoader.__dict__[n]
        for n in ("INPUT_TYPES", "IS_CHANGED", "load_unet")})
    nodes.NODE_CLASS_MAPPINGS[CLASS] = CineforgeBF16UNETLoader
    return CineforgeBF16UNETLoader


def installed_nodes() -> dict[str, Any]:
    nodes = importlib.import_module("nodes")
    cls = nodes.NODE_CLASS_MAPPINGS.get(CLASS)
    if (_INSTALLED is None or cls is not _INSTALLED[0]
            or any(cls.__dict__.get(n) is not f for n, f in _INSTALLED[1].items())
            or cls.__module__ != __name__
            or Path(inspect.getfile(cls)).resolve() != Path(__file__).resolve()):
        raise ValueError("first-party precision loader was removed or replaced")
    return FIRST_PARTY_NODES
