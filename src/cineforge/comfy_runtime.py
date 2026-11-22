"""Trusted M5 Comfy closure review and fixed remote setup; no default execution."""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cineforge.comfy_precision import POLICY, precision_receipt, precision_request
from cineforge.comfy_runtime_plan import comfy_plan, comfy_subject, comfy_worker
from cineforge.interruptions import preserve_interruption, remember_interruption
from cineforge.native_remote_plan import NR_BLOB_MAX, nr_digest, nr_encode, nr_hash, nr_json
from cineforge.native_speech_integrity import NativeSpeechError
from cineforge.s2v_service import LongSessionServiceContext

VERIFY_SUB_STEPS = frozenset({
    "not_prepared", "entry_admit", "entry_require", "verify_rpc",
    "fact_check", "exit_admit", "exit_require", "deadline"})
VERIFY_FACT_STEPS = frozenset({"fact_shape", "fact_closure"})


def _merge_verify_cause(target: dict[str, Any], source: Any) -> None:
    if type(source) is not dict:
        return
    for key in ("cause_type", "reason", "supervision_message"):
        value = source.get(key)
        if type(value) is str:
            target[key] = value
    returncode = source.get("returncode")
    if type(returncode) is int:
        target["returncode"] = returncode
    tail = source.get("child_output_tail")
    if (type(tail) is str and 0 < len(tail) <= 512
            and all(32 <= ord(character) <= 126 for character in tail)):
        target["child_output_tail"] = tail


def _verify_failure_diagnostic(sub_step: str, error: BaseException, *,
                               fact_step: Any = None,
                               call_failure: Any = None) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"sub_step": sub_step,
                                  "error_type": type(error).__name__[:64]}
    if type(fact_step) is str and fact_step in VERIFY_FACT_STEPS:
        diagnostic["fact_step"] = fact_step
    _merge_verify_cause(diagnostic, getattr(error, "cause_diagnostic", None))
    if ("cause_type" not in diagnostic and type(call_failure) is dict
            and call_failure.get("operation") == "native_comfy_verify"):
        _merge_verify_cause(diagnostic, call_failure)
    return diagnostic


def _finish_diagnostic(stage: str, reply: Any) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"stage": stage}
    if stage == "receipt_validation":
        if type(reply) is not dict:
            diagnostic["receipt_type"] = type(reply).__name__[:128]
        else:
            diagnostic["receipt_keys"] = sorted(k[:64] for k in reply if type(k) is str)[:32]
            receipt: dict[str, Any] = {}
            for key in ("model_nonce", "request_sha256", "model_sha256", "header_sha256",
                        "matrices_sha256", "enforcer_sha256", "audit_count", "prompt_id",
                        "manual_dtype", "effective_dtype", "version", "policy", "patch_uuid",
                        "step", "matrix_calls", "producer_pid"):
                if key not in reply:
                    continue
                value = reply[key]
                if type(value) is str:
                    receipt[key] = value[:256].encode("utf-8", "replace").decode("utf-8")
                elif (value is None or type(value) is bool
                      or type(value) is int and value.bit_length() <= 64
                      or type(value) is float and math.isfinite(value)):
                    receipt[key] = value
            diagnostic["receipt"] = receipt
    return diagnostic


@dataclass(frozen=True, slots=True)
class ComfyClosureReview:
    """Trusted reviewer assertion, not authentication or a metadata-only ABI measurement."""
    subject_sha256: str
    artifacts_sha256: str
    dependencies_sha256: str
    licenses_sha256: str
    advisories_sha256: str
    abi_sha256: str

    def require(self, plan: dict[str, Any]) -> None:
        for value in (self.subject_sha256, self.artifacts_sha256, self.dependencies_sha256,
                      self.licenses_sha256, self.advisories_sha256, self.abi_sha256):
            nr_hash(value)
        if (self.subject_sha256 != comfy_subject(plan)
                or {name: getattr(self, name + "_sha256") for name in plan["review"]}
                != plan["review"]):
            raise ValueError("closed Comfy artifact review is stale")


class ComfyClosureVerifier(Protocol):
    """Separately verify actual artifacts/manifests, full dependencies, notices and ABI.

    A resolver lock or source label cannot implement this review. No default verifier
    exists; authenticated operator selection remains outside these metadata contracts.
    """
    def verify(self, plan: dict[str, Any]) -> ComfyClosureReview: ...


_CHECKED = object()


@dataclass(frozen=True, slots=True, init=False)
class CheckedComfyRuntime:
    _declaration: bytes = field(repr=False)
    _verifier: ComfyClosureVerifier = field(repr=False)
    _checked: object = field(repr=False)

    @classmethod
    @preserve_interruption
    def reviewed(cls, plan: dict[str, Any], verifier: ComfyClosureVerifier) -> CheckedComfyRuntime:
        raw = None
        try:
            encoded = nr_encode(comfy_plan(plan))
            if not 1 <= len(encoded) <= NR_BLOB_MAX:
                raise ValueError
            raw = encoded
        except BaseException as interruption_error:
            remember_interruption(interruption_error)
            pass
        if raw is None:
            raise ValueError("invalid closed Comfy declaration")
        result = object.__new__(cls)
        object.__setattr__(result, "_declaration", raw)
        object.__setattr__(result, "_verifier", verifier)
        object.__setattr__(result, "_checked", _CHECKED)
        result.current()
        return result

    @preserve_interruption
    def current(self) -> dict[str, Any]:
        valid = None
        try:
            if self._checked is not _CHECKED:
                raise ValueError
            plan = comfy_plan(nr_json(self._declaration))
            proof = self._verifier.verify(nr_json(self._declaration))
            if type(proof) is not ComfyClosureReview:
                raise ValueError
            proof.require(plan)
            valid = plan
        except BaseException as interruption_error:
            remember_interruption(interruption_error)
            pass
        if valid is None:
            raise ValueError("current independently checked Comfy closure required")
        return valid


class ClosedComfyRuntime:
    """Use an existing remote108 package channel, never its native108 runtime target.

    Both channels retain their original controller/worker caps. Setup is consumed
    before transfer; uncertainty poisons this object and the bridge, never a retry.
    """
    def __init__(self, bridge: Any, checked: CheckedComfyRuntime) -> None:
        from cineforge.native_remote import RemoteNativeSpeechRuntime
        if type(bridge) is not RemoteNativeSpeechRuntime or type(checked) is not CheckedComfyRuntime:
            raise ValueError("fixed reviewed M5 Comfy bridge required")
        checked.current()
        self._bridge, self._checked = bridge, checked
        self._attempted = self._unresolved = False
        self._prepared = False
        self._subject: str | None = None
        self._expires_unix: float | None = None
        self._precision: dict[str, Any] | None = None
        self._precision_used: set[str] = set()
        self._precision_posted = False
        self._precision_finish_diagnostic: dict[str, Any] | None = None
        self._verify_diagnostic: dict[str, Any] | None = None

    def _poison(self) -> None:
        self._unresolved = True
        self._bridge._unresolved = True

    def _admit(self, worker: Any, deadline: float) -> dict[str, Any]:
        if (self._unresolved or self._bridge._worker is not worker
                or self._bridge._deadline is None or self._bridge._expires_unix is None
                or self._bridge._binding is None or time.monotonic() >= deadline):
            raise ValueError("closed Comfy runtime is unavailable")
        worker.native_admission()
        plan = self._checked.current()
        installed = self._bridge._installed
        if (comfy_worker(plan).profile_id != worker.selection.worker_profile["id"]
                or plan["image"] != worker.spec.image or type(installed) is not dict
                or plan["interpreter"] != installed.get("interpreter")):
            raise ValueError("closed Comfy interpreter differs from actual worker admission")
        for binary in plan["native"]:
            if "driver_mount" in binary:
                if binary["driver_mount"]["binding_sha256"] != nr_digest(self._bridge._binding):
                    raise ValueError("NVIDIA mount belongs to another native worker binding")
                captured = self._bridge._capabilities_payload
                if captured is None:
                    raise ValueError("NVIDIA declaration has no actual consumed capability measurement")
                facts = nr_json(captured)
                if (nr_digest(facts) != self._bridge._capabilities_sha256
                        or not any({key: row[key] for key in binary if key in row} == binary
                                   for row in facts["files"] if row.get("status") == "measured")):
                    raise ValueError("NVIDIA declaration differs from consumed capability measurement")
        return plan

    def _call(self, operation: str, deadline: float, **fields: Any) -> dict[str, Any]:
        if self._bridge._deadline is None or self._expires_unix is None:
            raise AssertionError
        cap = min(deadline, self._bridge._deadline)
        return self._bridge._call(operation, controller_deadline=cap,
            comfy_expires_unix=self._expires_unix,
            comfy_call_expires_unix=min(self._expires_unix, time.time() + cap - time.monotonic()),
            **fields)

    @preserve_interruption
    def prepare(self, worker: Any, selection: Any, *, deadline: float) -> dict[str, Any]:
        if self._attempted:
            raise ValueError("closed Comfy setup already consumed")
        plan = self._admit(worker, deadline)
        worker._require_s2v(selection)
        self._attempted = True
        if self._bridge._deadline is None or self._bridge._expires_unix is None:
            raise AssertionError
        if type(worker.selection) is not LongSessionServiceContext:
            self._bridge._deadline = min(deadline, self._bridge._deadline)
        self._expires_unix = min(self._bridge._expires_unix,
                                time.time() + self._bridge._deadline - time.monotonic())
        try:
            self._bridge._upload("comfy", nr_encode(plan), comfy_expires_unix=self._expires_unix)
            worker._require_s2v(selection)
            common = dict(comfy_subject_sha256=comfy_subject(plan), selection=selection.declarations,
                          image_digest=worker.spec.image.split("@", 1)[1])
            self._call("native_comfy_install", deadline, **common)
            self._admit(worker, deadline)
            worker._require_s2v(selection)
            for index, asset in enumerate(selection.declarations["model"]["files"]):
                self._admit(worker, deadline)
                worker._require_s2v(selection)
                received = self._call("native_comfy_model", deadline, model_index=index, **common)
                if received != {key: asset[key] for key in ("path", "size_bytes", "sha256")}:
                    raise ValueError("closed Comfy model transfer identity differs")
                worker._require_s2v(selection)
            self._admit(worker, deadline)
            worker._require_s2v(selection)
            reply = self._call("native_comfy_setup", deadline, **common)
            self._admit(worker, deadline)
            worker._require_s2v(selection)
            if set(reply) != {"verification", "runtime", "comfy"}:
                raise ValueError
            self._subject = comfy_subject(plan)
            self._check_fact(reply["comfy"])
            self._admit(worker, deadline)
            worker._require_s2v(selection)
            if time.monotonic() >= deadline:
                raise ValueError
            self._prepared = True
            return {"verification": reply["verification"], "runtime": reply["runtime"]}
        except BaseException as interruption_error:
            remember_interruption(interruption_error)
            self._poison()
            raise ValueError("closed Comfy setup incomplete or unconfirmed; never replay") from None

    def _check_fact(self, fact: Any, *, diagnostic: dict[str, Any] | None = None) -> None:
        if diagnostic is not None:
            diagnostic["fact_step"] = "fact_shape"
        if (type(fact) is not dict or set(fact) != {"subject_sha256", "installed_sha256", "probe_sha256",
                                                 "execution_sha256", "jit_enabled", "phase"}
                or fact["subject_sha256"] != self._subject or fact["jit_enabled"] is not False
                or fact["phase"] != "after_core_startup"):
            raise ValueError("closed Comfy worker facts differ")
        for name in ("subject_sha256", "installed_sha256", "probe_sha256", "execution_sha256"):
            nr_hash(fact[name])
        plan = self._checked.current()
        if diagnostic is not None:
            diagnostic["fact_step"] = "fact_closure"
        if (fact["installed_sha256"] != hashlib.sha256(nr_encode(plan["installed"])).hexdigest()
                or fact["execution_sha256"] != hashlib.sha256(nr_encode(plan["execution"])).hexdigest()):
            raise ValueError("closed Comfy installed closure differs")

    @preserve_interruption
    def verify(self, worker: Any, selection: Any, *, deadline: float) -> None:
        self._verify_diagnostic = None
        if not self._prepared:
            error = ValueError("closed Comfy preparation required")
            self._verify_diagnostic = {"sub_step": "not_prepared", "error_type": "ValueError"}
            raise error
        sub_step = "entry_admit"
        try:
            plan = self._admit(worker, deadline)
            sub_step = "entry_require"
            worker._require_s2v(selection)
        except BaseException as interruption_error:
            try:
                self._verify_diagnostic = _verify_failure_diagnostic(sub_step, interruption_error)
            except BaseException as diagnostic_error:
                remember_interruption(diagnostic_error)
            raise
        fact_diagnostic: dict[str, Any] = {}
        sub_step = "verify_rpc"
        try:
            reply = self._call("native_comfy_verify", deadline, comfy_subject_sha256=comfy_subject(plan))
            sub_step = "fact_check"
            self._check_fact(reply, diagnostic=fact_diagnostic)
            sub_step = "exit_admit"
            self._admit(worker, deadline)
            sub_step = "exit_require"
            worker._require_s2v(selection)
            sub_step = "deadline"
            if time.monotonic() >= deadline:
                raise ValueError
        except BaseException as interruption_error:
            try:
                call_failure = None
                if sub_step == "verify_rpc" and type(interruption_error) is NativeSpeechError:
                    call_failure = getattr(self._bridge, "_last_call_failure", None)
                self._verify_diagnostic = _verify_failure_diagnostic(
                    sub_step, interruption_error,
                    fact_step=fact_diagnostic.get("fact_step") if sub_step == "fact_check" else None,
                    call_failure=call_failure)
            except BaseException as diagnostic_error:
                remember_interruption(diagnostic_error)
            remember_interruption(interruption_error)
            self._poison()
            raise ValueError("closed Comfy current verification failed") from None


    @preserve_interruption
    def begin_precision(self, worker: Any, selection: Any, *, deadline: float) -> dict[str, Any]:
        """Consume one exact active intent before POST; ACK loss poisons, never replay."""
        from cineforge.native_remote_package import native_package
        from cineforge.native_remote_plan import nr_digest
        from cineforge.private_store import read_private_input
        from cineforge.workflows import strict_object
        long_session = type(worker.selection) is LongSessionServiceContext
        limit = self._bridge._max_operations - 1 if long_session else 2
        if not self._prepared or self._precision is not None or len(self._precision_used) >= limit:
            raise ValueError("current precision intent unavailable or consumed")
        plan = self._admit(worker, deadline)
        worker._require_s2v(selection)
        language = selection.gate.utterance.language
        use = selection.approved_subject_sha256 if long_session else language
        if language not in ("en", "zh") or use in self._precision_used:
            raise ValueError("language precision intent already consumed")
        self._precision_used.add(use)
        nonce = uuid.uuid4().hex
        package, _ = native_package()
        if self._bridge._binding is None:
            raise AssertionError
        expected = dict(version=1, policy=POLICY, nonce=nonce,
            nonce_graph_sha256=nr_digest(strict_object(selection.graph_json)),
            selection_sha256=selection.approved_subject_sha256,
            comfy_sha256=comfy_subject(plan), runtime_sha256=nr_digest(plan["installed"]),
            source_sha256=nr_digest(plan["source"]), package_sha256=hashlib.sha256(package).hexdigest(),
            binding_sha256=nr_digest(self._bridge._binding), worker_id=self._bridge._binding["worker"],
            session_id=self._bridge._binding["session"], profile_id=worker.selection.worker_profile["id"],
            enforcer_sha256=hashlib.sha256(read_private_input(
                Path(__file__).with_name("comfy_model_precision.py"), max_bytes=256*1024)).hexdigest())
        try:
            reply = precision_request(self._call("native_comfy_precision_begin", deadline,
                comfy_subject_sha256=comfy_subject(plan), precision_nonce=nonce,
                precision_binding={k: expected[k] for k in
                    ("version", "policy", "nonce_graph_sha256", "selection_sha256")}))
            if {k:v for k,v in reply.items() if k != "deadline"} != expected:
                raise ValueError
            self._admit(worker, deadline)
            worker._require_s2v(selection)
            if time.monotonic() >= deadline:
                raise ValueError
            self._precision = reply
            self._precision_posted = False
            return dict(reply)
        except BaseException as interruption_error:
            remember_interruption(interruption_error)
            self._poison()
            raise ValueError("actual precision begin incomplete; never replay") from None

    @preserve_interruption
    def finish_precision(self, worker: Any, selection: Any, prompt_id: str,
                         *, deadline: float) -> dict[str, Any]:
        self._precision_finish_diagnostic = None
        if self._precision is None or not self._precision_posted:
            self._precision_finish_diagnostic = {"stage": "intent_guard"}
            raise ValueError("current owned precision intent required")
        request, self._precision = self._precision, None  # Consume before fallible validation/I/O.
        stage, reply = "entry_admit", None
        try:
            plan = self._admit(worker, deadline)
            worker._require_s2v(selection)
            stage = "selection_identity"
            if request["selection_sha256"] != selection.approved_subject_sha256:
                raise ValueError
            stage = "finish_rpc"
            reply = self._call("native_comfy_precision_finish", deadline,
                comfy_subject_sha256=comfy_subject(plan), precision_nonce=request["nonce"],
                precision_prompt_id=prompt_id)
            stage = "receipt_validation"
            reply = precision_receipt(reply, request)
            stage = "prompt_identity"
            if reply["prompt_id"] != prompt_id:
                raise ValueError
            stage = "exit_admit"
            self._admit(worker, deadline)
            worker._require_s2v(selection)
            stage = "deadline"
            if time.monotonic() >= deadline:
                raise ValueError
            return reply
        except BaseException as interruption_error:
            # Diagnostics must never replace cancellation or the original failure path.
            try:
                diagnostic = _finish_diagnostic(stage, reply)
                diagnostic["error_type"] = type(interruption_error).__name__[:64]
                if stage == "prompt_identity" and reply is not None:
                    diagnostic.update(received_prompt_id=reply["prompt_id"],
                                      expected_prompt_id=prompt_id)
                self._precision_finish_diagnostic = diagnostic
            except BaseException as diagnostic_error:
                remember_interruption(diagnostic_error)
            remember_interruption(interruption_error)
            self._poison()
            raise ValueError("actual model precision missing, failed or stale; never replay") from None


    def require_precision_post(self, worker: Any, body: Any, *, deadline: float) -> None:
        from cineforge.native_remote_plan import nr_digest
        if self._precision is None or self._precision_posted:
            raise ValueError("active current precision intent required before POST")
        self._precision_posted = True  # One attempted POST, including preflight/ACK uncertainty.
        self._admit(worker, deadline)
        selection = worker._comfy_selection
        worker._require_s2v(selection)
        if (type(body) is not dict or set(body) != {"prompt"} or type(body["prompt"]) is not dict
                or any(type(n) is not dict or set(n) != {"class_type", "inputs"}
                       for n in body["prompt"].values())
                or nr_digest(body["prompt"]) != self._precision["nonce_graph_sha256"]
                or selection.approved_subject_sha256 != self._precision["selection_sha256"]
                or time.monotonic() >= deadline):
            raise ValueError("POST differs from current exact precision request")


    def abandon_precision(self) -> None:
        """Any incomplete current attempt poisons new native/Comfy work, never cleanup."""
        self._poison()
