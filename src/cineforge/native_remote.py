"""Explicit guarded remote speech bridge. No SDK import, acquisition or default live CLI.

Completed attempt facts are process-owned historical provenance, not permissions,
review signatures, human quality approval or authority for new remote work.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

from cineforge.audio_pcm import PCM16Wave
from cineforge.native_remote_package import native_package
from cineforge.native_remote_plan import (
    NR_BLOB_MAX,
    NR_CHUNK,
    nr_digest,
    nr_encode,
    nr_json,
    nr_recipe,
)
from cineforge.native_speech import NativeSpeechRuntime, OriginalVoiceDesign
from cineforge.native_speech_integrity import NativeSpeechError, NativeSpeechSelection, digest
from cineforge.native_speech_pcm import pcm_wav
from cineforge.native_speech_pins import BASE_FILES, DESIGN_FILES, SDK_FILES
from cineforge.native_speech_process import pack, unpack
from cineforge.s2v import NativeSpeechProof
from cineforge.s2v_service import S2VServiceContext
from cineforge.speech import QwenBaseRequest, SpeechArtifact
from cineforge.ssh_worker import SSHWorker
from cineforge.worker_guard import worker_guard_source_sha256


@dataclass(frozen=True, slots=True)
class RemoteNativeProof:
    """Immutable technical facts of one owned completed attempt, never live admission."""

    attempt: str
    worker_id: str
    session_id: str
    scope_sha256: str
    selection_sha256: str
    source_sha256: str
    package_sha256: str
    recipe_sha256: str
    operation: str
    request_sha256: str
    reference_sha256: str
    payload_sha256: str
    native_receipt_sha256: str
    owned_process_id: int
    cleanup_status: str
    _receipt: bytes = field(repr=False)


class RemoteNativeSpeechRuntime:
    """One consumed setup and at most three attempts on one existing guarded worker.

    Staging facts must be separately reviewed into a typed NativeSpeechSelection;
    returned JSON never grants rights, installed-source approval or native readiness.
    A lost ACK or any incomplete operation poisons new work until owned Pod cleanup.
    """

    def __init__(
        self,
        worker: SSHWorker,
        *,
        allow_native: bool = False,
        timeout_seconds: float = 1200,
        seed: int = 0,
        absolute_deadline: float | None = None,
    ) -> None:
        if (
            type(worker) is not SSHWorker
            or type(worker.selection) is not S2VServiceContext
            or allow_native is not True
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 3 <= timeout_seconds <= 1200
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise NativeSpeechError("explicit existing guarded M5 native bridge required")
        if absolute_deadline is not None and (
            type(absolute_deadline) not in (int, float)
            or not math.isfinite(absolute_deadline)
            or absolute_deadline <= time.monotonic()
        ):
            raise NativeSpeechError("invalid native controller absolute deadline")
        self._parent_deadline = absolute_deadline
        worker.selection.preflight()
        self._worker, self._timeout, self._seed = worker, float(timeout_seconds), seed
        self._deadline: float | None = None
        self._expires_unix: float | None = None
        self._binding: dict[str, Any] | None = None
        self._recipe: str | None = None
        self._installed: dict[str, Any] | None = None
        self._selection: NativeSpeechSelection | None = None
        self._lock = Lock()
        self._staging_attempted = False
        self._unresolved = False
        self._attempts = 0
        self._used: set[str] = set()
        self._proofs: dict[tuple[str, str, str], RemoteNativeProof] = {}
        self.last_attempt: str | None = None

    def _call(self, operation: str, *, controller_deadline: float | None = None,
              **fields: Any) -> dict[str, Any]:
        if self._unresolved or self._deadline is None or self._binding is None:
            raise NativeSpeechError("remote native work is unavailable or unresolved")
        worker = self._worker
        assert isinstance(worker.selection, S2VServiceContext)
        plan = worker.plan
        if (
            plan is None
            or (plan.pod_id, plan.session_id) != (self._binding["worker"], self._binding["session"])
            or worker_guard_source_sha256() != self._binding["guard"]
            or hashlib.sha256(worker._source).hexdigest() != self._binding["source"]
            or worker.selection.approval.subject_sha256 != self._binding["scope"]
        ):
            self._unresolved = True
            raise NativeSpeechError("remote native worker or source scope changed")
        failed = False
        result = None
        try:
            cap = self._deadline
            if controller_deadline is not None:
                if (type(controller_deadline) not in (int, float)
                        or not math.isfinite(controller_deadline)):
                    raise ValueError
                cap = min(cap, controller_deadline)
            if time.monotonic() >= cap:
                raise ValueError
            result = worker.native_request(
                dict(
                    fields,
                    operation=operation,
                    native_source_sha256=self._binding["source"],
                    native_sent_unix=time.time(),
                    native_expires_unix=self._expires_unix,
                ),
                absolute_deadline=cap,
            )
            if time.monotonic() >= cap:
                raise ValueError
        except BaseException:
            failed = True
        if failed or result is None:
            self._unresolved = True
            raise NativeSpeechError(
                "remote native operation incomplete or unconfirmed", cleanup_uncertain=True
            ) from None
        return result

    def _upload(self, role: str, data: bytes, *, token: str | None = None,
                comfy_expires_unix: float | None = None) -> str:
        if type(data) is not bytes or not 0 < len(data) <= NR_BLOB_MAX:
            raise NativeSpeechError("remote native input exceeds fixed bound")
        token = uuid.uuid4().hex if token is None else token
        common: dict[str, Any] = {"role": role, "token": token}
        if comfy_expires_unix is not None:
            if (role != "comfy" or type(comfy_expires_unix) not in (int, float)
                    or not math.isfinite(comfy_expires_unix)):
                raise NativeSpeechError("invalid fixed Comfy transfer deadline")
            common["comfy_expires_unix"] = comfy_expires_unix

        def call(operation: str, **fields: Any) -> None:
            reply = self._call(operation, **common, **fields)
            if reply != {"status": operation, "token": token}:
                self._unresolved = True
                raise NativeSpeechError("remote native transfer acknowledgement differs")

        call("native_begin", size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for offset in range(0, len(data), NR_CHUNK):
            call(
                "native_chunk",
                offset=offset,
                data=base64.b64encode(data[offset : offset + NR_CHUNK]).decode(),
            )
        call("native_finish")
        return token

    def _fetch(self, token: str, descriptor: dict[str, Any]) -> bytes:
        if (
            set(descriptor) != {"size_bytes", "sha256"}
            or type(descriptor["size_bytes"]) is not int
            or not 0 < descriptor["size_bytes"] <= NR_BLOB_MAX
        ):
            raise NativeSpeechError("invalid remote native result descriptor")
        raw = bytearray()
        while len(raw) < descriptor["size_bytes"]:
            reply = self._call("native_fetch", token=token, offset=len(raw))
            if (
                set(reply) != {"offset", "data"}
                or type(reply["offset"]) is not int
                or reply["offset"] != len(raw)
                or type(reply["data"]) is not str
            ):
                raise NativeSpeechError("remote native result framing differs")
            data = base64.b64decode(reply["data"], validate=True)
            if len(data) != min(NR_CHUNK, descriptor["size_bytes"] - len(raw)):
                raise NativeSpeechError("remote native result chunk differs")
            raw.extend(data)
        result = bytes(raw)
        if hashlib.sha256(result).hexdigest() != descriptor["sha256"]:
            raise NativeSpeechError("remote native result identity differs")
        return result

    def stage(self, recipe: dict[str, Any], built_sox_wheel: bytes) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise NativeSpeechError("concurrent remote native work refused")
        try:
            return self._stage(recipe, built_sox_wheel)
        finally:
            self._lock.release()

    def _stage(self, recipe: dict[str, Any], built_sox_wheel: bytes) -> dict[str, Any]:
        """Network staging opt-in. Returns measurements, never a selected execution gate."""
        if self._staging_attempted:
            raise NativeSpeechError("remote native setup already consumed")
        started_deadline = time.monotonic() + self._timeout
        if self._parent_deadline is not None:
            started_deadline = min(started_deadline, self._parent_deadline)
        recipe = nr_recipe(copy.deepcopy(recipe))
        sox = next(w for w in recipe["wheels"] if w["name"] == "sox")
        if (
            type(built_sox_wheel) is not bytes
            or len(built_sox_wheel) != sox["size_bytes"]
            or hashlib.sha256(built_sox_wheel).hexdigest() != sox["sha256"]
        ):
            raise NativeSpeechError("built native wheel differs from reviewed recipe")
        try:
            package, _ = native_package()
        except BaseException:
            raise NativeSpeechError("fixed first party native package unavailable") from None
        worker = self._worker
        worker.native_admission()
        plan = worker.plan
        assert plan is not None
        self._deadline = min(
            started_deadline, time.monotonic() + worker.budget.remaining_seconds(worker._key) - 120
        )
        self._expires_unix = time.time() + self._deadline - time.monotonic()
        assert isinstance(worker.selection, S2VServiceContext)
        self._binding = {
            "worker": plan.pod_id,
            "session": plan.session_id,
            "scope": worker.selection.approval.subject_sha256,
            "guard": worker_guard_source_sha256(),
            "package": hashlib.sha256(package).hexdigest(),
            "source": hashlib.sha256(worker._source).hexdigest(),
            "phase_expires_unix": self._expires_unix,
        }
        self._staging_attempted = True
        try:
            preflight = self._call("native_preflight")
            if (
                set(preflight) != {"interpreter", "seccomp", "abi", "binding_sha256"}
                or preflight["binding_sha256"] != nr_digest(self._binding)
                or preflight["abi"] != "cpython311-linux-x86_64-glibc2.28"
            ):
                raise ValueError
            self._upload("package", package)
            self._upload("recipe", nr_encode(recipe))
            self._upload("sox", built_sox_wheel)
            result = nr_json(self._fetch("installed", self._call("native_setup")))
            if (
                set(result)
                != {"interpreter", "seccomp", "runtime", "sdk", "base", "design", "recipe_sha256"}
                or result["interpreter"] != preflight["interpreter"]
                or result["seccomp"] != preflight["seccomp"]
                or result["recipe_sha256"] != recipe["reviewed_subject_sha256"]
            ):
                raise ValueError
            self._installed = copy.deepcopy(result)
            self._recipe = recipe["reviewed_subject_sha256"]
            if time.monotonic() >= self._deadline:
                raise ValueError
            return copy.deepcopy(result)
        except BaseException:
            self._unresolved = True
            raise NativeSpeechError(
                "remote native staging incomplete or unconfirmed", cleanup_uncertain=True
            ) from None

    def select(self, selection: NativeSpeechSelection) -> None:
        if not self._lock.acquire(blocking=False):
            raise NativeSpeechError("concurrent remote native work refused")
        try:
            self._select(selection)
        finally:
            self._lock.release()

    def _select(self, selection: NativeSpeechSelection) -> None:
        if (
            type(selection) is not NativeSpeechSelection
            or self._installed is None
            or self._selection is not None
        ):
            raise NativeSpeechError("separately reviewed installed native selection required")
        value = asdict(selection)
        for name in ("interpreter", "seccomp", "runtime", "sdk", "base", "design"):
            if nr_encode(value[name]) != nr_encode(self._installed[name]):
                raise NativeSpeechError("native selection differs from actual remote installation")
        if (
            self._binding is None
            or selection.hardware.operator_worker_id != self._binding["worker"]
        ):
            raise NativeSpeechError("native selection differs from owned worker")
        try:
            self._upload("selection", nr_encode(value))
            if self._call("native_select") != {"selection_sha256": selection.subject_sha256()}:
                raise ValueError
            self._selection = selection
        except BaseException:
            self._unresolved = True
            raise NativeSpeechError(
                "remote native selection incomplete or unconfirmed", cleanup_uncertain=True
            ) from None

    def synthesize(self, request: QwenBaseRequest, reference: PCM16Wave) -> bytes:
        if type(request) is not QwenBaseRequest or type(reference) is not PCM16Wave:
            raise NativeSpeechError("typed native speech request and reference required")
        raw = pcm_wav(reference.pcm, reference.sample_rate)
        if (
            reference.channels != 1
            or reference.frames != request.reference_frames
            or reference.sample_rate != request.reference_sample_rate
            or hashlib.sha256(raw).hexdigest() != request.reference.sha256
        ):
            raise NativeSpeechError("native speech reference differs from request")
        return self._invoke(request, raw)

    def preview(self, request: OriginalVoiceDesign) -> bytes:
        if type(request) is not OriginalVoiceDesign:
            raise NativeSpeechError("typed original native voice design required")
        return self._invoke(request, b"")

    def _invoke(self, request: QwenBaseRequest | OriginalVoiceDesign, reference: bytes) -> bytes:
        if not self._lock.acquire(blocking=False):
            raise NativeSpeechError("concurrent remote native work refused")
        try:
            return self._invoke_owned(request, reference)
        finally:
            self._lock.release()

    def _invoke_owned(
        self, request: QwenBaseRequest | OriginalVoiceDesign, reference: bytes
    ) -> bytes:
        if (
            self._selection is None
            or self._binding is None
            or self._recipe is None
            or self._attempts >= 3
        ):
            raise NativeSpeechError("remote native runtime unavailable or capacity exhausted")
        operation = "base" if type(request) is QwenBaseRequest else "design"
        use = "design" if operation == "design" else "base-" + request.language
        if use in self._used:
            raise NativeSpeechError("reviewed native operation already consumed")
        self._used.add(use)
        self._attempts += 1
        token = uuid.uuid4().hex
        self.last_attempt = token
        declaration = request.to_dict() if isinstance(request, QwenBaseRequest) else asdict(request)
        try:
            self._upload(
                "input",
                pack(
                    {"operation": operation, "request": declaration, "seed": self._seed}, reference
                ),
                token=token,
            )
            header, payload = unpack(self._fetch(token, self._call("native_invoke", token=token)))
            if (
                set(header)
                != {
                    "binding",
                    "attempt",
                    "selection",
                    "request",
                    "operation",
                    "reference",
                    "receipt",
                }
                or header["binding"] != self._binding
                or header["attempt"] != token
                or header["selection"] != self._selection.subject_sha256()
                or header["request"] != digest(declaration)
                or header["operation"] != operation
                or header["reference"] != hashlib.sha256(reference).hexdigest()
            ):
                raise ValueError
            receipt = header["receipt"]
            if (
                type(receipt) is not dict
                or receipt.get("owned_child_status") != "exit0-parent-reaped-group-signal-attempted"
                or type(receipt.get("owned_process_id")) is not int
                or receipt["owned_process_id"] <= 0
            ):
                raise ValueError
            wire = {
                k: v
                for k, v in receipt.items()
                if k not in ("owned_child_status", "owned_process_id")
            }
            pcm = reference
            if reference:
                from cineforge.audio_pcm import inspect_pcm16_wav

                pcm = inspect_pcm16_wav(reference).pcm
            identity = digest(
                {
                    "operation": operation,
                    "request": declaration,
                    "seed": self._seed,
                    "selection": self._selection.subject_sha256(),
                    "reference_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                }
            )
            # Use only the pure declaration validator; never populate local Native proofs.
            validator = NativeSpeechRuntime(self._selection, allow_native=True, seed=self._seed)
            validator._validate_receipt(wire, payload, identity, operation, digest(declaration))
            expected_trees = {
                "sdk_content_sha256": SDK_FILES,
                "runtime_content_sha256": self._selection.runtime.files,
                "model_content_sha256": BASE_FILES if operation == "base" else DESIGN_FILES,
            }
            for name, pins in expected_trees.items():
                if wire[name] != digest(sorted((p[0], p[3]) for p in pins)):
                    raise ValueError
            if isinstance(request, QwenBaseRequest):
                from cineforge.audio_pcm import inspect_pcm16_wav

                wave = inspect_pcm16_wav(payload)
                if (
                    wave.frames * request.timebase.ticks_per_second
                    > request.duration_ticks * wave.sample_rate
                ):
                    raise ValueError
            receipt_raw = nr_encode(receipt)
            proof = RemoteNativeProof(
                token,
                self._binding["worker"],
                self._binding["session"],
                self._binding["scope"],
                self._selection.subject_sha256(),
                self._binding["source"],
                self._binding["package"],
                self._recipe,
                operation,
                digest(declaration),
                hashlib.sha256(reference).hexdigest(),
                hashlib.sha256(payload).hexdigest(),
                hashlib.sha256(receipt_raw).hexdigest(),
                receipt["owned_process_id"],
                receipt["owned_child_status"],
                receipt_raw,
            )
            if self._deadline is None or time.monotonic() >= self._deadline:
                raise ValueError
            self._proofs[(operation, proof.request_sha256, proof.payload_sha256)] = proof
            return payload
        except BaseException:
            self._unresolved = True
            raise NativeSpeechError(
                "remote native attempt incomplete or unconfirmed", cleanup_uncertain=True
            ) from None

    def proof_for(
        self, request: QwenBaseRequest | OriginalVoiceDesign, payload: bytes
    ) -> RemoteNativeProof:
        """Offline historical facts only. Does not refresh rights or require a live Pod."""
        if (
            type(request) not in (QwenBaseRequest, OriginalVoiceDesign)
            or type(payload) is not bytes
        ):
            raise NativeSpeechError("typed owned remote result required")
        operation = "base" if type(request) is QwenBaseRequest else "design"
        value = request.to_dict() if isinstance(request, QwenBaseRequest) else asdict(request)
        proof = self._proofs.get((operation, digest(value), hashlib.sha256(payload).hexdigest()))
        if proof is None:
            raise NativeSpeechError("owned remote native proof unavailable")
        return proof

    def receipt_for(
        self, request: QwenBaseRequest | OriginalVoiceDesign, payload: bytes
    ) -> dict[str, Any]:
        return nr_json(self.proof_for(request, payload)._receipt)

    def verify_speech(
        self, speech: SpeechArtifact, service: S2VServiceContext
    ) -> NativeSpeechProof:
        """Trusted S2V callback; actual bytes only, separate from fresh S2V work gates."""
        if type(speech) is not SpeechArtifact or type(service) is not S2VServiceContext:
            raise NativeSpeechError("typed S2V speech and service required")
        service.preflight()
        proof = self.proof_for(speech.request, speech.payload)
        if (
            proof.scope_sha256 != service.approval.subject_sha256
            or proof.reference_sha256 != speech.request.reference.sha256
        ):
            raise NativeSpeechError("owned speech belongs to another S2V scope")
        return NativeSpeechProof(
            proof.request_sha256,
            proof.payload_sha256,
            proof.reference_sha256,
            proof.native_receipt_sha256,
            proof.worker_id,
            proof.scope_sha256,
        )
