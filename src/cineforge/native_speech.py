"""Trusted-code native speech opt-in; the ordinary adapter remains fixture-only.

No SDK, resolver, downloader, provisioning, model or native library loads on import
or construction. Real use requires an independently reviewed immutable staging
selection and existing rights/budget/worker admission by the trusted caller.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cineforge.audio_pcm import PCM16Wave, inspect_pcm16_wav
from cineforge.native_speech_integrity import (
    NativeSpeechError,
    NativeSpeechSelection,
    digest,
    verify_binary,
)
from cineforge.native_speech_pcm import pcm_wav
from cineforge.native_speech_pins import NETWORK_POLICY, PCM_POLICY
from cineforge.native_speech_process import OwnedChildResult, pack, run_child, unpack
from cineforge.speech import QwenBaseRequest

_BOOTSTRAP = ("import os,sys;sys.path.insert(0,sys.argv[1]);"
              "from cineforge.native_speech_worker import main;os._exit(main())")


@dataclass(frozen=True, slots=True)
class OriginalVoiceDesign:
    """Operator-selected fictional timbre, not a stock voice or likeness approval.

    Description semantics/originality require operator review; string validation
    cannot prove them. There is no person/reference/speaker ID input to this API.
    """

    text: str = field(repr=False)
    language: str
    description: str = field(repr=False)
    source_review_sha256: str

    def __post_init__(self) -> None:
        for value in (self.text, self.description):
            if (type(value) is not str or not 1 <= len(value) <= 4096 or not value.strip()
                    or any(0xD800 <= ord(c) <= 0xDFFF for c in value)
                    or any(ord(c) < 32 and c not in "\n\r\t" for c in value)):
                raise NativeSpeechError("invalid private voice-design input")
        if type(self.language) is not str or self.language not in ("English", "Chinese"):
            raise NativeSpeechError("unsupported voice-design language")
        from cineforge.native_speech_integrity import _hash
        _hash(self.source_review_sha256)


@dataclass(frozen=True, slots=True)
class NativeAttempt:
    input_sha256: str | None
    outcome: str
    cleanup_uncertain: bool = False
    owned_process_id: int | None = None


class NativeSpeechRuntime:
    """One explicit fresh child per call, with no retries or serializable code selector.

    The technical receipt and returned PCM remain provisional, including a designed
    reference. They never certify human quality, voice identity, consent or export.
    """

    def __init__(self, selection: NativeSpeechSelection, *, allow_native: bool = False,
                 timeout_seconds: float = 300, seed: int = 0) -> None:
        if (type(selection) is not NativeSpeechSelection or allow_native is not True
                or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or not 3 <= timeout_seconds <= 1800 or type(seed) is not int
                or not 0 <= seed <= 2**32 - 1):
            raise NativeSpeechError("explicit reviewed native speech opt-in required")
        self._selection = selection
        self._timeout = float(timeout_seconds)
        self._seed = seed
        self.last_attempt: NativeAttempt | None = None
        self._receipt: dict[str, Any] | None = None
        self._unresolved: NativeAttempt | None = None
        self._proofs: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._success_count = 0

    @property
    def last_receipt(self) -> dict[str, Any] | None:
        # Returned declarations are copies, never mutable execution authority.
        import copy
        return copy.deepcopy(self._receipt)

    def synthesize(self, request: QwenBaseRequest, reference: PCM16Wave) -> bytes:
        self._admit_call()
        self._receipt = None
        self.last_attempt = NativeAttempt(None, "refused")
        if type(request) is not QwenBaseRequest or type(reference) is not PCM16Wave:
            raise NativeSpeechError("typed speech request and reference required")
        failed = False
        declaration = None
        try:
            source = pcm_wav(reference.pcm, reference.sample_rate)
            if (reference.channels != 1 or reference.frames != request.reference_frames
                    or reference.sample_rate != request.reference_sample_rate
                    or hashlib.sha256(source).hexdigest() != request.reference.sha256):
                raise NativeSpeechError("native speech reference differs from request")
            declaration = request.to_dict()
        except BaseException:
            failed = True
        if failed or declaration is None:
            raise NativeSpeechError("native speech reference rejected") from None
        result = self._invoke("base", declaration, reference.pcm)
        wave = inspect_pcm16_wav(result)
        if wave.frames * request.timebase.ticks_per_second > request.duration_ticks * wave.sample_rate:
            self._receipt = None
            self.last_attempt = NativeAttempt(self.last_attempt.input_sha256, "failed")
            raise NativeSpeechError("native speech exceeds reserved duration")
        return result

    def preview(self, design: OriginalVoiceDesign) -> bytes:
        self._admit_call()
        self._receipt = None
        self.last_attempt = NativeAttempt(None, "refused")
        if type(design) is not OriginalVoiceDesign:
            raise NativeSpeechError("explicit original voice-design request required")
        return self._invoke("design", asdict(design), b"")

    def _require_clean(self) -> None:
        if self._unresolved is not None:
            raise NativeSpeechError("prior native speech cleanup remains unresolved",
                cleanup_uncertain=True, owned_process_id=self._unresolved.owned_process_id)

    def _admit_call(self) -> None:
        self._require_clean()
        if self._success_count >= 16:
            raise NativeSpeechError("native speech evidence capacity exhausted")

    def receipt_for(self, request: QwenBaseRequest | OriginalVoiceDesign,
                    payload: bytes) -> dict[str, Any]:
        """Revalidate an owned result, never a caller-supplied serialized receipt.

        This in-process technical evidence has no durable signature or human/rights
        authority. At most16 successes are retained without caching any current
        rights/permission decision. Later generation does not evict earlier facts.
        """
        self._require_clean()
        result = None
        failed = False
        try:
            if type(request) not in (QwenBaseRequest, OriginalVoiceDesign):
                raise NativeSpeechError("typed native receipt input required")
            declaration = request.to_dict() if isinstance(request, QwenBaseRequest) else asdict(request)
            operation = "base" if isinstance(request, QwenBaseRequest) else "design"
            key = (operation, digest(declaration), hashlib.sha256(payload).hexdigest())
            receipt = self._proofs.get(key)
            if (receipt is None or receipt.get("owned_child_status")
                    != "exit0-parent-reaped-group-signal-attempted"):
                raise NativeSpeechError("owned native receipt is unavailable")
            wire = {key: value for key, value in receipt.items()
                    if key not in ("owned_child_status", "owned_process_id")}
            self._validate_receipt(wire, payload, str(receipt["input_sha256"]),
                                   operation, digest(declaration))
            import copy
            result = copy.deepcopy(receipt)
        except BaseException:
            failed = True
        if failed or result is None:
            raise NativeSpeechError("native receipt does not match actual request and bytes") from None
        return result

    def _invoke(self, operation: str, request: dict[str, Any], reference: bytes) -> bytes:
        self._receipt = None
        identity = None
        self.last_attempt = NativeAttempt(None, "preparing")
        deadline = time.monotonic() + self._timeout
        failed = cancelled = uncertain = False
        process_id = None
        result = None
        try:
            identity = digest({"operation": operation, "request": request, "seed": self._seed,
                               "selection": self._selection.subject_sha256(),
                               "reference_pcm_sha256": hashlib.sha256(reference).hexdigest()})
            self.last_attempt = NativeAttempt(identity, "prepared")
            verify_binary(self._selection.interpreter, deadline - 2)
            with tempfile.TemporaryDirectory(prefix="cineforge-native-speech-") as directory:
                # No inherited settings, credentials, site/.pth hooks or executable
                # search path. pysox's probe cannot find an unpinned system SoX.
                empty_path = os.path.join(directory, "empty-path")
                os.mkdir(empty_path, 0o700)
                environment = {"PATH": empty_path, "HOME": directory, "LANG": "C",
                    "LC_ALL": "C", "HF_HOME": directory, "HF_HUB_CACHE": directory,
                    "TRANSFORMERS_CACHE": directory, "XDG_CACHE_HOME": directory,
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                    "TOKENIZERS_PARALLELISM": "false", "TORCH_COMPILE_DISABLE": "1",
                    "CUDA_VISIBLE_DEVICES": "0", "NUMBA_CACHE_DIR": directory,
                    "TRITON_CACHE_DIR": directory, "TORCHINDUCTOR_CACHE_DIR": directory}
                header = {"version": 1, "operation": operation, "request": request,
                    "selection": asdict(self._selection), "seed": self._seed,
                    "deadline": deadline - 2, "input_sha256": identity}
                package_root = str(Path(__file__).resolve().parent.parent)
                command = (self._selection.interpreter.path, "-I", "-B", "-S", "-c",
                           _BOOTSTRAP, package_root)
                child = run_child(command, payload=pack(header, reference), env=environment,
                                  cwd=directory, deadline=deadline)
                if type(child) is not OwnedChildResult:
                    raise NativeSpeechError("owned native speech result is missing")
                process_id = child.process_id
                receipt, result = unpack(child.payload)
                self._validate_receipt(receipt, result, identity, operation, digest(request))
                if operation == "base":
                    current = QwenBaseRequest.from_dict(request)
                    wave = inspect_pcm16_wav(result)
                    if (wave.frames * current.timebase.ticks_per_second
                            > current.duration_ticks * wave.sample_rate):
                        raise NativeSpeechError("native speech exceeds reserved duration")
                self._receipt = dict(receipt, owned_process_id=process_id,
                    owned_child_status="exit0-parent-reaped-group-signal-attempted")
        except KeyboardInterrupt:
            cancelled = True
        except BaseException as error:
            uncertain = isinstance(error, NativeSpeechError) and error.cleanup_uncertain
            # A late filesystem cleanup failure after the owned child returned is
            # also unresolved cleanup, rather than an apparently reusable runtime.
            uncertain = uncertain or isinstance(error, OSError) and process_id is not None
            if isinstance(error, NativeSpeechError) and error.owned_process_id is not None:
                process_id = error.owned_process_id
            failed = True
        if cancelled:
            self._receipt = None
            self.last_attempt = NativeAttempt(identity, "interrupted", owned_process_id=process_id)
            raise KeyboardInterrupt from None
        if failed or result is None:
            self._receipt = None
            self.last_attempt = NativeAttempt(identity, "failed", uncertain, process_id)
            if uncertain:
                self._unresolved = self.last_attempt
            raise NativeSpeechError("native speech attempt failed", cleanup_uncertain=uncertain,
                                    owned_process_id=process_id) \
                from None
        self.last_attempt = NativeAttempt(identity, "provisional", owned_process_id=process_id)
        self._success_count += 1
        if self._receipt is not None:
            key = (operation, digest(request), hashlib.sha256(result).hexdigest())
            import copy
            self._proofs[key] = copy.deepcopy(self._receipt)
        return result

    def _validate_receipt(self, receipt: dict[str, Any], payload: bytes, identity: str,
                          operation: str, request_sha256: str) -> None:
        names = {"version", "status", "input_sha256", "selection_sha256", "payload_sha256",
                 "frames", "sample_rate", "float_peak", "positive_endpoint_saturations",
                 "pcm_policy", "network_policy", "sdk_content_sha256", "runtime_content_sha256",
                 "model_content_sha256", "provenance", "request_sha256", "operation", "hardware",
                 "seed", "max_new_tokens", "non_streaming_mode"}
        wave = inspect_pcm16_wav(payload)
        if (set(receipt) != names or type(receipt["version"]) is not int or receipt["version"] != 1
                or receipt["status"] != "provisional" or receipt["input_sha256"] != identity
                or receipt["operation"] != operation or receipt["request_sha256"] != request_sha256
                or type(receipt["seed"]) is not int or receipt["seed"] != self._seed
                or type(receipt["max_new_tokens"]) is not int or receipt["max_new_tokens"] != 512
                or receipt["non_streaming_mode"] is not True
                or receipt["selection_sha256"] != self._selection.subject_sha256()
                or receipt["payload_sha256"] != hashlib.sha256(payload).hexdigest()
                or type(receipt["frames"]) is not int or receipt["frames"] != wave.frames
                or type(receipt["sample_rate"]) is not int
                or receipt["sample_rate"] != wave.sample_rate or wave.channels != 1
                or wave.sample_rate != 24000
                or type(receipt["float_peak"]) not in (int, float)
                or not math.isfinite(receipt["float_peak"]) or not 0 <= receipt["float_peak"] <= 1
                or type(receipt["positive_endpoint_saturations"]) is not int
                or not 0 <= receipt["positive_endpoint_saturations"] <= wave.frames
                or receipt["pcm_policy"] != PCM_POLICY or receipt["network_policy"] != NETWORK_POLICY
                or receipt["provenance"] != "unreviewed-native-speech"):
            raise NativeSpeechError("native speech result or provenance rejected")
        from cineforge.native_speech_integrity import _hash
        for name in ("sdk_content_sha256", "runtime_content_sha256", "model_content_sha256"):
            _hash(receipt[name])
        if type(receipt["hardware"]) is not dict:
            raise NativeSpeechError("native speech hardware observation is missing")
        self._selection.hardware.require_matches(receipt["hardware"])
