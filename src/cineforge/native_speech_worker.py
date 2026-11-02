"""Fixed internal native speech child. Never invoked by the ordinary CLI.

Only the trusted runtime supervisor supplies its private bounded stdin frame.
All SDK output is discarded at file-descriptor level; only a typed result frame
uses the original output descriptor. No resolver, downloader or model retry exists.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Any

from cineforge.native_speech import OriginalVoiceDesign
from cineforge.native_speech_integrity import (
    NativeBinary,
    NativeHardware,
    NativeSpeechError,
    NativeSpeechSelection,
    NativeTree,
    digest,
    verify_tree,
)
from cineforge.native_speech_network import deny_network
from cineforge.native_speech_pcm import Conversion, float_pcm
from cineforge.native_speech_pins import PCM_POLICY
from cineforge.native_speech_process import MAX_INPUT, pack, unpack
from cineforge.speech import QwenBaseRequest


def _selection(value: dict[str, Any]) -> NativeSpeechSelection:
    if set(value) != {"interpreter", "seccomp", "runtime", "sdk", "base", "design", "hardware",
                      "reviewed_subject_sha256", "review_evidence_sha256"}:
        raise NativeSpeechError("invalid native speech selection frame")

    def binary(name: str) -> NativeBinary:
        data = value[name]
        if type(data) is not dict or set(data) != {"path", "size_bytes", "sha256"}:
            raise NativeSpeechError("invalid speech capability frame")
        return NativeBinary(**data)

    def tree(name: str) -> NativeTree:
        data = value[name]
        if type(data) is not dict or set(data) != {"root", "files"} or type(data["files"]) is not list:
            raise NativeSpeechError("invalid speech tree frame")
        return NativeTree(data["root"], tuple(tuple(pin) for pin in data["files"]))

    hardware = value["hardware"]
    if type(hardware) is not dict or set(hardware) != {"operator_worker_id", "gpu_name",
                                                     "memory_mib", "capability"}:
        raise NativeSpeechError("invalid speech hardware frame")
    expected = NativeHardware(hardware["operator_worker_id"], hardware["gpu_name"],
                              hardware["memory_mib"], tuple(hardware["capability"]))
    return NativeSpeechSelection(binary("interpreter"), binary("seccomp"), tree("runtime"),
        tree("sdk"), tree("base"), tree("design"), expected, value["reviewed_subject_sha256"],
        value["review_evidence_sha256"])


def _generate(operation: str, request: QwenBaseRequest | OriginalVoiceDesign,
              reference: bytes, selection: NativeSpeechSelection,
              seed: int) -> tuple[bytes, Conversion, dict[str, Any]]:
    # No site.addsitedir: reviewed .pth files, if present in a distribution, never
    # become startup code. Standard library/system image are operator-trusted.
    sys.path.insert(0, selection.sdk.root)
    sys.path.append(selection.runtime.root)
    torch = importlib.import_module("torch")
    audio = importlib.import_module("torchaudio")
    numpy = importlib.import_module("numpy")
    sdk = importlib.import_module("qwen_tts")
    if (torch.__version__ != "2.8.0+cu128" or audio.__version__ != "2.8.0+cu128"
            or torch.version.cuda != "12.8" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise NativeSpeechError("native speech CUDA capability differs from reviewed runtime")
    properties = torch.cuda.get_device_properties(0)
    actual = {"torch_version": torch.__version__, "torchaudio_version": audio.__version__,
        "cuda_version": torch.version.cuda, "gpu_name": properties.name,
        "memory_bytes": properties.total_memory, "capability": list(torch.cuda.get_device_capability(0)),
        "operator_worker_id": selection.hardware.operator_worker_id, "bf16_supported": True,
        "attention": "sdpa-math"}
    selection.hardware.require_matches(actual)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(seed)
    model = sdk.Qwen3TTSModel.from_pretrained(
        selection.base.root if operation == "base" else selection.design.root,
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    if (model.model.tts_model_type != ("base" if operation == "base" else "voice_design")
            or model.model.tokenizer_type != "qwen3_tts_tokenizer_12hz"):
        raise NativeSpeechError("unsupported native speech model capability")
    for module in (model.model, model.model.speech_tokenizer.model):
        found = False
        for parameter in module.parameters():
            found = True
            if parameter.device.type != "cuda" or parameter.device.index != 0:
                raise NativeSpeechError("native speech cannot offload model parameters to CPU or MPS")
        if not found:
            raise NativeSpeechError("native speech model placement is unavailable")
    with torch.inference_mode():
        if isinstance(request, QwenBaseRequest):
            waveform = numpy.frombuffer(reference, dtype="<i2").astype(numpy.float32) / 32768
            prompt = model.create_voice_clone_prompt(
                ref_audio=(waveform, request.reference_sample_rate), ref_text=request.ref_text,
                x_vector_only_mode=False)
            waves, rate = model.generate_voice_clone(
                text=request.text, language=request.language, voice_clone_prompt=prompt,
                non_streaming_mode=True, max_new_tokens=512)
        else:
            waves, rate = model.generate_voice_design(
                text=request.text, language=request.language, instruct=request.description,
                non_streaming_mode=True, max_new_tokens=512)
    if (type(waves) is not list or len(waves) != 1 or type(rate) is not int or rate != 24000
            or type(waves[0]) is not numpy.ndarray or waves[0].ndim != 1
            or waves[0].dtype not in (numpy.dtype("float32"), numpy.dtype("float64"))
            or not 1 <= waves[0].size <= 60 * rate):
        raise NativeSpeechError("unsupported native speech output layout")
    payload, conversion = float_pcm(waves[0].tolist(), rate)
    if (isinstance(request, QwenBaseRequest)
            and conversion.frames * request.timebase.ticks_per_second
            > request.duration_ticks * conversion.sample_rate):
        raise NativeSpeechError("native speech exceeds reserved duration")
    return payload, conversion, actual


def handle(header: dict[str, Any], reference: bytes) -> tuple[dict[str, Any], bytes]:
    names = {"version", "operation", "request", "selection", "seed", "deadline", "input_sha256"}
    if (set(header) != names or type(header["version"]) is not int or header["version"] != 1
            or header["operation"] not in ("base", "design") or type(header["seed"]) is not int
            or not 0 <= header["seed"] <= 2**32 - 1
            or type(header["deadline"]) not in (int, float)
            or not math.isfinite(header["deadline"])
            or not 0 < header["deadline"] - time.monotonic() <= 1800
            or type(header["selection"]) is not dict or type(header["request"]) is not dict):
        raise NativeSpeechError("invalid native speech invocation")
    selection = _selection(header["selection"])
    request: QwenBaseRequest | OriginalVoiceDesign
    if header["operation"] == "base":
        request = QwenBaseRequest.from_dict(header["request"])
        from cineforge.native_speech_pcm import pcm_wav
        if (len(reference) != request.reference_frames * 2
                or hashlib.sha256(pcm_wav(reference, request.reference_sample_rate)).hexdigest()
                != request.reference.sha256):
            raise NativeSpeechError("native speech reference mismatch")
    else:
        if reference:
            raise NativeSpeechError("voice design never accepts a reference or stock voice")
        request = OriginalVoiceDesign(**header["request"])
    identity = digest({"operation": header["operation"], "request": header["request"],
                       "seed": header["seed"], "selection": selection.subject_sha256(),
                       "reference_pcm_sha256": hashlib.sha256(reference).hexdigest()})
    if identity != header["input_sha256"]:
        raise NativeSpeechError("native speech request binding differs")
    deadline = float(header["deadline"])
    # No third-party imports before the required OS policy succeeds. Its child/exec
    # inheritance also contains the upstream pysox import-time shell probe.
    network = deny_network(selection.seccomp, deadline)
    sdk_content = verify_tree(selection.sdk, deadline)
    runtime_content = verify_tree(selection.runtime, deadline, distributions=True)
    model_content = verify_tree(selection.base if header["operation"] == "base"
                                else selection.design, deadline)
    if time.monotonic() >= deadline:
        raise NativeSpeechError("native speech work deadline exhausted")
    payload, conversion, hardware = _generate(header["operation"], request, reference,
                                               selection, header["seed"])
    if time.monotonic() >= deadline:
        raise NativeSpeechError("native speech work deadline exhausted")
    return {"version": 1, "status": "provisional", "input_sha256": identity,
            "request_sha256": digest(header["request"]), "operation": header["operation"],
            "seed": header["seed"], "max_new_tokens": 512, "non_streaming_mode": True,
            "selection_sha256": selection.subject_sha256(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(), **asdict(conversion),
            "pcm_policy": PCM_POLICY, "network_policy": network,
            "sdk_content_sha256": sdk_content, "runtime_content_sha256": runtime_content,
            "model_content_sha256": model_content, "hardware": hardware,
            "provenance": "unreviewed-native-speech"}, payload


def main() -> int:
    """Only static framed failures leave the child, even if a private SDK traceback exists."""
    output = os.dup(1)
    try:
        null = os.open(os.devnull, os.O_RDWR)
        try:
            os.dup2(null, 1)
            os.dup2(null, 2)
        finally:
            os.close(null)
        raw = sys.stdin.buffer.read(MAX_INPUT + 1)
        header, reference = unpack(raw)
        receipt, payload = handle(header, reference)
        frame = pack(receipt, payload)
        code = 0
    except BaseException:
        frame = pack({"version": 1, "status": "failed"}, b"")
        code = 1
    # Keep this pipe descriptor open until the fixed bootstrap calls os._exit:
    # EOF then means process exit, rather than an untrusted atexit callback.
    view = memoryview(frame)
    while view:
        written = os.write(output, view)
        view = view[written:]
    return code
