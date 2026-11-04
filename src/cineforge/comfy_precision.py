"""Fixed Wan BF16 policy and current-job receipts; no Torch imports or JSON authority."""
from __future__ import annotations

import math
import uuid
from typing import Any

from cineforge.native_remote_plan import nr_digest, nr_hash, nr_token

POLICY = "m5-wan-s2v-bf16-quant-ops-v1"
CLASS = "CineforgeBF16UNETLoader"
MODULE = "cineforge.comfy_model_precision"
HEADER_SHA256 = "20437eddd3fc5d0708da7e52cb40eaa2fc352b61692799a4a6a758461d935acf"  # pragma: allowlist secret - public header hash
MATRICES_SHA256 = "befbf7897e6d3f55193d2a3f8538b125f27541d37a258e805e0c1328843762cf"  # pragma: allowlist secret - public metadata digest
MODEL_SHA256 = "140e75af5534ac3d91e710d9df756f7032addd64b341ba2c1c70e3e6da9aa216"  # pragma: allowlist secret - public model hash
MODEL_NAME = "wan2.2_s2v_14B_fp8_scaled.safetensors"
COUNT = 467
MAX_MATRIX_BYTES = 314572800
MAX_STEPS = 128
FIRST_PARTY_NODES = {CLASS: {"module": MODULE, "policy": POLICY}}
PRECISION = {"policy": POLICY, "manual_dtype": "bf16", "quant_format": "float8_e4m3fn",
             "header_sha256": HEADER_SHA256, "matrices_sha256": MATRICES_SHA256,
             "matrices": COUNT, "max_matrix_bytes": MAX_MATRIX_BYTES}
_HASHES = {"nonce_graph_sha256", "selection_sha256", "comfy_sha256", "runtime_sha256",
           "source_sha256", "package_sha256", "enforcer_sha256", "binding_sha256"}


def precision_request(value: Any) -> dict[str, Any]:
    if (type(value) is not dict or set(value) != _HASHES | {
            "version", "policy", "nonce", "worker_id", "session_id", "profile_id", "deadline"}
            or type(value["version"]) is not int or value["version"] != 1
            or value["policy"] != POLICY):
        raise ValueError("incomplete fixed model precision request")
    for key in _HASHES:
        nr_hash(value[key])
    nr_token(value["nonce"])
    for key in ("worker_id", "session_id", "profile_id"):
        if type(value[key]) is not str or not 1 <= len(value[key]) <= 128:
            raise ValueError("invalid precision producer identity")
    if (type(value["deadline"]) not in (int, float) or not math.isfinite(value["deadline"])
            or value["deadline"] <= 0):
        raise ValueError("invalid worker-local precision cap")
    return value


def precision_receipt(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    precision_request(request)
    if (type(value) is not dict or set(value) != {
            "version", "policy", "request_sha256", "model_sha256", "header_sha256",
            "matrices_sha256", "enforcer_sha256", "model_nonce", "patch_uuid", "audit_count",
            "manual_dtype", "effective_dtype", "step", "matrix_calls", "producer_pid", "prompt_id"}
            or type(value["version"]) is not int or value["version"] != 1
            or value["policy"] != POLICY or value["request_sha256"] != nr_digest(request)
            or value["model_sha256"] != MODEL_SHA256 or value["header_sha256"] != HEADER_SHA256
            or value["matrices_sha256"] != MATRICES_SHA256
            or value["enforcer_sha256"] != request["enforcer_sha256"]
            or value["manual_dtype"] != "bf16" or value["effective_dtype"] != "bf16"
            or type(value["audit_count"]) is not int or value["audit_count"] != COUNT):
        raise ValueError("actual current model precision observation missing or stale")
    if (type(value["prompt_id"]) is not str
            or str(uuid.UUID(value["prompt_id"])) != value["prompt_id"]):
        raise ValueError("actual Comfy prompt identity missing")
    nr_token(value["model_nonce"])
    if (type(value["patch_uuid"]) is not str
            or str(uuid.UUID(value["patch_uuid"])) != value["patch_uuid"]):
        raise ValueError("invalid precision patch identity")
    for key, lower, upper in (("step", 1, MAX_STEPS), ("matrix_calls", 1, COUNT * MAX_STEPS * 4),
                              ("producer_pid", 1, 2**31-1)):
        if type(value[key]) is not int or not lower <= value[key] <= upper:
            raise ValueError("invalid actual precision observation bounds")
    return value
