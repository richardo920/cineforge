"""Fixed guarded begin/finish only; worker-local caps and consumed historical evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from cineforge.comfy_precision import precision_receipt, precision_request
from cineforge.native_remote_bootstrap import nr_check, nr_read, nr_write
from cineforge.native_remote_plan import nr_digest, nr_encode, nr_json, nr_token


def precision_operation(request: dict[str, Any], native: Path, deadline: float,
                        binding: dict[str, Any]) -> dict[str, Any]:
    from cineforge.comfy_runtime_plan import comfy_plan, comfy_subject
    from cineforge.comfy_runtime_worker import current_comfy
    from cineforge.worker_bootstrap import worker_root
    current_comfy(request, native, deadline)
    root = worker_root(request["plan"]["session_id"])
    directory = root / "comfy-precision"
    nonce = request["precision_nonce"]
    nr_token(nonce)
    path = directory / (nonce + ".request.json")
    plan = comfy_plan(nr_json(nr_read(root / "comfy-selection.json")))
    producer = dict(package_sha256=binding["package"], binding_sha256=nr_digest(binding),
        worker_id=binding["worker"], session_id=binding["session"],
        profile_id=request["profile"]["id"], comfy_sha256=comfy_subject(plan),
        runtime_sha256=nr_digest(plan["installed"]), source_sha256=nr_digest(plan["source"]),
        enforcer_sha256=hashlib.sha256(nr_read(
            native / "firstparty/cineforge/comfy_model_precision.py")).hexdigest())
    if request["operation"] == "native_comfy_precision_begin":
        entries = list(directory.glob("*.request.json"))
        if len(entries) >= 2 or any(not p.with_name(p.name[:-13] + ".finished").exists()
                                    for p in entries):
            raise ValueError("precision request already active or exhausted")
        value = precision_request(dict(request["precision_binding"], nonce=nonce,
            deadline=deadline, **producer))
        nr_check(deadline)
        nr_write(path, nr_encode(value))  # Never overwrite a consumed/lost-ACK nonce.
        nr_check(deadline)
        return value
    value = precision_request(nr_json(nr_read(path, maximum=8192)))
    if any(value[key] != observed for key, observed in producer.items()):
        raise ValueError("precision producer identity is stale")
    cap = min(deadline, value["deadline"])
    nr_check(cap)
    nr_write(directory / (nonce + ".finish-consumed"), b"consumed before receipt validation")
    if (directory / (nonce + ".failed")).exists():
        raise ValueError("actual model precision enforcement failed")
    audit = nr_json(nr_read(directory / (nonce + ".audit.json"), maximum=8192))
    if type(audit.get("step")) is not int or audit["step"] != 0 or audit.get("matrix_calls") != 0:
        raise ValueError("actual pre-inference audit phase missing")
    steps = sorted(directory.glob(nonce + ".step-*.json"))
    if not 1 <= len(steps) <= 128 or [p.name for p in steps] != [
            nonce + ".step-" + str(i).zfill(3) + ".json" for i in range(1, len(steps)+1)]:
        raise ValueError("actual precision forward observations missing")
    pid = nr_json(nr_read(root / "server-pid.json"))["pid"]
    previous = 0
    for index, step in enumerate(steps, 1):
        nr_check(cap)
        receipt = precision_receipt(nr_json(nr_read(step, maximum=8192)), value)
        if (receipt["step"] != index or receipt["producer_pid"] != pid
                or receipt["prompt_id"] != request["precision_prompt_id"]
                or receipt["matrix_calls"] <= previous
                or {k:v for k,v in receipt.items() if k not in ("step", "matrix_calls")}
                != {k:v for k,v in audit.items() if k not in ("step", "matrix_calls")}):
            raise ValueError("precision producer or actual job changed")
        previous = receipt["matrix_calls"]
    closed = precision_receipt(nr_json(nr_read(directory / (nonce + ".closed.json"), maximum=8192)), value)
    if closed != receipt:
        raise ValueError("actual model precision cleanup observation missing")
    current_comfy(request, native, cap)
    nr_check(cap)
    nr_write(directory / (nonce + ".finished"), nr_encode(receipt))
    nr_check(cap)
    return receipt
