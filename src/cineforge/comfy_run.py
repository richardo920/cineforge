"""Explicitly opted-in, bounded single-shot execution and private local persistence."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cineforge.comfy import TERMINAL, ComfyExecutionProvider, JobDeadlineError
from cineforge.comfy_policy import ApprovedComfySelection, WorkerVerification
from cineforge.comfy_transport import (
    ComfyHTTPTransport,
    ComfyTransport,
    TransportError,
    positive_seconds,
)
from cineforge.contracts import ProjectPlan, RunManifest, RunStatus, ShotRun
from cineforge.private_store import PrivateRunDirectory
from cineforge.workflows import canonical, strict_object


class ComfyRunError(ValueError):
    """Sanitized failure; a claimed run directory retains the last known job state."""


class _BoundedTransport:
    def __init__(self, transport: ComfyTransport, deadline: float,
                 clock: Callable[[], float]) -> None:
        self.transport, self.deadline, self.clock = transport, deadline, clock

    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int) -> bytes:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise JobDeadlineError("single-shot deadline exceeded")
        failed = False
        raw = b""
        try:
            raw = self.transport.request(method, path, body,
                                         timeout_seconds=min(remaining, timeout_seconds),
                                         max_bytes=max_bytes)
        except Exception:
            failed = True
        # Raise outside the exception handler so neither traceback context nor cause
        # retains arbitrary transport/HTTP exception text containing credentials.
        if failed:
            raise TransportError("ComfyUI transport request failed")
        if self.clock() >= self.deadline:
            raise JobDeadlineError("single-shot deadline exceeded")
        return raw


def _verify_nodes(transport: ComfyTransport, selection: ApprovedComfySelection,
                  timeout_seconds: float) -> None:
    graph = strict_object(selection.graph_json)
    schemas = {}
    for kind in sorted(selection.workflow.allowed_classes):
        response = strict_object(transport.request("GET", "/object_info/" + kind,
                                                  timeout_seconds=timeout_seconds,
                                                  max_bytes=1024 * 1024))
        schema = response.get(kind)
        if not isinstance(schema, dict) or not isinstance(schema.get("input"), dict):
            raise ComfyRunError("worker is missing a reviewed core node schema")
        if (not isinstance(schema.get("output"), list)
                or not isinstance(schema["input"].get("required", {}), dict)
                or not isinstance(schema["input"].get("optional", {}), dict)):
            raise ComfyRunError("malformed core node schema")
        schemas[kind] = schema
    for node in graph.values():
        schema = schemas[node["class_type"]]
        required, optional = (schema["input"].get(key, {}) for key in ("required", "optional"))
        inputs = node["inputs"]
        if not set(required) <= set(inputs) or not set(inputs) <= set(required) | set(optional):
            raise ComfyRunError("worker node inputs differ from reviewed graph")
        for field, value in inputs.items():
            if isinstance(value, list):
                target = schemas[graph[value[0]]["class_type"]]
                if value[1] >= len(target["output"]):
                    raise ComfyRunError("worker node output slot is missing")
            if field in ("unet_name", "clip_name", "vae_name"):
                spec = (required | optional)[field]
                if (not isinstance(spec, list) or not spec or not isinstance(spec[0], list)
                        or value not in spec[0]):
                    raise ComfyRunError("worker does not expose the approved model file")
    output = graph[selection.workflow.output_node]
    if schemas[output["class_type"]].get("output_node") is not True:
        raise ComfyRunError("worker output node is not executable")


def execute_comfy(
    plan: ProjectPlan, selection: ApprovedComfySelection, worker: WorkerVerification,
    output_dir: str | Path, *, allow_live: bool = False,
    endpoint: str = "http://127.0.0.1:8188", timeout_seconds: float = 600,
    request_timeout_seconds: float = 10, poll_interval_seconds: float = 1,
    max_polls: int = 600, max_result_bytes: int = 32 * 1024 * 1024,
    transport: ComfyTransport | None = None, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunManifest:
    """Run exactly one approved shot; never provision, install, download weights or retry.

    Worker records and reviewed selection are operator-trusted declarations, separate
    from the creative plan. Runtime bounds do not enforce cloud rental/billing limits.
    A failure may leave active GPU work; run.json records bounded cancellation results.
    """
    failed = False
    provider = None
    bounded = None
    request = None
    deadline = 0.0
    try:
        if (allow_live is not True or not isinstance(plan, ProjectPlan)
                or len(plan.shots) != 1 or not isinstance(selection, ApprovedComfySelection)
                or not isinstance(worker, WorkerVerification)):
            raise ComfyRunError("explicit opt-in and one approved shot are required")
        worker.require_matches(selection)
        request = selection.request(plan.shots[0])
        timeout_seconds = positive_seconds(timeout_seconds)
        request_timeout_seconds = positive_seconds(request_timeout_seconds)
        if (type(poll_interval_seconds) not in (int, float)
                or not math.isfinite(poll_interval_seconds)
                or not 0 <= poll_interval_seconds <= 60):
            raise ComfyRunError("invalid polling interval")
        native = ComfyHTTPTransport(endpoint, allow_live=True)
        deadline = clock() + timeout_seconds
        bounded = _BoundedTransport(native if transport is None else transport, deadline, clock)
        provider = ComfyExecutionProvider(
            bounded, selection.workflow, request_timeout_seconds=request_timeout_seconds,
            job_timeout_seconds=timeout_seconds, max_polls=max_polls,
            max_result_bytes=max_result_bytes, clock=clock,
            verified_server_revision=worker.record["server_revision"],
        )
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or provider is None or bounded is None or request is None:
        raise ComfyRunError("ComfyUI single-shot preflight rejected")

    run_id = "comfy-" + uuid.uuid4().hex
    key = run_id + "-shot"
    record: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "outcome": "prepared",
        "plan": plan.to_dict(), "request": request.to_dict(), "idempotency_key": key,
        "translated_workflow": selection.workflow.translate(request),
        "selection_subject_sha256": selection.declarations["approval"]["model_workflow"][
            "approved_subject_sha256"],
        "worker_verification": worker.record, "known_run": None,
        "cloud_cost": {"status": "not_measured", "approved_setup_limit_usd":
                       selection.declarations["approval"]["spend_limit_usd"]},
        "cancellation": {"attempted": False, "confirmed_terminal": False},
    }
    run: ShotRun | None = None
    completed: RunManifest | None = None
    failed = False
    try:
        with PrivateRunDirectory(output_dir) as store:
            try:
                # Persist intent before any request; never replay this directory on restart.
                store.checkpoint(canonical(record))
                _verify_nodes(bounded, selection, request_timeout_seconds)
                record["outcome"] = "submitting"
                store.checkpoint(canonical(record))
                run = provider.submit(request, key)
                record.update(outcome="submitted", known_run=run.to_dict())
                store.checkpoint(canonical(record))
                while run.status not in TERMINAL:
                    run = provider.status(run.job_id)
                    record.update(outcome="awaiting_artifact" if run.status == RunStatus.SUCCEEDED
                                  else "submitted", known_run=run.to_dict())
                    store.checkpoint(canonical(record))
                    if run.status not in TERMINAL:
                        remaining = deadline - clock()
                        if remaining <= 0:
                            raise JobDeadlineError("single-shot deadline exceeded")
                        sleep(min(poll_interval_seconds, remaining))
                if run.status != RunStatus.SUCCEEDED:
                    raise ComfyRunError("ComfyUI shot did not succeed")
                result = provider.result(run.job_id)
                artifact = store.put(result)
                completed = RunManifest(run_id, plan, (result.run,), (artifact,))
                record.update(outcome="succeeded", known_run=result.run.to_dict(),
                              manifest=completed.to_dict())
                store.checkpoint(canonical(record))
            except (Exception, KeyboardInterrupt):
                failed = True
                # One bounded attempt, for this instance's confirmed job only. A missing
                # acknowledgement/history never becomes a fabricated cancelled state.
                if run is not None and run.status not in TERMINAL:
                    record["cancellation"]["attempted"] = True
                    bounded.deadline = clock() + request_timeout_seconds
                    try:
                        run = provider.cancel(run.job_id)
                        record["cancellation"]["confirmed_terminal"] = run.status in TERMINAL
                    except (Exception, KeyboardInterrupt):
                        pass
                record.update(outcome="incomplete", known_run=None if run is None else run.to_dict())
                try:
                    store.checkpoint(canonical(record))
                except (Exception, KeyboardInterrupt):
                    pass
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or completed is None:
        raise ComfyRunError("ComfyUI run incomplete; inspect private run.json before any new submission")
    return completed
