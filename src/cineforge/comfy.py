"""ComfyUI execution protocol; no provisioning, installation or local artifact writes.

    API source: ComfyUI server.py at 6d352802d8f8b9133f664a1757d6c6d26390c335.
    Cancellation uses that revision's atomic /api/jobs/{id}/cancel route only.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from urllib.parse import urlencode

from cineforge.comfy_transport import ComfyTransport, positive_seconds
from cineforge.contracts import Artifact, RunStatus, ShotRun
from cineforge.providers import (
    ExecutionRequest,
    ExecutionResult,
    IdempotencyConflictError,
    ProviderError,
    ResultNotReadyError,
    UnknownJobError,
    _text,
)
from cineforge.workflows import WorkflowTemplate, canonical, strict_object

TERMINAL = (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)
SUPPORTED_SERVER_REVISION = "6d352802d8f8b9133f664a1757d6c6d26390c335"  # pragma: allowlist secret


class SubmissionUncertainError(ProviderError):
    """Submission may have reached the server; this key will never be submitted again."""


class JobStateUncertainError(ProviderError):
    """A job is absent from both queue and history; do not infer failure or retry it."""


class JobDeadlineError(ProviderError):
    """Client polling budget exhausted; explicit bounded cancellation remains available."""


@dataclass
class _Job:
    request: ExecutionRequest
    graph: dict[str, Any]
    run: ShotRun
    deadline: float
    polls: int = 0
    output: dict[str, str] | None = None
    result: ExecutionResult | None = None


class ComfyExecutionProvider:
    """Instance-local idempotency only; never blindly replay uncertain submissions.

    Callers approve the server revision/model/workflow and invoke cancellation on
    budget expiry. No restart-safe recovery or automatic retries are claimed.
    """

    provider_id = "cineforge.comfy-v1"

    def __init__(self, transport: ComfyTransport, workflow: WorkflowTemplate, *,
                 request_timeout_seconds: float = 10, job_timeout_seconds: float = 60,
                 max_polls: int = 120, max_result_bytes: int = 32 * 1024 * 1024,
                 clock: Callable[[], float] = time.monotonic,
                 verified_server_revision: str | None = None) -> None:
        if (not isinstance(workflow, WorkflowTemplate) or type(max_polls) is not int
                or not 1 <= max_polls <= 10000 or type(max_result_bytes) is not int
                or not 1 <= max_result_bytes <= 256 * 1024 * 1024):
            raise ProviderError("invalid ComfyUI provider bounds")
        if verified_server_revision not in (None, SUPPORTED_SERVER_REVISION):
            raise ProviderError("unsupported verified ComfyUI server revision")
        self._transport = transport
        self._workflow = workflow
        self._request_timeout = positive_seconds(request_timeout_seconds)
        self._job_timeout = positive_seconds(job_timeout_seconds)
        self._max_polls = max_polls
        self._max_result_bytes = max_result_bytes
        self._clock = clock
        self._atomic_cancel = verified_server_revision is not None
        self._jobs: dict[str, _Job] = {}
        self._keys: dict[str, tuple[bytes, str | None]] = {}
        self._lock = RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _call(self, method: str, path: str, body: dict[str, object] | None = None,
              *, deadline: float, max_bytes: int = 4 * 1024 * 1024) -> bytes:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise JobDeadlineError("ComfyUI client deadline exceeded; cancel the job explicitly")
        try:
            raw = self._transport.request(method, path, body,
                                          timeout_seconds=min(remaining, self._request_timeout),
                                          max_bytes=max_bytes)
        except (OSError, TimeoutError):
            raise ProviderError("ComfyUI connection or timeout failure") from None
        if not isinstance(raw, bytes) or len(raw) > max_bytes:
            raise ProviderError("invalid or oversized ComfyUI response")
        if self._clock() >= deadline:
            raise JobDeadlineError("ComfyUI client deadline exceeded; cancel the job explicitly")
        return raw

    def submit(self, request: ExecutionRequest, idempotency_key: str) -> ShotRun:
        graph = self._workflow.translate(request)
        _text(idempotency_key, "idempotency_key")
        snapshot = canonical(request.to_dict())
        with self._lock:
            if idempotency_key in self._keys:
                previous, identity = self._keys[idempotency_key]
                if snapshot != previous:
                    raise IdempotencyConflictError("idempotency key has a different request")
                if identity is None:
                    raise SubmissionUncertainError("submission state is uncertain; key cannot replay")
                return self._jobs[identity].run
            # Reserve before I/O. ComfyUI does not provide server-side idempotency.
            self._keys[idempotency_key] = (snapshot, None)
            deadline = self._clock() + self._job_timeout
            try:
                response = strict_object(self._call("POST", "/prompt", {"prompt": graph},
                                                    deadline=deadline))
                identity = response.get("prompt_id")
                if (not isinstance(identity, str) or str(uuid.UUID(identity)) != identity
                        or identity in self._jobs or response.get("error")
                        or response.get("node_errors") not in ({}, None)):
                    raise ValueError
            except (ProviderError, ValueError, TypeError, AttributeError):
                raise SubmissionUncertainError("submission not confirmed; key cannot replay") from None
            run = ShotRun(shot_id=request.shot.shot_id, shot_revision=request.shot.revision,
                          attempt=request.attempt, provider_id=self.provider_id, job_id=identity,
                          idempotency_key=idempotency_key, status=RunStatus.QUEUED)
            self._jobs[identity] = _Job(request, graph, run, deadline)
            self._keys[idempotency_key] = (snapshot, identity)
            return run

    def _job(self, identity: str) -> _Job:
        _text(identity, "job_id")
        try:
            return self._jobs[identity]
        except KeyError:
            raise UnknownJobError("provider does not own this ComfyUI job") from None

    def _queue(self, deadline: float) -> tuple[set[str], set[str]]:
        data = strict_object(self._call("GET", "/queue", deadline=deadline))
        result = []
        for name in ("queue_running", "queue_pending"):
            rows = data.get(name)
            if (not isinstance(rows, list) or any(not isinstance(row, list) or len(row) < 2
                                                or not isinstance(row[1], str) for row in rows)):
                raise ProviderError("malformed ComfyUI queue")
            result.append({row[1] for row in rows})
        if result[0] & result[1]:
            raise ProviderError("inconsistent ComfyUI queue")
        return result[0], result[1]

    def _history(self, job: _Job, deadline: float) -> bool:
        identity = job.run.job_id
        data = strict_object(self._call("GET", f"/history/{identity}", deadline=deadline))
        if not data:
            return False
        item = data.get(identity)
        if not isinstance(item, dict):
            raise ProviderError("malformed ComfyUI history")
        prompt, status = item.get("prompt"), item.get("status")
        if (not isinstance(prompt, list) or len(prompt) < 3 or prompt[1] != identity
                or canonical(prompt[2]) != canonical(job.graph) or not isinstance(status, dict)
                or type(status.get("completed")) is not bool
                or status.get("status_str") not in ("success", "error")):
            raise ProviderError("ComfyUI history differs from the submitted job")
        if status["status_str"] == "error":
            messages = status.get("messages", [])
            cancelled = (isinstance(messages, list)
                         and any(isinstance(m, list) and m and m[0] == "execution_interrupted"
                                 for m in messages))
            job.run = replace(job.run, status=RunStatus.CANCELLED if cancelled else RunStatus.FAILED,
                              finished_at=self._now(), error=None if cancelled else
                              "ComfyUI execution failed")
        else:
            if not status["completed"]:
                raise ProviderError("incomplete ComfyUI success history")
            output = self._output(item.get("outputs"))
            job.output = output
            job.run = replace(job.run, status=RunStatus.SUCCEEDED,
                              started_at=job.run.started_at or self._now(), finished_at=self._now(),
                              artifact_ids=(f"comfy-{identity}",))
        return True

    def _output(self, outputs: object) -> dict[str, str]:
        if not isinstance(outputs, dict):
            raise ProviderError("ComfyUI output is missing")
        node = outputs.get(self._workflow.output_node)
        items = node.get(self._workflow.output_kind) if isinstance(node, dict) else None
        if not isinstance(items, list) or len(items) <= self._workflow.output_index:
            raise ProviderError("selected ComfyUI output is missing")
        item = items[self._workflow.output_index]
        if (not isinstance(item, dict) or item.get("type") != "output"
                or not isinstance(item.get("filename"), str)
                or not isinstance(item.get("subfolder"), str)):
            raise ProviderError("malformed ComfyUI output descriptor")
        filename, subfolder = item["filename"], item["subfolder"]
        for part in [filename] + (subfolder.split("/") if subfolder else []):
            if (not part or part in (".", "..") or ".." in part
                    or any(c in part for c in "/\\:%") or not part.isprintable()):
                raise ProviderError("unsafe ComfyUI output path")
        return {"filename": filename, "subfolder": subfolder, "type": "output"}

    def status(self, job_id: str) -> ShotRun:
        with self._lock:
            job = self._job(job_id)
            if job.run.status in TERMINAL:
                return job.run
            if job.polls >= self._max_polls:
                raise JobDeadlineError("ComfyUI poll limit exceeded; cancel the job explicitly")
            job.polls += 1
            if not self._history(job, job.deadline):
                running, pending = self._queue(job.deadline)
                if job_id in running:
                    job.run = replace(job.run, status=RunStatus.RUNNING,
                                      started_at=job.run.started_at or self._now())
                elif job_id not in pending:
                    # Queue->history completion can race the first history lookup.
                    if not self._history(job, job.deadline):
                        raise JobStateUncertainError("job absent from ComfyUI queue and history")
            return job.run

    def cancel(self, job_id: str) -> ShotRun:
        with self._lock:
            job = self._job(job_id)
            if job.run.status in TERMINAL:
                return job.run
            deadline = self._clock() + self._request_timeout
            if self._history(job, deadline):
                return job.run
            running, pending = self._queue(deadline)
            if job_id not in running | pending:
                if self._history(job, deadline):
                    return job.run
                raise JobStateUncertainError("job absent before cancellation; no action confirmed")
            if not self._atomic_cancel:
                raise ProviderError("cancellation requires a verified atomic job-cancel API")
            acknowledgement = strict_object(self._call(
                "POST", f"/api/jobs/{job_id}/cancel", deadline=deadline,
            ))
            if (set(acknowledgement) != {"cancelled"}
                    or type(acknowledgement["cancelled"]) is not bool):
                raise ProviderError("invalid ComfyUI cancellation acknowledgement")
            if self._history(job, deadline):
                return job.run
            running, pending = self._queue(deadline)
            if job_id in running:
                job.run = replace(job.run, status=RunStatus.RUNNING,
                                  started_at=job.run.started_at or self._now())
            elif job_id not in pending:
                # Completion publishes history under the queue mutex. Reconcile after
                # disappearance, not just before the queue read, before caching a terminal.
                if self._history(job, deadline):
                    return job.run
                # This API's boolean cannot distinguish dequeue from a dispatched
                # running interrupt. Even an earlier pending snapshot cannot prove
                # which action occurred; absence alone never establishes a terminal.
                raise JobStateUncertainError("cancellation outcome is not confirmed")
            return job.run

    def result(self, job_id: str) -> ExecutionResult:
        with self._lock:
            job = self._job(job_id)
            if job.run.status != RunStatus.SUCCEEDED or job.output is None:
                raise ResultNotReadyError("ComfyUI job has no succeeded result")
            if job.result is None:
                path = "/view?" + urlencode(job.output)
                raw = self._call("GET", path, deadline=job.deadline,
                                 max_bytes=self._max_result_bytes)
                artifact = Artifact(
                    artifact_id=job.run.artifact_ids[0], shot_id=job.request.shot.shot_id,
                    uri=f"comfy://{job_id}/{job.output['filename']}",
                    sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
                    license=self._workflow.artifact_license,
                    origin=f"{self.provider_id}:{job_id}",
                    retention_policy="server output; caller must persist downloaded bytes",
                )
                job.result = ExecutionResult(job.run, artifact, raw)
            return job.result
