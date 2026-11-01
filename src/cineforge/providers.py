"""Execution contracts and a deterministic metadata-only, in-memory fake runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from threading import Lock
from typing import Protocol, runtime_checkable

from cineforge.contracts import Artifact, RunStatus, Shot, ShotRun
from cineforge.devices import ModelSpec


class ProviderError(ValueError):
    """Invalid execution input or an unavailable job/result."""


class UnknownJobError(ProviderError):
    """The provider does not own this job."""


class IdempotencyConflictError(ProviderError):
    """An existing key was reused with a different execution request."""


class ResultNotReadyError(ProviderError):
    """Only succeeded jobs have a result, including after cancellation/failure."""


def _text(value: object, field: str) -> None:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(not character.isprintable() for character in value)):
        raise ProviderError(f"{field} must be nonempty text without whitespace padding or controls")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    shot: Shot
    model: ModelSpec
    attempt: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.shot, Shot) or not isinstance(self.model, ModelSpec):
            raise ProviderError("request requires a Shot and ModelSpec")
        if (self.shot.model_id, self.shot.model_revision) != (
            self.model.model_id, self.model.revision,
        ):
            raise ProviderError("shot model identity/revision must match ModelSpec")
        if type(self.attempt) is not int or not 1 <= self.attempt <= 2**63 - 1:
            raise ProviderError("attempt must be an integer in [1, 2**63 - 1]")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible provenance snapshot, including declared model rights."""
        return {
            "shot": self.shot.to_dict(),
            "model": {
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "task": self.model.task,
                "allowed_backends": sorted(backend.value for backend in self.model.allowed_backends),
                "minimum_memory_mib": self.model.minimum_memory_mib,
                "license": self.model.license,
                "required_precision": self.model.required_precision.value,
                "commercial_use": self.model.commercial_use,
            },
            "attempt": self.attempt,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Bytes and their immutable metadata; URI is a declaration, never a storage write."""

    run: ShotRun
    artifact: Artifact
    payload: bytes

    def __post_init__(self) -> None:
        if (not isinstance(self.run, ShotRun) or not isinstance(self.artifact, Artifact)
                or not isinstance(self.payload, bytes)):
            raise ProviderError("result requires a ShotRun, Artifact and immutable bytes")
        if (self.run.status != RunStatus.SUCCEEDED
                or self.run.shot_id != self.artifact.shot_id
                or self.run.artifact_ids != (self.artifact.artifact_id,)):
            raise ProviderError("result must contain the succeeded run's single artifact")
        if (self.artifact.sha256 != hashlib.sha256(self.payload).hexdigest()
                or self.artifact.size_bytes != len(self.payload)):
            raise ProviderError("artifact checksum/size does not match payload")


@runtime_checkable
class ExecutionProvider(Protocol):
    """Submit once per key/request; results require success and terminal states are stable."""

    def submit(self, request: ExecutionRequest, idempotency_key: str) -> ShotRun: ...

    def status(self, job_id: str) -> ShotRun: ...

    def cancel(self, job_id: str) -> ShotRun: ...

    def result(self, job_id: str) -> ExecutionResult: ...


@dataclass
class _Job:
    request_bytes: bytes
    run: ShotRun
    artifact: Artifact
    payload: bytes


class FakeExecutionProvider:
    """Each status poll advances queued→running→succeeded (or failed with fail=True).

    Times are synthetic logical ticks, not wall time. State lasts only for this instance;
    resubmitting the same key/request to a fresh instance recreates the same IDs and bytes.
    """

    provider_id = "cineforge.fake-v1"
    _start = "2000-01-01T00:00:00+00:00"
    _finish = "2000-01-01T00:00:01+00:00"

    def __init__(self, *, fail: bool = False) -> None:
        if type(fail) is not bool:
            raise ProviderError("fail must be bool")
        self._fail = fail
        self._jobs: dict[str, _Job] = {}
        self._keys: dict[str, str] = {}
        self._lock = Lock()

    def submit(self, request: ExecutionRequest, idempotency_key: str) -> ShotRun:
        if not isinstance(request, ExecutionRequest):
            raise ProviderError("request must be an ExecutionRequest")
        _text(idempotency_key, "idempotency_key")
        request_bytes = _json(request.to_dict())
        with self._lock:
            if idempotency_key in self._keys:
                job = self._jobs[self._keys[idempotency_key]]
                if job.request_bytes != request_bytes:
                    raise IdempotencyConflictError("idempotency key already has a different request")
                return job.run
            digest = hashlib.sha256(_json({
                "provider_id": self.provider_id, "idempotency_key": idempotency_key,
                "request": request.to_dict(),
            })).hexdigest()
            job_id = f"fake-{digest}"
            payload = _json({
                "schema_version": 1, "synthetic": True, "media_type": "application/json",
                "provider_id": self.provider_id, "job_id": job_id, "request": request.to_dict(),
            })
            artifact = Artifact(
                artifact_id=f"metadata-{digest}", shot_id=request.shot.shot_id,
                uri=f"memory://{self.provider_id}/{job_id}/metadata.json",
                sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
                license="LicenseRef-SyntheticFixture", origin=f"{self.provider_id}:{job_id}",
                retention_policy="ephemeral; caller must persist payload separately",
            )
            run = ShotRun(
                shot_id=request.shot.shot_id, shot_revision=request.shot.revision,
                attempt=request.attempt, provider_id=self.provider_id, job_id=job_id,
                idempotency_key=idempotency_key, status=RunStatus.QUEUED,
            )
            self._jobs[job_id] = _Job(request_bytes, run, artifact, payload)
            self._keys[idempotency_key] = job_id
            return run

    def _job(self, job_id: str) -> _Job:
        _text(job_id, "job_id")
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise UnknownJobError(f"unknown job: {job_id}") from exc

    def status(self, job_id: str) -> ShotRun:
        with self._lock:
            job = self._job(job_id)
            if job.run.status == RunStatus.QUEUED:
                job.run = replace(job.run, status=RunStatus.RUNNING, started_at=self._start)
            elif job.run.status == RunStatus.RUNNING:
                job.run = replace(
                    job.run, status=RunStatus.FAILED if self._fail else RunStatus.SUCCEEDED,
                    finished_at=self._finish,
                    artifact_ids=() if self._fail else (job.artifact.artifact_id,),
                    error="synthetic execution failure" if self._fail else None,
                )
            return job.run

    def cancel(self, job_id: str) -> ShotRun:
        with self._lock:
            job = self._job(job_id)
            if job.run.status in (RunStatus.QUEUED, RunStatus.RUNNING):
                job.run = replace(job.run, status=RunStatus.CANCELLED, finished_at=self._finish)
            return job.run

    def result(self, job_id: str) -> ExecutionResult:
        with self._lock:
            job = self._job(job_id)
            if job.run.status != RunStatus.SUCCEEDED:
                raise ResultNotReadyError(f"job has no result: {job.run.status.value}")
            return ExecutionResult(job.run, job.artifact, job.payload)
