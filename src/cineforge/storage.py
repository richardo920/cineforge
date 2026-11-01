"""Single-host POSIX storage for synthetic runs; cooperating writers hold one flock."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from cineforge.contracts import Artifact, RunManifest, RunStatus, ShotRun
from cineforge.providers import ExecutionResult, FakeExecutionProvider

if TYPE_CHECKING:
    from cineforge.mock import MockPlan


class StorageError(ValueError):
    """Unsafe, inconsistent, corrupt, or already-owned local run storage."""


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StorageError(f"duplicate checkpoint key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise StorageError(f"nonfinite checkpoint number: {value}")


class LocalRunStore:
    """Own a dedicated directory; no symlinks/hardlinks or unrelated files are accepted.

    Requires POSIX dir_fd, O_NOFOLLOW, flock and local filesystem atomic rename/fsync.
    Locks coordinate this implementation on one host, not hostile writers/distributed storage.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._directory = -1
        self._lock = -1

    def __enter__(self) -> LocalRunStore:
        try:
            import fcntl
        except ImportError as exc:
            raise StorageError("local run storage requires POSIX flock") from exc
        # Open each component relative to its parent: even ancestor symlinks are rejected.
        parts = self.path.absolute().parts
        descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in parts[1:]:
                if part in (".", ".."):
                    raise StorageError("output path must not contain traversal components")
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            self._directory = descriptor
            descriptor = -1
            self._lock = os.open(".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                                 0o600, dir_fd=self._directory)
            self._regular(self._lock)
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StorageError("output directory is locked by another run") from exc
            self._inventory()
            return self
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            self.__exit__(None, None, None)
            raise

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if self._lock >= 0:
            os.close(self._lock)
            self._lock = -1
        if self._directory >= 0:
            os.close(self._directory)
            self._directory = -1

    @staticmethod
    def _regular(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise StorageError("store files must be regular files with exactly one link")

    def _read(self, name: str) -> bytes:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=self._directory)
        try:
            self._regular(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(descriptor)

    def _inventory(self) -> None:
        for name in os.listdir(self._directory):
            if name not in (".lock", "checkpoint.json") and re.fullmatch(
                r"artifact-[0-9a-f]{64}\.json|\.pending-[0-9a-f]{32}", name,
            ) is None:
                raise StorageError(f"unexpected store entry: {name}")
            payload = self._read(name)
            if name.startswith("artifact-") and hashlib.sha256(payload).hexdigest() != name[9:-5]:
                raise StorageError("artifact filename checksum mismatch")

    def _atomic_write(self, name: str, payload: bytes) -> None:
        temporary = f".pending-{uuid.uuid4().hex}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self._directory)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(descriptor)
            os.replace(temporary, name, src_dir_fd=self._directory, dst_dir_fd=self._directory)
            os.fsync(self._directory)
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=self._directory)
            except FileNotFoundError:
                pass

    def start(self, plan: MockPlan, identity: str, *, resume: bool) -> RunManifest:
        from cineforge.mock import MockPlan

        try:
            raw = self._read("checkpoint.json")
        except FileNotFoundError:
            if resume:
                raise StorageError("resume requires an existing checkpoint") from None
            if set(os.listdir(self._directory)) != {".lock"}:
                raise StorageError("new run requires an empty output directory") from None
            manifest = RunManifest(run_id=identity, plan=plan.plan)
            self.checkpoint(plan, manifest)
            return manifest
        if not resume:
            raise StorageError("output already contains a run; use --resume")
        try:
            data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
            if (not isinstance(data, dict)
                    or set(data) != {"schema_version", "inputs", "manifest"}
                    or type(data["schema_version"]) is not int or data["schema_version"] != 1):
                raise StorageError("unsupported or malformed checkpoint")
            stored_plan = MockPlan.from_json(json.dumps(data["inputs"], allow_nan=False))
            manifest = RunManifest.from_dict(data["manifest"])
            if (stored_plan.to_json() != plan.to_json() or manifest.plan != plan.plan
                    or manifest.run_id != identity):
                raise StorageError("stored run identity or full execution inputs differ")
            self._validate(plan, manifest)
            return manifest
        except (ValueError, TypeError, RecursionError) as exc:
            raise StorageError(f"invalid checkpoint: {exc}") from exc

    def _validate(self, plan: MockPlan, manifest: RunManifest) -> None:
        from cineforge.mock import mock_idempotency_key

        requests = {request.shot.shot_id: request for request in plan.preflight()}
        latest: dict[str, ShotRun] = {}
        for run in manifest.shot_runs:
            previous = latest.get(run.shot_id)
            if (run.attempt != (1 if previous is None else previous.attempt + 1)
                    or (previous is not None and previous.status not in
                        (RunStatus.FAILED, RunStatus.CANCELLED))):
                raise StorageError("invalid attempt history")
            request = replace(requests[run.shot_id], attempt=run.attempt)
            digest = hashlib.sha256(_json({
                "provider_id": FakeExecutionProvider.provider_id,
                "idempotency_key": mock_idempotency_key(manifest.run_id, request),
                "request": request.to_dict(),
            })).hexdigest()
            if (run.job_id != f"fake-{digest}"
                    or run.provider_id != FakeExecutionProvider.provider_id
                    or run.idempotency_key != mock_idempotency_key(manifest.run_id, request)):
                raise StorageError("stored job does not match execution inputs")
            if run.status == RunStatus.SUCCEEDED:
                if len(run.artifact_ids) != 1:
                    raise StorageError("completed mock job must have one artifact")
                artifact = next(item for item in manifest.artifacts
                                if item.artifact_id == run.artifact_ids[0])
                if (artifact.artifact_id != f"metadata-{digest}"
                        or artifact.origin != f"{run.provider_id}:{run.job_id}"
                        or artifact.license != "LicenseRef-SyntheticFixture"):
                    raise StorageError("stored artifact identity differs from mock job")
                payload = self.payload(artifact)
                expected = _json({
                    "schema_version": 1, "synthetic": True, "media_type": "application/json",
                    "provider_id": run.provider_id, "job_id": run.job_id,
                    "request": request.to_dict(),
                })
                if payload != expected:
                    raise StorageError("artifact provenance differs from execution inputs")
            latest[run.shot_id] = run

    def checkpoint(self, plan: MockPlan, manifest: RunManifest) -> None:
        self._atomic_write("checkpoint.json", _json({
            "schema_version": 1, "inputs": plan.to_dict(), "manifest": manifest.to_dict(),
        }))

    def payload(self, artifact: Artifact) -> bytes:
        name = f"artifact-{artifact.sha256}.json"
        if artifact.uri != name:
            raise StorageError("unsafe or unexpected artifact URI")
        payload = self._read(name)
        if len(payload) != artifact.size_bytes or hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise StorageError("artifact checksum or size mismatch")
        return payload

    def put(self, result: ExecutionResult) -> Artifact:
        name = f"artifact-{result.artifact.sha256}.json"
        artifact = replace(result.artifact, uri=name,
                           retention_policy="local; retained until caller removes run directory")
        try:
            existing = self._read(name)
        except FileNotFoundError:
            self._atomic_write(name, result.payload)
        else:
            if existing != result.payload:
                raise StorageError("refusing to overwrite different artifact bytes")
        self.payload(artifact)
        return artifact
