"""Version-one, metadata-only production contracts; no I/O or execution adapters."""

from __future__ import annotations

import json
import math
import re
from dataclasses import MISSING, dataclass, fields
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Self

SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ContractError(ValueError):
    """Invalid schema data or inconsistent production references."""


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{label} must be nonempty text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError(f"{label} must be valid Unicode text") from exc


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{label} must be a 1–128 character identifier")


def _integer(value: object, label: str, minimum: int = 0, maximum: int = 2**63 - 1) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractError(f"{label} must be an integer in [{minimum}, {maximum}]")


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ContractError(f"{label} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except OverflowError as exc:
        raise ContractError(f"{label} must be a finite number") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ContractError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA256 hex digest")


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be a timezone-aware ISO8601 timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{label} must be a timezone-aware ISO8601 timestamp") from exc
    if result.utcoffset() is None:
        raise ContractError(f"{label} must include a timezone")
    return result


def _tuple(value: object, kind: type[object], label: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
        raise ContractError(f"{label} must be an immutable tuple of {kind.__name__}")


def _unique(values: tuple[object, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ContractError(f"duplicate {label}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ContractError(f"nonfinite JSON number: {value}")


def _encode(value: Any) -> Any:
    if isinstance(value, Contract):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Contract:
    """Strict JSON codec shared by the fixed production records below."""

    schema_version: int = SCHEMA_VERSION
    _objects: ClassVar[dict[str, type[Contract]]] = {}
    _sequences: ClassVar[dict[str, type[Contract]]] = {}
    _enums: ClassVar[dict[str, type[Enum]]] = {}

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _encode(getattr(self, field.name)) for field in fields(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
            raise ContractError(f"{cls.__name__} must be an object with string keys")
        declared = {field.name for field in fields(cls)}
        required = {field.name for field in fields(cls)
                    if field.default is MISSING and field.default_factory is MISSING}
        required.add("schema_version")
        if set(data) - declared or required - set(data):
            raise ContractError(f"{cls.__name__} has missing or unknown fields")
        values = dict(data)
        for name, kind in cls._objects.items():
            if name in values:
                values[name] = kind.from_dict(values[name])
        for name, kind in cls._sequences.items():
            if name in values:
                if not isinstance(values[name], list):
                    raise ContractError(f"{name} must be a JSON array")
                values[name] = tuple(kind.from_dict(item) for item in values[name])
        # String-tuples have no nested record type.
        if "artifact_ids" in values:
            if not isinstance(values["artifact_ids"], list):
                raise ContractError("artifact_ids must be a JSON array")
            values["artifact_ids"] = tuple(values["artifact_ids"])
        for name, enum in cls._enums.items():
            if name in values:
                try:
                    values[name] = enum(values[name])
                except (ValueError, TypeError) as exc:
                    raise ContractError(f"invalid {name}") from exc
        try:
            return cls(**values)
        except TypeError as exc:
            raise ContractError(f"invalid {cls.__name__} fields") from exc

    @classmethod
    def from_json(cls, raw: str) -> Self:
        if not isinstance(raw, str):
            raise ContractError("JSON input must be text")
        try:
            value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        except (ValueError, RecursionError) as exc:
            raise ContractError("invalid contract JSON") from exc
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class Project(Contract):
    project_id: str
    revision: int
    title: str

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.project_id, "project_id")
        _integer(self.revision, "revision", 1)
        _text(self.title, "title")


@dataclass(frozen=True, slots=True)
class Scene(Contract):
    scene_id: str
    project_id: str
    revision: int

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.scene_id, "scene_id")
        _identifier(self.project_id, "project_id")
        _integer(self.revision, "revision", 1)


@dataclass(frozen=True, slots=True)
class Shot(Contract):
    shot_id: str
    scene_id: str
    revision: int
    prompt: str
    seed: int | None
    duration_seconds: float
    model_id: str
    model_revision: str
    workflow_digest: str
    character_revision: str | None = None

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.shot_id, "shot_id")
        _identifier(self.scene_id, "scene_id")
        _integer(self.revision, "revision", 1)
        _text(self.prompt, "prompt")
        if self.seed is not None:
            _integer(self.seed, "seed", maximum=2**64 - 1)
        object.__setattr__(self, "duration_seconds",
                           _number(self.duration_seconds, "duration_seconds", positive=True))
        _text(self.model_id, "model_id")
        _text(self.model_revision, "model_revision")
        _digest(self.workflow_digest, "workflow_digest")
        if self.character_revision is not None:
            _text(self.character_revision, "character_revision")


@dataclass(frozen=True, slots=True)
class Artifact(Contract):
    artifact_id: str
    shot_id: str
    uri: str
    sha256: str
    size_bytes: int
    license: str
    origin: str
    retention_policy: str

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.artifact_id, "artifact_id")
        _identifier(self.shot_id, "shot_id")
        _text(self.uri, "uri")
        _digest(self.sha256, "sha256")
        _integer(self.size_bytes, "size_bytes")
        _text(self.license, "license")
        _text(self.origin, "origin")
        _text(self.retention_policy, "retention_policy")


@dataclass(frozen=True, slots=True)
class ProjectPlan(Contract):
    project: Project
    scenes: tuple[Scene, ...]
    shots: tuple[Shot, ...]
    _objects: ClassVar[dict[str, type[Contract]]] = {"project": Project}
    _sequences: ClassVar[dict[str, type[Contract]]] = {"scenes": Scene, "shots": Shot}

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        if not isinstance(self.project, Project):
            raise ContractError("project must be a Project")
        _tuple(self.scenes, Scene, "scenes")
        _tuple(self.shots, Shot, "shots")
        if not self.scenes or not self.shots:
            raise ContractError("a production plan requires scenes and shots")
        _unique(tuple(scene.scene_id for scene in self.scenes), "scene_id")
        _unique(tuple(shot.shot_id for shot in self.shots), "shot_id")
        if any(scene.project_id != self.project.project_id for scene in self.scenes):
            raise ContractError("scene references an unknown project")
        scene_ids = {scene.scene_id for scene in self.scenes}
        if any(shot.scene_id not in scene_ids for shot in self.shots):
            raise ContractError("shot references an unknown scene")
        if scene_ids != {shot.scene_id for shot in self.shots}:
            raise ContractError("every scene must contain a shot")

    def get_shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.shot_id == shot_id:
                return shot
        raise ContractError(f"unknown shot_id: {shot_id}")


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ShotRun(Contract):
    shot_id: str
    shot_revision: int
    attempt: int
    provider_id: str
    job_id: str
    idempotency_key: str
    status: RunStatus
    started_at: str | None = None
    finished_at: str | None = None
    cost_usd: float = 0.0
    artifact_ids: tuple[str, ...] = ()
    error: str | None = None
    _enums: ClassVar[dict[str, type[Enum]]] = {"status": RunStatus}

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.shot_id, "shot_id")
        _integer(self.shot_revision, "shot_revision", 1)
        _integer(self.attempt, "attempt", 1)
        for name in ("provider_id", "job_id", "idempotency_key"):
            _text(getattr(self, name), name)
        if not isinstance(self.status, RunStatus):
            raise ContractError("status must be a RunStatus")
        object.__setattr__(self, "cost_usd", _number(self.cost_usd, "cost_usd"))
        _tuple(self.artifact_ids, str, "artifact_ids")
        for artifact_id in self.artifact_ids:
            _identifier(artifact_id, "artifact_id")
        _unique(self.artifact_ids, "artifact_id")
        start = None if self.started_at is None else _timestamp(self.started_at, "started_at")
        finish = None if self.finished_at is None else _timestamp(self.finished_at, "finished_at")
        if start is not None and finish is not None and finish < start:
            raise ContractError("finished_at precedes started_at")
        if self.status == RunStatus.QUEUED and (start is not None or finish is not None):
            raise ContractError("queued runs cannot have execution timestamps")
        if self.status == RunStatus.RUNNING and (start is None or finish is not None):
            raise ContractError("running runs require only started_at")
        if self.status == RunStatus.SUCCEEDED and (start is None or finish is None):
            raise ContractError("succeeded runs require start and finish timestamps")
        if self.status in (RunStatus.FAILED, RunStatus.CANCELLED) and finish is None:
            raise ContractError("failed/cancelled runs require finished_at")
        if self.status != RunStatus.SUCCEEDED and self.artifact_ids:
            raise ContractError("only succeeded runs may publish artifacts")
        if self.status == RunStatus.SUCCEEDED and not self.artifact_ids:
            raise ContractError("succeeded runs require at least one artifact")
        if self.error is not None:
            _text(self.error, "error")
        if self.status == RunStatus.FAILED and self.error is None:
            raise ContractError("failed runs require an error")
        if self.status not in (RunStatus.FAILED, RunStatus.CANCELLED) and self.error is not None:
            raise ContractError("only failed/cancelled runs may have an error")


@dataclass(frozen=True, slots=True)
class RunManifest(Contract):
    run_id: str
    plan: ProjectPlan
    shot_runs: tuple[ShotRun, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    _objects: ClassVar[dict[str, type[Contract]]] = {"plan": ProjectPlan}
    _sequences: ClassVar[dict[str, type[Contract]]] = {
        "shot_runs": ShotRun, "artifacts": Artifact,
    }

    def __post_init__(self) -> None:
        Contract.__post_init__(self)
        _identifier(self.run_id, "run_id")
        if not isinstance(self.plan, ProjectPlan):
            raise ContractError("plan must be a ProjectPlan")
        _tuple(self.shot_runs, ShotRun, "shot_runs")
        _tuple(self.artifacts, Artifact, "artifacts")
        _unique(tuple((run.shot_id, run.attempt) for run in self.shot_runs), "shot attempt")
        _unique(tuple((run.provider_id, run.job_id) for run in self.shot_runs), "provider job")
        _unique(tuple(run.idempotency_key for run in self.shot_runs), "idempotency_key")
        _unique(tuple(artifact.artifact_id for artifact in self.artifacts), "artifact_id")
        artifacts = {artifact.artifact_id: artifact for artifact in self.artifacts}
        produced: list[str] = []
        for artifact in self.artifacts:
            self.plan.get_shot(artifact.shot_id)
        for run in self.shot_runs:
            shot = self.plan.get_shot(run.shot_id)
            if run.shot_revision != shot.revision:
                raise ContractError("shot run references a different shot revision")
            for artifact_id in run.artifact_ids:
                if artifact_id not in artifacts:
                    raise ContractError("shot run references an unknown artifact")
                if artifacts[artifact_id].shot_id != run.shot_id:
                    raise ContractError("artifact belongs to a different shot")
                produced.append(artifact_id)
        _unique(tuple(produced), "artifact producer")
        if set(produced) != set(artifacts):
            raise ContractError("artifact has no producing shot run")
