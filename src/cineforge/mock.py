"""Strict synthetic declarations and an offline, metadata-only execution path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineforge.contracts import ContractError, ProjectPlan, RunManifest, RunStatus
from cineforge.devices import Backend, GPUDevice, ModelSpec, Precision, select_device
from cineforge.providers import (
    ExecutionRequest,
    ExecutionResult,
    FakeExecutionProvider,
    ProviderError,
)


def _object(value: object, names: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ContractError(f"{label} has missing or unknown fields")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a nonempty JSON array")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ContractError(f"nonfinite JSON number: {value}")


def _model(value: object) -> ModelSpec:
    data = _object(value, {
        "model_id", "revision", "task", "allowed_backends", "minimum_memory_mib",
        "license", "required_precision", "commercial_use",
    }, "model")
    backends = [Backend(item) for item in _array(data["allowed_backends"], "allowed_backends")]
    if len(set(backends)) != len(backends):
        raise ContractError("duplicate allowed backend")
    return ModelSpec(**(data | {
        "allowed_backends": frozenset(backends),
        "required_precision": Precision(data["required_precision"]),
    }))


def _device(value: object) -> GPUDevice:
    data = _object(value, {
        "device_id", "backend", "memory_mib", "supported_precisions",
    }, "device")
    precisions = [Precision(item) for item in
                  _array(data["supported_precisions"], "supported_precisions")]
    if len(set(precisions)) != len(precisions):
        raise ContractError("duplicate supported precision")
    return GPUDevice(**(data | {
        "backend": Backend(data["backend"]), "supported_precisions": frozenset(precisions),
    }))


@dataclass(frozen=True, slots=True)
class MockPlan:
    """Explicit caller declarations; device capacity and model rights are synthetic, unverified."""

    plan: ProjectPlan
    models: tuple[ModelSpec, ...]
    devices: tuple[GPUDevice, ...]

    def __post_init__(self) -> None:
        self.preflight()

    def preflight(self) -> tuple[ExecutionRequest, ...]:
        """Validate every request and target before any submission; never inspect local hardware."""
        if not isinstance(self.plan, ProjectPlan):
            raise ContractError("plan must be a ProjectPlan")
        if (not isinstance(self.models, tuple) or not self.models
                or any(not isinstance(model, ModelSpec) for model in self.models)):
            raise ContractError("models must be a nonempty tuple of ModelSpec")
        if (not isinstance(self.devices, tuple) or not self.devices
                or any(not isinstance(device, GPUDevice) for device in self.devices)):
            raise ContractError("devices must be a nonempty tuple of GPUDevice")
        models = {(model.model_id, model.revision): model for model in self.models}
        if len(models) != len(self.models):
            raise ContractError("duplicate model identity/revision")
        requests = []
        for shot in self.plan.shots:
            model = models.get((shot.model_id, shot.model_revision))
            if model is None:
                raise ContractError(f"unknown model identity/revision for shot: {shot.shot_id}")
            select_device(model, self.devices)
            requests.append(ExecutionRequest(shot, model))
        return tuple(requests)

    def to_dict(self) -> dict[str, object]:
        models = [{
            "model_id": model.model_id, "revision": model.revision, "task": model.task,
            "allowed_backends": sorted(item.value for item in model.allowed_backends),
            "minimum_memory_mib": model.minimum_memory_mib, "license": model.license,
            "required_precision": model.required_precision.value,
            "commercial_use": model.commercial_use,
        } for model in self.models]
        return {
            "schema_version": 1, "plan": self.plan.to_dict(), "models": models,
            "devices": [{
                "device_id": device.device_id, "backend": device.backend.value,
                "memory_mib": device.memory_mib,
                "supported_precisions": sorted(item.value for item in device.supported_precisions),
            } for device in self.devices],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> MockPlan:
        try:
            value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
            data = _object(value, {"schema_version", "plan", "models", "devices"}, "mock plan")
            if type(data["schema_version"]) is not int or data["schema_version"] != 1:
                raise ContractError("unsupported mock plan schema_version")
            return cls(
                ProjectPlan.from_dict(data["plan"]),
                tuple(_model(item) for item in _array(data["models"], "models")),
                tuple(_device(item) for item in _array(data["devices"], "devices")),
            )
        except (ValueError, TypeError, RecursionError) as exc:
            raise ContractError(f"invalid mock plan: {exc}") from exc


def load_mock_plan(path: str | Path) -> MockPlan:
    """Read strict UTF-8 JSON without creating or changing files."""
    return MockPlan.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class MockRun:
    manifest: RunManifest
    results: tuple[ExecutionResult, ...]


def mock_run_id(plan: MockPlan, *, run_id: str | None = None) -> str:
    """Resolve and validate a run identity covering all declared execution inputs."""
    identity = run_id if run_id is not None else (
        f"mock-{hashlib.sha256(plan.to_json().encode('utf-8')).hexdigest()}"
    )
    RunManifest(run_id=identity, plan=plan.plan)
    return identity


def mock_idempotency_key(run_id: str, request: ExecutionRequest) -> str:
    """Stable per-request key; run_id should come from mock_run_id before any submission."""
    value = json.dumps({"run_id": run_id, "request": request.to_dict()},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"mock-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def execute_mock(plan: MockPlan, *, run_id: str | None = None) -> MockRun:
    """Run the fake provider with bounded polling; return metadata and ephemeral JSON bytes."""
    requests = plan.preflight()
    identity = mock_run_id(plan, run_id=run_id)
    provider = FakeExecutionProvider()
    results = []
    for request in requests:
        run = provider.submit(request, mock_idempotency_key(identity, request))
        for _ in range(2):
            run = provider.status(run.job_id)
            if run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
                break
        if run.status != RunStatus.SUCCEEDED:
            provider.cancel(run.job_id)
            raise ProviderError(f"mock job did not succeed: {run.status.value}")
        results.append(provider.result(run.job_id))
    manifest = RunManifest(
        run_id=identity, plan=plan.plan, shot_runs=tuple(result.run for result in results),
        artifacts=tuple(result.artifact for result in results),
    )
    return MockRun(manifest, tuple(results))
