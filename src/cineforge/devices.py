"""Declared device/model capabilities and deterministic, offline device selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    CPU = "cpu"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


def _text(value: object, field: str) -> None:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(not character.isprintable() for character in value)):
        raise ValueError(f"{field} must be nonempty text without surrounding whitespace or controls")


def _memory(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer number of MiB")


def _enum_set(value: object, kind: type[Enum], field: str) -> None:
    if (not isinstance(value, frozenset) or not value
            or any(not isinstance(item, kind) for item in value)):
        raise ValueError(f"{field} must be a nonempty frozenset of {kind.__name__} values")


@dataclass(frozen=True)
class GPUDevice:
    """Caller-declared capacity; CPU memory means its available execution-memory budget."""

    device_id: str
    backend: Backend
    memory_mib: int
    supported_precisions: frozenset[Precision] = frozenset({Precision.FP32})

    def __post_init__(self) -> None:
        _text(self.device_id, "device_id")
        if not isinstance(self.backend, Backend):
            raise ValueError("backend must be a Backend value")
        _memory(self.memory_mib, "memory_mib")
        _enum_set(self.supported_precisions, Precision, "supported_precisions")


@dataclass(frozen=True)
class ModelSpec:
    """A declared model requirement; revision and rights metadata are not remotely verified."""

    model_id: str
    revision: str
    task: str
    allowed_backends: frozenset[Backend]
    minimum_memory_mib: int
    license: str
    required_precision: Precision = Precision.FP32
    commercial_use: bool | None = None

    def __post_init__(self) -> None:
        for field in ("model_id", "revision", "task", "license"):
            _text(getattr(self, field), field)
        if self.revision.casefold() in {
            "main", "master", "latest", "head", "default", "stable", "nightly",
        }:
            raise ValueError("revision must identify a pinned revision, not a floating alias")
        _enum_set(self.allowed_backends, Backend, "allowed_backends")
        _memory(self.minimum_memory_mib, "minimum_memory_mib")
        if not isinstance(self.required_precision, Precision):
            raise ValueError("required_precision must be a Precision value")
        if self.commercial_use is not None and type(self.commercial_use) is not bool:
            raise ValueError("commercial_use must be bool or None (unknown)")


class DeviceSelectionError(ValueError):
    """No compatible device, invalid inventory, or missing execution capacity."""


def compatibility(device: GPUDevice, model: ModelSpec) -> tuple[str, ...]:
    """Return all declared incompatibilities; an empty tuple means compatible capabilities."""
    if not isinstance(device, GPUDevice) or not isinstance(model, ModelSpec):
        raise ValueError("compatibility requires a GPUDevice and ModelSpec")
    reasons = []
    if device.backend not in model.allowed_backends:
        reasons.append(f"backend {device.backend.value} is not allowed")
    if device.memory_mib < model.minimum_memory_mib:
        reasons.append(
            f"insufficient memory: {device.memory_mib} MiB < {model.minimum_memory_mib} MiB"
        )
    if model.required_precision not in device.supported_precisions:
        reasons.append(f"precision {model.required_precision.value} is not supported")
    return tuple(reasons)


def select_device(model: ModelSpec, devices: Iterable[GPUDevice]) -> GPUDevice:
    """Select least sufficient memory, then backend value, then device ID; never detect hardware."""
    if not isinstance(model, ModelSpec):
        raise DeviceSelectionError("model must be a ModelSpec")
    inventory = tuple(devices)
    if not inventory:
        raise DeviceSelectionError("no devices available")
    if any(not isinstance(device, GPUDevice) for device in inventory):
        raise DeviceSelectionError("inventory must contain GPUDevice values")
    identifiers = [device.device_id for device in inventory]
    if len(set(identifiers)) != len(identifiers):
        raise DeviceSelectionError("duplicate device IDs are not allowed")
    compatible = [device for device in inventory if not compatibility(device, model)]
    if compatible:
        return min(compatible, key=lambda device: (
            device.memory_mib, device.backend.value, device.device_id,
        ))
    rejections = "; ".join(
        f"{device.device_id}: {', '.join(compatibility(device, model))}"
        for device in sorted(inventory, key=lambda device: device.device_id)
    )
    raise DeviceSelectionError(f"no compatible device: {rejections}")
