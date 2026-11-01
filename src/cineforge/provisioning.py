"""Compute lifecycle boundary only; no hardware detection or provisioning implementation."""

from typing import Protocol, runtime_checkable

from cineforge.devices import GPUDevice, ModelSpec


@runtime_checkable
class Provisioner(Protocol):
    """Future adapters enforce budget/cleanup; execution providers need no provisioner."""

    def acquire(self, requirements: ModelSpec, *, max_cost_usd: float) -> GPUDevice: ...

    def inspect(self, device_id: str) -> GPUDevice: ...

    def release(self, device_id: str) -> None: ...
