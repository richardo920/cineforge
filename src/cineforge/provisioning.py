"""Provider-neutral compute lifecycle boundary; importing it performs no I/O."""

from typing import Protocol, runtime_checkable

from cineforge.devices import GPUDevice, ModelSpec


@runtime_checkable
class Provisioner(Protocol):
    """Adapters manage compute separately from workflow execution and artifact storage."""

    def acquire(self, requirements: ModelSpec, *, max_cost_usd: float) -> GPUDevice: ...

    def inspect(self, device_id: str) -> GPUDevice: ...

    def release(self, device_id: str) -> None: ...
