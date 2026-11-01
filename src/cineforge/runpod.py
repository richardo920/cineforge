"""Owned ephemeral Runpod Pod lifecycle; no workflow execution or durable billing policy.

Wire contract: https://api.runpod.io/v2/openapi.json (inspected 2026-11-01).
Pod gpu.memory is host RAM. Only the separately supplied runtime probe provides VRAM.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, TypeGuard

from cineforge.devices import Backend, GPUDevice, ModelSpec, Precision, compatibility
from cineforge.runpod_transport import CloudResponse, CloudTransport, bounded_seconds
from cineforge.workflows import canonical, strict_object


def _identifier(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is not None


def _money(value: object) -> TypeGuard[int | float]:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 < value <= 100000)


def _model_dict(model: ModelSpec) -> dict[str, Any]:
    return {"model_id": model.model_id, "revision": model.revision, "task": model.task,
            "allowed_backends": sorted(b.value for b in model.allowed_backends),
            "minimum_memory_mib": model.minimum_memory_mib, "license": model.license,
            "required_precision": model.required_precision.value,
            "commercial_use": model.commercial_use}


@dataclass(frozen=True)
class RunpodPodSpec:
    """Trusted operator configuration; the external policy hook must approve its scope."""

    session_id: str
    image: str
    gpu_id: str
    data_center_id: str
    ssh_public_key: str = field(repr=False)
    max_hourly_usd: float = 1.0
    disk_gb: int = 80
    min_cuda_version: str = "12.8"
    registry_id: str | None = None

    def __post_init__(self) -> None:
        if (not _identifier(self.session_id) or len(self.session_id) > 32
                or not isinstance(self.image, str)
                or not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", self.image)
                or not isinstance(self.gpu_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{1,100}", self.gpu_id)
                or not _identifier(self.data_center_id)
                or not isinstance(self.ssh_public_key, str)
                or not re.fullmatch(r"ssh-(?:ed25519|rsa) [A-Za-z0-9+/=]+(?: [A-Za-z0-9._-]+)?",
                                    self.ssh_public_key)
                or len(self.ssh_public_key) > 4096
                or not _money(self.max_hourly_usd)
                or type(self.disk_gb) is not int or not 10 <= self.disk_gb <= 1024
                or not isinstance(self.min_cuda_version, str)
                or not re.fullmatch(r"\d{1,2}\.\d{1,2}", self.min_cuda_version)
                or tuple(map(int, self.min_cuda_version.split("."))) < (12, 8)
                or self.registry_id is not None and not _identifier(self.registry_id)):
            raise ValueError("invalid immutable Runpod operator configuration")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class AcquisitionIntent:
    """Serialize this trusted operator intent before a paid create; it contains no API key."""

    key: str
    spec: RunpodPodSpec = field(repr=False)
    owner_token: str
    name: str
    requirements: ModelSpec = field(repr=False)
    max_cost_usd: float
    lease_seconds: float
    startup_timeout_seconds: float
    request_timeout_seconds: float

    def __post_init__(self) -> None:
        if (not _identifier(self.key) or not isinstance(self.requirements, ModelSpec)
                or not isinstance(self.spec, RunpodPodSpec)
                or not re.fullmatch(r"[0-9a-f]{32}", self.owner_token)
                or not _identifier(self.name) or not _money(self.max_cost_usd)):
            raise ValueError("invalid cloud acquisition intent")
        bounded_seconds(self.lease_seconds)
        bounded_seconds(self.startup_timeout_seconds)
        bounded_seconds(self.request_timeout_seconds)
        if self.startup_timeout_seconds > self.lease_seconds:
            raise ValueError("startup budget exceeds planned lease duration")

    @property
    def spec_digest(self) -> str:
        return self.spec.digest

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "key": self.key, "spec": self.spec.to_dict(),
                "spec_digest": self.spec_digest,
                "owner_token": self.owner_token, "name": self.name,
                "requirements": _model_dict(self.requirements), "max_cost_usd": self.max_cost_usd,
                "lease_seconds": self.lease_seconds,
                "startup_timeout_seconds": self.startup_timeout_seconds,
                "request_timeout_seconds": self.request_timeout_seconds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquisitionIntent:
        failed = False
        result = None
        try:
            if set(data) != {"schema_version", "key", "spec", "spec_digest", "owner_token", "name",
                             "requirements", "max_cost_usd", "lease_seconds",
                             "startup_timeout_seconds", "request_timeout_seconds"}:
                raise ValueError
            if type(data["schema_version"]) is not int or data["schema_version"] != 1:
                raise ValueError
            model = dict(data["requirements"])
            model["allowed_backends"] = frozenset(Backend(b) for b in model["allowed_backends"])
            model["required_precision"] = Precision(model["required_precision"])
            spec = RunpodPodSpec(**data["spec"])
            if spec.digest != data["spec_digest"]:
                raise ValueError
            result = cls(data["key"], spec, data["owner_token"], data["name"],
                         ModelSpec(**model), data["max_cost_usd"], data["lease_seconds"],
                         data["startup_timeout_seconds"], data["request_timeout_seconds"])
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise ValueError("invalid persisted acquisition intent")
        return result


_STATES = {"reserved", "created", "ready", "rejected", "create_uncertain", "cleanup_pending",
           "cleanup_uncertain", "released", "not_created"}


@dataclass(frozen=True)
class LeaseSnapshot:
    """Private journal event; a known identity survives every subsequent failure."""

    key: str
    state: str
    resource_id: str | None = None
    hourly_usd: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (not _identifier(self.key) or self.state not in _STATES
                or self.resource_id is not None and not _identifier(self.resource_id)
                or self.hourly_usd is not None and not _money(self.hourly_usd)
                or self.reason not in (None, "cancelled", "startup_timeout", "rejected",
                                       "transport", "invalid_worker", "event_sink", "restored")):
            raise ValueError("invalid cloud lease snapshot")
        if self.state in {"created", "ready", "cleanup_pending", "cleanup_uncertain", "released"} \
                and self.resource_id is None:
            raise ValueError("lease state requires a known resource identity")
        if self.state in {"reserved", "rejected", "not_created"} and self.resource_id is not None:
            raise ValueError("unsubmitted or rejected lease cannot own a resource")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "key": self.key, "state": self.state,
                "resource_id": self.resource_id, "hourly_usd": self.hourly_usd,
                "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeaseSnapshot:
        failed = False
        result = None
        try:
            if set(data) != {"schema_version", "key", "state", "resource_id", "hourly_usd",
                             "reason"} or type(data["schema_version"]) is not int \
                    or data["schema_version"] != 1:
                raise ValueError
            result = cls(data["key"], data["state"], data["resource_id"], data["hourly_usd"],
                         data["reason"])
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise ValueError("invalid persisted lease snapshot")
        return result


@dataclass(frozen=True)
class WorkerObservation:
    """Operator probe's measured result, separate from control-plane/catalog declarations."""

    device: GPUDevice
    raw_gpu_name: str

    def __post_init__(self) -> None:
        if (not isinstance(self.device, GPUDevice) or self.device.backend != Backend.CUDA
                or not isinstance(self.raw_gpu_name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{1,100}", self.raw_gpu_name)):
            raise ValueError("invalid measured worker observation")


class ProvisioningError(ValueError):
    """Static diagnostic with safe retained identity, never raw provider/credential text."""

    def __init__(self, message: str, snapshot: LeaseSnapshot | None = None) -> None:
        super().__init__(message)
        self.snapshot = snapshot


class AcquisitionUncertainError(ProvisioningError):
    """A create may have succeeded; this key and another paid create cannot replay it."""


class CleanupUncertainError(ProvisioningError):
    """Cleanup was not confirmed; the retained owned ID needs supervisor reconciliation."""


@dataclass
class _Lease:
    intent: AcquisitionIntent
    snapshot: LeaseSnapshot
    observation: WorkerObservation | None = None


class RunpodPodProvisioner:
    """Instance-local ownership/idempotency; M3.3 supplies durable policy and supervision.

    before_create MUST approve a fresh quote, budget and durable intent before POST.
    event_sink MUST persist each transition. Callback records are operator-trusted,
    not creative-plan inputs. Native HTTPS additionally needs a key and live opt-in.
    The provider has no server-side price ceiling: read-back catches price races only
    after allocation. The planned lease duration is not an autonomous shutdown timer.
    """

    def __init__(self, spec: RunpodPodSpec, transport: CloudTransport, *,
                 probe: Callable[[LeaseSnapshot, float], WorkerObservation | None],
                 before_create: Callable[[AcquisitionIntent], None],
                 event_sink: Callable[[AcquisitionIntent, LeaseSnapshot], None],
                 lease_seconds: float = 1800, startup_timeout_seconds: float = 120,
                 request_timeout_seconds: float = 10, cleanup_timeout_seconds: float = 30,
                 poll_interval_seconds: float = 1, max_polls: int = 120,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if (not isinstance(spec, RunpodPodSpec) or not callable(probe)
                or not callable(before_create) or not callable(event_sink)
                or type(max_polls) is not int or not 1 <= max_polls <= 10000
                or type(poll_interval_seconds) not in (int, float)
                or not math.isfinite(poll_interval_seconds)
                or not 0 <= poll_interval_seconds <= 60):
            raise ProvisioningError("invalid cloud lifecycle configuration")
        self.spec, self._transport = spec, transport
        self._probe, self._before, self._sink = probe, before_create, event_sink
        self._lease_seconds = bounded_seconds(lease_seconds)
        self._startup = bounded_seconds(startup_timeout_seconds)
        self._request = bounded_seconds(request_timeout_seconds)
        self._cleanup = bounded_seconds(cleanup_timeout_seconds)
        if self._startup > self._lease_seconds:
            raise ProvisioningError("startup budget exceeds planned lease duration")
        self._interval, self._max_polls = poll_interval_seconds, max_polls
        self._clock, self._sleep = clock, sleep
        self._leases: dict[str, _Lease] = {}
        self._ids: dict[str, str] = {}
        self._lock = RLock()

    @property
    def snapshots(self) -> tuple[LeaseSnapshot, ...]:
        with self._lock:
            return tuple(item.snapshot for item in self._leases.values())

    def intent(self, key: str) -> AcquisitionIntent:
        with self._lock:
            if key not in self._leases:
                raise ProvisioningError("unknown cloud lease")
            return self._leases[key].intent

    def _emit(self, lease: _Lease, **changes: Any) -> None:
        lease.snapshot = replace(lease.snapshot, **changes)
        failed = False
        try:
            self._sink(lease.intent, lease.snapshot)
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise ProvisioningError("cloud lease persistence failed", lease.snapshot)

    def _call(self, method: str, path: str, deadline: float,
              body: dict[str, object] | None = None) -> CloudResponse:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise ProvisioningError("cloud lifecycle deadline exceeded")
        failed = False
        result = None
        try:
            result = self._transport.request(method, path, body,
                                             timeout_seconds=min(remaining, self._request),
                                             max_bytes=2 * 1024 * 1024)
            if (not isinstance(result, CloudResponse) or type(result.status) is not int
                    or not 100 <= result.status <= 599 or not isinstance(result.body, bytes)
                    or len(result.body) > 2 * 1024 * 1024 or self._clock() >= deadline):
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise ProvisioningError("cloud lifecycle request failed")
        return result

    @staticmethod
    def _json(reply: CloudResponse) -> dict[str, Any]:
        failed = False
        result = None
        try:
            result = strict_object(reply.body)
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise ProvisioningError("invalid cloud lifecycle response")
        return result

    def _body(self, intent: AcquisitionIntent) -> dict[str, object]:
        return {"name": intent.name, "image": self.spec.image, "disk": self.spec.disk_gb,
                "cloud": "SECURE", "dataCenterIds": [self.spec.data_center_id],
                "gpu": {"id": self.spec.gpu_id, "count": 1,
                        "minCudaVersion": self.spec.min_cuda_version},
                "env": {"PUBLIC_KEY": self.spec.ssh_public_key,
                        "CINEFORGE_SESSION": self.spec.session_id,
                        "CINEFORGE_LEASE": intent.owner_token},
                "ports": ["22/tcp"], "mounts": {}, "startSsh": True,
                "startJupyter": False, "globalNetworking": False,
                "registry": self.spec.registry_id}

    def _owned(self, lease: _Lease, pod: dict[str, Any]) -> None:
        failed = False
        try:
            env, gpu = pod["env"], pod["gpu"]
            if (pod["id"] != lease.snapshot.resource_id or pod["name"] != lease.intent.name
                    or pod["image"] != self.spec.image or pod["cloud"] != "SECURE"
                    or type(pod["disk"]) is not int or pod["disk"] != self.spec.disk_gb
                    or pod["mounts"] != {} or pod["ports"] != ["22/tcp"]
                    or pod["registry"] != self.spec.registry_id or pod.get("cluster") is not None
                    or not isinstance(env, dict)
                    or env.get("CINEFORGE_SESSION") != self.spec.session_id
                    or env.get("CINEFORGE_LEASE") != lease.intent.owner_token
                    or env.get("PUBLIC_KEY") != self.spec.ssh_public_key
                    or not isinstance(gpu, dict) or gpu.get("id") != self.spec.gpu_id
                    or type(gpu.get("count")) is not int or gpu["count"] != 1
                    or pod["dataCenterId"] not in (None, self.spec.data_center_id)
                    or type(pod["locked"]) is not bool
                    or pod["status"] not in {"PROVISIONING", "STARTING", "RUNNING", "EXITED",
                                             "ERROR", "TERMINATED"}
                    or not isinstance(pod["actions"], list)
                    or any(a not in ("start", "stop", "restart", "terminate")
                           for a in pod["actions"])):
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise ProvisioningError("cloud ownership or configuration verification failed",
                                    lease.snapshot)

    def _get(self, lease: _Lease, deadline: float) -> dict[str, Any] | None:
        reply = self._call("GET", "/v2/pods/" + str(lease.snapshot.resource_id), deadline)
        if reply.status == 404:
            return None
        if reply.status != 200:
            raise ProvisioningError("cloud worker inspection failed", lease.snapshot)
        pod = self._json(reply)
        self._owned(lease, pod)
        return pod

    def _known(self, identity: str) -> _Lease:
        if not _identifier(identity) or identity not in self._ids:
            raise ProvisioningError("cloud provisioner does not own this resource")
        return self._leases[self._ids[identity]]

    def _clean(self, lease: _Lease) -> None:
        deadline = self._clock() + self._cleanup
        lease.snapshot = replace(lease.snapshot, state="cleanup_pending")
        # Persistence failure must not prevent cleanup of an already known resource.
        try:
            self._emit(lease)
        except ProvisioningError:
            pass
        pod = self._get(lease, deadline)
        if pod is not None and pod["status"] != "TERMINATED":
            if pod["locked"] or "terminate" not in pod["actions"]:
                raise ProvisioningError("owned worker cannot be released", lease.snapshot)
            try:
                self._call("DELETE", "/v2/pods/" + str(lease.snapshot.resource_id), deadline)
            except ProvisioningError:
                pass  # A lost acknowledgement is reconciled by a fresh owned-ID GET.
        for _ in range(self._max_polls):
            if self._get(lease, deadline) is None:
                self._emit(lease, state="released")
                return
            self._sleep(min(self._interval, max(0, deadline - self._clock())))
        raise ProvisioningError("owned worker release was not confirmed", lease.snapshot)

    def release(self, device_id: str) -> None:
        with self._lock:
            lease = self._known(device_id)
            if lease.snapshot.state == "released":
                return
            failed = False
            try:
                self._clean(lease)
            except (Exception, KeyboardInterrupt):
                failed = True
            if failed:
                lease.snapshot = replace(lease.snapshot, state="cleanup_uncertain")
                try:
                    self._emit(lease)
                except ProvisioningError:
                    pass
                raise CleanupUncertainError("owned worker cleanup is not confirmed", lease.snapshot)

    def inspect_lease(self, resource_id: str) -> LeaseSnapshot:
        with self._lock:
            lease = self._known(resource_id)
            pod = self._get(lease, self._clock() + self._request)
            if pod is None:
                self._emit(lease, state="released")
            elif _money(pod.get("cost")):
                self._emit(lease, hourly_usd=float(pod["cost"]))
            else:
                raise ProvisioningError("invalid worker hourly rate", lease.snapshot)
            return lease.snapshot

    def _measurement(self, lease: _Lease, timeout: float) -> WorkerObservation | None:
        failed = False
        observation = None
        try:
            observation = self._probe(lease.snapshot, timeout)
            if observation is not None and (
                    not isinstance(observation, WorkerObservation)
                    or observation.device.device_id != lease.snapshot.resource_id
                    or observation.raw_gpu_name != self.spec.gpu_id
                    or compatibility(observation.device, lease.intent.requirements)):
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise ProvisioningError("measured worker is invalid or incompatible", lease.snapshot)
        return observation

    def inspect(self, device_id: str) -> GPUDevice:
        with self._lock:
            lease = self._known(device_id)
            deadline = self._clock() + self._request
            pod = self._get(lease, deadline)
            if pod is None or pod["status"] != "RUNNING" or pod.get("runtime") is None:
                raise ProvisioningError("owned worker is not ready for measurement", lease.snapshot)
            observation = self._measurement(lease, max(0, deadline - self._clock()))
            if observation is None or self._clock() >= deadline:
                raise ProvisioningError("worker measurement did not complete", lease.snapshot)
            lease.observation = observation
            return observation.device

    def acquire(self, requirements: ModelSpec, *, max_cost_usd: float,
                idempotency_key: str | None = None,
                cancelled: Callable[[], bool] = lambda: False) -> GPUDevice:
        if (not isinstance(requirements, ModelSpec) or not _money(max_cost_usd)
                or not callable(cancelled)):
            raise ProvisioningError("invalid cloud acquisition requirements")
        key = uuid.uuid4().hex if idempotency_key is None else idempotency_key
        if not _identifier(key):
            raise ProvisioningError("invalid cloud idempotency key")
        with self._lock:
            if key in self._leases:
                old = self._leases[key]
                if (old.intent.requirements != requirements
                        or old.intent.max_cost_usd != max_cost_usd):
                    raise ProvisioningError("cloud idempotency key has a different request")
                if old.snapshot.state == "ready" and old.observation is not None:
                    return old.observation.device
                raise AcquisitionUncertainError("cloud acquisition key cannot be replayed",
                                                old.snapshot)
            if any(item.snapshot.state not in {"rejected", "released", "not_created"}
                   for item in self._leases.values()):
                raise AcquisitionUncertainError("another cloud lease needs reconciliation")
            intent = AcquisitionIntent(
                key, self.spec, uuid.uuid4().hex,
                "cineforge-" + self.spec.session_id + "-" + hashlib.sha256(key.encode()).hexdigest()[:16],
                requirements, max_cost_usd, self._lease_seconds, self._startup, self._request,
            )
            lease = _Lease(intent, LeaseSnapshot(key, "reserved"))
            self._leases[key] = lease
            reason = "invalid_worker"
            created_request = False
            rejected = False
            failed = False
            deadline = self._clock() + self._startup
            try:
                reason = "event_sink"
                self._emit(lease)
                self._before(intent)
                if cancelled():
                    reason = "cancelled"
                    raise ValueError
                reason = "transport"
                created_request = True
                reply = self._call("POST", "/v2/pods", deadline, self._body(intent))
                if 400 <= reply.status < 500 and reply.status != 408:
                    rejected = True
                    reason = "rejected"
                    raise ValueError
                if reply.status not in (200, 201):
                    raise ValueError
                pod = self._json(reply)
                identity = pod.get("id")
                if not _identifier(identity) or identity in self._ids:
                    raise ValueError
                lease.snapshot = replace(lease.snapshot, resource_id=identity, state="created")
                self._ids[identity] = key
                reason = "event_sink"
                self._emit(lease)  # Identity is retained before persistence/probe can fail.
                for _ in range(self._max_polls):
                    if cancelled():
                        reason = "cancelled"
                        raise ValueError
                    reason = "startup_timeout"
                    if self._clock() >= deadline:
                        raise ValueError
                    reason = "invalid_worker"
                    actual = self._get(lease, deadline)
                    if actual is None:
                        raise ValueError
                    rate = actual.get("cost")
                    if not _money(rate) or rate > self.spec.max_hourly_usd:
                        raise ValueError
                    self._emit(lease, hourly_usd=float(rate))
                    if actual["status"] in ("EXITED", "ERROR", "TERMINATED"):
                        raise ValueError
                    if actual["status"] == "RUNNING" and actual.get("runtime") is not None:
                        cuda = actual.get("cudaVersion")
                        if (not isinstance(cuda, str) or not re.fullmatch(r"\d+\.\d+", cuda)
                                or tuple(map(int, cuda.split("."))) < tuple(map(
                                    int, self.spec.min_cuda_version.split(".")))):
                            raise ValueError
                        observation = self._measurement(lease, min(
                            self._request, max(0, deadline - self._clock())))
                        if observation is not None:
                            if cancelled() or self._clock() >= deadline:
                                reason = "cancelled" if cancelled() else "startup_timeout"
                                raise ValueError
                            lease.observation = observation
                            self._emit(lease, state="ready")
                            return observation.device
                    reason = "startup_timeout"
                    self._sleep(min(self._interval, max(0, deadline - self._clock())))
            except (Exception, KeyboardInterrupt):
                failed = True
            if not failed:
                reason = "startup_timeout"
            if lease.snapshot.resource_id is not None:
                identity = lease.snapshot.resource_id
                lease.snapshot = replace(lease.snapshot, reason=reason)
                self.release(identity)
            else:
                state = "rejected" if rejected else "create_uncertain" if created_request \
                    else "not_created"
                try:
                    self._emit(lease, state=state, reason=reason)
                except ProvisioningError:
                    pass
            if created_request and lease.snapshot.resource_id is None and not rejected:
                raise AcquisitionUncertainError("cloud create outcome is not confirmed",
                                                lease.snapshot)
            raise ProvisioningError("cloud acquisition did not succeed", lease.snapshot)

    def restore_owned(self, intent: AcquisitionIntent, snapshot: LeaseSnapshot) -> None:
        """Attach only trusted operator journal records; never discover/adopt by name alone.

        No network call or mutation here. inspect/release must still prove remote
        ownership/configuration. Unknown-ID creates remain blocked for reconciliation.
        """
        with self._lock:
            if (not isinstance(intent, AcquisitionIntent) or not isinstance(snapshot, LeaseSnapshot)
                    or intent.spec_digest != self.spec.digest or intent.key != snapshot.key
                    or intent.key in self._leases
                    or snapshot.resource_id is not None and snapshot.resource_id in self._ids
                    or intent.name != "cineforge-" + self.spec.session_id + "-"
                    + hashlib.sha256(intent.key.encode()).hexdigest()[:16]):
                raise ProvisioningError("persisted cloud lease does not match operator configuration")
            self._leases[intent.key] = _Lease(intent, snapshot)
            if snapshot.resource_id is not None:
                self._ids[snapshot.resource_id] = intent.key
