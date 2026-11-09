"""Private, bounded SSH setup/heartbeats for a journal-owned disposable worker."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import math
import os
import shlex
import signal
import socket
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from types import TracebackType
from typing import Any, Callable, Protocol

from cineforge import worker_bootstrap, worker_guard
from cineforge.cloud_budget import CloudBudgetSession
from cineforge.comfy_policy import ApprovedComfySelection, WorkerVerification
from cineforge.comfy_runtime import ClosedComfyRuntime
from cineforge.comfy_transport import (
    ComfyHTTPTransport,
    request_capped,
    request_deadline,
    require_time,
)
from cineforge.devices import Backend, GPUDevice
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.runpod import LeaseSnapshot, RunpodPodSpec, WorkerObservation
from cineforge.runpod_transport import CloudTransport
from cineforge.s2v import S2VSelection, StagedS2VPair
from cineforge.s2v_service import S2VServiceContext
from cineforge.worker_bundle import worker_source
from cineforge.worker_guard import GuardReceipt, WorkerGuardPlan, worker_guard_source_sha256
from cineforge.worker_transport_source import worker_command_source
from cineforge.workflows import canonical, strict_object


class NativeInvoker(Protocol):
    def __call__(self, timeout: float, *, absolute_deadline: float) -> Any: ...


class WorkerError(ValueError):
    """Only fixed local diagnostics; never expose SSH/remote/API exception strings."""


MAX_CONTROL_NAMESPACES = 8


def private_file(path: str | Path, *, max_bytes: int, read: bool = True) -> bytes:
    """External owned0600 input, no links or Git ancestors; never log path or bytes."""
    descriptor = -1
    failed = False
    data = b""
    try:
        parts = Path(path).absolute().parts
        descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
        for index, part in enumerate(parts[1:]):
            if (not part or part in (".", "..") or not part.isprintable()
                    or any(c in part for c in "\\:%\x00")):
                raise ValueError
            PrivateRunDirectory._outside_git(descriptor)
            child = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                | (os.O_DIRECTORY if index < len(parts) - 2 else 0), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        parent = Path(path).absolute().parent.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > max_bytes or parent.st_uid != os.getuid()
                or stat.S_IMODE(parent.st_mode) & 0o077):
            raise ValueError
        if read:
            data = os.read(descriptor, max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError
    except (Exception, KeyboardInterrupt):
        failed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if failed:
        raise WorkerError("external private credential file rejected")
    return data


def ssh_environment() -> dict[str, str]:
    # The subprocess receives no account key, agent socket, proxy or SendEnv values.
    return {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
            "LANG": "C", "LC_ALL": "C"}


def run_process(command: list[str], *, timeout: float, payload: bytes = b"",
                check: Callable[[], None] = lambda: None, max_output: int = 65536,
                decode_setup_failure: bool = False,
                absolute_deadline: float | None = None) -> bytes:
    """No shell; bounded request/response, killed process group on timeout/interruption."""
    failed = False
    answer = None
    bounds: dict[str, Any] = {} if absolute_deadline is None else {"absolute_deadline": absolute_deadline}
    try:
        answer = worker_bootstrap.bounded_process(command, timeout=timeout, payload=payload,
            max_output=max_output, check=check, environment=ssh_environment(),
            decode_setup_failure=decode_setup_failure,
            **bounds)
    except worker_bootstrap.SetupFailure:
        raise
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or answer is None:
        raise WorkerError("bounded SSH process failed")
    return answer


def ssh_public_key(private_key: str | Path) -> str:
    private_file(private_key, max_bytes=16384, read=False)
    value = run_process(["/usr/bin/ssh-keygen", "-y", "-f", str(Path(private_key).absolute())],
                        timeout=5, max_output=8192).decode("ascii").strip()
    # RunpodPodSpec performs the complete public-key syntax check.
    return value


class SSHWorker:
    """One lease/driver; isolated TOFU and bounded retained control namespaces."""

    def __init__(self, spec: RunpodPodSpec, selection: ApprovedComfySelection | S2VServiceContext,
                 budget: CloudBudgetSession, transport: CloudTransport, private_key: str | Path,
                 state_dir: str | Path, *, allow_live: bool = False) -> None:
        if allow_live is not True:
            raise WorkerError("SSH worker needs explicit live opt-in")
        if type(selection) is S2VSelection:
            raise WorkerError("M5 worker acquisition requires its static service context")
        private_file(private_key, max_bytes=16384, read=False)
        self.spec, self.selection, self.budget, self.transport = spec, selection, budget, transport
        self.private_key, self.state_dir = Path(private_key).absolute(), Path(state_dir)
        if any(c.isspace() or not c.isprintable() for c in str(self.state_dir.absolute())):
            raise WorkerError("SSH state path cannot contain whitespace or controls")
        self.plan: WorkerGuardPlan | None = None
        self.observation: WorkerObservation | None = None
        self._host, self._port = "", 0
        self._stop, self._failed = Event(), Event()
        self._remote_suspended, self._remote_lock = Event(), Lock()
        self._thread: Thread | None = None
        self._tunnel: subprocess.Popen[bytes] | None = None
        self._endpoint_lock, self._tunnel_lock = Lock(), RLock()
        self._control_dir: Path | None = None
        self._control_identity: tuple[int, int] | None = None
        self._control_socket_identity: tuple[int, int] | None = None
        self._retired_controls: list[
            tuple[Path, tuple[int, int] | None, tuple[int, int] | None]] = []
        self._endpoint = ""
        self._claimed: PrivateRunDirectory | None = None
        self._enforce: Callable[[], tuple[str, ...]] | None = None
        self._source = worker_source()
        from cineforge.conditioning import ConditioningSelection
        self._profile: dict[str, Any] | None = None
        if type(selection) is ConditioningSelection:
            self._profile = selection.worker_profile
        elif type(selection) is S2VServiceContext:
            selection.preflight()
            if (getattr(budget, "approval", None) != selection.approval
                    or (spec.image, spec.gpu_id, spec.data_center_id, spec.min_ram_per_gpu_gb,
                        spec.disk_gb, spec.min_cuda_version)
                    != (selection.approval.image, selection.approval.gpu_id,
                        selection.approval.region, selection.approval.min_ram_per_gpu_gb,
                        selection.approval.disk_gb, selection.approval.min_cuda)
                    or spec.registry_id is not None
                    or spec.max_hourly_usd > selection.approval.hourly_ceiling_usd):
                raise WorkerError("M5 service and owned budget scope differ")
            self._profile = selection.worker_profile
        self._s2v_lock = Lock()
        self._s2v_consumed: set[str] = set()
        self._s2v_setup_attempted = False
        self._prepared_s2v: WorkerVerification | None = None
        self._closed_comfy: ClosedComfyRuntime | None = None
        self._comfy_selection: S2VSelection | None = None
        self.runtime_evidence: dict[str, Any] | None = None
        self.setup_failure: dict[str, Any] | None = None
        self._reference_staged = False

    def __enter__(self) -> SSHWorker:
        self._claimed = PrivateRunDirectory(self.state_dir).__enter__()
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        try:
            self.close()
        finally:
            if self._claimed is not None:
                claimed, self._claimed = self._claimed, None
                claimed.__exit__(kind, value, traceback)

    def _connection(self, snapshot: LeaseSnapshot, timeout: float) -> bool:
        # A previous connection must not survive a newly missing/rejected observation.
        self._host, self._port = "", 0
        failed = False
        try:
            remaining = self.budget.remaining_seconds(snapshot.key)
            if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                    or type(remaining) not in (int, float) or not math.isfinite(remaining)
                    or remaining <= 0):
                raise ValueError
            timeout = min(10, timeout, remaining)
            deadline = time.monotonic() + timeout
            reply = self.transport.request("GET", "/v2/pods/" + str(snapshot.resource_id),
                timeout_seconds=timeout, max_bytes=1024*1024)
            if reply.status != 200 or len(reply.body) > 1024*1024:
                raise ValueError
            pod = strict_object(reply.body)
            intent = self.budget.guard_plan(snapshot.key)
            env = pod.get("env")
            if (intent.pod_id != snapshot.resource_id or intent.session_id != self.spec.session_id
                    or pod.get("id") != snapshot.resource_id or pod.get("status") != "RUNNING"
                    or pod.get("dataCenterId") != self.spec.data_center_id
                    or pod.get("image") != self.spec.image or pod.get("mounts") != {}
                    or not isinstance(env, dict)
                    or env.get("CINEFORGE_SESSION") != self.spec.session_id
                    or env.get("CINEFORGE_LEASE") != intent.lease_token):
                raise ValueError
            if time.monotonic() >= deadline:
                return False
            ssh = pod.get("ssh")
            if ssh is None:
                return False
            if not isinstance(ssh, dict):
                raise ValueError
            direct = ssh.get("direct")
            if direct is None:
                return False
            if not isinstance(direct, dict) or not isinstance(direct.get("host"), str):
                raise ValueError
            address = ipaddress.ip_address(direct["host"])
            unicast = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) \
                and address.ipv4_mapped is not None else address
            if (not address.is_global or address.is_multicast
                    or isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None
                    or not unicast.is_global or unicast.is_multicast
                    or direct.get("username") != "root" or type(direct.get("port")) is not int
                    or not 1 <= direct["port"] <= 65535):
                raise ValueError
            self._host, self._port = str(address), direct["port"]
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise WorkerError("owned worker connection metadata rejected")
        return True

    def _command(self) -> list[str]:
        if not self._host or not self._port:
            raise WorkerError("direct SSH endpoint unavailable")
        return ["/usr/bin/ssh", "-F", "/dev/null", "-T", "-i", str(self.private_key),
            "-p", str(self._port), "-l", "root", "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "ForwardAgent=no",
            "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
            "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "SendEnv=-*",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "UpdateHostKeys=no",
            "-o", "UserKnownHostsFile=" + str(self.state_dir.absolute() / "known-hosts"),
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=1", "-o", "ExitOnForwardFailure=yes"]

    def _request(self, request: dict[str, Any], timeout: float, *, supervise: bool = True,
                 absolute_deadline: float | None = None) \
            -> dict[str, Any]:
        if type(self.selection) is S2VServiceContext:
            self.selection.preflight()
        request = dict(request, allow_live=True, timeout_seconds=timeout,
                       guard_sha256=worker_guard_source_sha256())
        if self._profile is not None:
            request["profile"] = self._profile
        native = str(request.get("operation", "")).startswith("native_")
        if absolute_deadline is not None:
            timeout = min(timeout, absolute_deadline-time.monotonic())
            if timeout <= 0:
                raise WorkerError("absolute SSH deadline exhausted")
            request["timeout_seconds"] = timeout
        if type(self.selection) is S2VServiceContext:
            flags = "/usr/bin/python3.11 -I -B -S -c "
        else:
            flags = "python3 -I -B -S -c " if native else "python3 -I -c "
        rendered = worker_command_source(self._source)
        command = self._command() + [self._host, flags + shlex.quote(rendered)]
        diagnostic_setup = self._profile is not None and request.get("operation") == "setup"
        bounds: dict[str, Any] = {} if absolute_deadline is None else {"absolute_deadline": absolute_deadline}
        try:
            raw = run_process(command, timeout=timeout, payload=canonical(request),
                check=self.check if supervise else lambda: None,
                **({"decode_setup_failure": True} if diagnostic_setup else {}),
                **({"max_output": 256*1024} if request.get("operation") == "native_fetch" else {}),
                **bounds)
        except worker_bootstrap.SetupFailure as error:
            if diagnostic_setup and self.setup_failure is None:
                self.setup_failure = worker_bootstrap.checked_setup_failure(error.diagnostic)
            raise WorkerError("worker setup failed") from None
        if absolute_deadline is not None and time.monotonic() >= absolute_deadline:
            raise WorkerError("absolute SSH deadline exceeded")
        reply = strict_object(raw)
        if diagnostic_setup and set(reply) != {"verification", "runtime"}:
            raise WorkerError("worker setup response rejected")
        return reply

    def probe(self, snapshot: LeaseSnapshot, timeout: float) -> WorkerObservation | None:
        if self.plan is not None:
            self.check()
            if self.plan.pod_id != snapshot.resource_id:
                raise WorkerError("driver cannot adopt another lease")
            return self.observation
        started = time.monotonic()
        if not self._connection(snapshot, timeout):
            return None
        # This read-only connectivity check may retry while sshd boots. An ARM request
        # is never retried after a lost acknowledgement: provisioner cleans up the Pod.
        try:
            remaining = min(timeout - (time.monotonic() - started),
                            self.budget.remaining_seconds(snapshot.key))
            if remaining <= 0:
                return None
            self._request({"operation": "ping"}, min(5, remaining), supervise=False)
        except (Exception, KeyboardInterrupt):
            return None
        remaining = min(timeout - (time.monotonic() - started),
                        self.budget.remaining_seconds(snapshot.key))
        if remaining <= 0:
            return None
        plan = self.budget.guard_plan(snapshot.key)
        source = read_private_input(Path(worker_guard.__file__), max_bytes=65536)
        reply = self._request({"operation": "arm", "plan": plan.to_dict(),
            "guard_source": base64.b64encode(source).decode(),
            "guard_sha256": worker_guard_source_sha256(), "controller_unix": time.time()},
            remaining, supervise=False)
        receipt = GuardReceipt.from_dict(reply["receipt"])
        self.budget.arm_guard(snapshot.key, receipt)
        runtime = reply["runtime"]
        if self._profile is not None:
            self.runtime_evidence = self._runtime_evidence(runtime,
                s2v=type(self.selection) is S2VServiceContext, admission=True,
                profile_id=self._profile["id"])
        self.observation = WorkerObservation(GPUDevice(str(snapshot.resource_id), Backend.CUDA,
            runtime["gpu_memory_mib"], frozenset({self.selection.model.required_precision})),
            runtime["gpu_raw_name"])
        self.plan, self._key = plan, snapshot.key
        self.budget.heartbeat(self._key)
        self._thread = Thread(target=self._supervise, daemon=True)
        self._thread.start()
        return self.observation

    def _supervise(self) -> None:
        while not self._stop.wait(10):
            try:
                self.check()
                assert self.plan is not None
                with self._remote_lock:
                    if not self._remote_suspended.is_set():
                        self._request({"operation": "heartbeat", "plan": self.plan.to_dict()},
                            min(5, self.budget.remaining_seconds(self._key)), supervise=False)
                self.budget.heartbeat(self._key)
            except (Exception, KeyboardInterrupt):
                self._failed.set()
                try:
                    self._close_tunnel()
                except WorkerError:
                    pass
                break

    def check(self) -> None:
        failed = False
        try:
            if self._stop.is_set():
                raise ValueError
            if type(self.selection) is S2VServiceContext and worker_source() != self._source:
                raise ValueError
            # Never hold the tunnel lock across provider/budget locks or callbacks.
            if self._enforce is not None and self._enforce():
                raise ValueError
            if (self._failed.is_set()
                    or self.plan is not None and self.budget.remaining_seconds(self._key) <= 0
                    or self._tunnel is not None and self._tunnel.poll() is not None):
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            try:
                self._close_tunnel()
            except WorkerError:
                pass
            raise WorkerError("worker supervision or lease deadline failed")

    def set_enforcer(self, enforce: Callable[[], tuple[str, ...]]) -> None:
        if self._enforce is not None or not callable(enforce):
            raise WorkerError("worker budget enforcer already configured")
        self._enforce = enforce

    def suspend_remote_heartbeat(self) -> WorkerGuardPlan:
        """Smoke fault injection only: keep local budget supervision, stop worker refreshes."""
        self.check()
        if self.plan is None:
            raise WorkerError("worker guard not armed")
        with self._remote_lock:
            self._remote_suspended.set()
        return self.plan

    def prepare(self, selection: S2VSelection | None = None, *,
                runtime: ClosedComfyRuntime | None = None,
                absolute_deadline: float | None = None) -> WorkerVerification:
        if type(self.selection) is not S2VServiceContext and (runtime is not None or absolute_deadline is not None):
            raise WorkerError("closed runtime parameters apply only to the M5 service")
        # Capture M5's cap before source hashing, rights or external review callbacks.
        deadline = (request_deadline(1200, absolute_deadline, time.monotonic)
                    if type(self.selection) is S2VServiceContext else None)
        self.check()
        if self.plan is None:
            raise WorkerError("worker guard not armed")
        timeout = min(1200, self.budget.remaining_seconds(self._key) - 120)
        if timeout <= 0:
            raise WorkerError("insufficient lease time for setup and shot retrieval")
        if type(self.selection) is S2VServiceContext:
            if type(runtime) is not ClosedComfyRuntime:
                raise WorkerError("M5 requires a separately checked closed Comfy runtime")
            if self._s2v_setup_attempted:
                raise WorkerError("M5 service setup already attempted; inspect owned worker state")
            assert deadline is not None
            deadline = min(deadline, time.monotonic() + timeout)
            self._require_s2v(selection)
            assert selection is not None
            runtime._admit(self, deadline)
            if time.monotonic() >= deadline:
                raise WorkerError("M5 preparation original deadline exhausted")
            self._s2v_setup_attempted = True
            return self._prepare_closed_comfy(selection, runtime, deadline)
        # Legacy M3/M4 policy and response declarations remain unchanged.
        reply = self._request({"operation": "setup", "plan": self.plan.to_dict(),
            "selection": self.selection.declarations, "image_digest": self.spec.image.split("@", 1)[1]}, timeout)
        result = WorkerVerification.from_dict(reply["verification"])
        result.require_matches(self.selection)
        if self._profile is not None:
            self.runtime_evidence = self._runtime_evidence(reply["runtime"])
        if (result.record["worker_id"] != self.plan.pod_id
                or result.record["image_digest"] != self.spec.image.split("@", 1)[1]):
            raise WorkerError("worker verification identity differs")
        return result

    def _prepare_closed_comfy(self, selection: S2VSelection, runtime: ClosedComfyRuntime,
                              deadline: float) -> WorkerVerification:
        result = None
        failed = False
        observed = None
        try:
            if type(self.selection) is not S2VServiceContext:
                raise WorkerError("closed Comfy requires the selected M5 service")
            reply = runtime.prepare(self, selection, deadline=deadline)
            worker_bootstrap.validate_s2v_runtime(reply["runtime"], profile_id=self.selection.worker_profile["id"])
            result = WorkerVerification.from_dict(reply["verification"])
            result.require_matches(self.selection)
            observed = self._runtime_evidence(reply["runtime"], s2v=True, profile_id=self.selection.worker_profile["id"])
            assert self.plan is not None
            if (result.record["worker_id"] != self.plan.pod_id
                    or result.record["image_digest"] != self.spec.image.split("@", 1)[1]):
                raise ValueError
            self._require_s2v(selection)
            self.check()
            if time.monotonic() >= deadline:
                raise ValueError
        except BaseException:
            failed = True
            runtime._poison()
        if failed or result is None or observed is None:
            raise WorkerError("M5 Comfy preparation incomplete or unconfirmed; never replay")
        self.runtime_evidence = observed
        self._closed_comfy, self._comfy_selection = runtime, selection
        self._prepared_s2v = result
        return result

    @staticmethod
    def _runtime_evidence(runtime: dict[str, Any], *, s2v: bool = False,
                          admission: bool = False, profile_id: str = "m5-wan22-s2v14b-5090-v1") -> dict[str, Any]:
        if s2v and admission and (type(runtime) is not dict or "admission_status" not in runtime):
            raise WorkerError("M5 arm cannot attest an unreviewed imported runtime")
        if s2v and type(runtime) is dict and "admission_status" in runtime:
            worker_bootstrap.validate_s2v_admission(runtime, profile_id=profile_id)
            if len(canonical(runtime)) > 4096:
                raise WorkerError("M5 hardware admission exceeds bound")
            return strict_object(canonical(runtime))
        fields = {"python_version", "torch_version", "torchvision_version", "cuda_version",
                  "gpu_raw_name", "gpu_memory_mib", "cuda_available", "arch_list", "capability",
                  "cuda_kernel_ok", "device_count", "nvml_total_mib",
                  "nvml_reserved_mib", "driver_version"} | worker_bootstrap.M4_RESOURCE_FIELDS
        if s2v:
            fields |= {"compute_precision"}
        if set(runtime) != fields or len(canonical(runtime)) > 4096:
            raise WorkerError("invalid bounded M4 runtime evidence")
        if s2v:
            worker_bootstrap.validate_s2v_runtime(runtime, profile_id=profile_id)
        else:
            worker_bootstrap.validate_m4_runtime(runtime)
        return strict_object(canonical(runtime))

    def _require_s2v(self, selection: S2VSelection | None) -> None:
        failed = False
        try:
            if (type(self.selection) is not S2VServiceContext or type(selection) is not S2VSelection
                    or selection.gate.service != self.selection or self.plan is None
                    or self.runtime_evidence is None):
                raise ValueError
            worker_bootstrap.validate_s2v_admission(self.runtime_evidence, profile_id=self.selection.worker_profile["id"])
            trace = selection.gate.preflight()
            selection.request(selection.gate.shot)
            if trace["speech_native_proof"]["worker_id"] != self.plan.pod_id:
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise WorkerError("current S2V request differs from guarded native worker scope")

    def native_admission(self) -> None:
        """Fresh work gate only; historical native provenance needs no live worker."""
        if (type(self.selection) is not S2VServiceContext or self.plan is None
                or self.runtime_evidence is None):
            raise WorkerError("native speech needs an existing guarded M5 service")
        worker_bootstrap.validate_s2v_admission(self.runtime_evidence, profile_id=self.selection.worker_profile["id"])
        self.selection.preflight()
        self.check()
        if self.budget.remaining_seconds(self._key) <= 120:
            raise WorkerError("insufficient owned lease for native speech")

    def native_request(self, request: dict[str, Any], *, absolute_deadline: float) -> dict[str, Any]:
        self.native_admission()
        if (type(absolute_deadline) not in (int, float) or not math.isfinite(absolute_deadline)
                or request.get("operation") not in {"native_capabilities", "native_preflight", "native_begin", "native_chunk",
                    "native_finish", "native_setup", "native_select", "native_invoke", "native_fetch",
                    "native_comfy_install", "native_comfy_model", "native_comfy_setup", "native_comfy_verify",
                    "native_comfy_precision_begin", "native_comfy_precision_finish"}):
            raise WorkerError("invalid fixed native dispatch")
        timeout = min(1200, absolute_deadline-time.monotonic(),
                      self.budget.remaining_seconds(self._key)-120)
        if timeout <= 0:
            raise WorkerError("native speech absolute deadline exhausted")
        assert self.plan is not None
        result = self._request(dict(request, plan=self.plan.to_dict()), timeout,
                               absolute_deadline=absolute_deadline)
        self.native_admission()
        if time.monotonic() >= absolute_deadline:
            raise WorkerError("native speech absolute deadline exceeded")
        return result

    def run_guarded_native(self, invoke: NativeInvoker, *, allow_live: bool = False) -> Any:
        """Invoke trusted controller-local code once under the current M5 guard.

        The callback must forward the original same-host absolute deadline to
        native103 or the separately reviewed remote108 controller. Remote code
        derives its own local cap; this method does not transport execution.
        No code/command comes from JSON. The callback owns bounded work/cleanup;
        admission and the current guard are checked before and after return.
        """
        if (allow_live is not True or type(self.selection) is not S2VServiceContext
                or self.plan is None or self.runtime_evidence is None or not callable(invoke)):
            raise WorkerError("native speech needs explicit guarded M5 service opt-in")
        worker_bootstrap.validate_s2v_admission(self.runtime_evidence, profile_id=self.selection.worker_profile["id"])
        self.selection.preflight()
        self.check()
        timeout = min(1200, self.budget.remaining_seconds(self._key) - 120)
        if timeout <= 0:
            raise WorkerError("insufficient owned lease for native speech")
        deadline = time.monotonic() + timeout
        failed = False
        result = None
        try:
            result = invoke(timeout, absolute_deadline=deadline)
            self.check()
            if time.monotonic() >= deadline:
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise WorkerError("guarded native speech incomplete or unconfirmed")
        return result

    @property
    def s2v_verification(self) -> WorkerVerification | None:
        """Observed preparation snapshot only; not a materialized graph authorization."""
        return self._prepared_s2v

    def stage_s2v(self, selection: S2VSelection, *,
                  absolute_deadline: float | None = None) -> StagedS2VPair:
        """One consumed transfer under a trusted same-host cap; never replay lost ACKs.

        Capture before lock wait or foreign admission. Remote workers receive only
        remaining durations, never this controller's monotonic timestamp.
        """
        deadline = None
        try:
            candidate = request_deadline(90, absolute_deadline, time.monotonic)
            sampled = time.monotonic()
            remaining = self.budget.remaining_seconds(self._key)
            if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                    or remaining <= 120):
                raise ValueError
            candidate = min(candidate, sampled + remaining - 120)
            require_time(candidate, time.monotonic)
            deadline = candidate
        except (Exception, KeyboardInterrupt):
            pass
        if deadline is None:
            raise WorkerError("invalid or expired private paired-transfer deadline")

        def window() -> None:
            valid = False
            try:
                require_time(deadline, time.monotonic)
                remaining = self.budget.remaining_seconds(self._key)
                if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                        or remaining <= 120):
                    raise ValueError
                require_time(deadline, time.monotonic)
                valid = True
            except (Exception, KeyboardInterrupt):
                pass
            if not valid:
                raise WorkerError("private paired transfer reached its deadline or cleanup reserve")

        with self._s2v_lock:
            window()
            self.check()
            self._require_s2v(selection)
            if self._prepared_s2v is None or self.plan is None:
                raise WorkerError("S2V service model/worker preparation is required before staging")
            self._prepared_s2v.require_matches(selection)
            self.verify_comfy_current(absolute_deadline=deadline)
            window()
            media = selection.gate.media
            scope = selection.gate.service.approval.subject_sha256
            subject = selection.approved_subject_sha256
            pair = hashlib.sha256(canonical({"scope_sha256": scope, "subject_sha256": subject,
                                             "media": media})).hexdigest()
            if pair in self._s2v_consumed or len(self._s2v_consumed) >= 4:
                raise WorkerError("S2V pair transfer already consumed or limit exceeded")
            window()
            self._s2v_consumed.add(pair)
            stage_id = uuid.uuid4().hex
            common = {"plan": self.plan.to_dict(), "stage_id": stage_id,
                      "subject_sha256": subject, "media": media}
            failed = False
            result = None
            try:
                def call(operation: str, **fields: Any) -> dict[str, Any]:
                    self.check()
                    self._require_s2v(selection)
                    timeout = min(10, deadline - time.monotonic(),
                                  self.budget.remaining_seconds(self._key) - 120)
                    if timeout <= 0:
                        raise ValueError
                    window()
                    reply = self._request(dict(common, operation=operation, **fields), timeout,
                                          absolute_deadline=deadline)
                    self.check()
                    window()
                    return reply
                beginning = call("s2v_pair_begin")
                if (set(beginning) != {"stage_id", "pair_sha256", "reused"}
                        or beginning["stage_id"] != stage_id or beginning["pair_sha256"] != pair
                        or set(beginning["reused"]) != {"image", "audio"}
                        or any(type(value) is not bool for value in beginning["reused"].values())):
                    raise ValueError
                for kind, payload in (("image", selection.gate.actor.read()),
                                      ("audio", selection.gate.speech.payload)):
                    if beginning["reused"][kind]:
                        continue
                    for offset in range(0, len(payload), 65536):
                        chunk = payload[offset:offset + 65536]
                        reply = call("s2v_pair_chunk", kind=kind, offset=offset,
                                     chunk=base64.b64encode(chunk).decode())
                        if reply != {"stage_id": stage_id, "pair_sha256": pair, "kind": kind,
                                     "offset": offset + len(chunk)}:
                            raise ValueError
                answer = call("s2v_pair_finish")
                if answer != {"stage_id": stage_id, "pair_sha256": pair, "scope_sha256": scope,
                        "subject_sha256": subject, "media": media, "worker_id": self.plan.pod_id}:
                    raise ValueError
                self._require_s2v(selection)
                self._comfy_selection = selection
                result = StagedS2VPair(self.plan.pod_id, scope, subject, canonical(media).decode())
            except (Exception, KeyboardInterrupt):
                failed = True
            if failed or result is None:
                raise WorkerError("private paired transfer incomplete or unconfirmed; never replay")
        # Lock release and every owned finalizer precede the last admission.
        self.check()
        self._require_s2v(selection)
        window()
        return result

    def stage_reference(self) -> dict[str, Any]:
        from cineforge.conditioning import ConditioningSelection
        if (type(self.selection) is not ConditioningSelection or self._profile is None
                or self.plan is None or self._reference_staged):
            raise WorkerError("reference staging is unavailable or already consumed")
        self.selection.gate.preflight()
        payload = self.selection.gate.reference.read()
        timeout = min(30, self.budget.remaining_seconds(self._key) - 60)
        if timeout <= 0:
            raise WorkerError("insufficient owned lease for reference staging")
        # Consume even uncertain outcomes. A lost acknowledgement never invites overwrite/retry.
        self._reference_staged = True
        reply = self._request({"operation": "stage_reference", "plan": self.plan.to_dict(),
            "payload": base64.b64encode(payload).decode()}, timeout)
        if reply != {"reference": self._profile["reference"]}:
            raise WorkerError("staged reference evidence differs")
        return reply["reference"]  # type: ignore[no-any-return]

    def _control_ready(self) -> bool:
        with self._tunnel_lock:
            if self._control_dir is None:
                raise WorkerError("owned SSH control path unavailable")
            info = self._control_dir.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o700
                    or (info.st_dev, info.st_ino) != self._control_identity):
                raise WorkerError("owned SSH control directory changed")
            try:
                control = (self._control_dir / "c").lstat()
            except FileNotFoundError:
                return False
            identity = (control.st_dev, control.st_ino)
            if (not stat.S_ISSOCK(control.st_mode) or control.st_uid != os.getuid()
                    or control.st_nlink != 1 or self._control_socket_identity is not None
                    and identity != self._control_socket_identity):
                raise WorkerError("owned SSH control socket changed")
            self._control_socket_identity = identity
            return True

    def _close_tunnel(self) -> None:
        failed = False
        with self._tunnel_lock:
            self._endpoint = ""
            process = self._tunnel
            if process is not None:
                killed = True
                try:
                    # A departed leader may leave descendants in its owned group.
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except (Exception, KeyboardInterrupt):
                    killed = False
                    failed = True
                try:
                    if type(process.wait(timeout=2)) is not int:
                        raise ValueError
                    if killed:
                        self._tunnel = None
                except (Exception, KeyboardInterrupt):
                    failed = True
            if self._tunnel is None and self._control_dir is not None:
                try:
                    self._control_ready()
                    # POSIX unlink/rmdir cannot condition deletion on an inode.
                    # Retain every namespace: even a replacement after this check
                    # is never deleted, traversed or reused by the driver.
                    self._retired_controls.append((self._control_dir,
                        self._control_identity, self._control_socket_identity))
                    self._control_dir = None
                    self._control_identity = self._control_socket_identity = None
                except (Exception, KeyboardInterrupt):
                    failed = True
        if failed:
            raise WorkerError("owned SSH tunnel cleanup is not confirmed") from None

    @property
    def retained_control_namespace_paths(self) -> tuple[Path, ...]:
        """Private residual paths for operator cleanup, not current ownership proof."""
        with self._tunnel_lock:
            paths = tuple(record[0] for record in self._retired_controls)
            return paths + ((self._control_dir,) if self._control_dir is not None else ())

    def _tunnel_remaining(self, deadline: float) -> float:
        self.check()
        reserve = 120 if type(self.selection) is S2VServiceContext else 0
        lease = self.budget.remaining_seconds(self._key) - reserve
        remaining = min(deadline - time.monotonic(), lease)
        if not math.isfinite(remaining) or remaining <= 0:
            raise WorkerError("private tunnel readiness deadline exceeded")
        return remaining

    def _mux(self, operation: str, deadline: float, forward: str | None = None) -> None:
        with self._tunnel_lock:
            directory = self._control_dir
            if directory is None or not self._control_ready():
                raise WorkerError("owned SSH control path unavailable")
            control = directory / "c"
        # Failed mux HELLO can fall back even with -O. Replace the first value:
        # appending ProxyCommand would be shadowed by _command's earlier "none".
        base = ["ProxyCommand=/usr/bin/false" if value == "ProxyCommand=none" else value
                for value in self._command()]
        command = base + ["-S", str(control), "-o", "ControlMaster=no", "-O", operation]
        if forward is not None:
            command.extend(["-L", forward])
        command.append(self._host)
        def check() -> None:
            self._tunnel_remaining(deadline)
        run_process(command, timeout=min(1, self._tunnel_remaining(deadline)), max_output=1024,
                    check=check, absolute_deadline=deadline)
        self._tunnel_remaining(deadline)
        if not self._control_ready():
            raise WorkerError("owned SSH control socket disappeared")

    def endpoint(self, *, absolute_deadline: float | None = None) -> str:
        """Bound tunnel setup to a same-host parent cap, including owned finalizers."""
        deadline = None
        try:
            candidate = request_deadline(10, absolute_deadline, time.monotonic)
            sampled = time.monotonic()
            reserve = 120 if type(self.selection) is S2VServiceContext else 0
            remaining = self.budget.remaining_seconds(self._key)
            if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                    or remaining <= reserve):
                raise ValueError
            candidate = min(candidate, sampled + remaining - reserve)
            require_time(candidate, time.monotonic)
            deadline = candidate
        except (Exception, KeyboardInterrupt):
            pass
        if deadline is None:
            raise WorkerError("invalid or expired private tunnel deadline")
        endpoint = None
        try:
            endpoint = self._endpoint_with_deadline(deadline)
            self._tunnel_remaining(deadline)
        except (Exception, KeyboardInterrupt):
            endpoint = None
        if endpoint is None or time.monotonic() >= deadline:
            cleanup_failed = False
            try:
                self._close_tunnel()
            except WorkerError:
                cleanup_failed = True
            if cleanup_failed:
                raise WorkerError("private tunnel failed with unconfirmed cleanup")
            raise WorkerError("private owned tunnel readiness failed")
        return endpoint

    def _endpoint_with_deadline(self, deadline: float) -> str:
        with self._endpoint_lock:
            failed = cleanup_failed = False
            try:
                self._tunnel_remaining(deadline)
                if self._endpoint:
                    return self._endpoint
                if self.plan is None:
                    raise ValueError
                # A failed prior teardown must never be overwritten by another child.
                self._close_tunnel()
                self._tunnel_remaining(deadline)
                with self._tunnel_lock:
                    require_time(deadline, time.monotonic)
                    if len(self._retired_controls) >= MAX_CONTROL_NAMESPACES:
                        raise ValueError
                    self._control_dir = Path(tempfile.mkdtemp(
                        prefix="cf-ssh-" + self.spec.session_id + "-", dir="/tmp"))
                    info = self._control_dir.lstat()
                    self._control_identity = (info.st_dev, info.st_ino)
                    control = self._control_dir / "c"
                    if len(os.fsencode(control)) > 100:
                        raise ValueError
                    command = self._command() + ["-M", "-S", str(control),
                        "-o", "ControlPersist=no", "-N", self._host]
                    environment = ssh_environment()
                    require_time(deadline, time.monotonic)
                    self._tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment,
                        start_new_session=True)
                for _ in range(101):
                    self._tunnel_remaining(deadline)
                    if self._control_ready():
                        try:
                            self._mux("check", deadline)
                            break
                        except WorkerError:
                            pass
                    time.sleep(min(.1, self._tunnel_remaining(deadline)))
                else:
                    raise ValueError
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                # Native mux acknowledgement proves this master owns the bind.
                # The mux-only deny proxy prevents a failed HELLO from connecting directly.
                self._mux("forward", deadline, f"127.0.0.1:{port}:127.0.0.1:8188")
                endpoint = f"http://127.0.0.1:{port}"
                health = ComfyHTTPTransport(endpoint, allow_live=True)
                for _ in range(101):
                    timeout = min(1, self._tunnel_remaining(deadline))
                    try:
                        raw = request_capped(health, "GET", "/system_stats", None,
                            timeout_seconds=timeout, max_bytes=65536,
                            absolute_deadline=deadline, clock=time.monotonic)
                    except Exception:
                        time.sleep(min(.1, self._tunnel_remaining(deadline)))
                        continue
                    data = strict_object(raw)
                    if (set(data) != {"system", "devices"} or not isinstance(data["system"], dict)
                            or not isinstance(data["devices"], list)
                            or any(not isinstance(device, dict) for device in data["devices"])):
                        raise ValueError
                    self._tunnel_remaining(deadline)
                    with self._tunnel_lock:
                        if (not self._control_ready() or self._tunnel is None
                                or self._tunnel.poll() is not None or self._stop.is_set()
                                or self._failed.is_set()):
                            raise ValueError
                        require_time(deadline, time.monotonic)
                        self._endpoint = endpoint
                        return endpoint
                raise ValueError
            except (Exception, KeyboardInterrupt):
                failed = True
            finally:
                if failed:
                    try:
                        self._close_tunnel()
                    except WorkerError:
                        cleanup_failed = True  # Keep uncertain references for close/retry.
            if cleanup_failed:
                raise WorkerError("private tunnel failed with unconfirmed cleanup")
            raise WorkerError("private owned tunnel readiness failed")

    def execution_transport(self, *, absolute_deadline: float | None = None) -> SupervisedComfyTransport:
        # Constructor validation and the final fresh guard belong to the original
        # same-host setup window too; never derive a new cap after either one.
        transport = None
        try:
            deadline = request_deadline(10, absolute_deadline, time.monotonic)
            sampled = time.monotonic()
            reserve = 120 if type(self.selection) is S2VServiceContext else 0
            remaining = self.budget.remaining_seconds(self._key)
            if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                    or remaining <= reserve):
                raise ValueError
            deadline = min(deadline, sampled + remaining - reserve)
            require_time(deadline, time.monotonic)
            candidate = SupervisedComfyTransport(self, ComfyHTTPTransport(
                self.endpoint(absolute_deadline=deadline), allow_live=True))
            self._tunnel_remaining(deadline)
            transport = candidate
        except (Exception, KeyboardInterrupt):
            pass
        if transport is None:
            cleanup_failed = False
            try:
                self._close_tunnel()
            except WorkerError:
                cleanup_failed = True
            if cleanup_failed:
                raise WorkerError("private transport handoff failed with unconfirmed cleanup")
            raise WorkerError("private transport handoff failed current guard or deadline")
        return transport

    def verify_comfy_current(self, *, absolute_deadline: float | None = None) -> None:
        if type(self.selection) is not S2VServiceContext:
            return
        if type(self._closed_comfy) is not ClosedComfyRuntime or self._comfy_selection is None:
            raise WorkerError("current fixed Comfy runtime proof required")
        deadline = None
        try:
            candidate = request_deadline(120, absolute_deadline, time.monotonic)
            sampled = time.monotonic()
            remaining = self.budget.remaining_seconds(self._key)
            if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                    or remaining <= 120):
                raise ValueError
            candidate = min(candidate, sampled + remaining - 120)
            require_time(candidate, time.monotonic)
            deadline = candidate
        except (Exception, KeyboardInterrupt):
            pass
        if deadline is None:
            raise WorkerError("invalid or expired current Comfy verification deadline")
        valid = False
        try:
            self._closed_comfy.verify(self, self._comfy_selection, deadline=deadline)
            require_time(deadline, time.monotonic)
            valid = True
        except (Exception, KeyboardInterrupt):
            pass
        if not valid:
            raise WorkerError("current Comfy verification incomplete or past original deadline")

    def close(self) -> None:
        self._stop.set()
        failed = False
        try:
            if self._thread is not None:
                self._thread.join(timeout=7)
        except (Exception, KeyboardInterrupt):
            failed = True
        try:
            self._close_tunnel()
        except WorkerError:
            failed = True
        if failed:
            raise WorkerError("owned SSH worker cleanup is not confirmed") from None


class SupervisedComfyTransport:
    def __init__(self, worker: SSHWorker, transport: ComfyHTTPTransport) -> None:
        self.worker, self.transport = worker, transport

    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int,
                absolute_deadline: float | None = None,
                clock: Callable[[], float] | None = None) -> bytes:
        clock = time.monotonic if clock is None else clock
        deadline = request_deadline(timeout_seconds, absolute_deadline, clock)
        self.worker.check()
        if (type(self.worker.selection) is S2VServiceContext
                and method == "POST" and path == "/prompt"):
            self.worker.verify_comfy_current(absolute_deadline=deadline)
            assert self.worker._closed_comfy is not None
            self.worker._closed_comfy.require_precision_post(self.worker, body, deadline=deadline)
        reserve = 120 if type(self.worker.selection) is S2VServiceContext else 0
        remaining = min(require_time(deadline, clock),
                        self.worker.budget.remaining_seconds(self.worker._key) - reserve)
        deadline = min(deadline, clock() + remaining)
        require_time(deadline, clock)
        value = request_capped(self.transport, method, path, body,
            timeout_seconds=remaining, max_bytes=max_bytes, absolute_deadline=deadline, clock=clock)
        self.worker.check()
        require_time(deadline, clock)
        if self.worker.budget.remaining_seconds(self.worker._key) <= reserve:
            raise WorkerError("ComfyUI response exceeded the owned lease reserve")
        return value
