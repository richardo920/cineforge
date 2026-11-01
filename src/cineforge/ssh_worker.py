"""Private, bounded SSH setup/heartbeats for a journal-owned disposable worker."""

from __future__ import annotations

import base64
import ipaddress
import math
import os
import shlex
import signal
import socket
import stat
import subprocess
import time
from pathlib import Path
from threading import Event, Lock, Thread
from types import TracebackType
from typing import Any, Callable

from cineforge import worker_bootstrap, worker_guard
from cineforge.cloud_budget import CloudBudgetSession
from cineforge.comfy_policy import ApprovedComfySelection, WorkerVerification
from cineforge.comfy_transport import ComfyHTTPTransport
from cineforge.devices import Backend, GPUDevice
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.runpod import LeaseSnapshot, RunpodPodSpec, WorkerObservation
from cineforge.runpod_transport import CloudTransport
from cineforge.worker_guard import GuardReceipt, WorkerGuardPlan, worker_guard_source_sha256
from cineforge.workflows import canonical, strict_object


class WorkerError(ValueError):
    """Only fixed local diagnostics; never expose SSH/remote/API exception strings."""


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
                check: Callable[[], None] = lambda: None, max_output: int = 65536) -> bytes:
    """No shell; bounded request/response, killed process group on timeout/interruption."""
    if (not isinstance(payload, bytes) or len(payload) > 256 * 1024
            or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise WorkerError("invalid process bounds")
    process = None
    failed = False
    answer = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=ssh_environment(), start_new_session=True)
        data = bytearray()
        overflow = Event()

        def read() -> None:
            assert process is not None and process.stdout is not None
            try:
                while block := process.stdout.read(4096):
                    if len(data) + len(block) > max_output:
                        overflow.set()
                        break
                    data.extend(block)
            except OSError:
                overflow.set()

        def write() -> None:
            assert process is not None and process.stdin is not None
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except OSError:
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        reader, writer = Thread(target=read, daemon=True), Thread(target=write, daemon=True)
        reader.start()
        writer.start()
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            check()
            if overflow.is_set() or time.monotonic() >= deadline:
                raise WorkerError("process bounds exceeded")
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
        reader.join(timeout=max(0, deadline - time.monotonic()))
        check()
        if reader.is_alive() or overflow.is_set() or process.returncode != 0:
            raise WorkerError("remote process failed")
        answer = bytes(data)
    except (Exception, KeyboardInterrupt):
        failed = True
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                failed = True
            if process.stdout is not None:
                process.stdout.close()
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
    """One lease/driver. First-contact host keys use isolated TOFU, never global config."""

    def __init__(self, spec: RunpodPodSpec, selection: ApprovedComfySelection,
                 budget: CloudBudgetSession, transport: CloudTransport, private_key: str | Path,
                 state_dir: str | Path, *, allow_live: bool = False) -> None:
        if allow_live is not True:
            raise WorkerError("SSH worker needs explicit live opt-in")
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
        self._endpoint = ""
        self._claimed: PrivateRunDirectory | None = None
        self._enforce: Callable[[], tuple[str, ...]] | None = None
        self._source = read_private_input(Path(worker_bootstrap.__file__), max_bytes=128*1024)

    def __enter__(self) -> SSHWorker:
        self._claimed = PrivateRunDirectory(self.state_dir).__enter__()
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()
        if self._claimed is not None:
            self._claimed.__exit__(kind, value, traceback)

    def _connection(self, snapshot: LeaseSnapshot, timeout: float) -> bool:
        reply = self.transport.request("GET", "/v2/pods/" + str(snapshot.resource_id),
            timeout_seconds=min(10, timeout), max_bytes=1024*1024)
        if reply.status != 200:
            raise WorkerError("owned worker connection unavailable")
        pod = strict_object(reply.body)
        intent = self.budget.guard_plan(snapshot.key)
        env = pod.get("env")
        if (pod.get("id") != snapshot.resource_id or pod.get("status") != "RUNNING"
                or pod.get("dataCenterId") != self.spec.data_center_id
                or pod.get("image") != self.spec.image or pod.get("mounts") != {}
                or not isinstance(env, dict) or env.get("CINEFORGE_SESSION") != self.spec.session_id
                or env.get("CINEFORGE_LEASE") != intent.lease_token):
            raise WorkerError("worker connection ownership differs")
        direct = pod["ssh"]["direct"]
        if direct is None:
            return False
        address = ipaddress.ip_address(direct["host"])
        unicast = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) \
            and address.ipv4_mapped is not None else address
        if (not address.is_global or address.is_multicast
                or not unicast.is_global or unicast.is_multicast or direct["username"] != "root"
                or type(direct["port"]) is not int or not 1 <= direct["port"] <= 65535):
            raise WorkerError("invalid direct SSH endpoint")
        self._host, self._port = str(address), direct["port"]
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

    def _request(self, request: dict[str, Any], timeout: float, *, supervise: bool = True) \
            -> dict[str, Any]:
        request = dict(request, allow_live=True, timeout_seconds=timeout,
                       guard_sha256=worker_guard_source_sha256())
        command = self._command() + [self._host, "python3 -c " + shlex.quote(self._source.decode())]
        raw = run_process(command, timeout=timeout, payload=canonical(request),
                          check=self.check if supervise else lambda: None)
        return strict_object(raw)

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
            self._request({"operation": "ping"}, min(5, timeout), supervise=False)
        except (Exception, KeyboardInterrupt):
            return None
        remaining = timeout - (time.monotonic() - started)
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
                break

    def check(self) -> None:
        if self._stop.is_set():
            raise WorkerError("worker supervision stopped")
        # Run admission's cleanup decision before interpreting zero remaining time;
        # the decision releases outside the budget lock and can race a blocked probe.
        if self._enforce is not None and self._enforce():
            raise WorkerError("worker lease or idle budget expired")
        if (self._failed.is_set()
                or self.plan is not None and self.budget.remaining_seconds(self._key) <= 0
                or self._tunnel is not None and self._tunnel.poll() is not None):
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

    def prepare(self) -> WorkerVerification:
        self.check()
        if self.plan is None:
            raise WorkerError("worker guard not armed")
        timeout = min(1200, self.budget.remaining_seconds(self._key) - 120)
        if timeout <= 0:
            raise WorkerError("insufficient lease time for setup and shot retrieval")
        reply = self._request({"operation": "setup", "plan": self.plan.to_dict(),
            "selection": self.selection.declarations, "image_digest": self.spec.image.split("@", 1)[1]},
            timeout)
        result = WorkerVerification.from_dict(reply["verification"])
        result.require_matches(self.selection)
        if (result.record["worker_id"] != self.plan.pod_id
                or result.record["image_digest"] != self.spec.image.split("@", 1)[1]):
            raise WorkerError("worker verification identity differs")
        return result

    def endpoint(self) -> str:
        self.check()
        if self._endpoint:
            return self._endpoint
        if self.plan is None:
            raise WorkerError("worker guard not armed")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = self._command() + ["-N", "-L", f"127.0.0.1:{port}:127.0.0.1:8188", self._host]
        self._tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ssh_environment(),
            start_new_session=True)
        deadline = time.monotonic() + min(10, self.budget.remaining_seconds(self._key))
        while time.monotonic() < deadline:
            self.check()
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    self._endpoint = f"http://127.0.0.1:{port}"
                    return self._endpoint
            except OSError:
                time.sleep(.1)
        raise WorkerError("private loopback tunnel unavailable")

    def execution_transport(self) -> SupervisedComfyTransport:
        return SupervisedComfyTransport(self, ComfyHTTPTransport(self.endpoint(), allow_live=True))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=7)
        if self._tunnel is not None:
            if self._tunnel.poll() is None:
                try:
                    os.killpg(self._tunnel.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._tunnel.wait(timeout=2)


class SupervisedComfyTransport:
    def __init__(self, worker: SSHWorker, transport: ComfyHTTPTransport) -> None:
        self.worker, self.transport = worker, transport

    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int) -> bytes:
        self.worker.check()
        value = self.transport.request(method, path, body, timeout_seconds=min(timeout_seconds,
            self.worker.budget.remaining_seconds(self.worker._key)), max_bytes=max_bytes)
        self.worker.check()
        return value
