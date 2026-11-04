"""Stdlib worker setup bundled with fixed reviewed profiles for disposable Wan workers.

Account credentials are never inputs. The separately copied guard owns shutdown.
Only selected version/hash/size fields leave this process; tool diagnostics do not.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

from cineforge.worker_profiles import (
    I2V_WORKER,
    S2V_PROFILE_IDS,
    S2V_WORKER,
    T2V_WORKER,
    WorkerProfile,
    s2v_worker,
)

# Existing inspection names remain available; operational data is immutable and shared.
REVISION = T2V_WORKER.runner_revision
GPU = T2V_WORKER.gpu
MODEL_REVISION = T2V_WORKER.model_revision
M4_PROFILE = I2V_WORKER.profile_id
M4_GPU = I2V_WORKER.gpu
M4_RESOURCE_FIELDS = {"host_memory_bytes", "host_available_memory_bytes", "cgroup_limits",
                      "memory_accounting", "memory_limit_bytes", "available_memory_bytes",
                      "filesystem_total_bytes", "filesystem_free_bytes"}
M4_REVISION = I2V_WORKER.model_revision
M4_FILES = {path: (size, digest) for path, size, digest in I2V_WORKER.files}


class BootstrapError(ValueError):
    """Static worker diagnostic, without subprocess, URL or environment content."""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


SETUP_REASONS = {
    "admission": {"setup_rejected", "unknown", "timeout"},
    "source": {"setup_rejected", "process_failed", "timeout", "unknown"},
    "environment": {"process_failed", "timeout", "unknown"},
    "dependencies": {"process_failed", "timeout", "unknown"},
    "runtime": {"setup_rejected", "process_failed", "timeout", "unknown"},
    "metadata": {"setup_rejected", "process_failed", "unknown", "timeout"},
    "weights": {"http_error", "network_error", "timeout", "integrity_failed",
                "setup_rejected", "unknown"},
    "verification": {"setup_rejected", "unknown", "timeout"},
    "server_start": {"server_exited", "unknown", "timeout"},
    "server_health": {"server_exited", "unknown", "timeout"},
}
SETUP_ASSETS = {"diffusion_model", "text_encoder", "vae"}


def checked_setup_failure(value: Any) -> dict[str, Any] | None:
    """Closed descriptive data only; never retains remote strings or arbitrary fields."""
    if not isinstance(value, dict):
        return None
    schema = value.get("schema")
    if type(schema) is not str or schema not in ("m4-setup-failure-v1", "m5-setup-failure-v1"):
        return None
    assets = SETUP_ASSETS | ({"audio_encoder"} if schema == "m5-setup-failure-v1" else set())
    base = {"schema", "stage", "asset", "reason"}
    if (set(value) != base | ({"http_status"} if value.get("reason") == "http_error" else set())
            or type(value.get("stage")) is not str or value["stage"] not in SETUP_REASONS
            or type(value.get("reason")) is not str
            or value["reason"] not in SETUP_REASONS[value["stage"]]
            or type(value.get("asset")) is not str
            or (value["asset"] not in assets if value["stage"] == "weights"
                else value["asset"] != "none")
            or value["reason"] == "http_error" and (
                type(value.get("http_status")) is not int or not 400 <= value["http_status"] <= 599)):
        return None
    return dict(value)


class SetupFailure(BootstrapError):
    """An allowlisted M4 setup description, never the underlying exception text."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        checked = checked_setup_failure(diagnostic)
        if checked is None:
            raise BootstrapError("invalid setup diagnostic")
        super().__init__("M4 worker setup failed" if checked["schema"] == "m4-setup-failure-v1"
                         else "M5 worker setup failed")
        self.diagnostic = checked


def setup_failure_envelope(raw: bytes) -> dict[str, Any] | None:
    """Accept exactly one small canonical failure frame, including its final newline."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 512:
        return None
    try:
        value = _json(raw)
        if (set(value) != {"status", "setup_failure"}
                or value["status"] != "worker_operation_failed"
                or encode(value) + b"\n" != raw):
            return None
        return checked_setup_failure(value["setup_failure"])
    except (ValueError, TypeError, RecursionError):
        return None


def validate_reference_png(payload: bytes) -> None:
    """Bounded static RGB8 PNG only; no Pillow, image downloads or metadata chunks."""
    if (not isinstance(payload, bytes) or not 0 < len(payload) <= 128*1024
            or payload[:8] != b"\x89PNG\r\n\x1a\n"):
        raise BootstrapError("unsupported reference image")
    offset, phase = 8, 0
    compressed = bytearray()
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise BootstrapError("malformed reference image")
        length = int.from_bytes(payload[offset:offset+4], "big")
        kind = payload[offset+4:offset+8]
        end = offset + 12 + length
        if end > len(payload):
            raise BootstrapError("malformed reference image")
        value = payload[offset+8:end-4]
        if zlib.crc32(kind + value) & 0xffffffff != int.from_bytes(payload[end-4:end], "big"):
            raise BootstrapError("reference image checksum differs")
        if kind == b"IHDR" and phase == 0:
            if value != struct.pack(">IIBBBBB", 1280, 704, 8, 2, 0, 0, 0):
                raise BootstrapError("reference image must be 1280x704 RGB8")
            phase = 1
        elif kind == b"IDAT" and phase in (1, 2):
            compressed.extend(value)
            phase = 2
        elif kind == b"IEND" and phase == 2 and length == 0 and end == len(payload):
            phase = 3
        else:
            raise BootstrapError("unsupported reference image chunk")
        offset = end
    if phase != 3:
        raise BootstrapError("incomplete reference image")
    expected = (1280*3+1)*704
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(bytes(compressed), expected+1)
        valid = (len(raw) == expected and decoder.eof and not decoder.unused_data
                 and not decoder.unconsumed_tail
                 and all(raw[i] <= 4 for i in range(0, expected, 1280*3+1)))
    except zlib.error:
        valid = False
    if not valid:
        raise BootstrapError("reference image exceeds decoded bounds or is corrupt")

def clean_environment() -> dict[str, str]:
    # Do not forward vendor/account keys, proxy/index credentials or user startup hooks.
    return {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1", "PIP_CONFIG_FILE": "/dev/null",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0"}


def bounded_process(command: list[str], *, timeout: float, cwd: Path | None = None,
                    payload: bytes | None = None, max_output: int = 65536,
                    check: Any = None, environment: dict[str, str] | None = None,
                    decode_setup_failure: bool = False,
                    absolute_deadline: float | None = None) -> bytes:
    """Bound wall time and buffered bytes; kill this process group on every failure."""
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
            or type(max_output) is not int or not 1 <= max_output <= 1024*1024
            or type(decode_setup_failure) is not bool
            or payload is not None and (not isinstance(payload, bytes) or len(payload) > 256*1024)):
        raise BootstrapError("invalid worker process bounds")
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    # A failed clock read cannot strand a newly spawned child; startup counts too.
    deadline = time.monotonic() + timeout
    if absolute_deadline is not None:
        if type(absolute_deadline) not in (int, float) or not math.isfinite(absolute_deadline):
            raise BootstrapError("invalid absolute worker deadline")
        deadline = min(deadline, absolute_deadline)
    result = b""
    try:
        if time.monotonic() >= deadline:
            raise BootstrapError("worker process deadline exhausted")
        process = subprocess.Popen(command, cwd=cwd,
            env=clean_environment() if environment is None else environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, bufsize=0)
        assert process.stdin is not None and process.stdout is not None
        output = bytearray()
        payload = payload or b""
        written = 0
        selector = selectors.DefaultSelector()
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        if payload:
            selector.register(process.stdin, selectors.EVENT_WRITE)
        else:
            process.stdin.close()
        while selector.get_map() or process.poll() is None:
            if time.monotonic() >= deadline:
                raise BootstrapError("worker process bounds exceeded", reason="timeout")
            if check is not None:
                check()
            for key, event in selector.select(min(.05, max(0, deadline-time.monotonic()))):
                if event & selectors.EVENT_READ:
                    try:
                        block = os.read(key.fd, min(4096, max_output+1-len(output)))
                    except BlockingIOError:
                        continue
                    output.extend(block)
                    if len(output) > max_output:
                        raise BootstrapError("worker process output exceeded its limit")
                    if not block:
                        selector.unregister(key.fileobj)
                        process.stdout.close()
                else:
                    try:
                        written += os.write(key.fd, payload[written:written+65536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        written = len(payload)
                    if written == len(payload):
                        selector.unregister(key.fileobj)
                        process.stdin.close()
        if process.returncode != 0:
            if decode_setup_failure and process.returncode == 2:
                diagnostic = setup_failure_envelope(bytes(output))
                if diagnostic is not None:
                    raise SetupFailure(diagnostic)
            raise BootstrapError("worker process did not complete", reason="process_failed")
        if check is not None:
            check()
        result = bytes(output)
    finally:
        # A completed parent can leave descendants holding its pipes open. Always
        # tear down the dedicated group; nonblocking pipes cannot hold close() hostage.
        if process is not None:
            try:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)
            finally:
                try:
                    if selector is not None:
                        selector.close()
                finally:
                    try:
                        if process.stdin is not None:
                            process.stdin.close()
                    finally:
                        if process.stdout is not None:
                            process.stdout.close()

    if absolute_deadline is not None and time.monotonic() >= deadline:
        raise BootstrapError("worker process cleanup exceeded absolute deadline")
    return result


def private_write(root: Path, name: str, payload: bytes, *, replace: bool = False) -> None:
    if re.fullmatch(r"[a-z0-9.-]{1,80}", name) is None:
        raise BootstrapError("invalid worker record name")
    temporary = root / (name + ".pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        if replace:
            old = os.stat(root / name, follow_symlinks=False)
            if not stat.S_ISREG(old.st_mode) or old.st_nlink != 1:
                raise BootstrapError("unsafe worker record")
            os.replace(temporary, root / name)
        else:
            os.link(temporary, root / name, follow_symlinks=False)
            temporary.unlink()
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BootstrapError("duplicate worker input")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise BootstrapError("invalid worker input")
    return value


def worker_root(session: str, *, create: bool = False) -> Path:
    if not isinstance(session, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", session) is None:
        raise BootstrapError("invalid worker session")
    root = Path("/workspace") / ("cineforge-m3-" + session)
    if create:
        root.mkdir(mode=0o700)
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077 or root.parent.is_symlink()):
        raise BootstrapError("unsafe worker directory")
    return root


def load_guard(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("cineforge_worker_guard", root / "guard.py")
    if spec is None or spec.loader is None:
        raise BootstrapError("worker guard unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def heartbeat(root: Path, plan: dict[str, Any], *, first: bool = False) -> None:
    private_write(root, "heartbeat.json", encode({"schema_version": 1,
        "pod_id": plan["pod_id"], "lease_token": plan["lease_token"],
        "heartbeat_unix": time.time()}), replace=not first)


def guard_alive(root: Path, pid: int) -> None:
    """Linux worker: reject a dead/zombie/reused PID rather than refreshing its heartbeat."""
    if type(pid) is not int or pid <= 0:
        raise BootstrapError("invalid worker guardian PID")
    valid = False
    try:
        os.kill(pid, 0)
        proc = Path("/proc") / str(pid)
        with (proc / "stat").open("rb") as stream:
            status = stream.read(8193)
        with (proc / "cmdline").open("rb") as stream:
            command = stream.read(8193)
        expected = [str(root / "guard.py").encode(), b"--plan", str(root / "plan.json").encode(),
                    b"--heartbeat", str(root / "heartbeat.json").encode(), b"--allow-live"]
        valid = (len(status) <= 8192 and len(command) <= 8192
                 and status.startswith(str(pid).encode() + b" (") and b")" in status
                 and status.rsplit(b")", 1)[-1].split()[0] in
                 (b"R", b"S", b"D", b"T", b"t", b"K", b"W", b"P", b"I")
                 and command.rstrip(b"\0").split(b"\0")[1:] == expected)
    except (Exception, KeyboardInterrupt):
        pass
    if not valid:
        raise BootstrapError("worker guardian is not the expected live process")



def memory_accounting(total: int, available: int, groups: list[dict[str, Any]],
                      version: str) -> dict[str, Any]:
    """Linux meminfo uses KiB; cgroup limits/current are bytes. No swap is counted."""
    if (type(total) is not int or type(available) is not int or not 0 <= available <= total
            or total <= 0 or version not in {"v1", "v2"}
            or not isinstance(groups, list) or not 1 <= len(groups) <= 64):
        raise BootstrapError("invalid memory accounting")
    effective_limit, headroom = total, available
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"limit_bytes", "usage_bytes"}:
            raise BootstrapError("invalid cgroup memory accounting")
        limit, usage = group["limit_bytes"], group["usage_bytes"]
        if (type(usage) is not int or usage < 0
                or limit is not None and (type(limit) is not int or limit <= 0)):
            raise BootstrapError("invalid cgroup memory accounting")
        if limit is not None:
            effective_limit = min(effective_limit, limit)
            headroom = min(headroom, max(0, limit - usage))
    return {"host_memory_bytes": total, "host_available_memory_bytes": available,
            "cgroup_limits": groups, "memory_accounting": version,
            "memory_limit_bytes": effective_limit, "available_memory_bytes": headroom}


def resource_evidence(root: Path, *, proc: Path = Path("/proc"),
                      cgroup: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    """Require this process's cgroup, including every visible ancestor constraint."""
    fields: dict[str, int] = {}
    for line in (proc / "meminfo").read_text().splitlines():
        match = re.fullmatch(r"(MemTotal|MemAvailable):\s+([0-9]+) kB", line)
        if match:
            fields[match[1]] = int(match[2]) * 1024
    groups: list[dict[str, Any]] = []
    membership = (proc / "self/cgroup").read_text().splitlines()
    selected = None
    for line in membership:
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise BootstrapError("unknown cgroup membership")
        if parts[0] == "0" and parts[1] == "":
            selected = ("v2", parts[2], cgroup, "memory.max", "memory.current", "cgroup.procs")
            break
        if "memory" in parts[1].split(","):
            selected = ("v1", parts[2], cgroup / "memory", "memory.limit_in_bytes",
                        "memory.usage_in_bytes", "cgroup.procs")
    if selected is None:
        raise BootstrapError("container memory limit unavailable")
    version, member, base, limit_file, usage_file, procs_file = selected
    member_path = Path(member)
    if not member_path.is_absolute() or ".." in member_path.parts:
        raise BootstrapError("unsupported cgroup membership path")
    candidates = [base / member.lstrip("/"), base]
    current = None
    for candidate in candidates:
        try:
            identities = (candidate / procs_file).read_text().splitlines()
        except OSError:
            continue
        if str(os.getpid()) in identities:
            current = candidate
            break
    if current is None:
        raise BootstrapError("current container cgroup not identified")
    while True:
        if current == base and groups and not (current / limit_file).exists():
            # A real cgroup-v2 hierarchy root has no memory.max; visible child/ancestor
            # bounds already apply. Namespace roots that expose a limit are still read.
            break
        limit = (current / limit_file).read_text().strip()
        usage = (current / usage_file).read_text().strip()
        if (not usage.isascii() or not usage.isdigit()
                or limit != "max" and (not limit.isascii() or not limit.isdigit())):
            raise BootstrapError("invalid cgroup memory values")
        groups.append({"limit_bytes": None if limit == "max" else int(limit),
                       "usage_bytes": int(usage)})
        if current == base:
            break
        current = current.parent
        if len(groups) >= 64 or not current.is_relative_to(base):
            raise BootstrapError("unsupported cgroup hierarchy")
    result = memory_accounting(fields.get("MemTotal", 0), fields.get("MemAvailable", -1),
                               groups, version)
    disk = shutil.disk_usage(root)
    return {**result, "filesystem_total_bytes": disk.total, "filesystem_free_bytes": disk.free}


def _validate_reference_runtime(runtime: dict[str, Any], configured: WorkerProfile) -> None:
    requirements = configured.reference_runtime
    if requirements is None:
        raise BootstrapError("reference runtime profile unavailable")
    fields = {"python_version", "torch_version", "torchvision_version", "cuda_version",
              "gpu_raw_name", "gpu_memory_mib", "cuda_available", "arch_list", "capability",
              "cuda_kernel_ok", "device_count", "nvml_total_mib",
              "nvml_reserved_mib", "driver_version"} | M4_RESOURCE_FIELDS
    if (set(runtime) != fields or runtime["gpu_raw_name"] != configured.gpu
            or runtime["cuda_available"] is not True or runtime["cuda_kernel_ok"] is not True
            or type(runtime["gpu_memory_mib"]) is not int
            or not configured.minimum_memory_mib <= runtime["gpu_memory_mib"]
            <= requirements.maximum_memory_mib
            or type(runtime["device_count"]) is not int or runtime["device_count"] != 1
            or runtime["capability"] != list(requirements.capability)
            or any(type(v) is not int for v in runtime["capability"])
            or runtime["torch_version"] != requirements.torch_version
            or runtime["cuda_version"] != configured.cuda_version
            or not isinstance(runtime["arch_list"], list)
            or not (requirements.architecture in runtime["arch_list"]
                or configured.profile_id == S2V_PROFILE_IDS[1] and "sm_86" in runtime["arch_list"])
            or len(runtime["arch_list"]) > 64
            or any(not isinstance(v, str) or re.fullmatch(r"(?:sm|compute)_[0-9a-z]{2,8}", v)
                   is None for v in runtime["arch_list"])
            or type(runtime["nvml_total_mib"]) is not int
            or not runtime["gpu_memory_mib"] <= runtime["nvml_total_mib"]
            <= requirements.maximum_memory_mib
            or type(runtime["nvml_reserved_mib"]) is not int
            or not 0 <= runtime["nvml_reserved_mib"] < runtime["nvml_total_mib"]
            or any(not isinstance(runtime[k], str) or re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}",
                   runtime[k]) is None for k in ("python_version", "torchvision_version"))
            or not isinstance(runtime["driver_version"], str)
            or re.fullmatch(r"[0-9.]{3,32}", runtime["driver_version"]) is None):
        raise BootstrapError("invalid bounded M4 runtime evidence")
    accounting = memory_accounting(runtime["host_memory_bytes"], runtime["host_available_memory_bytes"],
                                   runtime["cgroup_limits"], runtime["memory_accounting"])
    if (any(runtime[key] != value for key, value in accounting.items())
            or runtime["memory_limit_bytes"] < requirements.minimum_effective_ram_bytes
            or runtime["available_memory_bytes"] < requirements.minimum_available_ram_bytes
            or type(runtime["filesystem_free_bytes"]) is not int
            or type(runtime["filesystem_total_bytes"]) is not int
            or not requirements.minimum_filesystem_free_bytes <= runtime["filesystem_free_bytes"]
            <= runtime["filesystem_total_bytes"]):
        raise BootstrapError("M4 container memory or filesystem headroom insufficient")


def validate_m4_runtime(runtime: dict[str, Any]) -> None:
    _validate_reference_runtime(runtime, I2V_WORKER)

def checked_profile(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("id") in S2V_PROFILE_IDS:
        if (set(value) != {"id", "scope_sha256"} or type(value["scope_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value["scope_sha256"]) is None):
            raise BootstrapError("invalid closed M5 service scope")
        return value
    if not isinstance(value, dict) or set(value) != {"id", "reference"} or value["id"] != M4_PROFILE:
        raise BootstrapError("unknown worker profile")
    reference = value["reference"]
    if (not isinstance(reference, dict) or set(reference) != {"sha256", "size_bytes", "basename"}
            or not isinstance(reference["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
            or type(reference["size_bytes"]) is not int
            or not 0 < reference["size_bytes"] <= 128*1024
            or reference["basename"] != "ref-" + reference["sha256"] + ".png"):
        raise BootstrapError("invalid bound reference metadata")
    return value


def _configuration(value: Any) -> WorkerProfile:
    checked = checked_profile(value)
    return T2V_WORKER if checked is None else (
        s2v_worker(checked["id"]) if checked["id"] in S2V_PROFILE_IDS else I2V_WORKER)


def validate_s2v_runtime(runtime: dict[str, Any], *,
                         profile_id: str = S2V_WORKER.profile_id) -> None:
    """Exact selected M5 resources plus the observed BF16 attention-kernel probe."""
    configured = s2v_worker(profile_id)
    if (profile_id == S2V_PROFILE_IDS[1] and (type(runtime.get("python_version")) is not str
            or re.fullmatch(r"3\.11\.[0-9]{1,3}", runtime["python_version"]) is None)):
        raise BootstrapError("M5 fixed interpreter ABI unavailable")
    if runtime.get("compute_precision") != "bf16":
        raise BootstrapError("M5 BF16 compute capability unavailable")
    _validate_reference_runtime({key: value for key, value in runtime.items()
                                 if key != "compute_precision"}, configured)


def validate_s2v_admission(runtime: dict[str, Any], *,
                           profile_id: str = S2V_WORKER.profile_id) -> None:
    """NVML/stdlib hardware only; no claim that unreviewed Python imports are ready."""
    configured = s2v_worker(profile_id)
    if "admission_status" not in runtime:
        validate_s2v_runtime(runtime, profile_id=profile_id)
        return
    fields = {"admission_status", "python_version", "gpu_raw_name", "gpu_memory_mib",
              "capability", "device_count", "nvml_total_mib", "nvml_reserved_mib",
              "driver_version"} | M4_RESOURCE_FIELDS
    requirements = configured.reference_runtime
    if (requirements is None or set(runtime) != fields
            or runtime["admission_status"] != "hardware_only_unverified_runtime"
            or runtime["gpu_raw_name"] != configured.gpu
            or runtime["capability"] != list(requirements.capability)
            or any(type(v) is not int for v in runtime["capability"])
            or type(runtime["device_count"]) is not int or runtime["device_count"] != 1
            or type(runtime["gpu_memory_mib"]) is not int
            or not configured.minimum_memory_mib <= runtime["gpu_memory_mib"] <= requirements.maximum_memory_mib
            or runtime["nvml_total_mib"] != runtime["gpu_memory_mib"]
            or type(runtime["nvml_reserved_mib"]) is not int
            or not 0 <= runtime["nvml_reserved_mib"] < runtime["nvml_total_mib"]
            or type(runtime["python_version"]) is not str
            or re.fullmatch(r"3\.11\.[0-9]{1,3}", runtime["python_version"]) is None
            or type(runtime["driver_version"]) is not str
            or re.fullmatch(r"[0-9.]{3,32}", runtime["driver_version"]) is None):
        raise BootstrapError("M5 unimported hardware admission unavailable")
    accounting = memory_accounting(runtime["host_memory_bytes"], runtime["host_available_memory_bytes"],
                                   runtime["cgroup_limits"], runtime["memory_accounting"])
    if (any(runtime[k] != v for k, v in accounting.items())
            or runtime["memory_limit_bytes"] < requirements.minimum_effective_ram_bytes
            or runtime["available_memory_bytes"] < requirements.minimum_available_ram_bytes
            or type(runtime["filesystem_free_bytes"]) is not int
            or type(runtime["filesystem_total_bytes"]) is not int
            or not requirements.minimum_filesystem_free_bytes <= runtime["filesystem_free_bytes"]
            <= runtime["filesystem_total_bytes"]):
        raise BootstrapError("M5 hardware resources unavailable")


def s2v_hardware_admission(root: Path, deadline: float, *,
                           profile_id: str = S2V_WORKER.profile_id) -> dict[str, Any]:
    s2v_worker(profile_id)
    import platform
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11)
            or sys.platform != "linux" or platform.machine() != "x86_64"):
        raise BootstrapError("M5 fixed interpreter ABI unavailable")
    raw = bounded_process(["/usr/bin/nvidia-smi", "--query-gpu=name,memory.total,memory.reserved,compute_cap,driver_version",
        "--format=csv,noheader,nounits"], timeout=min(5, deadline-time.monotonic()),
        max_output=1024, environment={"PATH": "", "LANG": "C"}, absolute_deadline=deadline)
    rows = raw.decode("ascii").strip().splitlines()
    values = rows[0].split(",") if len(rows) == 1 else []
    if len(values) != 5:
        raise BootstrapError("M5 bounded NVML hardware evidence unavailable")
    result = dict(admission_status="hardware_only_unverified_runtime", python_version=platform.python_version(),
        gpu_raw_name=values[0].strip(), gpu_memory_mib=int(values[1]), nvml_total_mib=int(values[1]),
        nvml_reserved_mib=int(values[2]), capability=[int(v) for v in values[3].strip().split(".")],
        device_count=1, driver_version=values[4].strip(), **resource_evidence(root))
    validate_s2v_admission(result, profile_id=profile_id)
    if time.monotonic() >= deadline:
        raise BootstrapError("M5 hardware admission deadline exhausted")
    return result


def runtime_probe(timeout: float, python: str = sys.executable, *,
                  profile: dict[str, Any] | None = None, resource_root: Path | None = None) \
        -> dict[str, Any]:
    configured = _configuration(profile)
    if configured.profile_id in S2V_PROFILE_IDS:
        raise BootstrapError("M5 imports require the closed Comfy production launcher")
    requirements = configured.reference_runtime
    deadline = time.monotonic() + timeout
    code = ("import json,platform,torch,importlib.metadata as m;"
            "p=torch.cuda.get_device_properties(0);"
            "print(json.dumps(dict(python_version=platform.python_version(),"
            "torch_version=torch.__version__,torchvision_version=m.version('torchvision'),"
            "cuda_version=torch.version.cuda,gpu_raw_name=p.name,"
            "gpu_memory_mib=p.total_memory//1048576,cuda_available=torch.cuda.is_available())))")
    if requirements is not None:
        code = ("import json,os,platform,torch,importlib.metadata as m;"
                "p=torch.cuda.get_device_properties(0);"
                "x=torch.ones((1,1,2,16),device='cuda');"
                "y=torch.nn.functional.scaled_dot_product_attention(x,x,x);"
                "print(json.dumps(dict(python_version=platform.python_version(),"
                "torch_version=torch.__version__,torchvision_version=m.version('torchvision'),"
                "cuda_version=torch.version.cuda,gpu_raw_name=p.name,"
                "gpu_memory_mib=p.total_memory//1048576,cuda_available=torch.cuda.is_available(),"
                "arch_list=torch.cuda.get_arch_list(),capability=list(torch.cuda.get_device_capability()),"
                "device_count=torch.cuda.device_count(),"
                "cuda_kernel_ok=bool(torch.isfinite(y).all().item()))))")
        if configured.profile_id in S2V_PROFILE_IDS:
            code = code.replace("device='cuda');", "device='cuda',dtype=torch.bfloat16);")
            code = code.replace("cuda_kernel_ok=", "compute_precision='bf16' if x.dtype == torch.bfloat16 else 'invalid',cuda_kernel_ok=")
    runtime = _json(bounded_process([python, "-c", code], timeout=timeout))
    if (runtime["cuda_available"] is not True or runtime["gpu_raw_name"] != configured.gpu
            or type(runtime["gpu_memory_mib"]) is not int
            or runtime["gpu_memory_mib"] < configured.minimum_memory_mib):
        raise BootstrapError("worker runtime incompatible")
    for name in ("python_version", "torch_version", "torchvision_version", "cuda_version"):
        if re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", runtime[name]) is None:
            raise BootstrapError("invalid worker runtime version")
    if (re.fullmatch(r"\d{1,2}\.\d{1,2}", runtime["cuda_version"]) is None
            or tuple(map(int, runtime["cuda_version"].split(".")))
            < tuple(map(int, configured.cuda_version.split(".")))):
        raise BootstrapError("worker CUDA runtime incompatible")
    if requirements is not None and (runtime.get("torch_version") != requirements.torch_version
            or runtime.get("cuda_version") != configured.cuda_version
            or runtime.get("capability") != list(requirements.capability)
            or runtime.get("device_count") != 1 or type(runtime.get("device_count")) is not int
            or not isinstance(runtime.get("arch_list"), list)
            or requirements.architecture not in runtime["arch_list"]
            or runtime.get("cuda_kernel_ok") is not True):
        raise BootstrapError("M4 worker stack or kernel incompatible")
    if requirements is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BootstrapError("M4 runtime probe deadline exceeded")
        output = bounded_process(["nvidia-smi", "--id=0",
            "--query-gpu=memory.total,memory.reserved,driver_version",
            "--format=csv,noheader,nounits"], timeout=min(5, remaining), max_output=4096)
        values = output.decode("ascii").strip().split(",")
        if (len(values) != 3 or not all(v.strip().isdigit() for v in values[:2])
                or not re.fullmatch(r"[0-9.]{3,32}", values[2].strip())):
            raise BootstrapError("M4 NVML memory evidence unavailable")
        total, reserved = (int(v.strip()) for v in values[:2])
        if (not runtime["gpu_memory_mib"] <= total <= requirements.maximum_memory_mib
                or not 0 <= reserved < total):
            raise BootstrapError("M4 NVML and CUDA memory evidence incompatible")
        runtime.update(nvml_total_mib=total, nvml_reserved_mib=reserved,
                       driver_version=values[2].strip())
        runtime.update(resource_evidence(resource_root or Path("/workspace")))
        if configured.profile_id in S2V_PROFILE_IDS:
            validate_s2v_runtime(runtime, profile_id=configured.profile_id)
        else:
            validate_m4_runtime(runtime)
    return runtime


def arm(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    profile = checked_profile(request.get("profile"))
    plan = request["plan"]
    source = base64.b64decode(request["guard_source"], validate=True)
    if len(source) > 65536 or hashlib.sha256(source).hexdigest() != request["guard_sha256"]:
        raise BootstrapError("worker guard source differs")
    root = worker_root(plan["session_id"], create=True)
    private_write(root, "guard.py", source)
    private_write(root, "plan.json", encode(plan))
    if profile is not None:
        private_write(root, "profile.json", encode(profile))
    heartbeat(root, plan, first=True)
    guard = load_guard(root)
    parsed = guard.WorkerGuardPlan.from_dict(plan)
    if abs(time.time() - request["controller_unix"]) > 5:
        raise BootstrapError("worker clock differs")
    if not guard.verify_own_pod(parsed, allow_live=True):
        raise BootstrapError("worker own-Pod credential verification failed")
    process = subprocess.Popen([sys.executable, str(root / "guard.py"), "--plan",
        str(root / "plan.json"), "--heartbeat", str(root / "heartbeat.json"), "--allow-live"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    time.sleep(.1)
    if process.poll() is not None:
        raise BootstrapError("worker guard exited")
    guard_alive(root, process.pid)
    private_write(root, "guard-pid.json", encode({"pid": process.pid}))
    runtime = (s2v_hardware_admission(root, deadline, profile_id=profile["id"]) if profile is not None
               and profile["id"] in S2V_PROFILE_IDS else
               runtime_probe(max(.01, deadline - time.monotonic()), profile=profile, resource_root=root)
               if profile is not None else runtime_probe(max(.01, deadline - time.monotonic())))
    return {"receipt": {"plan": plan, "source_sha256": hashlib.sha256(source).hexdigest(),
        "pid": process.pid, "observed_unix": time.time(), "operator_verified_alive": True,
        "operator_verified_own_pod": True}, "runtime": runtime}


class _DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> Any:
        from urllib.parse import urlsplit
        url = urlsplit(newurl)
        host = url.hostname or ""
        if (url.scheme != "https" or url.username or url.password
                or not (host == "huggingface.co" or host.endswith(".huggingface.co")
                        or host.endswith(".hf.co") or host.endswith(".xethub.hf.co"))):
            raise BootstrapError("unapproved model redirect", reason="setup_rejected")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(root: Path, asset: dict[str, Any], deadline: float, *,
             profile: dict[str, Any] | None = None) -> dict[str, Any]:
    configured = _configuration(profile)
    revision, distribution = configured.model_revision, configured.distribution
    path = asset["path"]
    folders = "diffusion_models|text_encoders|vae" + ("|audio_encoders" if configured.profile_id in S2V_PROFILE_IDS else "")
    if (not isinstance(path, str) or re.fullmatch(r"split_files/(" + folders + r")"
            r"/[A-Za-z0-9._-]+\.safetensors", path) is None
            or asset["revision"] != revision
            or asset["url"] != "https://huggingface.co/Comfy-Org/" + distribution + "/resolve/"
            + revision + "/" + path or type(asset["size_bytes"]) is not int
            or not 0 < asset["size_bytes"] <= configured.maximum_asset_bytes
            or configured.files and (path, asset["size_bytes"], asset["sha256"])
            not in configured.files
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None):
        raise BootstrapError("unapproved worker model asset", reason="setup_rejected")
    _, folder, filename = path.split("/")
    target = root / "ComfyUI/models" / folder / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    digest, size = hashlib.sha256(), 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _DownloadRedirect())
    try:
        with opener.open(asset["url"], timeout=min(30, max(.01, deadline-time.monotonic()))) \
                as reply, partial.open("xb") as stream:
            while True:
                if time.monotonic() >= deadline:
                    raise BootstrapError("model download deadline exceeded", reason="timeout")
                block = reply.read1(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > asset["size_bytes"]:
                    raise BootstrapError("model download size exceeded", reason="integrity_failed")
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if size != asset["size_bytes"] or digest.hexdigest() != asset["sha256"]:
            raise BootstrapError("model download differs from approved bytes", reason="integrity_failed")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return {"path": path, "sha256": digest.hexdigest(), "size_bytes": size}


def setup(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    profile = checked_profile(request.get("profile"))
    progress = {"stage": "admission", "asset": "none"} if profile is not None else None
    try:
        return _setup(request, deadline, progress)
    except (Exception, KeyboardInterrupt) as error:
        if progress is None:
            raise
        reason = "unknown"
        status: dict[str, Any] = {}
        if isinstance(error, urllib.error.HTTPError) and type(error.code) is int and 400 <= error.code <= 599:
            reason, status = "http_error", {"http_status": error.code}
        elif isinstance(error, TimeoutError):
            reason = "timeout"
        elif isinstance(error, urllib.error.URLError):
            reason = "timeout" if isinstance(error.reason, TimeoutError) else "network_error"
        elif isinstance(error, BootstrapError):
            reason = error.reason
        if reason not in SETUP_REASONS[progress["stage"]]:
            reason, status = "unknown", {}
        schema = "m5-setup-failure-v1" if profile is not None and profile["id"] in S2V_PROFILE_IDS else "m4-setup-failure-v1"
        raise SetupFailure({"schema": schema, **progress,
                            "reason": reason, **status}) from None


def _setup(request: dict[str, Any], deadline: float,
           progress: dict[str, str] | None) -> dict[str, Any]:
    def stage(name: str, asset: str = "none") -> None:
        if progress is not None:
            progress.update(stage=name, asset=asset)

    plan, selection = request["plan"], request["selection"]
    profile = checked_profile(request.get("profile"))
    configured = _configuration(profile)
    revision = configured.model_revision
    root = worker_root(plan["session_id"])
    if configured.profile_id in S2V_PROFILE_IDS:
        raise BootstrapError("M5 requires the separately reviewed fixed Comfy installer", reason="setup_rejected")
    if configured.profile_id in S2V_PROFILE_IDS:
        subject = selection.get("approval", {}).get("model_workflow", {}).get("approved_subject_sha256")
        if type(subject) is not str or re.fullmatch(r"[0-9a-f]{64}", subject) is None:
            raise BootstrapError("M5 setup requires an exact materialized request subject")
        private_write(root, "s2v-setup-consumed.json", encode({
            "scope_sha256": profile["scope_sha256"] if profile is not None else None,
            "subject_sha256": subject}))
    if (selection["runner"]["revision"] != configured.runner_revision
            or selection["runner"]["custom_nodes"] != []
            or selection["model"]["revision"] != revision
            or request["image_digest"] != request["image_digest"].strip()
            or re.fullmatch(r"sha256:[0-9a-f]{64}", request["image_digest"]) is None):
        raise BootstrapError("worker setup differs from reviewed source", reason="setup_rejected")
    stage("runtime")
    limit = min(30, max(.01, deadline-time.monotonic()))
    runtime = (runtime_probe(limit, profile=profile, resource_root=root) if profile is not None
               else runtime_probe(limit))
    if runtime["gpu_memory_mib"] < selection["model"]["minimum_memory_mib"]:
        raise BootstrapError("worker memory below model requirement", reason="setup_rejected")
    comfy, venv = root / "ComfyUI", root / "venv"

    def run(command: list[str], limit: float = 180, cwd: Path | None = None) -> bytes:
        return bounded_process(command, timeout=min(limit, max(.01, deadline-time.monotonic())),
                               cwd=cwd)
    stage("source")
    run(["git", "init", str(comfy)])
    run(["git", "-C", str(comfy), "remote", "add", "origin",
         "https://github.com/Comfy-Org/ComfyUI.git"])
    run(["git", "-C", str(comfy), "fetch", "--depth", "1", "origin", configured.runner_revision])
    run(["git", "-C", str(comfy), "checkout", "--detach", "FETCH_HEAD"])
    if run(["git", "-C", str(comfy), "rev-parse", "HEAD"]).decode().strip() != configured.runner_revision:
        raise BootstrapError("worker source revision differs", reason="setup_rejected")
    stage("environment")
    run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], 60)
    python = str(venv / "bin/python")
    constraints = root / "constraints.txt"
    private_write(root, constraints.name, ("torch==" + runtime["torch_version"]
        + "\ntorchvision==" + runtime["torchvision_version"] + "\n").encode())
    stage("dependencies")
    run([python, "-m", "pip", "--isolated", "install", "--index-url", "https://pypi.org/simple",
         "--disable-pip-version-check", "--no-input", "--progress-bar", "off", "--quiet",
         "--constraint", str(constraints), "-r", str(comfy / "requirements.txt")], 500)
    stage("runtime")
    limit = min(30, max(.01, deadline-time.monotonic()))
    installed = (runtime_probe(limit, python, profile=profile, resource_root=root) if profile is not None
                 else runtime_probe(limit, python))
    if any(installed[name] != runtime[name] for name in ("torch_version", "cuda_version")):
        raise BootstrapError("worker GPU stack changed", reason="setup_rejected")
    stage("metadata")
    freeze = run([python, "-m", "pip", "--isolated", "freeze", "--all"], 30)
    private_write(root, "environment.json", encode({"source_revision": configured.runner_revision,
        "requirements_sha256": hashlib.sha256((comfy / "requirements.txt").read_bytes()).hexdigest(),
        "freeze_sha256": hashlib.sha256(freeze).hexdigest(),
        "dependency_policy": "source-pinned requirements; GPU stack constrained; resolved versions recorded",
        "resolved_packages": freeze.decode()}))
    requirements = configured.reference_runtime
    if requirements is not None:
        if (shutil.disk_usage(root).free < requirements.minimum_filesystem_free_bytes
                or len(selection["model"]["files"]) != len(configured.files)
                or {a["path"] for a in selection["model"]["files"]}
                != {path for path, _, _ in configured.files}):
            raise BootstrapError("M4 disk or weight manifest incompatible", reason="setup_rejected")
    files = []
    roles = {"diffusion_models": "diffusion_model", "text_encoders": "text_encoder", "vae": "vae",
             "audio_encoders": "audio_encoder"}
    for asset in selection["model"]["files"]:
        stage("weights", roles.get(asset["path"].split("/")[1], "none"))
        files.append(download(root, asset, deadline, profile=profile) if profile is not None
                     else download(root, asset, deadline))
    stage("verification")
    verification = {"schema_version": 1, "operator_verified": True, "worker_id": plan["pod_id"],
        "server_revision": configured.runner_revision, "image_digest": request["image_digest"],
        **{name: runtime[name] for name in ("python_version", "torch_version", "cuda_version")},
        "gpu": {"name": selection["hardware"]["gpu"], "backend": "cuda",
                "memory_mib": runtime["gpu_memory_mib"]},
        "model_files": files, "custom_nodes_disabled": True}
    private_write(root, "worker-verification.json", encode(verification))
    stage("server_start")
    command = [python, "main.py", "--listen", "127.0.0.1", "--port", "8188",
        "--disable-all-custom-nodes", "--disable-api-nodes", "--disable-auto-launch"]
    if configured.profile_id in S2V_PROFILE_IDS:
        command.append("--bf16-unet")
        private_write(root, "s2v-component-notices.json", encode(selection["component_notices"]))
    process = subprocess.Popen(command, cwd=comfy,
        env=clean_environment(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    private_write(root, "server-pid.json", encode({"pid": process.pid}))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    stage("server_health")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BootstrapError("worker server exited", reason="server_exited")
        try:
            with opener.open("http://127.0.0.1:8188/system_stats", timeout=2) as reply:
                if reply.status == 200:
                    return {"verification": verification,
                            **({"runtime": installed} if profile is not None else {})}
        except OSError:
            pass
        time.sleep(1)
    raise BootstrapError("worker server startup deadline exceeded", reason="timeout")



def stage_reference(request: dict[str, Any], root: Path) -> dict[str, Any]:
    profile = checked_profile(request.get("profile"))
    if profile is None:
        raise BootstrapError("legacy worker does not support reference staging")
    if profile["id"] != M4_PROFILE:
        raise BootstrapError("M5 media requires the private paired staging protocol")
    expected = profile["reference"]
    try:
        payload = base64.b64decode(request["payload"], validate=True)
    except (ValueError, TypeError):
        raise BootstrapError("invalid reference transfer") from None
    if (len(payload) != expected["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != expected["sha256"]):
        raise BootstrapError("reference transfer differs from bound bytes")
    validate_reference_png(payload)
    comfy = root / "ComfyUI"
    target = comfy / "input"
    if any(not path.is_dir() or path.is_symlink() for path in (comfy, target)):
        raise BootstrapError("worker reference destination unavailable")
    private_write(target, expected["basename"], payload)
    return {"reference": expected}

def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    timeout = request["timeout_seconds"]
    if (request.get("allow_live") is not True or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0 < timeout <= 1740):
        raise BootstrapError("worker operation requires bounded opt-in")
    deadline = time.monotonic() + timeout
    operation = request["operation"]
    if operation == "ping":
        return {"status": "ssh_ready"}
    if operation == "arm":
        return arm(request, deadline)
    root = worker_root(request["plan"]["session_id"])
    if encode(_json((root / "plan.json").read_bytes())) != encode(request["plan"]):
        raise BootstrapError("worker lease plan differs")
    profile = checked_profile(request.get("profile"))
    recorded = root / "profile.json"
    if ((profile is None and recorded.exists()) or (profile is not None
            and (not recorded.exists() or encode(_json(recorded.read_bytes())) != encode(profile)))):
        raise BootstrapError("worker profile differs from armed lease")
    pid = _json((root / "guard-pid.json").read_bytes())["pid"]
    if hashlib.sha256((root / "guard.py").read_bytes()).hexdigest() != request["guard_sha256"]:
        raise BootstrapError("worker guardian source changed")
    guard_alive(root, pid)
    if operation.startswith("native_"):
        if profile is None or profile["id"] not in S2V_PROFILE_IDS:
            raise BootstrapError("native speech requires the closed M5 service profile")
        from cineforge.native_remote_bootstrap import nr_dispatch
        return nr_dispatch(request, deadline)
    if operation == "heartbeat":
        heartbeat(root, request["plan"])
        return {"status": "heartbeat"}
    if operation == "setup":
        return setup(request, deadline)
    if operation == "stage_reference":
        return stage_reference(request, root)
    if operation in ("s2v_pair_begin", "s2v_pair_chunk", "s2v_pair_finish"):
        if profile is None or profile["id"] not in S2V_PROFILE_IDS:
            raise BootstrapError("paired media requires the closed M5 service profile")
        from cineforge.s2v_stage import paired_transfer
        return paired_transfer(request, root)
    raise BootstrapError("unknown worker operation")


def main() -> int:
    os.umask(0o077)
    failed = False
    answer = None
    diagnostic = None
    try:
        request = _json(sys.stdin.buffer.read(256 * 1024 + 1))
        if len(encode(request)) > 256 * 1024:
            raise BootstrapError("worker request too large")
        timeout = request.get("timeout_seconds")
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or not 0 < timeout <= 1740):
            raise BootstrapError("invalid worker deadline")

        def expire(_signum: int, _frame: Any) -> None:
            raise BootstrapError("worker absolute operation deadline exceeded", reason="timeout")
        signal.signal(signal.SIGALRM, expire)
        if str(request.get("operation", "")).startswith("native_"):
            signal.signal(signal.SIGHUP, expire)
            signal.signal(signal.SIGTERM, expire)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        answer = dispatch(request)
    except (Exception, KeyboardInterrupt) as error:
        failed = True
        if type(error) is SetupFailure:
            diagnostic = checked_setup_failure(error.diagnostic)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if failed or answer is None:
        if diagnostic is None:
            print('{"status":"worker_operation_failed"}')
        else:
            print(encode({"status": "worker_operation_failed", "setup_failure": diagnostic}).decode())
        return 2
    print(encode(answer).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
