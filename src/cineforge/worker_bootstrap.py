"""Copyable stdlib worker setup for the reviewed disposable Wan smoke only.

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
import urllib.request
import zlib
from pathlib import Path
from typing import Any

REVISION = "6d352802d8f8b9133f664a1757d6c6d26390c335"  # pragma: allowlist secret - public source revision
GPU = "NVIDIA GeForce RTX 4090"
MODEL_REVISION = "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d"  # pragma: allowlist secret - public model revision

M4_PROFILE = "m4-wan22-ti2v5b-5090-v1"
M4_GPU = "NVIDIA GeForce RTX 5090"
M4_RESOURCE_FIELDS = {"host_memory_bytes", "host_available_memory_bytes", "cgroup_limits",
                      "memory_accounting", "memory_limit_bytes", "available_memory_bytes",
                      "filesystem_total_bytes", "filesystem_free_bytes"}
M4_REVISION = 'ee6f4a40737a995bf5818954cfce6d59443b0f04'  # pragma: allowlist secret - public model revision
M4_FILES = {
    'split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors': (9999658848,
        '456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e'),  # pragma: allowlist secret - public weight digest
    'split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors': (6735906897,
        'c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68'),  # pragma: allowlist secret - public weight digest
    'split_files/vae/wan2.2_vae.safetensors': (1409400960,
        'e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156'),  # pragma: allowlist secret - public weight digest
}


class BootstrapError(ValueError):
    """Static worker diagnostic, without subprocess, URL or environment content."""



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
                    check: Any = None, environment: dict[str, str] | None = None) -> bytes:
    """Bound wall time and buffered bytes; kill this process group on every failure."""
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
            or type(max_output) is not int or not 1 <= max_output <= 1024*1024
            or payload is not None and (not isinstance(payload, bytes) or len(payload) > 256*1024)):
        raise BootstrapError("invalid worker process bounds")
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    # A failed clock read cannot strand a newly spawned child; startup counts too.
    deadline = time.monotonic() + timeout
    try:
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
                raise BootstrapError("worker process bounds exceeded")
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
            raise BootstrapError("worker process did not complete")
        if check is not None:
            check()
        return bytes(output)
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


def validate_m4_runtime(runtime: dict[str, Any]) -> None:
    fields = {"python_version", "torch_version", "torchvision_version", "cuda_version",
              "gpu_raw_name", "gpu_memory_mib", "cuda_available", "arch_list", "capability",
              "cuda_kernel_ok", "device_count", "nvml_total_mib",
              "nvml_reserved_mib", "driver_version"} | M4_RESOURCE_FIELDS
    if (set(runtime) != fields or runtime["gpu_raw_name"] != M4_GPU
            or runtime["cuda_available"] is not True or runtime["cuda_kernel_ok"] is not True
            or type(runtime["gpu_memory_mib"]) is not int
            or not 30720 <= runtime["gpu_memory_mib"] <= 32768
            or type(runtime["device_count"]) is not int or runtime["device_count"] != 1
            or runtime["capability"] != [12, 0]
            or any(type(v) is not int for v in runtime["capability"])
            or runtime["torch_version"] != "2.8.0+cu128" or runtime["cuda_version"] != "12.8"
            or not isinstance(runtime["arch_list"], list) or "sm_120" not in runtime["arch_list"]
            or len(runtime["arch_list"]) > 64
            or any(not isinstance(v, str) or re.fullmatch(r"(?:sm|compute)_[0-9a-z]{2,8}", v)
                   is None for v in runtime["arch_list"])
            or type(runtime["nvml_total_mib"]) is not int
            or not runtime["gpu_memory_mib"] <= runtime["nvml_total_mib"] <= 32768
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
            or runtime["memory_limit_bytes"] < 64_000_000_000
            or runtime["available_memory_bytes"] < 32*1024**3
            or type(runtime["filesystem_free_bytes"]) is not int
            or type(runtime["filesystem_total_bytes"]) is not int
            or not 30*1024**3 <= runtime["filesystem_free_bytes"] <= runtime["filesystem_total_bytes"]):
        raise BootstrapError("M4 container memory or filesystem headroom insufficient")


def checked_profile(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
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

def runtime_probe(timeout: float, python: str = sys.executable, *,
                  profile: dict[str, Any] | None = None, resource_root: Path | None = None) \
        -> dict[str, Any]:
    checked_profile(profile)
    deadline = time.monotonic() + timeout
    code = ("import json,platform,torch,importlib.metadata as m;"
            "p=torch.cuda.get_device_properties(0);"
            "print(json.dumps(dict(python_version=platform.python_version(),"
            "torch_version=torch.__version__,torchvision_version=m.version('torchvision'),"
            "cuda_version=torch.version.cuda,gpu_raw_name=p.name,"
            "gpu_memory_mib=p.total_memory//1048576,cuda_available=torch.cuda.is_available())))")
    if profile is not None:
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
    runtime = _json(bounded_process([python, "-c", code], timeout=timeout))
    if (runtime["cuda_available"] is not True or runtime["gpu_raw_name"] != (M4_GPU if profile is not None else GPU)
            or type(runtime["gpu_memory_mib"]) is not int
            or runtime["gpu_memory_mib"] < (30720 if profile is not None else 24082)):
        raise BootstrapError("worker runtime incompatible")
    for name in ("python_version", "torch_version", "torchvision_version", "cuda_version"):
        if re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", runtime[name]) is None:
            raise BootstrapError("invalid worker runtime version")
    if (re.fullmatch(r"\d{1,2}\.\d{1,2}", runtime["cuda_version"]) is None
            or tuple(map(int, runtime["cuda_version"].split("."))) < (12, 8)):
        raise BootstrapError("worker CUDA runtime incompatible")
    if profile is not None and (runtime.get("torch_version") != "2.8.0+cu128"
            or runtime.get("cuda_version") != "12.8" or runtime.get("capability") != [12, 0]
            or runtime.get("device_count") != 1 or type(runtime.get("device_count")) is not int
            or not isinstance(runtime.get("arch_list"), list) or "sm_120" not in runtime["arch_list"]
            or runtime.get("cuda_kernel_ok") is not True):
        raise BootstrapError("M4 worker stack or kernel incompatible")
    if profile is not None:
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
        if not runtime["gpu_memory_mib"] <= total <= 32768 or not 0 <= reserved < total:
            raise BootstrapError("M4 NVML and CUDA memory evidence incompatible")
        runtime.update(nvml_total_mib=total, nvml_reserved_mib=reserved,
                       driver_version=values[2].strip())
        runtime.update(resource_evidence(resource_root or Path("/workspace")))
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
    runtime = (runtime_probe(max(.01, deadline - time.monotonic()), profile=profile, resource_root=root)
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
            raise BootstrapError("unapproved model redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(root: Path, asset: dict[str, Any], deadline: float, *,
             profile: dict[str, Any] | None = None) -> dict[str, Any]:
    checked_profile(profile)
    revision = M4_REVISION if profile is not None else MODEL_REVISION
    distribution = "Wan-2.2" if profile is not None else "Wan-2.1"
    path = asset["path"]
    if (not isinstance(path, str) or re.fullmatch(r"split_files/(diffusion_models|text_encoders|vae)"
            r"/[A-Za-z0-9._-]+\.safetensors", path) is None
            or asset["revision"] != revision
            or asset["url"] != "https://huggingface.co/Comfy-Org/" + distribution + "/resolve/"
            + revision + "/" + path or type(asset["size_bytes"]) is not int
            or not 0 < asset["size_bytes"] <= (10_000_000_000 if profile is not None else 8 * 1024 ** 3)
            or profile is not None and M4_FILES.get(path) != (asset["size_bytes"], asset["sha256"])
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None):
        raise BootstrapError("unapproved worker model asset")
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
                    raise BootstrapError("model download deadline exceeded")
                block = reply.read1(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > asset["size_bytes"]:
                    raise BootstrapError("model download size exceeded")
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if size != asset["size_bytes"] or digest.hexdigest() != asset["sha256"]:
            raise BootstrapError("model download differs from approved bytes")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return {"path": path, "sha256": digest.hexdigest(), "size_bytes": size}


def setup(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    plan, selection = request["plan"], request["selection"]
    profile = checked_profile(request.get("profile"))
    revision = M4_REVISION if profile is not None else MODEL_REVISION
    root = worker_root(plan["session_id"])
    if (selection["runner"]["revision"] != REVISION
            or selection["runner"]["custom_nodes"] != []
            or selection["model"]["revision"] != revision
            or request["image_digest"] != request["image_digest"].strip()
            or re.fullmatch(r"sha256:[0-9a-f]{64}", request["image_digest"]) is None):
        raise BootstrapError("worker setup differs from reviewed source")
    limit = min(30, max(.01, deadline-time.monotonic()))
    runtime = (runtime_probe(limit, profile=profile, resource_root=root) if profile is not None
               else runtime_probe(limit))
    if runtime["gpu_memory_mib"] < selection["model"]["minimum_memory_mib"]:
        raise BootstrapError("worker memory below model requirement")
    comfy, venv = root / "ComfyUI", root / "venv"

    def run(command: list[str], limit: float = 180, cwd: Path | None = None) -> bytes:
        return bounded_process(command, timeout=min(limit, max(.01, deadline-time.monotonic())),
                               cwd=cwd)
    run(["git", "init", str(comfy)])
    run(["git", "-C", str(comfy), "remote", "add", "origin",
         "https://github.com/Comfy-Org/ComfyUI.git"])
    run(["git", "-C", str(comfy), "fetch", "--depth", "1", "origin", REVISION])
    run(["git", "-C", str(comfy), "checkout", "--detach", "FETCH_HEAD"])
    if run(["git", "-C", str(comfy), "rev-parse", "HEAD"]).decode().strip() != REVISION:
        raise BootstrapError("worker source revision differs")
    run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], 60)
    python = str(venv / "bin/python")
    constraints = root / "constraints.txt"
    private_write(root, constraints.name, ("torch==" + runtime["torch_version"]
        + "\ntorchvision==" + runtime["torchvision_version"] + "\n").encode())
    run([python, "-m", "pip", "--isolated", "install", "--index-url", "https://pypi.org/simple",
         "--disable-pip-version-check", "--no-input", "--progress-bar", "off", "--quiet",
         "--constraint", str(constraints), "-r", str(comfy / "requirements.txt")], 500)
    limit = min(30, max(.01, deadline-time.monotonic()))
    installed = (runtime_probe(limit, python, profile=profile, resource_root=root) if profile is not None
                 else runtime_probe(limit, python))
    if any(installed[name] != runtime[name] for name in ("torch_version", "cuda_version")):
        raise BootstrapError("worker GPU stack changed")
    freeze = run([python, "-m", "pip", "--isolated", "freeze", "--all"], 30)
    private_write(root, "environment.json", encode({"source_revision": REVISION,
        "requirements_sha256": hashlib.sha256((comfy / "requirements.txt").read_bytes()).hexdigest(),
        "freeze_sha256": hashlib.sha256(freeze).hexdigest(),
        "dependency_policy": "source-pinned requirements; GPU stack constrained; resolved versions recorded",
        "resolved_packages": freeze.decode()}))
    if profile is not None:
        if (shutil.disk_usage(root).free < 30*1024**3
                or len(selection["model"]["files"]) != 3
                or {a["path"] for a in selection["model"]["files"]} != set(M4_FILES)):
            raise BootstrapError("M4 disk or weight manifest incompatible")
    files = [download(root, asset, deadline, profile=profile) if profile is not None
             else download(root, asset, deadline) for asset in selection["model"]["files"]]
    verification = {"schema_version": 1, "operator_verified": True, "worker_id": plan["pod_id"],
        "server_revision": REVISION, "image_digest": request["image_digest"],
        **{name: runtime[name] for name in ("python_version", "torch_version", "cuda_version")},
        "gpu": {"name": selection["hardware"]["gpu"], "backend": "cuda",
                "memory_mib": runtime["gpu_memory_mib"]},
        "model_files": files, "custom_nodes_disabled": True}
    private_write(root, "worker-verification.json", encode(verification))
    process = subprocess.Popen([python, "main.py", "--listen", "127.0.0.1", "--port", "8188",
        "--disable-all-custom-nodes", "--disable-api-nodes", "--disable-auto-launch"], cwd=comfy,
        env=clean_environment(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    private_write(root, "server-pid.json", encode({"pid": process.pid}))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BootstrapError("worker server exited")
        try:
            with opener.open("http://127.0.0.1:8188/system_stats", timeout=2) as reply:
                if reply.status == 200:
                    return {"verification": verification,
                            **({"runtime": installed} if profile is not None else {})}
        except OSError:
            pass
        time.sleep(1)
    raise BootstrapError("worker server startup deadline exceeded")



def stage_reference(request: dict[str, Any], root: Path) -> dict[str, Any]:
    profile = checked_profile(request.get("profile"))
    if profile is None:
        raise BootstrapError("legacy worker does not support reference staging")
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
    if operation == "heartbeat":
        heartbeat(root, request["plan"])
        return {"status": "heartbeat"}
    if operation == "setup":
        return setup(request, deadline)
    if operation == "stage_reference":
        return stage_reference(request, root)
    raise BootstrapError("unknown worker operation")


def main() -> int:
    os.umask(0o077)
    failed = False
    answer = None
    try:
        request = _json(sys.stdin.buffer.read(256 * 1024 + 1))
        if len(encode(request)) > 256 * 1024:
            raise BootstrapError("worker request too large")
        timeout = request.get("timeout_seconds")
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or not 0 < timeout <= 1740):
            raise BootstrapError("invalid worker deadline")

        def expire(_signum: int, _frame: Any) -> None:
            raise BootstrapError("worker absolute operation deadline exceeded")
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        answer = dispatch(request)
    except (Exception, KeyboardInterrupt):
        failed = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if failed or answer is None:
        print('{"status":"worker_operation_failed"}')
        return 2
    print(encode(answer).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
