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
import signal
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REVISION = "6d352802d8f8b9133f664a1757d6c6d26390c335"  # pragma: allowlist secret - public source revision
GPU = "NVIDIA GeForce RTX 4090"
MODEL_REVISION = "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d"  # pragma: allowlist secret - public model revision


class BootstrapError(ValueError):
    """Static worker diagnostic, without subprocess, URL or environment content."""


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


def runtime_probe(timeout: float, python: str = sys.executable) -> dict[str, Any]:
    code = ("import json,platform,torch,importlib.metadata as m;"
            "p=torch.cuda.get_device_properties(0);"
            "print(json.dumps(dict(python_version=platform.python_version(),"
            "torch_version=torch.__version__,torchvision_version=m.version('torchvision'),"
            "cuda_version=torch.version.cuda,gpu_raw_name=p.name,"
            "gpu_memory_mib=p.total_memory//1048576,cuda_available=torch.cuda.is_available())))")
    runtime = _json(bounded_process([python, "-c", code], timeout=timeout))
    if (runtime["cuda_available"] is not True or runtime["gpu_raw_name"] != GPU
            or type(runtime["gpu_memory_mib"]) is not int
            or runtime["gpu_memory_mib"] < 24082):
        raise BootstrapError("worker runtime incompatible")
    for name in ("python_version", "torch_version", "torchvision_version", "cuda_version"):
        if re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", runtime[name]) is None:
            raise BootstrapError("invalid worker runtime version")
    if (re.fullmatch(r"\d{1,2}\.\d{1,2}", runtime["cuda_version"]) is None
            or tuple(map(int, runtime["cuda_version"].split("."))) < (12, 8)):
        raise BootstrapError("worker CUDA runtime incompatible")
    return runtime


def arm(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    plan = request["plan"]
    source = base64.b64decode(request["guard_source"], validate=True)
    if len(source) > 65536 or hashlib.sha256(source).hexdigest() != request["guard_sha256"]:
        raise BootstrapError("worker guard source differs")
    root = worker_root(plan["session_id"], create=True)
    private_write(root, "guard.py", source)
    private_write(root, "plan.json", encode(plan))
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
    runtime = runtime_probe(max(.01, deadline - time.monotonic()))
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


def download(root: Path, asset: dict[str, Any], deadline: float) -> dict[str, Any]:
    path = asset["path"]
    if (not isinstance(path, str) or re.fullmatch(r"split_files/(diffusion_models|text_encoders|vae)"
            r"/[A-Za-z0-9._-]+\.safetensors", path) is None
            or asset["revision"] != MODEL_REVISION
            or asset["url"] != "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/"
            + MODEL_REVISION + "/" + path or type(asset["size_bytes"]) is not int
            or not 0 < asset["size_bytes"] <= 8 * 1024 ** 3
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
    root = worker_root(plan["session_id"])
    if (selection["runner"]["revision"] != REVISION
            or selection["runner"]["custom_nodes"] != []
            or selection["model"]["revision"] != MODEL_REVISION
            or request["image_digest"] != request["image_digest"].strip()
            or re.fullmatch(r"sha256:[0-9a-f]{64}", request["image_digest"]) is None):
        raise BootstrapError("worker setup differs from reviewed source")
    runtime = runtime_probe(min(30, max(.01, deadline-time.monotonic())))
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
    installed = runtime_probe(min(30, max(.01, deadline-time.monotonic())), python)
    if any(installed[name] != runtime[name] for name in ("torch_version", "cuda_version")):
        raise BootstrapError("worker GPU stack changed")
    freeze = run([python, "-m", "pip", "--isolated", "freeze", "--all"], 30)
    private_write(root, "environment.json", encode({"source_revision": REVISION,
        "requirements_sha256": hashlib.sha256((comfy / "requirements.txt").read_bytes()).hexdigest(),
        "freeze_sha256": hashlib.sha256(freeze).hexdigest(),
        "dependency_policy": "source-pinned requirements; GPU stack constrained; resolved versions recorded",
        "resolved_packages": freeze.decode()}))
    files = [download(root, asset, deadline) for asset in selection["model"]["files"]]
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
                    return {"verification": verification}
        except OSError:
            pass
        time.sleep(1)
    raise BootstrapError("worker server startup deadline exceeded")


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
    pid = _json((root / "guard-pid.json").read_bytes())["pid"]
    if hashlib.sha256((root / "guard.py").read_bytes()).hexdigest() != request["guard_sha256"]:
        raise BootstrapError("worker guardian source changed")
    guard_alive(root, pid)
    if operation == "heartbeat":
        heartbeat(root, request["plan"])
        return {"status": "heartbeat"}
    if operation == "setup":
        return setup(request, deadline)
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
