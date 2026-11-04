"""Fixed stdlib transfer bootstrap. Included verbatim in the guarded SSH bundle."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import importlib
import math
import os
import platform
import re
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cineforge.native_remote_plan import (
    NR_BLOB_MAX,
    NR_CHUNK,
    nr_digest,
    nr_encode,
    nr_hash,
    nr_json,
    nr_token,
)
from cineforge.worker_bootstrap import bounded_process
from cineforge.worker_profiles import S2V_WORKER, s2v_worker

# Replaced by worker_bundle with identities of the complete installed first party.
NR_PACKAGE_SHA256 = ""
NR_PACKAGE_FILES: tuple[tuple[str, int, str], ...] = ()
NR_CAP_POLICY = "m5-fixed-native-facts-v1"
NR_CAP_ALIASES = tuple(sorted((role, path) for role, paths in (
    ("interpreter", ("/usr/bin/python3.11",)),
    ("seccomp", ("/lib/x86_64-linux-gnu/libseccomp.so.2", "/usr/lib/x86_64-linux-gnu/libseccomp.so.2")),
    ("nvidia_smi", ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi")),
    ("cuda_driver", ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1",
                     "/usr/local/nvidia/lib64/libcuda.so.1")),
    ("nvml", ("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "/usr/lib64/libnvidia-ml.so.1",
              "/usr/local/nvidia/lib64/libnvidia-ml.so.1")),
) for path in paths))
NR_CAP_ROOTS = ("/usr/bin/", "/usr/local/nvidia/bin/", "/usr/lib/x86_64-linux-gnu/",
                "/usr/lib64/", "/usr/local/nvidia/lib64/")


def nr_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def nr_cap_file(alias: str, deadline: float) -> dict[str, Any]:
    # The only caller enumerates fixed aliases. Resolve protected symlinks with
    # bounded descriptor traversal, then hash the actual no-follow owned body.
    path = alias
    links: list[tuple[str, tuple[int, ...], str]] = []
    for _ in range(17):
        nr_check(deadline)
        if len(path) > 4096 or not path.startswith("/") or "\x00" in path:
            raise ValueError("invalid fixed native capability path")
        parts = path.split("/")[1:]
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        restart = False
        try:
            for index, name in enumerate(parts):
                nr_check(deadline)
                if name in ("", ".", ".."):
                    raise ValueError("invalid fixed native capability component")
                before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if before.st_uid not in (0, os.getuid()):
                    raise ValueError("unowned fixed native capability")
                if stat.S_ISLNK(before.st_mode):
                    target = os.readlink(name, dir_fd=descriptor)
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if nr_file_identity(before) != nr_file_identity(after):
                        raise ValueError("fixed native capability link changed")
                    links.append((os.path.join("/", *parts[:index+1]), nr_file_identity(before), target))
                    path = os.path.normpath(os.path.join("/", *parts[:index], target, *parts[index+1:]))
                    restart = True
                    break
                if before.st_mode & 0o022 and not (before.st_uid == 0 and before.st_mode & stat.S_ISVTX):
                    raise ValueError("unprotected fixed native capability ancestor")
                if index < len(parts)-1:
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    if (os.fstat(child).st_dev, os.fstat(child).st_ino) != (before.st_dev, before.st_ino):
                        os.close(child)
                        raise ValueError("fixed native capability ancestor changed")
                    os.close(descriptor)
                    descriptor = child
                    continue
                if (not path.startswith(NR_CAP_ROOTS) or not stat.S_ISREG(before.st_mode)
                        or before.st_nlink != 1 or before.st_mode & 0o022
                        or not 0 < before.st_size <= 1024**3):
                    raise ValueError("unprotected fixed native capability body")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
                try:
                    if nr_file_identity(before) != nr_file_identity(os.fstat(fd)):
                        raise ValueError("fixed native capability changed before read")
                    sha, count = hashlib.sha256(), 0
                    while block := os.read(fd, 1024**2):
                        nr_check(deadline)
                        count += len(block)
                        if count > before.st_size:
                            raise ValueError("fixed native capability grew")
                        sha.update(block)
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if (count != before.st_size or nr_file_identity(before) != nr_file_identity(after)
                            or nr_file_identity(before) != nr_file_identity(os.fstat(fd))):
                        raise ValueError("fixed native capability changed during read")
                finally:
                    os.close(fd)
                nr_check(deadline)
                for link, identity, target in links:
                    if nr_file_identity(os.lstat(link)) != identity or os.readlink(link) != target:
                        raise ValueError("fixed native capability alias changed during read")
                    nr_check(deadline)
                return {"path": path, "size_bytes": count, "sha256": sha.hexdigest()}
        finally:
            os.close(descriptor)
        if not restart:
            break
    raise ValueError("fixed native capability link bound exceeded")


def nr_capabilities(deadline: float, binding: dict[str, Any]) -> dict[str, Any]:
    files = []
    total = 0
    for role, alias in NR_CAP_ALIASES:
        row: dict[str, Any] = {"role": role, "alias": alias}
        try:
            pin = nr_cap_file(alias, deadline)
        except FileNotFoundError:
            row["status"] = "missing"
        else:
            total += pin["size_bytes"]
            if total > 2 * 1024**3:
                raise ValueError("fixed native measurements exceed bound")
            row.update(pin, status="measured")
        files.append(row)
    nr_check(deadline)
    interpreter = str(Path(sys.executable).resolve(strict=True))
    if not any(row["role"] == "interpreter" and row.get("path") == interpreter for row in files):
        raise ValueError("running interpreter differs from fixed measurement candidate")
    nr_check(deadline)
    return {"version": 1, "policy": NR_CAP_POLICY, "binding_sha256": nr_digest(binding),
            "interpreter_path": interpreter, "files": files}


def nr_private(path: Path, *, directory: bool = False) -> os.stat_result:
    info = path.lstat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        and not directory
        or (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))
    ):
        raise ValueError("unsafe remote speech storage")
    return info


def nr_root(session: str, *, create: bool = False) -> Path:
    if type(session) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", session):
        raise ValueError("invalid remote speech session")
    parent = Path("/root")
    info = parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError("unsafe remote speech ancestor")
    root = parent / ("cineforge-native-m5-" + session)
    if create:
        root.mkdir(mode=0o700)
    nr_private(root, directory=True)
    return root


def nr_read(path: Path, *, maximum: int = NR_BLOB_MAX) -> bytes:
    before = nr_private(path)
    if before.st_size > maximum:
        raise ValueError("remote speech data exceeds bound")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        actual = os.fstat(fd)
        raw = bytearray()
        while True:
            block = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
            if len(raw) > maximum:
                raise ValueError("remote speech data exceeds bound")
        after = os.fstat(fd)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
                value.st_nlink,
                value.st_mode,
            )

        if (
            identity(before) != identity(actual)
            or identity(actual) != identity(after)
            or identity(after) != identity(path.lstat())
        ):
            raise ValueError("remote speech storage changed")
        return bytes(raw)
    finally:
        os.close(fd)


def nr_write(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def nr_lock(root: Path) -> Iterator[None]:
    path = root / "operation.lock"
    before = nr_private(path)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_ino != before.st_ino:
            raise ValueError("remote speech operation lock changed")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def nr_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise ValueError("remote speech deadline exhausted")


def nr_package(root: Path, *, install: bool = False) -> Path:
    raw = nr_read(root / "package.blob")
    if hashlib.sha256(raw).hexdigest() != NR_PACKAGE_SHA256:
        raise ValueError("remote first party package differs")
    value = nr_json(raw)
    if set(value) != {name for name, _, _ in NR_PACKAGE_FILES}:
        raise ValueError("remote first party closure differs")
    target = root / "firstparty"
    if install:
        target.mkdir(mode=0o700)
        (target / "cineforge").mkdir(mode=0o700)
    nr_private(target, directory=True)
    nr_private(target / "cineforge", directory=True)
    for name, size, sha in NR_PACKAGE_FILES:
        data = base64.b64decode(value[name], validate=True)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("remote first party source differs")
        path = target / "cineforge" / name
        if install:
            nr_write(path, data)
        elif nr_read(path) != data:
            raise ValueError("remote first party source changed")
    if set(p.name for p in (target / "cineforge").iterdir()) != set(value):
        raise ValueError("remote first party closure changed")
    return target


def nr_preflight(deadline: float, *, profile_id: str = S2V_WORKER.profile_id) -> dict[str, Any]:
    configured = s2v_worker(profile_id)
    requirements = configured.reference_runtime
    assert requirements is not None
    # Measurements precede network staging; no Torch/SDK imports or apt fallback.
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 11)
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
        or platform.libc_ver()[0] != "glibc"
        or tuple(int(p) for p in platform.libc_ver()[1].split(".")[:2]) < (2, 28)
    ):
        raise ValueError("unsupported fixed native worker ABI")
    candidates = (
        "/lib/x86_64-linux-gnu/libseccomp.so.2",
        "/usr/lib/x86_64-linux-gnu/libseccomp.so.2",
    )
    library = next(
        (str(Path(p).resolve(strict=True)) for p in candidates if Path(p).exists()), None
    )
    if library is None:
        raise ValueError("required native isolation library missing")

    def binary(path: str) -> dict[str, Any]:
        info = Path(path).lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_mode & 0o022
            or info.st_uid not in (0, os.getuid())
            or not 0 < info.st_size <= 1024**3
        ):
            raise ValueError("native worker binary is unprotected")
        sha = hashlib.sha256()
        with open(path, "rb") as stream:
            while block := stream.read(1024 * 1024):
                nr_check(deadline)
                sha.update(block)
        return {"path": path, "size_bytes": info.st_size, "sha256": sha.hexdigest()}

    probe = bounded_process(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=min(5, max(0.01, deadline - time.monotonic())),
        max_output=256,
        environment={"PATH": "", "LANG": "C"},
        absolute_deadline=deadline,
    )
    gpu = probe.decode("ascii").strip().split(",")
    if (
        len(gpu) != 3
        or gpu[0].strip() != configured.gpu
        or not configured.minimum_memory_mib <= int(gpu[1].strip()) <= requirements.maximum_memory_mib
        or gpu[2].strip() != ".".join(str(x) for x in requirements.capability)
    ):
        raise ValueError("native GPU capability differs from guarded M5 recipe")
    result = {
        "interpreter": binary(str(Path(sys.executable).resolve(strict=True))),
        "seccomp": binary(library),
        "abi": "cpython311-linux-x86_64-glibc2.28",
    }
    nr_check(deadline)
    return result


def nr_dispatch(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    result, cap = _nr_dispatch(request, deadline)
    # Lock release/descriptor cleanup precede final admission against the exact
    # derived worker cap. Late facts never become successful acknowledgements.
    try:
        nr_check(cap)
    except BaseException:
        root = nr_root(request["plan"]["session_id"])
        if not (root / "failed").exists():
            nr_write(root / "failed", b"late completion; owned Pod deletion required")
        raise
    return result


def _nr_dispatch(request: dict[str, Any], deadline: float) -> tuple[dict[str, Any], float]:
    # The dispatcher already verified guardian source, ownership, profile and plan.
    now, wall = time.monotonic(), time.time()
    sent, expiry = request["native_sent_unix"], request["native_expires_unix"]
    if (
        type(sent) not in (int, float)
        or not math.isfinite(sent)
        or abs(sent - wall) > 5
        or type(expiry) not in (int, float)
        or not math.isfinite(expiry)
    ):
        raise ValueError("native phase clock or delivery bound exceeded")
    deadline = min(deadline, now + request["plan"]["stop_at_unix"] - wall - 120)
    nr_check(deadline)
    op = request["operation"]
    session = request["plan"]["session_id"]
    binding = {
        "worker": request["plan"]["pod_id"],
        "session": session,
        "scope": request["profile"]["scope_sha256"],
        "guard": request["guard_sha256"],
        "package": NR_PACKAGE_SHA256,
        "source": request["native_source_sha256"],
        "phase_expires_unix": expiry,
    }
    nr_hash(binding["source"])
    if op == "native_capabilities" and set(request) - {
            "operation", "plan", "profile", "guard_sha256", "native_source_sha256",
            "native_sent_unix", "native_expires_unix", "allow_live", "timeout_seconds"}:
        raise ValueError("fixed native measurements accept no path or command selector")
    if op == "native_capabilities" or op == "native_preflight" and "capabilities_sha256" not in request:
        root = nr_root(session, create=True)  # Existing/uncertain setup cannot be restarted.
        nr_write(root / "operation.lock", b"")
        nr_write(root / "binding.json", nr_encode(binding))
        deadline = min(deadline, now + expiry - wall - 5)
        nr_check(deadline)
        nr_write(root / "local-cap.json", nr_encode({"deadline": deadline}))
    root = nr_root(session)
    if nr_json(nr_read(root / "binding.json")) != binding or (root / "failed").exists():
        raise ValueError("remote speech session differs or is unresolved")
    deadline = min(deadline, nr_json(nr_read(root / "local-cap.json"))["deadline"])
    nr_check(deadline)
    with nr_lock(root):
        try:
            if op == "native_capabilities":
                nr_write(root / "capabilities-consumed", b"consumed before fixed file measurement")
                facts = nr_capabilities(deadline, binding)
                nr_write(root / "capabilities.json", nr_encode(facts))
                nr_check(deadline)
                return facts, deadline
            if op == "native_preflight":
                nr_write(root / "preflight-consumed", b"consumed before fixed preflight")
                if "capabilities_sha256" in request:
                    nr_hash(request["capabilities_sha256"])
                    captured = nr_read(root / "capabilities.json")
                    if (hashlib.sha256(captured).hexdigest() != request["capabilities_sha256"]
                            or nr_encode(nr_capabilities(deadline, binding)) != captured):
                        raise ValueError("fixed native capability facts changed before staging")
                result = nr_preflight(deadline, profile_id=request["profile"]["id"])
                if "capabilities_sha256" in request:
                    rows = nr_json(captured)["files"]
                    for role in ("interpreter", "seccomp"):
                        if not any(row["role"] == role and row.get("status") == "measured"
                                   and {k: row[k] for k in ("path", "size_bytes", "sha256")} == result[role]
                                   for row in rows):
                            raise ValueError("preflight differs from captured native capability")
                nr_write(root / "preflight.json", nr_encode(result))
                nr_check(deadline)
                return dict(result, binding_sha256=nr_digest(binding)), deadline
            if op.startswith("native_comfy_") or request.get("role") == "comfy":
                cutoff: Any = request.get("comfy_expires_unix")
                if (type(cutoff) not in (int, float) or not math.isfinite(cutoff)
                        or cutoff > expiry):
                    raise ValueError("invalid fixed Comfy original cap")
                cap_file = root / "comfy-local-cap.json"
                if op == "native_begin":
                    cap: dict[str, Any] = {"expires_unix": cutoff, "deadline": min(deadline, now + cutoff - wall - 5)}
                    nr_check(cap["deadline"])
                    nr_write(cap_file, nr_encode(cap))
                else:
                    cap = nr_json(nr_read(cap_file))
                    if cap["expires_unix"] != cutoff:
                        raise ValueError("fixed Comfy original cap changed")
                deadline = min(deadline, cap["deadline"])
                if op.startswith("native_comfy_"):
                    call_cutoff: Any = request.get("comfy_call_expires_unix")
                    if (type(call_cutoff) not in (int, float) or not math.isfinite(call_cutoff)
                            or call_cutoff > cutoff):
                        raise ValueError("invalid fixed Comfy caller cap")
                    deadline = min(deadline, now + call_cutoff - wall - 5)
                nr_check(deadline)
            if op in ("native_begin", "native_chunk", "native_finish"):
                role = request["role"]
                if role not in ("package", "recipe", "sox", "selection", "input", "comfy"):
                    raise ValueError("unsupported remote speech transfer role")
                token = request["token"]
                nr_token(token)
                if role == "input":
                    blob = root / (token + ".input")
                else:
                    blob = root / (role + ".blob")
                declaration = blob.with_suffix(blob.suffix + ".json")
                done = blob.with_suffix(blob.suffix + ".done")
                if op == "native_begin":
                    size, sha = request["size_bytes"], request["sha256"]
                    nr_hash(sha)
                    if type(size) is not int or not 1 <= size <= NR_BLOB_MAX:
                        raise ValueError("remote speech transfer exceeds bound")
                    # Input capacity is a fixed reviewed one-design/two-utterance recipe.
                    if role == "input" and len(list(root.glob("*.input.json"))) >= 3:
                        raise ValueError("remote speech attempt capacity exhausted")
                    nr_write(
                        declaration, nr_encode({"token": token, "size_bytes": size, "sha256": sha})
                    )
                    nr_write(blob, b"")
                else:
                    item = nr_json(nr_read(declaration))
                    if item["token"] != token or done.exists():
                        raise ValueError("remote speech transfer consumed")
                    if op == "native_chunk":
                        data = base64.b64decode(request["data"], validate=True)
                        info = nr_private(blob)
                        offset = request["offset"]
                        if (
                            type(offset) is not int
                            or info.st_size != offset
                            or not 0 < len(data) <= NR_CHUNK
                            or offset + len(data) > item["size_bytes"]
                        ):
                            raise ValueError("invalid remote speech chunk")
                        fd = os.open(blob, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                        try:
                            if os.fstat(fd).st_ino != info.st_ino or os.write(fd, data) != len(
                                data
                            ):
                                raise ValueError("remote speech chunk write uncertain")
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    else:
                        data = nr_read(blob)
                        if (
                            len(data) != item["size_bytes"]
                            or hashlib.sha256(data).hexdigest() != item["sha256"]
                        ):
                            raise ValueError("remote speech transfer identity differs")
                        nr_write(done, b"complete")
                        if role == "package":
                            nr_package(root, install=True)
                nr_check(deadline)
                return {"status": op, "token": token}, deadline
            package = nr_package(root)
            # Only a fixed module from just-verified first-party bytes is importable.
            sys.path.insert(0, str(package))
            implementation = importlib.import_module("cineforge.native_remote_worker")
            reply: dict[str, Any] = implementation.remote_operation(
                request, root, deadline, binding
            )
            nr_check(deadline)
            return reply, deadline
        except BaseException:
            if not (root / "failed").exists():
                nr_write(root / "failed", b"unresolved; owned Pod deletion required")
            raise
