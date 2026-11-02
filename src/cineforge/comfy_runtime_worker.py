"""Fixed M5 wheel/source extraction and protected manifest verification; no resolver."""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import tarfile
import time
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from cineforge.comfy_runtime_plan import (
    COMFY_SOURCE,
    _files,
    comfy_plan,
    comfy_subject,
    notice_bytes,
)
from cineforge.native_remote_bootstrap import nr_check, nr_private, nr_read, nr_write
from cineforge.native_remote_plan import nr_digest, nr_encode, nr_json
from cineforge.native_remote_worker import _destination, _download, _install_wheel, _parent, _parts
from cineforge.native_speech_integrity import NativeBinary, NativeTree, verify_binary, verify_tree


def install_wheel(path: Path, target: Path, wheel: dict[str, Any],
                  manifest: dict[str, tuple[int, str]], deadline: float, *,
                  shared_files: tuple[dict[str, Any], ...] = ()) -> None:
    # Validate reviewed bytes, headers, tags, notices and all destination conflicts
    # before the shared RECORD verifier/exclusive writer sees any extraction target.
    verify_binary(NativeBinary(str(path), wheel["size_bytes"], wheel["sha256"]), deadline)
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 100000:
            raise ValueError("closed Comfy wheel entry bound exceeded")
        for info in archive.infolist():
            _parts(info.filename.rstrip("/") if info.is_dir() else info.filename)
            if info.is_dir() and stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFDIR):
                raise ValueError("closed Comfy wheel directory is not regular")
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        destinations = [_destination(name, (wheel["name"], wheel["version"]), comfy_layout=True)
                        for name in names]
        allowed_shared = {"site/" + entry["path"]: entry for entry in shared_files}
        if len(set(destinations)) != len(destinations):
            raise ValueError("closed Comfy wheel destination overlaps")
        for original, relative in zip(names, destinations):
            shared = allowed_shared.get(relative)
            if relative in manifest and shared is None:
                raise ValueError("closed Comfy wheel destination overlaps")
            if shared is not None:
                if wheel["name"] not in shared["owners"] or archive.getinfo(original).file_size != shared["size_bytes"]:
                    raise ValueError("closed Comfy shared file owner differs")
                if relative in manifest:
                    from cineforge.native_speech_integrity import _directory, _read_file
                    if manifest[relative] != (shared["size_bytes"], shared["sha256"]):
                        raise ValueError("closed Comfy shared file identity differs")
                    descriptor = _directory(str((target / relative).parent), private=True)
                    try:
                        _read_file(descriptor, Path(relative).name,
                            (Path(relative).name, shared["size_bytes"], "sha256", shared["sha256"]), deadline)
                    finally:
                        os.close(descriptor)
        for suffix, sha_name in (("METADATA", "metadata_sha256"), ("WHEEL", "wheel_sha256")):
            selected = [name for name in names if name.endswith(".dist-info/" + suffix)
                        and name.count("/") == 1]
            if len(selected) != 1 or archive.getinfo(selected[0]).file_size > 1024**2:
                raise ValueError("closed Comfy wheel headers differ")
            raw = archive.read(selected[0])
            if hashlib.sha256(raw).hexdigest() != wheel[sha_name]:
                raise ValueError("closed Comfy wheel metadata differs")
            headers = BytesParser().parsebytes(raw)
            if suffix == "METADATA":
                if (len(headers.get_all("Name", [])) != 1 or len(headers.get_all("Version", [])) != 1
                        or headers.get("Requires-Python", "") != wheel["requires_python"]
                        or sorted(headers.get_all("Requires-Dist", [])) != wheel["requires_dist"]):
                    raise ValueError("closed Comfy dependency metadata differs")
            elif (headers.get("Wheel-Version") != "1.0"
                    or sorted(headers.get_all("Tag", [])) != wheel["tags"]):
                raise ValueError("closed Comfy wheel tags differ")
        for name, size, sha in wheel["notices"]:
            info = archive.getinfo(name)
            if info.file_size != size or size > 4 * 1024**2:
                raise ValueError("closed Comfy notice identity differs")
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("closed Comfy notice bytes differ")
    _install_wheel(path, target, wheel, manifest, deadline, comfy_layout=True)
    nr_check(deadline)


def install_source(path: Path, target: Path, declaration: dict[str, Any], deadline: float) -> None:
    verify_binary(NativeBinary(str(path), declaration["size_bytes"], declaration["sha256"]), deadline)
    expected = {p: (size, sha) for p, size, sha in declaration["files"]}
    seen: set[str] = set()
    prefix = "ComfyUI-" + COMFY_SOURCE + "/"
    with tarfile.open(path, "r:gz") as archive:
        count = 0
        for member in archive:
            nr_check(deadline)
            count += 1
            if member.isdir() and member.name.rstrip("/") == prefix.rstrip("/"):
                continue
            if count > 100000 or not member.name.startswith(prefix):
                raise ValueError("closed Comfy source archive layout differs")
            if member.isdir():
                continue
            relative = member.name[len(prefix):]
            if (not member.isfile() or relative not in expected or relative in seen
                    or member.size != expected[relative][0]):
                raise ValueError("closed Comfy source closure differs")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("closed Comfy source body missing")
            # Source files are bounded independently of model/runtime binaries.
            if member.size > 16 * 1024**2:
                raise ValueError("closed Comfy source body exceeds bound")
            raw = source.read(member.size + 1)
            if len(raw) != member.size or hashlib.sha256(raw).hexdigest() != expected[relative][1]:
                raise ValueError("closed Comfy source hash differs")
            nr_write(_parent(target, relative), raw)
            seen.add(relative)
    if seen != set(expected):
        raise ValueError("closed Comfy source file missing")
    nr_check(deadline)


def verify_runtime(plan: dict[str, Any], root: Path, deadline: float) -> None:
    comfy_plan(plan)
    verify_binary(NativeBinary(**plan["interpreter"]), deadline)
    for binary in plan["native"]:
        verify_binary(NativeBinary(**binary), deadline)
    runtime = root / "comfy-runtime"
    verify_tree(NativeTree(str(runtime), _files(plan["installed"], str(runtime))), deadline)
    executable_paths = {entry["path"] for entry in plan["executables"]}
    for relative, _, _ in plan["installed"]:
        mode = stat.S_IMODE(nr_private(runtime / relative).st_mode)
        if mode != (0o700 if relative in executable_paths else 0o600):
            raise ValueError("closed Comfy installed mode differs")
    # Do not infer closure from installed-after-resolution freeze. Require the full
    # separately reviewed manifest and exactly one selected METADATA per package.
    actual = {}
    for relative, _, _ in plan["installed"]:
        if (relative.startswith("site/") and relative.endswith(".dist-info/METADATA")
                and relative.count("/") == 2):
            raw = nr_read(runtime / relative, maximum=1024**2)
            headers = BytesParser().parsebytes(raw)
            name = re.sub(r"[-_.]+", "-", headers.get("Name", "")).lower()
            if name in actual:
                raise ValueError("duplicate closed Comfy distribution")
            actual[name] = (headers.get("Version"), hashlib.sha256(raw).hexdigest())
    if actual != {w["name"]: (w["version"], w["metadata_sha256"]) for w in plan["wheels"]}:
        raise ValueError("closed Comfy installed distribution closure differs")
    notices = {p["sha256"] + ".txt": (p["size_bytes"], p["sha256"]) for p in plan["notices"]}
    target = root / "comfy-notices"
    nr_private(target, directory=True)
    if notices:
        if any(nr_private(target / name).st_mode & 0o111 for name in notices):
            raise ValueError("Comfy notice addenda must remain nonexecutable")
        pins = tuple((p, size, "sha256", sha) for p, (size, sha) in sorted(notices.items()))
        verify_tree(NativeTree(str(target), pins), deadline)
    elif list(target.iterdir()):
        raise ValueError("unknown Comfy notice addendum")
    if nr_read(root / "comfy-notices-manifest.json") != nr_encode(plan["notices"]):
        raise ValueError("Comfy notice manifest changed")
    nr_check(deadline)


def set_executables(plan: dict[str, Any], root: Path, deadline: float) -> None:
    """Only fixed reviewed ELF helpers receive owner execute; no script/command API."""
    from cineforge.native_speech_integrity import _directory, _read_file
    for executable in comfy_plan(plan)["executables"]:
        path = root / "comfy-runtime" / executable["path"]
        descriptor = _directory(str(path.parent), private=True)
        try:
            _read_file(descriptor, path.name,
                (path.name, executable["size_bytes"], "sha256", executable["sha256"]), deadline)
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            try:
                header = os.read(fd, 64)
                if (len(header) != 64 or header[:6] != b"\x7fELF\x02\x01"
                        or int.from_bytes(header[16:18], "little") not in (2, 3)
                        or int.from_bytes(header[18:20], "little") != 62):
                    raise ValueError("closed Comfy executable is not selected x86_64 ELF")
                os.fchmod(fd, 0o700)
            finally:
                os.close(fd)
        finally:
            os.close(descriptor)
        nr_check(deadline)


def verify_source(plan: dict[str, Any], root: Path, deadline: float, *,
                  required_models: tuple[str, ...] = ()) -> None:
    # Code is exact. Only hash-named paired input and the four fixed model bodies
    # may be added inside the source tree; generated output/user/temp live elsewhere.
    from cineforge.worker_profiles import S2V_WORKER
    source = root / "ComfyUI"
    pins = list(_files(plan["source"]["files"], str(source)))
    expected = {p for p, _, _, _ in pins}
    input_root = source / "input"
    if input_root.exists():
        nr_private(input_root, directory=True)
        for path in input_root.iterdir():
            relative = "input/" + path.name
            if relative in expected:
                continue
            match = re.fullmatch(r"(?:actor|speech)-([0-9a-f]{64})\.(png|wav)", path.name)
            if match is None:
                raise ValueError("unknown closed Comfy input file")
            info = nr_private(path)
            pins.append((relative, info.st_size, "sha256", match[1]))
    for relative, size, sha in S2V_WORKER.files:
        required = relative in required_models
        relative = relative.replace("split_files/", "models/", 1)
        if (required or (source / relative).exists()) and relative not in expected:
            pins.append((relative, size, "sha256", sha))
    verify_tree(NativeTree(str(source), tuple(pins)), deadline,
                empty_directories=frozenset({"input", "output", "temp", "user", "models"}))
    nr_check(deadline)


def install_comfy(request: dict[str, Any], native: Path, deadline: float) -> dict[str, Any]:
    # Imported only from nr_package's just-verified fixed first-party closure.
    from cineforge import worker_bootstrap as bootstrap
    from cineforge.comfy_runtime_launcher import launch_command, probe_result
    nr_write(native / "comfy-install-consumed", b"consumed before any Comfy effect")
    nr_read(native / "comfy.blob.done")
    plan = comfy_plan(nr_json(nr_read(native / "comfy.blob")))
    if (comfy_subject(plan) != request["comfy_subject_sha256"]
            or request["image_digest"] != plan["image"].split("@", 1)[1]
            or plan["interpreter"] != nr_json(nr_read(native / "preflight.json"))["interpreter"]):
        raise ValueError("closed Comfy source or image scope differs")
    root = bootstrap.worker_root(request["plan"]["session_id"])
    # Empty destinations only. A prior failed install is never overwritten/reused.
    for name in ("ComfyUI", "comfy-runtime", "comfy-downloads", "comfy-output", "comfy-user", "comfy-temp", "comfy-notices"):
        (root / name).mkdir(mode=0o700)
    for notice in plan["notices"]:
        path = root / "comfy-notices" / (notice["sha256"] + ".txt")
        if not path.exists():
            nr_write(path, notice_bytes(notice))
    nr_write(root / "comfy-notices-manifest.json", nr_encode(plan["notices"]))
    for binary in [plan["interpreter"], *plan["native"]]:
        verify_binary(NativeBinary(**binary), deadline)
    downloads = root / "comfy-downloads"
    manifest: dict[str, tuple[int, str]] = {}
    for wheel in plan["wheels"]:
        path = downloads / wheel["filename"]
        _download(path, wheel["url"], wheel["size_bytes"], wheel["sha256"], deadline)
        install_wheel(path, root / "comfy-runtime", wheel, manifest, deadline,
                      shared_files=tuple(plan["shared_files"]))
    if [[name, size, sha] for name, (size, sha) in sorted(manifest.items())] != plan["installed"]:
        raise ValueError("closed Comfy installed files differ from prior review")
    set_executables(plan, root, deadline)
    source = plan["source"]
    archive = downloads / "comfy-source.tar.gz"
    _download(archive, "https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/" + COMFY_SOURCE,
              source["size_bytes"], source["sha256"], deadline)
    install_source(archive, root / "ComfyUI", source, deadline)
    nr_write(root / "comfy-selection.json", nr_encode(plan))
    verify_runtime(plan, root, deadline)
    verify_source(plan, root, deadline)
    command, environment = launch_command(plan, root, native, "probe", deadline)
    raw = bootstrap.bounded_process(command, timeout=min(30, deadline-time.monotonic()),
        environment=environment, cwd=root, absolute_deadline=deadline, max_output=65536)
    runtime, facts = probe_result(nr_json(raw), root, plan, deadline)
    nr_write(root / "comfy-probe.json", nr_encode(facts))
    return {"runtime": runtime, "comfy": current_comfy(request, native, deadline)}


def stage_comfy_model(request: dict[str, Any], native: Path, deadline: float) -> dict[str, Any]:
    from cineforge import worker_bootstrap as bootstrap
    from cineforge.worker_profiles import S2V_WORKER
    root = bootstrap.worker_root(request["plan"]["session_id"])
    current_comfy(request, native, deadline)
    assets = request["selection"]["model"]["files"]
    if (type(assets) is not list or len(assets) != len(S2V_WORKER.files)
            or sorted((a["path"], a["size_bytes"], a["sha256"]) for a in assets)
            != sorted(S2V_WORKER.files)):
        raise ValueError("closed Comfy complete model closure required")
    index = request["model_index"]
    if type(index) is not int or not 0 <= index < len(assets):
        raise ValueError("unsupported fixed Comfy model index")
    nr_write(native / ("comfy-model-" + str(index) + "-consumed"), b"consumed before model download")
    result = bootstrap.download(root, assets[index], deadline, profile=request["profile"])
    nr_check(deadline)
    nr_write(native / ("comfy-model-" + str(index) + ".json"), nr_encode(result))
    return result


def setup_comfy(request: dict[str, Any], native: Path, deadline: float) -> dict[str, Any]:
    from cineforge import worker_bootstrap as bootstrap
    from cineforge.comfy_runtime_launcher import launch_command, probe_result
    from cineforge.worker_profiles import S2V_WORKER
    nr_write(native / "comfy-launch-consumed", b"consumed before production startup")
    root = bootstrap.worker_root(request["plan"]["session_id"])
    plan = comfy_plan(nr_json(nr_read(root / "comfy-selection.json")))
    current_comfy(request, native, deadline)
    files = [nr_json(nr_read(native / ("comfy-model-" + str(index) + ".json")))
             for index in range(len(S2V_WORKER.files))]
    if sorted((a["path"], a["size_bytes"], a["sha256"]) for a in files) != sorted(S2V_WORKER.files):
        raise ValueError("closed Comfy model transfer proof incomplete")
    verify_source(plan, root, deadline, required_models=tuple(p for p, _, _ in S2V_WORKER.files))
    command, environment = launch_command(plan, root, native, "probe", deadline)
    raw = bootstrap.bounded_process(command, timeout=min(30, deadline-time.monotonic()),
        environment=environment, cwd=root, absolute_deadline=deadline, max_output=65536)
    runtime, facts = probe_result(nr_json(raw), root, plan, deadline)
    nr_write(root / "comfy-launch-probe.json", nr_encode(facts))
    selection = request["selection"]
    verification = {"schema_version": 1, "operator_verified": True,
        "worker_id": request["plan"]["pod_id"], "server_revision": S2V_WORKER.runner_revision,
        "image_digest": request["image_digest"],
        **{name: runtime[name] for name in ("python_version", "torch_version", "cuda_version")},
        "gpu": {"name": selection["hardware"]["gpu"], "backend": "cuda",
                "memory_mib": runtime["gpu_memory_mib"]},
        "model_files": files, "custom_nodes_disabled": True}
    command, environment = launch_command(plan, root, native, "server", deadline)
    if (root / "server-pid.json").exists() or (root / "comfy-jit-startup.json").exists():
        raise ValueError("closed Comfy production state was preexisting")
    import subprocess
    spawn = subprocess.Popen
    options: dict[str, Any] = dict(cwd=root, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
    nr_check(deadline)
    process = spawn(command, **options)
    nr_write(root / "server-pid.json", nr_encode({"pid": process.pid, "start_ticks": _server_start(process.pid)}))
    nr_write(root / "worker-verification.json", nr_encode(verification))
    # Polling is fixed loopback health only; final14-node contracts are verified by
    # S2VSelection/Comfy execution admission and remain actual opt-in evidence.
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError("closed Comfy production launcher exited")
        try:
            with opener.open("http://127.0.0.1:8188/system_stats", timeout=min(2, deadline-time.monotonic())) as reply:
                if reply.status == 200 and (root / "comfy-jit-startup.json").exists():
                    nr_check(deadline)
                    return {"verification": verification, "runtime": runtime,
                            "comfy": current_comfy(request, native, deadline)}
        except OSError:
            pass
        time.sleep(.1)
    raise ValueError("closed Comfy production startup deadline exceeded")



def _server_start(pid: int) -> str:
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid owned Comfy process")
    with (Path("/proc") / str(pid) / "stat").open("rb") as stream:
        raw = stream.read(8193)
    suffix = raw.rsplit(b")", 1)[-1].split()
    if len(raw) > 8192 or len(suffix) < 20 or suffix[0] not in (b"R", b"S", b"D", b"T", b"t", b"K", b"W", b"P", b"I"):
        raise ValueError("owned Comfy process is not live")
    return suffix[19].decode("ascii")


def server_facts(plan: dict[str, Any], root: Path, native: Path, deadline: float) -> dict[str, Any]:
    from cineforge.comfy_runtime_launcher import launch_command
    from cineforge.native_speech_integrity import _directory, _read_file
    from cineforge.worker_profiles import S2V_WORKER
    record = nr_json(nr_read(root / "server-pid.json"))
    pid = record["pid"]
    if _server_start(pid) != record["start_ticks"]:
        raise ValueError("owned Comfy process identity changed")
    proc = Path("/proc") / str(pid)
    with (proc / "cmdline").open("rb") as stream:
        raw_command = stream.read(16385)
    if len(raw_command) > 16384:
        raise ValueError("owned Comfy command exceeds bound")
    command = [part.decode("utf-8") for part in raw_command.rstrip(b"\0").split(b"\0")]
    launch_cap = float(command[-1])
    if not math.isfinite(launch_cap) or launch_cap > nr_json(nr_read(native / "comfy-local-cap.json"))["deadline"]:
        raise ValueError("owned Comfy process cap changed")
    if command != launch_command(plan, root, native, "server", launch_cap, check_deadline=False)[0]:
        raise ValueError("owned Comfy command differs")
    listeners = set()
    with (proc / "net/tcp").open("rb") as stream:
        network = stream.read(1024**2 + 1)
    if len(network) > 1024**2:
        raise ValueError("owned Comfy listener table exceeds bound")
    for network_line in network.splitlines()[1:]:
        columns = network_line.split()
        if len(columns) > 9 and columns[1] == b"0100007F:1FFC" and columns[3] == b"0A":
            listeners.add(columns[9].decode("ascii"))
    sockets = set()
    for descriptor in (proc / "fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            sockets.add(target[8:-1])
    if not listeners & sockets:
        raise ValueError("loopback Comfy listener is not owned by the fixed process")
    allowed = {p["path"]: (p["size_bytes"], p["sha256"]) for p in [plan["interpreter"], *plan["native"]]}
    allowed.update({str(root / "comfy-runtime" / p): (size, sha) for p, size, sha in plan["installed"]})
    allowed.update({str(root / "ComfyUI" / p.replace("split_files/", "models/", 1)): (size, sha)
                    for p, size, sha in S2V_WORKER.files})
    with (proc / "maps").open("rb") as stream:
        raw_maps = stream.read(1024**2 + 1)
    if len(raw_maps) > 1024**2:
        raise ValueError("owned Comfy native map exceeds bound")
    observed = {}
    for line in raw_maps.decode("utf-8").splitlines():
        path = line.split(maxsplit=5)[-1]
        if not path.startswith("/") or path.startswith("/dev/") or path in observed:
            continue
        if path not in allowed:
            raise ValueError("owned Comfy loaded an unreviewed native file")
        size, sha = allowed[path]
        parent = _directory(str(Path(path).parent), private=False)
        try:
            _read_file(parent, Path(path).name, (Path(path).name, size, "sha256", sha), deadline)
        finally:
            os.close(parent)
        observed[path] = sha
    if _server_start(pid) != record["start_ticks"]:
        raise ValueError("owned Comfy process changed during verification")
    nr_check(deadline)
    return {"pid": pid, "start_ticks": record["start_ticks"], "native_files": observed}

def current_comfy(request: dict[str, Any], native: Path, deadline: float) -> dict[str, Any]:
    from cineforge.worker_bootstrap import s2v_hardware_admission, worker_root
    from cineforge.worker_profiles import S2V_WORKER
    root = worker_root(request["plan"]["session_id"])
    plan = comfy_plan(nr_json(nr_read(root / "comfy-selection.json")))
    if request["comfy_subject_sha256"] != comfy_subject(plan):
        raise ValueError("closed Comfy current selection differs")
    verify_runtime(plan, root, deadline)
    required = tuple(p for i, (p, _, _) in enumerate(S2V_WORKER.files)
                     if (native / ("comfy-model-" + str(i) + ".json")).exists())
    verify_source(plan, root, deadline, required_models=required)
    facts = nr_json(nr_read(root / "comfy-probe.json"))
    if (facts["execution_sha256"] != nr_digest(plan["execution"])
            or facts["jit_enabled_before"] is not False or facts["jit_enabled_after"] is not False):
        raise ValueError("closed Comfy JIT import policy changed")
    hardware = s2v_hardware_admission(root, deadline)
    for name in ("gpu_raw_name", "capability", "device_count", "nvml_total_mib",
                 "nvml_reserved_mib", "driver_version"):
        if hardware[name] != facts["runtime"][name]:
            raise ValueError("closed Comfy actual GPU or driver changed")
    observed = server_facts(plan, root, native, deadline) if (root / "server-pid.json").exists() else None
    phase = "after_pytorch_import"
    if observed is not None:
        startup = nr_json(nr_read(root / "comfy-jit-startup.json"))
        if (set(startup) != {"execution_sha256", "jit_enabled", "phase"}
                or startup["execution_sha256"] != nr_digest(plan["execution"])
                or startup["jit_enabled"] is not False or startup["phase"] != "after_core_startup"):
            raise ValueError("closed Comfy post-core JIT observation unavailable")
        phase = "after_core_startup"
        name = "comfy-native-" + nr_digest(observed) + ".json"
        if not (root / name).exists():
            if len(list(root.glob("comfy-native-*.json"))) >= 32:
                raise ValueError("owned Comfy native evidence capacity exhausted")
            nr_write(root / name, nr_encode(observed))
    nr_check(deadline)
    return {"subject_sha256": comfy_subject(plan),
            "installed_sha256": nr_digest(plan["installed"]), "probe_sha256": nr_digest({"probe": facts, "server": observed}),
            "execution_sha256": nr_digest(plan["execution"]), "jit_enabled": False, "phase": phase}
