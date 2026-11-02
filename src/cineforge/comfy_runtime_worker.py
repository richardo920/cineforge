"""Fixed M5 wheel/source extraction and protected manifest verification; no resolver."""
from __future__ import annotations

import hashlib
import re
import tarfile
import time
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from cineforge.comfy_runtime_plan import COMFY_SOURCE, _files, comfy_plan, comfy_subject
from cineforge.native_remote_bootstrap import nr_check, nr_private, nr_read, nr_write
from cineforge.native_remote_plan import nr_digest, nr_encode, nr_json
from cineforge.native_remote_worker import _destination, _download, _install_wheel, _parent
from cineforge.native_speech_integrity import NativeBinary, NativeTree, verify_binary, verify_tree


def install_wheel(path: Path, target: Path, wheel: dict[str, Any],
                  manifest: dict[str, tuple[int, str]], deadline: float) -> None:
    # Validate reviewed bytes, headers, tags, notices and all destination conflicts
    # before the shared RECORD verifier/exclusive writer sees any extraction target.
    verify_binary(NativeBinary(str(path), wheel["size_bytes"], wheel["sha256"]), deadline)
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        destinations = [_destination(name) for name in names]
        if len(set(destinations)) != len(destinations) or set(destinations) & set(manifest):
            raise ValueError("closed Comfy wheel destination overlaps")
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
    _install_wheel(path, target, wheel, manifest, deadline, top_level_metadata=True)
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
    # Do not infer closure from installed-after-resolution freeze. Require the full
    # separately reviewed manifest and exactly one selected METADATA per package.
    actual = {}
    for relative, _, _ in plan["installed"]:
        if relative.endswith(".dist-info/METADATA") and relative.count("/") == 1:
            raw = nr_read(runtime / relative, maximum=1024**2)
            headers = BytesParser().parsebytes(raw)
            name = re.sub(r"[-_.]+", "-", headers.get("Name", "")).lower()
            if name in actual:
                raise ValueError("duplicate closed Comfy distribution")
            actual[name] = (headers.get("Version"), hashlib.sha256(raw).hexdigest())
    if actual != {w["name"]: (w["version"], w["metadata_sha256"]) for w in plan["wheels"]}:
        raise ValueError("closed Comfy installed distribution closure differs")
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
    for name in ("ComfyUI", "comfy-runtime", "comfy-downloads", "comfy-output", "comfy-user", "comfy-temp"):
        (root / name).mkdir(mode=0o700)
    for binary in [plan["interpreter"], *plan["native"]]:
        verify_binary(NativeBinary(**binary), deadline)
    downloads = root / "comfy-downloads"
    manifest: dict[str, tuple[int, str]] = {}
    for wheel in plan["wheels"]:
        path = downloads / wheel["filename"]
        _download(path, wheel["url"], wheel["size_bytes"], wheel["sha256"], deadline)
        install_wheel(path, root / "comfy-runtime", wheel, manifest, deadline)
    if [[name, size, sha] for name, (size, sha) in sorted(manifest.items())] != plan["installed"]:
        raise ValueError("closed Comfy installed files differ from prior review")
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
    import subprocess
    process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
    nr_write(root / "server-pid.json", nr_encode({"pid": process.pid}))
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
                if reply.status == 200:
                    nr_check(deadline)
                    return {"verification": verification, "runtime": runtime,
                            "comfy": current_comfy(request, native, deadline)}
        except OSError:
            pass
        time.sleep(.1)
    raise ValueError("closed Comfy production startup deadline exceeded")


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
    hardware = s2v_hardware_admission(root, deadline)
    for name in ("gpu_raw_name", "capability", "device_count", "nvml_total_mib",
                 "nvml_reserved_mib", "driver_version"):
        if hardware[name] != facts["runtime"][name]:
            raise ValueError("closed Comfy actual GPU or driver changed")
    nr_check(deadline)
    return {"subject_sha256": comfy_subject(plan),
            "installed_sha256": nr_digest(plan["installed"]), "probe_sha256": nr_digest(facts)}
