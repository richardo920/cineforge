"""Closed M5 Comfy declarations; artifact/ABI review is a separate trusted input."""
from __future__ import annotations

import base64
import hashlib
import re
from typing import Any
from urllib.parse import urlsplit

from cineforge.native_remote_plan import nr_digest, nr_hash
from cineforge.native_speech_integrity import NativeBinary, NativeTree
from cineforge.s2v_profile import IMAGE
from cineforge.worker_profiles import S2V_WORKER

COMFY_POLICY = "m5-closed-wheel-runtime-v1"
COMFY_EXECUTION = {"PYTORCH_JIT": "0", "TORCH_COMPILE_DISABLE": "1",
                   "attention": "pytorch-sdpa", "serialized_jit": "forbidden",
                   "model_format": "four-fixed-safetensors", "custom_nodes": False,
                   "import_layout": "site-only-fixed-sympy-manpage-v1"}
COMFY_SHARED_OWNERS = tuple(sorted("nvidia-" + name + "-cu12" for name in (
    "cublas", "cuda-cupti", "cuda-nvrtc", "cuda-runtime", "cufft", "cufile", "curand",
    "cusolver", "cusparse", "nvjitlink", "nvtx")))
COMFY_SHARED_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # pragma: allowlist secret - public empty-body hash
COMFY_EXECUTABLES = frozenset({"site/torch/bin/protoc", "site/torch/bin/protoc-3.13.0.0",
    "site/torch/bin/torch_shm_manager", "site/triton/backends/nvidia/bin/cuobjdump",
    "site/triton/backends/nvidia/bin/nvdisasm", "site/triton/backends/nvidia/bin/ptxas"})
COMFY_ABI = "cpython311-linux-x86_64-glibc2.28"
COMFY_REQUIRES_DIST_MAX = 1024
COMFY_WHEEL_NOTICE_MAX = 32 * 1024**2
COMFY_SOURCE = S2V_WORKER.runner_revision
COMFY_REQUIREMENTS_SHA256 = "86abf78ac24e7dfd6ba66c8fc031b253be0eb4f3c6f048e7f32d5405986eb991"  # pragma: allowlist secret - public source hash
COMFY_ROOTS = frozenset((
    "comfyui-frontend-package", "comfyui-workflow-templates", "comfyui-embedded-docs",
    "torch", "torchsde", "torchvision", "numpy", "einops", "transformers", "tokenizers",
    "sentencepiece", "safetensors", "aiohttp", "yarl", "pyyaml", "pillow", "scipy", "tqdm",
    "psutil", "alembic", "sqlalchemy", "filelock", "av", "comfy-aimdo", "comfy-kitchen",
    "requests", "simpleeval", "blake3", "kornia", "spandrel", "pydantic", "pydantic-settings",
    "pyopengl", "comfy-angle",
))
COMFY_FIXED = {"torch": "2.8.0", "torchvision": "0.23.0", "torchaudio": "2.8.0",
    "transformers": "4.57.3", "numpy": "2.2.6", "comfyui-frontend-package": "1.57.6",
    "comfyui-workflow-templates": "0.11.100", "comfyui-embedded-docs": "0.5.15",
    "comfy-aimdo": "0.6.0", "comfy-kitchen": "0.2.40"}


def _keys(value: Any, names: set[str]) -> None:
    if type(value) is not dict or set(value) != names:
        raise ValueError("incomplete closed Comfy declaration")


def _files(value: Any, root: str) -> tuple[tuple[str, int, str, str], ...]:
    if type(value) is not list or any(type(row) is not list or len(row) != 3 for row in value):
        raise ValueError("invalid closed Comfy file manifest")
    pins = tuple((path, size, "sha256", sha) for path, size, sha in value)
    NativeTree(root, pins)
    if value != sorted(value):
        raise ValueError("closed Comfy file manifest must be uniquely ordered")
    return pins


def _wheel(value: Any) -> None:
    _keys(value, {"name", "version", "filename", "url", "size_bytes", "sha256",
        "metadata_sha256", "wheel_sha256", "requires_python", "requires_dist", "tags", "notices"})
    name, version, filename = value["name"], value["version"], value["filename"]
    if (type(name) is not str or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            or type(version) is not str or re.fullmatch(r"[A-Za-z0-9.+!_]{1,80}", version) is None
            or name in COMFY_FIXED and version != COMFY_FIXED[name]
            or type(filename) is not str or len(filename) > 240 or not filename.endswith(".whl")
            or type(value["size_bytes"]) is not int or not 1 <= value["size_bytes"] <= 2 * 1024**3):
        raise ValueError("unsupported closed Comfy wheel identity")
    parts = filename[:-4].split("-")
    if (len(parts) not in (5, 6) or re.sub(r"[-_.]+", "-", parts[0]).lower() != name
            or parts[1] != version or len(parts) == 6 and not parts[2].isdigit()):
        raise ValueError("unsupported closed Comfy wheel filename")
    python, abi, platform = parts[-3:]
    tags = set(python.split("."))
    if (abi == "cp311" and tags != {"cp311"}
            or abi == "abi3" and re.fullmatch(r"cp3(?:[2-9]|10|11)", python) is None
            or abi == "none" and not (tags <= {"py2", "py3", "cp311"} and tags & {"py3", "cp311"})
            or abi not in {"cp311", "abi3", "none"}):
        raise ValueError("unsupported closed Comfy Python ABI")
    for tag in platform.split("."):
        if tag == "any" and abi == "none" or re.fullmatch(r"manylinux(?:1|2010|2014)_x86_64", tag):
            continue
        match = re.fullmatch(r"manylinux_2_([0-9]+)_x86_64", tag)
        if match is None or not 5 <= int(match[1]) <= 28:
            raise ValueError("unsupported closed Comfy platform ABI")
    parsed = urlsplit(value["url"]) if type(value["url"]) is str else None
    if (parsed is None or parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org"
            or not parsed.path.startswith("/packages/") or parsed.path.rsplit("/", 1)[-1] != filename
            or parsed.query or parsed.fragment or "%" in parsed.path
            or any(p in ("", ".", "..") for p in parsed.path.split("/")[1:])):
        raise ValueError("unsupported closed Comfy wheel origin")
    for key in ("sha256", "metadata_sha256", "wheel_sha256"):
        nr_hash(value[key])
    if (type(value["requires_python"]) is not str or len(value["requires_python"]) > 256
            or type(value["requires_dist"]) is not list or len(value["requires_dist"]) > COMFY_REQUIRES_DIST_MAX
            or any(type(v) is not str or not 1 <= len(v) <= 4096 for v in value["requires_dist"])
            # Requires-Dist is an exact header multiset, including repetitions.
            or value["requires_dist"] != sorted(value["requires_dist"])
            or type(value["tags"]) is not list or not value["tags"]
            or any(type(v) is not str or len(v) > 256 for v in value["tags"])
            or value["tags"] != sorted(set(value["tags"]))):
        raise ValueError("incomplete closed Comfy distribution metadata")
    expected_tags = sorted(f"{p}-{abi}-{t}" for p in python.split(".") for t in platform.split("."))
    if value["tags"] != expected_tags:
        raise ValueError("closed Comfy filename and WHEEL tags disagree")
    if type(value["notices"]) is not list:
        raise ValueError("closed Comfy wheel notice declaration missing")
    if value["notices"]:
        pins = _files(value["notices"], "/closed-comfy/notices")
        if any(size > COMFY_WHEEL_NOTICE_MAX for _, size, _, _ in pins):
            raise ValueError("closed Comfy wheel notice exceeds bound")


def notice_bytes(value: Any) -> bytes:
    _keys(value, {"source_uri", "source_revision", "packages", "size_bytes", "sha256", "content_base64"})
    if (type(value["size_bytes"]) is not int or not 1 <= value["size_bytes"] <= 1024**2
            or type(value["content_base64"]) is not str or len(value["content_base64"]) > 1400000
            or type(value["source_revision"]) is not str
            or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value["source_revision"]) is None
            or value["source_revision"].lower() in {"unknown", "pending", "unreviewed", "main", "master", "head", "latest"}):
        raise ValueError("invalid fixed Comfy notice addendum")
    nr_hash(value["sha256"])
    uri = urlsplit(value["source_uri"]) if type(value["source_uri"]) is str else None
    if (uri is None or len(value["source_uri"]) > 4096 or uri.scheme != "https"
            or not uri.hostname or uri.username or uri.password or not uri.path
            or uri.query or uri.fragment):
        raise ValueError("invalid captured Comfy notice provenance")
    raw = base64.b64decode(value["content_base64"], validate=True)
    if (len(raw) != value["size_bytes"] or hashlib.sha256(raw).hexdigest() != value["sha256"]
            or base64.b64encode(raw).decode() != value["content_base64"] or "\x00" in raw.decode("utf-8")):
        raise ValueError("captured Comfy notice bytes differ")
    return raw


def comfy_plan(value: Any) -> dict[str, Any]:
    _keys(value, {"version", "policy", "image", "abi", "source", "wheels", "installed",
                  "interpreter", "native", "review", "notices", "execution", "shared_files", "executables"})
    _keys(value["execution"], set(COMFY_EXECUTION))
    if (type(value["version"]) is not int or value["version"] != 1
            or value["policy"] != COMFY_POLICY or value["image"] != IMAGE
            or value["abi"] != COMFY_ABI or value["execution"] != COMFY_EXECUTION
            or any(type(value["execution"].get(k)) is not type(v) for k, v in COMFY_EXECUTION.items())):
        raise ValueError("unsupported closed Comfy recipe")
    source = value["source"]
    _keys(source, {"revision", "size_bytes", "sha256", "files"})
    if (source["revision"] != COMFY_SOURCE or type(source["size_bytes"]) is not int
            or not 1 <= source["size_bytes"] <= 128 * 1024**2):
        raise ValueError("unsupported closed Comfy source")
    nr_hash(source["sha256"])
    files = _files(source["files"], "/closed-comfy/source")
    indexed = {p: (size, sha) for p, size, _, sha in files}
    if (indexed.get("requirements.txt") != (529, COMFY_REQUIREMENTS_SHA256)
            or not {"main.py", "LICENSE", "comfy/cli_args.py", "nodes.py"} <= set(indexed)):
        raise ValueError("closed Comfy source or notices incomplete")
    wheels = value["wheels"]
    if type(wheels) is not list or not len(COMFY_ROOTS) <= len(wheels) <= 256:
        raise ValueError("closed Comfy dependency set incomplete")
    for wheel in wheels:
        _wheel(wheel)
    names = [w["name"] for w in wheels]
    if (names != sorted(set(names)) or not COMFY_ROOTS | set(COMFY_FIXED) <= set(names)
            or sum(w["size_bytes"] for w in wheels) > 8 * 1024**3):
        raise ValueError("closed Comfy dependency set differs")
    expected_shared = [{"path": "nvidia/__init__.py", "size_bytes": 0,
        "sha256": COMFY_SHARED_EMPTY_SHA256, "owners": list(COMFY_SHARED_OWNERS)}]
    shared = value["shared_files"]
    if (type(shared) is not list or shared not in ([], expected_shared)
            or bool(set(names) & set(COMFY_SHARED_OWNERS)) != bool(shared)
            or shared and (not set(COMFY_SHARED_OWNERS) <= set(names)
                           or type(shared[0]["size_bytes"]) is not int)):
        raise ValueError("closed Comfy shared namespace declaration differs")
    if type(value["notices"]) is not list or len(value["notices"]) > 256:
        raise ValueError("incomplete captured Comfy notices")
    covered = set()
    total_notices = 0
    for notice in value["notices"]:
        total_notices += len(notice_bytes(notice))
        packages = notice["packages"]
        if type(packages) is not list or not packages or len(packages) > 256:
            raise ValueError("Comfy notice version mapping missing")
        selected = {w["name"]: w["version"] for w in wheels}
        for package in packages:
            _keys(package, {"name", "version"})
            if package["name"] not in selected or package["version"] != selected[package["name"]]:
                raise ValueError("Comfy notice version mapping differs")
            covered.add(package["name"])
    if total_notices > 4 * 1024**2 or any(not w["notices"] and w["name"] not in covered for w in wheels):
        raise ValueError("Comfy package notices incomplete or excessive")
    installed = _files(value["installed"], "/closed-comfy/runtime")
    if "site/comfyui_frontend_package/static/index.html" not in {p for p, _, _, _ in installed}:
        raise ValueError("closed Comfy installed manifest missing")
    indexed_installed = {p: (size, sha) for p, size, _, sha in installed}
    executables = value["executables"]
    if type(executables) is not list:
        raise ValueError("closed Comfy executable declaration missing")
    executable_paths = []
    for executable in executables:
        _keys(executable, {"path", "size_bytes", "sha256", "mode"})
        if (type(executable["path"]) is not str or executable["path"] not in COMFY_EXECUTABLES
                or type(executable["mode"]) is not int or executable["mode"] != 0o700
                or type(executable["size_bytes"]) is not int
                or indexed_installed.get(executable["path"]) != (executable["size_bytes"], executable["sha256"])):
            raise ValueError("closed Comfy executable identity differs")
        executable_paths.append(executable["path"])
    if (executable_paths != sorted(set(executable_paths))
            or set(executable_paths) != COMFY_EXECUTABLES & set(indexed_installed)):
        raise ValueError("closed Comfy executable closure incomplete")
    _keys(value["interpreter"], {"path", "size_bytes", "sha256"})
    NativeBinary(**value["interpreter"])
    if type(value["native"]) is not list or not 1 <= len(value["native"]) <= 256:
        raise ValueError("closed Comfy native identities missing")
    paths = []
    for binary in value["native"]:
        _keys(binary, {"path", "size_bytes", "sha256"})
        paths.append(NativeBinary(**binary).path)
    if paths != sorted(set(paths)) or "/usr/bin/nvidia-smi" not in paths:
        raise ValueError("ambiguous closed Comfy native identities")
    _keys(value["review"], {"artifacts", "dependencies", "licenses", "advisories", "abi"})
    for sha in value["review"].values():
        nr_hash(sha)
    result: dict[str, Any] = value
    return result


def comfy_subject(value: dict[str, Any]) -> str:
    return nr_digest(comfy_plan(value))
