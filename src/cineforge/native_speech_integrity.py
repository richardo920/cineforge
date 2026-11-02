"""Explicit local speech staging identities; no discovery, install or model import.

The operator supplies reviewed runtime files and evidence through trusted Python
code. These declarations do not authenticate an operator or grant media rights.
Private staging must stay unchanged; this is not a hostile same-UID writer sandbox.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass, field
from email.parser import BytesParser
from typing import Any

from cineforge.json_codec import strict_loads
from cineforge.native_speech_pins import BASE_FILES, DESIGN_FILES, RUNTIME_DISTRIBUTIONS, SDK_FILES

FilePin = tuple[str, int, str, str]
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TREE_BYTES = 64 * 1024**3
_MODEL_CONFIG_PATHS = frozenset({
    "config.json", "generation_config.json", "preprocessor_config.json",
    "speech_tokenizer/config.json", "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json", "tokenizer_config.json",
})


class NativeSpeechError(ValueError):
    """Only static diagnostics; cleanup uncertainty never becomes a success."""

    def __init__(self, message: str, *, cleanup_uncertain: bool = False,
                 owned_process_id: int | None = None) -> None:
        if owned_process_id is not None and (type(owned_process_id) is not int or owned_process_id <= 0):
            raise ValueError("invalid owned process provenance")
        super().__init__(message)
        self.cleanup_uncertain = cleanup_uncertain
        # Diagnostic provenance only; a historical PID is never future signal authority.
        self.owned_process_id = owned_process_id


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _absolute(path: object) -> None:
    if (type(path) is not str or not os.path.isabs(path) or "\x00" in path
            or len(path) > 4096
            or any(0xD800 <= ord(c) <= 0xDFFF for c in path)
            or any(p in ("", ".", "..") for p in path.split("/")[1:])):
        raise NativeSpeechError("invalid local speech staging path")


def _hash(value: object) -> None:
    if type(value) is not str or not _HASH.fullmatch(value) or value == "0" * 64:
        raise NativeSpeechError("missing speech staging identity")


@dataclass(frozen=True, slots=True)
class NativeBinary:
    path: str = field(repr=False)
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _absolute(self.path)
        _hash(self.sha256)
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= 1024**3:
            raise NativeSpeechError("invalid speech capability size")


@dataclass(frozen=True, slots=True)
class NativeTree:
    root: str = field(repr=False)
    files: tuple[FilePin, ...]

    def __post_init__(self) -> None:
        _absolute(self.root)
        if type(self.files) is not tuple or not 1 <= len(self.files) <= 100000:
            raise NativeSpeechError("invalid speech staging manifest")
        seen: set[str] = set()
        total = 0
        path_bytes = 0
        for pin in self.files:
            if type(pin) is not tuple or len(pin) != 4:
                raise NativeSpeechError("invalid speech staging file")
            path, size, algorithm, checksum = pin
            if (type(path) is not str or len(path) > 1024 or path.startswith("/")
                    or "\\" in path or "\x00" in path
                    or any(0xD800 <= ord(c) <= 0xDFFF for c in path)
                    or any(p in ("", ".", "..") for p in path.split("/")) or path in seen
                    or type(size) is not int or not 0 <= size <= 16 * 1024**3
                    or algorithm != "sha256" or type(checksum) is not str
                    or not _HASH.fullmatch(checksum)):
                raise NativeSpeechError("invalid speech staging file")
            seen.add(path)
            path_bytes += len(path.encode("utf-8"))
            total += size
        if total > _MAX_TREE_BYTES or path_bytes > 8 * 1024**2:
            raise NativeSpeechError("speech staging exceeds size bound")


@dataclass(frozen=True, slots=True)
class NativeHardware:
    """An operator's existing worker-admission expectation, not cloud discovery."""

    operator_worker_id: str
    gpu_name: str
    memory_mib: int
    capability: tuple[int, int]

    def __post_init__(self) -> None:
        if (type(self.operator_worker_id) is not str
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.operator_worker_id)
                or type(self.gpu_name) is not str or not 1 <= len(self.gpu_name) <= 128
                or not self.gpu_name.isprintable() or type(self.memory_mib) is not int
                or not 1024 <= self.memory_mib <= 1024**2 or type(self.capability) is not tuple
                or len(self.capability) != 2 or any(type(x) is not int for x in self.capability)
                or not 8 <= self.capability[0] <= 99 or not 0 <= self.capability[1] <= 99):
            raise NativeSpeechError("unsupported operator speech hardware expectation")

    def require_matches(self, actual: dict[str, Any]) -> None:
        if (set(actual) != {"torch_version", "torchaudio_version", "cuda_version", "gpu_name",
                            "memory_bytes", "capability", "operator_worker_id", "bf16_supported",
                            "attention"} or actual["torch_version"] != "2.8.0+cu128"
                or actual["torchaudio_version"] != "2.8.0+cu128" or actual["cuda_version"] != "12.8"
                or actual["gpu_name"] != self.gpu_name or type(actual["memory_bytes"]) is not int
                or actual["memory_bytes"] // 1024**2 != self.memory_mib
                or type(actual["capability"]) is not list
                or any(type(x) is not int for x in actual["capability"])
                or actual["capability"] != list(self.capability)
                or actual["operator_worker_id"] != self.operator_worker_id
                or actual["bf16_supported"] is not True or actual["attention"] != "sdpa-math"):
            raise NativeSpeechError("observed speech GPU differs from admitted worker")


@dataclass(frozen=True, slots=True)
class NativeSpeechSelection:
    """A trusted operator's exact staged runtime, never a creative-plan field.

Review evidence covers the full subject including actual installed interpreter,
native library and runtime-file hashes, dependency/advisory disposition and notices.
Hash equality binds the declaration; it is not an authenticated review signature.
"""

    interpreter: NativeBinary
    seccomp: NativeBinary
    runtime: NativeTree
    sdk: NativeTree
    base: NativeTree
    design: NativeTree
    hardware: NativeHardware
    reviewed_subject_sha256: str
    review_evidence_sha256: str

    def __post_init__(self) -> None:
        for value, kind in ((self.interpreter, NativeBinary), (self.seccomp, NativeBinary),
                            (self.runtime, NativeTree), (self.sdk, NativeTree),
                            (self.base, NativeTree), (self.design, NativeTree),
                            (self.hardware, NativeHardware)):
            if type(value) is not kind:
                raise NativeSpeechError("invalid explicit native speech selection")
        if (self.sdk.files != SDK_FILES or self.base.files != BASE_FILES
                or self.design.files != DESIGN_FILES
                or any(p[2] != "sha256" for p in self.runtime.files)):
            raise NativeSpeechError("speech source or model differs from fixed pins")
        _hash(self.reviewed_subject_sha256)
        _hash(self.review_evidence_sha256)
        if self.reviewed_subject_sha256 != self.subject_sha256():
            raise NativeSpeechError("speech runtime review is stale")

    def subject_sha256(self) -> str:
        return native_subject(self.interpreter, self.seccomp, self.runtime, self.sdk,
                              self.base, self.design, self.hardware)


def native_subject(interpreter: NativeBinary, seccomp: NativeBinary, runtime: NativeTree,
                   sdk: NativeTree, base: NativeTree, design: NativeTree,
                   hardware: NativeHardware) -> str:
    """Subject for a separately reviewed staging receipt; computing it is not review."""
    return digest({"interpreter": (interpreter.size_bytes, interpreter.sha256),
                   "seccomp": (seccomp.size_bytes, seccomp.sha256),
                   "runtime": runtime.files, "sdk": sdk.files,
                   "base": base.files, "design": design.files,
                   "hardware": (hardware.operator_worker_id, hardware.gpu_name,
                                hardware.memory_mib, hardware.capability),
                   "distributions": RUNTIME_DISTRIBUTIONS,
                   "policy": "local-qwen-bf16-sdpa-seccomp-v1"})


def _directory(path: str, *, private: bool) -> int:
    _absolute(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.split("/")[1:]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            parent = os.fstat(descriptor)
            if (parent.st_uid not in (0, os.getuid()) or parent.st_mode & 0o022
                    and not (parent.st_uid == 0 and parent.st_mode & stat.S_ISVTX)):
                raise NativeSpeechError("speech staging ancestor is not protected")
        info = os.fstat(descriptor)
        if private and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise NativeSpeechError("speech staging must be private and owned")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_file(parent: int, name: str, pin: FilePin, deadline: float,
               *, capture: bool = False) -> tuple[str, bytes]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size != pin[1] or before.st_uid not in (0, os.getuid())
                or before.st_mode & 0o022 or capture and before.st_size > 4 * 1024**2):
            raise NativeSpeechError("speech staging file identity rejected")
        sha256 = hashlib.sha256()
        data = bytearray()
        count = 0
        while True:
            if time.monotonic() >= deadline:
                raise NativeSpeechError("speech staging deadline exceeded")
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > pin[1]:
                raise NativeSpeechError("speech staging changed during read")
            sha256.update(block)
            if capture:
                data.extend(block)
        after = os.fstat(descriptor)
        def identity(s: os.stat_result) -> tuple[int, ...]:
            return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
                    s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        actual = sha256.hexdigest()
        linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (identity(before) != identity(after) or identity(before) != identity(linked)
                or count != pin[1] or actual != pin[3]):
            raise NativeSpeechError("speech staging hash or identity changed")
        return sha256.hexdigest(), bytes(data)
    finally:
        os.close(descriptor)


def verify_binary(binary: NativeBinary, deadline: float) -> None:
    parent, name = os.path.split(binary.path)
    descriptor = _directory(parent, private=False)
    try:
        _read_file(descriptor, name, (name, binary.size_bytes, "sha256", binary.sha256), deadline)
    finally:
        os.close(descriptor)


def _distribution_identity(path: str, data: bytes) -> tuple[str, str]:
    """Only root dist-info identifies an installed distribution; vendors stay data."""
    if path.count("/") != 1 or not path.endswith(".dist-info/METADATA"):
        raise NativeSpeechError("invalid root speech distribution metadata")
    parsed = BytesParser().parsebytes(data)
    if len(parsed.get_all("Name", [])) != 1 or len(parsed.get_all("Version", [])) != 1:
        raise NativeSpeechError("ambiguous speech runtime distribution identity")
    package = re.sub(r"[-_.]+", "-", str(parsed["Name"]).lower())
    version = str(parsed["Version"])
    directory = path.split("/")[0][:-len(".dist-info")]
    name, separator, directory_version = directory.rpartition("-")
    if (not package or not version or not separator or directory_version != version
            or re.sub(r"[-_.]+", "-", name.lower()) != package):
        raise NativeSpeechError("speech runtime metadata directory differs")
    return package, version


def verify_tree(tree: NativeTree, deadline: float, *, distributions: bool = False,
                model_role: str | None = None) -> str:
    """Hash exact resources; semantic config checks require the pinned model role.

    Runtime/SDK/source resources are opaque data, including JSONC/build configs.
    Base/Design callers must supply their fixed role and exact curated pin set.
    """
    if type(distributions) is not bool:
        raise NativeSpeechError("invalid speech tree role")
    if model_role is None:
        if tree.files in (BASE_FILES, DESIGN_FILES):
            raise NativeSpeechError("explicit pinned model role required")
        configs: frozenset[str] = frozenset()
    else:
        if type(model_role) is not str or model_role not in ("base", "design") or distributions:
            raise NativeSpeechError("invalid speech model role")
        pins = BASE_FILES if model_role == "base" else DESIGN_FILES
        if tree.files != pins:
            raise NativeSpeechError("speech model tree differs from role pins")
        configs = _MODEL_CONFIG_PATHS
    expected = {p[0]: p for p in tree.files}
    directories = {""} | {path.rsplit("/", i)[0] for path in expected
                             for i in range(1, path.count("/") + 1)}
    observed: set[str] = set()
    packages: dict[str, str] = {}
    checksums: list[tuple[str, str]] = []
    root = _directory(tree.root, private=True)

    def walk(descriptor: int, prefix: str) -> None:
        if time.monotonic() >= deadline:
            raise NativeSpeechError("speech staging deadline exceeded")
        for name in os.listdir(descriptor):
            path = prefix + name
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                if path not in directories:
                    raise NativeSpeechError("unexpected speech staging directory")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
                try:
                    walk(child, path + "/")
                finally:
                    os.close(child)
            else:
                if path not in expected:
                    raise NativeSpeechError("unexpected speech staging file")
                metadata = (distributions and path.count("/") == 1
                            and path.endswith(".dist-info/METADATA"))
                config = path in configs
                checksum, data = _read_file(descriptor, name, expected[path], deadline,
                                            capture=metadata or config)
                observed.add(path)
                checksums.append((path, checksum))
                if metadata:
                    package, version = _distribution_identity(path, data)
                    if package in packages:
                        raise NativeSpeechError("duplicate speech runtime distribution")
                    packages[package] = version
                if config:
                    _config(data)
    try:
        walk(root, "")
    finally:
        os.close(root)
    if observed != set(expected):
        raise NativeSpeechError("speech staging is incomplete")
    if distributions and packages != dict(RUNTIME_DISTRIBUTIONS):
        raise NativeSpeechError("speech runtime distribution closure differs")
    return digest(sorted(checksums))


def _config(data: bytes) -> None:
    value = None
    failed = False
    try:
        value = strict_loads(data)
    except (ValueError, UnicodeError, RecursionError):
        failed = True
    if failed or type(value) is not dict:
        raise NativeSpeechError("invalid strict speech model configuration") from None
    forbidden = {"auto_map", "trust_remote_code", "_attn_implementation_internal",
                 "custom_pipelines", "code_revision"}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if forbidden & set(item):
                raise NativeSpeechError("remote speech configuration is unsupported")
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif type(item) is float and not math.isfinite(item):
            raise NativeSpeechError("nonfinite speech model configuration")
    visit(value)
