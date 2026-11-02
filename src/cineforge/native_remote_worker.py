"""Fixed guarded native staging/invoke implementation; never a public CLI or resolver."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import stat
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.native_remote_bootstrap import nr_check, nr_private, nr_read, nr_write
from cineforge.native_remote_plan import (
    NR_BLOB_MAX,
    NR_CHUNK,
    nr_encode,
    nr_json,
    nr_recipe,
    nr_token,
)
from cineforge.native_speech import NativeSpeechRuntime, OriginalVoiceDesign
from cineforge.native_speech_integrity import (
    NativeSpeechSelection,
    _distribution_identity,
    digest,
    verify_binary,
    verify_tree,
)
from cineforge.native_speech_pins import (
    BASE_FILES,
    DESIGN_FILES,
    DESIGN_REVISION,
    SDK_FILES,
    SDK_REVISION,
)
from cineforge.native_speech_process import pack, unpack
from cineforge.native_speech_worker import _selection
from cineforge.speech import BASE_MODEL, BASE_REVISION, QwenBaseRequest

_RETAINED_DATA = {
    ("sympy", "1.14.0"): "sympy-1.14.0.data/data/share/man/man1/isympy.1",
    ("numba", "0.68.1"): "numba-0.68.1.data/scripts/numba",
    ("ruff", "0.17.3"): "ruff-0.17.3.data/scripts/ruff",
}


class _NativeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        parsed = urllib.parse.urlsplit(newurl)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.port not in (None, 443)
            or parsed.username
            or parsed.password
            or not (
                host in {"raw.githubusercontent.com", "huggingface.co", "files.pythonhosted.org"}
                or host.endswith(".hf.co")
                or host.endswith(".huggingface.co")
            )
        ):
            raise ValueError("unsupported native staging redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(path: Path, url: str, size: int, sha: str, deadline: float) -> None:
    nr_check(deadline)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NativeRedirect())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        actual, count = hashlib.sha256(), 0
        with opener.open(url, timeout=min(15, max(0.01, deadline - time.monotonic()))) as response:
            while True:
                nr_check(deadline)
                data = response.read(min(1024 * 1024, size + 1 - count))
                if not data:
                    break
                count += len(data)
                if count > size:
                    raise ValueError("native staging body exceeds pin")
                actual.update(data)
                with os.fdopen(fd, "ab", closefd=False) as stream:
                    stream.write(data)
        os.fsync(fd)
        if count != size or actual.hexdigest() != sha:
            raise ValueError("native staging body differs from pin")
    finally:
        os.close(fd)
    nr_check(deadline)


def _parts(name: str) -> tuple[str, ...]:
    if (
        type(name) is not str
        or len(name) > 1024
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise ValueError("unsupported native wheel path")
    return tuple(name.split("/"))


def _destination(name: str, distribution: tuple[str, str]) -> str:
    parts = _parts(name)
    if parts[0].endswith(".data"):
        expected = distribution[0].replace("-", "_") + "-" + distribution[1] + ".data"
        if parts[0] != expected or len(parts) < 3:
            raise ValueError("unsupported native wheel relocation")
        if name == _RETAINED_DATA.get(distribution):
            # Preserve only these pinned bodies at original archive paths, as
            # mode0600 data. Never install bin/PATH entries, wrappers or scripts.
            # These directories are not import roots; full manifests bind bytes.
            return name
        if parts[1] not in ("purelib", "platlib"):
            raise ValueError("unsupported native wheel relocation")
        return "/".join(parts[2:])
    return name


def _parent(root: Path, relative: str) -> Path:
    parts = _parts(relative)
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        if not parent.exists():
            parent.mkdir(mode=0o700)
        nr_private(parent, directory=True)
    return parent / parts[-1]


def _install_wheel(
    path: Path,
    root: Path,
    declaration: dict[str, Any],
    manifest: dict[str, tuple[int, str]],
    deadline: float,
) -> None:
    nr_private(path)
    with zipfile.ZipFile(path) as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in infos]
        if len(set(names)) != len(names) or len(infos) > 100000:
            raise ValueError("ambiguous native wheel closure")
        metadata = [n for n in names if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
        records = [n for n in names if n.count("/") == 1 and n.endswith(".dist-info/RECORD")]
        if (
            len(metadata) != 1
            or len(records) != 1
            or metadata[0].split("/")[0] != records[0].split("/")[0]
        ):
            raise ValueError("native wheel metadata closure differs")
        if (
            archive.getinfo(metadata[0]).file_size > 1024**2
            or archive.getinfo(records[0]).file_size > 8 * 1024**2
        ):
            raise ValueError("excessive native wheel metadata")
        if _distribution_identity(metadata[0], archive.read(metadata[0])) != (
            declaration["name"],
            declaration["version"],
        ):
            raise ValueError("native wheel distribution differs")
        record: dict[str, tuple[str, int]] = {}
        for row in csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8"))):
            if len(row) != 3 or row[0] in record:
                raise ValueError("ambiguous native wheel RECORD")
            _parts(row[0])
            if row[0] == records[0]:
                if row[1:] != ["", ""]:
                    raise ValueError("invalid native RECORD self entry")
                record[row[0]] = ("", -1)
            else:
                if not row[1].startswith("sha256=") or not row[2].isdigit():
                    raise ValueError("native wheel RECORD must hash every body")
                checksum = base64.urlsafe_b64decode(row[1][7:] + "==").hex()
                if len(checksum) != 64:
                    raise ValueError("invalid native wheel RECORD hash")
                record[row[0]] = (checksum, int(row[2]))
        if set(record) != set(names):
            raise ValueError("native wheel RECORD closure differs")
        total = sum(p[0] for p in manifest.values())
        destinations: set[str] = set()
        for info in infos:
            nr_check(deadline)
            mode = info.external_attr >> 16
            if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("native wheel special file rejected")
            relative = _destination(info.filename, (declaration["name"], declaration["version"]))
            if relative in destinations:
                raise ValueError("ambiguous native wheel destination")
            destinations.add(relative)
            target = _parent(root, relative)
            if info.file_size > 16 * 1024**3 or total + info.file_size > 64 * 1024**3:
                raise ValueError("native runtime exceeds manifest bound")
            actual, count = hashlib.sha256(), 0
            duplicate = relative in manifest
            fd = (
                None
                if duplicate
                else os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            )
            try:
                with archive.open(info) as source:
                    while data := source.read(1024 * 1024):
                        nr_check(deadline)
                        actual.update(data)
                        count += len(data)
                        if count > info.file_size:
                            raise ValueError("native wheel body exceeds declaration")
                        if fd is not None:
                            with os.fdopen(fd, "ab", closefd=False) as stream:
                                stream.write(data)
                if fd is not None:
                    os.fsync(fd)
            finally:
                if fd is not None:
                    os.close(fd)
            sha = actual.hexdigest()
            recorded_sha, recorded_size = record[info.filename]
            if (
                count != info.file_size
                or info.filename != records[0]
                and (sha != recorded_sha or count != recorded_size)
                or duplicate
                and manifest[relative] != (count, sha)
            ):
                raise ValueError("native wheel content identity differs")
            if not duplicate:
                total += count
            manifest[relative] = (count, sha)
            if len(manifest) > 100000:
                raise ValueError("native runtime file count exceeded")


def _setup(root: Path, deadline: float) -> dict[str, Any]:
    nr_write(root / "setup-consumed", b"consumed before installation")
    for role in ("package", "recipe", "sox"):
        nr_read(root / (role + ".blob.done"))
    recipe = nr_recipe(nr_json(nr_read(root / "recipe.blob")))
    preflight = nr_json(nr_read(root / "preflight.json"))
    # Recheck actual measured fixed binaries before any multi-GB transfer.
    from cineforge.native_speech_integrity import NativeBinary, NativeTree

    verify_binary(NativeBinary(**preflight["interpreter"]), deadline)
    verify_binary(NativeBinary(**preflight["seccomp"]), deadline)
    downloads = root / "downloads"
    downloads.mkdir(mode=0o700)
    runtime = root / "runtime"
    runtime.mkdir(mode=0o700)
    manifest: dict[str, tuple[int, str]] = {}
    for wheel in recipe["wheels"]:
        nr_check(deadline)
        if wheel["name"] == "sox":
            path = root / "sox.blob"
            raw = nr_read(path)
            if (
                len(raw) != wheel["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != wheel["sha256"]
            ):
                raise ValueError("built native wheel differs from reviewed recipe")
        else:
            path = downloads / wheel["filename"]
            _download(path, wheel["url"], wheel["size_bytes"], wheel["sha256"], deadline)
        _install_wheel(path, runtime, wheel, manifest, deadline)
    trees: dict[str, Any] = {}
    declarations = (
        (
            "sdk",
            SDK_FILES,
            "https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/" + SDK_REVISION + "/",
        ),
        (
            "base",
            BASE_FILES,
            "https://huggingface.co/" + BASE_MODEL + "/resolve/" + BASE_REVISION + "/",
        ),
        (
            "design",
            DESIGN_FILES,
            "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/"
            + DESIGN_REVISION
            + "/",
        ),
    )
    for name, pins, prefix in declarations:
        tree_root = root / name
        tree_root.mkdir(mode=0o700)
        for relative, size, _, sha in pins:
            _download(_parent(tree_root, relative), prefix + relative, size, sha, deadline)
        trees[name] = {"root": str(tree_root), "files": pins}
    trees["runtime"] = {
        "root": str(runtime),
        "files": tuple(
            (name, size, "sha256", sha) for name, (size, sha) in sorted(manifest.items())
        ),
    }
    # Actual complete file manifests, metadata and configs checked before selection.
    for name, tree in trees.items():
        verify_tree(
            NativeTree(tree["root"], tuple(tuple(p) for p in tree["files"])),
            deadline,
            distributions=name == "runtime",
            model_role=name if name in ("base", "design") else None,
        )
    result = {
        "interpreter": preflight["interpreter"],
        "seccomp": preflight["seccomp"],
        **trees,
        "recipe_sha256": recipe["reviewed_subject_sha256"],
    }
    raw = nr_encode(result)
    if len(raw) > NR_BLOB_MAX:
        raise ValueError("installed native manifest exceeds bound")
    nr_write(root / "installed.json", raw)
    return _descriptor(raw)


def _descriptor(raw: bytes) -> dict[str, Any]:
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _selected(root: Path, deadline: float) -> NativeSpeechSelection:
    nr_read(root / "selection.blob.done")
    value = nr_json(nr_read(root / "selection.blob"))
    selection = _selection(value)
    installed = nr_json(nr_read(root / "installed.json"))
    for name in ("interpreter", "seccomp", "runtime", "sdk", "base", "design"):
        if nr_encode(value[name]) != nr_encode(installed[name]):
            raise ValueError("selected native runtime differs from actual installation")
    verify_binary(selection.interpreter, deadline)
    verify_binary(selection.seccomp, deadline)
    return selection


def remote_operation(
    request: dict[str, Any], root: Path, deadline: float, binding: dict[str, Any]
) -> dict[str, Any]:
    op = request["operation"]
    if op == "native_setup":
        return _setup(root, deadline)
    if op == "native_select":
        nr_write(root / "selection-consumed", b"consumed")
        selection = _selected(root, deadline)
        if selection.hardware.operator_worker_id != binding["worker"]:
            raise ValueError("native selection belongs to another worker")
        return {"selection_sha256": selection.subject_sha256()}
    if op == "native_invoke":
        token = request["token"]
        nr_token(token)
        nr_read(root / "selection-consumed")
        nr_read(root / (token + ".input.done"))
        nr_write(root / (token + ".consumed"), b"consumed before child admission")
        selection = _selected(root, deadline)
        header, reference = unpack(nr_read(root / (token + ".input")))
        if set(header) != {"operation", "request", "seed"}:
            raise ValueError("invalid fixed native invocation")
        runtime = NativeSpeechRuntime(
            selection,
            allow_native=True,
            timeout_seconds=min(1200, deadline - time.monotonic()),
            seed=header["seed"],
            absolute_deadline=deadline,
        )
        if header["operation"] == "base":
            typed: QwenBaseRequest | OriginalVoiceDesign = QwenBaseRequest.from_dict(
                header["request"]
            )
            assert isinstance(typed, QwenBaseRequest)
            use = {"English": "base-en", "Chinese": "base-zh"}[typed.language]
            nr_write(root / ("used-" + use), b"consumed before child launch")
            payload = runtime.synthesize(typed, inspect_pcm16_wav(reference))
        elif header["operation"] == "design" and not reference:
            typed = OriginalVoiceDesign(**header["request"])
            nr_write(root / "used-design", b"consumed before child launch")
            payload = runtime.preview(typed)
        else:
            raise ValueError("unsupported fixed native operation")
        receipt = runtime.receipt_for(typed, payload)
        envelope = {
            "binding": binding,
            "attempt": token,
            "selection": selection.subject_sha256(),
            "request": digest(header["request"]),
            "operation": header["operation"],
            "reference": hashlib.sha256(reference).hexdigest(),
            "receipt": receipt,
        }
        result = pack(envelope, payload)
        if len(result) > NR_BLOB_MAX:
            raise ValueError("remote native result exceeds bound")
        nr_check(deadline)
        nr_write(root / (token + ".result"), result)
        nr_check(deadline)
        return _descriptor(result)
    if op == "native_fetch":
        token = request["token"]
        if token == "installed":
            path = root / "installed.json"
        else:
            nr_token(token)
            nr_read(root / (token + ".consumed"))
            path = root / (token + ".result")
        # Bounded random reads only, never relaunch and never future PID authority.
        info = nr_private(path)
        offset = request["offset"]
        if type(offset) is not int or not 0 <= offset < info.st_size <= NR_BLOB_MAX:
            raise ValueError("invalid native result range")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if os.fstat(fd).st_ino != info.st_ino:
                raise ValueError("remote result identity changed")
            data = os.pread(fd, NR_CHUNK, offset)
        finally:
            os.close(fd)
        nr_check(deadline)
        return {"offset": offset, "data": base64.b64encode(data).decode()}
    raise ValueError("unsupported fixed native operation")
