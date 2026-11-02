"""Fixed stdlib paired-transfer protocol for an already guarded S2V worker.

Consumed markers and incomplete parts are retained; there is no retry/overwrite or
cleanup-by-path inference. Whole owned Pod lifecycle remains the caller's concern.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from cineforge.paired_media import PNG_MAX_BYTES, SPEECH_MAX_BYTES, image_metadata, speech_metadata
from cineforge.worker_bootstrap import BootstrapError, encode, private_write

CHUNK_BYTES = 65536
MAX_PAIRS = 4


def _sha(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError


def _metadata(media: Any) -> dict[str, Any]:
    if not isinstance(media, dict) or set(media) != {"image", "audio"}:
        raise ValueError
    for kind, item in media.items():
        if not isinstance(item, dict):
            raise ValueError
        _sha(item.get("sha256"))
        common = {"sha256", "size_bytes", "basename"}
        expected = common | ({"width", "height", "channels", "bit_depth"} if kind == "image"
                             else {"sample_rate", "channels", "frames", "sample_format"})
        maximum = PNG_MAX_BYTES if kind == "image" else SPEECH_MAX_BYTES
        prefix, suffix = ("actor-", ".png") if kind == "image" else ("speech-", ".wav")
        if (set(item) != expected or type(item["size_bytes"]) is not int
                or not 0 < item["size_bytes"] <= maximum
                or item["basename"] != prefix + item["sha256"] + suffix):
            raise ValueError
        if kind == "image":
            if (type(item["width"]) is not int or not 16 <= item["width"] <= 4096
                    or type(item["height"]) is not int or not 16 <= item["height"] <= 4096
                    or type(item["channels"]) is not int or item["channels"] != 3
                    or type(item["bit_depth"]) is not int or item["bit_depth"] != 8):
                raise ValueError
        elif (type(item["sample_rate"]) is not int
                or item["sample_rate"] not in (16000, 24000, 44100, 48000)
                or type(item["channels"]) is not int or item["channels"] != 1
                or type(item["frames"]) is not int or not 0 < item["frames"]
                or item["frames"] * 16 > item["sample_rate"] * 77
                or item["size_bytes"] != 44 + item["frames"] * 2
                or item["sample_format"] != "pcm_s16le"):
            raise ValueError
    return media


def _read_at(directory: int, name: str, maximum: int,
             identity: list[int] | None = None) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > maximum
                or identity is not None and [info.st_dev, info.st_ino] != identity):
            raise ValueError
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(maximum + 1)
        after = os.fstat(fd)
        if (len(payload) != info.st_size or after.st_size != info.st_size
                or after.st_nlink != 1 or [after.st_dev, after.st_ino] != [info.st_dev, info.st_ino]):
            raise ValueError
        return payload
    finally:
        os.close(fd)


def _input_directory(root: Path) -> int:
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in ("ComfyUI", "input"):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
            info = os.fstat(fd)
            if (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022
                    or not stat.S_ISDIR(info.st_mode)):
                raise ValueError
        result, fd = fd, -1
        return result
    finally:
        if fd >= 0:
            os.close(fd)


def _pair_json(raw: bytes) -> dict[str, Any]:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    if not isinstance(value, dict):
        raise ValueError
    return value


def _check_payload(kind: str, payload: bytes, expected: dict[str, Any]) -> None:
    measured = image_metadata(payload) if kind == "image" else speech_metadata(payload)
    if measured != expected:
        raise ValueError


def _pair_key(scope: str, subject: str, media: dict[str, Any]) -> str:
    return hashlib.sha256(encode({"scope_sha256": scope, "subject_sha256": subject,
                                 "media": media})).hexdigest()


def paired_transfer(request: dict[str, Any], root: Path) -> dict[str, Any]:
    """Only dispatch after exact armed profile/lease/source and live guard checks."""
    result = None
    fd = root_fd = -1
    try:
        profile = request["profile"]
        scope, subject, stage_id = (profile["scope_sha256"], request["subject_sha256"],
                                    request["stage_id"])
        _sha(scope)
        _sha(subject)
        if type(stage_id) is not str or re.fullmatch(r"[0-9a-f]{32}", stage_id) is None:
            raise ValueError
        media = _metadata(request["media"])
        pair = _pair_key(scope, subject, media)
        record_name = "s2v-" + pair + ".json"
        fd, root_fd = _input_directory(root), os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        operation = request["operation"]
        if operation == "s2v_pair_begin":
            if len(tuple(root.glob("s2v-" + "[0-9a-f]" * 64 + ".json"))) >= MAX_PAIRS:
                raise ValueError
            state: dict[str, Any] = {"scope_sha256": scope, "subject_sha256": subject,
                "stage_id": stage_id, "media": media, "status": "receiving", "items": {}}
            # Create the permanent consumed marker before any parts/publications.
            private_write(root, record_name, encode(state))
            for kind in ("image", "audio"):
                expected = media[kind]
                try:
                    payload = _read_at(fd, expected["basename"], expected["size_bytes"])
                except FileNotFoundError:
                    payload = None
                if payload is not None:
                    _check_payload(kind, payload, expected)
                    state["items"][kind] = {"reused": True, "offset": expected["size_bytes"]}
                else:
                    part = "s2v-" + stage_id + "-" + kind + ".part"
                    private_write(root, part, b"")
                    info = os.stat(part, dir_fd=root_fd, follow_symlinks=False)
                    state["items"][kind] = {"reused": False, "offset": 0, "part": part,
                                            "identity": [info.st_dev, info.st_ino]}
            private_write(root, record_name, encode(state), replace=True)
            result = {"stage_id": stage_id, "pair_sha256": pair,
                      "reused": {kind: item["reused"] for kind, item in state["items"].items()}}
        else:
            state = _pair_json(_read_at(root_fd, record_name, 16384))
            if (state.get("scope_sha256") != scope or state.get("subject_sha256") != subject
                    or state.get("stage_id") != stage_id or state.get("media") != media
                    or state.get("status") != "receiving"):
                raise ValueError
            if operation == "s2v_pair_chunk":
                kind, offset = request["kind"], request["offset"]
                if kind not in ("image", "audio") or type(offset) is not int:
                    raise ValueError
                item = state["items"][kind]
                chunk = base64.b64decode(request["chunk"], validate=True)
                if (item["reused"] or offset != item["offset"] or not 0 < len(chunk) <= CHUNK_BYTES
                        or offset + len(chunk) > media[kind]["size_bytes"]):
                    raise ValueError
                part_fd = os.open(item["part"], os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW,
                                  dir_fd=root_fd)
                try:
                    info = os.fstat(part_fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                            or [info.st_dev, info.st_ino] != item["identity"] or info.st_size != offset):
                        raise ValueError
                    with os.fdopen(part_fd, "wb", closefd=False) as stream:
                        stream.write(chunk)
                        stream.flush()
                        os.fsync(part_fd)
                finally:
                    os.close(part_fd)
                item["offset"] += len(chunk)
                private_write(root, record_name, encode(state), replace=True)
                result = {"stage_id": stage_id, "pair_sha256": pair, "kind": kind,
                          "offset": item["offset"]}
            elif operation == "s2v_pair_finish":
                payloads = {}
                for kind in ("image", "audio"):
                    item, expected = state["items"][kind], media[kind]
                    if item["offset"] != expected["size_bytes"]:
                        raise ValueError
                    payload = (_read_at(fd, expected["basename"], expected["size_bytes"])
                        if item["reused"] else _read_at(root_fd, item["part"], expected["size_bytes"],
                                                        item["identity"]))
                    _check_payload(kind, payload, expected)
                    payloads[kind] = payload
                state["status"] = "publishing"
                private_write(root, record_name, encode(state), replace=True)
                for kind, payload in payloads.items():
                    if not state["items"][kind]["reused"]:
                        target_fd = os.open(media[kind]["basename"], os.O_WRONLY | os.O_CREAT
                            | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
                        try:
                            with os.fdopen(target_fd, "wb", closefd=False) as stream:
                                stream.write(payload)
                                stream.flush()
                                os.fsync(target_fd)
                        finally:
                            os.close(target_fd)
                    _check_payload(kind, _read_at(fd, media[kind]["basename"], len(payload)), media[kind])
                os.fsync(fd)
                state["status"] = "complete"
                private_write(root, record_name, encode(state), replace=True)
                result = {"stage_id": stage_id, "pair_sha256": pair, "scope_sha256": scope,
                          "subject_sha256": subject, "media": media,
                          "worker_id": request["plan"]["pod_id"]}
            else:
                raise ValueError
    except (Exception, KeyboardInterrupt):
        pass
    finally:
        for descriptor in (fd, root_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    result = None
    if result is None:
        raise BootstrapError("private paired staging incomplete or unconfirmed")
    return result
