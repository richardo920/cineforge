"""Closed, stdlib-only remote speech declarations; metadata never grants execution."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

from cineforge.native_speech_pins import BASE_FILES, DESIGN_FILES, RUNTIME_DISTRIBUTIONS, SDK_FILES

NR_CHUNK = 32768
NR_BLOB_MAX = 16 * 1024**2
NR_POLICY = "native-fixed-wheel-unpack-v2-sympy-manpage"


def nr_encode(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def nr_digest(value: Any) -> str:
    return hashlib.sha256(nr_encode(value)).hexdigest()


def nr_hash(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
        raise ValueError("invalid remote speech identity")


def nr_token(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("invalid remote speech operation")


def nr_json(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate remote speech field")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError("invalid remote speech number")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    if type(value) is not dict:
        raise ValueError("invalid remote speech object")
    return value


def nr_recipe_subject(wheels: list[dict[str, Any]]) -> str:
    return nr_digest(
        {
            "wheels": wheels,
            "sdk": SDK_FILES,
            "base": BASE_FILES,
            "design": DESIGN_FILES,
            "policy": NR_POLICY,
        }
    )


def nr_recipe(value: Any) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != {"version", "wheels", "reviewed_subject_sha256", "review_evidence_sha256"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["wheels"]) is not list
        or len(value["wheels"]) != len(RUNTIME_DISTRIBUTIONS)
    ):
        raise ValueError("incomplete remote speech recipe")
    expected = dict(RUNTIME_DISTRIBUTIONS)
    names = []
    total = 0
    for wheel in value["wheels"]:
        if type(wheel) is not dict or set(wheel) != {
            "name",
            "version",
            "filename",
            "size_bytes",
            "sha256",
            "url",
        }:
            raise ValueError("invalid pinned speech wheel")
        name, filename = wheel["name"], wheel["filename"]
        if (
            type(name) is not str
            or name not in expected
            or wheel["version"] != expected[name]
            or type(filename) is not str
            or re.fullmatch(r"[A-Za-z0-9_.+\-]{1,240}\.whl", filename) is None
            or type(wheel["size_bytes"]) is not int
            or not 1 <= wheel["size_bytes"] <= 2 * 1024**3
        ):
            raise ValueError("unsupported speech wheel identity")
        parts = filename[:-4].split("-")
        if (
            len(parts) not in (5, 6)
            or parts[0].lower().replace("_", "-") != name
            or parts[1] != expected[name]
            or len(parts) == 6
            and not parts[2].isdigit()
        ):
            raise ValueError("unsupported Linux speech wheel layout")
        python, abi, platform = parts[-3:]
        tags = set(python.split("."))
        if (
            abi == "cp311"
            and tags != {"cp311"}
            or abi == "abi3"
            and not re.fullmatch(r"cp3(?:[2-9]|10|11)", python)
            or abi == "none"
            and not (tags <= {"py2", "py3", "cp311"} and tags & {"py3", "cp311"})
            or abi not in {"none", "abi3", "cp311"}
        ):
            raise ValueError("unsupported CPython311 speech wheel ABI")
        for tag in platform.split("."):
            if tag == "any" and abi == "none":
                continue
            if re.fullmatch(r"manylinux(?:1|2010|2014)_x86_64", tag):
                continue
            match = re.fullmatch(r"manylinux_2_([0-9]+)_x86_64", tag)
            if match is None or not 5 <= int(match[1]) <= 28:
                raise ValueError("unsupported fixed Linux speech wheel platform")
        nr_hash(wheel["sha256"])
        url = wheel["url"]
        if url is None:
            if name != "sox":
                raise ValueError("only the audited built sox wheel is uploaded")
        else:
            parsed = urlsplit(url) if type(url) is str else None
            if (
                parsed is None
                or parsed.scheme != "https"
                or parsed.netloc != "files.pythonhosted.org"
                or not parsed.path.startswith("/packages/")
                or parsed.path.rsplit("/", 1)[-1] != filename
                or parsed.query
                or parsed.fragment
                or "%" in parsed.path
                or any(part in {"", ".", ".."} for part in parsed.path.split("/")[1:])
            ):
                raise ValueError("unsupported pinned speech wheel origin")
        names.append(name)
        total += wheel["size_bytes"]
    if names != sorted(expected) or total > 8 * 1024**3:
        raise ValueError("incomplete or excessive speech runtime closure")
    nr_hash(value["review_evidence_sha256"])
    if value["reviewed_subject_sha256"] != nr_recipe_subject(value["wheels"]):
        raise ValueError("remote staging recipe review is stale")
    return value
