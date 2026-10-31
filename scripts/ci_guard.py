"""Offline policy for tracked artifacts and explicitly reviewed dependency pins."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

MAX_BLOB_BYTES = 1024 * 1024
PRIVATE_PARTS = {"artifacts", "weights", "models", "private", "secrets", ".secrets"}
PRIVATE_SUFFIXES = {
    ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".mp4", ".mov", ".mkv",
    ".wav", ".mp3", ".flac", ".png", ".jpg", ".jpeg", ".webp", ".pem", ".key", ".zip",
}
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_blob(path: str, size: int, mode: str) -> list[str]:
    """Check metadata before reading content; reject external submodule/symlink entries."""
    errors = []
    parsed = PurePosixPath(path)
    if mode not in {"100644", "100755"}:
        errors.append(f"{path}: only regular files allowed (mode {mode})")
    if size > MAX_BLOB_BYTES:
        errors.append(f"{path}: exceeds {MAX_BLOB_BYTES} byte Git blob limit")
    if (set(part.lower() for part in parsed.parts) & PRIVATE_PARTS
            or parsed.suffix.lower() in PRIVATE_SUFFIXES or "credentials" in parsed.name.lower()
            or (parsed.name.startswith(".env") and parsed.name != ".env.example")):
        errors.append(f"{path}: private/media/model/credential artifact prohibited")
    return errors


def tracked_artifacts(root: Path) -> list[str]:
    """Inspect the Git index, including staged blobs hidden by working-tree edits."""
    listing = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    errors = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="replace")
        if stage != "0":
            errors.append(f"{path}: unresolved index entry")
            continue
        size = int(subprocess.check_output(["git", "cat-file", "-s", oid], cwd=root))
        errors.extend(check_blob(path, size, mode))
    return errors


def pins(lines: list[str], scope: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if not isinstance(line, str):
            raise ValueError(f"{scope}: dependency must be a string")
        match = PIN.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"{scope}: exact name==version pins only: {line!r}")
        name, version = match.groups()
        name = normalize(name)
        if name in result:
            raise ValueError(f"{scope}: duplicate dependency {name}")
        result[name] = version
    return result


def dependencies(project: dict[str, Any], ci_lines: list[str],
                 policy: dict[str, Any]) -> list[str]:
    try:
        if project["project"].get("dynamic"):
            raise ValueError("dynamic package metadata is not reviewed")
        if project["project"].get("optional-dependencies"):
            raise ValueError("optional dependencies require policy support and review")
        scopes = {
            "runtime": pins(project["project"].get("dependencies", []), "runtime"),
            "build": pins(project["build-system"]["requires"], "build"),
            "ci": pins([line for line in ci_lines if line.strip()
                        and not line.lstrip().startswith("#")], "ci"),
        }
        return [f"{scope}: dependencies differ from reviewed exact pins"
                for scope, values in scopes.items() if values != policy[scope]]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"dependency policy: {exc}"]


def secret_scan(root: Path) -> list[str]:
    """Scan staged regular blobs offline without exposing secret values in output."""
    errors = tracked_artifacts(root)
    if errors:
        return errors
    # Materialize the index, not the working tree; artifact checks reject links first.
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(["git", "checkout-index", "--all", f"--prefix={directory}/"],
                       cwd=root, check=True, capture_output=True)
        result = subprocess.run([sys.executable, "-m", "detect_secrets", "scan", "--no-verify",
                                 "--all-files", "."], cwd=directory,
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
    return [f"{path}: potential secret detected" for path in report["results"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["artifacts", "dependencies", "secrets"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        if args.check == "artifacts":
            errors = tracked_artifacts(args.root)
        elif args.check == "dependencies":
            project = tomllib.loads((args.root / "pyproject.toml").read_text())
            policy = json.loads((args.root / "config/dependency-policy.json").read_text())
            errors = dependencies(project, (args.root / "requirements-ci.txt").read_text()
                                  .splitlines(), policy)
        else:
            errors = secret_scan(args.root)
    except (OSError, ValueError, subprocess.SubprocessError):
        print(f"{args.check}: guard could not complete", file=sys.stderr)
        return 1
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"{args.check}: PASS")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
