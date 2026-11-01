"""Exclusive POSIX run directories for private bytes, never inside a Git worktree."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import replace
from pathlib import Path
from types import TracebackType

from cineforge.contracts import Artifact
from cineforge.providers import ExecutionResult


class ArtifactStoreError(ValueError):
    """A destination is unsafe, already used, or could not be written atomically."""


def _component(value: str) -> None:
    if (not value or value in (".", "..", ".git") or not value.isprintable()
            or any(c in value for c in "\\:%")):
        raise ArtifactStoreError("unsafe local path component")


def read_private_input(path: str | Path, *, max_bytes: int = 1024 * 1024) -> bytes:
    """Read one bounded regular file through no-follow ancestors; never log its path."""
    descriptor = -1
    failed = False
    payload = b""
    try:
        parts = Path(path).absolute().parts
        descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
        for index, part in enumerate(parts[1:]):
            _component(part)
            final = index == len(parts) - 2
            child = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                            | (0 if final else os.O_DIRECTORY), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > max_bytes:
            raise ArtifactStoreError("input must be a bounded regular file with one link")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ArtifactStoreError("input exceeds its size limit")
    except (OSError, TypeError, ValueError):
        failed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if failed:
        raise ArtifactStoreError("could not read safe bounded input")
    return payload


class PrivateRunDirectory:
    """Claim an empty, owner-only directory once; no resume or hostile-writer claim.

    A retained exclusive .lock makes crashes/nonterminal runs refuse a fresh submission.
    Same-host POSIX dir_fd/O_NOFOLLOW, hard links, atomic rename and fsync are required.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._directory = -1
        self._record_inode: int | None = None

    def __enter__(self) -> PrivateRunDirectory:
        descriptor = -1
        failed = False
        try:
            parts = self.path.absolute().parts
            descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
            for part in parts[1:]:
                _component(part)
                self._outside_git(descriptor)
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            self._outside_git(descriptor)
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or os.listdir(descriptor):
                raise ArtifactStoreError("run destination must be owned and empty")
            os.fchmod(descriptor, 0o700)
            lock = os.open(".lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                           0o600, dir_fd=descriptor)
            try:
                os.fsync(lock)
            finally:
                os.close(lock)
            os.fsync(descriptor)
            if set(os.listdir(descriptor)) != {".lock"}:
                raise ArtifactStoreError("run destination changed during claim")
            self._directory = descriptor
            descriptor = -1
        except (OSError, TypeError, ValueError):
            failed = True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if failed:
            raise ArtifactStoreError("could not claim a private empty destination outside Git")
        return self

    @staticmethod
    def _outside_git(descriptor: int) -> None:
        try:
            os.stat(".git", dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ArtifactStoreError("private run outputs must be outside Git worktrees")

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if self._directory >= 0:
            os.close(self._directory)
            self._directory = -1

    def _write(self, name: str, payload: bytes, *, record: bool = False) -> None:
        temporary = ".pending-" + uuid.uuid4().hex
        descriptor = -1
        failed = False
        try:
            if self._directory < 0:
                raise ArtifactStoreError("run directory is not open")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=self._directory)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(descriptor)
            if record and self._record_inode is not None:
                info = os.stat(name, dir_fd=self._directory, follow_symlinks=False)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_ino != self._record_inode):
                    raise ArtifactStoreError("run record was replaced externally")
                os.replace(temporary, name, src_dir_fd=self._directory, dst_dir_fd=self._directory)
            else:
                # Atomic create without overwriting any existing path, including a link.
                os.link(temporary, name, src_dir_fd=self._directory,
                        dst_dir_fd=self._directory, follow_symlinks=False)
                os.unlink(temporary, dir_fd=self._directory)
            if record:
                self._record_inode = os.stat(name, dir_fd=self._directory,
                                            follow_symlinks=False).st_ino
            os.fsync(self._directory)
        except (OSError, ValueError):
            failed = True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if self._directory >= 0:
                try:
                    os.unlink(temporary, dir_fd=self._directory)
                except FileNotFoundError:
                    pass
        if failed:
            raise ArtifactStoreError("private atomic write failed")

    def checkpoint(self, payload: bytes) -> None:
        self._write("run.json", payload, record=True)

    def put(self, result: ExecutionResult) -> Artifact:
        if (not isinstance(result, ExecutionResult) or not result.payload
                or hashlib.sha256(result.payload).hexdigest() != result.artifact.sha256
                or len(result.payload) != result.artifact.size_bytes):
            raise ArtifactStoreError("artifact bytes do not match their metadata")
        name = f"artifact-{result.artifact.sha256}.webm"
        self._write(name, result.payload)
        return replace(result.artifact, uri=name,
                       retention_policy="private local run; retained until operator removes directory")
