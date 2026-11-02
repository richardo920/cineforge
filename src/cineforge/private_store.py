"""Exclusive POSIX run directories for private bytes, never inside a Git worktree."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType

from cineforge.contracts import Artifact
from cineforge.providers import ExecutionResult


class ArtifactStoreError(ValueError):
    """A destination is unsafe, already used, or could not be written atomically."""


@dataclass(frozen=True)
class _Pending:
    device: int
    inode: int
    published: str | None = None


def _same_file(info: os.stat_result, pending: _Pending, links: int) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_dev == pending.device
            and info.st_ino == pending.inode and info.st_nlink == links)


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
        self._record_device: int | None = None
        self._pending: dict[str, _Pending | None] = {}

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

    def _cleanup_pending(self) -> bool:
        """Retry only this instance's exact captured aliases; uncertainty blocks writes."""
        failed = False
        for name, pending in list(self._pending.items()):
            try:
                try:
                    info = os.stat(name, dir_fd=self._directory, follow_symlinks=False)
                except FileNotFoundError:
                    del self._pending[name]
                    continue
                # If the initial fstat failed, retain the name without inventing an
                # identity. An operator must remove it; never adopt it from a path.
                if pending is None:
                    raise ArtifactStoreError("temporary identity is uncertain")
                links = 1 if pending.published is None else 2
                if not _same_file(info, pending, links):
                    raise ArtifactStoreError("temporary identity changed")
                if pending.published is not None:
                    target = os.stat(pending.published, dir_fd=self._directory,
                                     follow_symlinks=False)
                    if not _same_file(target, pending, links):
                        raise ArtifactStoreError("published identity changed")
                os.unlink(name, dir_fd=self._directory)
                del self._pending[name]
            except (OSError, ValueError):
                failed = True
        return not failed

    def _write(self, name: str, payload: bytes, *, record: bool = False) -> None:
        temporary = ".pending-" + uuid.uuid4().hex
        descriptor = -1
        failed = False
        try:
            if self._directory < 0 or not self._cleanup_pending():
                raise ArtifactStoreError("run directory or pending cleanup is unavailable")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=self._directory)
            self._pending[temporary] = None
            info = os.fstat(descriptor)
            identity = _Pending(info.st_dev, info.st_ino)
            self._pending[temporary] = identity
            if not _same_file(info, identity, 1):
                raise ArtifactStoreError("temporary identity is not exclusive")
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(descriptor)
            if not _same_file(os.stat(temporary, dir_fd=self._directory,
                                      follow_symlinks=False), identity, 1):
                raise ArtifactStoreError("temporary identity changed before publication")
            if record and self._record_inode is not None:
                info = os.stat(name, dir_fd=self._directory, follow_symlinks=False)
                if (self._record_device is None or not _same_file(
                        info, _Pending(self._record_device, self._record_inode), 1)):
                    raise ArtifactStoreError("run record was replaced externally")
                os.replace(temporary, name, src_dir_fd=self._directory, dst_dir_fd=self._directory)
                del self._pending[temporary]
            else:
                # Atomic create without overwriting any existing path, including a link.
                os.link(temporary, name, src_dir_fd=self._directory,
                        dst_dir_fd=self._directory, follow_symlinks=False)
                self._pending[temporary] = replace(identity, published=name)
            if record:
                # Publication succeeded: adopt the descriptor's identity before any
                # fallible unlink/stat/fsync, so a later checkpoint can recover.
                self._record_device, self._record_inode = identity.device, identity.inode
            if not self._cleanup_pending():
                raise ArtifactStoreError("temporary cleanup is incomplete")
            if not _same_file(os.stat(name, dir_fd=self._directory,
                                      follow_symlinks=False), identity, 1):
                raise ArtifactStoreError("published identity changed")
            os.fsync(self._directory)
        except (OSError, ValueError):
            failed = True
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    failed = True
            if self._directory >= 0 and not self._cleanup_pending():
                failed = True
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
