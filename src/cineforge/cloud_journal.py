"""Single-host private cloud journals; no cloud credentials, network, or media bytes."""

from __future__ import annotations

import fcntl
import os
import stat
import uuid
from pathlib import Path
from types import TracebackType

from cineforge.private_store import PrivateRunDirectory, _component


class CloudJournalError(ValueError):
    """A private journal is unsafe, locked, corrupt, or cannot commit durably."""


class CloudJournal:
    """POSIX no-follow private directory, nonblocking flock, atomic fsynced checkpoint.

    Same-host processes coordinate through the retained lock. Hostile writers with the
    same account and distributed filesystems are not authenticated or supported.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._directory = -1
        self._lock = -1
        self._record_inode: int | None = None
        self._lock_inode: int | None = None
        self.initial: bytes | None = None

    @staticmethod
    def _regular(info: os.stat_result) -> None:
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
            raise CloudJournalError("private journal file is unsafe")

    def __enter__(self) -> CloudJournal:
        directory = -1
        lock = -1
        failed = False
        try:
            parts = self.path.absolute().parts
            directory = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
            for part in parts[1:]:
                _component(part)
                PrivateRunDirectory._outside_git(directory)
                try:
                    os.mkdir(part, 0o700, dir_fd=directory)
                    os.fsync(directory)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            PrivateRunDirectory._outside_git(directory)
            info = os.fstat(directory)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise CloudJournalError("private journal directory is unsafe")
            names = os.listdir(directory)
            if any(name not in {".lock", "budget.json"} and not name.startswith(".pending-")
                   for name in names):
                raise CloudJournalError("private journal contains unrelated files")
            if names and ".lock" not in names:
                raise CloudJournalError("private journal lock is missing")
            lock = os.open(".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                           0o600, dir_fd=directory)
            self._regular(os.fstat(lock))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_inode = os.fstat(lock).st_ino
            os.fsync(lock)
            os.fsync(directory)
            if "budget.json" in names:
                fd = os.open("budget.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
                try:
                    info = os.fstat(fd)
                    self._regular(info)
                    if info.st_size > 1024 * 1024:
                        raise CloudJournalError("private journal is oversized")
                    self.initial = os.read(fd, 1024 * 1024 + 1)
                    self._record_inode = info.st_ino
                finally:
                    os.close(fd)
            elif names:
                # An interrupted first write is not assumed to be an empty fresh budget.
                raise CloudJournalError("private journal has no committed budget")
            for name in names:
                if name.startswith(".pending-"):
                    self._regular(os.stat(name, dir_fd=directory, follow_symlinks=False))
                    os.unlink(name, dir_fd=directory)
            self._directory, self._lock = directory, lock
            directory = lock = -1
        except (OSError, ValueError, TypeError):
            failed = True
        finally:
            if lock >= 0:
                os.close(lock)
            if directory >= 0:
                os.close(directory)
        if failed:
            raise CloudJournalError("could not open exclusive private cloud journal")
        return self

    def checkpoint(self, payload: bytes) -> None:
        temporary = ".pending-" + uuid.uuid4().hex
        fd = -1
        failed = False
        try:
            if self._directory < 0 or not isinstance(payload, bytes) or len(payload) > 1024 * 1024:
                raise CloudJournalError("invalid private cloud checkpoint")
            info = os.stat(".lock", dir_fd=self._directory, follow_symlinks=False)
            self._regular(info)
            if info.st_ino != self._lock_inode:
                raise CloudJournalError("private cloud lock changed")
            if self._record_inode is not None:
                info = os.stat("budget.json", dir_fd=self._directory, follow_symlinks=False)
                self._regular(info)
                if info.st_ino != self._record_inode:
                    raise CloudJournalError("private cloud record changed")
            else:
                try:
                    os.stat("budget.json", dir_fd=self._directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise CloudJournalError("private cloud record unexpectedly exists")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=self._directory)
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(fd)
            os.replace(temporary, "budget.json", src_dir_fd=self._directory,
                       dst_dir_fd=self._directory)
            self._record_inode = os.stat("budget.json", dir_fd=self._directory,
                                        follow_symlinks=False).st_ino
            os.fsync(self._directory)
        except (OSError, ValueError, TypeError):
            failed = True
        finally:
            if fd >= 0:
                os.close(fd)
            if self._directory >= 0:
                try:
                    os.unlink(temporary, dir_fd=self._directory)
                except FileNotFoundError:
                    pass
        if failed:
            raise CloudJournalError("private cloud checkpoint failed")

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if self._lock >= 0:
            os.close(self._lock)
            self._lock = -1
        if self._directory >= 0:
            os.close(self._directory)
            self._directory = -1
