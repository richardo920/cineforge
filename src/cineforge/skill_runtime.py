"""Verified instruction loading and a closed, local tool broker.

Instruction text is data, never Python/shell. Only this broker's bounded file
operations are supported; network and arbitrary code execution always fail.
This is not an OS sandbox and does not constrain tools used outside the broker.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

from cineforge.contracts import ContractError, Shot, _digest
from cineforge.devices import GPUDevice
from cineforge.rights import RightsLedger, Use
from cineforge.skills import (
    CatalogEntry,
    Permissions,
    ShotSkills,
    SkillCatalog,
    SkillPin,
    bind_shot,
)


class SkillRuntimeError(ValueError):
    """Denied or invalid managed operation; diagnostics exclude supplied content."""


def _relative(value: object) -> tuple[str, ...]:
    try:
        if not isinstance(value, str):
            raise ValueError
        Permissions(read_paths=(value,))
        parts = PurePosixPath(value).parts
        if ".git" in parts:
            raise ValueError
        return parts
    except (ContractError, ValueError, TypeError):
        raise SkillRuntimeError("invalid managed relative path") from None


def _root(path: Path) -> int:
    """Open an existing owner-private root through no-follow ancestors, outside Git."""
    descriptor = -1
    try:
        parts = path.absolute().parts
        descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
        for part in parts[1:]:
            if part in (".", "..", ".git") or not part.isprintable():
                raise ValueError
            try:
                os.stat(".git", dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        try:
            os.stat(".git", dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError
        result, descriptor = descriptor, -1
        return result
    except (OSError, ValueError, TypeError):
        raise SkillRuntimeError("managed roots must be private directories outside Git") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parent(root: int, parts: tuple[str, ...]) -> int:
    descriptor = os.dup(root)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise


def _read(root: int, parts: tuple[str, ...], limit: int) -> bytes:
    parent = _parent(root, parts)
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_size > limit):
            raise ValueError
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError
        return data
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _create(root: int, parts: tuple[str, ...], data: bytes) -> None:
    parent = _parent(root, parts)
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
        os.fsync(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """Trusted operator ceiling; paths are exact files, not prefixes or globs."""

    permissions: Permissions = Permissions()
    max_source_bytes: int = 64 * 1024
    max_io_bytes: int = 64 * 1024
    max_calls: int = 32
    audience: str = "internal"
    commercial: bool = False

    def __post_init__(self) -> None:
        if type(self.permissions) is not Permissions:
            raise SkillRuntimeError("invalid operator capabilities")
        for value in (self.max_source_bytes, self.max_io_bytes):
            if type(value) is not int or not 1 <= value <= 1024 * 1024:
                raise SkillRuntimeError("invalid byte limit")
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 1000:
            raise SkillRuntimeError("invalid call limit")
        try:
            Use("reference", self.audience, self.commercial)
            for path in (*self.permissions.read_paths, *self.permissions.write_paths):
                _relative(path)
        except (ContractError, ValueError, TypeError):
            raise SkillRuntimeError("invalid operator context") from None


@dataclass(frozen=True, slots=True)
class InstalledSkill:
    pin: SkillPin
    shot_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        try:
            if type(self.pin) is not SkillPin:
                raise ValueError
            _digest(self.shot_sha256, "installed shot digest")
            _digest(self.receipt_sha256, "installed receipt digest")
        except (ContractError, ValueError, TypeError):
            raise SkillRuntimeError("invalid installed skill binding") from None

    @property
    def cache_key(self) -> str:
        raw = self.pin.to_json() + ":" + self.shot_sha256
        return hashlib.sha256(raw.encode()).hexdigest()


class SkillRuntime:
    """Caller supplies verified source bytes, never a fetch URL or executable callback.

    Existing roots must be owner-only, disjoint and outside Git. Failed installs
    retain an incomplete immutable slot and cannot be silently overwritten. Same-UID
    hostile writers and host tools outside this broker require separate OS isolation.
    """

    def __init__(self, catalog: SkillCatalog, ledger: RightsLedger, *,
                 install_root: str | Path, workspace: str | Path,
                 policy: RuntimePolicy = RuntimePolicy()) -> None:
        if (type(catalog) is not SkillCatalog or type(ledger) is not RightsLedger
                or type(policy) is not RuntimePolicy):
            raise SkillRuntimeError("invalid managed runtime context")
        self.catalog, self.ledger, self.policy = catalog, ledger, policy
        try:
            self._install_path, self._workspace_path = Path(install_root), Path(workspace)
        except (ValueError, TypeError):
            raise SkillRuntimeError("invalid managed roots") from None
        self._install = self._workspace = -1

    def __enter__(self) -> SkillRuntime:
        if self._install >= 0 or self._workspace >= 0:
            raise SkillRuntimeError("managed runtime is already open")
        try:
            self._install = _root(self._install_path)
            self._workspace = _root(self._workspace_path)
            a, b = self._install_path.resolve(), self._workspace_path.resolve()
            if a == b or a in b.parents or b in a.parents:
                raise SkillRuntimeError("managed roots must be disjoint")
        except (OSError, ValueError):
            self.__exit__(None, None, None)
            raise SkillRuntimeError("could not open isolated managed roots") from None
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        for descriptor in (self._install, self._workspace):
            if descriptor >= 0:
                os.close(descriptor)
        self._install = self._workspace = -1

    def _entry(self, pin: SkillPin, binding: ShotSkills, shot: Shot,
               device: GPUDevice | None, *, execute: bool) -> CatalogEntry:
        if self._install < 0 or self._workspace < 0:
            raise SkillRuntimeError("managed runtime is closed")
        try:
            if (type(binding) is not ShotSkills or pin not in binding.skills
                    or bind_shot(shot, binding.skills) != binding):
                raise ValueError
            entry = self.catalog.lookup(pin)
            readiness = self.catalog.reference_readiness(
                pin, shot=shot, ledger=self.ledger, device=device,
                use=Use("reference", self.policy.audience, self.policy.commercial))
            if not readiness.ready or entry.mode != "isolated_skill":
                raise ValueError
            for action in (("install", "execute") if execute else ("install",)):
                result = self.ledger.preflight_use(
                    entry.subject, entry.rights,
                    Use(action, self.policy.audience, self.policy.commercial))
                if not result.allowed:
                    raise ValueError
            return entry
        except (ContractError, ValueError, TypeError):
            raise SkillRuntimeError("skill is not approved for this managed use") from None

    def _receipt(self, entry: CatalogEntry, binding: ShotSkills) -> bytes:
        attribution = self.ledger.preflight_use(
            entry.subject, entry.rights,
            Use("install", self.policy.audience, self.policy.commercial)).attributions
        review = entry.review
        receipt = json.dumps({
            "schema_version": 1, "pin": entry.pin.to_dict(),
            "shot_sha256": binding.shot_sha256, "source": entry.source.to_dict(),
            "entrypoint": entry.entrypoint, "permissions": entry.permissions.to_dict(),
            "source_review": ({"declaration_sha256": review.declaration_sha256,
                               "reviewer": review.reviewer,
                               "evidence_sha256": review.evidence_sha256} if review else None),
            "attributions": [asdict(item) for item in attribution],
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        if len(receipt) > 1024 * 1024:
            raise SkillRuntimeError("installation receipt exceeds limit")
        return receipt

    def install(self, pin: SkillPin, *, binding: ShotSkills, shot: Shot, source: bytes,
                device: GPUDevice | None = None) -> InstalledSkill:
        """Install only the requested bound entry, retaining its exact notices and pins."""
        entry = self._entry(pin, binding, shot, device, execute=False)
        try:
            if (type(source) is not bytes or not source
                    or len(source) > self.policy.max_source_bytes
                    or hashlib.sha256(source).hexdigest() != entry.source.sha256):
                raise ValueError
            text = source.decode("utf-8")
            if not text.strip() or "\x00" in text:
                raise ValueError
            receipt = self._receipt(entry, binding)
            installed = InstalledSkill(pin, binding.shot_sha256, hashlib.sha256(receipt).hexdigest())
            os.mkdir(installed.cache_key, 0o700, dir_fd=self._install)
            _create(self._install, (installed.cache_key, "instructions"), source)
            _create(self._install, (installed.cache_key, "receipt"), receipt)
            return installed
        except (OSError, ValueError, TypeError, UnicodeError):
            raise SkillRuntimeError("could not install exact reviewed instruction bytes") from None

    def open_session(self, installed: InstalledSkill, *, binding: ShotSkills, shot: Shot,
                     device: GPUDevice | None = None) -> SkillSession:
        session = SkillSession(self, installed, binding, shot, device)
        session.instructions()  # Fail before returning an unauthorized or corrupt session.
        return session


class SkillSession:
    """Untrusted instructions plus an explicitly mediated local file tool surface."""

    def __init__(self, runtime: SkillRuntime, installed: InstalledSkill, binding: ShotSkills,
                 shot: Shot, device: GPUDevice | None) -> None:
        if type(installed) is not InstalledSkill:
            raise SkillRuntimeError("invalid installed skill")
        self._runtime, self._installed = runtime, installed
        self._binding, self._shot, self._device = binding, shot, device
        self._calls = 0

    def _checked(self) -> tuple[CatalogEntry, bytes]:
        try:
            entry = self._runtime._entry(self._installed.pin, self._binding, self._shot,
                                         self._device, execute=True)
            if self._installed.shot_sha256 != self._binding.shot_sha256:
                raise ValueError
            content = _read(self._runtime._install,
                            (self._installed.cache_key, "instructions"),
                            self._runtime.policy.max_source_bytes)
            receipt = _read(self._runtime._install, (self._installed.cache_key, "receipt"),
                            1024 * 1024)
            if (hashlib.sha256(content).hexdigest() != entry.source.sha256
                    or hashlib.sha256(receipt).hexdigest() != self._installed.receipt_sha256):
                raise ValueError
            if receipt != self._runtime._receipt(entry, self._binding):
                raise ValueError
            text = content.decode("utf-8")
            if not text.strip() or "\x00" in text:
                raise ValueError
            return entry, content
        except (OSError, ValueError, TypeError, KeyError):
            raise SkillRuntimeError("installed skill is unavailable or no longer approved") from None

    def instructions(self) -> str:
        """Preserve portable UTF-8 instruction text verbatim; never interpret code/frontmatter."""
        _, content = self._checked()
        try:
            return content.decode("utf-8")
        except UnicodeError:
            raise SkillRuntimeError("invalid installed instruction encoding") from None

    def invoke(self, tool: str, *, path: str | None = None,
               content: bytes | None = None, url: str | None = None) -> bytes:
        """Only read_file/write_file; no network, redirects, shell or general callbacks."""
        entry, _ = self._checked()
        self._calls += 1
        if self._calls > self._runtime.policy.max_calls:
            raise SkillRuntimeError("managed call limit exceeded")
        if type(tool) is not str or tool not in {"read_file", "write_file"} or url is not None:
            raise SkillRuntimeError("unsupported managed tool or network operation")
        policy = self._runtime.policy
        if tool not in entry.permissions.tools or tool not in policy.permissions.tools:
            raise SkillRuntimeError("managed tool is not allowed")
        parts = _relative(path)
        declared = (entry.permissions.read_paths if tool == "read_file"
                    else entry.permissions.write_paths)
        allowed = (policy.permissions.read_paths if tool == "read_file"
                   else policy.permissions.write_paths)
        if path not in declared or path not in allowed:
            raise SkillRuntimeError("managed path is not allowed")
        try:
            if tool == "read_file":
                if content is not None:
                    raise ValueError
                return _read(self._runtime._workspace, parts, policy.max_io_bytes)
            if type(content) is not bytes or len(content) > policy.max_io_bytes:
                raise ValueError
            _create(self._runtime._workspace, parts, content)
            return b""
        except (OSError, ValueError, TypeError):
            raise SkillRuntimeError("managed file operation denied or failed") from None
