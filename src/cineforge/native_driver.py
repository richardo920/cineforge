"""Fixed NVIDIA file-mount qualification; host administrators remain trusted.

Read-only container mounts do not prevent host changes. Every consumer remeasures
identity; this policy neither authenticates a vendor nor admits model execution.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from typing import Any

DRIVER_POLICY = "nvidia-mapped-readonly-file-v1"
DRIVER_ALIASES = {
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1": "cuda_driver",
    "/usr/lib64/libcuda.so.1": "cuda_driver",
    "/usr/local/nvidia/lib64/libcuda.so.1": "cuda_driver",
    "/usr/bin/nvidia-smi": "nvidia_smi",
    "/usr/local/nvidia/bin/nvidia-smi": "nvidia_smi",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1": "nvml",
    "/usr/lib64/libnvidia-ml.so.1": "nvml",
    "/usr/local/nvidia/lib64/libnvidia-ml.so.1": "nvml",
}
DRIVER_ROOTS = ("/usr/bin/", "/usr/local/nvidia/bin/", "/usr/lib/x86_64-linux-gnu/",
                "/usr/lib64/", "/usr/local/nvidia/lib64/")


def driver_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise ValueError("native file deadline exhausted")


def _proc(path: str, limit: int, deadline: float) -> str:
    driver_check(deadline)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        raw = bytearray()
        while block := os.read(fd, min(65536, limit+1-len(raw))):
            driver_check(deadline)
            raw.extend(block)
            if len(raw) > limit:
                raise ValueError("native mount evidence exceeds bound")
        return raw.decode("ascii", "strict")
    finally:
        os.close(fd)


def _mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def driver_snapshot(alias: str, path: str, fd: int, info: os.stat_result,
                    deadline: float, binding_sha256: str | None) -> dict[str, Any]:
    driver_check(deadline)
    if (alias not in DRIVER_ALIASES or not path.startswith(DRIVER_ROOTS)
            or os.getuid() != 0 or info.st_uid != 65534 or not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o022 or info.st_nlink != 1
            or type(binding_sha256) is not str
            or re.fullmatch("[0-9a-f]{64}", binding_sha256) is None or binding_sha256 == "0"*64
            or not os.fstatvfs(fd).f_flag & os.ST_RDONLY):
        raise ValueError("NVIDIA body is not a qualified protected mount")
    mapping = []
    for line in _proc("/proc/self/uid_map", 4096, deadline).splitlines():
        parts = line.split()
        if len(parts) != 3 or any(not p.isascii() or not p.isdigit() for p in parts):
            raise ValueError("invalid namespace mapping")
        row = list(map(int, parts))
        if any(not 0 <= v < 2**32 for v in row) or row[2] == 0:
            raise ValueError("invalid namespace range")
        mapping.append(row)
    if (not 1 <= len(mapping) <= 64 or sum(row[0] == 0 for row in mapping) != 1
            or not any(row[0] == 0 and row[1] > 0 for row in mapping)
            or any(row[1] == 0 for row in mapping)):
        raise ValueError("NVIDIA qualification requires mapped root without host root")
    user_ns = os.readlink("/proc/self/ns/user")
    mount_ns = os.readlink("/proc/self/ns/mnt")
    if re.fullmatch(r"user:\[\d+\]", user_ns) is None or re.fullmatch(r"mnt:\[\d+\]", mount_ns) is None:
        raise ValueError("invalid kernel namespace identity")
    mounts = []
    for line in _proc("/proc/self/mountinfo", 1024**2, deadline).splitlines():
        driver_check(deadline)
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            raise ValueError("invalid kernel mount evidence")
        sep = fields.index("-")
        if sep < 6 or sep+3 >= len(fields):
            raise ValueError("incomplete mount evidence")
        point = _mount_path(fields[4])
        if path == point:
            if (not fields[0].isdigit() or not fields[1].isdigit()
                    or re.fullmatch(r"\d+:\d+", fields[2]) is None
                    or fields[sep+1] != "ext4" or "ro" not in fields[5].split(",")):
                raise ValueError("NVIDIA mount is not a read-only ext4 file mount")
            mounts.append([int(fields[0]), int(fields[1]), fields[2], point, fields[sep+1], fields[5]])
    if len(mounts) != 1:
        raise ValueError("missing or ambiguous individual NVIDIA mount")
    descriptor_mount = [line.split()[1] for line in _proc("/proc/self/fdinfo/"+str(fd), 8192, deadline).splitlines()
                        if line.startswith("mnt_id:")]
    if (descriptor_mount != [str(mounts[0][0])]
            or mounts[0][2] != str(os.major(info.st_dev))+":"+str(os.minor(info.st_dev))):
        raise ValueError("NVIDIA descriptor mount differs from path mount")
    if driver_file_identity(os.fstat(fd)) != driver_file_identity(info):
        raise ValueError("NVIDIA descriptor changed")
    driver_check(deadline)
    return {"policy": DRIVER_POLICY, "role": DRIVER_ALIASES[alias], "alias": alias,
            "binding_sha256": binding_sha256, "user_namespace": user_ns,
            "mount_namespace": mount_ns, "uid_map": mapping, "mount": mounts[0],
            "identity": list(driver_file_identity(info))}


def driver_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_native_file(alias: str, deadline: float, roots: tuple[str, ...], *,
                     binding_sha256: str | None = None) -> dict[str, Any]:
    # The only caller enumerates fixed aliases. Resolve protected symlinks with
    # bounded descriptor traversal, then hash the actual no-follow owned body.
    path = alias
    links: list[tuple[str, tuple[int, ...], str]] = []
    ancestors: list[tuple[str, tuple[int, ...]]] = []
    for _ in range(17):
        driver_check(deadline)
        if len(path) > 4096 or not path.startswith("/") or "\x00" in path:
            raise ValueError("invalid fixed native capability path")
        parts = path.split("/")[1:]
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        restart = False
        try:
            root_info = os.fstat(descriptor)
            if root_info.st_uid not in (0, os.getuid()) or root_info.st_mode & 0o022:
                raise ValueError("unprotected native filesystem root")
            ancestors.append(("/", driver_file_identity(root_info)))
            for index, name in enumerate(parts):
                driver_check(deadline)
                if name in ("", ".", ".."):
                    raise ValueError("invalid fixed native capability component")
                before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (before.st_uid not in (0, os.getuid())
                        and not (index == len(parts)-1 and stat.S_ISREG(before.st_mode)
                                 and before.st_uid == 65534 and binding_sha256 is not None
                                 and alias in DRIVER_ALIASES)):
                    raise ValueError("unowned fixed native capability")
                if stat.S_ISLNK(before.st_mode):
                    target = os.readlink(name, dir_fd=descriptor)
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if driver_file_identity(before) != driver_file_identity(after):
                        raise ValueError("fixed native capability link changed")
                    links.append((os.path.join("/", *parts[:index+1]), driver_file_identity(before), target))
                    path = os.path.normpath(os.path.join("/", *parts[:index], target, *parts[index+1:]))
                    restart = True
                    break
                if before.st_mode & 0o022 and not (before.st_uid == 0 and before.st_mode & stat.S_ISVTX):
                    raise ValueError("unprotected fixed native capability ancestor")
                if index < len(parts)-1:
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    if (os.fstat(child).st_dev, os.fstat(child).st_ino) != (before.st_dev, before.st_ino):
                        os.close(child)
                        raise ValueError("fixed native capability ancestor changed")
                    ancestors.append((os.path.join("/", *parts[:index+1]), driver_file_identity(before)))
                    os.close(descriptor)
                    descriptor = child
                    continue
                if (not path.startswith(roots) or not stat.S_ISREG(before.st_mode)
                        or before.st_nlink != 1 or before.st_mode & 0o022
                        or not 0 < before.st_size <= 1024**3):
                    raise ValueError("unprotected fixed native capability body")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
                try:
                    if driver_file_identity(before) != driver_file_identity(os.fstat(fd)):
                        raise ValueError("fixed native capability changed before read")
                    witness = (driver_snapshot(alias, path, fd, before, deadline, binding_sha256)
                        if before.st_uid not in (0, os.getuid()) else None)
                    sha, count = hashlib.sha256(), 0
                    while block := os.read(fd, 1024**2):
                        driver_check(deadline)
                        count += len(block)
                        if count > before.st_size:
                            raise ValueError("fixed native capability grew")
                        sha.update(block)
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if (count != before.st_size or driver_file_identity(before) != driver_file_identity(after)
                            or driver_file_identity(before) != driver_file_identity(os.fstat(fd))):
                        raise ValueError("fixed native capability changed during read")
                    if witness is not None and driver_snapshot(alias, path, fd, after, deadline, binding_sha256) != witness:
                        raise ValueError("NVIDIA mount changed during read")
                finally:
                    os.close(fd)
                driver_check(deadline)
                for ancestor, identity in ancestors:
                    if driver_file_identity(os.lstat(ancestor)) != identity:
                        raise ValueError("native ancestor changed during read")
                    driver_check(deadline)
                for link, identity, target in links:
                    if driver_file_identity(os.lstat(link)) != identity or os.readlink(link) != target:
                        raise ValueError("fixed native capability alias changed during read")
                    driver_check(deadline)
                result: dict[str, Any] = {"path": path, "size_bytes": count, "sha256": sha.hexdigest()}
                if witness is not None:
                    witness["links"] = [[link, list(identity), target] for link, identity, target in links]
                    witness["ancestors"] = [[ancestor, list(identity)] for ancestor, identity in ancestors]
                    result["driver_mount"] = witness
                return result
        finally:
            os.close(descriptor)
        if not restart:
            break
    raise ValueError("fixed native capability link bound exceeded")



def linux_device_parts(value: int) -> tuple[int, int]:
    """Decode the Linux uint64 dev_t wire value, independently of this host ABI."""
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("invalid Linux device identity")
    major = ((value >> 8) & 0xfff) | ((value >> 32) & 0xfffff000)
    minor = (value & 0xff) | ((value >> 12) & 0xffffff00)
    return major, minor


def validate_driver_receipt(receipt: Any, *, path: str, size_bytes: int, sha256: str,
                            binding_sha256: str, role: str | None = None,
                            alias: str | None = None) -> None:
    """Validate declarations only; actual mount qualification is always remeasured."""
    keys = {"policy", "role", "alias", "binding_sha256", "user_namespace", "mount_namespace",
            "uid_map", "mount", "identity", "links", "ancestors"}
    if (type(receipt) is not dict or set(receipt) != keys
            or receipt["policy"] != DRIVER_POLICY or receipt["alias"] not in DRIVER_ALIASES
            or receipt["role"] != DRIVER_ALIASES[receipt["alias"]]
            or role is not None and receipt["role"] != role
            or alias is not None and receipt["alias"] != alias
            or type(binding_sha256) is not str or re.fullmatch("[0-9a-f]{64}", binding_sha256) is None
            or binding_sha256 == "0"*64 or receipt["binding_sha256"] != binding_sha256
            or type(path) is not str or not path.startswith(DRIVER_ROOTS)
            or len(path) > 4096 or any(p in ("", ".", "..") for p in path.split("/")[1:])
            or type(sha256) is not str or re.fullmatch("[0-9a-f]{64}", sha256) is None
            or sha256 == "0"*64 or type(size_bytes) is not int or not 0 < size_bytes <= 1024**3):
        raise ValueError("invalid fixed NVIDIA receipt")
    identity = receipt["identity"]
    if (type(identity) is not list or len(identity) != 8
            or any(type(v) is not int or not 0 <= v < 2**128 for v in identity)
            or identity[2] != 65534 or not stat.S_ISREG(identity[3])
            or identity[3] & 0o022 or identity[4] != 1 or identity[5] != size_bytes):
        raise ValueError("invalid mounted NVIDIA body identity")
    for key, pattern in (("user_namespace", r"user:\[\d+\]"), ("mount_namespace", r"mnt:\[\d+\]")):
        if type(receipt[key]) is not str or re.fullmatch(pattern, receipt[key]) is None:
            raise ValueError("invalid NVIDIA namespace identity")
    mapping = receipt["uid_map"]
    if (type(mapping) is not list or not 1 <= len(mapping) <= 64
            or any(type(row) is not list or len(row) != 3
                   or any(type(v) is not int or not 0 <= v < 2**32 for v in row)
                   or row[2] == 0 or row[0]+row[2] > 2**32 or row[1]+row[2] > 2**32 for row in mapping)
            or sum(row[0] == 0 for row in mapping) != 1
            or not any(row[0] == 0 and row[1] > 0 for row in mapping)
            or any(row[1] == 0 for row in mapping)):
        raise ValueError("invalid mapped NVIDIA namespace")
    for offset, row in enumerate(mapping):
        for other in mapping[offset+1:]:
            if any(max(row[i], other[i]) < min(row[i]+row[2], other[i]+other[2]) for i in (0, 1)):
                raise ValueError("overlapping namespace ranges")
    mount = receipt["mount"]
    if (type(mount) is not list or len(mount) != 6
            or any(type(mount[i]) is not int or not 0 < mount[i] < 2**64 for i in (0, 1))
            or mount[2] != ":".join(map(str, linux_device_parts(identity[0])))
            or mount[3] != path or mount[4] != "ext4" or type(mount[5]) is not str
            or len(mount[5]) > 4096 or "ro" not in mount[5].split(",") or "rw" in mount[5].split(",")):
        raise ValueError("invalid individual NVIDIA mount")
    ancestors = receipt["ancestors"]
    if type(ancestors) is not list or not 1 <= len(ancestors) <= 1024:
        raise ValueError("invalid protected NVIDIA ancestors")
    for row in ancestors:
        if (type(row) is not list or len(row) != 2 or type(row[0]) is not str
                or not row[0].startswith("/") or len(row[0]) > 4096
                or row[0] != "/" and any(p in ("", ".", "..") for p in row[0].split("/")[1:])
                or type(row[1]) is not list or len(row[1]) != 8
                or any(type(v) is not int or not 0 <= v < 2**128 for v in row[1])
                or row[1][2] != 0 or not stat.S_ISDIR(row[1][3])
                or row[1][3] & 0o022 and not row[1][3] & stat.S_ISVTX):
            raise ValueError("unprotected NVIDIA ancestor declaration")
    links = receipt["links"]
    if type(links) is not list or len(links) > 16:
        raise ValueError("invalid NVIDIA alias chain")
    for row in links:
        if (type(row) is not list or len(row) != 3 or type(row[0]) is not str
                or not row[0].startswith("/") or len(row[0]) > 4096
                or any(p in ("", ".", "..") for p in row[0].split("/")[1:])
                or type(row[1]) is not list or len(row[1]) != 8
                or any(type(v) is not int or not 0 <= v < 2**128 for v in row[1])
                or row[1][2] != 0 or not stat.S_ISLNK(row[1][3])
                or type(row[2]) is not str or not 0 < len(row[2]) <= 4096 or "\x00" in row[2]):
            raise ValueError("unprotected NVIDIA alias")


def verify_driver_binary(binary: dict[str, Any], deadline: float, binding_sha256: str) -> None:
    """Reopen a closed alias; declarations cannot substitute live namespace/mount facts."""
    if type(binary) is not dict or set(binary) != {"path", "size_bytes", "sha256", "driver_mount"}:
        raise ValueError("invalid NVIDIA binary declaration")
    validate_driver_receipt(binary["driver_mount"], path=binary["path"],
        size_bytes=binary["size_bytes"], sha256=binary["sha256"], binding_sha256=binding_sha256)
    actual = read_native_file(binary["driver_mount"]["alias"], deadline, DRIVER_ROOTS,
                              binding_sha256=binding_sha256)
    if actual != binary:
        raise ValueError("mounted NVIDIA binary identity changed")
