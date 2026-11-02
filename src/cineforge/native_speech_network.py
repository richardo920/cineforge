"""Linux socket-denial capability for the owned speech child, before SDK imports.

This limits outbound network sockets, not arbitrary hostile code or filesystem/IPC
access. AF_UNIX remains available for local runtime IPC. The operator pins the
actual installed libseccomp bytes; no library search or installation occurs here.
The system image and local IPC services remain trusted, including any UNIX proxy.
"""

from __future__ import annotations

import ctypes
import errno
import platform
import socket
import sys

from cineforge.native_speech_integrity import NativeBinary, NativeSpeechError, verify_binary
from cineforge.native_speech_pins import NETWORK_POLICY


class _Compare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def deny_network(library: NativeBinary, deadline: float) -> str:
    """Fail closed unless no_new_privs and the complete inherited filter load succeed."""
    failed = False
    result = None
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise NativeSpeechError("native speech requires Linux x86_64 socket isolation")
        verify_binary(library, deadline)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                               ctypes.c_ulong, ctypes.c_ulong]
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(39, 0, 0, 0, 0) != 1:
            raise NativeSpeechError("speech no-new-privileges policy unavailable")
        seccomp = ctypes.CDLL(library.path, use_errno=True)
        seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
        seccomp.seccomp_init.restype = ctypes.c_void_p
        seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
        seccomp.seccomp_release.restype = None
        seccomp.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
        seccomp.seccomp_attr_set.restype = ctypes.c_int
        seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
        seccomp.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_Compare)]
        seccomp.seccomp_rule_add_array.restype = ctypes.c_int
        seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
        seccomp.seccomp_load.restype = ctypes.c_int
        context = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
        if not context:
            raise NativeSpeechError("speech socket isolation unavailable")
        try:
            # libseccomp attributes: CTL_NNP=3, CTL_TSYNC=4. Apply before SDK threads.
            if (seccomp.seccomp_attr_set(context, 3, 1) != 0
                    or seccomp.seccomp_attr_set(context, 4, 1) != 0):
                raise NativeSpeechError("speech socket isolation attributes rejected")
            action = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO(EPERM)
            for name in (b"socket", b"socketpair"):
                number = seccomp.seccomp_syscall_resolve_name(name)
                compare = _Compare(0, 1, socket.AF_UNIX, 0)  # SCMP_CMP_NE
                if number < 0 or seccomp.seccomp_rule_add_array(
                        context, action, number, 1, ctypes.byref(compare)) != 0:
                    raise NativeSpeechError("speech socket isolation rule rejected")
            # Do not permit asynchronous socket creation or importing another task's
            # open socket descriptor as an alternative to the socket syscall.
            for name in (b"io_uring_setup", b"io_uring_enter", b"io_uring_register",
                         b"pidfd_getfd"):
                number = seccomp.seccomp_syscall_resolve_name(name)
                if number < 0 or seccomp.seccomp_rule_add_array(
                        context, action, number, 0, None) != 0:
                    raise NativeSpeechError("speech socket isolation rule rejected")
            if seccomp.seccomp_load(context) != 0:
                raise NativeSpeechError("speech socket isolation load rejected")
            result = NETWORK_POLICY
        finally:
            seccomp.seccomp_release(context)
    except BaseException:
        failed = True
    if failed or result is None:
        raise NativeSpeechError("required speech socket isolation unavailable") from None
    return result
