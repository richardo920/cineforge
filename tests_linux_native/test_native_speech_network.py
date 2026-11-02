"""Required Linux CI proof using installed libseccomp and synthetic stdlib children.

No model, SDK, GPU, credentials or outbound connection is used. This suite is kept
outside default offline discovery because the actual kernel capability is required.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SpeechSocketIsolationTests(unittest.TestCase):
    def test_native_filter_denies_network_and_survives_exec(self):
        self.assertEqual(sys.platform, "linux", "Linux native evidence cannot be skipped")
        library = next((p.resolve() for p in (
            Path("/lib/x86_64-linux-gnu/libseccomp.so.2"),
            Path("/usr/lib/x86_64-linux-gnu/libseccomp.so.2")) if p.exists()), None)
        self.assertIsNotNone(library, "required libseccomp capability unavailable")
        with tempfile.TemporaryDirectory(prefix="cineforge-native-network-") as temporary:
            root = Path(temporary).resolve()
            staged = root / "libseccomp.so.2"
            shutil.copyfile(library, staged)
            staged.chmod(0o600)
            metadata = dict(path=str(staged), size_bytes=staged.stat().st_size,
                            sha256=hashlib.sha256(staged.read_bytes()).hexdigest())
            package = str(Path(__file__).resolve().parents[1] / "src")
            code = r'''
import ctypes,errno,json,os,socket,subprocess,sys,time
sys.path.insert(0,sys.argv[1])
from cineforge.native_speech_integrity import NativeBinary
from cineforge.native_speech_network import deny_network
cap=NativeBinary(**json.loads(sys.argv[2]))
try:os.fstat(int(sys.argv[3]));raise AssertionError('inherited descriptor')
except OSError:pass
policy=deny_network(cap,time.monotonic()+10)
for family in (socket.AF_INET,socket.AF_INET6,socket.AF_PACKET):
 try:socket.socket(family,socket.SOCK_STREAM);raise AssertionError('network socket')
 except OSError as error:assert error.errno==errno.EPERM
a,b=socket.socketpair(socket.AF_UNIX);a.close();b.close()
lib=ctypes.CDLL(cap.path);lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
for name in (b'io_uring_setup',b'io_uring_enter',b'io_uring_register',b'pidfd_getfd'):
 number=lib.seccomp_syscall_resolve_name(name)
 assert libc.syscall(number,0,0,0,0,0,0)==-1 and ctypes.get_errno()==errno.EPERM
probe="import errno,socket;\ntry:socket.socket();raise AssertionError('exec bypass')\nexcept OSError as e:assert e.errno==errno.EPERM\n"
child=subprocess.run([sys.executable,'-I','-S','-c',probe],capture_output=True,timeout=5)
assert child.returncode==0
shell=subprocess.run(['/bin/sh','-c','exec "$1" -I -S -c "$2"','fixture',sys.executable,probe],
 capture_output=True,timeout=5)
assert shell.returncode==0
print(json.dumps({'policy':policy,'network_families_denied':3,'unix_allowed':True,
 'exec_inherits':True,'shell_exec_inherits':True,'alternative_syscalls_denied':4,
 'inherited_descriptor_closed':True,'libseccomp_sha256':cap.sha256}))
'''
            with socket.socket() as inherited:
                os.set_inheritable(inherited.fileno(), True)
                result = subprocess.run([sys.executable, "-I", "-B", "-S", "-c", code,
                    package, json.dumps(metadata), str(inherited.fileno())], close_fds=True,
                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(result.stdout)
            self.assertEqual(evidence["network_families_denied"], 3)
            self.assertTrue(evidence["exec_inherits"])
            self.assertTrue(evidence["shell_exec_inherits"])
            self.assertEqual(evidence["libseccomp_sha256"], metadata["sha256"])
            print("NATIVE_SPEECH_NETWORK_EVIDENCE=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
