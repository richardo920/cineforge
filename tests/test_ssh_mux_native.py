"""Native OpenSSH mux fixtures use only local sockets and /dev/null identity."""

import os
import socket
import struct
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from test_cloud_budget import selection
from test_runpod import spec
from test_ssh_worker import Budget, Transport

from cineforge.ssh_worker import SSHWorker, WorkerError, run_process


def packet(*values):
    body = struct.pack(">" + "I" * len(values), *values)
    return struct.pack(">I", len(body)) + body


def read_packet(stream):
    def exact(size):
        data = b""
        while len(data) < size:
            block = stream.recv(size - len(data))
            if not block:
                raise EOFError
            data += block
        return data
    size = struct.unpack(">I", exact(4))[0]
    if not 4 <= size <= 4096:
        raise ValueError("invalid local mux fixture packet")
    return exact(size)


class Master:
    def __init__(self, path, mode="valid"):
        self.path, self.mode = path, mode
        self.requests, self.errors, self.forwarded = [], [], []
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.bind(str(path))
        self.socket.listen(2)
        self.socket.settimeout(.1)
        self.stop = Event()
        self.thread = Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while not self.stop.is_set():
            try:
                stream, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                with stream:
                    stream.settimeout(1)
                    hello = read_packet(stream)
                    if struct.unpack(">II", hello[:8]) != (1, 4):
                        raise ValueError("unexpected native mux hello")
                    if self.mode == "eof":
                        continue
                    if self.mode == "version":
                        stream.sendall(packet(1, 999))
                        continue
                    stream.sendall(packet(1, 4))
                    request = read_packet(stream)
                    kind, rid = struct.unpack(">II", request[:8])
                    self.requests.append(kind)
                    if kind == 0x10000004:  # MUX_C_ALIVE_CHECK
                        stream.sendall(packet(0x80000005, rid, os.getpid()))
                    elif kind == 0x10000006:  # MUX_C_OPEN_FWD
                        kind, rid, forwarding = struct.unpack(">III", request[:12])
                        offset = 12
                        size = struct.unpack(">I", request[offset:offset+4])[0]
                        host = request[offset+4:offset+4+size].decode()
                        offset += 4 + size
                        port = struct.unpack(">I", request[offset:offset+4])[0]
                        if forwarding != 1 or host != "127.0.0.1":
                            raise ValueError("unexpected native forward target")
                        listener = socket.socket()
                        listener.bind((host, port))
                        listener.listen(1)
                        self.forwarded.append(listener)
                        stream.sendall(packet(0x80000001, rid))
                    else:
                        raise ValueError("unexpected native mux request")
            except (OSError, EOFError, ValueError) as error:
                self.errors.append(type(error).__name__)

    def close(self):
        self.stop.set()
        self.socket.close()
        self.thread.join(2)
        for listener in self.forwarded:
            listener.close()


class NativeMuxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cf-native-mux-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        key = self.root / "synthetic-key"
        key.write_bytes(b"synthetic fixture, not private key")
        key.chmod(0o600)
        self.worker = SSHWorker(spec(), selection(), Budget(), Transport(), key,
                                self.root / "state", allow_live=True)
        self.worker.private_key = Path("/dev/null")
        self.worker.plan, self.worker._key = self.worker.budget.plan, "first"
        self.worker._host, self.worker._port = "127.0.0.1", 1
        self.worker._control_dir = self.root
        info = self.root.stat()
        self.worker._control_identity = (info.st_dev, info.st_ino)

    def test_native_hello_failure_cannot_fall_back_to_loopback_network(self):
        for mode in ("eof", "version"):
            for operation in ("check", "forward"):
                with self.subTest(mode=mode, operation=operation):
                    monitor = socket.socket()
                    monitor.bind(("127.0.0.1", 0))
                    monitor.listen(1)
                    monitor.settimeout(.15)
                    self.worker._port = monitor.getsockname()[1]
                    path = self.root / "c"
                    master = Master(path, mode)
                    connected, stop = Event(), Event()
                    def observe():
                        while not stop.is_set():
                            try:
                                stream, _ = monitor.accept()
                                connected.set()
                                stream.close()
                            except socket.timeout:
                                continue
                            except OSError:
                                return
                    observer = Thread(target=observe, daemon=True)
                    observer.start()
                    self.worker._control_socket_identity = None
                    try:
                        with self.assertRaises(WorkerError):
                            self.worker._mux(operation, time.monotonic()+2,
                                "127.0.0.1:12345:127.0.0.1:8188" if operation == "forward" else None)
                        self.assertEqual(master.requests, [])
                        self.assertFalse(connected.is_set(), "mux HELLO failure opened a TCP fallback")
                    finally:
                        stop.set()
                        monitor.close()
                        observer.join(2)
                        master.close()
                        path.unlink()

    def test_native_valid_hello_alive_and_forward_ack_still_succeed(self):
        path = self.root / "c"
        master = Master(path)
        try:
            self.worker._mux("check", time.monotonic()+2)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            self.worker._mux("forward", time.monotonic()+2,
                             f"127.0.0.1:{port}:127.0.0.1:8188")
            self.assertEqual(master.requests, [0x10000004, 0x10000006])
            self.assertEqual(master.errors, [])
            self.assertEqual(len(master.forwarded), 1)
        finally:
            master.close()
            path.unlink()

    def test_native_effective_mux_proxy_is_deny_not_later_shadowed_option(self):
        # Inspect actual production argv: OpenSSH uses the first configured value.
        path = self.root / "c"
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            with patch("cineforge.ssh_worker.run_process", return_value=b"") as invoke:
                self.worker._mux("check", time.monotonic()+2)
            command = invoke.call_args.args[0]
        path.unlink()
        self.assertNotIn("ProxyCommand=none", command)
        self.assertEqual(command.count("ProxyCommand=/usr/bin/false"), 1)
        index = command.index("-O")
        command = command[:index] + command[index+2:]
        value = run_process(command[:-1] + ["-G", command[-1]], timeout=2)
        proxy = [line for line in value.decode().splitlines() if line.startswith("proxycommand ")]
        self.assertEqual(proxy, ["proxycommand /usr/bin/false"])
        self.assertIn("ProxyCommand=none", self.worker._command())


if __name__ == "__main__":
    unittest.main()
