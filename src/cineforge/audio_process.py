"""Bounded local audio-tool process transport, without a shell or inherited settings."""

from __future__ import annotations

import math
import os
import selectors
import signal
import subprocess
import time


class AudioProcessError(ValueError):
    """Static diagnostics only; never retain argv, input, environment or tool output."""


def run_audio_process(command: tuple[str, ...], *, payload: bytes, timeout_seconds: float,
                      max_output_bytes: int) -> bytes:
    """Bound both pipes and wall time; always kill/reap the dedicated process group."""
    if (not isinstance(command, tuple) or not command
            or any(not isinstance(item, str) or not item or "\x00" in item for item in command)
            or not os.path.isabs(command[0]) or not isinstance(payload, bytes)
            or len(payload) > 16 * 1024 * 1024
            or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 30 or type(max_output_bytes) is not int
            or not 4096 <= max_output_bytes <= 1024 * 1024):
        raise AudioProcessError("invalid audio process bounds")
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    try:
        deadline = time.monotonic() + timeout_seconds
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True, bufsize=0,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
        assert process.stdin is not None and process.stdout is not None
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        written = 0
        output = bytearray()
        if payload:
            selector.register(process.stdin, selectors.EVENT_WRITE)
        else:
            process.stdin.close()
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AudioProcessError("audio process deadline exceeded")
            for key, event in selector.select(min(.05, remaining)):
                if event & selectors.EVENT_READ:
                    try:
                        block = os.read(key.fd, min(4096, max_output_bytes + 1 - len(output)))
                    except BlockingIOError:
                        continue
                    output.extend(block)
                    if len(output) > max_output_bytes:
                        raise AudioProcessError("audio process output bound exceeded")
                    if not block:
                        selector.unregister(key.fileobj)
                        process.stdout.close()
                else:
                    try:
                        written += os.write(key.fd, payload[written:written + 65536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        written = len(payload)
                    if written == len(payload):
                        selector.unregister(key.fileobj)
                        process.stdin.close()
        if process.returncode != 0:
            raise AudioProcessError("audio process failed")
        return bytes(output)
    except Exception:
        raise AudioProcessError("bounded audio process failed") from None
    finally:
        if process is not None:
            try:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except Exception:
                    raise AudioProcessError("audio process cleanup failed") from None
            finally:
                if selector is not None:
                    selector.close()
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
