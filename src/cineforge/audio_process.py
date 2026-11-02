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

    def __init__(self, message: str, *, cleanup_uncertain: bool = False) -> None:
        super().__init__(message)
        self.cleanup_uncertain = cleanup_uncertain


def _finalize(process: subprocess.Popen[bytes] | None,
              selector: selectors.BaseSelector | None) -> bool:
    """Attempt every cleanup independently; OS refusal cannot imply confirmed cleanup."""
    uncertain = False
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException:
            uncertain = True
            # This can terminate the owned parent if group signaling was refused;
            # it does not confirm the state of any descendants.
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except BaseException:
                uncertain = True
        try:
            process.wait(timeout=2)
        except BaseException:
            uncertain = True
    if selector is not None:
        try:
            selector.close()
        except BaseException:
            uncertain = True
    if process is not None:
        for name in ("stdin", "stdout"):
            try:
                stream = getattr(process, name)
                if stream is not None:
                    stream.close()
            except BaseException:
                uncertain = True
    return uncertain


def run_audio_process(command: tuple[str, ...], *, payload: bytes, timeout_seconds: float,
                      max_output_bytes: int) -> bytes:
    """Bound pipes/time; attempt group termination, bounded reap and every descriptor close."""
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
    result = None
    failed = cancelled = False
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
        result = bytes(output)
    except KeyboardInterrupt:
        cancelled = True
    except BaseException:
        failed = True
    finally:
        cleanup_uncertain = _finalize(process, selector)
    # Raise only after leaving all private primary/cleanup exception contexts.
    if cleanup_uncertain:
        raise AudioProcessError("audio process cleanup incomplete or unconfirmed",
                                cleanup_uncertain=True) from None
    if cancelled:
        raise KeyboardInterrupt from None
    if failed or result is None:
        raise AudioProcessError("bounded audio process failed") from None
    return result
