"""Bounded binary pipes for one owned native speech child; no shell or retries."""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import struct
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from cineforge.json_codec import strict_loads
from cineforge.native_speech_integrity import NativeSpeechError

MAX_HEADER = 16 * 1024**2
MAX_INPUT = MAX_HEADER + 16 * 1024**2 + 4
MAX_OUTPUT = 17 * 1024**2


@dataclass(frozen=True, slots=True)
class OwnedChildResult:
    payload: bytes = field(repr=False)
    process_id: int

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or type(self.process_id) is not int or self.process_id <= 0:
            raise NativeSpeechError("invalid owned speech child result")


def pack(header: dict[str, Any], payload: bytes) -> bytes:
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) > MAX_HEADER or len(payload) > 16 * 1024**2:
        raise NativeSpeechError("native speech protocol bound exceeded")
    return struct.pack(">I", len(encoded)) + encoded + payload


def unpack(raw: bytes) -> tuple[dict[str, Any], bytes]:
    if type(raw) is not bytes or not 4 <= len(raw) <= MAX_INPUT:
        raise NativeSpeechError("invalid native speech protocol frame")
    count = struct.unpack(">I", raw[:4])[0]
    if not 1 <= count <= MAX_HEADER or 4 + count > len(raw):
        raise NativeSpeechError("invalid native speech protocol header")
    header = strict_loads(raw[4:4 + count])
    if not isinstance(header, dict):
        raise NativeSpeechError("invalid native speech protocol header")
    return header, raw[4 + count:]


def _finish(process: subprocess.Popen[bytes] | None, selector: selectors.BaseSelector | None,
            deadline: float) -> bool:
    uncertain = False
    if process is not None:
        try:
            # The parent is not polled/reaped before this signal, retaining its PID
            # until owned-group cleanup. This also attempts to stop descendants.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException:
            uncertain = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except BaseException:
                uncertain = True
        try:
            remaining = max(0.0, min(2.0, deadline - time.monotonic()))
            process.wait(timeout=remaining)
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


def run_child(command: tuple[str, ...], *, payload: bytes, env: dict[str, str],
              cwd: str, deadline: float) -> OwnedChildResult:
    """The original absolute deadline includes a two-second cleanup reserve.

    A completed frame alone is insufficient: require parent exit0/reap and no known
    cleanup-call failures. Group signaling does not independently measure every
    descendant's terminal state. OS refusal is explicit uncertainty, never success.
    """
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    result = None
    failed = cancelled = False
    if type(payload) is not bytes or len(payload) > MAX_INPUT:
        raise NativeSpeechError("native speech input bound exceeded")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise NativeSpeechError("invalid native speech deadline")
    try:
        work_deadline = deadline - 2
        if time.monotonic() >= work_deadline:
            raise NativeSpeechError("native speech deadline exhausted")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True, bufsize=0,
            shell=False, env=env, cwd=cwd)
        if process.stdin is None or process.stdout is None:
            raise NativeSpeechError("native speech pipe unavailable")
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stdin, selectors.EVENT_WRITE)
        output = bytearray()
        written = 0
        while selector.get_map():
            remaining = work_deadline - time.monotonic()
            if remaining <= 0:
                raise NativeSpeechError("native speech deadline exceeded")
            for key, event in selector.select(min(.05, remaining)):
                if event & selectors.EVENT_READ:
                    try:
                        chunk = os.read(key.fd, min(65536, MAX_OUTPUT + 1 - len(output)))
                    except BlockingIOError:
                        continue
                    output.extend(chunk)
                    if len(output) > MAX_OUTPUT:
                        raise NativeSpeechError("native speech output bound exceeded")
                    if not chunk:
                        selector.unregister(key.fileobj)
                        process.stdout.close()
                else:
                    try:
                        written += os.write(key.fd, payload[written:written + 65536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        raise NativeSpeechError("native speech input was not consumed") from None
                    if written == len(payload):
                        selector.unregister(key.fileobj)
                        process.stdin.close()
        if time.monotonic() >= work_deadline:
            raise NativeSpeechError("native speech deadline exceeded")
        result = bytes(output)
    except KeyboardInterrupt:
        cancelled = True
    except BaseException:
        failed = True
    finally:
        uncertain = _finish(process, selector, deadline)
    if uncertain:
        raise NativeSpeechError("native speech cleanup incomplete or unconfirmed",
                                cleanup_uncertain=True,
                                owned_process_id=None if process is None else process.pid) from None
    if cancelled:
        raise KeyboardInterrupt from None
    if failed or result is None or process is None or process.returncode != 0:
        raise NativeSpeechError("bounded native speech child failed",
                                owned_process_id=None if process is None else process.pid) from None
    owned = OwnedChildResult(result, process.pid)
    # Cleanup and scheduler delays consume the original cap too. A known-clean
    # late result is a failed attempt, not invented cleanup uncertainty.
    if time.monotonic() >= deadline:
        raise NativeSpeechError("native speech deadline exceeded", owned_process_id=process.pid)
    return owned
