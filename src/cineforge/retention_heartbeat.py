"""Detached SSH heartbeat: no journal, budget session, account API or release authority."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def feed_until(expiry: float, beat: Callable[[float], None], *,
               clock: Callable[[], float] = time.time,
               monotonic: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep) -> None:
    """Beat immediately, then every ten seconds; failure/rollback/expiry ends authority."""
    wall, mono = clock(), monotonic()
    if not math.isfinite(expiry) or not math.isfinite(wall) or not math.isfinite(mono):
        raise ValueError("finite heartbeat expiry required")
    cap = mono + max(0., expiry - wall)
    previous = wall
    while True:
        now, current = clock(), monotonic()
        if (not math.isfinite(now) or not math.isfinite(current) or now < previous
                or current < mono or now >= expiry or current >= cap):
            return
        previous, mono = now, current
        beat(min(5., expiry - now, cap - current))
        sleep(max(0., min(10., expiry - clock(), cap - monotonic())))


def spawn_heartbeat(directory: Path, command: list[str], request: dict[str, Any],
                    expiry: float, environment: dict[str, str], *,
                    popen: Callable[..., Any] = subprocess.Popen) -> int:
    """Private config avoids argv size limits; child needs only SSH state and a finite cap."""
    if not math.isfinite(expiry):
        raise ValueError("finite heartbeat expiry required")
    path = directory / "retention-heartbeat.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"command": command, "request": request, "expires_unix": expiry},
                  stream, allow_nan=False)
    # Execute the frozen standalone source; no import path or budget/session dependency.
    source = Path(__file__).read_text()
    process = popen([sys.executable, "-I", "-c", source, str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, env=environment)
    if process.poll() is not None:
        raise RuntimeError("detached heartbeat exited during handoff")
    return int(process.pid)


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())

    def beat(timeout: float) -> None:
        request = dict(config["request"], timeout_seconds=timeout)
        reply = subprocess.run(config["command"], input=json.dumps(request).encode(),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, check=True)
        if json.loads(reply.stdout) != {"status": "heartbeat"}:
            raise RuntimeError("heartbeat rejected")

    feed_until(config["expires_unix"], beat)


if __name__ == "__main__":
    main()
