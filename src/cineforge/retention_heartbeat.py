"""Detached SSH feeder with bounded own-pod stop authority; no account key or journal."""

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


def make_beat(config: dict[str, Any], *, run: Callable[..., Any] = subprocess.run,
              clock: Callable[[], float] = time.time,
              monotonic: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> Callable[[float], None]:
    """Retry transient SSH failures within both the monotonic retry cap and expiry."""
    def beat(timeout: float) -> None:
        wall, mono = clock(), monotonic()
        deadline = mono + max(0., config["expires_unix"] - wall)
        retrying = False
        previous_wall, previous_mono = wall, mono
        error: Exception = RuntimeError("heartbeat retry deadline exhausted")
        while True:
            now, current = clock(), monotonic()
            if (not math.isfinite(now) or not math.isfinite(current)
                    or now < previous_wall or current < previous_mono
                    or now >= config["expires_unix"] or current >= deadline):
                raise error
            previous_wall, previous_mono = now, current
            bound = min(5., config["expires_unix"] - now, deadline - current)
            try:
                request = dict(config["request"], timeout_seconds=bound)
                reply = run(config["command"], input=json.dumps(request).encode(),
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=bound, check=True)
                if json.loads(reply.stdout) != {"status": "heartbeat"}:
                    raise RuntimeError("heartbeat rejected")
                return
            except Exception as failure:
                error = failure
                if not retrying:
                    deadline = min(deadline, monotonic() + config.get("retry_budget_seconds", 60))
                    retrying = True
            sleep(max(0., min(2., deadline - monotonic(), config["expires_unix"] - clock())))
    return beat


def spawn_heartbeat(directory: Path, command: list[str], request: dict[str, Any],
                    expiry: float, environment: dict[str, str], *,
                    popen: Callable[..., Any] = subprocess.Popen,
                    retry_budget_seconds: float = 60, stop_timeout_seconds: float = 30,
                    settle_seconds: float = .5) -> int:
    """Private config avoids argv size limits; child needs only SSH state and a finite cap."""
    if (not math.isfinite(expiry)
            or not math.isfinite(retry_budget_seconds) or not 0 < retry_budget_seconds <= 120
            or not math.isfinite(stop_timeout_seconds) or not 0 < stop_timeout_seconds <= 60
            or not math.isfinite(settle_seconds) or not 0 <= settle_seconds <= 5):
        raise ValueError("finite heartbeat bounds required")
    path = directory / "retention-heartbeat.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"command": command, "request": request, "expires_unix": expiry,
                   "retry_budget_seconds": retry_budget_seconds,
                   "stop_timeout_seconds": stop_timeout_seconds}, stream, allow_nan=False)
    # Execute the frozen standalone source; no import path or budget/session dependency.
    source = Path(__file__).read_text()
    try:
        process = popen([sys.executable, "-I", "-c", source, str(path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=environment)
    except BaseException:
        # A failed spawn must not leave a stale config behind: it blocks any
        # later O_EXCL spawn in the same state dir.
        try:
            path.unlink(missing_ok=True)
        except BaseException:
            pass
        raise
    try:
        time.sleep(settle_seconds)
        if process.poll() is not None:
            raise RuntimeError("detached heartbeat exited during handoff")
        return int(process.pid)
    except BaseException:
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=2)
        except BaseException:
            pass
        # SIGTERM skips the child's finally unlink, so the parent removes the
        # config the child can no longer be trusted to remove.
        try:
            path.unlink(missing_ok=True)
        except BaseException:
            pass
        raise


def main(*, run: Callable[..., Any] = subprocess.run,
         clock: Callable[[], float] = time.time,
         monotonic: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], None] = time.sleep) -> None:
    path = Path(sys.argv[1])
    try:
        config = json.loads(path.read_text())
        try:
            feed_until(config["expires_unix"], make_beat(config, run=run, clock=clock,
                monotonic=monotonic, sleep=sleep), clock=clock, monotonic=monotonic, sleep=sleep)
        except BaseException:
            pass
        finally:
            # Bounded retries: one transient Runpod/SSH failure must not leave a
            # guard-less pod running. Retry until dispatch reports "stopped" or
            # the stop budget is exhausted.
            try:
                timeout = config.get("stop_timeout_seconds", 30)
                if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                        or not 0 < timeout <= 60):
                    timeout = 30
                deadline = monotonic() + timeout
                while True:
                    remaining = deadline - monotonic()
                    if not math.isfinite(remaining) or remaining <= 0:
                        break
                    attempt = min(timeout, remaining)
                    request = dict(config["request"], operation="stop",
                                   timeout_seconds=attempt)
                    try:
                        reply = run(config["command"], input=json.dumps(request).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=attempt, check=True)
                        if json.loads(reply.stdout) == {"status": "stopped"}:
                            break
                    except BaseException:
                        pass
                    remaining = deadline - monotonic()
                    if not math.isfinite(remaining) or remaining <= 0:
                        break
                    sleep(min(1., remaining))
            except BaseException:
                pass
    finally:
        try:
            path.unlink(missing_ok=True)
        except BaseException:
            pass


if __name__ == "__main__":
    main()
