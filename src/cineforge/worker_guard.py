"""Copyable stdlib worker-side guard: own Pod only, STOP only, no account key.

The operator verifies this exact source and a live guard PID before admitting work.
This is an independent best-effort shutdown path, not a provider hard billing cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import ssl
import stat
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Mapping


class WorkerGuardError(ValueError):
    """Static diagnostics only; never propagate HTTP, credential, or file contents."""


def _finite(value: object) -> bool:
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def _strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError

    result = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError
    return result


@dataclass(frozen=True)
class WorkerGuardPlan:
    pod_id: str
    session_id: str
    lease_token: str
    stop_at_unix: float
    idle_seconds: float

    def __post_init__(self) -> None:
        if (not isinstance(self.pod_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.pod_id) is None
                or not isinstance(self.session_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", self.session_id) is None
                or not isinstance(self.lease_token, str)
                or re.fullmatch(r"[0-9a-f]{32}", self.lease_token) is None
                or not _finite(self.stop_at_unix) or self.stop_at_unix <= 0
                or not _finite(self.idle_seconds) or not 0 < self.idle_seconds <= 1800):
            raise WorkerGuardError("invalid scoped worker guard plan")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "pod_id": self.pod_id, "session_id": self.session_id,
                "lease_token": self.lease_token, "stop_at_unix": self.stop_at_unix,
                "idle_seconds": self.idle_seconds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerGuardPlan:
        failed = False
        result = None
        try:
            if (set(data) != {"schema_version", "pod_id", "session_id", "lease_token",
                              "stop_at_unix", "idle_seconds"}
                    or type(data["schema_version"]) is not int or data["schema_version"] != 1):
                raise ValueError
            result = cls(**{k: v for k, v in data.items() if k != "schema_version"})
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise WorkerGuardError("invalid scoped worker guard plan")
        return result


@dataclass(frozen=True)
class GuardReceipt:
    """Operator attestation of remote source/PID verification, not future shutdown proof."""

    plan: WorkerGuardPlan
    source_sha256: str
    pid: int
    observed_unix: float
    operator_verified_alive: bool
    operator_verified_own_pod: bool

    def __post_init__(self) -> None:
        if (not isinstance(self.plan, WorkerGuardPlan)
                or not isinstance(self.source_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
                or type(self.pid) is not int or self.pid <= 0
                or not _finite(self.observed_unix)
                or self.operator_verified_alive is not True
                or self.operator_verified_own_pod is not True):
            raise WorkerGuardError("invalid worker guard verification")

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.to_dict(), "source_sha256": self.source_sha256,
                "pid": self.pid, "observed_unix": self.observed_unix,
                "operator_verified_alive": self.operator_verified_alive,
                "operator_verified_own_pod": self.operator_verified_own_pod}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardReceipt:
        failed = False
        result = None
        try:
            if set(data) != {"plan", "source_sha256", "pid", "observed_unix",
                             "operator_verified_alive", "operator_verified_own_pod"}:
                raise ValueError
            result = cls(WorkerGuardPlan.from_dict(data["plan"]), data["source_sha256"],
                         data["pid"], data["observed_unix"], data["operator_verified_alive"],
                         data["operator_verified_own_pod"])
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise WorkerGuardError("invalid worker guard verification")
        return result


def worker_guard_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_private_json(path: str | Path) -> dict[str, Any]:
    """No-follow owner-only heartbeat; callers write via atomic rename."""
    descriptor = -1
    failed = False
    result: dict[str, Any] | None = None
    try:
        parts = Path(path).absolute().parts
        descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
        for index, part in enumerate(parts[1:]):
            if (not part or part in (".", "..") or not part.isprintable()
                    or any(c in part for c in "\\:%")):
                raise ValueError
            child = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                            | (os.O_DIRECTORY if index < len(parts) - 2 else 0),
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size > 2048):
            raise ValueError
        result = _strict_json(os.read(descriptor, 2049))
    except (Exception, KeyboardInterrupt):
        failed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if failed or result is None:
        raise WorkerGuardError("invalid private worker input")
    return result


def read_heartbeat(path: str | Path, plan: WorkerGuardPlan) -> float:
    data = read_private_json(path)
    if (set(data) != {"schema_version", "pod_id", "lease_token", "heartbeat_unix"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["pod_id"] != plan.pod_id or data["lease_token"] != plan.lease_token
            or not _finite(data["heartbeat_unix"])):
        raise WorkerGuardError("invalid private worker heartbeat")
    return float(data["heartbeat_unix"])


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def worker_credential(plan: WorkerGuardPlan) -> str:
    """Existing worker environment/PID1 pair only; never creates or exports credentials."""
    values: Mapping[str, str] = os.environ
    if not values.get("RUNPOD_API_KEY") or not values.get("RUNPOD_POD_ID"):
        try:
            descriptor = os.open("/proc/1/environ", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                raw = os.read(descriptor, 1024 * 1024 + 1)
            finally:
                os.close(descriptor)
            if len(raw) > 1024 * 1024:
                raise ValueError
            parsed = dict(item.decode().split("=", 1) for item in raw.split(b"\0")
                          if b"=" in item)
            values = parsed
        except (Exception, KeyboardInterrupt):
            raise WorkerGuardError("worker-scoped shutdown credential is unavailable") from None
    if values.get("RUNPOD_POD_ID") != plan.pod_id:
        raise WorkerGuardError("worker identity differs from guard plan")
    key = values.get("RUNPOD_API_KEY", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{8,512}", key):
        raise WorkerGuardError("worker-scoped shutdown credential is unavailable")
    return key


def _worker_request(plan: WorkerGuardPlan) -> Callable[[str, bool], dict[str, Any]]:
    key = worker_credential(plan)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    url = "https://api.runpod.io/v2/pods/" + plan.pod_id

    def request(method: str, stop: bool = False) -> dict[str, Any]:
        req = urllib.request.Request(url + ("/action" if stop else ""),
            data=b'{"action":"stop"}' if stop else None, method=method,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with opener.open(req, timeout=3) as reply:
            data = reply.read(65537)
            if len(data) > 65536:
                raise ValueError
            return _strict_json(data)
    return request


def _owned(pod: dict[str, Any], plan: WorkerGuardPlan) -> bool:
    env = pod.get("env")
    return (pod.get("id") == plan.pod_id and isinstance(env, dict)
            and env.get("CINEFORGE_SESSION") == plan.session_id
            and env.get("CINEFORGE_LEASE") == plan.lease_token and pod.get("mounts") == {})


def verify_own_pod(plan: WorkerGuardPlan, *, allow_live: bool = False) -> bool:
    """Worker-only arming preflight using its existing scoped credential; GET only."""
    if allow_live is not True:
        raise WorkerGuardError("worker verification needs explicit live opt-in")
    try:
        return _owned(_worker_request(plan)("GET", False), plan)
    except (Exception, KeyboardInterrupt):
        return False


def stop_own_pod(plan: WorkerGuardPlan, *, allow_live: bool = False) -> bool:
    """GET ownership -> own-ID STOP -> GET terminal runtime. No DELETE or foreign ID."""
    if allow_live is not True:
        raise WorkerGuardError("worker shutdown needs explicit live opt-in")
    try:
        request = _worker_request(plan)
        pod = request("GET", False)
        if not _owned(pod, plan):
            return False
        if pod.get("status") not in ("EXITED", "TERMINATED"):
            if "stop" not in pod.get("actions", []):
                return False
            request("POST", True)
            pod = request("GET", False)
        return (_owned(pod, plan) and pod.get("status") in ("EXITED", "TERMINATED")
                and pod.get("runtime") is None)
    except (Exception, KeyboardInterrupt):
        return False


def run_guard(plan: WorkerGuardPlan, heartbeat: Callable[[], float],
              stop: Callable[[], bool], *, clock: Callable[[], float] = time.time,
              monotonic: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep,
              poll_seconds: float = 1, max_stop_attempts: int = 3,
              attempt_timeout_seconds: float = 5) -> dict[str, Any]:
    """Independent process; absent/invalid heartbeats, rollback or expiry request STOP.

    A heartbeat means the orchestrator is still supervising startup/generation; it
    must continue during long model downloads and inference. It never extends the
    absolute lease deadline. Timed-out stop threads can finish late, but only stop
    this same worker; they cannot create resources or mutate another Pod.
    """
    if (not isinstance(plan, WorkerGuardPlan) or not callable(heartbeat) or not callable(stop)
            or not _finite(poll_seconds) or not 0 < poll_seconds <= 30
            or type(max_stop_attempts) is not int or not 1 <= max_stop_attempts <= 5
            or not _finite(attempt_timeout_seconds) or not 0 < attempt_timeout_seconds <= 10):
        raise WorkerGuardError("invalid worker guard bounds")
    started_wall, started_mono = clock(), monotonic()
    if not _finite(started_wall) or not _finite(started_mono):
        raise WorkerGuardError("invalid worker guard clock")
    stop_mono = started_mono + max(0, plan.stop_at_unix - started_wall)
    previous_wall = started_wall
    previous_mono = started_mono
    previous_heartbeat = 0.0
    idle_anchor_mono = started_mono
    reason = "deadline"
    while True:
        now, mono = clock(), monotonic()
        if (not _finite(now) or not _finite(mono) or now < previous_wall
                or mono < previous_mono):
            reason = "clock_rollback"
            break
        previous_wall, previous_mono = now, mono
        if now >= plan.stop_at_unix or mono >= stop_mono:
            break
        try:
            beat = heartbeat()
            if not _finite(beat) or beat > now + 5 or beat < previous_heartbeat:
                raise ValueError
        except (Exception, KeyboardInterrupt):
            reason = "invalid_heartbeat"
            break
        if beat > previous_heartbeat:
            previous_heartbeat = beat
            idle_anchor_mono = mono
        if now - beat >= plan.idle_seconds or mono - idle_anchor_mono >= plan.idle_seconds:
            reason = "idle"
            break
        sleep(min(poll_seconds, max(0, stop_mono - mono)))
    for attempt in range(max_stop_attempts):
        answers: queue.Queue[bool] = queue.Queue(maxsize=1)

        def invoke(destination: queue.Queue[bool] = answers) -> None:
            try:
                destination.put(stop() is True)
            except (Exception, KeyboardInterrupt):
                destination.put(False)

        Thread(target=invoke, daemon=True).start()
        try:
            if answers.get(timeout=attempt_timeout_seconds):
                return {"status": "compute_stop_confirmed", "reason": reason,
                        "attempts": attempt + 1, "resource_deleted": False}
        except queue.Empty:
            pass
        if attempt + 1 < max_stop_attempts:
            sleep(1)
    return {"status": "compute_stop_unconfirmed", "reason": reason,
            "attempts": max_stop_attempts, "resource_deleted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Own-worker deadline guard (no account key).")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--verify", action="store_true", help="GET-only scoped arming preflight")
    args = parser.parse_args()
    failed = False
    answer = None
    try:
        plan = WorkerGuardPlan.from_dict(read_private_json(args.plan))
        if args.allow_live is not True:
            raise WorkerGuardError("worker guard identity or opt-in missing")
        worker_credential(plan)
        if args.verify:
            verified = verify_own_pod(plan, allow_live=True)
            print(json.dumps({"status": "own_pod_verified" if verified else "guard_failed"}))
            return 0 if verified else 2
        answer = run_guard(plan, lambda: read_heartbeat(args.heartbeat, plan),
                           lambda: stop_own_pod(plan, allow_live=True))
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or answer is None:
        print('{"status":"guard_failed","resource_deleted":false}')
        return 2
    print(json.dumps(answer, sort_keys=True))
    return 0 if answer["status"] == "compute_stop_confirmed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
