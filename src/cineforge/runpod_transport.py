"""Opted-in, fixed-origin Runpod REST v2 HTTPS; no retries or credential persistence."""

from __future__ import annotations

import http.client
import json
import math
import queue
import re
import socket
import ssl
import time
from dataclasses import dataclass
from threading import Thread, Timer
from typing import Protocol
from urllib.parse import parse_qsl, unquote, urlsplit


class CloudTransportError(ValueError):
    """Sanitized transport failure; a mutation's outcome may be unknown."""


def bounded_seconds(value: object) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            or not 0 < value <= 86400):
        raise CloudTransportError("invalid cloud request timeout")
    return float(value)


@dataclass(frozen=True)
class CloudResponse:
    status: int
    body: bytes


class CloudTransport(Protocol):
    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int) -> CloudResponse: ...


def _request_path(method: str, path: object) -> bool:
    if not isinstance(path, str):
        return False
    if re.fullmatch(r"/v2/pods(?:/[A-Za-z0-9_-]{1,80})?", path):
        return True
    try:
        url = urlsplit(path)
        fields = parse_qsl(url.query, strict_parsing=True, max_num_fields=8)
        if (method != "GET" or url.scheme or url.netloc or url.fragment
                or len(dict(fields)) != len(fields)):
            return False
        if url.path == "/v2/pods":
            return bool(fields) and all(
                (key == "limit" and value.isascii() and value.isdigit() and 1 <= int(value) <= 1000)
                or (key == "cursor" and 0 < len(value) <= 8192 and value.isprintable())
                for key, value in fields)
        prefix = "/v2/catalog/gpus/"
        if not url.path.startswith(prefix) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9 ._-]{1,100}", unquote(url.path[len(prefix):])):
            return False
        allowed = {"include": {"AVAILABILITY"}, "product": {"POD"},
                   "cloud": {"SECURE"}, "count": {"1"}}
        return all(value in allowed.get(key, set()) or (
            key == "minCudaVersion" and re.fullmatch(r"\d{1,2}\.\d{1,2}", value) is not None
            and tuple(map(int, value.split("."))) >= (12, 8)) for key, value in fields)
    except ValueError:
        return False


def _open_connection(deadline: float) -> http.client.HTTPSConnection:
    # DNS may outlive a socket timeout. Its daemon performs resolution only, never I/O
    # that could submit a late paid request after the caller's deadline.
    results: queue.Queue[object] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            results.put(socket.getaddrinfo("api.runpod.io", 443, type=socket.SOCK_STREAM))
        except OSError:
            results.put(None)

    Thread(target=resolve, daemon=True).start()
    try:
        addresses = results.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty:
        addresses = None
    if not isinstance(addresses, list):
        raise CloudTransportError("cloud name resolution failed or exceeded deadline")
    context = ssl.create_default_context()
    for family, kind, proto, _canonical, address in addresses:
        raw = socket.socket(family, kind, proto)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            raw.settimeout(remaining)
            raw.connect(address)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            raw.settimeout(remaining)
            secured = context.wrap_socket(raw, server_hostname="api.runpod.io")
            connection = http.client.HTTPSConnection("api.runpod.io", timeout=remaining)
            connection.sock = secured
            return connection
        except (OSError, ssl.SSLError):
            pass
        finally:
            try:
                raw.close()
            except (Exception, KeyboardInterrupt):
                pass
    raise CloudTransportError("cloud TLS connection failed or exceeded deadline")


class RunpodHTTPTransport:
    """Supply a key externally; no redirects, proxy environment, custom host or disk reads.

    Construction sends nothing. This low-level opt-in is not approval authentication or
    billing enforcement; the provisioner additionally requires its operator policy hook.
    """

    def __init__(self, api_key: str, *, allow_live: bool = False) -> None:
        if (not isinstance(api_key, str) or not re.fullmatch(r"[A-Za-z0-9._-]{8,512}", api_key)
                or type(allow_live) is not bool):
            raise CloudTransportError("invalid externally supplied cloud credentials or opt-in")
        self._api_key, self._allow_live = api_key, allow_live

    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int) -> CloudResponse:
        if not self._allow_live:
            raise CloudTransportError("live Runpod transport requires explicit opt-in")
        timeout = bounded_seconds(timeout_seconds)
        if (method not in ("GET", "POST", "DELETE")
                or not _request_path(method, path)
                or type(max_bytes) is not int or not 1 <= max_bytes <= 4 * 1024 * 1024
                or method in ("GET", "DELETE") and body is not None):
            raise CloudTransportError("invalid cloud request or bounds")
        failed = False
        encoded = None
        connection = None
        timer = None
        answer = None
        try:
            encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
            if encoded is not None and len(encoded) > 64 * 1024:
                raise ValueError
            deadline = time.monotonic() + timeout
            connection = _open_connection(deadline)
            connected = connection.sock
            if connected is None or time.monotonic() >= deadline:
                raise ValueError

            def expire() -> None:
                try:
                    connected.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = Timer(deadline - time.monotonic(), expire)
            timer.daemon = True
            timer.start()
            connection.request(method, path, body=encoded, headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json", "Accept": "application/json",
            })
            response = connection.getresponse()
            chunks: list[bytes] = []
            count = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                connected.settimeout(remaining)
                chunk = response.read1(min(65536, max_bytes + 1 - count))
                count += len(chunk)
                if count > max_bytes or time.monotonic() >= deadline:
                    raise ValueError
                if not chunk:
                    if response.length is not None and response.length != 0:
                        raise ValueError
                    answer = CloudResponse(response.status, b"".join(chunks))
                    break
                chunks.append(chunk)
        except (Exception, KeyboardInterrupt):
            failed = True
        finally:
            if timer is not None:
                timer.cancel()
            if connection is not None:
                try:
                    connection.close()
                except (Exception, KeyboardInterrupt):
                    # Local socket cleanup cannot erase a complete create response's
                    # known identity, or leak diagnostics while masking a prior error.
                    pass
        # Raise outside the handler: arbitrary socket/HTTP errors cannot survive in
        # traceback context or expose a key, response body or provider diagnostic.
        if failed or answer is None:
            raise CloudTransportError("Runpod HTTPS request failed")
        return answer
