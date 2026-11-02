"""Bounded HTTP transport for an explicitly opted-in loopback ComfyUI server."""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import socket
import time
from collections.abc import Callable
from threading import Timer
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from cineforge.providers import ProviderError


class TransportError(ProviderError):
    """A sanitized connection, HTTP, timeout, or response-size failure."""


def positive_seconds(value: object) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            or value <= 0 or value > 86400):
        raise ProviderError("timeout must be finite and in (0, 86400] seconds")
    return float(value)


class ComfyTransport(Protocol):
    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int) -> bytes: ...


def request_deadline(timeout_seconds: float, absolute_deadline: float | None,
                     clock: Callable[[], float]) -> float:
    """Keep an optional trusted cap in the caller's clock domain, never renew it."""
    now = clock()
    if (type(now) not in (int, float) or not math.isfinite(now)
            or absolute_deadline is not None and (type(absolute_deadline) not in (int, float)
            or not math.isfinite(absolute_deadline))):
        raise TransportError("invalid ComfyUI absolute deadline")
    deadline = now + positive_seconds(timeout_seconds)
    if absolute_deadline is not None:
        deadline = min(deadline, absolute_deadline)
    if deadline <= now:
        raise TransportError("ComfyUI request deadline exceeded")
    return deadline


def require_time(deadline: float, clock: Callable[[], float]) -> float:
    now = clock()
    if type(now) not in (int, float) or not math.isfinite(now) or now >= deadline:
        raise TransportError("ComfyUI request deadline exceeded")
    return deadline - now


def request_capped(transport: ComfyTransport, method: str, path: str,
                   body: dict[str, object] | None, *, timeout_seconds: float,
                   max_bytes: int, absolute_deadline: float,
                   clock: Callable[[], float]) -> bytes:
    """First-party transports preserve the cap through their own admission/sends.

    Injected legacy transports remain trusted code with the existing relative API;
    entry/return are checked, with no TypeError retry or capability inference.
    """
    from cineforge.ssh_worker import SupervisedComfyTransport
    remaining = require_time(absolute_deadline, clock)
    if type(transport) in (ComfyHTTPTransport, SupervisedComfyTransport):
        native = cast("ComfyHTTPTransport | SupervisedComfyTransport", transport)
        raw = native.request(method, path, body, timeout_seconds=min(timeout_seconds, remaining),
            max_bytes=max_bytes, absolute_deadline=absolute_deadline, clock=clock)
    else:
        raw = transport.request(method, path, body,
            timeout_seconds=min(timeout_seconds, remaining), max_bytes=max_bytes)
    require_time(absolute_deadline, clock)
    return raw


class _CappedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, *, deadline: float,
                 clock: Callable[[], float]) -> None:
        self._deadline, self._clock = deadline, clock
        super().__init__(host, port, timeout=require_time(deadline, clock))

    def send(self, data: Any) -> None:
        # HTTPConnection calls send separately for headers and body/chunks.
        require_time(self._deadline, self._clock)
        if self.sock is None:
            self.connect()
        remaining = require_time(self._deadline, self._clock)
        if self.sock is None:
            raise TransportError("ComfyUI connection unavailable")
        self.sock.settimeout(remaining)
        super().send(data)
        require_time(self._deadline, self._clock)


class ComfyHTTPTransport:
    """No proxies, redirects, credentials, DNS resolution, or construction-time requests.

    Remote workers can be reached through an owner-configured loopback tunnel. This
    object neither establishes tunnels nor provisions/installs anything.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8188", *,
                 allow_live: bool = False) -> None:
        if type(allow_live) is not bool or not isinstance(endpoint, str):
            raise ProviderError("invalid transport configuration")
        try:
            url = urlsplit(endpoint)
            address = ipaddress.ip_address(url.hostname or "")
            port = url.port or 8188
            if (url.scheme != "http" or not address.is_loopback or url.username is not None
                    or url.password is not None or url.path not in ("", "/")
                    or url.query or url.fragment or not 1 <= port <= 65535):
                raise ValueError
        except ValueError:
            raise ProviderError("ComfyUI endpoint must be a credential-free loopback HTTP origin") \
                from None
        self._host = str(address)
        self._port = port
        self._allow_live = allow_live

    def request(self, method: str, path: str, body: dict[str, object] | None = None,
                *, timeout_seconds: float, max_bytes: int,
                absolute_deadline: float | None = None,
                clock: Callable[[], float] | None = None) -> bytes:
        if not self._allow_live:
            raise TransportError("live ComfyUI transport requires explicit opt-in")
        timeout = positive_seconds(timeout_seconds)
        clock = time.monotonic if clock is None else clock
        deadline = request_deadline(timeout, absolute_deadline, clock)
        if (method not in ("GET", "POST") or not path.startswith("/")
                or path.startswith("//") or any(ord(c) < 32 for c in path)
                or type(max_bytes) is not int or not 1 <= max_bytes <= 256 * 1024 * 1024):
            raise TransportError("invalid HTTP request bounds")
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        if encoded is not None and len(encoded) > 4 * 1024 * 1024:
            raise TransportError("ComfyUI request exceeds size limit")
        require_time(deadline, clock)
        connection = (http.client.HTTPConnection(self._host, self._port, timeout=timeout)
                      if absolute_deadline is None else
                      _CappedHTTPConnection(self._host, self._port, deadline=deadline, clock=clock))
        timer = None
        try:
            connection.connect()
            if connection.sock is None:
                raise TransportError("ComfyUI connection unavailable")
            remaining = require_time(deadline, clock)
            connected_socket = connection.sock

            def expire() -> None:
                try:
                    connected_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            # Socket timeouts alone do not bound a peer slowly trickling HTTP headers.
            timer = Timer(remaining, expire)
            timer.daemon = True
            timer.start()
            connected_socket.settimeout(remaining)
            require_time(deadline, clock)
            connection.request(method, path, body=encoded,
                               headers={"Content-Type": "application/json"})
            require_time(deadline, clock)
            response = connection.getresponse()
            if response.status != 200:
                raise TransportError("ComfyUI HTTP request rejected")
            chunks: list[bytes] = []
            total = 0
            while True:
                remaining = require_time(deadline, clock)
                connected_socket.settimeout(remaining)
                chunk = response.read1(min(65536, max_bytes + 1 - total))
                require_time(deadline, clock)
                if not chunk:
                    if response.length is not None and response.length != 0:
                        raise TransportError("truncated ComfyUI response")
                    return b"".join(chunks)
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise TransportError("ComfyUI response exceeds size limit")
        except (OSError, http.client.HTTPException):
            raise TransportError("ComfyUI connection or timeout failure") from None
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()
