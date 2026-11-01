"""Bounded HTTP transport for an explicitly opted-in loopback ComfyUI server."""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import socket
import time
from threading import Timer
from typing import Protocol
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
                *, timeout_seconds: float, max_bytes: int) -> bytes:
        if not self._allow_live:
            raise TransportError("live ComfyUI transport requires explicit opt-in")
        timeout = positive_seconds(timeout_seconds)
        if (method not in ("GET", "POST") or not path.startswith("/")
                or path.startswith("//") or any(ord(c) < 32 for c in path)
                or type(max_bytes) is not int or not 1 <= max_bytes <= 256 * 1024 * 1024):
            raise TransportError("invalid HTTP request bounds")
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        if encoded is not None and len(encoded) > 4 * 1024 * 1024:
            raise TransportError("ComfyUI request exceeds size limit")
        deadline = time.monotonic() + timeout
        connection = http.client.HTTPConnection(self._host, self._port, timeout=timeout)
        timer = None
        try:
            connection.connect()
            if connection.sock is None:
                raise TransportError("ComfyUI connection unavailable")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportError("ComfyUI request deadline exceeded")
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
            connection.request(method, path, body=encoded,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise TransportError("ComfyUI HTTP request rejected")
            chunks: list[bytes] = []
            total = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError("ComfyUI request deadline exceeded")
                connected_socket.settimeout(remaining)
                chunk = response.read1(min(65536, max_bytes + 1 - total))
                if time.monotonic() >= deadline:
                    raise TransportError("ComfyUI request deadline exceeded")
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
