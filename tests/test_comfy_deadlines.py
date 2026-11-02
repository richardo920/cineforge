"""Synthetic HTTP socket boundaries only; no TCP service or live request."""
from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from cineforge.comfy_transport import ComfyHTTPTransport, TransportError


class SocketFixture:
    def __init__(self, now, *, after_send=None):
        self.now, self.after_send = now, after_send
        self.sent = []
        self.closed = False

    def setsockopt(self, *args):
        pass

    def settimeout(self, value):
        if value <= 0:
            raise AssertionError("nonpositive synthetic socket timeout")

    def sendall(self, data):
        self.sent.append((self.now[0], bytes(data)))
        if self.after_send is not None:
            self.after_send()

    def shutdown(self, *args):
        pass

    def close(self):
        self.closed = True


class ComfyAbsoluteHTTPTests(unittest.TestCase):
    def test_slow_body_encoding_is_in_original_cap_and_never_connects(self):
        now = [100.0]
        dumps = json.dumps
        def encode(*args, **kwargs):
            value = dumps(*args, **kwargs)
            now[0] = 104
            return value
        with patch("cineforge.comfy_transport.json.dumps", side_effect=encode), \
                patch("cineforge.comfy_transport.socket.create_connection") as connect:
            with self.assertRaises(TransportError):
                ComfyHTTPTransport(allow_live=True).request("POST", "/prompt", {"prompt": {}},
                    timeout_seconds=10, max_bytes=1024, absolute_deadline=103, clock=lambda: now[0])
        connect.assert_not_called()

    def test_connect_returning_late_sends_neither_headers_nor_body(self):
        now = [100.0]
        sock = SocketFixture(now)
        def connect(*args, **kwargs):
            now[0] = 104
            return sock
        with patch("cineforge.comfy_transport.socket.create_connection", side_effect=connect):
            with self.assertRaises(TransportError):
                ComfyHTTPTransport(allow_live=True).request("POST", "/prompt", {"prompt": {}},
                    timeout_seconds=10, max_bytes=1024, absolute_deadline=103, clock=lambda: now[0])
        self.assertEqual(sock.sent, [])
        self.assertTrue(sock.closed)

    def test_header_send_exhausting_cap_does_not_send_private_body(self):
        now = [100.0]
        sock = SocketFixture(now, after_send=lambda: now.__setitem__(0, 104))
        with patch("cineforge.comfy_transport.socket.create_connection", return_value=sock):
            with self.assertRaises(TransportError):
                ComfyHTTPTransport(allow_live=True).request("POST", "/prompt",
                    {"private": "CANARY synthetic body"}, timeout_seconds=10, max_bytes=1024,
                    absolute_deadline=103, clock=lambda: now[0])
        self.assertEqual(len(sock.sent), 1)
        self.assertEqual(sock.sent[0][0], 100)
        self.assertTrue(sock.sent[0][1].startswith(b"POST /prompt HTTP/1.1"))
        self.assertNotIn(b"CANARY", sock.sent[0][1])
        self.assertTrue(sock.closed)

    def test_response_read_crossing_cap_cannot_return_success_or_read_again(self):
        now = [100.0]
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.length = None
        def read(*args):
            now[0] = 104
            return b"late bytes"
        response.read1.side_effect = read
        with patch("cineforge.comfy_transport._CappedHTTPConnection", return_value=connection):
            with self.assertRaises(TransportError):
                ComfyHTTPTransport(allow_live=True).request("GET", "/queue", timeout_seconds=10,
                    max_bytes=1024, absolute_deadline=103, clock=lambda: now[0])
        self.assertEqual(response.read1.call_count, 1)
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
