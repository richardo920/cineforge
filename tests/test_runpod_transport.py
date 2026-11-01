"""The native transport is exercised through sockets/HTTP fakes, never the network."""

import socket
import threading
import time
import traceback
import unittest
from unittest.mock import Mock, patch

from cineforge.runpod_transport import CloudTransportError, RunpodHTTPTransport, _open_connection


class FakeSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, seconds):
        self.timeouts.append(seconds)

    def shutdown(self, _how):
        pass


class FakeResponse:
    status = 200

    def __init__(self, payload=b'{}', unread=0):
        self.payload = payload
        self.length = unread

    def read1(self, size):
        result, self.payload = self.payload[:size], self.payload[size:]
        return result


class FakeConnection:
    def __init__(self, response=None):
        self.sock = FakeSocket()
        self.response = response or FakeResponse()
        self.closed = False
        self.sent = None

    def request(self, method, path, *, body, headers):
        self.sent = method, path, body, headers

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class RunpodTransportTests(unittest.TestCase):
    def test_no_request_without_live_optin_and_no_credentials_in_repr(self):
        transport = RunpodHTTPTransport("fixture-sensitive-value")
        with patch("cineforge.runpod_transport._open_connection") as connection:
            with self.assertRaises(CloudTransportError):
                transport.request("POST", "/v2/pods", {}, timeout_seconds=1, max_bytes=100)
            connection.assert_not_called()
        self.assertNotIn("fixture-sensitive-value", repr(transport))

    def test_valid_native_wire_request_carries_key_only_in_https_header(self):
        transport = RunpodHTTPTransport("fixture-sensitive-value", allow_live=True)
        fake = FakeConnection()
        with patch("cineforge.runpod_transport._open_connection", return_value=fake):
            response = transport.request("POST", "/v2/pods", {"name": "fixture"},
                                         timeout_seconds=1, max_bytes=100)
        self.assertEqual(response.body, b'{}')
        method, path, body, headers = fake.sent
        self.assertEqual((method, path), ("POST", "/v2/pods"))
        self.assertNotIn("fixture-sensitive-value", path)
        self.assertNotIn(b"fixture-sensitive-value", body)
        self.assertEqual(headers["Authorization"], "Bearer fixture-sensitive-value")
        self.assertTrue(fake.closed)

    def test_arbitrary_socket_diagnostic_cannot_survive_exception_context(self):
        transport = RunpodHTTPTransport("fixture-sensitive-value", allow_live=True)
        with patch("cineforge.runpod_transport._open_connection",
                   side_effect=OSError("https://user:fixture-sensitive-value@host")):
            with self.assertRaises(CloudTransportError) as cm:
                transport.request("GET", "/v2/pods", timeout_seconds=1, max_bytes=100)
        self.assertIsNone(cm.exception.__context__)
        self.assertNotIn("fixture-sensitive", "".join(traceback.format_exception(cm.exception)))

    def test_close_failure_does_not_erase_complete_create_identity(self):
        fake = FakeConnection(FakeResponse(b'{"id":"pod_fixture"}'))
        fake.response.status = 201
        fake.close = Mock(side_effect=RuntimeError("Bearer fixture-sensitive-value"))
        with patch("cineforge.runpod_transport._open_connection", return_value=fake):
            response = RunpodHTTPTransport("fixture-value", allow_live=True).request(
                "POST", "/v2/pods", {"name": "fixture"}, timeout_seconds=1, max_bytes=100)
        self.assertEqual(response.status, 201)
        self.assertEqual(response.body, b'{"id":"pod_fixture"}')
        fake.close.assert_called_once()

    def test_close_failure_cannot_replace_or_chain_a_sanitized_request_error(self):
        fake = FakeConnection()
        fake.getresponse = Mock(side_effect=OSError("private request diagnostic"))
        fake.close = Mock(side_effect=RuntimeError("Bearer fixture-sensitive-value"))
        with patch("cineforge.runpod_transport._open_connection", return_value=fake):
            with self.assertRaises(CloudTransportError) as cm:
                RunpodHTTPTransport("fixture-value", allow_live=True).request(
                    "GET", "/v2/pods", timeout_seconds=1, max_bytes=100)
        self.assertIsNone(cm.exception.__cause__)
        self.assertIsNone(cm.exception.__context__)
        rendered = "".join(traceback.format_exception(cm.exception))
        self.assertNotIn("fixture-sensitive", rendered)
        self.assertNotIn("private request diagnostic", rendered)

    def test_raw_socket_finally_failure_does_not_mask_the_verified_tls_socket(self):
        raw, secured = Mock(), Mock()
        raw.close.side_effect = RuntimeError("Bearer fixture-sensitive-value")
        context = Mock()
        context.wrap_socket.return_value = secured
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with patch("cineforge.runpod_transport.socket.getaddrinfo", return_value=addresses), \
                patch("cineforge.runpod_transport.socket.socket", return_value=raw), \
                patch("cineforge.runpod_transport.ssl.create_default_context", return_value=context):
            connection = _open_connection(time.monotonic() + 1)
        self.assertIs(connection.sock, secured)
        context.wrap_socket.assert_called_once_with(raw, server_hostname="api.runpod.io")
        raw.close.assert_called_once()
        connection.close()

    def test_no_redirect_following_or_provider_body_in_exception(self):
        fake = FakeConnection(FakeResponse(b'private provider diagnostic'))
        fake.response.status = 302
        with patch("cineforge.runpod_transport._open_connection", return_value=fake) as opening:
            response = RunpodHTTPTransport("fixture-value", allow_live=True).request(
                "GET", "/v2/pods", timeout_seconds=1, max_bytes=100)
        self.assertEqual(response.status, 302)
        self.assertEqual(opening.call_count, 1)

    def test_readonly_pagination_supports_future_owned_orphan_discovery(self):
        for path in ("/v2/pods?limit=1000", "/v2/pods?limit=20&cursor=opaque%2Bcursor%3D"):
            with patch("cineforge.runpod_transport._open_connection",
                       return_value=FakeConnection()) as opening:
                RunpodHTTPTransport("fixture-value", allow_live=True).request(
                    "GET", path, timeout_seconds=1, max_bytes=100)
                self.assertEqual(opening.call_count, 1)

    def test_catalog_read_for_policy_preflight_is_fixed_origin_and_exact(self):
        path = ("/v2/catalog/gpus/NVIDIA%20GeForce%20RTX%204090?include=AVAILABILITY"
                "&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8")
        with patch("cineforge.runpod_transport._open_connection",
                   return_value=FakeConnection()) as opening:
            RunpodHTTPTransport("fixture-value", allow_live=True).request(
                "GET", path, timeout_seconds=1, max_bytes=100)
            self.assertEqual(opening.call_count, 1)

    def test_oversized_and_truncated_responses_fail_and_close(self):
        for response in (FakeResponse(b'123456'), FakeResponse(b'{}', unread=4)):
            with self.subTest(response=response):
                fake = FakeConnection(response)
                with patch("cineforge.runpod_transport._open_connection", return_value=fake):
                    with self.assertRaises(CloudTransportError):
                        RunpodHTTPTransport("fixture-value", allow_live=True).request(
                            "GET", "/v2/pods", timeout_seconds=1, max_bytes=5)
                self.assertTrue(fake.closed)

    def test_path_method_payload_and_bound_validation_happens_before_dns(self):
        transport = RunpodHTTPTransport("fixture-value", allow_live=True)
        requests = [("GET", "https://user:secret@other", None),  # pragma: allowlist secret - invalid synthetic credential fixture
                    ("GET", "//other/v2/pods", None), ("GET", "/v2/pods/../account", None),
                    ("GET", "/v2/pods?api_key=private", None),
                    ("GET", "/v2/pods?limit=1001", None),
                    ("GET", "/v2/pods?limit=1&limit=2", None),
                    ("POST", "/v2/pods?limit=20", {}),
                    ("POST", "/v2/catalog/gpus/NVIDIA%20GeForce%20RTX%204090", {}),
                    ("GET", "/v2/catalog/gpus/..%2Faccount", None),
                    ("GET", "/v2/catalog/gpus/fixture?api_key=private", None),
                    ("GET", "/v2/pods/fixture\r\nX-Key: secret", None),
                    ("PATCH", "/v2/pods/fixture", {}), ("GET", "/v2/pods", {}),
                    ("POST", "/v2/pods", {"x": float("nan")})]
        with patch("cineforge.runpod_transport._open_connection") as connection:
            for method, path, body in requests:
                with self.assertRaises(CloudTransportError):
                    transport.request(method, path, body, timeout_seconds=1, max_bytes=100)
            for timeout, size in [(0, 100), (float("inf"), 100), (True, 100), (1, 0), (1, True)]:
                with self.assertRaises(CloudTransportError):
                    transport.request("GET", "/v2/pods", timeout_seconds=timeout, max_bytes=size)
            connection.assert_not_called()

    def test_dns_deadline_cannot_leave_a_late_paid_connection(self):
        release = threading.Event()

        def held_resolution(*_args, **_kwargs):
            release.wait(1)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

        try:
            with patch("cineforge.runpod_transport.socket.getaddrinfo", held_resolution), \
                    patch("cineforge.runpod_transport.socket.socket") as socket_factory:
                started = time.monotonic()
                with self.assertRaises(CloudTransportError):
                    _open_connection(started + .02)
                self.assertLess(time.monotonic() - started, .5)
                release.set()
                socket_factory.assert_not_called()
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
