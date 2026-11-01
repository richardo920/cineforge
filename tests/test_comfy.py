"""Offline ComfyUI protocol, translation and failure fixtures; no server or media."""

import copy
import hashlib
import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from cineforge.comfy import (
    SUPPORTED_SERVER_REVISION,
    ComfyExecutionProvider,
    JobDeadlineError,
    JobStateUncertainError,
    SubmissionUncertainError,
)
from cineforge.comfy_transport import ComfyHTTPTransport, TransportError
from cineforge.contracts import RunStatus, Shot
from cineforge.devices import Backend, ModelSpec
from cineforge.providers import (
    ExecutionProvider,
    ExecutionRequest,
    IdempotencyConflictError,
    ProviderError,
    ResultNotReadyError,
    UnknownJobError,
)
from cineforge.workflows import InputBinding, WorkflowTemplate

IDENTITY = "66a36419-0c15-4f86-bdfa-0506433e6b74"
OTHER = "b23ddf08-4417-4f58-a4a5-d92656917772"
GRAPH = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
         "2": {"class_type": "KSampler", "inputs": {"seed": 0}},
         "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}


def workflow(**changes):
    values = dict(graph_json=json.dumps(GRAPH), model_id="synthetic-model", model_revision="r1",
                  allowed_classes=frozenset({"CLIPTextEncode", "KSampler", "SaveImage"}),
                  prompt=InputBinding("1", "text"), seed=InputBinding("2", "seed"),
                  output_node="3", output_kind="images", artifact_license="LicenseRef-Synthetic")
    return WorkflowTemplate(**dict(values, **changes))


def request(template):
    model = ModelSpec("synthetic-model", "r1", "synthetic", frozenset({Backend.CPU}),
                      0, "LicenseRef-Synthetic")
    shot = Shot("shot-1", "scene-1", 1, "a synthetic scene", 42, 1, model.model_id,
                model.revision, template.digest)
    return ExecutionRequest(shot, model)


class ServerFixture:
    def __init__(self):
        self.calls = []
        self.mode = "queued"
        self.graph = None
        self.error = None
        self.raw = None
        self.descriptor = {"filename": "synthetic.bin", "subfolder": "scene-1", "type": "output"}

    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        self.calls.append((method, path, copy.deepcopy(body), timeout_seconds, max_bytes))
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        if self.raw is not None:
            raw, self.raw = self.raw, None
            return raw
        if path == "/prompt":
            self.graph = copy.deepcopy(body["prompt"])
            value = {"prompt_id": IDENTITY, "number": 0, "node_errors": {}}
        elif path == "/queue" and method == "POST":
            if self.mode == "queued" and body == {"delete": [IDENTITY]}:
                self.mode = "deleted"
            return b""
        elif path == "/interrupt":
            if self.mode == "running" and body == {"prompt_id": IDENTITY}:
                self.mode = "cancelled"
            return b""
        elif path == "/queue":
            value = {"queue_running": [[0, IDENTITY]] if self.mode == "running" else [],
                     "queue_pending": [[0, IDENTITY]] if self.mode == "queued" else []}
        elif path.startswith("/history/"):
            value = {}
            if self.mode in ("success", "error", "cancelled"):
                value = {IDENTITY: {
                    "prompt": [0, IDENTITY, self.graph, {}, ["3"]],
                    "status": {"status_str": "success" if self.mode == "success" else "error",
                               "completed": self.mode == "success",
                               "messages": [["execution_interrupted", {}]]
                               if self.mode == "cancelled" else []},
                    "outputs": {"3": {"images": [self.descriptor]}},
                }}
        elif path.startswith("/view?"):
            return b"synthetic bytes, no media"
        else:
            raise AssertionError(path)
        return json.dumps(value).encode()


class WorkflowTests(unittest.TestCase):
    def test_canonical_immutable_translation_and_identity(self):
        template = workflow()
        self.assertEqual(template.digest, workflow(graph_json=json.dumps(GRAPH, indent=2)).digest)
        translated = template.translate(request(template))
        self.assertEqual(translated["1"]["inputs"]["text"], "a synthetic scene")
        self.assertEqual(translated["2"]["inputs"]["seed"], 42)
        translated["1"]["inputs"]["text"] = "mutated"
        self.assertEqual(json.loads(template.graph_json), GRAPH)
        for shot in (replace(request(template).shot, workflow_digest="0" * 64),
                     replace(request(template).shot, seed=None),
                     replace(request(template).shot, duration_seconds=2)):
            with self.assertRaises(ProviderError):
                template.translate(replace(request(template), shot=shot))
        with self.assertRaises(ProviderError):
            workflow(model_revision="r2").translate(request(template))

    def test_bad_workflow_declarations_rejected(self):
        for changes in ({"graph_json": "{"}, {"graph_json": '{"x":1,"x":2}'},
                        {"graph_json": '{"x":NaN}'}, {"graph_json": "[]"},
                        {"allowed_classes": frozenset({"Other"})},
                        {"output_node": "unknown"}, {"output_kind": "arbitrary"},
                        {"prompt": InputBinding("missing", "text")},
                        {"seed": InputBinding("1", "text")}, {"output_index": True},
                        {"duration_seconds": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(ProviderError):
                workflow(**changes)


class ComfyProviderTests(unittest.TestCase):
    def setUp(self):
        self.server = ServerFixture()
        self.template = workflow()
        self.request = request(self.template)
        self.provider = ComfyExecutionProvider(self.server, self.template)

    def submit(self):
        return self.provider.submit(self.request, "synthetic-key")

    def test_lifecycle_result_integrity_and_terminal_stability(self):
        self.assertIsInstance(self.provider, ExecutionProvider)
        run = self.submit()
        self.assertEqual(run.status, RunStatus.QUEUED)
        with self.assertRaises(ResultNotReadyError):
            self.provider.result(run.job_id)
        self.assertEqual(self.provider.status(run.job_id).status, RunStatus.QUEUED)
        self.server.mode = "running"
        self.assertEqual(self.provider.status(run.job_id).status, RunStatus.RUNNING)
        self.server.mode = "success"
        succeeded = self.provider.status(run.job_id)
        self.assertEqual(succeeded.status, RunStatus.SUCCEEDED)
        result = self.provider.result(run.job_id)
        self.assertEqual(hashlib.sha256(result.payload).hexdigest(), result.artifact.sha256)
        self.assertEqual(len(result.payload), result.artifact.size_bytes)
        self.assertEqual(result.artifact.origin, f"cineforge.comfy-v1:{IDENTITY}")
        count = len(self.server.calls)
        self.assertEqual(self.provider.result(run.job_id), result)
        self.assertEqual(self.provider.status(run.job_id), succeeded)
        self.assertEqual(self.provider.cancel(run.job_id), succeeded)
        self.assertEqual(len(self.server.calls), count)

    def test_idempotency_conflicts_and_invalid_request_preflight(self):
        run = self.submit()
        self.assertEqual(self.submit(), run)
        self.assertEqual(len(self.server.calls), 1)
        changed = replace(self.request, shot=replace(self.request.shot, prompt="other"))
        with self.assertRaises(IdempotencyConflictError):
            self.provider.submit(changed, "synthetic-key")
        with self.assertRaises(ProviderError):
            self.provider.submit(replace(self.request, shot=replace(
                self.request.shot, workflow_digest="0" * 64)), "new-key")
        self.assertEqual(len(self.server.calls), 1)
        for method in (self.provider.status, self.provider.cancel, self.provider.result):
            with self.assertRaises(UnknownJobError):
                method(OTHER)

    def test_uncertain_submit_never_replays(self):
        for failure in (OSError("fake-credential-value"), TimeoutError("timeout"),
                        b"{", b'{"prompt_id":"not-a-uuid"}',
                        b'{"error":"fake-credential-value"}'):
            with self.subTest(failure=failure):
                server = ServerFixture()
                if isinstance(failure, bytes):
                    server.raw = failure
                else:
                    server.error = failure
                provider = ComfyExecutionProvider(server, self.template)
                for _ in range(2):
                    with self.assertRaises(SubmissionUncertainError) as caught:
                        provider.submit(self.request, "reserved-key")
                    self.assertNotIn("fake-credential-value", str(caught.exception))
                self.assertEqual(len(server.calls), 1)

    def test_failure_and_interruption_history(self):
        for mode, expected in (("error", RunStatus.FAILED), ("cancelled", RunStatus.CANCELLED)):
            with self.subTest(mode=mode):
                provider = ComfyExecutionProvider(self.server, self.template)
                run = provider.submit(self.request, mode)
                self.server.mode = mode
                self.assertEqual(provider.status(run.job_id).status, expected)
                with self.assertRaises(ResultNotReadyError):
                    provider.result(run.job_id)

    def test_queued_cancel_never_uses_interrupt(self):
        run = self.submit()
        self.assertEqual(self.provider.cancel(run.job_id).status, RunStatus.CANCELLED)
        self.assertFalse(any(c[1] == "/interrupt" for c in self.server.calls))
        self.assertIn(("POST", "/queue", {"delete": [IDENTITY]}),
                      [c[:3] for c in self.server.calls])

    def test_running_cancel_requires_verified_targeted_api(self):
        run = self.submit()
        self.server.mode = "running"
        with self.assertRaisesRegex(ProviderError, "verified server"):
            self.provider.cancel(run.job_id)
        self.assertFalse(any(c[1] == "/interrupt" for c in self.server.calls))
        trusted = ComfyExecutionProvider(self.server, self.template,
                                        verified_server_revision=SUPPORTED_SERVER_REVISION)
        trusted.submit(self.request, "trusted-key")
        self.assertEqual(trusted.cancel(run.job_id).status, RunStatus.CANCELLED)
        interrupts = [c for c in self.server.calls if c[1] == "/interrupt"]
        self.assertEqual([c[2] for c in interrupts], [{"prompt_id": IDENTITY}])
        with self.assertRaises(ProviderError):
            ComfyExecutionProvider(self.server, self.template, verified_server_revision="old")

    def test_malformed_reply_connection_loss_and_unknown_state_do_not_claim_failure(self):
        run = self.submit()
        for raw in (b"{", b"[]", b'{"unknown":{}}',
                    b'{"queue_running":"wrong","queue_pending":[]}'):
            self.server.raw = raw
            with self.assertRaises(ProviderError):
                self.provider.status(run.job_id)
            self.assertEqual(self.submit().status, RunStatus.QUEUED)
        self.server.error = OSError("secret exception detail")
        with self.assertRaises(ProviderError) as caught:
            self.provider.status(run.job_id)
        self.assertNotIn("secret", str(caught.exception))
        self.server.mode = "lost"
        with self.assertRaises(JobStateUncertainError):
            self.provider.status(run.job_id)

    def test_history_identity_and_output_paths_fail_closed(self):
        run = self.submit()
        self.server.mode = "success"
        self.server.graph = {"tampered": {}}
        with self.assertRaises(ProviderError):
            self.provider.status(run.job_id)
        self.server.graph = self.template.translate(self.request)
        for field, value in (("filename", "../outside"), ("filename", "x\\outside"),
                             ("filename", "x%2e"), ("subfolder", "/outside"),
                             ("subfolder", "a/../b"), ("type", "input")):
            with self.subTest(field=field, value=value):
                self.server.descriptor = {"filename": "safe.bin", "subfolder": "", "type": "output"}
                self.server.descriptor[field] = value
                with self.assertRaises(ProviderError):
                    self.provider.status(run.job_id)
        self.assertFalse(any(c[1].startswith("/view") for c in self.server.calls))

    def test_deadline_poll_limit_and_cancellation_budget(self):
        now = [0.0]
        provider = ComfyExecutionProvider(self.server, self.template, job_timeout_seconds=1,
                                         max_polls=1, clock=lambda: now[0])
        run = provider.submit(self.request, "key")
        provider.status(run.job_id)
        with self.assertRaises(JobDeadlineError):
            provider.status(run.job_id)
        now[0] = 2
        self.assertEqual(provider.cancel(run.job_id).status, RunStatus.CANCELLED)
        another = ComfyExecutionProvider(self.server, self.template, clock=lambda: now[0])
        self.server.mode = "queued"
        run = another.submit(self.request, "another")
        count = len(self.server.calls)
        now[0] = 100
        with self.assertRaises(JobDeadlineError):
            another.status(run.job_id)
        self.assertEqual(len(self.server.calls), count)

    def test_oversized_result_is_not_cached(self):
        provider = ComfyExecutionProvider(self.server, self.template, max_result_bytes=1)
        run = provider.submit(self.request, "key")
        self.server.mode = "success"
        provider.status(run.job_id)
        with self.assertRaisesRegex(ProviderError, "oversized"):
            provider.result(run.job_id)

    def test_invalid_bounds_and_strict_json_fail_without_submission(self):
        for options in ({"max_polls": True}, {"max_polls": 0},
                        {"max_result_bytes": 0}, {"job_timeout_seconds": float("nan")},
                        {"request_timeout_seconds": True}):
            with self.subTest(options=options), self.assertRaises(ProviderError):
                ComfyExecutionProvider(self.server, self.template, **options)
        self.assertEqual(self.server.calls, [])
        for raw in (b"\xff", b'{"prompt_id":1,"prompt_id":2}', b'{"value":Infinity}'):
            server = ServerFixture()
            server.raw = raw
            provider = ComfyExecutionProvider(server, self.template)
            with self.assertRaises(SubmissionUncertainError):
                provider.submit(self.request, "key")
            self.assertEqual(len(server.calls), 1)


class HTTPTransportTests(unittest.TestCase):
    def test_constructor_and_opt_in_have_no_network_side_effects(self):
        with patch("http.client.HTTPConnection") as connection:
            transport = ComfyHTTPTransport()
            with self.assertRaises(TransportError):
                transport.request("GET", "/queue", timeout_seconds=1, max_bytes=10)
            connection.assert_not_called()
        for endpoint in ("https://127.0.0.1", "http://user:fake@127.0.0.1:8188",  # pragma: allowlist secret
                         "http://example.com", "http://192.168.1.2", "http://127.0.0.1/x",
                         "http://127.0.0.1?token=fake", "http://127.0.0.1#x"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ProviderError):
                ComfyHTTPTransport(endpoint)

    def test_bounded_read_http_failure_cleanup_and_no_redirect(self):
        transport = ComfyHTTPTransport(allow_live=True)
        for status, chunks, expected in ((200, [b"ab", b""], b"ab"),
                                         (200, [b"oversize"], None), (302, [], None)):
            connection = Mock()
            response = connection.getresponse.return_value
            response.status = status
            response.length = None
            response.read1.side_effect = chunks
            with patch("http.client.HTTPConnection", return_value=connection):
                if expected is None:
                    with self.assertRaises(TransportError):
                        transport.request("GET", "/queue", timeout_seconds=1, max_bytes=2)
                else:
                    self.assertEqual(transport.request("GET", "/queue", timeout_seconds=1,
                                                       max_bytes=2), expected)
                connection.close.assert_called_once()
                self.assertEqual(connection.request.call_count, 1)

    def test_network_errors_are_sanitized(self):
        connection = Mock()
        connection.connect.side_effect = OSError("fake-credential-value")
        with patch("http.client.HTTPConnection", return_value=connection):
            with self.assertRaises(TransportError) as caught:
                ComfyHTTPTransport(allow_live=True).request("GET", "/queue", timeout_seconds=1,
                                                          max_bytes=10)
            self.assertNotIn("fake-credential-value", str(caught.exception))
            connection.close.assert_called_once()

    def test_total_deadline_and_truncation_reject_partial_results(self):
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.length = 10
        response.read1.return_value = b""
        transport = ComfyHTTPTransport(allow_live=True)
        with patch("http.client.HTTPConnection", return_value=connection):
            with self.assertRaisesRegex(TransportError, "truncated"):
                transport.request("GET", "/view", timeout_seconds=1, max_bytes=100)
        connection = Mock()
        connection.getresponse.return_value.status = 200
        with patch("http.client.HTTPConnection", return_value=connection), \
                patch("cineforge.comfy_transport.time.monotonic", side_effect=[0, 0, 2]):
            with self.assertRaisesRegex(TransportError, "deadline"):
                transport.request("GET", "/view", timeout_seconds=1, max_bytes=100)
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
