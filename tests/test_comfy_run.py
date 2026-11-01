"""Offline admission, filesystem, redaction and one-shot fixtures; no server or media."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cineforge.comfy import SUPPORTED_SERVER_REVISION
from cineforge.comfy_policy import (
    ApprovedComfySelection,
    ComfyPolicyError,
    WorkerVerification,
    load_comfy_selection,
)
from cineforge.comfy_run import ComfyRunError, execute_comfy
from cineforge.contracts import Project, ProjectPlan, RunManifest, RunStatus, Scene, Shot
from cineforge.private_store import ArtifactStoreError, PrivateRunDirectory, read_private_input
from cineforge.providers import FakeExecutionProvider
from cineforge.selection import selection_subject
from cineforge.workflows import canonical

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "config" / "comfy" / "wan21-t2v-1.3b.json"
IDENTITY = "66a36419-0c15-4f86-bdfa-0506433e6b74"
PAYLOAD = b"synthetic fixture bytes, not encoded media"


def worker_record(selection):
    data = selection.declarations
    return {
        "schema_version": 1, "operator_verified": True, "worker_id": "synthetic-worker",
        "server_revision": SUPPORTED_SERVER_REVISION,
        "image_digest": "sha256:" + hashlib.sha256(b"synthetic image").hexdigest(),
        "python_version": "3.12.12", "torch_version": "2.7.1+cu128", "cuda_version": "12.8",
        "gpu": {"name": data["hardware"]["gpu"], "backend": "cuda", "memory_mib": 24576},
        "model_files": [{k: asset[k] for k in ("path", "sha256", "size_bytes")}
                        for asset in data["model"]["files"]],
        "custom_nodes_disabled": True,
    }


def plan_for(selection):
    shot = Shot("shot-1", "scene-1", 1, "A synthetic paper boat.", 42,
                selection.workflow.duration_seconds, selection.model.model_id,
                selection.model.revision, selection.workflow.digest)
    return ProjectPlan(Project("project-1", 1, "Synthetic test"),
                       (Scene("scene-1", "project-1", 1),), (shot,))


def revised_selection(selection, graph, change):
    """Even separately approved declarations must satisfy fixed safety boundaries."""
    data, modified = selection.declarations, copy.deepcopy(graph)
    change(data, modified)
    data["workflow"]["canonical_sha256"] = hashlib.sha256(canonical(modified)).hexdigest()
    data["approval"]["model_workflow"]["approved_subject_sha256"] = hashlib.sha256(
        canonical(selection_subject(data))).hexdigest()
    return ApprovedComfySelection(canonical(data).decode(), canonical(modified).decode(),
                                  selection.license_bytes)


class WireFixture:
    def __init__(self, selection, *, mode="success", failure=None):
        self.template = json.loads(selection.graph_json)
        self.graph = None
        self.calls = []
        self.mode, self.failure = mode, failure
        self.schema_failure = False
        self.on_submit = None
        self.history_graph_change = None

    def request(self, method, path, body=None, *, timeout_seconds, max_bytes):
        self.calls.append((method, path, copy.deepcopy(body), timeout_seconds, max_bytes))
        if self.failure is not None:
            raise self.failure
        if path.startswith("/object_info/"):
            kind = path.rsplit("/", 1)[1]
            inputs = {}
            for node in self.template.values():
                if node["class_type"] == kind:
                    for key, value in node["inputs"].items():
                        inputs[key] = [[value]] if key in ("unet_name", "clip_name", "vae_name") \
                            else ["SYNTHETIC"]
            value = {kind: {"input": {"required": inputs}, "output": ["SYNTHETIC"],
                            "output_node": kind == "SaveWEBM"}}
            if self.schema_failure:
                value = {}
        elif path == "/prompt":
            self.graph = copy.deepcopy(body["prompt"])
            if self.on_submit:
                self.on_submit()
            value = {"prompt_id": IDENTITY, "node_errors": {}}
        elif path.startswith("/history/"):
            value = {}
            if self.mode in ("success", "error", "cancelled"):
                history_graph = copy.deepcopy(self.graph)
                if self.history_graph_change:
                    self.history_graph_change(history_graph)
                value = {IDENTITY: {
                    "prompt": [0, IDENTITY, history_graph, {}, ["47"]],
                    "status": {"status_str": "success" if self.mode == "success" else "error",
                               "completed": self.mode == "success", "messages":
                               [["execution_interrupted", {}]] if self.mode == "cancelled" else []},
                    "outputs": {"47": {"images": [{"filename": "fixture.webm",
                                                    "subfolder": "", "type": "output",
                                                    "metadata": {"kind": "video", "width": 832,
                                                                 "height": 480, "duration": 2.063,
                                                                 "fps": 16.0}}],
                                       "animated": [True]}},
                }}
        elif path == "/queue":
            value = {"queue_running": [[0, IDENTITY]] if self.mode == "running" else [],
                     "queue_pending": [[0, IDENTITY]] if self.mode == "queued" else []}
        elif path == f"/api/jobs/{IDENTITY}/cancel":
            self.mode = "cancelled"
            value = {"cancelled": True}
        elif path.startswith("/view?"):
            return PAYLOAD
        else:
            raise AssertionError("unexpected transport operation")
        return json.dumps(value).encode()


class ComfyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.selection = load_comfy_selection(LEDGER)
        self.graph = json.loads(self.selection.graph_json)

    def test_approved_snapshot_is_immutable_and_requires_per_run_opt_in(self):
        data = self.selection.declarations
        self.assertFalse(data["readiness"]["live_ready"])
        self.assertFalse(data["approval"]["live_execution"])
        data["model"]["revision"] = "changed"
        self.assertNotEqual(self.selection.model.revision, "changed")
        worker = WorkerVerification.from_dict(worker_record(self.selection))
        worker.require_matches(self.selection)
        record = worker.record
        record["operator_verified"] = False
        self.assertTrue(worker.record["operator_verified"])

    def test_unapproved_or_tampered_selection_rejected(self):
        for field in ("model_approved", "workflow_approved", "hardware_approved"):
            data = self.selection.declarations
            data["approval"]["model_workflow"][field] = False
            with self.subTest(field=field), self.assertRaises(ComfyPolicyError):
                ApprovedComfySelection(json.dumps(data), self.selection.graph_json,
                                       self.selection.license_bytes)
        data = self.selection.declarations
        data["model"]["files"][0]["sha256"] = "a" * 64
        with self.assertRaises(ComfyPolicyError):
            ApprovedComfySelection(json.dumps(data), self.selection.graph_json,
                                   self.selection.license_bytes)

    def test_even_consistent_approval_cannot_admit_custom_nodes_or_unsafe_paths(self):
        changes = [
            lambda s, g: s["runner"].update(custom_nodes=["unreviewed"]),
            lambda s, g: g["47"]["inputs"].update(filename_prefix="../outside"),
            lambda s, g: g["47"]["inputs"].update(filename_prefix="/outside"),
            lambda s, g: g["37"]["inputs"].update(unet_name="../../outside.safetensors"),
            lambda s, g: g["8"]["inputs"].update(samples=["missing", 0]),
            lambda s, g: g["8"]["inputs"].update(samples=["47", 0]),
            lambda s, g: s["workflow"]["bindings"]["prompt"].update(node_id="37",
                                                                      input_name="unet_name"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ComfyPolicyError):
                revised_selection(self.selection, self.graph, change)

    def test_source_file_references_and_links_are_no_follow(self):
        with tempfile.TemporaryDirectory(prefix="cineforge-m23-input-") as tmp:
            root = Path(tmp).resolve()
            data = self.selection.declarations
            for name in ("../outside.json", "/outside.json", "nested/graph.json"):
                data["workflow"]["path"] = name
                path = root / "selection.json"
                path.write_text(json.dumps(data))
                with self.subTest(name=name), self.assertRaises(ComfyPolicyError):
                    load_comfy_selection(path)
            outside = root / "safe.json"
            outside.write_bytes(b"{}")
            linked = root / "linked.json"
            linked.symlink_to(outside)
            with self.assertRaises(ArtifactStoreError):
                read_private_input(linked)
            linked.unlink()
            os.link(outside, linked)
            with self.assertRaises(ArtifactStoreError):
                read_private_input(outside)

    def test_worker_requires_verified_pinned_environment_and_exact_model_hashes(self):
        for change in (lambda d: d.update(operator_verified=False),
                       lambda d: d.update(custom_nodes_disabled=False),
                       lambda d: d.update(server_revision="unverified"),
                       lambda d: d.update(image_digest="latest"),
                       lambda d: d["gpu"].update(memory_mib=1024),
                       lambda d: d["gpu"].update(backend="metal"),
                       lambda d: d["gpu"].update(name="other"),
                       lambda d: d["model_files"][0].update(sha256="a" * 64)):
            data = worker_record(self.selection)
            change(data)
            with self.subTest(data=data), self.assertRaises(ComfyPolicyError):
                WorkerVerification.from_dict(data).require_matches(self.selection)

    def test_measured_cuda_memory_is_never_rounded_up_or_used_for_another_gpu(self):
        record = worker_record(self.selection)
        record["gpu"]["memory_mib"] = 24082
        worker = WorkerVerification.from_dict(record)
        worker.require_matches(self.selection)
        self.assertEqual(worker.record["gpu"]["memory_mib"], 24082)
        record["gpu"]["memory_mib"] = 24081
        with self.assertRaises(ComfyPolicyError):
            WorkerVerification.from_dict(record).require_matches(self.selection)
        record["gpu"].update(memory_mib=24082, name="NVIDIA RTX 3090")
        with self.assertRaises(ComfyPolicyError):
            WorkerVerification.from_dict(record).require_matches(self.selection)


class ComfyRunTests(unittest.TestCase):
    def setUp(self):
        self.selection = load_comfy_selection(LEDGER)
        self.worker = WorkerVerification.from_dict(worker_record(self.selection))
        self.plan = plan_for(self.selection)
        self.temporary = tempfile.TemporaryDirectory(prefix="cineforge-m23-run-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "run"
        self.wire = WireFixture(self.selection)

    def execute(self, **changes):
        values = dict(allow_live=True, transport=self.wire, poll_interval_seconds=0)
        values.update(changes)
        return execute_comfy(self.plan, self.selection, self.worker, self.output, **values)

    def test_success_persists_verified_bytes_and_manifest_outside_git(self):
        manifest = self.execute()
        artifact = manifest.artifacts[0]
        payload = (self.output / artifact.uri).read_bytes()
        self.assertEqual(payload, PAYLOAD)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact.sha256)
        self.assertEqual(len(payload), artifact.size_bytes)
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["outcome"], "succeeded")
        self.assertEqual(RunManifest.from_dict(record["manifest"]), manifest)
        self.assertEqual(record["cloud_cost"]["status"], "not_measured")
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        for path in self.output.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)
        with self.assertRaises(ComfyRunError):
            self.execute()
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)

    def test_observed_upstream_float_coercions_persist_success_without_changing_template(self):
        # These exact five coercions occurred in the approved M2.4 server history.
        # Pinned execution.py:1008-1013 writes int()/float() casts back into inputs.
        fields = (("3", "cfg"), ("3", "denoise"), ("47", "crf"),
                  ("47", "fps"), ("48", "shift"))

        def coerce(graph):
            for node, field in fields:
                graph[node]["inputs"][field] = float(graph[node]["inputs"][field])

        original = self.selection.graph_json
        self.wire.history_graph_change = coerce
        manifest = self.execute()
        self.assertEqual(manifest.shot_runs[0].status, RunStatus.SUCCEEDED)
        self.assertEqual(self.selection.graph_json, original)
        for node, field in fields:
            self.assertIs(type(self.wire.graph[node]["inputs"][field]), int)
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)
        self.assertEqual((self.output / manifest.artifacts[0].uri).read_bytes(), PAYLOAD)

    def test_history_bool_numeric_string_structure_or_value_changes_still_fail(self):
        changes = (
            lambda g: g["3"]["inputs"].update(denoise=True),
            lambda g: g["3"]["inputs"].update(cfg=6.000000000000001),
            lambda g: g["3"]["inputs"].update(cfg="6.0"),
            lambda g: g["3"]["inputs"].update(extra=0),
            lambda g: g["3"]["inputs"].update(positive=[0, "6"]),
            lambda g: g["3"]["inputs"].update(positive=["6", 0, 0]),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.wire = WireFixture(self.selection)
                self.wire.history_graph_change = change
                self.output = self.root / f"altered-{index}"
                with self.assertRaises(ComfyRunError):
                    self.execute()
                journal = json.loads((self.output / "run.json").read_text())
                self.assertEqual(journal["outcome"], "incomplete")
                self.assertNotIn("manifest", journal)

    def test_default_opt_in_wrong_plan_or_endpoint_rejects_before_io(self):
        for options in ({"allow_live": False}, {"endpoint": "http://public.example"},
                        {"endpoint": "http://user:fixture-value@127.0.0.1:8188"},  # pragma: allowlist secret
                        {"timeout_seconds": float("nan")}, {"max_polls": True}):
            with self.subTest(options=options), self.assertRaises(ComfyRunError):
                self.execute(**options)
            self.assertEqual(self.wire.calls, [])
            self.assertFalse(self.output.exists())
        for shot in (replace(self.plan.shots[0], model_revision="wrong"),
                     replace(self.plan.shots[0], workflow_digest="0" * 64),
                     replace(self.plan.shots[0], prompt="embedding:../../unapproved")):
            self.plan = replace(self.plan, shots=(shot,))
            with self.assertRaises(ComfyRunError):
                self.execute()
            self.assertEqual(self.wire.calls, [])
        self.plan = plan_for(self.selection)
        self.plan = replace(self.plan, shots=self.plan.shots +
                            (replace(self.plan.shots[0], shot_id="shot-2"),))
        with self.assertRaises(ComfyRunError):
            self.execute()
        self.assertEqual(self.wire.calls, [])

    def test_prompt_is_opaque_data_and_only_bound_inputs_change(self):
        prompt = 'A paper boat; $(touch private-file) {"class_type":"RunCode"}\nLiteral text.'
        self.plan = replace(self.plan, shots=(replace(self.plan.shots[0], prompt=prompt, seed=17),))
        self.execute()
        self.assertEqual(self.wire.graph["6"]["inputs"]["text"], prompt)
        self.assertEqual(self.wire.graph["3"]["inputs"]["seed"], 17)
        expected = json.loads(self.selection.graph_json)
        expected["6"]["inputs"]["text"] = prompt
        expected["3"]["inputs"]["seed"] = 17
        self.assertEqual(expected, self.wire.graph)

    def test_node_preflight_fails_without_submission_or_installation(self):
        self.wire.schema_failure = True
        with self.assertRaises(ComfyRunError):
            self.execute()
        self.assertFalse(any(c[0] == "POST" for c in self.wire.calls))
        self.assertEqual(json.loads((self.output / "run.json").read_bytes())["outcome"], "incomplete")

    def test_failed_job_retains_known_identity_and_never_claims_success(self):
        self.wire.mode = "error"
        with self.assertRaises(ComfyRunError):
            self.execute()
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["known_run"]["job_id"], IDENTITY)
        self.assertEqual(record["known_run"]["status"], "failed")
        self.assertEqual(record["outcome"], "incomplete")
        self.assertNotIn("manifest", record)
        self.assertFalse(list(self.output.glob("artifact-*")))

    def test_unknown_job_does_not_blind_retry_or_fabricate_cancelled(self):
        self.wire.mode = "lost"
        with self.assertRaises(ComfyRunError):
            self.execute()
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["known_run"]["status"], "queued")
        self.assertFalse(record["cancellation"]["confirmed_terminal"])
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)
        self.assertFalse(any(c[1] == "/interrupt" for c in self.wire.calls))
        with self.assertRaises(ComfyRunError):
            self.execute()
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)

    def test_poll_budget_attempts_only_owned_job_cancel(self):
        self.wire.mode = "running"
        with self.assertRaises(ComfyRunError):
            self.execute(max_polls=1)
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["known_run"]["status"], RunStatus.CANCELLED.value)
        self.assertTrue(record["cancellation"]["confirmed_terminal"])
        self.assertEqual([c[1] for c in self.wire.calls if c[1].endswith("/cancel")],
                         [f"/api/jobs/{IDENTITY}/cancel"])

    def test_arbitrary_credential_error_has_no_message_traceback_or_record_leak(self):
        secret = "synthetic-private-value"  # pragma: allowlist secret - synthetic rejection fixture
        self.wire.failure = RuntimeError("Authorization: Bearer " + secret)
        with self.assertRaises(ComfyRunError) as caught:
            self.execute()
        self.assertNotIn(secret, "".join(traceback.format_exception(caught.exception)))
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(secret, (self.output / "run.json").read_text())

    def test_uncertain_submission_intent_is_durable_and_never_retried(self):
        def fail():
            raise RuntimeError("synthetic auth detail")
        self.wire.on_submit = fail
        with self.assertRaises(ComfyRunError):
            self.execute()
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertIsNone(record["known_run"])
        self.assertTrue(record["idempotency_key"])
        self.assertIn("translated_workflow", record)
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)

    def test_expired_wall_deadline_gets_separate_bounded_cleanup(self):
        now = [0.0]
        self.wire.mode = "running"

        def sleep(seconds):
            now[0] += seconds

        with self.assertRaises(ComfyRunError):
            self.execute(timeout_seconds=1, request_timeout_seconds=0.25,
                         poll_interval_seconds=1, clock=lambda: now[0], sleep=sleep)
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["known_run"]["status"], "cancelled")
        self.assertTrue(record["cancellation"]["confirmed_terminal"])
        self.assertTrue(all(0 < call[3] <= 0.25 for call in self.wire.calls))

    def test_connection_failure_after_submit_preserves_identity_without_error_leak(self):
        secret = "synthetic-private-status"  # pragma: allowlist secret - synthetic rejection fixture

        def submitted():
            self.wire.failure = RuntimeError("Bearer " + secret)

        self.wire.on_submit = submitted
        with self.assertRaises(ComfyRunError) as caught:
            self.execute()
        record = (self.output / "run.json").read_text()
        self.assertIn(IDENTITY, record)
        self.assertNotIn(secret, record)
        self.assertNotIn(secret, "".join(traceback.format_exception(caught.exception)))
        self.assertFalse(json.loads(record)["cancellation"]["confirmed_terminal"])

    def test_artifact_write_fault_never_emits_success_or_replays(self):
        original = os.link

        def linking(source, destination, **kwargs):
            if destination.startswith("artifact-"):
                raise OSError("synthetic artifact write fault")
            return original(source, destination, **kwargs)

        with patch("cineforge.private_store.os.link", side_effect=linking):
            with self.assertRaises(ComfyRunError):
                self.execute()
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["outcome"], "incomplete")
        self.assertEqual(record["known_run"]["status"], "succeeded")
        self.assertNotIn("manifest", record)
        self.assertFalse(list(self.output.glob("artifact-*")))
        with self.assertRaises(ComfyRunError):
            self.execute()
        self.assertEqual(sum(c[1] == "/prompt" for c in self.wire.calls), 1)

    def test_empty_download_cannot_publish_a_success_manifest(self):
        original = self.wire.request

        def request(method, path, body=None, **kwargs):
            return b"" if path.startswith("/view?") else original(method, path, body, **kwargs)

        with patch.object(self.wire, "request", side_effect=request):
            with self.assertRaises(ComfyRunError):
                self.execute()
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["outcome"], "incomplete")
        self.assertNotIn("manifest", record)

    def test_abrupt_process_death_preserves_job_and_refuses_fresh_submission(self):
        code = """
import os, sys
from test_comfy_run import LEDGER, WireFixture, worker_record, plan_for
from cineforge.comfy_policy import load_comfy_selection, WorkerVerification
from cineforge.comfy_run import execute_comfy
selection = load_comfy_selection(LEDGER)
class CrashingWire(WireFixture):
    def request(self, method, path, body=None, **kwargs):
        if path.startswith('/history/'):
            os._exit(73)
        return super().request(method, path, body, **kwargs)
execute_comfy(plan_for(selection), selection,
              WorkerVerification.from_dict(worker_record(selection)), sys.argv[1],
              allow_live=True, transport=CrashingWire(selection), poll_interval_seconds=0)
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))))
        result = subprocess.run([sys.executable, "-c", code, str(self.output)], env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 73, result.stderr)
        record = json.loads((self.output / "run.json").read_bytes())
        self.assertEqual(record["known_run"]["job_id"], IDENTITY)
        self.assertEqual(record["known_run"]["status"], "queued")
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        with self.assertRaises(ComfyRunError):
            self.execute()
        self.assertEqual(self.wire.calls, [])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                                  for p in self.output.iterdir()})


class PrivateStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cineforge-m23-store-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_traversal_git_and_symlink_ancestors_reject_before_output(self):
        repo = self.root / "repository"
        repo.mkdir()
        (repo / ".git").write_text("synthetic git marker")
        link = self.root / "linked"
        link.symlink_to(repo, target_is_directory=True)
        for path in (self.root / ".." / "outside", repo / "run", link / "run"):
            with self.subTest(path=path), self.assertRaises(ArtifactStoreError):
                with PrivateRunDirectory(path):
                    pass
        self.assertFalse((repo / "run").exists())

    def test_nonempty_or_already_claimed_destinations_are_never_overwritten(self):
        path = self.root / "run"
        path.mkdir()
        (path / "unrelated").write_bytes(b"preserve")
        with self.assertRaises(ArtifactStoreError):
            with PrivateRunDirectory(path):
                pass
        self.assertEqual((path / "unrelated").read_bytes(), b"preserve")
        (path / "unrelated").unlink()
        with PrivateRunDirectory(path):
            with self.assertRaises(ArtifactStoreError):
                with PrivateRunDirectory(path):
                    pass
        with self.assertRaises(ArtifactStoreError):
            with PrivateRunDirectory(path):
                pass

    def test_atomic_fault_keeps_valid_record_and_closes_claim_descriptor(self):
        path = self.root / "run"
        with PrivateRunDirectory(path) as store:
            store.checkpoint(b'{"stage":"old"}')
            for operation in ("fsync", "replace"):
                with patch("cineforge.private_store.os." + operation,
                           side_effect=OSError("synthetic failure")):
                    with self.assertRaises(ArtifactStoreError):
                        store.checkpoint(b'{"stage":"new"}')
                self.assertEqual((path / "run.json").read_bytes(), b'{"stage":"old"}')
                self.assertFalse(list(path.glob(".pending-*")))
        opened, closed = [], []
        real_open, real_close, real_fsync = os.open, os.close, os.fsync

        def opening(*args, **kwargs):
            descriptor = real_open(*args, **kwargs)
            opened.append(descriptor)
            return descriptor

        def closing(descriptor):
            closed.append(descriptor)
            return real_close(descriptor)

        def syncing(descriptor):
            if os.fstat(descriptor).st_mode & 0o170000 == 0o100000:
                raise OSError("synthetic claim fsync failure")
            return real_fsync(descriptor)

        with patch("cineforge.private_store.os.open", side_effect=opening), \
                patch("cineforge.private_store.os.close", side_effect=closing), \
                patch("cineforge.private_store.os.fsync", side_effect=syncing):
            with self.assertRaises(ArtifactStoreError):
                with PrivateRunDirectory(self.root / "claim-fault"):
                    pass
        self.assertCountEqual(opened, closed)

    def test_record_hardlink_or_symlink_replacement_is_rejected(self):
        for mode in ("hardlink", "symlink"):
            with self.subTest(mode=mode):
                path = self.root / mode
                outside = self.root / (mode + "-outside")
                with PrivateRunDirectory(path) as store:
                    store.checkpoint(b"original")
                    if mode == "hardlink":
                        os.link(path / "run.json", outside)
                    else:
                        outside.write_bytes(b"original")
                        (path / "run.json").unlink()
                        (path / "run.json").symlink_to(outside)
                    with self.assertRaises(ArtifactStoreError):
                        store.checkpoint(b"changed")
                    self.assertEqual(outside.read_bytes(), b"original")

    def test_payload_destination_never_overwrites_existing_file_or_links(self):
        selection = load_comfy_selection(LEDGER)
        request = selection.request(plan_for(selection).shots[0])
        provider = FakeExecutionProvider()
        run = provider.submit(request, "synthetic-artifact-fixture")
        provider.status(run.job_id)
        provider.status(run.job_id)
        result = provider.result(run.job_id)
        for mode in ("file", "hardlink", "symlink"):
            path = self.root / mode
            outside = self.root / (mode + "-outside")
            outside.write_bytes(b"preserve outside")
            with self.subTest(mode=mode), PrivateRunDirectory(path) as store:
                destination = path / f"artifact-{result.artifact.sha256}.webm"
                if mode == "file":
                    destination.write_bytes(b"unrelated")
                elif mode == "hardlink":
                    os.link(outside, destination)
                else:
                    destination.symlink_to(outside)
                with self.assertRaises(ArtifactStoreError):
                    store.put(result)
                self.assertEqual(outside.read_bytes(), b"preserve outside")
                if mode == "file":
                    self.assertEqual(destination.read_bytes(), b"unrelated")


class ComfyCLITests(unittest.TestCase):
    def test_all_parse_error_paths_omit_arbitrary_argument_values(self):
        credential = "https://synthetic-user:synthetic-private-token@example.invalid"  # pragma: allowlist secret - synthetic rejection fixture
        base = ["comfy-run", "missing-plan", "--selection", "missing-selection",
                "--worker-verification", "missing-worker", "--output-dir", "missing-output"]
        cases = [base + [flag, credential] for flag in (
            "--timeout-seconds", "--request-timeout-seconds", "--poll-interval-seconds",
            "--max-polls", "--max-result-bytes")]
        cases += [base + ["--unrecognized=" + credential], base + [credential],
                  ["comfy-run", credential, "--selection"],
                  ["comfy-run", credential, "--worker-verification", "missing-worker",
                   "--output-dir", "missing-output"],
                  [credential], ["--unknown-root", credential] + base,
                  base + ["--unexpected", "arbitrary-sensitive-value"]]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-m", "cineforge"] + arguments,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                for raw in (credential, "synthetic-private-token", "synthetic-user",
                            "arbitrary-sensitive-value"):
                    self.assertNotIn(raw, result.stderr)
                self.assertIn("invalid command arguments", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_default_is_offline_nonzero_and_does_not_leak_credentials(self):
        with tempfile.TemporaryDirectory(prefix="cineforge-m23-cli-") as tmp:
            root = Path(tmp).resolve()
            selection = load_comfy_selection(LEDGER)
            plan = root / "plan.json"
            plan.write_text(plan_for(selection).to_json())
            worker = root / "worker.json"
            worker.write_text(json.dumps(worker_record(selection)))
            command = [sys.executable, "-m", "cineforge", "comfy-run", str(plan),
                       "--selection", str(LEDGER), "--worker-verification", str(worker),
                       "--output-dir", str(root / "run")]
            result = subprocess.run(command, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertFalse((root / "run").exists())
            result = subprocess.run(command + ["--allow-live", "--endpoint",
                                    "http://user:fixture-private@127.0.0.1:8188"],  # pragma: allowlist secret
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("fixture-private", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
