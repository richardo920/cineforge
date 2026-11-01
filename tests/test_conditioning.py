"""Offline M4 gates; PNGs are generated synthetic fixtures, outputs are not video."""

import base64
import copy
import hashlib
import json
import os
import struct
import tempfile
import time
import traceback
import unittest
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_cloud_budget import Clock, Cloud
from test_cloud_budget import approval as legacy_budget
from test_cloud_budget import selection as legacy_selection
from test_comfy_run import WireFixture, plan_for, worker_record
from test_runpod import spec as legacy_spec

from cineforge import conditioning_profile as profile
from cineforge import worker_bootstrap as bootstrap
from cineforge.characters import (
    CharacterRegistry,
    CharacterRevision,
    ConditioningHint,
    ReferenceArtifact,
    RightsBinding,
    Wardrobe,
    with_character_review,
)
from cineforge.cloud_budget import (
    BudgetApproval,
    CloudBudgetError,
    CloudBudgetSession,
    ConditioningBudgetApproval,
)
from cineforge.comfy import ComfyExecutionProvider
from cineforge.comfy_policy import ComfyPolicyError, WorkerVerification
from cineforge.comfy_run import ComfyRunError, execute_comfy
from cineforge.conditioning import (
    ConditioningApproval,
    ConditioningError,
    PrivateReference,
    VerifiedConditioning,
    digest,
    load_context,
    model_subject,
    selection_from_approval,
    shots,
)
from cineforge.conditioning_cli import command as conditioning_command
from cineforge.conditioning_run import prepare_journal, recover_conditioned, run_conditioned
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.providers import ExecutionRequest, ProviderError
from cineforge.rights import EvidenceRef, RightsLedger, RightsRecord, RightsRef, SubjectRef, Use
from cineforge.rights import with_review as review_rights
from cineforge.runpod import WorkerObservation
from cineforge.skills import (
    CatalogEntry,
    Compatibility,
    Permissions,
    SkillCatalog,
    SourcePin,
    SourceReview,
    bind_shot,
)
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.worker_guard import GuardReceipt, worker_guard_source_sha256
from cineforge.workflows import canonical

EVIDENCE = EvidenceRef("private://synthetic/review", "f"*64)


def png():
    def chunk(kind, value):
        return (struct.pack(">I", len(value)) + kind + value
                + struct.pack(">I", zlib.crc32(kind+value) & 0xffffffff))
    raw = (b"\0" + bytes((20, 140, 150))*1280)*704
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1280, 704, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def chunk(kind, value):
    return (struct.pack(">I", len(value)) + kind + value
            + struct.pack(">I", zlib.crc32(kind+value) & 0xffffffff))


def grant(ref, subject, *, consent=False):
    value = RightsRecord(ref, subject, source=EVIDENCE, source_revision="v1",
        license_id="LicenseRef-Synthetic", terms=EVIDENCE, notices=("Synthetic fixture notice",),
        attribution="Synthetic fixture author", consent=EVIDENCE if consent else None,
        allowed_uses=(Use("reference"), Use("condition"), Use("execute")))
    if subject.kind == "model":
        value = replace(value, license_id="Apache-2.0", source_revision=profile.MODEL_REVISION,
            source=EvidenceRef(profile.MODEL_CARD_URI, profile.MODEL_CARD_SHA256),
            terms=EvidenceRef(profile.MODEL_TERMS_URI, profile.MODEL_TERMS_SHA256))
    if subject.kind == "workflow":
        value = replace(value, license_id="MIT", source_revision=profile.SOURCE_REVISION,
            source=EvidenceRef(profile.SOURCE_REPOSITORY + "/blob/" + profile.SOURCE_REVISION + "/"
                               + profile.SOURCE_PATH, profile.SOURCE_SHA256),
            terms=EvidenceRef("urn:synthetic:pinned-mit-license",
                              hashlib.sha256(profile.WORKFLOW_LICENSE).hexdigest()))
    return review_rights(value, reviewer="synthetic-reviewer", evidence=EVIDENCE)


def context(root):
    payload = png()
    path = root / "reference.png"
    path.write_bytes(payload)
    reference_hash = hashlib.sha256(payload).hexdigest()
    image_subject = SubjectRef("character_reference", "synthetic-front", "v1", reference_hash)
    reference = ReferenceArtifact("front", "identity", EvidenceRef("private://synthetic/front",
        reference_hash), len(payload), "image/png", RightsBinding(image_subject,
        RightsRef("reference-permission", "v1")))
    graph_digest = digest(profile.graph(reference_hash))
    character = with_character_review(CharacterRevision("synthetic-robot", 1, "fictional",
        "Synthetic geometric robot", Wardrobe("scarf", "Synthetic orange scarf"), (reference,),
        conditioning=(ConditioningHint(profile.MODE, profile.MODEL_ID, profile.MODEL_REVISION,
            graph_digest, ("front",)),)), reviewer="synthetic-reviewer", evidence=EVIDENCE)
    registry = CharacterRegistry()
    registry.register(character)
    source = SourcePin(profile.SOURCE_REPOSITORY, profile.SOURCE_REVISION,
                       profile.SOURCE_PATH, profile.SOURCE_SHA256)
    entry = CatalogEntry("wan22-workflow", "r1", "workflow", "Synthetic reviewed metadata",
        source, source.path, Permissions(), Compatibility(), RightsRef("workflow-rights", "v1"))
    entry = replace(entry, review=SourceReview(entry.digest, "synthetic-reviewer",
                                             "urn:synthetic:review", "f"*64))
    records = (grant(reference.permission.rights, image_subject, consent=True),
        grant(entry.rights, entry.subject), grant(RightsRef("model-rights", "v1"), model_subject()))
    ledger = RightsLedger()
    for record in records:
        ledger.register(record)
    requests = shots(character.pin(), graph_digest)
    bindings = tuple(bind_shot(shot, (entry.pin,)) for shot in requests)
    gate = VerifiedConditioning(PrivateReference(reference, path), registry, ledger,
        SkillCatalog((entry,)), bindings, character.pin(), records[-1].ref, str(root / "budget"))
    return gate, records, entry, character


def selected(gate):
    # Synthetic assertion only, never a real owner grant.
    return selection_from_approval(gate, {"status": "approved", "subject_sha256": digest(gate.subject),
        "approved_by": "synthetic-owner", "receipt":
        "https://github.com/richardo920/cineforge/issues/60#issuecomment-1"})


def technical(_path):
    return dict(media_type="video/webm", codec="vp9", pixel_format="yuv420p",
                width=1280, height=704, decoded_frames=49, fps=24.,
                duration_seconds=profile.DURATION)


def runtime_fixture():
    resources = bootstrap.memory_accounting(512*1024**3, 128*1024**3,
        [{"limit_bytes": 64_000_000_000, "usage_bytes": 8_000_000_000}], "v2")
    return dict(resources, python_version="3.12.12", torch_version="2.8.0+cu128",
        torchvision_version="0.23.0+cu128", cuda_version="12.8", gpu_raw_name=profile.GPU_ID,
        gpu_memory_mib=32286, cuda_available=True, arch_list=["sm_120"], capability=[12, 0],
        device_count=1, cuda_kernel_ok=True, nvml_total_mib=32768, nvml_reserved_mib=482,
        driver_version="595.91.07", filesystem_total_bytes=80_000_000_000,
        filesystem_free_bytes=50_000_000_000)


class Wire(WireFixture):
    def request(self, method, path, body=None, **bounds):
        if path.startswith("/view?"):
            return b"synthetic output, not video:" + str(self.graph["3"]["inputs"]["seed"]).encode()
        raw = super().request(method, path, body, **bounds)
        if path == "/object_info/LoadImage":
            value = json.loads(raw)
            value["LoadImage"]["input"]["required"]["image"] = [[
                self.template["56"]["inputs"]["image"]]]
            return canonical(value)
        return raw


class Worker:
    def __init__(self, budget, selection, spec, mode="success"):
        self.budget, self.selection, self.spec, self.mode = budget, selection, spec, mode
        self.runtime_evidence = runtime_fixture()
        self.events = []
        self.wire = Wire(selection)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.events.append("closed")

    def set_enforcer(self, enforce):
        self.enforce = enforce

    def check(self):
        if self.enforce():
            raise ConditioningError("synthetic lease expired")

    def probe(self, snapshot, timeout):
        self.budget.arm_guard(snapshot.key, GuardReceipt(self.budget.guard_plan(snapshot.key),
            worker_guard_source_sha256(), 123, time.time(), True, True))
        self.identity = snapshot.resource_id
        self.events.append("guarded")
        return WorkerObservation(GPUDevice(self.identity, Backend.CUDA, 32768,
            frozenset({Precision.FP16})), profile.GPU_ID)

    def prepare(self):
        self.events.append("prepared")
        if self.mode == "setup_failure":
            raise TimeoutError("synthetic-sensitive-marker")
        record = worker_record(self.selection)
        record.update(worker_id=self.identity, image_digest=profile.IMAGE.split("@", 1)[1])
        if self.mode == "stale_worker":
            record["worker_id"] = "other-worker"
        record["gpu"]["memory_mib"] = 32286
        record["torch_version"] = "2.8.0+cu128"
        return WorkerVerification.from_dict(record)

    def stage_reference(self):
        self.events.append("staged")
        return self.selection.gate.reference.worker_metadata

    def endpoint(self):
        return "http://127.0.0.1:8188"

    def execution_transport(self):
        return self.wire


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cineforge-m42-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.gate, self.records, self.entry, self.character = context(self.root)

    def context_file(self):
        data = dict(character=self.character.to_dict(), rights=[r.to_dict() for r in self.records],
            revocations=[], catalog=[self.entry.to_dict()],
            bindings=[b.to_dict() for b in self.gate.bindings], model_rights=self.records[-1].ref.to_dict(),
            journal_directory=self.gate.journal_directory)
        path = self.root / "context.json"
        path.write_bytes(canonical(data))
        return path


class AdmissionTests(Fixture):
    def test_public_proposal_contains_only_exact_unapproved_model_profile(self):
        root = Path(__file__).resolve().parents[1]
        proposal = json.loads((root / "config/comfy/wan22-ti2v5b-proposal.json").read_text())
        self.assertEqual(proposal["state"], "proposed_not_approved")
        self.assertFalse(proposal["readiness"]["live_ready"])
        self.assertNotIn("approval", proposal)
        actual = [{k: value[k] for k in expected} for value, expected in
                  zip(proposal["model"]["files"], profile.model_files())]
        self.assertEqual(actual, profile.model_files())
        self.assertEqual((root / "config/comfy/WAN22-WORKFLOW-LICENSE").read_bytes(),
                         profile.WORKFLOW_LICENSE)

    def test_real_registry_and_catalog_on_same_shot_translate_reference_and_trace(self):
        chosen = selected(self.gate)
        for shot in self.gate.requests:
            graph = chosen.workflow.translate(chosen.request(shot))
            self.assertEqual(graph["55"]["inputs"]["start_image"], ["56", 0])
            self.assertEqual(graph["56"]["inputs"]["image"], self.gate.reference.basename)
            self.assertEqual(graph["3"]["inputs"]["seed"], shot.seed)
            self.assertEqual(graph["55"]["inputs"]["length"], 49)
            trace = self.gate.trace(shot)
            self.assertEqual(trace["character"], self.character.to_dict())
            self.assertEqual(trace["catalog"], [self.entry.to_dict()])
            self.assertEqual(trace["skills"]["shot_sha256"], hashlib.sha256(
                shot.to_json().encode()).hexdigest())
            self.assertEqual(len(trace["rights"]), 3)
        self.assertEqual(sum(size for _, size, _ in profile.FILES), 18144966705)

    def test_old_selection_and_lower_level_t2v_reject_character_before_transport(self):
        old = legacy_selection()
        shot = replace(plan_for(old).shots[0], character_revision=self.character.pin().to_key())
        with self.assertRaises(ComfyPolicyError):
            old.request(shot)
        wire = Mock()
        with self.assertRaises(ProviderError):
            ComfyExecutionProvider(wire, old.workflow).submit(ExecutionRequest(shot, old.model), "one")
        wire.request.assert_not_called()
        with self.assertRaises(ComfyRunError):
            execute_comfy(replace(plan_for(old), shots=(shot,)), old,
                WorkerVerification.from_dict(worker_record(old)), self.root / "not-created",
                allow_live=True, transport=wire)
        self.assertFalse((self.root / "not-created").exists())

    def test_current_character_catalog_and_model_rights_are_rechecked(self):
        for record in self.records:
            with self.subTest(kind=record.subject.kind):
                gate, records, _, _ = context(self.root)
                chosen = selected(gate)
                grant_to_revoke = next(r for r in records if r.subject.kind == record.subject.kind)
                gate.ledger.revoke(grant_to_revoke.ref, reason="Synthetic withdrawal")
                with self.assertRaises(ConditioningError):
                    chosen.request(gate.requests[0])

    def test_graph_shot_and_capability_changes_fail_closed(self):
        chosen = selected(self.gate)
        shot = self.gate.requests[0]
        for changed in (replace(shot, prompt="changed"), replace(shot, seed=99),
                        replace(shot, character_revision=None), replace(shot, model_id="unsupported")):
            with self.assertRaises((ValueError, ProviderError)):
                chosen.request(changed)
        graph = copy.deepcopy(self.gate.graph)
        del graph["55"]["inputs"]["start_image"]
        for cap in (self.gate, True, {"approved": True}):
            workflow = replace(chosen.workflow, graph_json=canonical(graph).decode(), conditioning=cap)
            wrong = replace(shot, workflow_digest=workflow.digest)
            with self.assertRaises((ValueError, ProviderError)):
                workflow.translate(ExecutionRequest(wrong, profile.model()))
        with self.assertRaises(ComfyPolicyError):
            replace(chosen, graph_json=canonical(graph).decode())

    def test_submission_requires_this_workers_verified_runtime_and_reference_receipt(self):
        chosen = selected(self.gate)
        shot = self.gate.requests[0]
        wire = Mock()
        with self.assertRaises(ConditioningError):
            ComfyExecutionProvider(wire, chosen.workflow).submit(chosen.request(shot), "unstaged")
        wire.request.assert_not_called()
        record = worker_record(chosen)
        record.update(image_digest=profile.IMAGE.split("@", 1)[1], torch_version="2.8.0+cu128")
        record["gpu"]["memory_mib"] = 32286
        worker = WorkerVerification.from_dict(record)
        bound = chosen.bind_worker(worker, self.gate.reference.worker_metadata, runtime_fixture())
        with self.assertRaises(ConditioningError):
            ComfyExecutionProvider(wire, bound.workflow, verified_worker_id="other-worker").submit(
                bound.request(shot), "wrong-worker")
        wire.request.assert_not_called()
        with self.assertRaises(ConditioningError):
            chosen.bind_worker(worker, dict(self.gate.reference.worker_metadata, sha256="b"*64),
                               runtime_fixture())

    def test_reference_bytes_and_paths_are_verified_without_echo(self):
        marker = self.root / "sensitive-marker"
        marker.write_bytes(png())
        for name in ("sym", "hard"):
            path = self.root / name
            if name == "sym":
                path.symlink_to(marker)
            else:
                os.link(marker, path)
            with self.assertRaises(ConditioningError):
                replace(self.gate.reference, path=path).read()
        self.gate.reference.path.write_bytes(png()[:-1] + b"x")
        try:
            self.gate.preflight()
        except ConditioningError as error:
            self.assertNotIn("sensitive-marker", "".join(traceback.format_exception(error)))
        else:
            self.fail("changed reference bytes accepted")

    def test_unknown_png_chunks_crc_bombs_and_trailing_bytes_are_rejected(self):
        animated = png()[:33] + chunk(b"acTL", struct.pack(">II", 2, 0)) + png()[33:]
        bomb = (png()[:33] + chunk(b"IDAT", zlib.compress(b"x"*((1280*3+1)*704+1)))
                + chunk(b"IEND", b""))
        for value in (png()+b"extra", animated, bomb, png()[:-3],
                      b"\x89PNG\r\n\x1a\n"+b"x"*128*1024):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.validate_reference_png(value)
        bootstrap.validate_reference_png(png())

    def test_approval_is_separate_exact_and_never_fabricated_by_preflight(self):
        subject = self.gate.subject
        self.assertNotIn("approved_by", subject)
        record = {"status": "approved", "subject_sha256": digest(subject),
                  "approved_by": "synthetic-owner", "receipt":
                  "https://github.com/richardo920/cineforge/issues/48#issuecomment-1"}
        with self.assertRaises(ConditioningError):
            selection_from_approval(self.gate, record)
        with self.assertRaises(ConditioningError):
            ConditioningApproval(canonical(dict(record, status="proposed")).decode(),
                                 canonical(subject).decode())

    def test_private_context_roundtrip_and_current_source_binding(self):
        path = self.context_file()
        data = json.loads(path.read_bytes())
        self.assertEqual(load_context(path, self.gate.reference.path).subject, self.gate.subject)
        data["catalog"][0]["source"]["revision"] = "b"*40
        path.write_bytes(canonical(data))
        with self.assertRaises(ConditioningError):
            load_context(path, self.gate.reference.path)

    def test_offline_cli_proposes_only_and_initializes_persistent_scope_without_keys(self):
        args = SimpleNamespace(command="conditioning-preflight", context=str(self.context_file()),
            reference=str(self.gate.reference.path), output_dir=str(self.root / "proposal"))
        with patch("cineforge.conditioning_cli.external_api_key") as key, \
                patch("cineforge.conditioning_cli.RunpodHTTPTransport") as transport:
            report = conditioning_command(args)
        key.assert_not_called()
        transport.assert_not_called()
        self.assertEqual(report["status"], "proposed")
        stored = json.loads((self.root / "proposal/run.json").read_bytes())
        self.assertEqual(stored["subject"], self.gate.subject)
        self.assertNotIn("approval", stored)
        self.assertEqual(json.loads((self.root / "budget/budget.json").read_bytes())["records"], {})
        args.command = "conditioning-run"
        args.allow_live = False
        with patch("cineforge.conditioning_cli._approval") as read, \
                patch("cineforge.conditioning_cli.ssh_public_key") as ssh:
            with self.assertRaises(ConditioningError):
                conditioning_command(args)
        read.assert_not_called()
        ssh.assert_not_called()

    def test_model_terms_cannot_be_replaced_by_a_reviewed_unknown_license(self):
        model_record = self.records[-1]
        ledger = RightsLedger()
        for record in self.records[:-1]:
            ledger.register(record)
        changed = review_rights(replace(model_record, license_id="LicenseRef-Other", review=None),
                               reviewer="synthetic-reviewer", evidence=EVIDENCE)
        ledger.register(changed)
        with self.assertRaises(ConditioningError):
            replace(self.gate, ledger=ledger).preflight()


class ManagedTests(Fixture):
    def run_case(self, name="run", mode="success", **options):
        chosen = selected(self.gate)
        spec = replace(legacy_spec(), image=profile.IMAGE, gpu_id=profile.GPU_ID,
                       data_center_id=profile.REGION, max_hourly_usd=1.25)
        cloud = options.pop("cloud", Cloud(Clock()))
        cloud.catalog.update(id=profile.GPU_ID, price=dict(secure=1.19), memory=32,
                             dataCenters=[dict(id=profile.REGION, availability="LOW")])
        self.cloud, self.workers = cloud, []
        def factory(budget, state):
            worker = Worker(budget, chosen, spec, mode)
            self.workers.append(worker)
            return worker
        values = dict(allow_live=True, inspect_media=technical)
        values.update(options)
        if values["allow_live"]:
            prepare_journal(self.gate)  # Explicit offline proposal initialization, never a grant.
        return run_conditioned(chosen, spec, cloud, self.root / "budget", self.root / name,
                               factory, **values)

    def test_three_shots_stage_once_and_release_then_wait_for_human_review(self):
        report = self.run_case()
        self.assertEqual(report["outcome"], "pending_human_review")
        self.assertTrue(report["release_verified"])
        self.assertEqual(len(report["shots"]), 3)
        self.assertEqual(self.workers[0].events, ["guarded", "prepared", "staged", "closed"])
        self.assertEqual(sum(method == "POST" for method, *_ in self.cloud.calls), 1)
        self.assertEqual(sum(method == "DELETE" for method, *_ in self.cloud.calls), 1)
        self.assertEqual(report["costs"]["attempts"], 1)
        self.assertEqual(report["costs"]["reserved_worst_case_usd"], .635)
        self.assertIsNone(report["costs"]["actual_total_usd"])
        self.assertEqual(len({s["translated_graph_sha256"] for s in report["shots"]}), 3)

    def test_no_optin_or_revoked_context_never_touches_cloud(self):
        with self.assertRaises(ConditioningError):
            self.run_case(allow_live=False)
        self.assertEqual(self.cloud.calls, [])
        self.assertFalse((self.root / "run").exists())

    def test_setup_and_mid_shot_failure_release_without_retry(self):
        with self.assertRaises(ConditioningError):
            self.run_case(mode="setup_failure")
        self.assertFalse(self.cloud.pods)
        self.assertNotIn("staged", self.workers[0].events)
        self.assertEqual(sum(c[0] == "POST" for c in self.cloud.calls), 1)
        report = json.loads((self.root / "run/run.json").read_bytes())
        self.assertEqual(report["outcome"], "incomplete")
        self.assertTrue(report["release_verified"])

    def test_stale_prepared_worker_rejected_before_staging(self):
        with self.assertRaises(ConditioningError):
            self.run_case(mode="stale_worker")
        self.assertNotIn("staged", self.workers[0].events)
        self.assertFalse(self.cloud.pods)

    def test_revocation_between_outputs_blocks_later_submissions_and_cleans_up(self):
        def execute(*args, **kwargs):
            result = execute_comfy(*args, **kwargs)
            self.gate.ledger.revoke(self.records[0].ref, reason="Synthetic withdrawal")
            return result
        with self.assertRaises(ConditioningError):
            self.run_case(execute=execute)
        self.assertFalse(self.cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in self.workers[0].wire.calls), 1)

    def test_remaining_lease_estimate_never_starts_another_shot(self):
        with self.assertRaises(ConditioningError):
            self.run_case(clock=Mock(side_effect=[0., 2000.]))
        self.assertFalse(self.cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in self.workers[0].wire.calls), 1)

    def test_technical_failure_after_release_never_passes_human_review(self):
        with self.assertRaises(ConditioningError):
            self.run_case(inspect_media=lambda path: dict(technical(path), decoded_frames=48))
        self.assertFalse(self.cloud.pods)
        report = json.loads((self.root / "run/run.json").read_bytes())
        self.assertEqual(report["outcome"], "incomplete")
        self.assertEqual(report["identity_review"], "pending_human_review")
        self.assertTrue(report["release_verified"])

    def test_cross_scope_budget_and_journal_cannot_be_reused(self):
        budget = ConditioningBudgetApproval.for_selection(selected(self.gate))
        for change in ({"gpu_id": legacy_spec().gpu_id}, {"region": "US-IL-1"},
                       {"aggregate_usd": 3}, {"max_attempts": 2}, {"lifetime_seconds": 1801}):
            with self.assertRaises(CloudBudgetError):
                replace(budget, **change)
        with self.assertRaises(CloudBudgetError):
            BudgetApproval(budget.subject_sha256, budget.model_workflow_sha256, budget.model,
                budget.image, budget.gpu_id, budget.region, budget.min_cuda, 1, 1, 1, 1800, 1.25, 80)
        old = legacy_budget()
        cloud = Cloud(Clock())
        with CloudBudgetSession(self.root / "old-budget", old, cloud):
            pass
        with self.assertRaises(CloudBudgetError):
            with CloudBudgetSession(self.root / "old-budget", budget, cloud):
                pass

    def test_unknown_create_retained_recovery_without_reference_bytes(self):
        cloud = Cloud(Clock())
        cloud.create_lost = True
        with self.assertRaises(ConditioningError):
            self.run_case(cloud=cloud)
        self.assertEqual(len(cloud.pods), 1)
        self.assertEqual(self.workers[0].events, ["closed"])
        budget = ConditioningBudgetApproval.for_selection(selected(self.gate))
        self.gate.ledger.revoke(self.records[0].ref, reason="Synthetic processing withdrawal")
        self.gate.reference.path.unlink()
        cloud.create_lost = False
        report = recover_conditioned(budget, cloud, self.root / "budget", allow_live=True)
        self.assertTrue(report["release_verified"])
        self.assertFalse(cloud.pods)
        self.assertEqual(sum(c[0] == "POST" for c in cloud.calls), 1)

    def test_missing_or_alternate_journal_cannot_automatically_refund_attempt(self):
        budget = ConditioningBudgetApproval.for_selection(selected(self.gate))
        cloud = Cloud(Clock())
        with self.assertRaises(CloudBudgetError):
            with CloudBudgetSession(self.root / "missing-new-path", budget, cloud):
                pass
        with self.assertRaises(CloudBudgetError):
            with CloudBudgetSession(self.gate.journal_directory, budget, cloud):
                pass
        # Interrupted first initialization is retained, not guessed to be fresh.
        with self.assertRaises(ValueError):
            prepare_journal(self.gate)
        self.assertEqual(cloud.calls, [])


class BootstrapTests(Fixture):
    def test_m4_profile_is_closed_and_legacy_8gib_download_limit_stays(self):
        metadata = selected(self.gate).worker_profile
        self.assertEqual(bootstrap.checked_profile(metadata), metadata)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.checked_profile({**metadata, "id": "arbitrary"})
        for path, size, sha in profile.FILES:
            self.assertEqual(bootstrap.M4_FILES[path], (size, sha))
        asset = profile.model_files()[0]
        self.assertGreater(asset["size_bytes"], 8*1024**3)
        with patch.object(bootstrap.urllib.request, "build_opener") as opener:
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.download(self.root, asset, time.monotonic()+1)
            opener.assert_not_called()
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.download(self.root, dict(asset, sha256="a"*64), time.monotonic()+1,
                                   profile=metadata)
            opener.assert_not_called()

    def test_stage_exact_bytes_once_and_reject_legacy_or_changed_reference(self):
        metadata = selected(self.gate).worker_profile
        target = self.root / "ComfyUI/input"
        target.mkdir(parents=True)
        request = dict(profile=metadata, payload=base64.b64encode(png()).decode())
        result = bootstrap.stage_reference(request, self.root)
        self.assertEqual(result, {"reference": metadata["reference"]})
        self.assertEqual((target / self.gate.reference.basename).read_bytes(), png())
        with self.assertRaises((bootstrap.BootstrapError, FileExistsError)):
            bootstrap.stage_reference(request, self.root)
        for changed in ({**request, "profile": None}, {**request, "payload": "eA=="}):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.stage_reference(changed, self.root)

    def test_reference_stage_rejects_symlinked_ancestor(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "input").mkdir()
        (self.root / "ComfyUI").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.stage_reference(dict(profile=selected(self.gate).worker_profile,
                payload=base64.b64encode(png()).decode()), self.root)
        self.assertEqual(list((outside / "input").iterdir()), [])

    def test_m4_runtime_requires_blackwell_kernel_not_host_ram(self):
        metadata = selected(self.gate).worker_profile
        raw = runtime_fixture()
        resources = {k: raw[k] for k in bootstrap.M4_RESOURCE_FIELDS}
        runtime = {k: value for k, value in raw.items() if k not in bootstrap.M4_RESOURCE_FIELDS
                   and k not in {"nvml_total_mib", "nvml_reserved_mib", "driver_version"}}
        with patch.object(bootstrap, "bounded_process", side_effect=[canonical(runtime),
                b"32768, 482, 595.91.07\n"]), \
                patch.object(bootstrap, "resource_evidence", return_value=resources):
            actual = bootstrap.runtime_probe(1, profile=metadata)
            self.assertEqual(actual["gpu_memory_mib"], 32286)
            self.assertEqual(actual["nvml_reserved_mib"], 482)
        for change in ({"gpu_memory_mib": 41}, {"gpu_raw_name": bootstrap.GPU},
                       {"arch_list": ["sm_89"]}, {"cuda_kernel_ok": False},
                       {"torch_version": "2.6.0+cu124"}, {"capability": [8, 9]},
                       {"device_count": 2}):
            with patch.object(bootstrap, "bounded_process", return_value=canonical(dict(runtime, **change))):
                with self.assertRaises(bootstrap.BootstrapError):
                    bootstrap.runtime_probe(1, profile=metadata)

    def test_decimal64gb_cgroup_allocation_and_available_headroom_are_distinct(self):
        actual = runtime_fixture()
        self.assertEqual(actual["memory_limit_bytes"], 64_000_000_000)
        self.assertLess(actual["memory_limit_bytes"], 64*1024**3)
        bootstrap.validate_m4_runtime(actual)
        self.assertEqual(profile.RESOURCE_POLICY["minimum_effective_ram_limit_bytes"],
                         actual["memory_limit_bytes"])
        for groups in ([{"limit_bytes": 32_000_000_000, "usage_bytes": 0}],
                       [{"limit_bytes": 64_000_000_000, "usage_bytes": 40_000_000_000}]):
            # Huge host RAM does not bypass a smaller allocation or charged parent group.
            memory = bootstrap.memory_accounting(512*1024**3, 128*1024**3, groups, "v2")
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.validate_m4_runtime(dict(actual, **memory))
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.validate_m4_runtime(dict(actual, filesystem_free_bytes=29*1024**3))

    def test_resource_probe_applies_parent_cgroup_and_memavailable_without_swap(self):
        proc, cgroup = self.root / "proc", self.root / "cgroup"
        (proc / "self").mkdir(parents=True)
        proc.joinpath("meminfo").write_text("MemTotal: 536870912 kB\nMemAvailable: 134217728 kB\n"
                                         "SwapFree: 999999999999 kB\n")
        proc.joinpath("self/cgroup").write_text("0::/parent/child\n")
        for subdir, limit, usage in (("parent/child", "max", 1_000_000_000),
                                    ("parent", 64_000_000_000, 40_000_000_000)):
            path = cgroup / subdir
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath("memory.max").write_text(str(limit))
            path.joinpath("memory.current").write_text(str(usage))
            path.joinpath("cgroup.procs").write_text(str(os.getpid()) if subdir.endswith("child")
                                                     else "99999999")
        disk = SimpleNamespace(total=80_000_000_000, free=50_000_000_000)
        with patch.object(bootstrap.shutil, "disk_usage", return_value=disk):
            evidence = bootstrap.resource_evidence(self.root, proc=proc, cgroup=cgroup)
        self.assertEqual(evidence["memory_limit_bytes"], 64_000_000_000)
        self.assertEqual(evidence["available_memory_bytes"], 24_000_000_000)
        self.assertEqual(len(evidence["cgroup_limits"]), 2)
        self.assertEqual(evidence["filesystem_free_bytes"], 50_000_000_000)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.validate_m4_runtime(dict(runtime_fixture(), **evidence))

    def test_ssh_stage_uncertainty_cannot_retry_or_write_another_reference(self):
        # Construct only the pre-network admission surface; no keys or sockets are used.
        worker = object.__new__(SSHWorker)
        worker.selection = selected(self.gate)
        worker._profile = worker.selection.worker_profile
        worker.plan = Mock()
        worker.plan.to_dict.return_value = {}
        worker._reference_staged = False
        worker.budget = Mock()
        worker.budget.remaining_seconds.return_value = 300
        worker._key = "synthetic"
        worker._request = Mock(side_effect=TimeoutError("synthetic marker"))
        with self.assertRaises(TimeoutError):
            worker.stage_reference()
        with self.assertRaises(WorkerError):
            worker.stage_reference()
        worker._request.assert_called_once()
