"""One guarded Pod, three sequential I2V shots, private evidence and separate human review."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from cineforge import conditioning_profile as profile
from cineforge.cloud_budget import CloudBudgetSession, ConditioningBudgetApproval
from cineforge.cloud_journal import CloudJournal
from cineforge.cloud_smoke import probe_media
from cineforge.comfy_policy import WorkerVerification
from cineforge.comfy_run import execute_comfy
from cineforge.conditioning import (
    ConditioningError,
    ConditioningSelection,
    VerifiedConditioning,
    digest,
)
from cineforge.contracts import RunManifest
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.runpod import AcquisitionIntent, LeaseSnapshot, RunpodPodProvisioner, RunpodPodSpec
from cineforge.runpod_transport import CloudTransport
from cineforge.workflows import canonical, strict_object


class ConditioningWorker(Protocol):
    runtime_evidence: dict[str, Any] | None
    def __enter__(self) -> Any: ...
    def __exit__(self, *args: Any) -> None: ...
    def probe(self, snapshot: LeaseSnapshot, timeout: float) -> Any: ...
    def prepare(self) -> WorkerVerification: ...
    def stage_reference(self) -> dict[str, Any]: ...
    def endpoint(self) -> str: ...
    def execution_transport(self) -> Any: ...
    def check(self) -> None: ...
    def set_enforcer(self, enforce: Callable[[], tuple[str, ...]]) -> None: ...


WorkerFactory = Callable[[CloudBudgetSession, Path], ConditioningWorker]


def prepare_journal(gate: VerifiedConditioning) -> None:
    """Offline initialization only; existing state is never reset or refunded.

    Same-account administrators can delete all local state. This is not tamperproof
    or account-wide spending enforcement; preserve the owner-private directory.
    """
    if type(gate) is not VerifiedConditioning:
        raise ConditioningError("M4 journal requires its verified proposal")
    subject_sha256 = digest(gate.subject)
    with CloudJournal(gate.journal_directory) as journal:
        if journal.initial is None:
            journal.checkpoint(canonical({"schema_version": 1, "approval_sha256": subject_sha256,
                                         "last_wall": time.time(), "records": {}}))
        else:
            data = strict_object(journal.initial)
            if data.get("approval_sha256") != subject_sha256:
                raise ConditioningError("existing M4 journal belongs to a different proposal")


def _provider(spec: RunpodPodSpec, transport: CloudTransport, budget: CloudBudgetSession,
              probe: Callable[..., Any], selection: ConditioningSelection | None = None) \
        -> RunpodPodProvisioner:
    def progress(intent: AcquisitionIntent, snapshot: LeaseSnapshot) -> None:
        budget.event_sink(intent, snapshot)
        if snapshot.state in {"created", "ready"}:
            budget.heartbeat(snapshot.key)

    def before_create(intent: AcquisitionIntent) -> None:
        if selection is None:
            raise ConditioningError("recovery cannot acquire a worker")
        selection.gate.preflight()
        budget.before_create(intent)

    return RunpodPodProvisioner(spec, transport, probe=probe, before_create=before_create,
        event_sink=progress, lease_seconds=1800, startup_timeout_seconds=240,
        request_timeout_seconds=30, cleanup_timeout_seconds=20, max_polls=300)


def _require_spec(spec: RunpodPodSpec, region: str) -> None:
    if region not in profile.REGIONS:
        raise ConditioningError("M4 requires one explicitly approved data center")
    if ((spec.image, spec.gpu_id, spec.data_center_id, spec.disk_gb,
         spec.min_cuda_version, spec.max_hourly_usd, spec.registry_id, spec.min_ram_per_gpu_gb) !=
            (profile.IMAGE, profile.GPU_ID, region, 80, "12.8", 1.25, None,
             profile.MIN_RAM_PER_GPU_GB)):
        raise ConditioningError("worker spec differs from the closed M4 scope")


def technical_result(manifest: RunManifest, directory: Path,
                     inspect_media: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    if (len(manifest.artifacts) != 1 or len(manifest.shot_runs) != 1
            or manifest.shot_runs[0].status.value != "succeeded"):
        raise ConditioningError("incomplete M4 output")
    artifact = manifest.artifacts[0]
    if artifact.uri != "artifact-" + artifact.sha256 + ".webm":
        raise ConditioningError("M4 output must be private adjacent WebM bytes")
    path = directory / artifact.uri
    payload = read_private_input(path, max_bytes=32*1024*1024)
    if (len(payload) != artifact.size_bytes
            or hashlib.sha256(payload).hexdigest() != artifact.sha256):
        raise ConditioningError("M4 output checksum differs")
    media = inspect_media(path)
    expected = {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
                "width": 1280, "height": 704, "decoded_frames": 49, "fps": 24.0}
    measured = media.get("duration_seconds")
    duration = float(measured) if isinstance(measured, (int, float)) and not isinstance(
        measured, bool) else float("nan")
    if (any(type(media.get(k)) is bool or media.get(k) != v for k, v in expected.items())
            or not profile.DURATION - 1/24 <= duration <= profile.DURATION + 1/24):
        raise ConditioningError("M4 output media contract differs")
    return {"manifest": manifest.to_dict(), "media": media,
            "identity_review": "pending_human_review"}


def run_conditioned(selection: ConditioningSelection, spec: RunpodPodSpec,
                    transport: CloudTransport, budget_dir: str | Path, output_dir: str | Path,
                    worker_factory: WorkerFactory, *, allow_live: bool = False,
                    execute: Callable[..., RunManifest] = execute_comfy,
                    inspect_media: Callable[[Path], dict[str, Any]] = probe_media,
                    clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Opt-in execution only. No retries; success never means human identity approval."""
    if allow_live is not True or type(selection) is not ConditioningSelection:
        raise ConditioningError("M4 execution requires a checked selection and explicit opt-in")
    _require_spec(spec, selection.gate.region)
    selection.gate.preflight()  # All three exact requests and current rights before any create.
    approval = ConditioningBudgetApproval.for_selection(selection)
    report: dict[str, Any] = {"schema_version": 1, "mode": "m4-three-shot",
        "approved_subject_sha256": approval.subject_sha256, "outcome": "prepared",
        "identity_review": "pending_human_review", "release_verified": False,
        "shots": [], "costs": None, "leases": []}
    failed = False
    output_dir = Path(output_dir)
    try:
        with PrivateRunDirectory(output_dir) as store:
            store.checkpoint(canonical(report))
            with CloudBudgetSession(budget_dir, approval, transport,
                                    guard_deployment_configured=True) as budget:
                provider = None
                manifests: list[tuple[RunManifest, Path]] = []
                try:
                    if budget.intents:
                        raise ConditioningError("M4 run requires a fresh scoped journal; recover first")
                    worker = worker_factory(budget, output_dir / "ssh")
                    with worker:
                        provider = _provider(spec, transport, budget, worker.probe, selection)
                        worker.set_enforcer(lambda: budget.enforce_limits(provider))
                        device = provider.acquire(selection.model, max_cost_usd=.64)
                        key = provider.snapshots[-1].key
                        report.update(outcome="acquired", worker_id=device.device_id)
                        store.checkpoint(canonical(report))
                        verification = worker.prepare()
                        if verification.record["worker_id"] != device.device_id:
                            raise ConditioningError("prepared worker differs from the owned acquisition")
                        selection.gate.preflight()
                        worker.check()
                        report["staged_reference"] = worker.stage_reference()
                        if report["staged_reference"] != selection.gate.reference.worker_metadata:
                            raise ConditioningError("worker staged different reference bytes")
                        report["runtime"] = worker.runtime_evidence
                        report["worker_verification"] = verification.record
                        selection = selection.bind_worker(verification, report["staged_reference"],
                                                          worker.runtime_evidence)
                        # Reserve cleanup/retrieval time; do not begin another shot after cutoff.
                        previous_seconds = 300.0
                        for shot in selection.gate.requests:
                            worker.check()
                            trace = selection.gate.trace(shot)  # Fresh ledger/catalog/bytes per shot.
                            remaining = budget.remaining_seconds(key)
                            if remaining < previous_seconds + 120:
                                raise ConditioningError("insufficient lease for next bounded shot")
                            started = clock()
                            directory = output_dir / shot.shot_id
                            manifest = execute(selection.plan(shot), selection, verification,
                                directory, allow_live=True, endpoint=worker.endpoint(),
                                timeout_seconds=min(600, remaining-120),
                                transport=worker.execution_transport())
                            worker.check()
                            previous_seconds = max(30, clock() - started)
                            manifests.append((manifest, directory))
                            translated = selection.workflow.translate(selection.request(shot))
                            report["shots"].append({"shot_id": shot.shot_id, "lineage": trace,
                                "translated_graph_sha256": hashlib.sha256(canonical(translated)).hexdigest(),
                                "manifest": manifest.to_dict()})
                            report["outcome"] = "generating"
                            store.checkpoint(canonical(report))
                except (Exception, KeyboardInterrupt):
                    failed = True
                    report["outcome"] = "incomplete"
                finally:
                    if provider is not None:
                        for snapshot in provider.snapshots:
                            if snapshot.resource_id is not None and snapshot.state != "released":
                                try:
                                    provider.release(snapshot.resource_id)
                                except (Exception, KeyboardInterrupt):
                                    failed = True
                    report["leases"] = [s.to_dict() for s in budget.snapshots]
                    report["release_verified"] = (bool(budget.snapshots)
                        and all(s.state == "released" for s in budget.snapshots))
                    report["costs"] = budget.costs()
                    if not report["release_verified"]:
                        failed = True
                    store.checkpoint(canonical(report))
                if not failed:
                    try:
                        if len(manifests) != 3:
                            raise ConditioningError("three outputs are required")
                        media = [technical_result(m, d, inspect_media) for m, d in manifests]
                        if len({m.artifacts[0].sha256 for m, _ in manifests}) != 3:
                            raise ConditioningError("M4 requires three distinct output artifacts")
                        report.update(outcome="pending_human_review", technical_results=media)
                    except (Exception, KeyboardInterrupt):
                        failed = True
                        report["outcome"] = "incomplete"
                    store.checkpoint(canonical(report))
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed:
        raise ConditioningError("M4 run incomplete; retain private run and budget journals")
    return report


def recover_conditioned(approval: ConditioningBudgetApproval, transport: CloudTransport,
                        budget_dir: str | Path, *, allow_live: bool = False) -> dict[str, Any]:
    """Owned recovery does not need reference bytes/current consent or create/probe anything."""
    if allow_live is not True or type(approval) is not ConditioningBudgetApproval:
        raise ConditioningError("M4 recovery requires its checked approval and explicit opt-in")
    if str(Path(budget_dir).absolute()) != approval.journal_directory:
        raise ConditioningError("M4 recovery path differs from the approved journal")
    read_private_input(Path(budget_dir) / "budget.json")  # Reject missing path; never start fresh.
    failed = False
    result = None
    try:
        with CloudBudgetSession(budget_dir, approval, transport) as budget:
            if not budget.intents:
                raise ConditioningError("M4 recovery journal has no retained intents")
            spec = budget.intents[0].spec
            _require_spec(spec, approval.region)
            if any(intent.spec.digest != spec.digest for intent in budget.intents):
                raise ConditioningError("M4 recovery journal contains differing worker specs")
            def no_probe(*args: Any) -> Any:
                raise ConditioningError("recovery cannot probe a worker")
            provider = _provider(spec, transport, budget, no_probe)
            budget.recover_owned(provider)
            released = all(s.state in {"released", "not_created", "rejected"}
                           for s in budget.snapshots)
            result = {"mode": "m4-recovery-only", "release_verified": released,
                "leases": [s.to_dict() for s in budget.snapshots], "costs": budget.costs(),
                "uncertain_identities": budget.uncertain_identities}
            if not released:
                raise ConditioningError("M4 owned recovery remains uncertain")
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or result is None:
        raise ConditioningError("M4 recovery incomplete; retain private budget journal")
    return result
