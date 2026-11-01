"""Opt-in single-shot managed-cloud parity; not a general production scheduler."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from cineforge.cloud_budget import BudgetApproval, CloudBudgetSession
from cineforge.comfy import _history_graph_matches
from cineforge.comfy_policy import ApprovedComfySelection, WorkerVerification, load_comfy_selection
from cineforge.comfy_run import execute_comfy
from cineforge.comfy_transport import ComfyTransport
from cineforge.contracts import Project, ProjectPlan, RunManifest, RunStatus, Scene, Shot
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.runpod import (
    AcquisitionIntent,
    LeaseSnapshot,
    RunpodPodProvisioner,
    RunpodPodSpec,
    WorkerObservation,
)
from cineforge.runpod_transport import CloudTransport, RunpodHTTPTransport
from cineforge.ssh_worker import SSHWorker, private_file, run_process, ssh_public_key
from cineforge.worker_guard import WorkerGuardPlan, pod_exited
from cineforge.workflows import canonical, strict_object

# Public owner receipt binds these declarations; the coordinator authenticates it separately.
SCOPE_SHA256 = "9646e99a04b51064b08b75b56151370ef1a9df7def0f543bdb8faa70dbfc7349"  # pragma: allowlist secret
MODEL_WORKFLOW_SHA256 = "1aaa7e024932fe008bf0c13304ce80be66991d03a2b7d94aa4bd0f37987b15d0"  # pragma: allowlist secret
PROMPT = "A tiny red paper boat gently drifting across a calm pond, soft morning light, static camera."


class CloudSmokeError(ValueError):
    """Static diagnostic. Private run/budget records retain the actual known state."""


class _InjectedFailure(Exception):
    """Deliberate smoke-only exception after compatible guarded acquisition."""


def approved_plan(selection: ApprovedComfySelection) -> ProjectPlan:
    if selection.declarations["approval"]["model_workflow"][
            "approved_subject_sha256"] != MODEL_WORKFLOW_SHA256:
        raise CloudSmokeError("smoke requires the approved pinned model/workflow")
    return ProjectPlan(Project("m2-live-demo", 1, "Approved M2 real-shot validation"),
        (Scene("pond", "m2-live-demo", 1),),
        (Shot("paper-boat", "pond", 1, PROMPT, 42, selection.workflow.duration_seconds,
              selection.model.model_id, selection.model.revision, selection.workflow.digest),))


def probe_media(path: Path) -> dict[str, Any]:
    binary = shutil.which("ffprobe")
    if binary is None:
        raise CloudSmokeError("ffprobe is required before the live smoke")
    data = strict_object(run_process([binary, "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path)], timeout=10))
    streams = data["streams"]
    if not isinstance(streams, list) or len(streams) != 1:
        raise CloudSmokeError("single video stream required")
    stream = streams[0]
    rate = stream["avg_frame_rate"]
    if not isinstance(rate, str) or len(rate) > 40:
        raise CloudSmokeError("invalid measured video rate")
    fps, duration = float(Fraction(rate)), float(data["format"]["duration"])
    result = {"media_type": "video/webm", "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"], "width": stream["width"], "height": stream["height"],
        "decoded_frames": int(stream["nb_read_frames"]), "fps": fps,
        "duration_seconds": duration}
    require_media(result)
    return result


def require_media(data: dict[str, Any]) -> None:
    expected = {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
                "width": 832, "height": 480, "decoded_frames": 33, "fps": 16}
    if (any(type(data.get(name)) is bool or data.get(name) != value
            for name, value in expected.items())
            or type(data.get("duration_seconds")) not in (float, int)
            or not abs(data["duration_seconds"] - 33/16) < 1/16):
        raise CloudSmokeError("artifact technical contract differs from the approved shot")


def normalize(manifest: RunManifest, selection: ApprovedComfySelection, directory: Path, *,
              inspect_media: Callable[[Path], dict[str, Any]] = probe_media) -> dict[str, Any]:
    if (manifest.plan != approved_plan(selection) or len(manifest.shot_runs) != 1
            or len(manifest.artifacts) != 1 or manifest.shot_runs[0].status != RunStatus.SUCCEEDED
            or manifest.shot_runs[0].provider_id != "cineforge.comfy-v1"):
        raise CloudSmokeError("parity requires the same successful approved shot")
    artifact = manifest.artifacts[0]
    if artifact.uri != "artifact-" + artifact.sha256 + ".webm":
        raise CloudSmokeError("parity artifact must be private adjacent bytes")
    path = directory / artifact.uri
    payload = read_private_input(path, max_bytes=32*1024*1024)
    if len(payload) != artifact.size_bytes or hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise CloudSmokeError("parity artifact checksum or size differs")
    technical = inspect_media(path)
    require_media(technical)
    # Container rounding can differ within the validated frame interval. Compare the
    # normalized planned duration while retaining measured duration in lineage.
    contract = {"schema_version": 1, "plan": manifest.plan.to_dict(), "status": "succeeded",
        "media": {**technical, "duration_seconds": manifest.plan.shots[0].duration_seconds},
        "artifact_license": artifact.license,
        "model_workflow_subject_sha256": MODEL_WORKFLOW_SHA256}
    run = manifest.shot_runs[0]
    return {"contract": contract, "contract_sha256": hashlib.sha256(canonical(contract)).hexdigest(),
        "artifact": {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes,
            "origin": artifact.origin, "measured_duration_seconds": technical["duration_seconds"]},
        "volatile_execution": {"run_id": manifest.run_id, "provider_id": run.provider_id,
            "job_id": run.job_id, "started_at": run.started_at, "finished_at": run.finished_at,
            "manifest_cost_is_rental_invoice": False}}


@dataclass(frozen=True)
class ParityReference:
    """Validated private prepared-endpoint evidence, including independent byte checks."""
    normalized_json: str
    record_sha256: str
    provenance: str

    @property
    def normalized(self) -> dict[str, Any]:
        return strict_object(self.normalized_json)


def load_reference(path: str | Path, selection: ApprovedComfySelection, *,
                   inspect_media: Callable[[Path], dict[str, Any]] = probe_media) -> ParityReference:
    path = Path(path)
    payload = read_private_input(path)
    record = strict_object(payload)
    manifest = RunManifest.from_dict(record["manifest"])
    worker = WorkerVerification.from_dict(record["worker_verification"])
    worker.require_matches(selection)
    expected = selection.workflow.translate(selection.request(approved_plan(selection).shots[0]))
    recovered = (record.get("outcome") == "operator_recovered"
                 and record.get("mode") == "manual_get_only_operator_reconciliation")
    if (not (record.get("outcome") == "succeeded" or recovered)
            or record["plan"] != manifest.plan.to_dict()
            or not _history_graph_matches(expected, record["translated_workflow"])):
        raise CloudSmokeError("prepared reference differs from approved execution")
    normalized = normalize(manifest, selection, path.parent, inspect_media=inspect_media)
    normalized["worker_image_digest"] = worker.record["image_digest"]
    provenance = "operator_recovered_existing_job" if recovered \
        else "normal_prepared_endpoint"
    return ParityReference(canonical(normalized).decode(), hashlib.sha256(payload).hexdigest(),
                           provenance)


class ManagedWorker(Protocol):
    def __enter__(self) -> ManagedWorker: ...
    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None: ...
    def probe(self, snapshot: LeaseSnapshot, timeout: float) -> WorkerObservation | None: ...
    def prepare(self) -> WorkerVerification: ...
    def endpoint(self) -> str: ...
    def execution_transport(self) -> ComfyTransport: ...
    def check(self) -> None: ...
    def set_enforcer(self, enforce: Callable[[], tuple[str, ...]]) -> None: ...
    def suspend_remote_heartbeat(self) -> WorkerGuardPlan: ...


WorkerFactory = Callable[[CloudBudgetSession, Path], ManagedWorker]


def _provider(spec: RunpodPodSpec, transport: CloudTransport, budget: CloudBudgetSession,
              probe: Callable[[LeaseSnapshot, float], WorkerObservation | None]) \
        -> RunpodPodProvisioner:
    def progress(intent: AcquisitionIntent, snapshot: LeaseSnapshot) -> None:
        budget.event_sink(intent, snapshot)
        if snapshot.state in {"created", "ready"}:
            budget.heartbeat(snapshot.key)
    return RunpodPodProvisioner(spec, transport, probe=probe, before_create=budget.before_create,
        event_sink=progress, lease_seconds=1800, startup_timeout_seconds=240,
        request_timeout_seconds=30, cleanup_timeout_seconds=20, max_polls=300)


def _observe_guard_stop(worker: ManagedWorker, provider: RunpodPodProvisioner,
                        budget: CloudBudgetSession, transport: CloudTransport,
                        key: str, report: dict[str, Any], *,
                        clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
    plan = worker.suspend_remote_heartbeat()
    if plan != budget.guard_plan(key):
        raise CloudSmokeError("injected heartbeat-loss plan differs from owned lease")
    intent = provider.intent(key)
    started = clock()
    wait = plan.idle_seconds + 20
    if budget.remaining_seconds(key) < wait + 10:
        raise CloudSmokeError("insufficient lease time for bounded guardian observation")
    report["guard_observation"] = {"remote_heartbeat_withdrawn": True,
        "local_supervision_continues": True, "wait_limit_seconds": wait,
        "observed_status": None, "runtime_absent": False}
    while clock() - started < wait:
        worker.check()  # Real enforcer stays active; local heartbeat cannot revive expired leases.
        budget.heartbeat(key)
        # Acquisition proved the immutable lease; each read rechecks its unguessable
        # owner nonce before interpreting state. Deletion performs full verification.
        snapshot = next(s for s in provider.snapshots if s.key == key)
        if snapshot.resource_id is None:
            raise CloudSmokeError("guard observation lost its known identity")
        reply = transport.request("GET", "/v2/pods/" + snapshot.resource_id,
            timeout_seconds=min(3, wait-(clock()-started), budget.remaining_seconds(key)),
            max_bytes=1024*1024)
        if reply.status != 200:
            raise CloudSmokeError("independent worker stop was not observed")
        pod = strict_object(reply.body)
        env = pod.get("env")
        if (pod.get("id") != plan.pod_id or not isinstance(env, dict)
                or env.get("CINEFORGE_SESSION") != plan.session_id
                or env.get("CINEFORGE_LEASE") != plan.lease_token
                or pod.get("image") != intent.spec.image or pod.get("mounts") != {}
                or pod.get("dataCenterId") != intent.spec.data_center_id):
            raise CloudSmokeError("guard observation ownership differs")
        report["guard_observation"].update(observed_status=pod.get("status")
            if pod.get("status") in {"RUNNING", "EXITED", "ERROR", "TERMINATED"} else "unknown",
            runtime_absent=pod.get("runtime") is None,
            elapsed_seconds=clock()-started)
        if pod_exited(pod.get("status")):
            report["guard_observation"]["stop_status_basis"] = "observed_lifecycle_EXITED"
            report["guard_stop_verified"] = True
            return
        sleep(min(2, max(0, wait-(clock()-started))))
    raise CloudSmokeError("independent worker stop was not confirmed within its grace")


def run_managed_smoke(selection: ApprovedComfySelection, approval: BudgetApproval,
                      spec: RunpodPodSpec, transport: CloudTransport, budget_dir: str | Path,
                      output_dir: str | Path, reference: ParityReference, worker_factory: WorkerFactory,
                      *, case: str = "success", allow_live: bool = False,
                      inspect_media: Callable[[Path], dict[str, Any]] = probe_media,
                      execute: Callable[..., RunManifest] = execute_comfy,
                      clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """One attempt, always cleanup; no automatic live job/submission resumption."""
    if (allow_live is not True or case not in {"success", "post-acquire-failure"}
            or not isinstance(reference, ParityReference)
            or approval.model != selection.model
            or reference.normalized["contract"]["plan"] != approved_plan(selection).to_dict()
            or approval.subject_sha256 != SCOPE_SHA256
            or approval.model_workflow_sha256 != MODEL_WORKFLOW_SHA256
            or reference.normalized["worker_image_digest"] != approval.image.split("@", 1)[1]):
        raise CloudSmokeError("managed smoke preflight rejected")
    report: dict[str, Any] = {"schema_version": 1, "case": case, "outcome": "prepared",
        "reference_record_sha256": reference.record_sha256,
        "reference_provenance": reference.provenance, "reference": reference.normalized,
        "approved_scope_sha256": approval.subject_sha256, "release_verified": False,
        "costs": None, "leases": [], "guard_stop_verified": False}
    failed = False
    output_dir = Path(output_dir)
    try:
        with PrivateRunDirectory(output_dir) as store:
            store.checkpoint(canonical(report))
            with CloudBudgetSession(budget_dir, approval, transport,
                                    guard_deployment_configured=True) as budget:
                provider = None
                worker = None
                manifest = None
                try:
                    if any(intent.spec.digest != spec.digest for intent in budget.intents):
                        raise CloudSmokeError("budget journal belongs to another session/spec")
                    worker = worker_factory(budget, output_dir / "ssh")
                    with worker:
                        provider = _provider(spec, transport, budget, worker.probe)
                        worker.set_enforcer(lambda: budget.enforce_limits(provider))
                        device = provider.acquire(selection.model, max_cost_usd=.51)
                        report.update(outcome="acquired", worker_id=device.device_id)
                        store.checkpoint(canonical(report))
                        if case == "post-acquire-failure":
                            _observe_guard_stop(worker, provider, budget, transport,
                                provider.snapshots[-1].key, report, clock=clock, sleep=sleep)
                            raise _InjectedFailure
                        else:
                            verification = worker.prepare()
                            worker.check()
                            key = provider.snapshots[-1].key
                            timeout = min(600, budget.remaining_seconds(key))
                            if timeout <= 0:
                                raise CloudSmokeError("lease cutoff reached before execution")
                            manifest = execute(approved_plan(selection), selection, verification,
                                output_dir / "shot", allow_live=True, endpoint=worker.endpoint(),
                                timeout_seconds=timeout, transport=worker.execution_transport())
                            worker.check()
                            report.update(outcome="generated", manifest=manifest.to_dict())
                except _InjectedFailure:
                    report["outcome"] = "injected_post_acquire_failure"
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
                if not failed and manifest is not None:
                    try:
                        # Inspect local bytes after confirmed release; ffprobe never
                        # prolongs GPU rental or the worker's cleanup margin.
                        normalized = normalize(manifest, selection, output_dir / "shot",
                                               inspect_media=inspect_media)
                        if normalized["contract"] != reference.normalized["contract"]:
                            raise CloudSmokeError("managed/prepared normalized contract mismatch")
                        report.update(outcome="succeeded", managed=normalized)
                    except (Exception, KeyboardInterrupt):
                        failed = True
                        report["outcome"] = "incomplete"
                    store.checkpoint(canonical(report))
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed:
        raise CloudSmokeError("managed smoke incomplete; inspect private run and budget records")
    return report


def recover_managed(spec: RunpodPodSpec, approval: BudgetApproval, transport: CloudTransport,
                    budget_dir: str | Path, *, allow_live: bool = False) -> dict[str, Any]:
    """Discover/release only existing journal-owned leases; never calls acquire/probe."""
    if allow_live is not True or approval.subject_sha256 != SCOPE_SHA256:
        raise CloudSmokeError("recovery preflight rejected")
    failed = False
    report = None
    try:
        with CloudBudgetSession(budget_dir, approval, transport) as budget:
            if any(intent.spec.digest != spec.digest for intent in budget.intents):
                raise CloudSmokeError("recovery journal spec differs")
            def no_probe(snapshot: LeaseSnapshot, timeout: float) -> WorkerObservation | None:
                raise CloudSmokeError("recovery must not probe or execute")
            provider = _provider(spec, transport, budget, no_probe)
            budget.recover_owned(provider)
            report = {"schema_version": 1, "mode": "recovery_only",
                "leases": [s.to_dict() for s in budget.snapshots], "costs": budget.costs(),
                "uncertain_identities": budget.uncertain_identities,
                "release_verified": all(s.state in {"released", "not_created", "rejected"}
                                        for s in budget.snapshots)}
            if not report["release_verified"]:
                failed = True
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or report is None:
        raise CloudSmokeError("owned recovery incomplete; retain private budget journal")
    return report


def external_api_key(path: str | None = None) -> str:
    value = private_file(Path(path).expanduser(), max_bytes=1024).decode("ascii").strip() if path \
        else os.environ.get("RUNPOD_API_KEY", "")
    if re.fullmatch(r"[A-Za-z0-9._-]{8,512}", value) is None:
        raise CloudSmokeError("external programmatic Runpod credential unavailable")
    return value


def cli_inputs(selection_path: str, proposal_path: str, session_id: str, private_key: str,
               key_file: str | None, *, allow_live: bool = False) \
        -> tuple[ApprovedComfySelection, BudgetApproval, RunpodPodSpec, RunpodHTTPTransport]:
    if allow_live is not True:
        raise CloudSmokeError("cloud command requires explicit live opt-in")
    selection = load_comfy_selection(selection_path)
    approval = BudgetApproval.from_proposal(strict_object(read_private_input(proposal_path)),
        expected_subject_sha256=SCOPE_SHA256,
        expected_model_workflow_sha256=MODEL_WORKFLOW_SHA256,
        model=selection.model, selection=selection)
    spec = RunpodPodSpec(session_id, approval.image, approval.gpu_id, approval.region,
                         ssh_public_key(Path(private_key).expanduser()))
    return selection, approval, spec, RunpodHTTPTransport(external_api_key(key_file), allow_live=True)


def recovery_inputs(selection_path: str, proposal_path: str, budget_dir: str, key_file: str | None,
                    *, allow_live: bool = False) \
        -> tuple[BudgetApproval, RunpodPodSpec, RunpodHTTPTransport]:
    if allow_live is not True:
        raise CloudSmokeError("recovery requires explicit live opt-in")
    selection = load_comfy_selection(selection_path)
    approval = BudgetApproval.from_proposal(strict_object(read_private_input(proposal_path)),
        expected_subject_sha256=SCOPE_SHA256,
        expected_model_workflow_sha256=MODEL_WORKFLOW_SHA256,
        model=selection.model, selection=selection)
    # Refuse a mistyped/empty path instead of creating a fresh recovery journal.
    private_file(Path(budget_dir) / "budget.json", max_bytes=1024*1024, read=False)
    transport = RunpodHTTPTransport(external_api_key(key_file), allow_live=True)
    with CloudBudgetSession(budget_dir, approval, transport) as budget:
        if not budget.intents:
            raise CloudSmokeError("recovery journal has no retained intents")
        spec = budget.intents[0].spec
        if any(intent.spec.digest != spec.digest for intent in budget.intents):
            raise CloudSmokeError("recovery journal contains differing session/specs")
    return approval, spec, transport


def ssh_factory(spec: RunpodPodSpec, selection: ApprovedComfySelection, transport: CloudTransport,
                private_key: str) -> WorkerFactory:
    return lambda budget, state: SSHWorker(spec, selection, budget, transport,
        Path(private_key).expanduser(), state, allow_live=True)
