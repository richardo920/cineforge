"""Durable single-host admission and recovery policy; construction performs no cloud calls."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, Callable
from urllib.parse import quote

from cineforge.cloud_journal import CloudJournal
from cineforge.comfy_policy import ApprovedComfySelection
from cineforge.devices import ModelSpec
from cineforge.runpod import AcquisitionIntent, LeaseSnapshot, RunpodPodProvisioner
from cineforge.runpod_transport import CloudResponse, CloudTransport
from cineforge.worker_guard import GuardReceipt, WorkerGuardPlan, worker_guard_source_sha256
from cineforge.workflows import canonical, strict_object


class CloudBudgetError(ValueError):
    """Admission, journal, or recovery failed closed; diagnostics contain no source payloads."""


def _number(value: object, *, zero: bool = False) -> bool:
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and (value >= 0 if zero else value > 0))
    except OverflowError:
        return False


@dataclass(frozen=True)
class BudgetApproval:
    """A scope binding, not authentication. Caller independently verifies the owner receipt."""

    subject_sha256: str
    model_workflow_sha256: str
    model: ModelSpec
    image: str
    gpu_id: str
    region: str
    min_cuda: str
    aggregate_usd: float
    max_attempts: int
    max_concurrent: int
    lifetime_seconds: float
    hourly_ceiling_usd: float
    disk_gb: int

    def __post_init__(self) -> None:
        if (not isinstance(self.subject_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.subject_sha256) is None
                or self.subject_sha256 == "0" * 64
                or not isinstance(self.model_workflow_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.model_workflow_sha256) is None
                or self.model_workflow_sha256 == "0" * 64
                or not isinstance(self.model, ModelSpec)
                or type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 2
                or type(self.max_concurrent) is not int or self.max_concurrent != 1
                or type(self.disk_gb) is not int or self.disk_gb != 80
                or not _number(self.aggregate_usd) or self.aggregate_usd > 3
                or not _number(self.lifetime_seconds) or self.lifetime_seconds > 1800
                or not _number(self.hourly_ceiling_usd) or self.hourly_ceiling_usd > 1
                or self.gpu_id != "NVIDIA GeForce RTX 4090" or self.region != "US-IL-1"
                or not isinstance(self.image, str)
                or re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", self.image) is None
                or not isinstance(self.min_cuda, str)
                or re.fullmatch(r"\d{1,2}\.\d{1,2}", self.min_cuda) is None
                or tuple(map(int, self.min_cuda.split("."))) < (12, 8)):
            raise CloudBudgetError("invalid bounded budget approval")

    @classmethod
    def from_proposal(cls, proposal: dict[str, Any], *, expected_subject_sha256: str,
                      expected_model_workflow_sha256: str, model: ModelSpec,
                      selection: ApprovedComfySelection) -> BudgetApproval:
        """Compare the operational proposal with the exact independently approved subject."""
        failed = False
        result = None
        try:
            approval = proposal["approval"]["m3_live_spend"]
            subject = approval["approved_subject"]
            operational = proposal["proposed_validation"]
            recommendation = proposal["recommendation"]
            digest = hashlib.sha256(canonical(subject)).hexdigest()
            if (proposal["state"] != "owner_scope_approved" or approval["status"] != "approved"
                    or digest != expected_subject_sha256
                    or approval["approved_subject_sha256"] != digest
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_model_workflow_sha256)
                    or not isinstance(model, ModelSpec)
                    or not isinstance(selection, ApprovedComfySelection)
                    or selection.model != model
                    or selection.declarations["approval"]["model_workflow"][
                        "approved_subject_sha256"] != expected_model_workflow_sha256
                    or subject["model_workflow_subject_sha256"] != expected_model_workflow_sha256
                    or re.fullmatch(r"https://github\.com/richardo920/cineforge/issues/48"
                                    r"#issuecomment-[0-9]+", approval["receipt"]) is None):
                raise ValueError
            fixed = {"provider": "runpod", "product": "pod", "cloud": "SECURE",
                     "gpu_count": 1, "region_fallback": False, "max_concurrent_pods": 1,
                     "persistent_volume_gb": 0, "network_volume_ids": [], "ports": ["22/tcp"],
                     "startup_and_downloads_count_toward_lifetime": True,
                     "synthetic_data_only": True}
            for name, expected in fixed.items():
                if canonical(subject[name]) != canonical(expected):
                    raise ValueError
            for name in ("provider", "product", "cloud"):
                if recommendation[name] != subject[name]:
                    raise ValueError
            for name in ("gpu_id", "gpu_count", "region", "region_fallback", "image",
                         "min_host_cuda", "max_concurrent_pods", "max_create_attempts",
                         "worker_lifetime_limit_seconds", "startup_and_downloads_count_toward_lifetime",
                         "compute_price_ceiling_usd_per_hour", "container_disk_gb",
                         "persistent_volume_gb", "network_volume_ids", "ports"):
                # JSON numeric representation can differ (1 vs 1.0); bool must not equal 1.
                a, b = subject[name], operational[name]
                if isinstance(a, bool) or isinstance(b, bool):
                    if type(a) is not type(b) or a != b:
                        raise ValueError
                elif a != b:
                    raise ValueError
            for name in ("gpu_count", "max_concurrent_pods", "max_create_attempts",
                         "container_disk_gb", "persistent_volume_gb"):
                if type(operational[name]) is not int:
                    raise ValueError
            if (operational["actual_worker_verification_required"] is not True
                    or type(subject["max_create_attempts"]) is not int
                    or not 1 <= subject["max_create_attempts"] <= 2
                    or type(subject["max_concurrent_pods"]) is not int
                    or type(subject["container_disk_gb"]) is not int
                    or subject["container_disk_gb"] != 80
                    or subject["gpu_id"] != "NVIDIA GeForce RTX 4090"
                    or subject["region"] != "US-IL-1"
                    or not _number(subject["aggregate_limit_usd"])
                    or subject["aggregate_limit_usd"] > 3
                    or not _number(approval["aggregate_limit_usd"])
                    or approval["aggregate_limit_usd"] != subject["aggregate_limit_usd"]
                    or not _number(subject["worker_lifetime_limit_seconds"])
                    or subject["worker_lifetime_limit_seconds"] > 1800
                    or not _number(subject["compute_price_ceiling_usd_per_hour"])
                    or subject["compute_price_ceiling_usd_per_hour"] > 1
                    or not isinstance(subject["image"], str)
                    or re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}",
                                    subject["image"]) is None
                    or not isinstance(subject["min_host_cuda"], str)
                    or re.fullmatch(r"\d{1,2}\.\d{1,2}", subject["min_host_cuda"]) is None
                    or tuple(map(int, subject["min_host_cuda"].split("."))) < (12, 8)):
                raise ValueError
            result = cls(digest, expected_model_workflow_sha256, model, subject["image"],
                         subject["gpu_id"], subject["region"], subject["min_host_cuda"],
                         float(subject["aggregate_limit_usd"]), subject["max_create_attempts"],
                         1, float(subject["worker_lifetime_limit_seconds"]),
                         float(subject["compute_price_ceiling_usd_per_hour"]), 80)
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise CloudBudgetError("proposal does not match independently approved scope")
        return result

    def admits(self, intent: AcquisitionIntent) -> bool:
        spec = intent.spec
        return (intent.requirements == self.model and spec.image == self.image
                and spec.gpu_id == self.gpu_id and spec.data_center_id == self.region
                and spec.min_cuda_version == self.min_cuda and spec.disk_gb == self.disk_gb
                and spec.registry_id is None and spec.max_hourly_usd <= self.hourly_ceiling_usd
                and intent.startup_timeout_seconds + 60 < intent.lease_seconds
                and intent.lease_seconds <= self.lifetime_seconds)


_TERMINAL = {"released", "rejected", "not_created"}
_TRANSITIONS = {
    "reserved": {"reserved", "created", "not_created", "rejected", "create_uncertain"},
    "created": {"created", "ready", "cleanup_pending", "cleanup_uncertain", "released"},
    "ready": {"ready", "cleanup_pending", "cleanup_uncertain", "released"},
    "create_uncertain": {"create_uncertain", "cleanup_pending", "cleanup_uncertain", "released"},
    "cleanup_pending": {"cleanup_pending", "cleanup_uncertain", "released"},
    "cleanup_uncertain": {"cleanup_pending", "cleanup_uncertain", "released"},
    "released": {"released"}, "rejected": {"rejected"}, "not_created": {"not_created"},
}
_RECORD_FIELDS = {"intent", "snapshot", "authorized", "reservation_usd", "estimate_usd", "quote",
                  "reserved_at", "deadline", "last_activity", "guard", "actual_usd",
                  "actual_reference", "discovered_ids"}


class CloudBudgetSession:
    """Hold an exclusive private journal through all driver calls and recovery.

    Only hooks wired to this object authorize paid creates. The separately installed
    worker guard and heartbeat bridge must be configured before construction enables
    admission; READY additionally requires an exact verified guard receipt.
    """

    def __init__(self, path: str | Path, approval: BudgetApproval, transport: CloudTransport, *,
                 guard_deployment_configured: bool = False,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 idle_seconds: float = 120, quote_max_age_seconds: float = 30) -> None:
        if (not isinstance(approval, BudgetApproval) or type(guard_deployment_configured) is not bool
                or not _number(idle_seconds) or idle_seconds > 300
                or not _number(quote_max_age_seconds) or quote_max_age_seconds > 30):
            raise CloudBudgetError("invalid cloud budget configuration")
        self.approval, self.transport = approval, transport
        self.clock, self.monotonic = clock, monotonic
        self.idle_seconds, self.quote_max_age_seconds = idle_seconds, quote_max_age_seconds
        self.guard_deployment_configured = guard_deployment_configured
        self._journal = CloudJournal(path)
        self._lock = RLock()
        self._data: dict[str, Any] = {}
        self._poisoned = False
        self._last_mono = 0.0
        self._effective = 0.0
        self._last_wall = 0.0
        self._entered = False
        self._cleanup_inflight: set[str] = set()

    def __enter__(self) -> CloudBudgetSession:
        self._journal.__enter__()
        failed = False
        try:
            wall, mono = self.clock(), self.monotonic()
            if not _number(wall, zero=True) or not _number(mono, zero=True):
                raise ValueError
            self._last_wall, self._last_mono, self._effective = wall, mono, wall
            if self._journal.initial is None:
                self._data = {"schema_version": 1, "approval_sha256": self.approval.subject_sha256,
                              "last_wall": wall, "records": {}}
            else:
                self._data = strict_object(self._journal.initial)
                self._validate_loaded()
                if wall < self._data["last_wall"]:
                    # Opening for recovery remains possible; new admission fails closed.
                    self._poisoned = True
            self._entered = True
            self._save()
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            self._journal.__exit__(None, None, None)
            self._entered = False
            raise CloudBudgetError("private budget state could not be resumed")
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self._entered = False
        self._journal.__exit__(kind, value, traceback)

    def _validate_loaded(self) -> None:
        data = self._data
        if (set(data) != {"schema_version", "approval_sha256", "last_wall", "records"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1
                or data["approval_sha256"] != self.approval.subject_sha256
                or not _number(data["last_wall"], zero=True)
                or not isinstance(data["records"], dict) or len(data["records"]) > 100):
            raise ValueError
        for key, record in data["records"].items():
            if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
                raise ValueError
            intent = AcquisitionIntent.from_dict(record["intent"])
            snapshot = LeaseSnapshot.from_dict(record["snapshot"])
            if (key != intent.key or key != snapshot.key or not self.approval.admits(intent)
                    or type(record["authorized"]) is not bool
                    or not isinstance(record["discovered_ids"], list)
                    or any(not isinstance(i, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", i)
                           is None for i in record["discovered_ids"])):
                raise ValueError
            for name in ("reservation_usd", "estimate_usd", "reserved_at", "deadline",
                         "last_activity", "actual_usd"):
                if record[name] is not None and not _number(record[name], zero=True):
                    raise ValueError
            if record["authorized"] and (record["reservation_usd"] is None
                                         or record["deadline"] is None
                                         or record["reserved_at"] is None
                                         or record["last_activity"] is None
                                         or record["quote"] is None):
                raise ValueError
            if record["authorized"]:
                expected_reserve = math.ceil((intent.spec.max_hourly_usd + .02)
                    * intent.lease_seconds / 3600 * 1_000_000) / 1_000_000
                if (record["reservation_usd"] != expected_reserve
                        or record["deadline"] != record["reserved_at"] + intent.lease_seconds
                        or record["last_activity"] < record["reserved_at"]
                        or not _number(record["estimate_usd"])
                        or not isinstance(record["quote"], dict)
                        or not _number(record["quote"].get("compute_hourly_usd"))
                        or record["quote"]["compute_hourly_usd"] > intent.spec.max_hourly_usd):
                    raise ValueError
            elif snapshot.resource_id is not None:
                raise ValueError
            if record["guard"] is not None:
                receipt = GuardReceipt.from_dict(record["guard"])
                if (receipt.plan.pod_id != snapshot.resource_id
                        or receipt.plan.session_id != intent.spec.session_id
                        or receipt.plan.lease_token != intent.owner_token):
                    raise ValueError
            if record["actual_usd"] is not None and (snapshot.state != "released"
                    or not self._billing_reference(record["actual_reference"])):
                raise ValueError
        if sum(r["authorized"] for r in data["records"].values()) > self.approval.max_attempts:
            raise ValueError

    def _now(self) -> float:
        if not self._entered:
            raise CloudBudgetError("budget session is not locked")
        wall, mono = self.clock(), self.monotonic()
        if (not _number(wall, zero=True) or not _number(mono, zero=True)
                or wall < self._last_wall or mono < self._last_mono):
            self._poisoned = True
            raise CloudBudgetError("cloud budget clock moved backwards")
        self._effective = max(wall, self._effective + mono - self._last_mono)
        self._last_wall, self._last_mono = wall, mono
        self._data["last_wall"] = max(self._data["last_wall"], self._effective)
        return self._effective

    def _save(self) -> None:
        try:
            self._journal.checkpoint(canonical(self._data))
        except (Exception, KeyboardInterrupt):
            self._poisoned = True
            raise CloudBudgetError("durable cloud budget checkpoint failed") from None

    def _record(self, key: str) -> dict[str, Any]:
        record: dict[str, Any] | None = self._data.get("records", {}).get(key)
        if record is None:
            raise CloudBudgetError("unknown journal lease")
        return record

    def event_sink(self, intent: AcquisitionIntent, snapshot: LeaseSnapshot) -> None:
        with self._lock:
            if not self._entered or intent.key != snapshot.key:
                raise CloudBudgetError("invalid journal lifecycle event")
            records = self._data["records"]
            if intent.key not in records:
                if snapshot.state != "reserved" or not self.approval.admits(intent):
                    raise CloudBudgetError("unapproved journal lease")
                records[intent.key] = dict(intent=intent.to_dict(), snapshot=snapshot.to_dict(),
                    authorized=False, reservation_usd=None, estimate_usd=None, quote=None,
                    reserved_at=None, deadline=None, last_activity=None, guard=None,
                    actual_usd=None, actual_reference=None, discovered_ids=[])
            record = self._record(intent.key)
            previous = LeaseSnapshot.from_dict(record["snapshot"])
            if (record["intent"] != intent.to_dict()
                    or previous.resource_id is not None
                    and previous.resource_id != snapshot.resource_id
                    or snapshot.state not in _TRANSITIONS[previous.state]
                    or snapshot.resource_id is not None and not record["authorized"]
                    or snapshot.state == "ready" and (record["guard"] is None
                        or self._now() >= record["guard"]["plan"]["stop_at_unix"])):
                raise CloudBudgetError("invalid or unguarded journal lifecycle transition")
            record["snapshot"] = snapshot.to_dict()
            # Identity checkpoint is synchronous and precedes probe/work in the driver.
            self._save()

    def _fresh_quote(self, intent: AcquisitionIntent) -> dict[str, Any]:
        observed = self._now()
        path = ("/v2/catalog/gpus/" + quote(intent.spec.gpu_id, safe="")
                + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion="
                + quote(intent.spec.min_cuda_version, safe=""))
        failed = False
        result = None
        try:
            reply = self.transport.request("GET", path, timeout_seconds=min(10,
                intent.request_timeout_seconds), max_bytes=1024 * 1024)
            if (not isinstance(reply, CloudResponse) or type(reply.status) is not int
                    or reply.status != 200 or not isinstance(reply.body, bytes)
                    or len(reply.body) > 1024 * 1024):
                raise ValueError
            catalog = strict_object(reply.body)
            rate = catalog["price"]["secure"]
            locations = catalog["dataCenters"]
            location = next(v for v in locations if v["id"] == intent.spec.data_center_id)
            available_cuda = catalog["cudaVersions"]
            if (catalog["id"] != intent.spec.gpu_id or catalog["secure"] is not True
                    or catalog["availability"] not in ("LOW", "MEDIUM", "HIGH")
                    or type(catalog["maxCount"]["secure"]) is not int
                    or catalog["maxCount"]["secure"] < 1 or not _number(rate)
                    or rate > min(intent.spec.max_hourly_usd, self.approval.hourly_ceiling_usd)
                    or location["availability"] not in ("LOW", "MEDIUM", "HIGH")
                    or not isinstance(available_cuda, list)
                    or not any(isinstance(v, dict) and v.get("available") is True
                               and isinstance(v.get("version"), str)
                               and re.fullmatch(r"\d{1,2}\.\d{1,2}", v["version"])
                               and tuple(map(int, v["version"].split("."))) >= tuple(map(int,
                                   intent.spec.min_cuda_version.split(".")))
                               for v in available_cuda)
                    or self._now() - observed > self.quote_max_age_seconds):
                raise ValueError
            result = {"observed_unix": observed, "compute_hourly_usd": rate,
                      "gpu_id": catalog["id"], "region": location["id"],
                      "cuda_versions": available_cuda}
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or result is None:
            raise CloudBudgetError("fresh compatible regional compute quote unavailable")
        return result

    def before_create(self, intent: AcquisitionIntent) -> None:
        with self._lock:
            now = self._now()
            record = self._record(intent.key)
            others = [r for key, r in self._data["records"].items() if key != intent.key]
            attempts = sum(r["authorized"] for r in self._data["records"].values())
            if (self._poisoned or not self.guard_deployment_configured
                    or record["intent"] != intent.to_dict() or record["authorized"]
                    or not self.approval.admits(intent) or attempts >= self.approval.max_attempts
                    or any(LeaseSnapshot.from_dict(r["snapshot"]).state not in _TERMINAL
                           for r in others)):
                raise CloudBudgetError("cloud acquisition blocked by durable admission policy")
            # Conservative disk allowance exceeds proposal's illustrative disk estimate.
            # This reservation is an admission estimate, never an invoice/hard spend cap.
            reserve = math.ceil((intent.spec.max_hourly_usd + .02)
                                * intent.lease_seconds / 3600 * 1_000_000) / 1_000_000
            used = sum(r["actual_usd"] if r["actual_usd"] is not None
                       else r["reservation_usd"] or 0 for r in self._data["records"].values())
            if reserve > intent.max_cost_usd or used + reserve > self.approval.aggregate_usd:
                raise CloudBudgetError("aggregate estimated budget cannot admit this lease")
            fresh = self._fresh_quote(intent)
            now = self._now()
            record.update(authorized=True, reservation_usd=reserve,
                          estimate_usd=(fresh["compute_hourly_usd"] + .0109589)
                          * intent.lease_seconds / 3600, quote=fresh, reserved_at=now,
                          deadline=now + intent.lease_seconds, last_activity=now)
            self._save()  # Paid attempt consumed before POST; crash cannot refund it.
            # Atomic persistence can itself take long enough to invalidate admission.
            # Keep the durable reservation/attempt even when no POST may now follow.
            now = self._now()
            if (self._poisoned or now - fresh["observed_unix"] > self.quote_max_age_seconds
                    or now >= record["deadline"] - 60):
                raise CloudBudgetError("durable admission expired before cloud creation")

    def guard_plan(self, key: str) -> WorkerGuardPlan:
        with self._lock:
            record = self._record(key)
            intent = AcquisitionIntent.from_dict(record["intent"])
            snapshot = LeaseSnapshot.from_dict(record["snapshot"])
            if (not record["authorized"] or snapshot.resource_id is None
                    or snapshot.state not in {"created", "ready"}):
                raise CloudBudgetError("worker guard requires journal-owned identity")
            return WorkerGuardPlan(snapshot.resource_id, intent.spec.session_id, intent.owner_token,
                record["deadline"] - 60, self.idle_seconds)

    def arm_guard(self, key: str, receipt: GuardReceipt) -> None:
        with self._lock:
            now = self._now()
            if (not isinstance(receipt, GuardReceipt) or receipt.plan != self.guard_plan(key)
                    or receipt.source_sha256 != worker_guard_source_sha256()
                    or abs(receipt.observed_unix - now) > 30 or now >= receipt.plan.stop_at_unix):
                raise CloudBudgetError("worker guard attestation does not match owned lease")
            self._record(key)["guard"] = receipt.to_dict()
            self._save()

    def remaining_seconds(self, key: str) -> float:
        """Execution/setup timeout ceiling; reserve final 60s for STOP and deletion."""
        with self._lock:
            record = self._record(key)
            if not record["authorized"]:
                raise CloudBudgetError("lease is not admitted")
            now = self._now()
            if self._poisoned or LeaseSnapshot.from_dict(record["snapshot"]).state not in {
                    "created", "ready"}:
                return 0.
            return max(0., float(record["deadline"]) - 60 - now)

    def heartbeat(self, key: str) -> None:
        """Local supervision only; M3.4 separately updates heartbeat on worker's clock."""
        with self._lock:
            record = self._record(key)
            if (not record["authorized"] or LeaseSnapshot.from_dict(record["snapshot"]).state
                    not in {"created", "ready"}):
                raise CloudBudgetError("cannot heartbeat an inactive lease")
            record["last_activity"] = self._now()
            self._save()

    def _claim_cleanup(self, record: dict[str, Any]) -> tuple[AcquisitionIntent,
                                                           LeaseSnapshot] | None:
        """Budget lock held: persist the decision, deny READY, deduplicate local supervisors."""
        intent = AcquisitionIntent.from_dict(record["intent"])
        snapshot = LeaseSnapshot.from_dict(record["snapshot"])
        if (snapshot.resource_id is None or snapshot.state in _TERMINAL
                or intent.key in self._cleanup_inflight):
            return None
        snapshot = replace(snapshot, state="cleanup_pending")
        record["snapshot"] = snapshot.to_dict()
        self._cleanup_inflight.add(intent.key)
        try:
            self._save()
        except CloudBudgetError:
            # Persistence failure must not prevent cleanup of an already known ID.
            # The poisoned session still blocks all further spending/work admission.
            pass
        return intent, snapshot

    def _release(self, provisioner: RunpodPodProvisioner, intent: AcquisitionIntent,
                 snapshot: LeaseSnapshot) -> None:
        """Never hold the budget lock across provider locks, callbacks, or lifecycle I/O."""
        try:
            if snapshot.resource_id is None:
                raise CloudBudgetError("cannot release an unknown worker")
            if not any(v.key == intent.key for v in provisioner.snapshots):
                provisioner.restore_owned(intent, snapshot)
            provisioner.release(snapshot.resource_id)
        finally:
            with self._lock:
                self._cleanup_inflight.discard(intent.key)

    def enforce_limits(self, provisioner: RunpodPodProvisioner) -> tuple[str, ...]:
        """Call frequently during setup/download/inference; heartbeat never extends deadline."""
        claims = []
        with self._lock:
            try:
                now = self._now()
            except CloudBudgetError:
                now = math.inf
            for record in self._data["records"].values():
                snapshot = LeaseSnapshot.from_dict(record["snapshot"])
                if (snapshot.resource_id and snapshot.state not in _TERMINAL
                        and record["authorized"] and (self._poisoned or now >= record["deadline"] - 60
                        or now - record["last_activity"] >= self.idle_seconds)):
                    claim = self._claim_cleanup(record)
                    if claim is not None:
                        claims.append(claim)
        stopped = []
        for intent, snapshot in claims:
            self._release(provisioner, intent, snapshot)
            stopped.append(intent.key)
        return tuple(stopped)

    def recover_owned(self, provisioner: RunpodPodProvisioner, *, max_pages: int = 10) -> tuple[str, ...]:
        """GET-only discovery followed by verified owned deletion; never recreate/retry POST."""
        if type(max_pages) is not int or not 1 <= max_pages <= 20:
            raise CloudBudgetError("invalid recovery pagination bound")
        claims = []
        with self._lock:
            for record in self._data["records"].values():
                snapshot = LeaseSnapshot.from_dict(record["snapshot"])
                if snapshot.state in _TERMINAL or snapshot.key in self._cleanup_inflight:
                    continue
                if snapshot.resource_id is None:
                    self._discover(record, max_pages)
                claim = self._claim_cleanup(record)
                if claim is not None:
                    claims.append(claim)
        recovered = []
        for intent, snapshot in claims:
            self._release(provisioner, intent, snapshot)
            recovered.append(intent.key)
        return tuple(recovered)

    def _discover(self, record: dict[str, Any], max_pages: int) -> None:
        intent = AcquisitionIntent.from_dict(record["intent"])
        matches = []
        cursor: str | None = None
        visited: set[str] = set()
        started = self.monotonic()
        complete = False
        try:
            for _ in range(max_pages):
                remaining = 30 - (self.monotonic() - started)
                if not 0 < remaining <= 30:
                    break
                path = "/v2/pods?limit=100" + ("&cursor=" + quote(cursor, safe="") if cursor else "")
                reply = self.transport.request("GET", path, timeout_seconds=min(5, remaining),
                                               max_bytes=1024*1024)
                if (not isinstance(reply, CloudResponse) or type(reply.status) is not int
                        or reply.status != 200 or not isinstance(reply.body, bytes)
                        or len(reply.body) > 1024 * 1024
                        or not 0 <= self.monotonic() - started < 30):
                    break
                page = strict_object(reply.body)
                if not isinstance(page["pods"], list):
                    break
                for pod in page["pods"]:
                    if self._discovery_match(pod, intent):
                        matches.append(pod["id"])
                pagination = page["pagination"]
                if (not isinstance(pagination, dict)
                        or type(pagination.get("hasNextPage")) is not bool):
                    break
                cursor = pagination.get("nextCursor")
                if pagination["hasNextPage"] is False and cursor is None:
                    complete = True
                    break
                if pagination["hasNextPage"] is False:
                    break
                if (not isinstance(cursor, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", cursor)
                        or cursor in visited):
                    break
                visited.add(cursor)
        except (Exception, KeyboardInterrupt):
            complete = False
        record["discovered_ids"] = sorted(set(record["discovered_ids"] + matches))
        if complete and len(record["discovered_ids"]) == 1:
            record["snapshot"] = LeaseSnapshot(intent.key, "create_uncertain",
                resource_id=record["discovered_ids"][0], reason="restored").to_dict()
        else:
            # Empty listing is not evidence of absence after ambiguous POST.
            record["snapshot"] = LeaseSnapshot(intent.key, "create_uncertain",
                                                reason="transport").to_dict()
        self._save()

    @staticmethod
    def _discovery_match(pod: Any, intent: AcquisitionIntent) -> bool:
        if not isinstance(pod, dict):
            return False
        env = pod.get("env")
        return (isinstance(pod.get("id"), str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", pod["id"]) is not None
                and pod.get("name") == intent.name and isinstance(env, dict)
                and env.get("CINEFORGE_SESSION") == intent.spec.session_id
                and env.get("CINEFORGE_LEASE") == intent.owner_token
                and pod.get("image") == intent.spec.image and pod.get("cloud") == "SECURE"
                and pod.get("disk") == intent.spec.disk_gb
                and pod.get("dataCenterId") == intent.spec.data_center_id
                and isinstance(pod.get("gpu"), dict)
                and pod["gpu"].get("id") == intent.spec.gpu_id
                and pod["gpu"].get("count") == 1 and pod.get("mounts") == {})

    @staticmethod
    def _billing_reference(value: object) -> bool:
        return (isinstance(value, str)
                and re.fullmatch(r"billing:[A-Za-z0-9_-]{1,100}", value) is not None)

    def record_actual(self, key: str, amount_usd: float, *, billing_reference: str,
                      authoritative_final: bool = False) -> None:
        """Operator-verified final provider billing only; elapsed-time estimates are not actuals."""
        with self._lock:
            record = self._record(key)
            if (authoritative_final is not True or not self._billing_reference(billing_reference)
                    or not _number(amount_usd, zero=True) or record["actual_usd"] is not None
                    or LeaseSnapshot.from_dict(record["snapshot"]).state != "released"):
                raise CloudBudgetError("actual cost requires authoritative final billing evidence")
            record.update(actual_usd=amount_usd, actual_reference=billing_reference)
            self._save()

    @property
    def intents(self) -> tuple[AcquisitionIntent, ...]:
        """Immutable copies of trusted intents for exact-spec recovery reconstruction."""
        with self._lock:
            return tuple(AcquisitionIntent.from_dict(r["intent"])
                         for r in self._data.get("records", {}).values())

    @property
    def snapshots(self) -> tuple[LeaseSnapshot, ...]:
        with self._lock:
            return tuple(LeaseSnapshot.from_dict(r["snapshot"])
                         for r in self._data.get("records", {}).values())

    @property
    def uncertain_identities(self) -> dict[str, tuple[str, ...]]:
        with self._lock:
            return {key: tuple(r["discovered_ids"]) for key, r in
                    self._data.get("records", {}).items() if r["discovered_ids"]
                    and LeaseSnapshot.from_dict(r["snapshot"]).state not in _TERMINAL}

    def costs(self) -> dict[str, Any]:
        with self._lock:
            records = [r for r in self._data.get("records", {}).values() if r["authorized"]]
            unknown = [r for r in records if r["actual_usd"] is None]
            reserved = sum(r["reservation_usd"] for r in unknown)
            actual = sum(r["actual_usd"] for r in records if r["actual_usd"] is not None)
            return {"attempts": len(records), "reserved_worst_case_usd": reserved,
                    "estimated_usd": sum(r["estimate_usd"] for r in records),
                    "actual_known_usd": actual, "actual_total_usd": None if unknown else actual,
                    "admission_committed_usd": reserved + actual,
                    "invoice_unknown_keys": [AcquisitionIntent.from_dict(r["intent"]).key
                                             for r in unknown],
                    "hard_account_spend_cap": False}
