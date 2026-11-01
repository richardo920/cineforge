"""Closed I2V admission and private reference resolution; never provisions or uploads."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cineforge import conditioning_profile as profile
from cineforge.characters import CharacterPin, CharacterRegistry, ReferenceArtifact
from cineforge.comfy_policy import ApprovedComfySelection, ComfyPolicyError, WorkerVerification
from cineforge.contracts import Project, ProjectPlan, Scene, Shot
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.private_store import read_private_input
from cineforge.providers import ExecutionRequest
from cineforge.rights import EvidenceRef, RightsLedger, RightsRecord, RightsRef, SubjectRef, Use
from cineforge.skills import CatalogEntry, ShotSkills, SkillCatalog
from cineforge.worker_bootstrap import validate_m4_runtime, validate_reference_png
from cineforge.workflows import InputBinding, WorkflowTemplate, canonical, strict_object


class ConditioningError(ValueError):
    """Static diagnostics; reference paths and parser exceptions never leave admission."""


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def model_subject() -> SubjectRef:
    return SubjectRef("model", "wan22-ti2v-5b", profile.MODEL_REVISION,
                      digest(profile.model_files()))


def shots(character: CharacterPin, workflow_digest: str) -> tuple[Shot, ...]:
    return tuple(Shot(f"m4-identity-0{i+1}", "identity", 1, prompt, seed, profile.DURATION,
                      profile.MODEL_ID, profile.MODEL_REVISION, workflow_digest,
                      character.to_key())
                 for i, (prompt, seed) in enumerate(zip(profile.PROMPTS, profile.SEEDS)))


@dataclass(frozen=True)
class PrivateReference:
    artifact: ReferenceArtifact = field(repr=False)
    path: Path = field(repr=False)

    @property
    def basename(self) -> str:
        return "ref-" + self.artifact.artifact.sha256 + ".png"

    def read(self) -> bytes:
        failed = False
        payload = b""
        try:
            if (self.artifact.role != "identity" or self.artifact.media_type != "image/png"
                    or not 0 < self.artifact.size_bytes <= 128*1024):
                raise ValueError
            payload = read_private_input(self.path, max_bytes=128*1024)
            if (len(payload) != self.artifact.size_bytes
                    or hashlib.sha256(payload).hexdigest() != self.artifact.artifact.sha256):
                raise ValueError
            validate_reference_png(payload)
        except (ValueError, OSError, TypeError):
            failed = True
        if failed:
            raise ConditioningError("private reference bytes rejected")
        return payload

    @property
    def worker_metadata(self) -> dict[str, Any]:
        return {"sha256": self.artifact.artifact.sha256,
                "size_bytes": self.artifact.size_bytes, "basename": self.basename}


@dataclass(frozen=True)
class StagedReference:
    """Checked owned-SSH attestation, not a creative-plan or self-authenticating claim."""

    worker_id: str
    reference_sha256: str
    runtime_json: str = field(repr=False)

    def __post_init__(self) -> None:
        failed = False
        try:
            if (not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.worker_id)
                    or not re.fullmatch(r"[0-9a-f]{64}", self.reference_sha256)):
                raise ValueError
            validate_m4_runtime(strict_object(self.runtime_json))
        except (ValueError, TypeError):
            failed = True
        if failed:
            raise ConditioningError("invalid staged-worker runtime attestation")


@dataclass(frozen=True)
class VerifiedConditioning:
    """Non-serialized capability binds exact graph/shot, current rights and private bytes.

    Constructed by the checked adapter; no creative JSON flag grants this capability.
    Registries are intentionally live references so revocation is rechecked per shot.
    """

    reference: PrivateReference = field(repr=False)
    registry: CharacterRegistry = field(repr=False)
    ledger: RightsLedger = field(repr=False)
    catalog: SkillCatalog = field(repr=False)
    bindings: tuple[ShotSkills, ...] = field(repr=False)
    character: CharacterPin
    model_rights: RightsRef
    journal_directory: str = field(repr=False)
    region: str
    staging: StagedReference | None = field(default=None, repr=False)

    @property
    def graph(self) -> dict[str, Any]:
        return profile.graph(self.reference.artifact.artifact.sha256)

    @property
    def workflow_digest(self) -> str:
        return digest(self.graph)

    @property
    def requests(self) -> tuple[Shot, ...]:
        return shots(self.character, self.workflow_digest)

    def require(self, request: ExecutionRequest, workflow_digest: str) -> None:
        failed = False
        try:
            if (type(request) is not ExecutionRequest or request.model != profile.model()
                    or workflow_digest != self.workflow_digest
                    or request.shot not in self.requests or request.attempt != 1
                    or len(self.bindings) != 3):
                raise ValueError
            index = self.requests.index(request.shot)
            resolved = self.registry.resolve_shot(request.shot, self.ledger, use=Use("condition"))
            character = resolved.character
            if (not resolved.allowed or character is None
                    or character.identity_kind != "fictional"
                    or character.references != (self.reference.artifact,)
                    or len(character.conditioning) != 1):
                raise ValueError
            hint = character.conditioning[0]
            if ((hint.mode, hint.model_id, hint.model_revision, hint.workflow_digest,
                 hint.reference_ids) != (profile.MODE, profile.MODEL_ID, profile.MODEL_REVISION,
                     workflow_digest, (self.reference.artifact.reference_id,))):
                raise ValueError
            binding = self.bindings[index]
            device = GPUDevice("proposed-m4-worker", Backend.CUDA, profile.MEMORY_MIB,
                               frozenset({Precision.FP16}))
            ready = self.catalog.resolve_shot(binding, shot=request.shot, ledger=self.ledger,
                                               device=device)
            entries = tuple(self.catalog.lookup(pin) for pin in binding.skills)
            workflow_entries = tuple(entry for entry in entries if entry.category == "workflow"
                and (entry.source.repository, entry.source.revision, entry.source.path,
                     entry.source.sha256) == (profile.SOURCE_REPOSITORY, profile.SOURCE_REVISION,
                                              profile.SOURCE_PATH, profile.SOURCE_SHA256))
            if (not ready or not all(item.ready for item in ready)
                    or len(workflow_entries) != 1
                    or not self.ledger.preflight_use(model_subject(), self.model_rights,
                                                     Use("execute")).allowed):
                raise ValueError
            model_terms = self.ledger.lookup(self.model_rights)
            workflow_terms = self.ledger.lookup(workflow_entries[0].rights)
            if (model_terms is None or model_terms.license_id != "Apache-2.0"
                    or model_terms.source != EvidenceRef(profile.MODEL_CARD_URI,
                                                        profile.MODEL_CARD_SHA256)
                    or model_terms.source_revision != profile.MODEL_REVISION
                    or model_terms.terms != EvidenceRef(profile.MODEL_TERMS_URI,
                                                       profile.MODEL_TERMS_SHA256)
                    or workflow_terms is None or workflow_terms.license_id != "MIT"
                    or workflow_terms.source != EvidenceRef(workflow_entries[0].source.uri,
                                                           profile.SOURCE_SHA256)
                    or workflow_terms.source_revision != profile.SOURCE_REVISION
                    or workflow_terms.terms is None or workflow_terms.terms.sha256 !=
                       hashlib.sha256(profile.WORKFLOW_LICENSE).hexdigest()):
                raise ValueError
            self.reference.read()
        except (ValueError, TypeError, KeyError, OSError, AttributeError):
            failed = True
        if failed:
            raise ConditioningError("conditioning profile, current rights or reference rejected")

    def trace(self, shot: Shot) -> dict[str, Any]:
        self.require(ExecutionRequest(shot, profile.model()), self.workflow_digest)
        binding = self.bindings[self.requests.index(shot)]
        record = self.registry.lookup(self.character)
        assert record is not None
        refs = (self.reference.artifact.permission.rights, self.model_rights,
                *(self.catalog.lookup(pin).rights for pin in binding.skills))
        if record.likeness is not None:
            refs += (record.likeness.rights,)
        return {"profile_id": profile.PROFILE_ID, "character": record.to_dict(),
                "reference": self.reference.artifact.to_dict(), "skills": binding.to_dict(),
                "catalog": [self.catalog.lookup(pin).to_dict() for pin in binding.skills],
                "rights": [self.ledger.lookup(ref).to_dict() for ref in refs  # type: ignore[union-attr]
                           if self.ledger.lookup(ref) is not None]}

    def preflight(self) -> None:
        path = Path(self.journal_directory)
        if (not path.is_absolute() or ".." in path.parts
                or str(path) != self.journal_directory or not self.journal_directory.isprintable()
                or any(c in self.journal_directory for c in "\\:%")):
            raise ConditioningError("M4 requires a fixed absolute private journal directory")
        if self.region not in profile.REGIONS:
            raise ConditioningError("M4 requires one explicitly reviewed data center")
        for shot in self.requests:
            self.require(ExecutionRequest(shot, profile.model()), self.workflow_digest)

    def require_submission(self, request: ExecutionRequest, worker_id: str | None) -> None:
        self.require(request, self.workflow_digest)
        if (type(self.staging) is not StagedReference or self.staging.worker_id != worker_id
                or self.staging.reference_sha256 != self.reference.artifact.artifact.sha256):
            raise ConditioningError("I2V submission requires this worker's staged reference")

    @property
    def subject(self) -> dict[str, Any]:
        self.preflight()
        return {"schema_version": 1, "task_id": "M4.2", "profile_id": profile.PROFILE_ID,
                "selection": profile.declarations(self.reference.artifact.artifact.sha256),
                "graph": self.graph, "shots": [shot.to_dict() for shot in self.requests],
                "lineage": [self.trace(shot) for shot in self.requests],
                "reference_staging": self.reference.worker_metadata,
                "proposed_operations": ["download_exact_model_files", "stage_exact_reference",
                    "acquire_one_guarded_pod", "generate_three_private_shots"],
                "cloud": {"image": profile.IMAGE, "gpu_id": profile.GPU_ID,
                    "journal_directory": self.journal_directory,
                    "region": self.region, "region_fallback": False,
                    "min_cuda": "12.8", "aggregate_usd": 1,
                    "max_attempts": 1, "max_concurrent": 1, "lifetime_seconds": 1800,
                    "hourly_ceiling_usd": 1.25, "disk_gb": 80, "volumes": [], "ports": ["22/tcp"]}}


@dataclass(frozen=True)
class ConditioningApproval:
    """Operator-trusted receipt assertion; does not authenticate the approver or spend."""

    record_json: str = field(repr=False)
    subject_json: str = field(repr=False)

    def __post_init__(self) -> None:
        failed = False
        try:
            record, subject = strict_object(self.record_json), strict_object(self.subject_json)
            if (set(record) != {"status", "subject_sha256", "approved_by", "receipt"}
                    or record["status"] != "approved" or record["subject_sha256"] != digest(subject)
                    or not isinstance(record["approved_by"], str) or not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", record["approved_by"])
                    or not isinstance(record["receipt"], str) or not re.fullmatch(
                        r"https://github\.com/richardo920/cineforge/issues/60#issuecomment-[0-9]+",
                        record["receipt"])):
                raise ValueError
            object.__setattr__(self, "record_json", canonical(record).decode())
            object.__setattr__(self, "subject_json", canonical(subject).decode())
        except (ValueError, TypeError, KeyError):
            failed = True
        if failed:
            raise ConditioningError("missing or mismatched M4 operator approval")

    @property
    def subject_sha256(self) -> str:
        return digest(strict_object(self.subject_json))


@dataclass(frozen=True)
class ConditioningSelection(ApprovedComfySelection):
    gate: VerifiedConditioning = field(repr=False)
    approval: ConditioningApproval = field(repr=False)

    def __post_init__(self) -> None:
        if (type(self.gate) is not VerifiedConditioning
                or type(self.approval) is not ConditioningApproval):
            raise ComfyPolicyError("invalid checked conditioning selection")
        expected = profile.declarations(self.gate.reference.artifact.artifact.sha256)
        expected["approval"] = {"model_workflow": {"approved_subject_sha256":
                                    self.approval.subject_sha256}, "spend_limit_usd": 1}
        if (self.approval.subject_json != canonical(self.gate.subject).decode()
                or self.selection_json != canonical(expected).decode()
                or self.graph_json != canonical(self.gate.graph).decode()
                or self.license_bytes != profile.WORKFLOW_LICENSE):
            raise ComfyPolicyError("conditioning selection differs from exact approved subject")
        self.workflow

    @property
    def workflow(self) -> WorkflowTemplate:
        data = self.declarations["workflow"]
        return WorkflowTemplate(self.graph_json, profile.MODEL_ID, profile.MODEL_REVISION,
            frozenset(data["allowed_classes"]), InputBinding("6", "text"), InputBinding("3", "seed"),
            "47", "images", "LicenseRef-Generated-Unreviewed", 0, profile.DURATION, self.gate)

    def request(self, plan_shot: Shot) -> ExecutionRequest:
        request = ExecutionRequest(plan_shot, profile.model())
        self.workflow.translate(request)
        return request

    def bind_worker(self, worker: WorkerVerification, staged: dict[str, Any],
                    runtime: dict[str, Any] | None) -> ConditioningSelection:
        worker.require_matches(self)
        if (staged != self.gate.reference.worker_metadata or runtime is None
                or worker.record["image_digest"] != profile.IMAGE.split("@", 1)[1]
                or worker.record["torch_version"] != "2.8.0+cu128"
                or worker.record["cuda_version"] != "12.8"
                or runtime.get("gpu_memory_mib") != worker.record["gpu"]["memory_mib"]):
            raise ConditioningError("staged reference or worker differs from M4 approval")
        proof = StagedReference(worker.record["worker_id"], self.gate.reference.artifact.artifact.sha256,
                                canonical(runtime).decode())
        return replace(self, gate=replace(self.gate, staging=proof))

    def require_execution_worker(self, worker: WorkerVerification) -> None:
        self.gate.require_submission(self.request(self.gate.requests[0]), worker.record["worker_id"])
        assert self.gate.staging is not None
        runtime = strict_object(self.gate.staging.runtime_json)
        if (runtime["gpu_memory_mib"] != worker.record["gpu"]["memory_mib"]
                or worker.record["image_digest"] != profile.IMAGE.split("@", 1)[1]
                or worker.record["torch_version"] != runtime["torch_version"]
                or worker.record["cuda_version"] != runtime["cuda_version"]):
            raise ConditioningError("M4 worker verification differs from staged capability")

    @property
    def worker_profile(self) -> dict[str, Any]:
        self.gate.preflight()
        return {"id": profile.PROFILE_ID, "reference": self.gate.reference.worker_metadata}

    def plan(self, shot: Shot) -> ProjectPlan:
        self.request(shot)
        project = Project("m4-identity", 1, "Private identity review")
        return ProjectPlan(project, (Scene("identity", project.project_id, 1),), (shot,))


def selection_from_approval(gate: VerifiedConditioning, receipt: dict[str, Any]) -> ConditioningSelection:
    approval = ConditioningApproval(canonical(receipt).decode(), canonical(gate.subject).decode())
    declarations = profile.declarations(gate.reference.artifact.artifact.sha256)
    declarations["approval"] = {"model_workflow": {"approved_subject_sha256":
                                approval.subject_sha256}, "spend_limit_usd": 1}
    return ConditioningSelection(canonical(declarations).decode(), canonical(gate.graph).decode(),
                                 profile.WORKFLOW_LICENSE, gate, approval)


def load_context(path: str | Path, reference_path: str | Path) -> VerifiedConditioning:
    """Load private metadata separately from the operator approval; no network or uploads."""
    failed = False
    gate = None
    try:
        data = strict_object(read_private_input(path))
        if set(data) != {"character", "rights", "revocations", "catalog", "bindings", "model_rights",
                         "journal_directory", "region"}:
            raise ValueError
        from cineforge.characters import CharacterRevision
        character = CharacterRevision.from_dict(data["character"])
        registry, ledger = CharacterRegistry(), RightsLedger()
        registry.register(character)
        for value in data["rights"]:
            ledger.register(RightsRecord.from_dict(value))
        for value in data["revocations"]:
            ledger.revoke(RightsRef.from_dict(value), reason="Operator recorded revocation")
        catalog = SkillCatalog(tuple(CatalogEntry.from_dict(value) for value in data["catalog"]))
        if len(character.references) != 1:
            raise ValueError
        gate = VerifiedConditioning(PrivateReference(character.references[0], Path(reference_path)),
            registry, ledger, catalog, tuple(ShotSkills.from_dict(v) for v in data["bindings"]),
            character.pin(), RightsRef.from_dict(data["model_rights"]), data["journal_directory"],
            data["region"])
        gate.preflight()
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        failed = True
    if failed or gate is None:
        raise ConditioningError("private conditioning context rejected")
    return gate
