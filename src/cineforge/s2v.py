"""Exact native core S2V admission; trusted code verifies actor/native-speech evidence.

No JSON receipt, pose descriptor or fixture waveform authenticates live provenance.
Registries, rights and proof callbacks remain current on every translation/submission.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from cineforge import s2v_profile as profile
from cineforge.audio import CueSheet, RevisionPin, Utterance
from cineforge.characters import CharacterPin, CharacterRegistry, ReferenceArtifact
from cineforge.comfy_policy import ApprovedComfySelection, WorkerVerification
from cineforge.contracts import ProjectPlan, Shot
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.paired_media import PNG_MAX_BYTES, image_metadata, speech_metadata
from cineforge.private_store import read_private_input
from cineforge.providers import ExecutionRequest
from cineforge.rights import EvidenceRef, RightsLedger, RightsRef, SubjectRef, Use
from cineforge.s2v_service import S2VServiceContext
from cineforge.skills import ShotSkills, SkillCatalog
from cineforge.speech import SpeechArtifact
from cineforge.workflows import InputBinding, WorkflowTemplate, canonical, strict_object


class S2VError(ValueError):
    """Static boundary diagnostics; private reference/proof/runtime strings never escape."""


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def model_subject() -> SubjectRef:
    return SubjectRef("model", "wan22-s2v-14b", profile.MODEL_REVISION,
                      digest(profile.model_files()))


def _sha(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise S2VError("invalid S2V evidence digest")


@dataclass(frozen=True)
class ActorDescriptor:
    """Requested baseline only; independent operator proof below is still mandatory."""
    identity: str = "original-fictional"
    age: str = "adult"
    people: int = 1
    pose: str = "front-facing"
    style: str = "photoreal"
    mouth: str = "visible"
    language_mode: str = "single-language"

    def __post_init__(self) -> None:
        if ((self.identity, self.age, self.pose, self.style, self.mouth, self.language_mode)
                != ("original-fictional", "adult", "front-facing", "photoreal", "visible",
                    "single-language") or type(self.people) is not int or self.people != 1):
            raise S2VError("unsupported S2V actor, pose, style or language baseline")


@dataclass(frozen=True)
class PrivateActorReference:
    artifact: ReferenceArtifact = field(repr=False)
    path: Path = field(repr=False)

    def read(self) -> bytes:
        payload = None
        try:
            if (type(self.artifact) is not ReferenceArtifact or self.artifact.role != "identity"
                    or self.artifact.media_type != "image/png"):
                raise ValueError
            value = read_private_input(self.path, max_bytes=PNG_MAX_BYTES)
            metadata = image_metadata(value)
            if (metadata["sha256"] != self.artifact.artifact.sha256
                    or metadata["size_bytes"] != self.artifact.size_bytes):
                raise ValueError
            payload = value
        except (ValueError, TypeError, OSError, AttributeError):
            pass
        if payload is None:
            raise S2VError("private original actor reference rejected")
        return payload


@dataclass(frozen=True)
class ActorReviewProof:
    """Nonserialized result of trusted operator review, not a face classifier."""
    character: CharacterPin
    reference_sha256: str
    evidence: EvidenceRef = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.character) is not CharacterPin or type(self.evidence) is not EvidenceRef:
            raise S2VError("invalid actor review proof")
        _sha(self.reference_sha256)


@dataclass(frozen=True)
class NativeSpeechProof:
    """Nonserialized verification result; callback must inspect actual native103 evidence."""
    request_sha256: str
    speech_sha256: str
    reference_sha256: str
    native_receipt_sha256: str
    worker_id: str
    scope_sha256: str

    def __post_init__(self) -> None:
        for value in (self.request_sha256, self.speech_sha256, self.reference_sha256,
                      self.native_receipt_sha256, self.scope_sha256):
            _sha(value)
        if self.native_receipt_sha256 == "0" * 64:
            raise S2VError("missing native speech receipt identity")
        if (type(self.worker_id) is not str
                or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.worker_id) is None):
            raise S2VError("invalid native speech worker identity")


ActorVerifier = Callable[[CharacterPin, ReferenceArtifact, bytes, ActorDescriptor], ActorReviewProof]
SpeechVerifier = Callable[[SpeechArtifact, S2VServiceContext], NativeSpeechProof]
SubjectVerifier = Callable[[dict[str, Any]], object]


@dataclass(frozen=True)
class StagedS2VPair:
    """Checked owned-worker attestation created after both transfers and decoded metadata."""
    worker_id: str
    scope_sha256: str
    subject_sha256: str
    metadata_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _sha(self.scope_sha256)
        _sha(self.subject_sha256)
        if (type(self.worker_id) is not str
                or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.worker_id) is None):
            raise S2VError("invalid staged S2V worker identity")
        value = strict_object(self.metadata_json)
        if set(value) != {"image", "audio"}:
            raise S2VError("invalid staged S2V pair metadata")
        object.__setattr__(self, "metadata_json", canonical(value).decode())

    @property
    def pair_sha256(self) -> str:
        return digest({"scope_sha256": self.scope_sha256,
                       "subject_sha256": self.subject_sha256,
                       "media": strict_object(self.metadata_json)})


@dataclass(frozen=True)
class VerifiedS2V:
    service: S2VServiceContext = field(repr=False)
    sheet: CueSheet = field(repr=False)
    plan: ProjectPlan = field(repr=False)
    current_picture: RevisionPin
    current_script: RevisionPin
    actor: PrivateActorReference = field(repr=False)
    speech: SpeechArtifact = field(repr=False)
    registry: CharacterRegistry = field(repr=False)
    ledger: RightsLedger = field(repr=False)
    catalog: SkillCatalog = field(repr=False)
    binding: ShotSkills = field(repr=False)
    model_rights: RightsRef
    descriptor: ActorDescriptor
    verify_actor: ActorVerifier = field(repr=False, compare=False)
    verify_speech: SpeechVerifier = field(repr=False, compare=False)
    authorize: SubjectVerifier = field(repr=False, compare=False)
    staging: StagedS2VPair | None = field(default=None, repr=False)

    @property
    def utterance(self) -> Utterance:
        value = next((item for item in self.sheet.script.utterances
                      if item.utterance_id == self.speech.request.utterance.input_id), None)
        if value is None:
            raise S2VError("missing current S2V utterance")
        return value

    @property
    def graph(self) -> dict[str, Any]:
        return profile.graph(self.actor.artifact.artifact.sha256, self.speech.pin.sha256,
                             self.utterance.language)

    @property
    def workflow_digest(self) -> str:
        return digest(self.graph)

    @property
    def shot(self) -> Shot:
        original = next((shot for shot in self.plan.shots if shot.shot_id == self.utterance.shot_id), None)
        if original is None:
            raise S2VError("missing current S2V picture shot")
        graph = self.graph
        return replace(original, prompt=graph["159"]["inputs"]["text"],
            seed=graph["152"]["inputs"]["seed"], model_id=profile.MODEL_ID,
            model_revision=profile.MODEL_REVISION, workflow_digest=self.workflow_digest)

    @property
    def media(self) -> dict[str, Any]:
        return {"image": image_metadata(self.actor.read()),
                "audio": speech_metadata(self.speech.payload)}

    def _check(self) -> dict[str, Any]:
        for value, kind in ((self.service, S2VServiceContext), (self.sheet, CueSheet),
                (self.plan, ProjectPlan), (self.actor, PrivateActorReference),
                (self.speech, SpeechArtifact), (self.registry, CharacterRegistry),
                (self.ledger, RightsLedger), (self.catalog, SkillCatalog),
                (self.binding, ShotSkills), (self.descriptor, ActorDescriptor)):
            if type(value) is not kind:
                raise ValueError
        self.service.preflight()
        self.descriptor.__post_init__()
        if (not callable(self.verify_actor) or not callable(self.verify_speech)
                or not callable(self.authorize) or self.speech.runtime_kind == "synthetic_fixture"
                or self.sheet.script.pin() != self.current_script
                or not self.sheet.preflight(self.plan, self.registry, self.ledger,
                        current_picture=self.current_picture).allowed):
            raise ValueError
        utterance = self.utterance
        character = self.registry.lookup(utterance.character)
        shot = self.shot
        span = next(item for item in self.sheet.shots if item.shot_id == utterance.shot_id)
        if (character is None or character.identity_kind != "fictional"
                or self.actor.artifact not in character.references
                or not self.registry.resolve_shot(shot, self.ledger, use=Use("condition")).allowed
                or len({item.character for item in self.sheet.script.utterances}) != 1
                or len({item.voice for item in self.sheet.script.utterances}) != 1
                or sum(item.shot_id == utterance.shot_id for item in self.sheet.script.utterances) != 1
                or utterance.start_ticks != span.start_ticks or shot.duration_seconds != profile.DURATION
                or self.speech.request.sheet != RevisionPin(self.sheet.sheet_id, self.sheet.revision,
                                                            self.sheet.digest())
                or self.speech.request.script != self.current_script
                or self.speech.request.utterance != RevisionPin(utterance.utterance_id,
                                                              utterance.text_revision, utterance.digest())
                or (self.speech.request.character, self.speech.request.voice,
                    self.speech.request.text, self.speech.request.language,
                    self.speech.request.start_ticks, self.speech.request.duration_ticks,
                    self.speech.request.timebase) != (utterance.character, utterance.voice,
                    utterance.text, {"en": "English", "zh": "Chinese"}[utterance.language],
                    utterance.start_ticks, utterance.duration_ticks, self.sheet.timebase)):
            raise ValueError
        SpeechArtifact(self.speech.pin, self.speech.request, self.speech.payload,
                       self.speech.runtime_kind, self.speech.sample_rate, self.speech.frames)
        reference = next((item for item in character.references
                          if item.reference_id == utterance.voice.reference_id), None)
        if (reference is None or reference.role != "voice"
                or reference.permission != self.speech.request.recording_permission
                or reference.artifact.sha256 != self.speech.request.reference.sha256):
            raise ValueError
        for binding in (self.actor.artifact.permission, utterance.voice.permission,
                        self.speech.request.recording_permission):
            for action in ("reference", "condition"):
                if not self.ledger.preflight_use(binding.subject, binding.rights, Use(action)).allowed:
                    raise ValueError
        actor_bytes = self.actor.read()
        media = {"image": image_metadata(actor_bytes), "audio": speech_metadata(self.speech.payload)}
        reviewed_actor = self.verify_actor(character.pin(), self.actor.artifact, actor_bytes,
                                           self.descriptor)
        speech_proof = self.verify_speech(self.speech, self.service)
        if (type(reviewed_actor) is not ActorReviewProof
                or reviewed_actor.character != character.pin()
                or reviewed_actor.reference_sha256 != media["image"]["sha256"]
                or type(speech_proof) is not NativeSpeechProof
                or speech_proof.request_sha256 != self.speech.request.digest()
                or speech_proof.speech_sha256 != media["audio"]["sha256"]
                or speech_proof.reference_sha256 != self.speech.request.reference.sha256
                or speech_proof.scope_sha256 != self.service.approval.subject_sha256):
            raise ValueError
        device = GPUDevice("s2v-profile", Backend.CUDA, profile.MEMORY_MIB,
                           frozenset({Precision.BF16}))
        ready = self.catalog.resolve_shot(self.binding, shot=shot, ledger=self.ledger, device=device)
        entries = tuple(self.catalog.lookup(pin) for pin in self.binding.skills)
        workflows = tuple(entry for entry in entries if entry is not None and entry.category == "workflow"
            and (entry.source.repository, entry.source.revision, entry.source.path, entry.source.sha256)
            == (profile.SOURCE_REPOSITORY, profile.SOURCE_REVISION, profile.SOURCE_PATH,
                profile.SOURCE_SHA256))
        if (not ready or not all(item.ready for item in ready) or len(workflows) != 1
                or not self.ledger.preflight_use(model_subject(), self.model_rights, Use("execute")).allowed):
            raise ValueError
        model_terms = self.ledger.lookup(self.model_rights)
        workflow_terms = self.ledger.lookup(workflows[0].rights)
        if (model_terms is None or model_terms.license_id != "Apache-2.0"
                or model_terms.source != EvidenceRef(profile.MODEL_CARD_URI, profile.MODEL_CARD_SHA256)
                or model_terms.source_revision != profile.MODEL_REVISION
                or model_terms.terms != EvidenceRef(profile.MODEL_TERMS_URI, profile.MODEL_TERMS_SHA256)
                or workflow_terms is None or workflow_terms.license_id != "MIT"
                or workflow_terms.source_revision != profile.SOURCE_REVISION
                or workflow_terms.source != EvidenceRef(workflows[0].source.uri, profile.SOURCE_SHA256)
                or workflow_terms.terms is None or workflow_terms.terms.sha256
                != hashlib.sha256(profile.WORKFLOW_LICENSE).hexdigest()):
            raise ValueError
        return {"character": character.to_dict(), "actor": self.actor.artifact.to_dict(),
            "actor_review": reviewed_actor.evidence.to_dict(), "speech": self.speech.pin.to_dict(),
            "speech_request": self.speech.request.to_dict(), "speech_native_proof": {
                key: getattr(speech_proof, key) for key in speech_proof.__dataclass_fields__},
            "input_media": media, "catalog": [entry.to_dict() for entry in entries if entry is not None],
            "binding": self.binding.to_dict(), "source_plan_sha256": digest(self.plan.to_dict()),
            "sheet_sha256": self.sheet.digest(), "picture": self.current_picture.to_dict(),
            "script": self.current_script.to_dict()}

    def preflight(self) -> dict[str, Any]:
        result = None
        try:
            result = self._check()
        except (Exception, KeyboardInterrupt):
            pass
        if result is None:
            raise S2VError("current S2V rights, actor, native speech or source rejected")
        return result

    @property
    def subject(self) -> dict[str, Any]:
        trace = self.preflight()
        return {"schema_version": 1, "task_id": "M5.6", "profile_id": profile.PROFILE_ID,
            "scope_sha256": self.service.approval.subject_sha256,
            "recipe_sha256": profile.recipe_sha256(), "graph": self.graph,
            "shot": self.shot.to_dict(), "lineage": trace}

    def require(self, request: ExecutionRequest, workflow_digest: str) -> None:
        self.preflight()
        if (type(request) is not ExecutionRequest or request.model != profile.model()
                or request.shot != self.shot or request.attempt != 1
                or workflow_digest != self.workflow_digest):
            raise S2VError("request differs from exact checked S2V input")
        subject = self.subject
        failed = False
        try:
            if self.authorize(subject) is not None:
                raise ValueError
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            raise S2VError("current exact S2V request authorization rejected")

    def require_submission(self, request: ExecutionRequest, worker_id: str | None) -> None:
        self.require(request, self.workflow_digest)
        proof = self.staging
        current_subject = self.subject
        speech_proof = current_subject["lineage"]["speech_native_proof"]
        if (type(proof) is not StagedS2VPair or proof.worker_id != worker_id
                or speech_proof["worker_id"] != worker_id
                or proof.scope_sha256 != self.service.approval.subject_sha256
                or proof.subject_sha256 != digest(current_subject)
                or strict_object(proof.metadata_json) != self.media):
            raise S2VError("S2V submission requires this owned worker's exact staged pair")


@dataclass(frozen=True)
class S2VSelection(ApprovedComfySelection):
    gate: VerifiedS2V = field(repr=False)
    approved_subject_sha256: str

    def __post_init__(self) -> None:
        if (type(self.gate) is not VerifiedS2V or self.license_bytes != profile.WORKFLOW_LICENSE
                or self.graph_json != canonical(self.gate.graph).decode()
                or self.approved_subject_sha256 != digest(self.gate.subject)
                or self.selection_json != canonical(_declarations(self.gate,
                                                        self.approved_subject_sha256)).decode()):
            raise S2VError("S2V selection differs from exact checked subject")
        self.gate.require(ExecutionRequest(self.gate.shot, profile.model()), self.gate.workflow_digest)

    @property
    def workflow(self) -> WorkflowTemplate:
        graph = self.gate.graph
        return WorkflowTemplate(canonical(graph).decode(), profile.MODEL_ID, profile.MODEL_REVISION,
            frozenset(node["class_type"] for node in graph.values()), InputBinding("159", "text"),
            InputBinding("152", "seed"), "168", "images", "LicenseRef-Generated-Unreviewed", 0,
            profile.DURATION, self.gate)

    def request(self, plan_shot: Shot) -> ExecutionRequest:
        request = ExecutionRequest(plan_shot, profile.model())
        self.workflow.translate(request)
        return request

    def require_execution_worker(self, worker: WorkerVerification) -> None:
        worker.require_matches(self)
        self.gate.require_submission(self.request(self.gate.shot), worker.record["worker_id"])
        if (worker.record["image_digest"] != profile.IMAGE.split("@", 1)[1]
                or worker.record["torch_version"] != "2.8.0+cu128"
                or worker.record["cuda_version"] != "12.8"):
            raise S2VError("S2V worker runtime differs from checked service")

    def bind_worker(self, worker: WorkerVerification, staged: StagedS2VPair) -> S2VSelection:
        worker.require_matches(self)
        result = replace(self, gate=replace(self.gate, staging=staged))
        result.require_execution_worker(worker)
        return result

    def plan(self) -> ProjectPlan:
        return ProjectPlan(self.gate.plan.project, tuple(scene for scene in self.gate.plan.scenes
            if scene.scene_id == self.gate.shot.scene_id), (self.gate.shot,))


def _declarations(gate: VerifiedS2V, subject_sha256: str) -> dict[str, Any]:
    data = profile.service_declarations()
    graph = gate.graph
    data["workflow"] = {"canonical_sha256": digest(graph),
        "source_repository": profile.SOURCE_REPOSITORY, "source_revision": profile.SOURCE_REVISION,
        "source_path": profile.SOURCE_PATH, "source_raw_sha256": profile.SOURCE_SHA256,
        "license": "MIT", "license_sha256": hashlib.sha256(profile.WORKFLOW_LICENSE).hexdigest(),
        "allowed_classes": sorted({node["class_type"] for node in graph.values()}),
        "bindings": {"prompt": {"node_id": "159", "input_name": "text"},
                     "seed": {"node_id": "152", "input_name": "seed"}},
        "output_node": "168", "output_kind": "images", "output_index": 0,
        "artifact_license": "LicenseRef-Generated-Unreviewed", "media_type": "video/webm",
        "duration_seconds": profile.DURATION}
    data["approval"] = {"model_workflow": {"approved_subject_sha256": subject_sha256},
                        "spend_limit_usd": gate.service.approval.aggregate_usd}
    return data


def select_s2v(gate: VerifiedS2V) -> S2VSelection:
    if type(gate) is not VerifiedS2V:
        raise S2VError("S2V requires the nonserialized checked adapter gate")
    subject_sha256 = digest(gate.subject)
    return S2VSelection(canonical(_declarations(gate, subject_sha256)).decode(),
        canonical(gate.graph).decode(), profile.WORKFLOW_LICENSE, gate, subject_sha256)
