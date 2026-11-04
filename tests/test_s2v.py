"""Synthetic actor/audio/proof callbacks; no native claim, weights, account or SSH."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_s2v_profile import png
from test_speech import EVIDENCE, Runtime, adapter, fixture, reviewed, wav_bytes

from cineforge import s2v_profile as profile
from cineforge.audio import AudioTimebase, Cue, DialogueScript, ShotTiming, Stem
from cineforge.characters import CharacterRegistry, with_character_review
from cineforge.comfy_policy import ApprovedComfySelection, ComfyPolicyError, WorkerVerification
from cineforge.contracts import ProjectPlan
from cineforge.providers import ExecutionRequest, ProviderError
from cineforge.rights import EvidenceRef, RightsLedger, RightsRef, Use, with_review
from cineforge.s2v import (
    ActorDescriptor,
    ActorReviewProof,
    NativeSpeechProof,
    PrivateActorReference,
    S2VError,
    StagedS2VPair,
    VerifiedS2V,
    digest,
    model_subject,
    select_s2v,
)
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext
from cineforge.skills import (
    CatalogEntry,
    Compatibility,
    Permissions,
    SkillCatalog,
    SourcePin,
    SourceReview,
    bind_shot,
)
from cineforge.workflows import InputBinding, WorkflowTemplate, canonical


def context(root, *, language="en", actor_png=None, gpu_id=profile.GPU_ID):
    args, actor, _ = fixture()
    payload = png(width=832, height=480) if actor_png is None else actor_png
    actor_sha = hashlib.sha256(payload).hexdigest()
    old_image = actor.references[0]
    image = replace(old_image, artifact=EvidenceRef("private://synthetic/original-actor", actor_sha),
        size_bytes=len(payload), permission=replace(old_image.permission,
            subject=replace(old_image.permission.subject, sha256=actor_sha)))
    actor = with_character_review(replace(actor, references=(image, actor.references[1])),
                                  reviewer="fixture-reviewer", evidence=EVIDENCE)
    path = root / "actor.png"
    path.write_bytes(payload)
    registry = CharacterRegistry()
    registry.register(actor)
    line = replace(args["sheet"].script.utterances[0], character=actor.pin(), start_ticks=0,
                   duration_ticks=48125, language=language)
    sheet = replace(args["sheet"], script=DialogueScript("script", 1, (line,)),
        timebase=AudioTimebase(10000, 48125), stems=(Stem("dialogue", "dialogue"),),
        shots=(ShotTiming("shot", 1, 0, 48125),), cues=(Cue("dialogue", "dialogue", "dialogue",
            "scene", "shot", 0, 48125, line.voice.permission, line.utterance_id),))
    original = replace(args["plan"].shots[0], duration_seconds=profile.DURATION,
                       character_revision=actor.pin().to_key())
    plan = ProjectPlan(args["plan"].project, args["plan"].scenes, (original,))
    ledger = RightsLedger()
    for binding in {*(ref.permission for ref in actor.references), line.voice.permission}:
        record = reviewed(binding)
        ledger.register(with_review(replace(record, allowed_uses=(*record.allowed_uses, Use("execute"))),
                                    reviewer="fixture-reviewer", evidence=EVIDENCE))
    args.update(sheet=sheet, plan=plan, registry=registry, ledger=ledger,
                current_script=sheet.script.pin())
    speech_adapter = adapter(args, runtime=Runtime(payload=wav_bytes(frames=24000)))
    speech = speech_adapter.synthesize(utterance_id=line.utterance_id, output_id="speech-" + language,
                                      output_revision=1, **args)
    service = S2VServiceContext(S2VBudgetApproval.for_scope(str(root / "budget"), "EU-RO-1", EVIDENCE, gpu_id=gpu_id))
    source = SourcePin(profile.SOURCE_REPOSITORY, profile.SOURCE_REVISION, profile.SOURCE_PATH,
                       profile.SOURCE_SHA256)
    entry = CatalogEntry("s2v-workflow", "r1", "workflow", "Synthetic source review", source,
        source.path, Permissions(), Compatibility(), RightsRef("workflow-rights", "v1"))
    entry = replace(entry, review=SourceReview(entry.digest, "fixture-reviewer",
                                             "private://synthetic/source-review", "f" * 64))
    graph = profile.graph(actor_sha, speech.pin.sha256, language)
    shot = replace(original, prompt=graph["159"]["inputs"]["text"], seed=graph["152"]["inputs"]["seed"],
        model_id=profile.MODEL_ID, model_revision=profile.MODEL_REVISION, workflow_digest=digest(graph))
    model_rights = RightsRef("model-rights", "v1")
    for binding, model in ((replace(image.permission, subject=entry.subject, rights=entry.rights), False),
                           (replace(image.permission, subject=model_subject(), rights=model_rights), True)):
        record = reviewed(binding)
        record = replace(record, license_id="Apache-2.0" if model else "MIT",
            source_revision=profile.MODEL_REVISION if model else profile.SOURCE_REVISION,
            source=EvidenceRef(profile.MODEL_CARD_URI, profile.MODEL_CARD_SHA256) if model
                else EvidenceRef(source.uri, profile.SOURCE_SHA256),
            terms=EvidenceRef(profile.MODEL_TERMS_URI, profile.MODEL_TERMS_SHA256) if model
                else EvidenceRef("private://synthetic/mit", hashlib.sha256(profile.WORKFLOW_LICENSE).hexdigest()),
            allowed_uses=(*record.allowed_uses, Use("execute")))
        ledger.register(with_review(record, reviewer="fixture-reviewer", evidence=EVIDENCE))
    calls = {"actor": 0, "speech": 0, "authorize": 0}
    approved = [None]
    def verify_actor(character, artifact, value, descriptor):
        calls["actor"] += 1
        assert descriptor == ActorDescriptor() and value == payload
        return ActorReviewProof(character, artifact.artifact.sha256, EVIDENCE)
    def verify_speech(value, configured):
        calls["speech"] += 1
        return NativeSpeechProof(value.request.digest(), value.pin.sha256,
            value.request.reference.sha256, "c" * 64, "owned-worker", configured.approval.subject_sha256)
    def authorize(subject):
        calls["authorize"] += 1
        if digest(subject) != approved[0]:
            raise RuntimeError("CANARY approval mismatch")
    gate = VerifiedS2V(service, sheet, plan, sheet.picture, sheet.script.pin(),
        PrivateActorReference(image, path), speech, registry, ledger, SkillCatalog((entry,)),
        bind_shot(shot, (entry.pin,)), model_rights, ActorDescriptor(), verify_actor,
        verify_speech, authorize)
    approved[0] = digest(gate.subject)
    return gate, calls


def worker(gate):
    return WorkerVerification.from_dict({"schema_version": 1, "operator_verified": True,
        "worker_id": "owned-worker", "server_revision": profile.RUNNER_REVISION,
        "image_digest": profile.IMAGE.split("@", 1)[1], "python_version": "3.11.5",
        "torch_version": "2.8.0+cu128", "cuda_version": "12.8",
        "gpu": {"name": gate.service.approval.gpu_id, "backend": "cuda",
                "memory_mib": 32286 if gate.service.approval.gpu_id == profile.GPU_ID else 46000},
        "model_files": [{key: asset[key] for key in ("path", "size_bytes", "sha256")}
                        for asset in profile.model_files()], "custom_nodes_disabled": True})


class S2VGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cf-s2v-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.gate, self.calls = context(self.root)

    def test_exact_four_asset_selection_and_bilingual_graph_with_current_typed_proofs(self):
        selection = select_s2v(self.gate)
        request = selection.request(self.gate.shot)
        graph = selection.workflow.translate(request)
        self.assertEqual(graph, self.gate.graph)
        self.assertEqual(len(selection.declarations["model"]["files"]), 4)
        self.assertEqual(selection.declarations["model"]["required_precision"], "bf16")
        self.assertGreater(self.calls["actor"], 0)
        self.assertGreater(self.calls["speech"], 0)
        self.assertGreater(self.calls["authorize"], 0)
        with self.assertRaises(ComfyPolicyError):
            ApprovedComfySelection(selection.selection_json, selection.graph_json, selection.license_bytes)
        other = self.root / "zh"
        other.mkdir()
        mandarin, _ = context(other, language="zh")
        self.assertEqual(select_s2v(mandarin).request(mandarin.shot).shot.seed, 5202)

    def test_json_flags_fake_speech_duck_gates_and_wrong_proof_types_never_ready(self):
        for gate in (replace(self.gate, verify_speech=lambda *_: {"verified": True}),
                     replace(self.gate, verify_actor=lambda *_: {"visible_mouth": True}),
                     replace(self.gate, authorize=lambda *_: True),
                     replace(self.gate, speech=replace(self.gate.speech,
                                                      runtime_kind="synthetic_fixture"))):
            with self.subTest(gate=type(gate)), self.assertRaises(S2VError):
                select_s2v(gate)
        class Duck:
            def require(self, *_):
                return None
        selection = select_s2v(self.gate)
        template = replace(selection.workflow, conditioning=Duck())
        with self.assertRaises(ProviderError):
            template.translate(ExecutionRequest(self.gate.shot, profile.model()))

    def test_staged_pair_requires_exact_worker_scope_subject_and_unchanged_current_rights(self):
        selection = select_s2v(self.gate)
        verification = worker(self.gate)
        with self.assertRaises(S2VError):
            selection.require_execution_worker(verification)
        staged = StagedS2VPair("owned-worker", self.gate.service.approval.subject_sha256,
            selection.approved_subject_sha256, canonical(self.gate.media).decode())
        bound = selection.bind_worker(verification, staged)
        bound.require_execution_worker(verification)
        for bad in (replace(staged, worker_id="foreign-worker"),
                    replace(staged, scope_sha256="e" * 64),
                    replace(staged, subject_sha256="e" * 64)):
            with self.assertRaises(S2VError):
                selection.bind_worker(verification, bad)
        self.gate.ledger.revoke(self.gate.model_rights, reason="CANARY")
        with self.assertRaises(S2VError):
            bound.require_execution_worker(verification)

    def test_private_reference_replacement_and_unsupported_actor_or_multispeaker_refuse(self):
        for changes in (dict(age="child"), dict(people=True), dict(people=2), dict(pose="profile"),
                        dict(style="cartoon"), dict(mouth="covered"), dict(language_mode="code-switch")):
            with self.assertRaises(S2VError):
                ActorDescriptor(**changes)
        self.gate.actor.path.write_bytes(b"CANARY")
        with self.assertRaises(S2VError) as caught:
            self.gate.preflight()
        self.assertNotIn("CANARY", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_all_hidden_audio_loader_and_explicit_audio_paths_require_exact_firstparty_gate(self):
        graph = {"p": {"class_type": "Text", "inputs": {"text": "prompt"}},
                 "s": {"class_type": "Sampler", "inputs": {"seed": 1}},
                 "out": {"class_type": "Output", "inputs": {"sample": ["s", 0]}}}
        from cineforge.contracts import Shot
        from cineforge.devices import Backend, ModelSpec
        model = ModelSpec("generic", "r1", "text-to-video", frozenset({Backend.CUDA}), 1, "synthetic")
        for node in ({"class_type": "LoadAudio", "inputs": {"audio": "private.wav"}},
                     {"class_type": "Utility", "inputs": {"audio_path": "private.wav"}},
                     {"class_type": "Utility", "inputs": {"reference_audio": "private.wav"}}):
            graph["hidden"] = node
            template = WorkflowTemplate(json.dumps(graph), "generic", "r1",
                frozenset(x["class_type"] for x in graph.values()), InputBinding("p", "text"),
                InputBinding("s", "seed"), "out", "images", "synthetic")
            shot = Shot("generic-shot", "scene", 1, "prompt", 1, 1, "generic", "r1", template.digest)
            with self.assertRaises(ProviderError):
                template.translate(ExecutionRequest(shot, model))


if __name__ == "__main__":
    unittest.main()
