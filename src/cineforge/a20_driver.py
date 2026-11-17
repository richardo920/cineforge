"""A20 orchestration over guarded components; operator proofs remain trusted code.

No default runtime, creative inputs, proof verifiers, or live opt-in. One acquisition,
one prepare, two language intents; operator retention may skip owned release.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cineforge.a20_delivery import mux_delivery
from cineforge.a20_inputs import A20Error, FrozenInputs
from cineforge.audio import CueSheet, RevisionPin
from cineforge.audio_analysis import FFmpegTool, ToolIdentity
from cineforge.audio_mix import DeliveryPreset, MixSettings, mix_scene
from cineforge.characters import CharacterRegistry
from cineforge.cloud_budget import CloudBudgetSession
from cineforge.comfy_runtime import CheckedComfyRuntime, ClosedComfyRuntime
from cineforge.contracts import ProjectPlan
from cineforge.interruptions import preserve_interruption, remember_interruption
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.native_speech_integrity import NativeSpeechSelection
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.rights import EvidenceRef, RightsLedger
from cineforge.runpod import RunpodPodProvisioner, RunpodPodSpec
from cineforge.runpod_transport import CloudTransport
from cineforge.s2v import VerifiedS2V, select_s2v
from cineforge.s2v_run import run_s2v
from cineforge.s2v_service import S2VServiceContext, prepare_service_journal
from cineforge.speech import (
    CanonicalVoiceBinding,
    QwenBaseSpeechAdapter,
    SpeechArtifact,
    SyntheticReferenceInput,
)
from cineforge.ssh_worker import SSHWorker
from cineforge.stem_import import import_original_stem
from cineforge.worker_guard import WorkerGuardPlan
from cineforge.workflows import canonical, strict_object

CHECKPOINTS = ("postinstall", "select", "design", "EN", "ZH")


@dataclass(frozen=True)
class StemInput:
    name: str
    cue_id: str
    pin: RevisionPin
    review: EvidenceRef


@dataclass(frozen=True)
class LanguageInputs:
    """Approved domain bindings; factory may bind the actual designed reference hash."""
    sheet: CueSheet
    utterance_id: str
    reference: SyntheticReferenceInput
    plan: ProjectPlan
    registry: CharacterRegistry
    ledger: RightsLedger
    picture: RevisionPin
    voices: tuple[CanonicalVoiceBinding, ...]
    stems: tuple[StemInput, ...]
    mix_settings: MixSettings
    preset: DeliveryPreset


@dataclass(frozen=True)
class A20Bindings:
    """Explicit operator Python composition, never a verifier selected by creative JSON.

Callbacks review/authorize existing facts; they must not perform runtime effects.
All their local code/data dependencies must reside in the frozen controller root.
"""
    service: S2VServiceContext
    spec: RunpodPodSpec
    transport: CloudTransport
    ssh_private_key: Path
    acquisition_id: str
    recipe: dict[str, Any]
    sox_wheel: Path
    design: OriginalVoiceDesign
    select_native: Callable[[dict[str, Any], SSHWorker], NativeSpeechSelection]
    check_comfy: Callable[[RemoteNativeSpeechRuntime], CheckedComfyRuntime]
    language: Callable[[str, bytes], LanguageInputs]
    s2v_gate: Callable[[LanguageInputs, SpeechArtifact, RemoteNativeSpeechRuntime], VerifiedS2V]
    checkpoint: Callable[[str, dict[str, Any]], None]
    ffmpeg: Path
    ffprobe: Path
    tool_identity: ToolIdentity


class CoreProbe:
    """Closed persisted-facts diagnostic. No worker/bridge/transport reference is accepted."""
    def __init__(self) -> None:
        self.invocations = 0

    def run(self, *, deadline: float, prep: dict[str, Any], staged: dict[str, Any] | None,
            frozen: dict[str, Any], snapshot: Callable[[], Any],
            clock: Callable[[], float] | None = None) -> dict[str, Any]:
        if clock is None:
            clock = time.monotonic
        if self.invocations:
            raise A20Error("A20_CORE_ALREADY_CONSUMED")
        self.invocations += 1
        reads: list[str] = []
        findings = {"failed_step": "comfy-prepare", "failure_code": "A20_CORE_CAP_EXHAUSTED",
                    "staged_selection_present": False, "frozen_receipt_present": False,
                    "journal_state": "not-read"}
        result = {"probe": "core", "trigger": "comfy-prepare-failure", "invocation": 1,
                  "reads": reads, "findings": findings,
                  "verdict": "diagnostic-only; no retry; proceed to cleanup"}
        try:
            if clock() >= deadline:
                return result
            reads.append("prep_attempt")
            findings["failure_code"] = prep["failure_code"]
            if clock() >= deadline:
                return result
            reads.append("staged_selection_manifest")
            findings["staged_selection_present"] = staged is not None
            if clock() >= deadline:
                return result
            reads.append("frozen_code_data_receipt")
            findings["frozen_receipt_present"] = bool(frozen)
            if clock() >= deadline:
                return result
            reads.append("budget_lifecycle_snapshot")
            lifecycle = snapshot()
            findings["journal_state"] = "absent" if lifecycle is None else lifecycle.state
        except (Exception, KeyboardInterrupt) as error:
            remember_interruption(error)
            findings["failure_code"] = "A20_CORE_PROBE_ERROR"
            findings["journal_state"] = "probe-error"
        return result


def attempt_cap(parent: float, remaining: float, now: float) -> float:
    """600 s hold means execution attempt cap, retaining the separate lease reserve."""
    if (any(not math.isfinite(v) for v in (parent, remaining, now))
            or remaining <= 150 or parent <= now):
        raise A20Error("A20_ATTEMPT_CAP_EXHAUSTED")
    return min(parent, now + 600, now + remaining - 120)


def _language_inputs(b: A20Bindings, frozen: FrozenInputs, language: str,
                     reference: bytes) -> LanguageInputs:
    value = b.language(language, reference)
    data = frozen.creative.read()
    utterance = next((u for u in value.sheet.script.utterances
                      if u.utterance_id == value.utterance_id), None)
    if (utterance is None or utterance.language != language
            or utterance.text != data[language + "_line"].decode("utf-8").strip()
            or {s.name for s in value.stems} != {"music", "ambience", "foley"}
            or len(value.stems) != 3):
        raise A20Error("A20_CREATIVE_BINDING")
    return value


@preserve_interruption
def run_a20(b: A20Bindings, frozen: FrozenInputs, output: Path, *,
            absolute_deadline: float, allow_live: bool = False) -> dict[str, Any]:
    """Run (a)-(l); budget before_create persists reservation before acquisition POST.

    Live execution is separately authorized by the caller. Do not run this during
    implementation validation. Output is a private review candidate, never approval.
    """
    if allow_live is not True:
        raise A20Error("A20_LIVE_OPT_IN_REQUIRED")
    if not math.isfinite(absolute_deadline) or absolute_deadline <= time.monotonic():
        raise A20Error("A20_ORIGINAL_CAP")
    frozen.check()
    data = frozen.creative.read()
    b.service.preflight()
    approval, spec = b.service.approval, b.spec
    if (approval.region != "EUR-IS-2" or approval.gpu_id != "NVIDIA L40S"
            or spec.gpu_id != approval.gpu_id or spec.data_center_id != approval.region
            or spec.disk_gb != 150 or spec.min_ram_per_gpu_gb != 64
            or spec.min_cuda_version != "12.8" or spec.max_hourly_usd > 1.25
            or b.design.description != data["voice_design"].decode("utf-8").strip()
            or b.design.text != data[{"English": "en_line", "Chinese": "zh_line"}[
                b.design.language]].decode("utf-8").strip()
            or b.tool_identity.kind != "native" or b.tool_identity.version != "9.0.2"):
        raise A20Error("A20_SCOPE_OR_CREATIVE_INPUT")
    # Freeze controller configuration and the actual staging wheel too. These are
    # execution data, separate from the eight mandatory creative payloads.
    wheel = read_private_input(b.sox_wheel, max_bytes=16 * 1024 * 1024)
    def configuration() -> bytes:
        return canonical({"recipe": b.recipe, "spec": spec.to_dict(),
            "design": asdict(b.design), "tool": b.tool_identity.to_dict(),
            "acquisition_id": b.acquisition_id, "approval": approval.subject_sha256})
    config = configuration()
    probe = CoreProbe()
    record: dict[str, Any] = {"schema_version": 1, "mode": "a20-validation",
        "outcome": "incomplete", "frozen": frozen.receipt, "checkpoints": [],
        "languages": {}, "release_verified": False, "core_probe": None,
        "human_review": "pending_human_review", "export_authority": "none"}
    record["frozen"].update(configuration_sha256=hashlib.sha256(config).hexdigest(),
                            staging_wheel_sha256=hashlib.sha256(wheel).hexdigest())
    failed = False
    with PrivateRunDirectory(output) as store:
        store.checkpoint(canonical(record))  # receipt is durable BEFORE acquisition
        prepare_service_journal(b.service)
        with CloudBudgetSession(approval.journal_directory, approval, b.transport,
                                guard_deployment_configured=True) as budget:
            with SSHWorker(spec, b.service, budget, b.transport, b.ssh_private_key,
                           output / "worker", allow_live=allow_live) as worker:
                provisioner = RunpodPodProvisioner(spec, b.transport, probe=worker.probe,
                    before_create=budget.before_create, event_sink=budget.event_sink,
                    lease_seconds=approval.lifetime_seconds)  # DELETE default remains 30s
                worker.set_enforcer(lambda: budget.enforce_limits(provisioner))

                def guard() -> None:
                    frozen.check()
                    if (configuration() != config or read_private_input(b.sox_wheel,
                            max_bytes=16 * 1024 * 1024) != wheel):
                        raise A20Error("A20_FROZEN_CONFIGURATION_CHANGED")
                    if time.monotonic() >= absolute_deadline:
                        raise A20Error("A20_ORIGINAL_CAP_EXHAUSTED")

                def checkpoint(name: str) -> None:
                    guard()
                    preview = canonical(record)
                    # Full accumulated facts, copied across the operator boundary.
                    # Callback edits cannot alter driver evidence or approval state.
                    b.checkpoint(name, strict_object(preview))
                    guard()
                    facts = {"name": name, "frozen": record["frozen"],
                             "preview": store.put_payload(preview, "json"),
                             "preview_sha256": hashlib.sha256(preview).hexdigest()}
                    record["checkpoints"].append(facts)
                    store.checkpoint(canonical(record))

                device = None
                try:
                    guard()
                    device = provisioner.acquire(b.service.model, max_cost_usd=.635,
                                                 idempotency_key=b.acquisition_id)
                    guard()
                    # stage() includes native_preflight BEFORE installing runtime.
                    native_cap = min(absolute_deadline, time.monotonic() + 1200)
                    record["original_native_comfy_deadline"] = native_cap
                    bridge = RemoteNativeSpeechRuntime(worker, allow_native=allow_live,
                        timeout_seconds=1200, seed=int(data["seed"].strip()),
                        absolute_deadline=native_cap)
                    # Normal GPU/native-file probe, once BEFORE staging. This is
                    # runtime work and is never the failure-only Core diagnostic.
                    record["capabilities"] = bridge.measure_capabilities(absolute_deadline=native_cap)
                    guard()
                    installed = bridge.stage(b.recipe, wheel)
                    record["native_installation"] = installed
                    checkpoint("postinstall")
                    native = b.select_native(installed, worker)
                    guard()
                    bridge.select(native)
                    checkpoint("select")
                    reference = bridge.preview(b.design)
                    reference_uri = store.put_payload(reference, "wav")
                    record["voice_reference"] = reference_uri
                    checkpoint("design")
                    verification = None
                    comfy_cap = native_cap
                    voices = None
                    generated: dict[str, Any] = {}
                    for language, boundary in (("en", "EN"), ("zh", "ZH")):
                        guard()
                        context = _language_inputs(b, frozen, language, reference)
                        if context.reference.transcript != b.design.text:
                            raise A20Error("A20_REFERENCE_TRANSCRIPT_CHANGED")
                        if voices is not None and voices != context.voices:
                            raise A20Error("A20_CANONICAL_VOICE_CHANGED")
                        voices = context.voices
                        adapter = QwenBaseSpeechAdapter(voices, runtime=bridge)
                        guard()
                        speech = adapter.synthesize(context.sheet, context.utterance_id,
                            context.reference, reference, plan=context.plan, registry=context.registry,
                            ledger=context.ledger, current_picture=context.picture,
                            current_script=context.sheet.script.pin(),
                            output_id="a20-speech-" + language, output_revision=1)
                        record["languages"][language] = {"speech": store.put_payload(speech.payload, "wav")}
                        checkpoint(boundary)  # after synthesize, before that language's S2V
                        gate = b.s2v_gate(context, speech, bridge)
                        if (gate.speech is not speech or gate.service != b.service
                                or gate.actor.read() != data["actor_png"]
                                or gate.verify_speech != bridge.verify_speech
                                or gate.sheet != context.sheet):
                            raise A20Error("A20_S2V_BINDING")
                        selection = select_s2v(gate)
                        guard()
                        if verification is None:
                            closed = ClosedComfyRuntime(bridge, b.check_comfy(bridge))
                            guard()
                            prep_cap = min(native_cap, time.monotonic()
                                           + budget.remaining_seconds(b.acquisition_id) - 120)
                            comfy_cap = prep_cap
                            prep = {"attempt": 1, "outcome": "attempted",
                                    "failure_code": "A20_COMFY_PREPARE_FAILED",
                                    "original_deadline": prep_cap}
                            record["prep_attempt"] = prep
                            store.checkpoint(canonical(record))
                            try:
                                verification = worker.prepare(selection, runtime=closed,
                                                              absolute_deadline=prep_cap)
                            except (Exception, KeyboardInterrupt) as error:
                                # Do not call guard, bridge or any runtime probe here.
                                # Pair transfer requires successful prepare: no staged pair yet.
                                remember_interruption(error)
                                prep["outcome"] = "failed"
                                prep["failure_type"] = ("interrupted" if isinstance(error, KeyboardInterrupt)
                                                        else "exception")
                                record["core_probe"] = probe.run(deadline=prep_cap, prep=prep,
                                    staged=None, frozen=record["frozen"],
                                    snapshot=lambda: budget.snapshot_for(b.acquisition_id))
                                store.checkpoint(canonical(record))
                                raise
                            prep["outcome"] = "succeeded"
                        guard()
                        pair = worker.stage_s2v(selection, absolute_deadline=comfy_cap)
                        guard()
                        bound = selection.bind_worker(verification, pair)
                        cap = attempt_cap(comfy_cap, budget.remaining_seconds(b.acquisition_id),
                                          time.monotonic())
                        result = run_s2v(bound, verification, worker, output / language,
                                         allow_live=allow_live, absolute_deadline=cap)
                        guard()
                        record["languages"][language].update(result=result, hold_deadline=cap)
                        store.checkpoint(canonical(record))
                        generated[language] = (context, speech, result)
                    # Retrieval is owned by synthesize()/run_s2v(); never resubmit a job.
                    for language in ("en", "zh"):
                        guard()
                        context, speech, result = generated[language]
                        sources = [speech]
                        for stem in context.stems:
                            sources.append(import_original_stem(context.sheet, stem.cue_id,
                                data[stem.name], stem.pin, ledger=context.ledger,
                                original_source_review=stem.review))
                        tool = FFmpegTool(b.ffmpeg)
                        guard()
                        mixed = mix_scene(context.sheet, tuple(sources), context.mix_settings,
                            context.preset, plan=context.plan, registry=context.registry,
                            ledger=context.ledger, current_picture=context.picture,
                            current_script=context.sheet.script.pin(), output_id="a20-mix-" + language,
                            output_revision=1, tool=tool, expected_tool=b.tool_identity)
                        guard()
                        soundtrack = store.put_payload(mixed.payload, "wav")
                        artifacts = result["manifest"]["artifacts"]
                        if len(artifacts) != 1:
                            raise A20Error("A20_RETRIEVAL")
                        video = output / language / "shot" / artifacts[0]["uri"]
                        # No URI from a receipt may escape its exact content-addressed name.
                        if artifacts[0]["uri"] != "artifact-" + artifacts[0]["sha256"] + ".webm":
                            raise A20Error("A20_RETRIEVAL")
                        delivery = mux_delivery(video, mixed, store, ffmpeg=b.ffmpeg,
                            ffprobe=b.ffprobe, expected_tool=b.tool_identity, deadline=absolute_deadline)
                        if delivery["video_sha256"] != artifacts[0]["sha256"]:
                            raise A20Error("A20_DELIVERY_VIDEO_CHANGED")
                        record["languages"][language].update(soundtrack=soundtrack, delivery=delivery)
                        guard()
                        store.checkpoint(canonical(record))
                    guard()
                except (Exception, KeyboardInterrupt) as error:
                    remember_interruption(error)
                    failed = True
                finally:
                    # Worker.close is insufficient for release. Retention is explicit
                    # and applies even after a checkpoint or runtime failure.
                    if device is not None:
                        if spec.retain_node:
                            retained: dict[str, Any] = {"pod_id": device.device_id,
                                "region": spec.data_center_id, "gpu_id": spec.gpu_id,
                                "state": None, "hourly_usd": None, "driver_version": None,
                                "driver_mounts": [row["driver_mount"] for row in
                                    record.get("capabilities", {}).get("files", [])
                                    if "driver_mount" in row],
                                "ssh": worker.connection_info}
                            record["retained_node"] = retained
                            old_plan: WorkerGuardPlan | None
                            try:
                                old_plan = budget.guard_plan(b.acquisition_id)
                            except Exception:
                                old_plan = None
                            facts: dict[str, float] = {}
                            try:
                                facts = budget.extend_retention(b.acquisition_id)
                                result = worker.retain_live(facts)
                                retained.update(facts)
                                retained.update(result)
                            except (Exception, KeyboardInterrupt) as error:
                                remember_interruption(error)
                                retained.update(retention_heartbeat="failed",
                                    retention_expires_unix=None, lease_expires_unix=None,
                                    guard_stop_at_unix=None, guard_idle_seconds=None)
                                for name in ("retention_acquired_unix", "retention_bound_seconds"):
                                    if name in facts:
                                        retained[name] = facts[name]
                                stop_at = getattr(worker, "retention_guard_stop_at_unix", None)
                                if type(stop_at) in (int, float):
                                    retained.update(guard_stop_at_unix=stop_at,
                                        guard_idle_seconds=budget.idle_seconds)
                                elif (old_plan is not None
                                        and getattr(worker, "retention_guard_alive", None) is True):
                                    retained.update(guard_stop_at_unix=old_plan.stop_at_unix,
                                        guard_idle_seconds=old_plan.idle_seconds)
                                if getattr(worker, "retention_guard_alive", None) is False:
                                    try:
                                        provisioner.release(device.device_id)
                                        record["release_verified"] = True
                                    except (Exception, KeyboardInterrupt) as release_error:
                                        remember_interruption(release_error)
                                record["cleanup_error"] = "A20_LIVE_RETENTION_FAILED"
                                failed = True
                            try:
                                snap = provisioner.inspect_lease(device.device_id)
                                retained.update(state=snap.state, hourly_usd=
                                    None if snap.state == "released" else snap.hourly_usd)
                            except (Exception, KeyboardInterrupt) as error:
                                remember_interruption(error)
                                if isinstance(error, KeyboardInterrupt):
                                    failed = True
                            if type(retained.get("retention_window_seconds_effective")) in (int, float):
                                rate = retained["hourly_usd"]
                                retained["max_retention_cost_usd"] = (
                                    spec.max_hourly_usd if rate is None else rate
                                ) * retained["retention_window_seconds_effective"] / 3600
                        else:
                            try:
                                provisioner.release(device.device_id)
                                record["release_verified"] = True
                            except (Exception, KeyboardInterrupt) as error:
                                remember_interruption(error)
                                record["cleanup_error"] = "A20_RELEASE_UNCERTAIN"
                                failed = True
                    record["costs"] = budget.costs()
                    store.checkpoint(canonical(record))
        # Worker/budget finalizers must complete before candidate readiness.
        guard()
        lifecycle_complete = ("retained_node" in record if spec.retain_node
                              else record["release_verified"])
        if not failed and lifecycle_complete:
            record["outcome"] = "pending_human_review"
        store.checkpoint(canonical(record))
    if failed or not lifecycle_complete:
        raise A20Error("A20_INCOMPLETE_RETAIN_PRIVATE_RUN")
    return record
