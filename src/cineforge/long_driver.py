"""One guarded lease, one canonical voice and ordered, locally verified segments.

A lock-only output directory (crash before the first run.json) requires a new
output directory; the shared private store refuses to adopt it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cineforge.a20_driver import LanguageInputs, acquisition_max_cost_usd, attempt_cap
from cineforge.a20_inputs import A20Error, CreativeInputs, FrozenInputs
from cineforge.cloud_budget import CloudBudgetSession
from cineforge.comfy_runtime import CheckedComfyRuntime, ClosedComfyRuntime
from cineforge.interruptions import preserve_interruption, remember_interruption
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_speech import OriginalVoiceDesign
from cineforge.native_speech_integrity import NativeSpeechSelection
from cineforge.paired_media import image_metadata
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.runpod import (
    AcquisitionUncertainError,
    CleanupUncertainError,
    ProvisioningError,
    RunpodPodProvisioner,
    RunpodPodSpec,
)
from cineforge.runpod_transport import CloudTransport
from cineforge.s2v import VerifiedS2V, select_s2v
from cineforge.s2v_run import ATTEMPT_CAP_SECONDS, run_s2v
from cineforge.s2v_service import LongSessionServiceContext, prepare_long_session_journal
from cineforge.speech import QwenBaseSpeechAdapter, SpeechArtifact
from cineforge.ssh_worker import SSHWorker, run_process
from cineforge.workflows import canonical, strict_object

# Recovery planning allowances: shot work per segment, one-time staging/design,
# and lease slack. Admission still enforces the journal-wide aggregate cap.
PER_SEGMENT_SECONDS = 1500
SETUP_SECONDS = 1500
SLACK_SECONDS = 900
SPEECH_MARGIN_SECONDS = 300


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LongCreativeInputs(CreativeInputs):
    """Private UTF-8 JSON array `segments`; zero-based integer `design_index`."""

    def read(self) -> dict[str, bytes]:
        if set(self.paths) != {"segments", "design_index", "actor_png", "voice_design", "seed"}:
            raise A20Error("LONG_INPUT_KEYS")
        data = {name: read_private_input(path, max_bytes=16 * 1024 * 1024)
                for name, path in self.paths.items()}
        lines = json.loads(data["segments"])
        index = json.loads(data["design_index"])
        if (not isinstance(lines, list) or not lines or len(lines) > 10000
                or any(type(line) is not str or not line.strip() or len(line) > 4096
                       or line != line.strip() for line in lines)
                or type(index) is not int or not 0 <= index < len(lines)
                or not data["voice_design"].decode().strip()
                or data["seed"].strip() != b"15001"):
            raise A20Error("LONG_CREATIVE_INPUT")
        image_metadata(data["actor_png"])
        return data


@dataclass(frozen=True)
class LongSessionBindings:
    """Trusted operator composition; callbacks and dependencies belong in frozen roots."""
    service: LongSessionServiceContext
    spec: RunpodPodSpec
    transport: CloudTransport
    ssh_private_key: Path
    acquisition_id: str
    recipe: dict[str, Any]
    sox_wheel: Path
    design: OriginalVoiceDesign
    select_native: Callable[[dict[str, Any], SSHWorker], NativeSpeechSelection]
    check_comfy: Callable[[RemoteNativeSpeechRuntime], CheckedComfyRuntime]
    segment: Callable[[int, bytes], LanguageInputs]
    s2v_gate: Callable[[LanguageInputs, SpeechArtifact, RemoteNativeSpeechRuntime], VerifiedS2V]
    checkpoint: Callable[[str, dict[str, Any]], None]
    ffprobe: Path | None = None


class _DeadlineStopped(A20Error):
    pass


def _verified(segment: dict[str, Any], index: int, line: str, output: Path) -> bool:
    if (segment.get("index") != index or segment.get("status") != "verified"
            or segment.get("line_sha256") != _sha(line.encode())):
        return False
    try:
        for kind in ("speech", "shot"):
            uri = segment[kind + "_uri"]
            path = Path(uri)
            if (not isinstance(uri, str) or path.is_absolute() or ".." in path.parts
                    or str(path) != uri):
                return False
            if _sha(read_private_input(output / path, max_bytes=64 * 1024 * 1024)) != segment[kind + "_sha256"]:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


@preserve_interruption
def run_long_session(b: LongSessionBindings, frozen: FrozenInputs, output: Path, *,
                     absolute_deadline: float, allow_live: bool = False,
                     planned_segments: int | None = None) -> dict[str, Any]:
    """Resume only a released local run; every new lease is journal-wide admission.

    planned_segments caps NEW segments generated this admission, regardless of verified
    segments. None means all remaining lines; otherwise size the lease from the
    smaller of planned_segments and the number of remaining lines.
    """
    if allow_live is not True:
        raise A20Error("LONG_LIVE_OPT_IN_REQUIRED")
    if planned_segments is not None and (type(planned_segments) is not int or planned_segments < 1):
        raise A20Error("LONG_PLANNED_SEGMENTS")
    if not math.isfinite(absolute_deadline):
        raise A20Error("LONG_ORIGINAL_CAP")
    if b.ffprobe is not None:
        try:
            if not b.ffprobe.is_absolute() or not b.ffprobe.is_file():
                raise ValueError
            run_process([str(b.ffprobe), "-version"], timeout=10, max_output=65536,
                        absolute_deadline=absolute_deadline)
        except Exception as error:
            raise A20Error("LONG_FFPROBE_UNAVAILABLE: explicit ffprobe must be an absolute, working file") from error
    frozen.check()
    if type(frozen.creative) is not LongCreativeInputs or type(b.service) is not LongSessionServiceContext:
        raise A20Error("LONG_SCOPE_OR_CREATIVE_INPUT")
    data = frozen.creative.read()
    lines = json.loads(data["segments"])
    design_index = json.loads(data["design_index"])
    b.service.preflight()
    approval, spec = b.service.approval, b.spec
    if (approval.region != "EUR-IS-2" or approval.gpu_id != "NVIDIA L40S"
            or spec.gpu_id != approval.gpu_id or spec.data_center_id != approval.region
            or spec.disk_gb != 150 or spec.min_ram_per_gpu_gb != 64
            or spec.min_cuda_version != "12.8" or spec.max_hourly_usd > 1.25
            or b.design.language != "English" or b.design.text != lines[design_index]
            or b.design.description != data["voice_design"].decode().strip()):
        raise A20Error("LONG_SCOPE_OR_CREATIVE_INPUT")
    if spec.retain_node:
        raise A20Error("LONG_RETAIN_NODE_UNSUPPORTED")
    wheel = read_private_input(b.sox_wheel, max_bytes=16 * 1024 * 1024)

    def configuration() -> bytes:
        # Acquisition id is deliberately per attempt; resume still pins all execution data.
        return canonical({"recipe": b.recipe, "spec": spec.to_dict(),
            "design": asdict(b.design), "approval": approval.subject_sha256})

    config = configuration()
    acquisition_id = b.acquisition_id
    receipt = dict(frozen.receipt, configuration_sha256=_sha(config), staging_wheel_sha256=_sha(wheel))
    record: dict[str, Any] = {"schema_version": 1, "mode": "long-session",
        "outcome": "incomplete", "frozen": receipt, "segments": [], "failures": [],
        "checkpoints": [], "release_verified": True, "human_review": "pending_human_review",
        "export_authority": "none", "budget_notes": [
            "max_attempts counts paid creation admissions journal-wide; a capacity failure consumes one.",
            "Recovery leases use remaining verified work; journal-wide reservations include disk reserve.",
            "acquisition_max_cost_usd is the approved per-acquisition ceiling; admission reserves the requested lease against aggregate_usd."],
        "generation_timing": "elapsed shot execution including technical verification"}
    stage, index = "preflight", None
    with PrivateRunDirectory(output, resume=True) as store:
        if (output / "run.json").exists():
            previous = strict_object(read_private_input(output / "run.json", max_bytes=16 * 1024 * 1024))
            if (previous.get("mode") != "long-session" or previous.get("frozen") != receipt
                    or previous.get("release_verified") is not True
                    or not isinstance(previous.get("segments"), list)):
                raise A20Error("LONG_RESUME_SCOPE_OR_RELEASE")
            previous["budget_notes"] = record["budget_notes"]
            record = previous
        complete = {s["index"]: s for s in record["segments"]
                    if isinstance(s, dict) and type(s.get("index")) is int
                    and 0 <= s["index"] < len(lines)
                    and _verified(s, s["index"], lines[s["index"]], output)}
        record["segments"] = [complete[i] for i in sorted(complete)]
        record.update(outcome="incomplete", human_review="pending_human_review", export_authority="none")
        store.checkpoint(canonical(record))
        if len(complete) == len(lines):
            record["outcome"] = "complete"
            store.checkpoint(canonical(record))
            return record
        lease_seconds = (min(approval.lifetime_seconds, math.ceil(
            (len(lines) - len(complete)) * PER_SEGMENT_SECONDS + SETUP_SECONDS + SLACK_SECONDS))
            if complete else approval.lifetime_seconds)
        if planned_segments is not None:
            lease_seconds = min(approval.lifetime_seconds, math.ceil(
                min(planned_segments, len(lines) - len(complete)) * PER_SEGMENT_SECONDS
                + SETUP_SECONDS + SLACK_SECONDS))
        max_cost_usd = acquisition_max_cost_usd(spec, approval)
        record["budget"] = {"lease_seconds": lease_seconds,
            "acquisition_max_cost_usd": max_cost_usd, "aggregate_usd": approval.aggregate_usd,
            "max_hourly_usd": spec.max_hourly_usd}
        attempt = "attempt-" + uuid.uuid4().hex

        def guard() -> None:
            frozen.check()
            if configuration() != config or b.acquisition_id != acquisition_id or read_private_input(b.sox_wheel, max_bytes=16 * 1024 * 1024) != wheel:
                raise A20Error("LONG_FROZEN_CONFIGURATION_CHANGED")
            if time.monotonic() >= absolute_deadline:
                raise _DeadlineStopped("LONG_ABSOLUTE_DEADLINE")

        bridge: RemoteNativeSpeechRuntime | None = None

        def failure(error: BaseException) -> None:
            remember_interruption(error)
            record["failures"].append({"index": index, "stage": stage,
                "error_type": type(error).__name__, "message": str(error)})
            if stage == "stage" and bridge is not None:
                diagnostic = getattr(bridge, "last_staging_failure", None)
                if type(diagnostic) is dict:
                    record["failures"][-1]["staging_failure"] = copy.deepcopy(diagnostic)
            if stage == "speech" and bridge is not None:
                diagnostic = getattr(bridge, "last_speech_failure", None)
                if type(diagnostic) is dict:
                    record["failures"][-1]["speech_failure"] = copy.deepcopy(diagnostic)
            record["outcome"] = ("deadline_stopped" if stage != "release" and (
                isinstance(error, _DeadlineStopped) or time.monotonic() >= absolute_deadline)
                else "failed")
            store.checkpoint(canonical(record))

        def checkpoint(name: str) -> None:
            guard()
            b.checkpoint(name, strict_object(canonical(record)))
            guard()
            record["checkpoints"].append({"name": name, "attempt": attempt})
            store.checkpoint(canonical(record))

        try:
            guard()
            prepare_long_session_journal(b.service)
            with CloudBudgetSession(approval.journal_directory, approval, b.transport,
                                    guard_deployment_configured=True) as budget:
                with SSHWorker(spec, b.service, budget, b.transport, b.ssh_private_key,
                               output / attempt / "worker", allow_live=allow_live) as worker:
                    provisioner = RunpodPodProvisioner(spec, b.transport, probe=worker.probe,
                        before_create=budget.before_create, event_sink=budget.event_sink,
                        lease_seconds=lease_seconds)
                    worker.set_enforcer(lambda: budget.enforce_limits(provisioner))
                    device = None
                    try:
                        stage = "acquire"
                        guard()
                        record["release_verified"] = False
                        store.checkpoint(canonical(record))
                        try:
                            device = provisioner.acquire(b.service.model,
                                max_cost_usd=max_cost_usd, idempotency_key=b.acquisition_id)
                        except ProvisioningError as error:
                            record["acquisition_snapshot"] = (asdict(error.snapshot)
                                if error.snapshot is not None else None)
                            if not isinstance(error, (AcquisitionUncertainError, CleanupUncertainError)):
                                record["release_verified"] = True
                            raise
                        stage = "stage"
                        guard()
                        native_cap = min(absolute_deadline, time.monotonic() + lease_seconds)
                        bridge = RemoteNativeSpeechRuntime(worker, allow_native=allow_live,
                            timeout_seconds=lease_seconds, max_operations=len(lines) + 1, seed=int(data["seed"].strip()),
                            absolute_deadline=native_cap)
                        record["capabilities"] = bridge.measure_capabilities(absolute_deadline=native_cap)
                        guard()
                        installed = bridge.stage(b.recipe, wheel)
                        checkpoint("postinstall")
                        stage = "select"
                        guard()
                        native = b.select_native(installed, worker)
                        guard()
                        bridge.select(native)
                        checkpoint("select")
                        stage = "design"
                        guard()
                        # Preserve the original design bytes across resumed acquisitions.
                        if "voice_reference" in record:
                            reference = read_private_input(output / record["voice_reference"], max_bytes=64 * 1024 * 1024)
                            if _sha(reference) != record["voice_reference_sha256"]:
                                raise A20Error("LONG_VOICE_REFERENCE_CHANGED")
                        else:
                            reference = bridge.preview(b.design)
                        with PrivateRunDirectory(output / attempt / "payloads") as payloads:
                            record["voice_reference"] = attempt + "/payloads/" + payloads.put_payload(reference, "wav")
                            record["voice_reference_sha256"] = _sha(reference)
                            checkpoint("design")
                            verification = None
                            voices = record.get("canonical_voices_sha256")
                            generated = 0
                            for index, line in enumerate(lines):
                                if index in complete:
                                    continue
                                stage = "speech"
                                if planned_segments is not None and generated >= planned_segments:
                                    raise _DeadlineStopped("LONG_PLANNED_SEGMENTS_REACHED")
                                guard()
                                if (budget.remaining_seconds(b.acquisition_id) - 120
                                        < ATTEMPT_CAP_SECONDS + SPEECH_MARGIN_SECONDS):
                                    raise _DeadlineStopped("LONG_SEGMENT_LEASE_EXHAUSTED")
                                context = b.segment(index, reference)
                                utterance = next((u for u in context.sheet.script.utterances
                                    if u.utterance_id == context.utterance_id), None)
                                if utterance is None or utterance.language != "en" or utterance.text != line:
                                    raise A20Error("LONG_CREATIVE_BINDING")
                                if context.reference.transcript != b.design.text:
                                    raise A20Error("A20_REFERENCE_TRANSCRIPT_CHANGED")
                                voice_hash = _sha(canonical([asdict(v) for v in context.voices]))
                                if voices is not None and voices != voice_hash:
                                    raise A20Error("A20_CANONICAL_VOICE_CHANGED")
                                voices = voice_hash
                                record["canonical_voices_sha256"] = voices
                                guard()
                                if not bridge.speech_health_check():
                                    raise A20Error("LONG_SPEECH_HEALTH_CHECK")
                                speech = QwenBaseSpeechAdapter(context.voices, runtime=bridge).synthesize(
                                    context.sheet, context.utterance_id, context.reference, reference,
                                    plan=context.plan, registry=context.registry, ledger=context.ledger,
                                    current_picture=context.picture, current_script=context.sheet.script.pin(),
                                    output_id=f"long-speech-{index}", output_revision=1)
                                stage = "s2v_gate"
                                guard()
                                gate = b.s2v_gate(context, speech, bridge)
                                if (gate.speech is not speech or gate.service != b.service
                                        or gate.actor.read() != data["actor_png"]
                                        or gate.verify_speech != bridge.verify_speech or gate.sheet != context.sheet):
                                    raise A20Error("LONG_S2V_BINDING")
                                selection = select_s2v(gate)
                                stage = "prepare"
                                guard()
                                if verification is None:
                                    closed = ClosedComfyRuntime(bridge, b.check_comfy(bridge))
                                    guard()
                                    verification = worker.prepare(selection, runtime=closed,
                                        absolute_deadline=min(native_cap, time.monotonic() + budget.remaining_seconds(b.acquisition_id) - 120))
                                stage = "shot"
                                guard()
                                pair = worker.stage_s2v(selection, absolute_deadline=native_cap)
                                guard()
                                bound = selection.bind_worker(verification, pair)
                                cap = attempt_cap(native_cap, budget.remaining_seconds(b.acquisition_id), time.monotonic())
                                started = time.monotonic()
                                shot_dir = output / attempt / f"segment-{index}"
                                result = run_s2v(bound, verification, worker, shot_dir,
                                    allow_live=allow_live, absolute_deadline=cap, ffprobe=b.ffprobe)
                                elapsed = time.monotonic() - started
                                stage = "verify"
                                guard()
                                # run_s2v performs technical_result before returning; bind its
                                # checked hash to the exact bytes copied into the session store.
                                artifact, = result["manifest"]["artifacts"]
                                if (result["outcome"] != "pending_human_review"
                                        or result["technical_result"]["video_sha256"] != artifact["sha256"]
                                        or artifact["uri"] != "artifact-" + artifact["sha256"] + ".webm"):
                                    raise A20Error("LONG_SHOT_NOT_VERIFIED")
                                shot = read_private_input(shot_dir / "shot" / artifact["uri"], max_bytes=64 * 1024 * 1024)
                                if _sha(shot) != artifact["sha256"]:
                                    raise A20Error("LONG_SHOT_HASH_CHANGED")
                                stage = "pull"
                                guard()
                                speech_uri = payloads.put_payload(speech.payload, "wav")
                                shot_uri = payloads.put_payload(shot, "webm")
                                facts = {"index": index, "line_sha256": _sha(line.encode()),
                                    "speech_uri": attempt + "/payloads/" + speech_uri, "speech_sha256": _sha(speech.payload),
                                    "shot_uri": attempt + "/payloads/" + shot_uri, "shot_sha256": _sha(shot),
                                    "generation_seconds": elapsed, "status": "verified"}
                                complete[index] = facts
                                record["segments"] = [complete[i] for i in sorted(complete)]
                                stage = "checkpoint"
                                checkpoint(f"segment-{index}")
                                generated += 1
                            record["outcome"] = "complete"
                            store.checkpoint(canonical(record))
                    except (Exception, KeyboardInterrupt) as error:
                        failure(error)
                    finally:
                        if device is not None:
                            try:
                                provisioner.release(device.device_id)
                                record["release_verified"] = True
                            except (Exception, KeyboardInterrupt) as error:
                                stage = "release"
                                failure(error)
                            finally:
                                store.checkpoint(canonical(record))
        except (Exception, KeyboardInterrupt) as error:
            failure(error)
        store.checkpoint(canonical(record))
    return record
