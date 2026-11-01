"""Private M4 operator entry points; explicit live opt-in precedes credential access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cineforge import conditioning_profile as profile
from cineforge.cloud_budget import ConditioningBudgetApproval
from cineforge.cloud_smoke import external_api_key
from cineforge.conditioning import (
    ConditioningApproval,
    ConditioningError,
    digest,
    load_context,
    selection_from_approval,
)
from cineforge.conditioning_run import prepare_journal, recover_conditioned, run_conditioned
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.runpod import RunpodPodSpec
from cineforge.runpod_transport import RunpodHTTPTransport
from cineforge.ssh_worker import SSHWorker, ssh_public_key
from cineforge.workflows import canonical, strict_object


def _approval(path: str) -> ConditioningApproval:
    value = strict_object(read_private_input(path))
    if set(value) != {"approval", "subject"}:
        raise ConditioningError("invalid separate M4 approval record")
    return ConditioningApproval(canonical(value["approval"]).decode(),
                                canonical(value["subject"]).decode())


def command(args: Any) -> dict[str, Any]:
    if args.command == "conditioning-preflight":
        gate = load_context(args.context, args.reference)
        subject = gate.subject
        with PrivateRunDirectory(args.output_dir) as store:
            store.checkpoint(canonical({"status": "proposed", "subject": subject,
                "subject_sha256": digest(subject), "identity_review": "pending_human_review"}))
        prepare_journal(gate)
        return {"status": "proposed", "subject_sha256": digest(subject)}
    if args.allow_live is not True:
        raise ConditioningError("M4 cloud commands require explicit live opt-in")
    approval = _approval(args.approval)
    approved_budget = ConditioningBudgetApproval.for_approval(approval)
    if str(Path(args.budget_dir).absolute()) != approved_budget.journal_directory:
        raise ConditioningError("M4 journal path differs from the operator approval")
    read_private_input(Path(args.budget_dir) / "budget.json")
    if args.command == "conditioning-recover":
        transport = RunpodHTTPTransport(external_api_key(args.api_key_file), allow_live=True)
        return recover_conditioned(approved_budget, transport, args.budget_dir, allow_live=True)
    gate = load_context(args.context, args.reference)
    if approval.subject_json != canonical(gate.subject).decode():
        raise ConditioningError("approval differs from current M4 context")
    selection = selection_from_approval(gate, strict_object(approval.record_json))
    # Only after exact context/receipt admission may externally supplied credentials be used.
    spec = RunpodPodSpec(args.session_id, profile.IMAGE, profile.GPU_ID, profile.REGION,
        ssh_public_key(Path(args.ssh_private_key).expanduser()), max_hourly_usd=1.25)
    transport = RunpodHTTPTransport(external_api_key(args.api_key_file), allow_live=True)
    report = run_conditioned(selection, spec, transport, args.budget_dir, args.output_dir,
        lambda budget, state: SSHWorker(spec, selection, budget, transport,
            Path(args.ssh_private_key).expanduser(), state, allow_live=True), allow_live=True)
    return {key: report[key] for key in ("outcome", "identity_review", "release_verified", "costs")}
