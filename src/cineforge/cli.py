"""CLI compatibility contract (RF-20).

Success/help/version/bare invocation: return None (process 0). Usage AND application
failures: SystemExit(2), preserving existing shell and Python callers. Ctrl-C:
SystemExit(130) after bounded owned cleanup/private checkpoint attempts. Historical
run records retain their original outcomes; exit codes describe this invocation.
Diagnostics contain only fixed stage/error codes, never exception text or chains.
"""

import argparse
import sys
from typing import Any, NoReturn

from cineforge import __version__
from cineforge.a20_inputs import INPUT_NAMES, A20Error
from cineforge.cloud_budget import CloudBudgetError
from cineforge.cloud_smoke import (
    CloudSmokeError,
    cli_inputs,
    load_reference,
    recover_managed,
    recovery_inputs,
    run_managed_smoke,
    ssh_factory,
)
from cineforge.comfy_policy import WorkerVerification, load_comfy_selection
from cineforge.comfy_run import ComfyRunError, execute_comfy
from cineforge.conditioning import ConditioningError
from cineforge.contracts import ProjectPlan
from cineforge.mock import execute_mock, load_mock_plan
from cineforge.private_store import ArtifactStoreError, read_private_input
from cineforge.storage import StorageError
from cineforge.workflows import canonical as canonical_json
from cineforge.workflows import strict_object


class _ArgumentParser(argparse.ArgumentParser):
    """Never include arbitrary supplied tokens in parser diagnostics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["exit_on_error"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        super().error("[stage=usage code=CF_USAGE] invalid command arguments; use --help for syntax")

    def application_error(self, message: str, reason: str = "unexpected") -> NoReturn:
        """Only internal constant messages may reach this compatibility path."""
        super().error(f"[stage={_stage()} code=CF_APPLICATION reason={reason}] {message}")


_STAGES = frozenset({"validate-plan", "mock-run", "comfy-run", "cloud-smoke",
                     "cloud-recover", "conditioning-preflight", "conditioning-run",
                     "conditioning-recover", "a20-run"})


def _stage() -> str:
    return sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in _STAGES else "cli"


def _reason(error: Exception) -> str:
    """Closed vocabulary: subclass names, messages and chains are never displayed."""
    if type(error) is A20Error:
        fixed = {"A20_MISSING_" + name.upper(): "a20-missing-" + name for name in INPUT_NAMES}
        return fixed.get(str(error), "a20")
    for known, reason in ((CloudBudgetError, "cloud-budget"), (CloudSmokeError, "cloud-smoke"),
                          (ComfyRunError, "comfy-run"), (ConditioningError, "conditioning"),
                          (ArtifactStoreError, "private-store"), (StorageError, "storage"),
                          (TimeoutError, "timeout"), (OSError, "io"), (ValueError, "invalid-data")):
        if isinstance(error, known):
            return reason
    return "unexpected"


def main() -> None:
    """Preserve normal returns and SystemExit(2); sanitize unexpected failures too."""
    code = ""
    reason = "interrupted"
    exit_code = 2
    try:
        _main()
    except KeyboardInterrupt:
        code, exit_code = "CF_INTERRUPTED", 130
    except Exception as error:
        code, reason = "CF_APPLICATION", _reason(error)
    # Exit outside the handler so no private exception survives as displayed context.
    if code:
        print(f"cineforge: error: [stage={_stage()} code={code} reason={reason}] "
              "command incomplete; retain private journals", file=sys.stderr)
        raise SystemExit(exit_code)


def _main() -> None:
    parser = _ArgumentParser(prog="cineforge", description="Cineforge production metadata and execution",
        epilog="Exit codes: 0 success/help/version/bare invocation; 2 usage or application "
               "failure (compatible); 130 interruption after bounded owned cleanup. "
               "Errors use fixed stage and CF_USAGE/CF_APPLICATION/CF_INTERRUPTED codes; "
               "allowlisted reasons identify domain, IO, invalid data or unexpected failure. "
               "Retain private checkpoints; historical outcomes are unchanged.")
    parser.add_argument("--version", action="version", version=f"cineforge {__version__}")
    commands = parser.add_subparsers(dest="command")
    a20 = commands.add_parser("a20-run", help="separately authorized A20 validation; no defaults")
    a20.add_argument("--inputs", required=True, help="private JSON: " + ", ".join(INPUT_NAMES) + " paths")
    a20.add_argument("--controller", required=True, help="explicit trusted operator Python; build_bindings(inputs)")
    a20.add_argument("--controller-root", required=True, help="complete local controller code/data dependency root")
    a20.add_argument("--controller-sha256", required=True, help="independently reviewed controller source digest")
    a20.add_argument("--output-dir", required=True, help="new private directory outside Git")
    a20.add_argument("--allow-live", action="store_true", help="separate cloud/native/workflow opt-in")
    validate = commands.add_parser("validate-plan", help="validate a synthetic JSON plan")
    validate.add_argument("plan", help="UTF-8 JSON plan with explicit models and devices")
    mock = commands.add_parser("mock-run", help="emit an offline metadata-only RunManifest")
    mock.add_argument("plan", help="UTF-8 JSON plan with explicit models and devices")
    mock.add_argument("--run-id", help="override the deterministic run identifier")
    mock.add_argument("--output-dir", help="dedicated directory for local metadata/checkpoints")
    mock.add_argument("--resume", action="store_true", help="resume a stored run in --output-dir")
    comfy = commands.add_parser("comfy-run", help="one approved shot through private ComfyUI")
    comfy.add_argument("plan", help="single-shot domain ProjectPlan JSON")
    comfy.add_argument("--selection", required=True, help="separately trusted reviewed ledger")
    comfy.add_argument("--worker-verification", required=True, help="operator-verified worker JSON")
    comfy.add_argument("--output-dir", required=True, help="private empty directory outside Git")
    comfy.add_argument("--endpoint", default="http://127.0.0.1:8188", help="loopback tunnel origin")
    comfy.add_argument("--allow-live", action="store_true", help="explicitly permit ComfyUI HTTP")
    comfy.add_argument("--timeout-seconds", type=float, default=600)
    comfy.add_argument("--request-timeout-seconds", type=float, default=10)
    comfy.add_argument("--poll-interval-seconds", type=float, default=1)
    comfy.add_argument("--max-polls", type=int, default=600)
    comfy.add_argument("--max-result-bytes", type=int, default=32 * 1024 * 1024)
    cloud = commands.add_parser("cloud-smoke", help="opt-in bounded approved paper-boat parity test")
    recovery = commands.add_parser("cloud-recover", help="release journal-owned Pods without acquiring")
    for command in (cloud, recovery):
        command.add_argument("--selection", required=True, help="trusted pinned ComfyUI ledger")
        command.add_argument("--proposal", required=True, help="trusted approved M3 scope ledger")
        command.add_argument("--budget-dir", required=True, help="same private durable budget directory")
        command.add_argument("--api-key-file", help="external owned0600 key; default RUNPOD_API_KEY env")
        command.add_argument("--allow-live", action="store_true", help="explicit cloud API opt-in")
    cloud.add_argument("--session-id", required=True, help="stable session for both sequential attempts")
    cloud.add_argument("--ssh-private-key", required=True, help="external owned0600 SSH key")
    cloud.add_argument("--output-dir", required=True, help="new private directory outside Git")
    cloud.add_argument("--reference-run", required=True, help="private prepared-endpoint run.json")
    cloud.add_argument("--case", choices=("success", "post-acquire-failure"), default="success")
    m4_check = commands.add_parser("conditioning-preflight", help="offline private I2V proposal")
    m4_run = commands.add_parser("conditioning-run", help="opt-in approved three-shot I2V test")
    m4_recover = commands.add_parser("conditioning-recover", help="release only M4 journal-owned Pods")
    for command in (m4_check, m4_run):
        command.add_argument("--context", required=True, help="private registry/rights/catalog bindings")
        command.add_argument("--reference", required=True, help="private approved static PNG")
        command.add_argument("--output-dir", required=True, help="new private directory outside Git")
    for command in (m4_run, m4_recover):
        command.add_argument("--approval", required=True, help="separate exact M4 subject/receipt")
        command.add_argument("--budget-dir", required=True, help="new M4 scoped journal; retain for recovery")
        command.add_argument("--api-key-file", help="external programmatic credential")
        command.add_argument("--allow-live", action="store_true", help="explicit cloud/worker opt-in")
    m4_run.add_argument("--session-id", required=True)
    m4_run.add_argument("--ssh-private-key", required=True, help="external owned0600 SSH key")
    parse_failed = False
    args = None
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parse_failed = True
    # Conversion errors may retain the supplied value. Exit only after leaving
    # their handler, so those exceptions never become a displayed error context.
    if parse_failed or args is None:
        parser.error("invalid arguments")
    if args.command is None:
        parser.print_help()
        return
    reason = "unexpected"
    if args.command == "a20-run":
        from cineforge.a20_cli import command as a20_command
        print(canonical_json(a20_command(args)).decode())
        return
    if args.command.startswith("conditioning-"):
        from cineforge.conditioning_cli import command as conditioning_command
        m4_failed = False
        m4_report = None
        try:
            m4_report = conditioning_command(args)
        except Exception as error:
            reason = _reason(error)
            m4_failed = True
        if m4_failed or m4_report is None:
            parser.application_error("conditioning command rejected or incomplete; retain private journals", reason)
        print(canonical_json(m4_report).decode())
        return
    if args.command in ("cloud-smoke", "cloud-recover"):
        failed = False
        report = None
        try:
            if args.command == "cloud-recover":
                approval, spec, transport = recovery_inputs(args.selection, args.proposal,
                    args.budget_dir, args.api_key_file, allow_live=args.allow_live)
                report = recover_managed(spec, approval, transport, args.budget_dir,
                                         allow_live=args.allow_live)
            else:
                selection, approval, spec, transport = cli_inputs(args.selection, args.proposal,
                    args.session_id, args.ssh_private_key, args.api_key_file,
                    allow_live=args.allow_live)
                reference = load_reference(args.reference_run, selection)
                report = run_managed_smoke(selection, approval, spec, transport, args.budget_dir,
                    args.output_dir, reference, ssh_factory(spec, selection, transport,
                    args.ssh_private_key), case=args.case, allow_live=args.allow_live)
        except Exception as error:
            reason = _reason(error)
            failed = True
        if failed or report is None:
            parser.application_error("cloud command rejected or incomplete; retain private journals", reason)
        output_report = {name: report[name] for name in ("release_verified", "costs", "leases")}
        if "guard_stop_verified" in report:
            output_report["guard_stop_verified"] = report["guard_stop_verified"]
        print(canonical_json(output_report).decode())
        return
    if args.command == "comfy-run":
        failed = False
        output = ""
        try:
            comfy_plan = ProjectPlan.from_json(read_private_input(args.plan).decode("utf-8"))
            selection = load_comfy_selection(args.selection)
            worker = WorkerVerification.from_dict(strict_object(
                read_private_input(args.worker_verification)))
            output = execute_comfy(
                comfy_plan, selection, worker, args.output_dir, allow_live=args.allow_live,
                endpoint=args.endpoint, timeout_seconds=args.timeout_seconds,
                request_timeout_seconds=args.request_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds, max_polls=args.max_polls,
                max_result_bytes=args.max_result_bytes,
            ).to_json()
        except Exception as error:
            reason = _reason(error)
            failed = True
        if failed:
            parser.application_error(
                "ComfyUI command rejected or incomplete; inspect private run.json if present", reason)
        print(output)
        return
    offline_failed = False
    output = ""
    try:
        plan = load_mock_plan(args.plan)
        output = (plan.plan.to_json() if args.command == "validate-plan" else
                  execute_mock(plan, run_id=args.run_id, output_dir=args.output_dir,
                               resume=args.resume).manifest.to_json())
    except (OSError, ValueError, RecursionError) as error:
        reason = _reason(error)
        offline_failed = True
    if offline_failed:
        parser.application_error("offline command rejected or incomplete", reason)
    print(output)


if __name__ == "__main__":
    main()
