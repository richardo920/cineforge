"""Offline defaults and an explicitly opted-in private ComfyUI entry point."""

import argparse
from typing import Any, NoReturn

from cineforge import __version__
from cineforge.cloud_smoke import (
    cli_inputs,
    load_reference,
    recover_managed,
    recovery_inputs,
    run_managed_smoke,
    ssh_factory,
)
from cineforge.comfy_policy import WorkerVerification, load_comfy_selection
from cineforge.comfy_run import execute_comfy
from cineforge.contracts import ProjectPlan
from cineforge.mock import execute_mock, load_mock_plan
from cineforge.private_store import read_private_input
from cineforge.workflows import canonical as canonical_json
from cineforge.workflows import strict_object


class _ArgumentParser(argparse.ArgumentParser):
    """Never include arbitrary supplied tokens in parser diagnostics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["exit_on_error"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        super().error("invalid command arguments; use --help for syntax")

    def application_error(self, message: str) -> NoReturn:
        """Preserve existing offline domain errors; live callers pass fixed safe text."""
        super().error(message)


def main() -> None:
    parser = _ArgumentParser(prog="cineforge", description="Cineforge production metadata and execution")
    parser.add_argument("--version", action="version", version=f"cineforge {__version__}")
    commands = parser.add_subparsers(dest="command")
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
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed or report is None:
            parser.application_error("cloud command rejected or incomplete; retain private journals")
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
        except (Exception, KeyboardInterrupt):
            failed = True
        if failed:
            parser.application_error(
                "ComfyUI command rejected or incomplete; inspect private run.json if present")
        print(output)
        return
    try:
        plan = load_mock_plan(args.plan)
        output = (plan.plan.to_json() if args.command == "validate-plan" else
                  execute_mock(plan, run_id=args.run_id, output_dir=args.output_dir,
                               resume=args.resume).manifest.to_json())
    except (OSError, ValueError, RecursionError) as exc:
        parser.application_error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
