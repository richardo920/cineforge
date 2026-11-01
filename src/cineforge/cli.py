"""Offline package entry point."""

import argparse

from cineforge import __version__
from cineforge.mock import execute_mock, load_mock_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Cineforge offline filmmaking metadata")
    parser.add_argument("--version", action="version", version=f"cineforge {__version__}")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate-plan", help="validate a synthetic JSON plan")
    validate.add_argument("plan", help="UTF-8 JSON plan with explicit models and devices")
    mock = commands.add_parser("mock-run", help="emit an offline metadata-only RunManifest")
    mock.add_argument("plan", help="UTF-8 JSON plan with explicit models and devices")
    mock.add_argument("--run-id", help="override the deterministic run identifier")
    mock.add_argument("--output-dir", help="dedicated directory for local metadata/checkpoints")
    mock.add_argument("--resume", action="store_true", help="resume a stored run in --output-dir")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        plan = load_mock_plan(args.plan)
        output = (plan.plan.to_json() if args.command == "validate-plan" else
                  execute_mock(plan, run_id=args.run_id, output_dir=args.output_dir,
                               resume=args.resume).manifest.to_json())
    except (OSError, ValueError, RecursionError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
