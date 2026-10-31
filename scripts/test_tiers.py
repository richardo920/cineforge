"""Run offline tests and report unavailable live tiers without executing them."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("offline", "contract", "integration", "media-quality")


def admit_live_run(
    *,
    opt_in: bool,
    max_cost_usd: str,
    estimated_cost_usd: str,
    timeout_seconds: str,
) -> None:
    """Validate preflight limits; an adapter must enforce actual spend/deadlines."""
    if not opt_in:
        raise ValueError("live tests require explicit opt-in")
    try:
        maximum = Decimal(max_cost_usd)
        estimate = Decimal(estimated_cost_usd)
        deadline = Decimal(timeout_seconds)
    except InvalidOperation as exc:
        raise ValueError("cost and timeout limits must be numeric") from exc
    if not all(value.is_finite() for value in (maximum, estimate, deadline)):
        raise ValueError("cost and timeout limits must be finite")
    if maximum <= 0 or estimate < 0 or deadline <= 0:
        raise ValueError("budget and timeout must be positive; estimate cannot be negative")
    if estimate > maximum:
        raise ValueError("estimated cost exceeds the approved test budget")


def run_offline_tier(tier: str) -> dict[str, object]:
    """Live suites live outside default discovery and cannot enter this runner."""
    if tier not in ("offline", "contract"):
        raise ValueError("only offline and contract suites can execute")
    directory = ROOT / ("tests" if tier == "offline" else "tests_contract")
    if not directory.is_dir():
        return {"tier": tier, "status": "skipped", "reason": "no suite implemented"}
    suite = unittest.defaultTestLoader.discover(str(directory))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    status = "passed" if result.wasSuccessful() else "failed"
    if result.wasSuccessful() and result.testsRun == len(result.skipped):
        status = "skipped"
    return {
        "tier": tier,
        "status": status,
        "tests_run": result.testsRun,
        "tests_skipped": len(result.skipped),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", choices=TIERS)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--estimated-cost-usd")
    parser.add_argument("--timeout-seconds")
    args = parser.parse_args()
    failed = False
    for tier in dict.fromkeys(args.tier or ("offline", "contract")):
        if tier in ("offline", "contract"):
            report = run_offline_tier(tier)
        elif not args.allow_live:
            report = {"tier": tier, "status": "skipped", "reason": "live tests not opted in"}
        else:
            try:
                if any(value is None for value in (
                    args.max_cost_usd, args.estimated_cost_usd, args.timeout_seconds,
                )):
                    raise ValueError("live opt-in requires budget, estimate, and timeout")
                admit_live_run(
                    opt_in=args.allow_live,
                    max_cost_usd=args.max_cost_usd,
                    estimated_cost_usd=args.estimated_cost_usd,
                    timeout_seconds=args.timeout_seconds,
                )
                report = {
                    "tier": tier,
                    "status": "skipped",
                    "reason": "no bounded live adapter implemented; no live calls made",
                }
            except ValueError as exc:
                report = {"tier": tier, "status": "failed", "reason": str(exc)}
        failed |= report["status"] == "failed"
        print(json.dumps(report, sort_keys=True))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
