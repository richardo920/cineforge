"""Synthetic checks for test-tier admission and honest execution reports."""

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_tiers.py"
SPEC = importlib.util.spec_from_file_location("test_tiers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TIERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIERS)


class LiveAdmissionTests(unittest.TestCase):
    def admit(self, **changes):
        values = {
            "opt_in": True,
            "max_cost_usd": "1.00",
            "estimated_cost_usd": "0.50",
            "timeout_seconds": "60",
        }
        values.update(changes)
        TIERS.admit_live_run(**values)

    def test_valid_bounded_preflight(self):
        self.admit(estimated_cost_usd="1.00")

    def test_opt_in_is_required(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.admit(opt_in=False)

    def test_over_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.admit(estimated_cost_usd="1.01")

    def test_invalid_limits_are_rejected(self):
        for field in ("max_cost_usd", "estimated_cost_usd", "timeout_seconds"):
            for value in ("NaN", "Infinity", "-Infinity", "nonsense", "-1"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.admit(**{field: value})
        for field in ("max_cost_usd", "timeout_seconds"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit(**{field: "0"})


class TierRunnerTests(unittest.TestCase):
    def run_tier(self, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True, text=True, check=False,
        )
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        return result.returncode, reports

    def test_live_tiers_skip_without_opt_in(self):
        code, reports = self.run_tier("--tier", "integration", "--tier", "media-quality")
        self.assertEqual(code, 0)
        self.assertEqual([report["status"] for report in reports], ["skipped", "skipped"])

    def test_live_opt_in_requires_limits(self):
        code, reports = self.run_tier("--tier", "integration", "--allow-live")
        self.assertEqual(code, 1)
        self.assertEqual(reports[0]["status"], "failed")

    def test_bounded_opt_in_does_not_claim_unimplemented_adapter_ran(self):
        code, reports = self.run_tier(
            "--tier", "integration", "--allow-live", "--max-cost-usd", "1",
            "--estimated-cost-usd", "0.5", "--timeout-seconds", "60",
        )
        self.assertEqual(code, 0)
        self.assertEqual(reports[0]["status"], "skipped")
        self.assertIn("no live calls made", reports[0]["reason"])

    def test_missing_contract_suite_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(TIERS, "ROOT", Path(directory)):
                report = TIERS.run_offline_tier("contract")
        self.assertEqual(report["status"], "skipped")

    def test_default_run_discovers_both_offline_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            runner = root / "scripts" / SCRIPT.name
            shutil.copy2(SCRIPT, runner)
            for suite in ("tests", "tests_contract"):
                (root / suite).mkdir()
                (root / suite / f"test_{suite}_fixture.py").write_text(
                    "import unittest\n"
                    "class SyntheticTests(unittest.TestCase):\n"
                    "    def test_synthetic_case(self):\n"
                    "        self.assertTrue(True)\n",
                    encoding="utf-8",
                )
            result = subprocess.run(
                [sys.executable, str(runner)],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(
            [(report["tier"], report["status"], report["tests_run"]) for report in reports],
            [("offline", "passed", 1), ("contract", "passed", 1)],
        )

    def test_failed_and_all_skipped_suites_fail(self):
        def fail():
            raise AssertionError("synthetic failure")

        def skip():
            raise unittest.SkipTest("synthetic unavailable check")

        for case, status in ((fail, "failed"), (skip, "failed")):
            suite = unittest.TestSuite([unittest.FunctionTestCase(case)])
            with self.subTest(status=status), mock.patch.object(
                TIERS.unittest.TestLoader, "discover", return_value=suite,
            ), mock.patch.object(TIERS.sys, "stderr", io.StringIO()):
                report = TIERS.run_offline_tier("offline")
            self.assertEqual(report["status"], status)

    def test_contract_cli_pass_failure_missing_empty_and_all_skipped(self):
        cases = (
            (None, "skipped", 0),
            ("", "failed", 1),
            ("self.assertTrue(True)", "passed", 0),
            ("self.fail('synthetic contract failure')", "failed", 1),
            ("self.skipTest('synthetic unavailable contract')", "failed", 1),
            ("__import__('os')._exit(0)", "failed", 1),
            ("__import__('os').write(1, b'malformed worker output')", "failed", 1),
            ("__import__('os').write(1, b'{\"tier\":\"contract\",\"status\":\"passed\"}'); "
             "__import__('os')._exit(0)", "failed", 1),
            ("__import__('os').write(1, b'{\"tier\":\"contract\",\"status\":\"skipped\"}'); "
             "__import__('os')._exit(0)", "failed", 1),
        )
        for body, status, code in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "scripts").mkdir()
                runner = root / "scripts" / SCRIPT.name
                shutil.copy2(SCRIPT, runner)
                if body is not None:
                    (root / "tests_contract").mkdir()
                    if body:
                        (root / "tests_contract" / "test_fixture.py").write_text(
                            "import unittest\n"
                            "class SyntheticContract(unittest.TestCase):\n"
                            "    def test_contract(self):\n"
                            f"        {body}\n",
                            encoding="utf-8",
                        )
                result = subprocess.run(
                    [sys.executable, str(runner), "--tier", "contract"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, code, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["tier"], "contract")
                self.assertEqual(report["status"], status)

    def test_same_module_name_does_not_hide_failing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            runner = root / "scripts" / SCRIPT.name
            shutil.copy2(SCRIPT, runner)
            for suite, assertion in (
                ("tests", "self.assertTrue(True)"),
                ("tests_contract", "self.fail('independent contract failure')"),
            ):
                (root / suite).mkdir()
                (root / suite / "test_shared.py").write_text(
                    "import unittest\n"
                    "class SyntheticTests(unittest.TestCase):\n"
                    "    def test_synthetic_case(self):\n"
                    f"        {assertion}\n",
                    encoding="utf-8",
                )
            result = subprocess.run(
                [sys.executable, str(runner), "--tier", "offline", "--tier", "contract"],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 1, result.stderr)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(
            [(report["tier"], report["status"], report["tests_run"]) for report in reports],
            [("offline", "passed", 1), ("contract", "failed", 1)],
        )
        self.assertIn("independent contract failure", result.stderr)

    def test_offline_runner_refuses_live_tiers(self):
        with self.assertRaises(ValueError):
            TIERS.run_offline_tier("integration")


if __name__ == "__main__":
    unittest.main()
