"""Offline safety and failure-classification tests; no GitHub calls."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protection_probe", ROOT / "scripts/protection_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((ROOT / "config/main-ruleset.json").read_text())

    def test_production_rules_keep_required_sources_and_no_bypass(self):
        self.assertEqual(self.template["conditions"]["ref_name"],
                         {"include": ["refs/heads/main"], "exclude": []})
        self.assertEqual(self.template["enforcement"], "active")
        self.assertEqual(self.template["bypass_actors"], [])
        rules = {rule["type"]: rule.get("parameters", {}) for rule in self.template["rules"]}
        self.assertIn("deletion", rules)
        self.assertIn("non_fast_forward", rules)
        self.assertTrue(rules["pull_request"]["dismiss_stale_reviews_on_push"])
        self.assertTrue(rules["pull_request"]["required_review_thread_resolution"])
        self.assertEqual(rules["pull_request"]["required_approving_review_count"], 0)
        checks = rules["required_status_checks"]
        self.assertTrue(checks["strict_required_status_checks_policy"])
        self.assertEqual({c["context"] for c in checks["required_status_checks"]},
                         {"Python 3.11", "Python 3.12", "Python 3.13", "Python 3.14",
                          "Quality guards", "Dependency review", "Agent evidence"})
        self.assertTrue(all(c["integration_id"] == 15368 for c in checks["required_status_checks"]))

    def test_untrusted_session_cannot_select_production_ref_or_path(self):
        for token in ("main", "refs/heads/main", "../../main", "abc;rm", "", "a" * 33):
            with self.subTest(token=token), self.assertRaises(ValueError):
                probe.probe_names(token)
        base, head = probe.probe_names("ac2f0ff0")
        self.assertNotEqual(base, head)
        self.assertTrue(base.startswith("probe/m0-2-"))

    def test_fixture_does_not_modify_main_template_or_claim_actions_identity(self):
        original = json.dumps(self.template, sort_keys=True)
        fixture = probe.fixture_ruleset(self.template, "ac2f0ff0")
        self.assertEqual(json.dumps(self.template, sort_keys=True), original)
        self.assertEqual(fixture["bypass_actors"], [])
        self.assertEqual(fixture["conditions"]["ref_name"]["include"],
                         ["refs/heads/probe/m0-2-ac2f0ff0-base"])
        rules = {rule["type"]: rule for rule in fixture["rules"]}
        checks = rules["required_status_checks"]["parameters"]["required_status_checks"]
        self.assertEqual(checks, [{"context": name} for name in probe.CONTEXTS])

    def test_transcript_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.jsonl"
            output.write_text("prior session")
            with self.assertRaises(FileExistsError):
                probe.GitHub(output)
            self.assertEqual(output.read_text(), "prior session")


class RejectionTests(unittest.TestCase):
    def test_unrelated_merge_rejection_is_not_success(self):
        api = unittest.mock.Mock()
        api.request.return_value = {"message": "Base branch was modified. Review and try merge again."}
        with patch.object(probe, "ready_pr", return_value={"mergeable_state": "blocked"}):
            with self.assertRaisesRegex(RuntimeError, "did not identify"):
                probe.blocked(api, 42, "a" * 40, "negative case")
        api.record.assert_not_called()

    def test_protection_rejection_is_recorded_after_conflict_free_check(self):
        api = unittest.mock.Mock()
        api.request.return_value = {"message": "Required status check 'M0 probe evidence' is expected."}
        with patch.object(probe, "ready_pr", return_value={"mergeable_state": "blocked"}) as ready:
            probe.blocked(api, 42, "a" * 40, "missing evidence")
        ready.assert_called_once_with(api, 42, "a" * 40, blocked_state=True)
        self.assertEqual(api.request.call_args.kwargs["expected"], (405,))
        api.record.assert_called_once()

    def test_conflict_or_wrong_head_never_reaches_merge_attempt(self):
        api = unittest.mock.Mock()
        api.request.return_value = {"head": {"sha": "b" * 40}, "mergeable": True}
        with patch.object(probe.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "expected head"):
                probe.blocked(api, 42, "a" * 40, "wrong head")
        self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))


if __name__ == "__main__":
    unittest.main()
