"""Synthetic authentication and acceptance failures; no GitHub or credentials."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_evidence.py"
SPEC = importlib.util.spec_from_file_location("agent_evidence", SCRIPT)
assert SPEC and SPEC.loader
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)

KEY = b"x" * 32  # Deliberately non-secret synthetic key.
NOW = 1_800_000_000_000_000_000
REPO = "example/cineforge"
HEAD = "a" * 40
BASE = "b" * 40
PR_URL = f"https://github.com/{REPO}/pull/12"


def payload():
    reports = []
    for index, role in enumerate(("implementation", "review", "testing"), start=1):
        reports.append({
            "report_url": f"{PR_URL}#issuecomment-{index}",
            "report": {
                "version": 1, "task_id": "M0.3", "role": role, "agent_id": f"agent-{role}",
                "pr_url": PR_URL, "head_sha": HEAD, "base_sha": BASE, "decision": "pass",
                "acceptance": [{"criterion": "synthetic check", "status": "pass",
                                "evidence": "offline test passed"}],
                "findings": [],
            },
        })
    return {"repository": REPO, "pr_number": 12, "head_sha": HEAD, "base_sha": BASE,
            "coordinator_agent_id": "agent-coordinator", "reports": reports}


def report_comments(data):
    return [{"id": index, "body": "Report:\n\n```json\n" + json.dumps(entry["report"]) + "\n```"}
            for index, entry in enumerate(data["reports"], start=1)]


class EvidenceTests(unittest.TestCase):
    def bundle(self, data=None, issued=NOW):
        return evidence.sign_payload(payload() if data is None else data, KEY, issued_at_ns=issued)

    def verify(self, bundle, **overrides):
        context = {"repository": REPO, "pr_number": 12, "head_sha": HEAD,
                   "base_sha": BASE, "now_ns": NOW}
        context.update(overrides)
        return evidence.verify_bundle(bundle, KEY, **context)

    def test_valid_bundle_and_canonical_order(self):
        original = payload()
        self.assertEqual(self.verify(self.bundle())["head_sha"], HEAD)
        reversed_order = dict(reversed(list(original.items())))
        self.assertEqual(self.bundle(original)["signature"], self.bundle(reversed_order)["signature"])

    def test_bad_signature_missing_key_and_tampered_body_fail(self):
        for mutation in (
            lambda b: b.update(signature="0" * 64),
            lambda b: b.pop("signature"),
            lambda b: b["payload"].update(head_sha="c" * 40),
            lambda b: b.update(version=True),
        ):
            bundle = self.bundle()
            mutation(bundle)
            with self.subTest(bundle=bundle), self.assertRaises(evidence.EvidenceError):
                self.verify(bundle)
        with self.assertRaises(evidence.EvidenceError):
            evidence.authenticate(self.bundle(), b"")

    def test_current_context_and_expiry_are_required(self):
        for overrides in (
            {"repository": "other/repo"}, {"pr_number": 13}, {"pr_number": True},
            {"head_sha": "c" * 40}, {"base_sha": "c" * 40}, {"head_sha": "abc"},
            {"now_ns": NOW + evidence.MAX_AGE_NS + 1}, {"now_ns": NOW - 300_000_000_001},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(evidence.EvidenceError):
                self.verify(self.bundle(), **overrides)

    def test_missing_duplicate_and_self_certified_roles_fail(self):
        for change in (
            lambda p: p.update(reports=[]),
            lambda p: p["reports"].pop(),
            lambda p: p["reports"][1]["report"].update(role="implementation"),
            lambda p: p["reports"][1]["report"].update(agent_id="agent-implementation"),
            lambda p: p["reports"][1]["report"].update(agent_id="agent-coordinator"),
            lambda p: p["reports"][1]["report"].update(task_id="M1.1"),
        ):
            data = payload()
            change(data)
            with self.subTest(data=data), self.assertRaises(evidence.EvidenceError):
                self.verify(self.bundle(data))

    def test_report_binding_templates_and_required_evidence_fail_closed(self):
        for change in (
            lambda r: r.update(template_only=True), lambda r: r.update(version=True),
            lambda r: r.update(head_sha="c" * 40), lambda r: r.update(base_sha="c" * 40),
            lambda r: r.update(pr_url=PR_URL + "3"), lambda r: r.update(acceptance=[]),
            lambda r: r["acceptance"][0].update(evidence=""),
            lambda r: r.update(findings=None), lambda r: r.update(agent_id=" "),
        ):
            data = payload()
            change(data["reports"][1]["report"])
            with self.subTest(data=data), self.assertRaises(evidence.EvidenceError):
                self.verify(self.bundle(data))
        data = payload()
        data["reports"][1]["report_url"] = "https://untrusted.example/approval"
        with self.assertRaises(evidence.EvidenceError):
            self.verify(self.bundle(data))

    def test_failed_pending_and_skipped_approval_block_merge(self):
        for role in (1, 2):
            for decision in ("pending", "fail", "blocked", "SUCCESS", None):
                data = payload()
                data["reports"][role]["report"]["decision"] = decision
                with self.subTest(role=role, decision=decision):
                    with self.assertRaises(evidence.EvidenceError):
                        self.verify(self.bundle(data))
        for status in ("fail", "skipped", "unavailable", None):
            data = payload()
            data["reports"][2]["report"]["acceptance"][0]["status"] = status
            with self.subTest(status=status), self.assertRaises(evidence.EvidenceError):
                self.verify(self.bundle(data))

    def test_findings_require_verified_resolution_or_reviewed_minor_deferral(self):
        for severity in ("critical", "high", "medium", "low", "unknown"):
            for status in ("open", "deferred", "verified_fixed"):
                data = payload()
                data["reports"][1]["report"]["findings"] = [
                    {"severity": severity, "status": status},
                ]
                with self.subTest(severity=severity, status=status):
                    with self.assertRaises(evidence.EvidenceError):
                        self.verify(self.bundle(data))
        data = payload()
        finding = {"severity": "high", "status": "verified_fixed", "verification": "retest passed"}
        data["reports"][1]["report"]["findings"] = [finding]
        self.verify(self.bundle(data))
        finding.update(severity="low", status="deferred",
                       reviewer_agreement=f"{PR_URL}#issuecomment-2",
                       follow_up_issue=f"https://github.com/{REPO}/issues/13")
        self.verify(self.bundle(data))
        finding["severity"] = "high"
        with self.assertRaises(evidence.EvidenceError):
            self.verify(self.bundle(data))

    def test_untrusted_comment_text_cannot_forge_approval(self):
        fake = self.bundle()
        fake["signature"] = "0" * 64
        for comments in ([], [{"body": "APPROVED " + json.dumps(self.bundle())}],
                         [{"body": evidence.MARKER + "\n" + json.dumps(fake)}]):
            with self.subTest(comments=comments), self.assertRaises(evidence.EvidenceError):
                evidence.select_bundle(comments, KEY)

    def test_newest_signed_failure_never_falls_back_to_old_pass(self):
        failed = payload()
        failed["reports"][1]["report"]["decision"] = "fail"
        older = self.bundle(issued=NOW - 2)
        newer = self.bundle(failed, issued=NOW - 1)
        comments = [{"body": evidence.MARKER + "\n" + json.dumps(b)} for b in (newer, older)]
        selected = evidence.select_bundle(comments, KEY)
        self.assertEqual(selected["signature"], newer["signature"])
        with self.assertRaises(evidence.EvidenceError):
            self.verify(selected)

    def test_duplicate_json_nan_and_malformed_envelope_fail(self):
        with self.assertRaises(evidence.EvidenceError):
            evidence.load_json('{"version":1,"version":2}')
        for raw in ("[]", "{", "null"):
            with self.subTest(raw=raw), self.assertRaises(evidence.EvidenceError):
                evidence.load_json(raw)
        data = payload()
        data["nan"] = float("nan")
        with self.assertRaises(evidence.EvidenceError):
            self.bundle(data)
        for raw in ("", "not-a-key", "ab" * 31, "ab" * 33):
            with self.subTest(raw=raw), self.assertRaises(evidence.EvidenceError):
                evidence.read_key(raw)

    def test_deleted_edited_or_ambiguous_source_reports_fail(self):
        data = payload()
        evidence.verify_report_sources(data, report_comments(data))
        for mode in ("deleted", "edited", "ambiguous"):
            comments = report_comments(data)
            if mode == "deleted":
                comments.pop()
            elif mode == "edited":
                comments[1]["body"] = comments[1]["body"].replace('"pass"', '"fail"')
            else:
                comments[1]["body"] += "\n```json\n{}\n```"
            with self.subTest(mode=mode), self.assertRaises(evidence.EvidenceError):
                evidence.verify_report_sources(data, comments)

    def test_cli_round_trip_and_stale_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key_file, payload_file, bundle_file, comments_file = [
                root / name for name in ("key", "payload", "bundle", "comments")
            ]
            key_file.write_text(KEY.hex())
            payload_file.write_text(json.dumps(payload()))
            comments_file.write_text(json.dumps(report_comments(payload())))
            signed = subprocess.run([sys.executable, str(SCRIPT), "sign", "--payload",
                                     str(payload_file), "--key-file", str(key_file)],
                                    text=True, capture_output=True, check=True)
            self.assertNotIn(KEY.hex(), signed.stdout + signed.stderr)
            bundle_file.write_text(signed.stdout)
            command = [sys.executable, str(SCRIPT), "verify", "--bundle", str(bundle_file),
                       "--key-file", str(key_file), "--repository", REPO, "--pr", "12",
                       "--comments-file", str(comments_file), "--head", HEAD, "--base", BASE]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            command[-1] = "c" * 40
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)


class FakeAPI:
    repository = REPO

    def __init__(self, bundle):
        self.bundle = bundle
        self.statuses = []
        self.base = BASE
        self.head = HEAD
        self.moving = False

    def request(self, path):
        if path == "commits/main":
            if self.moving:
                self.base = "c" * 40
            return {"sha": self.base}
        if path == "pulls/12":
            return {"state": "open", "head": {"sha": self.head},
                    "base": {"ref": "main", "repo": {"full_name": REPO}}}
        raise AssertionError(f"unexpected API path {path}")

    def pages(self, path):
        if path != "issues/12/comments":
            raise AssertionError(f"unexpected API path {path}")
        if self.moving:
            self.head = "d" * 40
        return (report_comments(self.bundle["payload"])
                + [{"id": 4, "body": evidence.MARKER + "\n" + json.dumps(self.bundle)}])

    def status(self, head, state, description):
        self.statuses.append((head, state, description))


class WorkflowTests(unittest.TestCase):
    def test_publish_only_after_current_context_validation(self):
        bundle = evidence.sign_payload(payload(), KEY)
        api = FakeAPI(bundle)
        with patch("builtins.print"):
            self.assertTrue(evidence.check_pr(api, 12, KEY))
        self.assertEqual([s[1] for s in api.statuses], ["pending", "success"])
        self.assertTrue(all(s[0] == HEAD for s in api.statuses))

    def test_missing_or_stale_approvals_and_concurrent_head_change_publish_failure(self):
        for mode in ("missing", "stale", "moving"):
            data = payload()
            if mode == "stale":
                data["base_sha"] = "c" * 40
            api = FakeAPI(evidence.sign_payload(data, KEY))
            if mode == "missing":
                api.bundle["signature"] = "0" * 64
            api.moving = mode == "moving"
            with self.subTest(mode=mode), patch("builtins.print"):
                self.assertFalse(evidence.check_pr(api, 12, KEY))
                self.assertEqual(api.statuses[-1][1], "failure")

    def test_privileged_workflow_never_checks_out_pr_code(self):
        workflow = (SCRIPT.parents[1] / ".github/workflows/agent-evidence.yml").read_text()
        self.assertIn("environment: cineforge-agent-evidence", workflow)
        self.assertIn("ref: ${{ steps.trusted.outputs.sha }}", workflow)
        self.assertNotIn("github.event.pull_request.head", workflow)
        self.assertNotIn("github.event.comment.body", workflow)
        self.assertNotIn("pip install", workflow)
        self.assertIn("persist-credentials: false", workflow)

    def test_missing_key_sets_failure_and_regular_issue_comments_do_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(json.dumps({"number": 12}))
            env = {"GITHUB_REPOSITORY": REPO, "GITHUB_TOKEN": "synthetic",
                   "GITHUB_EVENT_NAME": "pull_request_target", "GITHUB_EVENT_PATH": str(event_path)}
            api = FakeAPI(evidence.sign_payload(payload(), KEY))
            with patch.dict("os.environ", env, clear=True), patch.object(evidence, "GitHubAPI") as api_type:
                api_type.return_value = api
                with self.assertRaises(evidence.EvidenceError):
                    evidence.check_event()
                self.assertEqual(api.statuses[-1][1], "failure")
                event_path.write_text(json.dumps({"issue": {"number": 12}}))
                with patch.dict("os.environ", {"GITHUB_EVENT_NAME": "issue_comment"}):
                    self.assertEqual(evidence.check_event(), 0)


if __name__ == "__main__":
    unittest.main()
