"""Verify coordinator-signed agent completion reports without executing PR code."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MARKER = "<!-- cineforge-agent-evidence:v1 -->"
CONTEXT = "Agent evidence"
DOMAIN = b"cineforge-agent-evidence-v1\x00"
MAX_AGE_NS = 86_400 * 1_000_000_000
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


class EvidenceError(ValueError):
    """Missing, unauthenticated, stale, or incomplete evidence."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be nonempty text")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate JSON key")
        result[key] = value
    return result


def load_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (ValueError, RecursionError) as exc:
        raise EvidenceError("invalid evidence JSON") from exc
    return _object(value, "JSON root")


def read_key(raw: str) -> bytes:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", raw.strip()):
        raise EvidenceError("signing key must be 32 random bytes encoded as hex")
    return bytes.fromhex(raw.strip())


def canonical(payload: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise EvidenceError("payload is not canonical JSON") from exc


def sign_payload(
    payload: dict[str, Any], key: bytes, *, issued_at_ns: int | None = None,
) -> dict[str, Any]:
    if len(key) != 32:
        raise EvidenceError("signing key must contain 32 bytes")
    stamped = dict(payload, issued_at_ns=issued_at_ns or time.time_ns())
    signature = hmac.new(key, DOMAIN + canonical(stamped), hashlib.sha256).hexdigest()
    return {"version": 1, "payload": stamped, "signature": signature}


def authenticate(bundle: dict[str, Any], key: bytes) -> dict[str, Any]:
    if len(key) != 32 or type(bundle.get("version")) is not int or bundle["version"] != 1:
        raise EvidenceError("unsupported envelope or missing signing key")
    payload = _object(bundle.get("payload"), "payload")
    signature = bundle.get("signature")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise EvidenceError("missing signature")
    expected = hmac.new(key, DOMAIN + canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise EvidenceError("invalid coordinator signature")
    return payload


def verify_bundle(
    bundle: dict[str, Any], key: bytes, *, repository: str, pr_number: int,
    head_sha: str, base_sha: str, now_ns: int | None = None,
) -> dict[str, Any]:
    """Validate a signed completion attestation against independently resolved Git refs."""
    payload = authenticate(bundle, key)
    if not REPOSITORY.fullmatch(repository) or type(pr_number) is not int or pr_number <= 0:
        raise EvidenceError("invalid repository or PR number")
    if not SHA.fullmatch(head_sha) or not SHA.fullmatch(base_sha):
        raise EvidenceError("full Git commit SHAs are required")
    for field, expected in (
        ("repository", repository), ("pr_number", pr_number),
        ("head_sha", head_sha), ("base_sha", base_sha),
    ):
        if type(payload.get(field)) is not type(expected) or payload[field] != expected:
            raise EvidenceError(f"attestation does not match current {field}")
    issued = payload.get("issued_at_ns")
    current = time.time_ns() if now_ns is None else now_ns
    if type(issued) is not int or not -300_000_000_000 <= current - issued <= MAX_AGE_NS:
        raise EvidenceError("attestation expired or has an invalid timestamp")
    coordinator = _text(payload.get("coordinator_agent_id"), "coordinator agent")
    reports = payload.get("reports")
    if not isinstance(reports, list) or len(reports) != 3:
        raise EvidenceError("one implementation, review, and testing report is required")
    seen_roles: set[str] = set()
    seen_agents = {coordinator}
    task_ids: set[str] = set()
    pr_url = f"https://github.com/{repository}/pull/{pr_number}"
    for entry in reports:
        entry = _object(entry, "report entry")
        report_url = _text(entry.get("report_url"), "report URL")
        if not re.fullmatch(re.escape(pr_url) + r"#issuecomment-[0-9]+", report_url):
            raise EvidenceError("report URL must reference this PR's report comment")
        report = _object(entry.get("report"), "report")
        if report.get("template_only") is not None:
            raise EvidenceError("templates cannot authorize merge")
        if type(report.get("version")) is not int or report["version"] != 1:
            raise EvidenceError("unsupported report version")
        role = _text(report.get("role"), "report role")
        if role not in {"implementation", "review", "testing"} or role in seen_roles:
            raise EvidenceError("duplicate or invalid report role")
        seen_roles.add(role)
        agent = _text(report.get("agent_id"), "agent ID")
        if agent in seen_agents:
            raise EvidenceError("coordinator, author, reviewer, and tester must be distinct")
        seen_agents.add(agent)
        task_ids.add(_text(report.get("task_id"), "task ID"))
        for field, expected in (("pr_url", pr_url), ("head_sha", head_sha), ("base_sha", base_sha)):
            if report.get(field) != expected:
                raise EvidenceError(f"stale or mismatched report {field}")
        allowed = {"pass", "pending"} if role == "implementation" else {"pass"}
        if report.get("decision") not in allowed:
            raise EvidenceError("independent review and testing must explicitly pass")
        acceptance = report.get("acceptance")
        if not isinstance(acceptance, list) or not acceptance:
            raise EvidenceError("acceptance evidence is required")
        for criterion in acceptance:
            criterion = _object(criterion, "acceptance criterion")
            _text(criterion.get("criterion"), "criterion")
            _text(criterion.get("evidence"), "criterion evidence")
            if criterion.get("status") != "pass":
                raise EvidenceError("failed, skipped, or unavailable acceptance blocks merge")
        findings = report.get("findings")
        if not isinstance(findings, list):
            raise EvidenceError("explicit findings list is required")
        for finding in findings:
            finding = _object(finding, "finding")
            severity = finding.get("severity")
            status = finding.get("status")
            if severity not in {"critical", "high", "medium", "low"}:
                raise EvidenceError("unknown finding severity")
            if status == "verified_fixed":
                _text(finding.get("verification"), "finding verification")
            elif status == "deferred" and severity in {"medium", "low"}:
                _text(finding.get("reviewer_agreement"), "deferral agreement")
                _text(finding.get("follow_up_issue"), "deferral follow-up issue")
            else:
                raise EvidenceError("unresolved findings block merge")
    if len(task_ids) != 1:
        raise EvidenceError("reports must concern the same approved task")
    return payload


def select_bundle(comments: list[dict[str, Any]], key: bytes) -> dict[str, Any]:
    """Use the newest authenticated statement; never fall back after a signed failure."""
    candidates: list[tuple[int, dict[str, Any]]] = []
    for comment in comments:
        body = comment.get("body", "")
        if not isinstance(body, str) or not body.startswith(MARKER + "\n"):
            continue
        try:
            bundle = load_json(body[len(MARKER):].strip())
            payload = authenticate(bundle, key)
            issued = payload.get("issued_at_ns")
            if type(issued) is int:
                candidates.append((issued, bundle))
        except EvidenceError:
            continue
    if not candidates:
        raise EvidenceError("no authenticated coordinator attestation")
    return max(candidates, key=lambda candidate: candidate[0])[1]


def verify_report_sources(payload: dict[str, Any], comments: list[dict[str, Any]]) -> None:
    """Deleted or edited source approvals invalidate their signed copies."""
    sources = {comment.get("id"): comment for comment in comments}
    for entry in payload["reports"]:
        identifier = int(entry["report_url"].rsplit("-", 1)[1])
        source = sources.get(identifier)
        if source is None:
            raise EvidenceError("an attested report comment is missing")
        body = source.get("body")
        if not isinstance(body, str):
            raise EvidenceError("an attested report comment is malformed")
        raw = body.strip()
        if not raw.startswith("{"):
            blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
            if len(blocks) != 1:
                raise EvidenceError("an attested report comment is malformed")
            raw = blocks[0]
        if canonical(load_json(raw)) != canonical(entry["report"]):
            raise EvidenceError("an attested report comment changed")


class GitHubAPI:
    """Minimal fixed-origin API client; errors never include tokens or response bodies."""

    def __init__(self, repository: str, token: str) -> None:
        if not REPOSITORY.fullmatch(repository) or not token:
            raise EvidenceError("GitHub repository and token are required")
        self.repository = repository
        self.token = token

    def request(self, path: str, data: dict[str, Any] | None = None) -> Any:
        encoded = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repository}/{path}", data=encoded,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise EvidenceError("GitHub API response exceeds safe size")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise EvidenceError(f"GitHub API request failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, ValueError) as exc:
            raise EvidenceError("GitHub API request failed") from exc

    def pages(self, path: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            batch = self.request(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                raise EvidenceError("unexpected GitHub list response")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise EvidenceError("GitHub pagination limit exceeded")

    def status(self, head: str, state: str, description: str) -> None:
        if not SHA.fullmatch(head):
            raise EvidenceError("invalid status target SHA")
        self.request(f"statuses/{head}", {
            "state": state, "context": CONTEXT, "description": description[:140],
        })


def check_pr(api: GitHubAPI, number: int, key: bytes) -> bool:
    pr = api.request(f"pulls/{number}")
    if pr.get("state") != "open":
        return True
    head = pr["head"]["sha"]
    api.status(head, "pending", "Verifying coordinator-signed independent agent evidence")
    try:
        if pr["base"]["repo"]["full_name"] != api.repository or pr["base"]["ref"] != "main":
            raise EvidenceError("trusted evidence gate supports this repository's main only")
        base = api.request("commits/main")["sha"]
        comments = api.pages(f"issues/{number}/comments")
        bundle = select_bundle(comments, key)
        payload = verify_bundle(bundle, key, repository=api.repository, pr_number=number,
                                head_sha=head, base_sha=base)
        verify_report_sources(payload, comments)
        latest = api.request(f"pulls/{number}")
        if latest["head"]["sha"] != head or api.request("commits/main")["sha"] != base:
            raise EvidenceError("head or main changed during verification")
    except (EvidenceError, KeyError, TypeError, ValueError, RecursionError) as exc:
        description = str(exc) if isinstance(exc, EvidenceError) else "Malformed evidence or API data"
        api.status(head, "failure", description)
        print(f"PR #{number}: FAIL — {description}")
        return False
    api.status(head, "success", "Signed current-head review/testing completion verified")
    print(f"PR #{number}: PASS — coordinator completion attestation verified")
    return True


def check_event() -> int:
    api = GitHubAPI(os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_TOKEN", ""))
    event = load_json(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    name = os.environ.get("GITHUB_EVENT_NAME")
    if name == "pull_request_target":
        numbers = [event["number"]]
    elif name == "issue_comment":
        if not event.get("issue", {}).get("pull_request"):
            return 0
        numbers = [event["issue"]["number"]]
    elif name == "workflow_dispatch":
        raw = str(event.get("inputs", {}).get("pr_number", ""))
        if not raw.isdigit() or int(raw) <= 0:
            raise EvidenceError("dispatch requires a positive PR number")
        numbers = [int(raw)]
    elif name == "push":
        numbers = [pr["number"] for pr in api.pages("pulls?state=open&base=main")]
    else:
        raise EvidenceError("unsupported evidence event")
    try:
        key = read_key(os.environ.get("AGENT_EVIDENCE_HMAC_KEY", ""))
    except EvidenceError:
        for number in numbers:
            pr = api.request(f"pulls/{number}")
            if pr.get("state") == "open":
                api.status(pr["head"]["sha"], "failure", "Trusted signing key is not configured")
        raise
    results = [check_pr(api, number, key) for number in numbers]
    return 0 if all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sign = commands.add_parser("sign", help="trusted coordinator only; emits no key material")
    sign.add_argument("--payload", type=Path, required=True)
    sign.add_argument("--key-file", type=Path, required=True)
    verify = commands.add_parser("verify", help="independently bind evidence to current Git refs")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--key-file", type=Path, required=True)
    verify.add_argument("--comments-file", type=Path, required=True,
                        help="fresh GitHub issue-comments API JSON array")
    verify.add_argument("--repository", required=True)
    verify.add_argument("--pr", type=int, required=True)
    verify.add_argument("--head", required=True)
    verify.add_argument("--base", required=True)
    commands.add_parser("check", help="trusted Actions workflow entrypoint")
    args = parser.parse_args()
    try:
        if args.command == "check":
            return check_event()
        key = read_key(args.key_file.read_text())
        if args.command == "sign":
            bundle = sign_payload(load_json(args.payload.read_text()), key)
            print(MARKER)
            print(json.dumps(bundle, indent=2, allow_nan=False))
        else:
            raw = args.bundle.read_text()
            if raw.startswith(MARKER + "\n"):
                raw = raw[len(MARKER):].strip()
            payload = verify_bundle(load_json(raw), key, repository=args.repository,
                                    pr_number=args.pr, head_sha=args.head, base_sha=args.base)
            comments = json.loads(args.comments_file.read_text())
            if not isinstance(comments, list) or any(not isinstance(c, dict) for c in comments):
                raise EvidenceError("comments snapshot must be a GitHub API array")
            verify_report_sources(payload, comments)
            print("PASS: signed current-head agent completion attestation")
        return 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError):
        print("FAIL: missing, malformed, stale, or failed trusted evidence", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
