"""Coordinator-only opt-in GitHub ruleset probe; never writes main or cleans up resources."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPOSITORY = "richardo920/cineforge"
CONTEXTS = ("M0 probe CI", "M0 probe evidence")
TOKEN = re.compile(r"[a-z0-9]{8,32}\Z")


def probe_names(token: str) -> tuple[str, str]:
    if not TOKEN.fullmatch(token):
        raise ValueError("session must be 8–32 lowercase letters/digits")
    return f"probe/m0-2-{token}-base", f"probe/m0-2-{token}-head"


def fixture_ruleset(template: dict[str, Any], token: str) -> dict[str, Any]:
    """Keep production PR/strictness rules; synthetic statuses cannot claim Actions identity."""
    base, _ = probe_names(token)
    result = copy.deepcopy(template)
    result["name"] = f"m0-2-probe-{token}"
    result["conditions"] = {"ref_name": {"include": [f"refs/heads/{base}"], "exclude": []}}
    for rule in result["rules"]:
        if rule["type"] == "required_status_checks":
            rule["parameters"]["required_status_checks"] = [
                {"context": context} for context in CONTEXTS
            ]
    return result


class GitHub:
    def __init__(self, output: Path) -> None:
        # Exclusive creation prevents overwriting another session's evidence.
        self.stream = output.open("x")

    def record(self, item: dict[str, Any]) -> None:
        self.stream.write(json.dumps(item, sort_keys=True) + "\n")
        self.stream.flush()

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None,
                *, expected: tuple[int, ...] = (200, 201)) -> dict[str, Any]:
        args = ["gh", "api", "--include", "--method", method,
                f"repos/{REPOSITORY}/{path}"]
        if payload is not None:
            args += ["--input", "-"]
        response = subprocess.run(args, input=None if payload is None else json.dumps(payload),
                                  capture_output=True, text=True, timeout=60)
        headers, separator, body = response.stdout.partition("\n\n")
        match = re.match(r"HTTP/\S+ (\d+)", headers)
        if not separator or match is None:
            raise RuntimeError("GitHub did not return an HTTP/JSON response")
        status = int(match[1])
        data = json.loads(body)
        self.record({"method": method, "path": path, "request": payload,
                     "http_status": status, "response": data})
        if status not in expected:
            raise RuntimeError(f"Unexpected HTTP {status}: {method} {path}")
        if not isinstance(data, dict):
            raise RuntimeError("Expected a GitHub object response")
        return data


def git(*args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", *args], input=input_data, stderr=subprocess.PIPE)


def commit(parent: str, sequence: int) -> str:
    """Create synthetic metadata using default Git author/committer, without touching the index."""
    blob = git("hash-object", "-w", "--stdin", input_data=f'{{"step":{sequence}}}\n'.encode())
    entries = git("ls-tree", "-z", parent).split(b"\0")
    entries = [entry for entry in entries if entry and not entry.endswith(b"\tprobe-fixture.json")]
    entries.append(b"100644 blob " + blob.strip() + b"\tprobe-fixture.json")
    tree = git("mktree", "-z", input_data=b"\0".join(entries) + b"\0").strip().decode()
    return git("commit-tree", tree, "-p", parent,
               input_data=f"M0.2 disposable protection fixture {sequence}\n".encode()).strip().decode()


def ready_pr(api: GitHub, number: int, head: str, *, blocked_state: bool | None = None
             ) -> dict[str, Any]:
    for _ in range(20):
        pr = api.request("GET", f"pulls/{number}")
        expected_state = (blocked_state is None or
                          (pr.get("mergeable_state") == "blocked") is blocked_state)
        if pr["head"]["sha"] == head and pr.get("mergeable") is True and expected_state:
            return pr
        time.sleep(1)
    raise RuntimeError("Probe PR is not conflict-free at the expected head")


def status(api: GitHub, head: str, context: str, state: str) -> None:
    api.request("POST", f"statuses/{head}", {"context": context, "state": state,
                                           "description": "Disposable M0.2 fixture only"})


def blocked(api: GitHub, number: int, head: str, label: str) -> None:
    pr = ready_pr(api, number, head, blocked_state=True)
    if pr.get("mergeable_state") != "blocked":
        raise RuntimeError("Expected protected PR mergeable_state=blocked")
    response = api.request("PUT", f"pulls/{number}/merge",
                           {"sha": head, "merge_method": "merge"}, expected=(405,))
    message = str(response.get("message", ""))
    if not re.search(r"required|status check|repository rule|protected", message, re.IGNORECASE):
        raise RuntimeError("Merge rejection did not identify a protection/check requirement")
    api.record({"assertion": label, "result": "pass", "head_sha": head})


def run_probe(api: GitHub, template: dict[str, Any], token: str) -> None:
    base, head_branch = probe_names(token)
    remote = git("remote", "get-url", "origin").strip().decode()
    if remote.removesuffix(".git") not in {
        f"https://github.com/{REPOSITORY}", f"git@github.com:{REPOSITORY}",
    }:
        raise ValueError("origin must be the Cineforge repository")
    main = api.request("GET", "branches/main")["commit"]["sha"]
    git("cat-file", "-e", f"{main}^{{commit}}")
    api.request("POST", "git/refs", {"ref": f"refs/heads/{base}", "sha": main})
    first = commit(main, 1)
    git("push", "origin", f"{first}:refs/heads/{head_branch}")
    config = fixture_ruleset(template, token)
    ruleset = api.request("POST", "rulesets", config)
    api.record({"resources": {"base": base, "head": head_branch, "ruleset": ruleset["id"]},
                "production_difference": "Two unbound user-issued synthetic check contexts"})
    pr = api.request("POST", "pulls", {"title": f"M0.2 disposable gate probe {token}",
                     "body": "Disposable synthetic check-enforcement fixture; no main changes.",
                     "base": base, "head": head_branch})
    number = pr["number"]
    api.record({"pr": number, "url": pr["html_url"], "starting_main": main})
    blocked(api, number, first, "missing checks reject merge")
    status(api, first, CONTEXTS[0], "success")
    blocked(api, number, first, "CI success without evidence rejects merge")
    status(api, first, CONTEXTS[1], "failure")
    blocked(api, number, first, "failed evidence rejects merge")
    status(api, first, CONTEXTS[1], "success")
    old = api.request("GET", f"commits/{first}/status")
    if old.get("state") != "success":
        raise RuntimeError("First head did not receive successful fixture statuses")
    second = commit(first, 2)
    if first == second:
        raise RuntimeError("Probe must change the head")
    git("push", "origin", f"{second}:refs/heads/{head_branch}")
    ready_pr(api, number, second)
    fresh = api.request("GET", f"commits/{second}/status")
    if fresh.get("statuses"):
        raise RuntimeError("New head unexpectedly inherited commit statuses")
    blocked(api, number, second, "old successful checks cannot approve new head")
    status(api, second, CONTEXTS[1], "success")
    status(api, second, CONTEXTS[0], "failure")
    blocked(api, number, second, "failed CI rejects merge despite evidence success")
    status(api, second, CONTEXTS[0], "success")
    ready_pr(api, number, second, blocked_state=False)
    merged = api.request("PUT", f"pulls/{number}/merge", {"sha": second, "merge_method": "merge"})
    if merged.get("merged") is not True:
        raise RuntimeError("Positive control did not merge")
    if api.request("GET", f"branches/{quote(base, safe='')}")["commit"]["sha"] != merged["sha"]:
        raise RuntimeError("Disposable target does not match verified merge")
    if api.request("GET", "branches/main")["commit"]["sha"] != main:
        raise RuntimeError("Main changed concurrently; repeat integration assessment")
    api.record({"assertion": "positive control merged with fresh successful checks",
                "result": "pass", "merge_sha": merged["sha"], "main_unchanged": main})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-live-probe", action="store_true", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new transcript outside Git")
    args = parser.parse_args()
    probe_names(args.session)
    checkout = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if args.output.resolve().is_relative_to(checkout):
        parser.error("transcript must be outside the repository checkout")
    template = json.loads((Path(__file__).resolve().parents[1] /
                           "config/main-ruleset.json").read_text())
    api = GitHub(args.output)
    try:
        run_probe(api, template, args.session)
        print(f"PASS: disposable probe; retained resources and transcript {args.output}")
        return 0
    except (ValueError, KeyError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        api.record({"result": "fail", "error": str(exc)})
        print("FAIL: probe incomplete; inspect retained transcript/resources")
        return 1
    finally:
        api.stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
