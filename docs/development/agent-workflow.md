# Agent development workflow proposal

Use this chat for design and prioritization; use Codex in a repository checkout for task implementation, diffs, and review. The source of truth is the merged design, task ID, acceptance criteria, and PR history, not an agent's chat memory.

| Role | Responsibility | Evidence |
| --- | --- | --- |
| Implementation agent | One approved task on an isolated branch | Diff, task ID, tests, limitations |
| Review agent | Independently inspect diff and affected callers | Head SHA, severity, file/line, failure scenario |
| Testing agent | Independently test task acceptance and failure paths | Head SHA, commands, fixtures, results, gaps |
| Coordinator / merge agent | Verify all required gates and merge approved tasks | Current-head CI, independent review/test approval, merge record |
| Owner (optional involvement) | Set product direction, approve initial plan, resolve scope decisions | Product decisions; no routine code review required |

Review and testing agents should use separate contexts and inspect requirements directly. They may share the same underlying model, so correlated mistakes remain possible; automated checks and human judgment still matter.

For every implementation PR: author opens PR → reviewer and tester report on its head SHA → author fixes findings → affected reviews/checks repeat → coordinator/merge agent verifies all gates and merges without another owner confirmation. Block critical/high findings and failed acceptance criteria; document lower-severity deferrals with independent reviewer agreement and a tracked follow-up. Do not let the author approve its own work.

Planned enforcement: required CI, stale approval dismissal, current-head review/test evidence, and conditional agent merge after every required gate passes. Agent prose is not a GitHub approval or status check; milestone M0 must implement trusted checks and verify protections supported by the account before calling these enforced. Default CI receives no model/cloud credentials; paid/GPU integration runs require explicit opt-in and budgets.

## Operating ten tasks in parallel

1. Reserve `~/Desktop/cineforge` for coordinator inspection/synchronization, never development edits. Record one active coordinator in the shared queue; only it mutates the GitHub queue, security settings, or merges. Other agents send proposed updates to it.
2. Coordinator selects two or three independent tasks with stable interfaces. After inspecting worktree ownership/status, give every editing task/session a unique worktree path and branch from current `origin/main`; dependent tasks wait for prerequisites or use an explicitly tracked PR stack.
3. Each author opens a PR and supplies task ID, acceptance criteria, test evidence, and head SHA. Authors stop changing the branch while reviewers inspect it.
4. Open a separate review chat with its own uniquely named detached worktree at the full PR head SHA. Review the diff against its intended base, affected callers, architecture, security, cleanup, and missing tests.
5. Open a testing chat with another uniquely named detached worktree at that exact head. It derives tests from task acceptance criteria, runs existing checks, and probes failures independently. Keep proposed test changes on a unique branch and hand them back to the author.
6. Coordinator routes findings to the author. After fixes, rerun affected tests/review on the new head SHA and update the task record; findings remain open until verified.
7. Before agent merge, verify current-head CI and integration with current `main`. After related PRs merge, update affected branches and repeat integration checks/review for any resolution changes.
8. Coordinator/merge agent merges ready implementation PRs in dependency order once every required check, independent review, and independent acceptance test passes for the current head; no further owner confirmation is needed. Coordinator records the merge and schedules the next ready tasks. The initial design/planning PR remains for owner review.

Before editing, inspect `git worktree list --porcelain` and `git -C PATH status --short`. Never assume an existing path, including `/tmp`, is unowned: reuse requires a recorded handoff, clean status, and a stopped prior owner; pause on uncertain ownership.

```sh
git worktree list --porcelain
git status --short
git fetch origin
task_token=$(uuidgen | tr '[:upper:]' '[:lower:]')
task_path="../cineforge-worktrees/M0.x-${task_token}"
git worktree add -b "task/M0.x-${task_token}" "$task_path" origin/main
git worktree add --detach "${task_path}-review" FULL_PR_HEAD_SHA
git worktree add --detach "${task_path}-test" FULL_PR_HEAD_SHA
```

Worktrees share Git refs/configuration, credentials, and caches; they do not isolate ports, databases, or cloud spend. Namespace environments, temporary outputs, databases, and ports per session; avoid shared Git configuration changes and other sessions' branches/refs. Clean up only trees your session created after they are clean, inactive, and released in the queue; never force-remove another session's tree.

Task record: `task | session owner | worktree path | branch/detached SHA | PR/head | dependencies | author/review/testing | CI | merge result`. Record handoffs and coordinator ownership here; reuse agent capacity with fresh task worktrees.

## Role prompts

Use [role instructions](../../.github/agents/roles.json): supply `inputs`, then apply `shared` plus the selected `roles` entry. Author, reviewer and tester use distinct contexts/agent IDs; the tester also uses a separate worktree.

Copy [the report template](../../.github/agents/report.example.json) into each PR report, remove `template_only`, replace placeholders and empty unused arrays. Publish author, review and testing reports for the same full head SHA; reviewers/testers explicitly pass or fail, and new commits require renewed reports.

Critical/high findings and any failed acceptance criterion block merge. Lower-severity deferrals require independent reviewer agreement and a linked follow-up; report skipped checks and uncertainty. These templates standardize content, not identity or trust enforcement; shared-account reports are not independent GitHub approvals (M0.2/M0.3).

## Built-in review versus reusable skills

A special skill is not required for review: Codex provides `/review` for local diffs, and Code Review provides Review with Codex for PRs. GitHub-triggered `@codex review` or automatic cloud review requires the connector, repository access, and review settings; none is enabled by this scaffold.

M0.1 can package the role prompts as reusable project skills/instructions after approval. A skill standardizes behavior; a coordinator schedules work; CI/branch protections enforce merge conditions. These are separate mechanisms, and review chat output alone neither posts a GitHub review nor approves a PR.

Product references: [Code review](https://learn.chatgpt.com/docs/code-review?surface=app), [Worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees).

## Agent merge gate

Merge is authorized for assigned implementation tasks from the owner-approved plan when all required CI checks succeed, the independent review agent explicitly passes the current head with no unresolved blocking findings, and the independent testing agent explicitly passes all acceptance criteria for that head. Pending, skipped, missing, unavailable, or stale required evidence blocks merge. New commits require renewed independent approvals and current-head tests.

Check that the PR head has not changed immediately before merging and use a merge operation conditioned on that head SHA. Verify integration with the current target branch; if related merges or conflict resolution change the tested combination, repeat affected integration checks and review. Never use administrative bypass. Agent approval evidence must be independently attributable; if agents share a GitHub account, a second chat does not constitute a separate GitHub approving identity. M0 must define trusted checks for that case rather than counting author self-approval.

### Trusted completion attestation

The coordinator signs a bundle only after observing distinct author/reviewer/tester contexts and passing current-head reports. `Agent evidence` verifies that signature, repo/PR/current head/current main, complete passing criteria, and finding resolutions; unsigned text, missing/stale reports and expired bundles fail closed (24-hour lifetime).

Keep the 32-byte hex signing key outside Git and in environment secret `AGENT_EVIDENCE_HMAC_KEY` under **`cineforge-agent-evidence`, restricted to deployment branch `main`**. Never use a repository-wide secret: PR/push workflows could request it. The privileged workflow resolves and executes only the trusted default-branch commit, never PR code; main pushes recheck open PRs.

```sh
python scripts/agent_evidence.py sign --payload /tmp/reports.json --key-file /secure/key
python scripts/agent_evidence.py verify --bundle /tmp/signed-comment.txt --key-file /secure/key \
  --comments-file /tmp/fresh-api-comments.json --repository richardo920/cineforge \
  --pr 12 --head FULL_HEAD --base CURRENT_MAIN
```

Payload fields: `repository`, `pr_number`, `head_sha`, `base_sha`, `coordinator_agent_id`, and three `reports: [{report_url, report}]`; reports follow the existing template. Signing adds `issued_at_ns`; publish the complete output as one PR comment. After current-head CI passes, independently resolve refs, fetch fresh report comments, and verify locally immediately before a SHA-conditioned merge; edited/deleted source reports invalidate approval.

This authenticates a trusted coordinator's completion claims, not independent GitHub identities or semantic correctness. Shared maintainer credentials/host and environment administrators remain trusted; a same-name status/check alone cannot establish report authenticity or authorize merge.

## Full agent maintenance

The owner is not expected to write code, review code, or merge routine PRs. After initial plan approval, the coordinator schedules ready tasks within that scope and agents handle implementation, independent review/testing, merges, issue triage, bug fixes, dependency updates, documentation, and CI failures. Routine maintenance uses the same gates; urgency does not waive review or testing.

The owner may inspect or intervene at any time. Escalate unresolved blockers or product-scope decisions; do not request routine code approval. Preserve existing approval requirements for spending, rights/consent, public release, and changes to security-sensitive access. This is the intended operating model; the scaffold does not yet run a continuous maintenance service or an automated merge controller.

## GitHub Projects task queue

Use a private GitHub Project linked to Cineforge as the coordinator's shared queue; each approved task becomes a repository issue, linked to its implementation PR. Keep requirements and acceptance criteria in the issue; record CI/review/testing evidence on the PR against its head SHA.

| Field | Values / purpose |
| --- | --- |
| Status | Backlog → Ready → In progress → Review & testing → Ready to merge → Done; Blocked for unresolved dependencies/failures |
| Task / milestone | Stable plan ID such as M1.2 and its milestone |
| Dependencies | Blocking issue links; Ready requires prerequisites satisfied |
| Agent role / chat | Responsible implementation chat, independent reviewer, and tester links |
| PR / head SHA | Change under review and exact approved commit |

Coordinator selects Ready tasks, assigns isolated worktrees, links PRs, and refreshes stale evidence after changes. Ready to merge requires every gate in the agent merge policy; Done requires verified merge. A green status field never grants merge authority.

The private [Cineforge — Agent Development project](https://github.com/users/richardo920/projects/1) is linked to the repository. Projects access is verified; the proposed plan tasks are draft cards in Backlog pending approval. M0.6 covers conversion to issues and operational task/PR tracking after approval; no automatic dispatch or merge controller is configured. Use issues/PRs as the fallback if Projects becomes unavailable.

## Starting a development session

After the owner approves and merges the initial planning PR, open `~/Desktop/cineforge` as a Codex project and start a coordinator chat with: **“Start the next ready milestone.”** `AGENTS.md` defines the full startup procedure, delegation, merge gates, and stopping point; no separate `skills.md` is required. Repeat the same prompt to resume an interrupted milestone or start the next one after completion.
