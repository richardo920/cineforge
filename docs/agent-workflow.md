# Agent development workflow proposal

Use this chat for design and prioritization; use Codex in a repository checkout for task implementation, diffs, and review. The source of truth is the merged design, task ID, acceptance criteria, and PR history, not an agent's chat memory.

| Role | Responsibility | Evidence |
| --- | --- | --- |
| Implementation agent | One approved task on an isolated branch | Diff, task ID, tests, limitations |
| Review agent | Independently inspect diff and affected callers | Head SHA, severity, file/line, failure scenario |
| Testing agent | Independently test task acceptance and failure paths | Head SHA, commands, fixtures, results, gaps |
| Owner | Resolve design tradeoffs and approve merge | Explicit PR approval/merge |

Review and testing agents should use separate contexts and inspect requirements directly. They may share the same underlying model, so correlated mistakes remain possible; automated checks and human judgment still matter.

For every implementation PR: author opens PR → reviewer and tester report on its head SHA → author fixes findings → affected reviews/checks repeat → owner merges. Block critical/high findings and failed acceptance criteria; document lower-severity deferrals with owner agreement. Do not let the author approve its own work.

Planned enforcement: required CI, stale approval dismissal, current-head review/test evidence, and owner-controlled merge. Agent prose is not a GitHub approval or status check; milestone M0 must implement trusted checks and verify protections supported by the account before calling these enforced. Default CI receives no model/cloud credentials; paid/GPU integration runs require explicit opt-in and budgets.

## Operating ten tasks in parallel

1. Open `~/Desktop/cineforge` as a Codex project. Keep one coordinator chat responsible for the approved backlog, dependencies, PR links, and review state.
2. Coordinator selects two or three independent tasks with stable interfaces. Create one implementation chat/worktree/branch per task from current `main`; dependent tasks wait for prerequisites or use an explicitly tracked PR stack.
3. Each author opens a PR and supplies task ID, acceptance criteria, test evidence, and head SHA. Authors stop changing the branch while reviewers inspect it.
4. Open a separate review chat for the PR; use Review with Codex in Code Review, or `/review` against the intended base branch in a separate checkout. Review the diff, affected callers, architecture, security, cleanup, and missing tests.
5. Open a testing chat with its own disposable worktree at the PR head commit. It derives tests from task acceptance criteria, runs existing checks, and probes failures independently. Keep new test changes on a separate branch and hand them back to the author.
6. Coordinator routes findings to the author. After fixes, rerun affected tests/review on the new head SHA and update the task record; findings remain open until verified.
7. Before owner merge, verify current-head CI and integration with current `main`. After related PRs merge, update affected branches and repeat integration checks/review for any resolution changes.
8. Owner merges ready PRs in dependency order; coordinator schedules the next ready tasks. Agents do not merge automatically.

Worktrees isolate checked-out files, not service ports, databases, credentials, shared caches, or cloud spend. Use per-task environments, output directories, database namespaces, and ports; avoid concurrent edits to shared Git configuration or the same branch.

Suggested task record: `task ID | dependencies | branch/PR | head SHA | author | review | testing | CI | owner decision`. Start with three implementation slots and reuse reviewer/tester capacity as PRs arrive rather than starting ten tightly coupled tasks together.

## Role prompts

Implementation: “Implement only approved task TASK-ID in your own worktree. Read DESIGN.md and its acceptance criteria. Add meaningful tests, open a PR with evidence, and wait for independent review. Do not merge.”

Review: “Independently review PR URL at HEAD-SHA against its intended base and TASK-ID. Do not modify implementation. Inspect affected callers and failure paths. Report actionable findings with severity, file/line, reproduction, and required fix; state residual uncertainty even if no issues are found.”

Testing: “Independently verify TASK-ID for PR URL at HEAD-SHA in a separate worktree. Derive cases from acceptance criteria, including invalid input and recovery. Record commands, results, skipped checks, and gaps. No paid calls or production credentials; propose needed test changes separately.”

Coordinator: “Track the approved backlog and dependencies. Dispatch only ready tasks, route PR findings, ensure review/testing/CI evidence matches the current head, and present ready PRs for owner merge. Do not implement unrelated work or auto-merge.”

## Built-in review versus reusable skills

A special skill is not required for review: Codex provides `/review` for local diffs, and Code Review provides Review with Codex for PRs. GitHub-triggered `@codex review` or automatic cloud review requires the connector, repository access, and review settings; none is enabled by this scaffold.

M0.1 can package the role prompts as reusable project skills/instructions after approval. A skill standardizes behavior; a coordinator schedules work; CI/branch protections enforce merge conditions. These are separate mechanisms, and review chat output alone neither posts a GitHub review nor approves a PR.

Product references: [Code review](https://learn.chatgpt.com/docs/code-review?surface=app), [Worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees).
