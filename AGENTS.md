# Repository guidance

- Agents maintain the repository end to end. Implement tasks from the owner-approved, merged milestone plan and routine maintenance within that scope; keep one task per PR.
- Never add Codex as a commit author or co-author. Use default Git identity.
- Never commit secrets, media, model weights, private character references, or unreviewed third-party skills.
- Do not create Markdown files unless explicitly requested. Keep requested documentation concise; prefer examples.
- Keep orchestration, GPU provisioning, workflow execution, model adapters, and storage separate.
- Default tests must run offline without credentials, GPUs, or paid services. Run `python -m unittest discover -s tests -v` and the CLI smoke check.
- Use a separate review agent and testing agent for implementation PRs. Review correctness/security and test acceptance criteria independently; record findings and evidence against the current PR head SHA.
- The coordinator/merge agent may merge approved implementation tasks without another owner confirmation after all required CI checks pass, independent review explicitly passes, and independent acceptance testing passes for the current PR head. Authors cannot self-certify review or testing.
- New commits invalidate prior review/testing approval. Verify the current head and integration with current main before merging; never bypass protections or treat pending, skipped, missing, or unavailable required checks as passed. The initial design/planning PR still requires owner review.
- Agent reports supplement CI and human review. These instructions alone do not enforce GitHub branch protection.
- No routine owner coding, code review, or merge action is required. Agents triage issues, schedule ready tasks, fix bugs/CI, maintain dependencies/docs, and merge through the same independent quality gates. Owner participation is optional; escalate product-scope decisions or unresolved blockers.

## Session isolation

- Each editing task/session gets a unique branch and worktree; main is for coordinator inspection/synchronization only, never development edits. Before editing, inspect `git worktree list --porcelain` and the intended tree's `git status --short`. Existing paths, including `/tmp`, require a recorded handoff, clean status, and a stopped prior owner; pause on unclear ownership.
- Reviewers and testers each use a separate, uniquely named detached worktree at the full PR head SHA. Record session owner, task, path, branch (or detached SHA), and PR in the shared queue.
- Only one coordinator may be active; only that coordinator mutates the GitHub queue, security settings, or merges. Other agents send proposed queue updates to it.
- Worktrees share refs/configuration, credentials, and caches; namespace environments, temporary outputs, databases, and ports per session. Do not change shared Git configuration or another session's branch/ref. Remove only your session's clean, inactive trees after ownership is released in the queue; never force-remove another session's tree.

## Startup: “Start the next ready milestone”

- Act as coordinator. Read `docs/design/overview.md`, `docs/plans/milestones.md`, and `docs/development/agent-workflow.md`; inspect current GitHub issues, PRs, CI, and the linked Project before scheduling work.
- Confirm the initial planning PR is merged. If it is still open, report its link and wait for owner plan approval; do not treat this startup prompt as approval of an unmerged plan.
- Choose the earliest incomplete milestone with satisfied prerequisites, starting with M0. Work through its ready tasks; milestone completion requires its exit criteria, not merely green tests.
- Use the Project as the shared queue. Reconcile draft cards with existing issues before creating anything; convert only scheduled approved drafts into Cineforge issues with acceptance criteria and dependencies.
- Confirm the coordinator's exclusive ownership in the shared queue before mutations. Delegate implementation, independent code review, and independent acceptance testing to separate agents with unique session worktrees/author branches and separate detached review/test worktrees; run at most three independent implementation tasks concurrently. Follow the role prompts in the workflow document.
- Open one PR per task. Route findings back to its author, obtain renewed approvals after fixes, verify all current-head gates and integration with main, then merge as coordinator without routine owner confirmation. Record PR/head/evidence links and move cards to Done only after verified merge.
- Maintain these rules and the workflow as the source of truth; do not rely on another chat's memory. Do not count chat reports as enforced GitHub checks; M0 establishes and verifies the automation.
- Continue until the selected milestone meets its exit criteria or a blocker prevents meaningful progress. Finish with merged PRs, test/review evidence, Project status, blockers, and the next ready milestone; do not start the following milestone unless requested.
