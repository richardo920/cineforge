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
