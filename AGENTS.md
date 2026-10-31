# Repository guidance

- Implement only owner-approved tasks from a merged milestone plan; keep one task per PR.
- Never add Codex as a commit author or co-author. Use default Git identity.
- Never commit secrets, media, model weights, private character references, or unreviewed third-party skills.
- Do not create Markdown files unless explicitly requested. Keep requested documentation concise; prefer examples.
- Keep orchestration, GPU provisioning, workflow execution, model adapters, and storage separate.
- Default tests must run offline without credentials, GPUs, or paid services. Run `python -m unittest discover -s tests -v` and the CLI smoke check.
- Use a separate review agent and testing agent for implementation PRs. Review correctness/security and test acceptance criteria independently; record findings and evidence against the current PR head SHA.
- The coordinator/merge agent may merge approved implementation tasks without another owner confirmation after all required CI checks pass, independent review explicitly passes, and independent acceptance testing passes for the current PR head. Authors cannot self-certify review or testing.
- New commits invalidate prior review/testing approval. Verify the current head and integration with current main before merging; never bypass protections or treat pending, skipped, missing, or unavailable required checks as passed. The initial design/planning PR still requires owner review.
- Agent reports supplement CI and human review. These instructions alone do not enforce GitHub branch protection.
