# Cineforge milestone plan

**Approved in merged planning PR #1.** Agents schedule within this scope without routine owner coding or review; completion follows verified task evidence in the Project and PRs.

One task per PR by default. Each task needs current-head CI, independent review-agent findings, independent testing-agent approval, and coordinator/merge-agent verification before agent merge; failures and unavailable checks must be reported explicitly.

## M0 — Make agent delivery reviewable

Depends on plan approval. Exit: a sample PR demonstrates the full author → review → testing → coordinator merge loop and enforced protections are verified, or unsupported protections are explicitly recorded.

M0 completion marks take effect only after [PR #15](https://github.com/richardo920/cineforge/pull/15) passes final-head CI, independent review/testing, signed workflow and fresh local evidence verification, then coordinator verifies its merge; [live protection evidence](https://github.com/richardo920/cineforge/pull/15#issuecomment-6545843729) records platform limits.

- [x] M0.1 Define reusable coordinator/implementation/review/testing agent instructions with task boundaries, maintenance ownership, and severity rules. Acceptance: a sample task yields three distinct reports tied to the same PR head SHA; authors cannot self-certify review.
- [x] M0.2 Configure available main-branch protections and required CI. Acceptance: failing checks block merge, new commits invalidate stale approval where supported, and conditional agent-merge policy/any platform limitations are documented and tested on a disposable branch.
- [x] M0.3 Add trusted review/testing evidence checks. Acceptance: missing or stale reports fail; untrusted PR text cannot forge approval; successful CI alone never authorizes merge, and absent/failed/stale independent approval blocks agent merge; do not use privileged workflows to execute untrusted PR code. Record whether evidence verifies identity, completion, or findings resolution.
- [x] M0.4 Add lint/type checks, package build/install checks, secret scanning, dependency review, and a repository artifact-size policy. Acceptance: synthetic bad fixtures fail each relevant guard without including a real secret; pin tool versions and action revisions.
- [x] M0.5 Define offline/contract/integration/media-quality test tiers and CI fixture policy. Acceptance: default tests need no credentials/GPU/network at runtime; paid runs are opt-in, cost-bounded, and clearly reported as run/skipped.

- [x] M0.6 Operationalize the private Cineforge GitHub Project (board created; proposed tasks start as Backlog drafts). Acceptance: each approved task is an issue with task ID, milestone, acceptance criteria, dependencies, linked PR, owner agent, and workflow status; coordinator moves tasks through independent review/testing to Done only after verified merge. Board state cannot substitute for merge checks.

## M1 — Domain contracts and offline production demo

Depends on M0. Exit: a synthetic shot plan completes through a mock provider with reproducible metadata and no media generation.

M1 completion marks take effect only after M1.5 passes current-head CI and independent review/testing, the coordinator verifies its merge, and a synthetic plan → mock → stored manifest → fresh-process resume demonstration passes. M1 remains metadata-only; live execution is M2+.

- [x] M1.1 Define versioned project, scene, shot, artifact, and run-manifest schemas. Acceptance: round-trip fixture, missing IDs, incompatible schema versions, invalid numeric values, and unknown references are tested.
- [x] M1.2 Define GPU/model compatibility and deterministic device selection. Acceptance: CUDA/ROCm/Metal/CPU fixtures; incompatible backend, insufficient memory, and no-device cases fail explicitly.
- [x] M1.3 Define execution-provider and separate provisioner protocols; implement a fake execution provider. Acceptance: submit/status/cancel/result transitions and duplicate idempotency keys are tested without external calls.
- [x] M1.4 Add CLI plan validation and mock-run commands with synthetic examples. Acceptance: valid plan produces metadata-only output and provenance; invalid plans return nonzero without partial success claims.
- [x] M1.5 Implement local artifact manifests and resumable job records. Acceptance: checksums, atomic writes, interrupted runs, failed jobs, and retry-without-regenerating-completed-shots are tested.

## M2 — First real shot through a private ComfyUI endpoint

Depends on M1. The laptop orchestrates a private ComfyUI endpoint on a manually prepared Runpod GPU worker; M3 retains automated cloud lifecycle and budgets. Exit: one explicitly approved model/workflow produces a short shot with provenance and bounded execution.

The [pinned Wan2.1 selection](../../config/comfy/wan21-t2v-1.3b.json) and RTX4090/24GB worker downloads are [owner-approved](https://github.com/richardo920/cineforge/issues/40#issuecomment-6553169092) within a USD3 cap. [Measured evidence](../evidence/m2-live-shot.json) records one real shot and manual read-only recovery; M2 completion takes effect after M2.4 passes current-head CI, independent review/testing and verified merge.

- [x] M2.1 Select and review a pinned model/workflow with supported hardware and usage terms. Acceptance: rights ledger, memory expectations, immutable versions, and owner approval precede downloading weights or paid execution.
- [x] M2.2 Implement ComfyUI transport and workflow translation. Acceptance: mocked transport covers submit/poll/result/cancel, malformed replies, timeouts, and connection loss; default CI stays offline.
- [x] M2.3 Enforce artifact destinations, safe workflow inputs, secret redaction, and private endpoint defaults. Acceptance: path traversal and credential-bearing error fixtures are blocked/redacted; unapproved custom-node code is never installed automatically.
- [x] M2.4 Run one opt-in end-to-end shot with an approved hardware setup. Acceptance: output checksum, prompt/model/workflow revision, timing, failure recovery, and visual review are recorded; no hardware compatibility is claimed without evidence.

## M3 — GPU providers and budgets

Depends on M2. Exit: one chosen cloud GPU provider can run the same shot with cleanup and cost controls.

The [owner-approved Runpod scope](../../config/cloud/runpod-pod-proposal.json) has a separate USD3 M3 validation cap; the M3.1 mark takes effect only after [PR #52](https://github.com/richardo920/cineforge/pull/52) passes current-head review/testing/CI and its merge is verified.

- [x] M3.1 Compare candidate providers and select one after terms, region, storage, pricing, and data-retention review. Acceptance: owner approves a provider and explicit spend limit; no provisioning during research.
- [ ] M3.2 Implement worker acquire/inspect/release with credentials supplied externally. Acceptance: fake-provider tests cover capacity failure, timeout, duplicate requests, cancellation, and cleanup after every failure path.
- [ ] M3.3 Add budget preflight, execution deadlines, idle shutdown, and cost records. Acceptance: over-budget jobs refuse before launch; interrupted coordinator recovery finds/releases orphaned workers; cost estimates and actuals remain distinguishable.
- [ ] M3.4 Add opt-in parity smoke test for local/cloud execution. Acceptance: normalized output contracts agree, secrets remain absent from logs, and worker release is verified after success and injected failure.

## M4 — Characters, skills, and rights

Depends on M1; live character validation also depends on M2. Exit: a shot can pin a reviewed character revision and a reviewed skill source with traceable permissions.

- [ ] M4.1 Implement versioned character registry. Acceptance: stable IDs, immutable revisions, missing/duplicate records, wardrobe/reference links, and revoked consent are tested.
- [ ] M4.2 Implement conditioning adapter for the selected workflow. Acceptance: supported references resolve with checksums; unsupported model conditioning fails explicitly; human review compares identity across at least three shots.
- [ ] M4.3 Implement skill provenance/rights manifest and review readiness checks. Acceptance: unpinned revision, unknown license, missing notices, and unreviewed content cannot become execution-ready.
- [ ] M4.4 Implement isolated skill installation/execution policies. Acceptance: test fixtures cannot escape allowed paths, invoke unapproved tools, or reach unapproved network destinations; no wholesale upstream copying.
- [ ] M4.5 Implement rights ledger for code/model/skill/voice/music/assets. Acceptance: export preflight blocks missing or incompatible permissions and produces an attribution record; unknown terms remain blocked pending owner review.

## M5 — Sound as a first-class production track

Depends on M1 and M4 rights checks; picture integration depends on M2. Exit: one short scene has synchronized stems, technical checks, and human listening approval.

- [ ] M5.1 Implement dialogue, ambience, Foley/effects, and music cue schemas. Acceptance: shared timebase, overlaps, finite timing, end bounds, character references, and picture-revision invalidation are tested.
- [ ] M5.2 Select licensed sources and implement one dialogue and one sound/music adapter. Acceptance: provenance/rights recorded for every cue, consent gates for voices, mocked default tests, and bounded opt-in generation.
- [ ] M5.3 Implement objective audio analysis and optional audio-model review. Acceptance: known fixtures detect clipping, silence, loudness/true peak, speech timing, and synchronization problems; unavailable qualitative analysis is reported honestly.
- [ ] M5.4 Implement stem mixing and delivery presets. Acceptance: pinned tooling and deterministic fixture checks cover ducking, fades, offsets, clipping prevention, and documented target-specific loudness settings.
- [ ] M5.5 Define human listening checklist and approval record. Acceptance: owner reviews intelligibility, sync, musical fit, artifacts, transitions, and balance; failed listening review blocks final export.

## M6 — Auteur orchestration and editorial delivery

Depends on M2, M4, M5; cloud mode also depends on M3. Exit: the owner reviews and approves a versioned plan through a private internal UI before production execution, then reviews the resulting multi-shot scene with final audio and complete lineage. Build CLI/backend contracts before the UI.

- [ ] M6.1 Implement brief → scene/shot/storyboard plans with structured outputs and owner approval of the exact plan revision. Acceptance: malformed plans, unsupported requests, and missing/stale approval block production generation/spend; any paid preview generation requires its own explicit budget approval.
- [ ] M6.2 Implement dependency-aware scheduling, bounded retries, cancellation, and resume. Acceptance: injected failures preserve completed work, prevent duplicate charges where provider semantics permit, and report uncertain submission states without blind retries.
- [ ] M6.3 Implement editorial timeline and picture-lock invalidation. Acceptance: trim/reorder transitions preserve timebase and invalidate dependent audio, captions, and prior approvals.
- [ ] M6.4 Implement CLI-first scene review/export backend. Acceptance: a short scene exports with captions, stems, picture/audio approvals, rights ledger, and full manifest; private assets never enter Git.
- [ ] M6.5 Build the private internal review UI after validating the CLI/backend contracts: storyboard, source images/video inputs, prompts, audio, characters, GPU/provider, pinned open models, and estimated cost/budget. Acceptance: comments are submitted when the owner finishes a review; the agent produces a new plan revision for repeated review until explicit approval of that exact snapshot permits production execution. Plan/model/provider/GPU/asset/budget changes invalidate prior approval; final picture/audio approval remains separate.

## M7 — Release readiness and possible open source

Depends on M6. Exit: owner decides whether to stay private or approve a separately reviewed public release.

- [ ] M7.1 Measure end-to-end reliability, cost, latency, continuity, and audio quality on a small approved benchmark. Acceptance: repeatable inputs, failure cases, costs, and human ratings are recorded with limits stated.
- [ ] M7.2 Audit dependencies, copied code/skills, model terms, notices, and artifact permissions. Acceptance: inventory has immutable versions and resolved distribution rights; owner selects a code license explicitly.
- [ ] M7.3 Audit security and full Git history for secrets/private assets; practice backup/recovery. Acceptance: clean scan evidence and recovery drill; any exposed credential is rotated rather than merely deleted from a file.
- [ ] M7.4 Prepare minimal onboarding and release checklist. Acceptance: clean-checkout setup works, supported hardware is evidenced, and public visibility remains unchanged until explicit owner approval.

## First assignments after approval

Start with M0.1 and M0.5; follow with M0.2–M0.4 and M0.6, then M1. Projects setup need not block otherwise-ready foundation work. Do not schedule live generation, provider provisioning, skill execution, or public release until the corresponding acceptance gates are satisfied.
