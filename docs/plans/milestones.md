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

The [owner-approved Runpod scope](../../config/cloud/runpod-pod-proposal.json) has a separate USD3 M3 validation cap; M3.1–M3.3 merged through [#52](https://github.com/richardo920/cineforge/pull/52), [#54](https://github.com/richardo920/cineforge/pull/54), and [#56](https://github.com/richardo920/cineforge/pull/56). M3.4's proposed post-merge mark takes effect only after [PR #57](https://github.com/richardo920/cineforge/pull/57) passes current-head independent gates and its merge is verified. [Recorded evidence](../evidence/m3-live-shot.json) preserves both actual runs, including failure-test CLI exit2 after observed `EXITED`/404 and the subsequent offline classifier correction without another GPU run.

- [x] M3.1 Compare candidate providers and select one after terms, region, storage, pricing, and data-retention review. Acceptance: owner approves a provider and explicit spend limit; no provisioning during research.
- [x] M3.2 Implement worker acquire/inspect/release with credentials supplied externally. Acceptance: fake-provider tests cover capacity failure, timeout, duplicate requests, cancellation, and cleanup after every failure path.
- [x] M3.3 Add budget preflight, execution deadlines, idle shutdown, and cost records. Acceptance: over-budget jobs refuse before launch; interrupted coordinator recovery finds/releases orphaned workers; cost estimates and actuals remain distinguishable.
- [x] M3.4 Add opt-in parity smoke test for local/cloud execution. Acceptance: normalized output contracts agree, secrets remain absent from logs, and worker release is verified after success and injected failure.

## M4 — Characters, skills, and rights

Depends on M1; live character validation also depends on M2. Exit: a shot can pin a reviewed character revision and a reviewed skill source with traceable permissions.

The proposed M4 completion marks take effect only after [PR #67](https://github.com/richardo920/cineforge/pull/67) passes current-head independent gates and its merge is verified. [Evidence](../evidence/m4-conditioning.json) preserves the native exit2/local postcheck correction and [owner approval](https://github.com/richardo920/cineforge/issues/60#issuecomment-6565098422), including accepted Shot 2 eye/head distortion; generated-media rights remain internal and unreviewed for publication.

- [x] M4.1 Implement versioned character registry ([PR65](https://github.com/richardo920/cineforge/pull/65)). Acceptance: stable IDs, immutable revisions, missing/duplicate records, wardrobe/reference links, and revoked consent are tested.
- [x] M4.2 Implement conditioning adapter for the selected workflow ([PR67](https://github.com/richardo920/cineforge/pull/67)). Acceptance: supported references resolve with checksums; unsupported model conditioning fails explicitly; human review compares identity across at least three shots.
- [x] M4.3 Implement skill provenance/rights manifest and review readiness checks ([PR63](https://github.com/richardo920/cineforge/pull/63)). Acceptance: unpinned revision, unknown license, missing notices, and unreviewed content cannot become execution-ready.
- [x] M4.4 Implement isolated skill installation/execution policies ([PR66](https://github.com/richardo920/cineforge/pull/66)). Acceptance: test fixtures cannot escape allowed paths, invoke unapproved tools, or reach unapproved network destinations; no wholesale upstream copying.
- [x] M4.5 Implement rights ledger for code/model/skill/voice/music/assets ([PR61](https://github.com/richardo920/cineforge/pull/61)). Acceptance: export preflight blocks missing or incompatible permissions and produces an attribution record; unknown terms remain blocked pending owner review.

## M5 — Sound as a first-class production track

Depends on M1 and M4 characters/rights; picture/execution integration depends on M2/M3 controls. Exit: one short scene has synchronized dialogue, ambience, Foley/effects, and music stems, plus an original actor with a visible mouth speaking approved English and Mandarin lines (at least one short clip per language), technical audio checks, and actual human listening/visual lip-sync approval. The mouthless M4 robot is not lip-sync evidence; supported results must preserve actor identity without claiming universal quality.

M5 consumes short approved script/台词 lines; full brief-to-screenplay planning remains M6. Reuse reviewed upstream models/workflows, with later pinned model/weights/dependencies, rights, hardware and budget approval before any download, upload, paid test or voice cloning; this roadmap grants no execution permission and default CI stays offline.

- [ ] M5.1 Implement dialogue, ambience, Foley/effects, and music cue schemas. Acceptance: dialogue pins approved text/revision, language (at least English/Mandarin), speaker/character/voice identity, emotion/delivery, pronunciation and timing; shared timebase, overlaps, finite timing, end bounds and references are tested. Script, speech, character or picture changes invalidate affected audio/video, sync checks and approvals.
- [ ] M5.2 Select licensed sources and implement one speech-synthesis and one sound/music adapter. Acceptance: English and Mandarin approved lines, provenance/rights for every cue, voice consent gates, explicit unsupported-language failures, mocked default tests and bounded opt-in generation; speech synthesis remains separate from M5.6 talking video and M5.4 mixing.
- [ ] M5.3 Implement objective audio analysis and optional audio-model review. Acceptance: known fixtures detect clipping, silence, loudness/true peak, speech timing and synchronization problems; audio timeline alignment alone cannot certify visible lip sync, and unavailable qualitative analysis is reported honestly.
- [ ] M5.4 Implement stem mixing and delivery presets. Acceptance: pinned tooling and deterministic fixture checks cover ducking, fades, offsets, clipping prevention, and documented target-specific loudness settings.
- [ ] M5.5 Define human listening/visual review checklist and approval record, including M5.6 results. Acceptance: actual owner review records English/Mandarin intelligibility, visible lip sync, actor identity, musical fit, artifacts, transitions and balance against exact media revisions; missing, failed or stale approval blocks final export.
- [ ] M5.6 Implement a separate lip-sync/talking-video adapter. Depends on M5.1/M5.2, M4 characters/rights and M2/M3 video/execution controls. Acceptance: approved speech drives talking video or post-process lip sync; an original visible-mouth actor speaks at least one short approved clip per language (English/Mandarin), preserving identity. Pin speech/video lineage and timing; changes invalidate affected media and approvals. Unsupported face, pose, style or language fails explicitly; technical checks and M5.5 human listening/visual review both pass. Code-switching and multi-person scenes are outside the M5 baseline.

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
