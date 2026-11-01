# Cineforge design

## Vision and scope

A modular AI filmmaking platform: brief → screenplay → storyboard → shots → picture edit → sound → review → delivery. Keep creative intent and production metadata portable across models, GPU providers, and editing tools; require human approval of final picture, voice likeness, music, and mix.

The private Python package implements strict production contracts, model/device selection, offline mock checkpoint/resume, and an explicitly opted-in private ComfyUI single-shot command. [M2 measured one RTX 4090 shot](../evidence/m2-live-shot.json), including manual recovery after a now-fixed history-number mismatch; automated provisioning, live resume, audio processing and a web UI remain future work.

## File structure

```text
cineforge/
├── README.md                  # Install and current capabilities
├── AGENTS.md                  # Contributor and agent rules
├── pyproject.toml             # Python package and CLI
├── .env.example               # Empty credential placeholders
├── .github/workflows/ci.yml   # Offline CI
├── docs/
│   ├── design/
│   │   ├── overview.md        # Vision, layout, and contracts (this file)
│   │   ├── architecture.md    # Production flow and reproducibility
│   │   ├── providers.md       # GPU and model contracts
│   │   ├── characters.md      # Identity, references, and consent
│   │   ├── audio.md           # Sound planning and review
│   │   ├── skills.md          # Skill provenance and trust
│   │   └── licensing.md       # Code, model, and asset rights
│   ├── plans/milestones.md    # Tasks, dependencies, acceptance criteria
│   └── development/
│       └── agent-workflow.md  # Parallel agents and PR quality gates
├── src/cineforge/             # Contracts, devices, fake provider, mock CLI
└── tests/                     # Offline unit and CLI tests
```

Current modules include `contracts.py`, `devices.py`, `providers.py`, `mock.py`, `storage.py`, `comfy.py`, `comfy_transport.py`, and `workflows.py`; character, audio, and skills registries remain planned. `examples/synthetic-plan.json` declares synthetic models and devices explicitly; neither hardware discovery nor model downloads occur.

`comfy_policy.py`, `comfy_run.py`, and `private_store.py` separate trusted selection/worker admission, bounded one-shot orchestration, and private local persistence. Creative plans cannot supply approval records; the live command requires explicit opt-in and separate operator-verified configuration, with no installation or cloud provisioning.

## Architecture and data flow

The future orchestration layer, Auteur, translates a brief into versioned scenes, shots, character references, and an audio cue sheet. Each shot resolves a model requirement, selects a compatible execution target, submits a job, and records output artifacts before editorial review; failed shots can be retried without regenerating the whole film.

M6 requires a private internal review UI after the CLI/backend contracts are validated: the owner reviews storyboard/assets, prompts, audio, characters, GPU/model choices and cost, then submits completed-review comments for the agent to produce a new revision. Production execution requires approval of the exact plan snapshot; changes invalidate that approval, while final picture/audio review remains separate. See the [review flow](architecture.md) and [M6.5 acceptance criteria](../plans/milestones.md#m6--auteur-orchestration-and-editorial-delivery).

Keep orchestration independent from execution. GPU providers provision compute; runners such as ComfyUI execute workflows; model adapters translate creative requests into runner inputs. Hosted APIs can implement the execution protocol without owning a GPU provisioner.

The version-one production manifest records project/scene/shot IDs, revision, prompt, seed where supported, model revision, workflow digest, character revision, provider/job ID, timestamps, cost, and artifact checksums. Immutable run records support retries, comparison, and lineage; seeds are evidence, not a guarantee of identical outputs across hardware.

## GPU-provider and model abstraction

`GPUDevice` describes backend, VRAM, and device identity; `ModelSpec` declares supported backends, minimum memory, task, and pinned revision. Select only compatible devices and report an explicit failure when none qualifies; never infer AMD, NVIDIA, Apple, or CPU compatibility from memory alone.

Initial backend vocabulary: CUDA, ROCm, Metal, CPU. Future provisioners cover local devices and cloud workers; a provisioner manages acquire/release and cost limits, while an execution provider manages submit/status/cancel/result. Model registries must carry license and commercial-use restrictions before live execution is enabled.

Production adapters need timeouts, bounded retries, idempotency keys, cancellation, per-job budgets, and guaranteed cleanup in failure paths. The fake provider implements submit/status/cancel/result with metadata-only JSON payloads and synthetic logical timestamps. `mock-run` preflights every request and device before submission, bounds polling, and emits a canonical manifest; it neither provisions GPUs nor generates media.

`MockPlan.from_json` / `load_mock_plan` parse the strict `{schema_version, plan, models, devices}` wrapper; `to_json` covers all declarations. `MockPlan.preflight()` resolves every `ExecutionRequest`; `mock_run_id` and `mock_idempotency_key` supply stable identities. `execute_mock` returns a `MockRun` containing the manifest and `ExecutionResult` payloads. Optional `output_dir` / `resume` use `LocalRunStore`: a version-one checkpoint binds full execution declarations to a validated manifest/attempt history, and immutable metadata payloads carry verified checksums/sizes. Completed shots are verified before any resumed submission; failed jobs retry with new attempts and interrupted fake jobs replay deterministically.

Storage holds a nonblocking POSIX `flock` for the run and uses same-directory temporary writes, `fsync`, and atomic replacement; payload persistence precedes success checkpointing. Symlinks/hardlinks, unsafe artifact paths, changed inputs, and corrupt/missing completed bytes fail closed. This is single-host local-filesystem mock recovery; live execution recovery, hostile writers, and distributed locking remain outside M1.

## Character registry

Identify characters by stable IDs and explicit revisions. Separate identity description, reference image URIs, voice references, wardrobe, and model-specific embeddings/LoRAs; shot plans pin revisions instead of relying on a mutable name.

Keep actual images, embeddings, and voice recordings in private artifact storage. Store rights/consent records beside references and prohibit unapproved voice or likeness use; reference metadata must not imply verified consent. The first registry implementation will validate IDs and duplicate revisions in memory; persistence and conditioning adapters follow later.

## Audio subsystem

Plan dialogue, ambience, Foley, effects, and music as separate stems on a shared timeline. M5 takes short approved script/台词 lines with language (at least English/Mandarin), speaker/character/voice identity, emotion/delivery, pronunciation and timing; full brief-to-screenplay planning remains M6. Cue sheets retain source/rights and media revisions; script, speech, character or picture changes invalidate affected audio/video, synchronization checks and approvals.

Separate speech synthesis, lip-sync/talking-video adapters, mixing and audio/visual evaluation. Approved speech drives talking video or post-process lip sync; M5 proves an original visible-mouth actor speaking at least one short clip each in English and Mandarin, preserving identity. Unsupported face, pose, style or language fails explicitly; the mouthless M4 robot is not lip-sync evidence.

Future adapters analyze speech timing, loudness/true peak/clipping/silence and mix licensed stems; audio-model feedback is optional. Timeline alignment alone cannot prove visible lip sync: technical checks and actual human listening/visual review of exact media revisions must pass before export. See the [audio design](audio.md) for scope and later pinned model, rights, hardware and budget approvals; no download or execution is authorized here.

## Skills management

Maintain a reviewed manifest of skill source URL, immutable revision, license, local entrypoint, and review state. Prefer references or isolated, pinned installation to copying third-party repositories wholesale; copied material requires compatible permissions and retained notices.

Treat skill content as untrusted input. Installation and execution are separate operations; future execution must enforce allowed tools, filesystem boundaries, network destinations, and human approvals for spending, publishing, or likeness use. The first skills implementation will store provenance and expose review readiness without downloading or executing skills.

## Artifacts, security, and licensing

Git contains code, documentation, workflow definitions, and synthetic metadata. Keep secrets in environment variables or a secret manager, and media/model weights in private object storage; artifact manifests carry URI, checksum, license, origin, and retention policy. Git ignore rules are a convenience, not a secret scanner.

Do not select an open-source code license on the owner's behalf. The initial package is proprietary (`LicenseRef-Proprietary`); before any public release, choose a code license and audit dependencies, imported skills, model terms, character/voice consent, and media rights separately. A private repository does not grant rights to third-party content.

## Implementation sequence

1. Foundation: package, mock provider, GPU selection, character revisions, cue validation, skill provenance, demo, and offline CI.
2. One real vertical slice: the laptop orchestrates a private ComfyUI endpoint on a manually prepared Runpod GPU worker, with a reviewed model/workflow, private artifact store, run manifest, and one-shot generation. Automated cloud lifecycle and budgets remain M3.
3. Cloud lifecycle and budgets; character conditioning; audio generation, analysis, mixing, and human review.
4. Timeline orchestration, resumable multi-shot jobs, editorial UI, and a rights/security review before an open-source release.

## Verification

`python -m unittest discover -s tests -v` and the CLI smoke check are the default offline checks. `python scripts/test_tiers.py` reports each tier as passed, failed, or skipped; a skipped required tier never satisfies a merge gate.

CI runs offline and contract tiers in separate interpreters. An absent contract suite is explicitly skipped; an existing suite with zero tests or all tests skipped fails.

| Tier | Suite / policy |
| --- | --- |
| Offline | `tests/`: synthetic unit/CLI fixtures; no runtime network, credentials, GPU, or paid calls |
| Contract | `tests_contract/`: fake-provider lifecycle/idempotency/failure conformance; same offline rules |
| Integration | `tests_integration/`: opt-in real adapters outside default discovery; currently unavailable |
| Media quality | `tests_media/`: opt-in approved media and technical measurements plus separate human review; currently unavailable |

Live admission requires `--allow-live --max-cost-usd 1 --estimated-cost-usd 0.5 --timeout-seconds 60`; invalid or exceeded limits fail. No live tier executes until an adapter enforces actual spend, deadlines, and cleanup; valid opt-in currently reports skipped with no calls made.

CI fixtures must be synthetic, small, deterministic, and rights-cleared; use fake credentials and temporary outputs. Keep private media/weights outside Git, mock transport/provider failures offline, and record commands, head SHA, skips, and limits in test evidence.

## Agent delivery and review

The design and milestone plan are owner-approved and merged. Each task ships through an implementation agent, an independent review agent, and an independent testing agent before coordinator/merge-agent merge once all required checks and both independent approvals pass for the current head; see `docs/development/agent-workflow.md` for evidence and enforcement requirements.

Agents own ongoing repository maintenance and routine merges under the approved plan; the owner provides product direction and may optionally inspect code, with no routine coding or code-review obligation.
