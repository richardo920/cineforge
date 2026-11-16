# Cineforge

Private foundation for a modular AI filmmaking platform: brief → screenplay → storyboard → shots → edit → sound → review → delivery.

**Capabilities:** strict production metadata, offline mock checkpoint/resume, private ComfyUI execution, managed Runpod smoke/recovery, character conditioning, and audio contracts/analysis/mixing/review. Use `cineforge --help` for available commands; live runs require separate exact approvals.

## Quick start

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cineforge --version
cineforge validate-plan examples/synthetic-plan.json
cineforge mock-run examples/synthetic-plan.json --run-id demo-run
cineforge mock-run examples/synthetic-plan.json --output-dir ./demo-run
cineforge mock-run examples/synthetic-plan.json --output-dir ./demo-run --resume
python -m unittest discover -s tests -v
```

Python 3.11+. Runtime has no third-party dependencies. The package does not load `.env`. `.env.example` documents only the optional `RUNPOD_API_KEY` shell fallback for cloud commands; `--api-key-file` takes precedence. Mock runs need no credentials. Worker-guard credentials/Pod identity and fixed PyTorch launcher controls are injected worker inputs, not local configuration.

The example wraps a `ProjectPlan` in `{schema_version: 1, plan, models, devices}` with explicit synthetic declarations; see its fields for the strict input schema. `validate-plan` checks the entire wrapper and emits its canonical domain `ProjectPlan`; `mock-run` consumes the wrapper and emits a canonical `RunManifest` only after all shots succeed.

Both commands reject malformed UTF-8/JSON, unknown fields/references, invalid model metadata, and incompatible devices before submission. Without `--output-dir`, artifacts are ephemeral metadata (`memory://`). A dedicated output directory stores `checkpoint.json` (full inputs and versioned manifest) plus checksum-named JSON payloads; `--resume` verifies all completed bytes before skipping them. Identical inputs produce identical output; `--run-id` overrides the default input digest.

Persistence requires a local POSIX filesystem with `flock`, no-follow file opens, atomic rename, and `fsync`; another writer fails immediately. Symlink ancestors/files, hardlinks, unrelated files, changed inputs, and corrupt/missing completed artifacts are rejected. Locks coordinate cooperating processes on one host; network filesystems and hostile directory writers are unsupported.

Failed attempts remain in the manifest; resume retries only unfinished/failed shots. Interrupted attempts replay the deterministic fake provider with the same identity; this recovery guarantee applies only to synthetic mock jobs. Artifact bytes precede the atomic success checkpoint; interrupted temporary files and valid unreferenced payloads can remain after a crash.

## Approved ComfyUI shot

Choose a trusted reviewed selection separately from the creative `ProjectPlan`; its approval hash binds declarations and does not authenticate the owner. Complete [the worker record](examples/worker-verification.example.json) from verified image/runtime/GPU, runner revision, and actual model-file hashes; examples deliberately refuse execution.

```sh
cineforge comfy-run /private/path/shot-plan.json \
  --selection config/comfy/wan21-t2v-1.3b.json \
  --worker-verification /private/path/worker-verification.json \
  --output-dir /private/path/new-run --allow-live
```

The default endpoint is a separately established `http://127.0.0.1:8188` tunnel; no cloud rental, SSH, weight download, or node installation occurs. The command checks core-node schemas/model filenames, runs exactly one matching shot, and stores `run.json` plus checksum-named bytes in a new private POSIX directory outside Git.

`--timeout-seconds`, `--request-timeout-seconds`, `--max-polls`, and `--max-result-bytes` bound client work; a failure allows one separately bounded owned-job cancellation attempt. Acknowledgements do not prove GPU work stopped: inspect the retained job ID/state before any new submission, manage the Pod separately, and never treat these bounds as cloud billing enforcement; live resume is unsupported.

## Managed cloud smoke

Cloud smoke uses an exactly approved paper-boat selection and a verified prepared-endpoint reference. Supply external owned `0600` API/SSH files in private directories, `ffprobe`, and one durable budget directory for attempts and recovery. Historical runs and classifier corrections remain in [M3 evidence](docs/evidence/m3-live-shot.json); these examples grant no new live authorization.

```sh
cineforge cloud-smoke --allow-live \
  --selection config/comfy/wan21-t2v-1.3b.json \
  --proposal config/cloud/runpod-pod-proposal.json \
  --session-id mysession --ssh-private-key "$HOME/.ssh/cineforge" \
  --api-key-file "$HOME/.config/cineforge/runpod-api-key" \
  --budget-dir /path/to/private/m3-budget \
  --reference-run /path/to/prepared/run.json --output-dir /path/to/new-run
```

The historical M3 scope capped two sequential RTX4090/US-IL-1 Pods within USD3. Within an authorized scope, `--case post-acquire-failure` withdraws only the remote heartbeat, observes independent STOP, then verifies deletion. Keep the same session/key/budget; use a new output directory, never reset the journal to retry.

```sh
cineforge cloud-recover --allow-live \
  --selection config/comfy/wan21-t2v-1.3b.json \
  --proposal config/cloud/runpod-pod-proposal.json \
  --api-key-file "$HOME/.config/cineforge/runpod-api-key" \
  --budget-dir /path/to/private/m3-budget
```

Recovery discovers/releases journal-owned workers without acquiring or requiring SSH; unavailable invoices remain unknown. [Controls and limitations](docs/design/providers.md#m3-provider-review) include independent worker supervision, private outputs, isolated SSH host-key TOFU, and no account-wide hard billing cap.

## CI guards

CI builds/installs wheel and sdist on Python 3.11–3.14; pinned Ruff, mypy, detect-secrets and synthetic failure fixtures run offline. `python scripts/ci_guard.py artifacts` limits tracked blobs to 1 MiB and rejects private/media/model/credential paths, symlinks, and submodules; dependency changes require matching reviewed pins in `config/dependency-policy.json`.

Dependency review also queries advisory services in CI (network required; failures block the job), including GitHub's PR dependency review. Secret scanning covers current staged blobs, not full history; no scanner proves that credentials are absent.

## Layout

- `src/cineforge/`: contracts, device declarations, fake execution provider, local storage, and CLI.
- `tests/`: offline unit and CLI tests; `.github/workflows/ci.yml`: CI.
- `AGENTS.md`: contributor rules and agent review expectations.
- [Design overview](docs/design/overview.md): vision, file structure, and subsystem contracts.
- [Milestone plan](docs/plans/milestones.md): tasks, dependencies, and acceptance criteria.
- [Agent workflow](docs/development/agent-workflow.md): implementation, review, testing, and coordination.
- `docs/design/`: architecture, GPU/model providers, characters, audio, skills, and licensing.

Code is proprietary pending an explicit open-source license decision. Keep credentials, model weights, recordings, and generated media outside Git; review third-party code, skill, model, and asset rights separately before reuse.
