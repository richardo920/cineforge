# Cineforge

Private foundation for a modular AI filmmaking platform: brief → screenplay → storyboard → shots → edit → sound → review → delivery.

**Status:** strict production metadata, offline mock runs, and an explicitly opted-in private ComfyUI single-shot command. [One approved RTX 4090 shot](docs/evidence/m2-live-shot.json) was generated and recovered manually after a history-number mismatch, now fixed; automated GPU provisioning remains M3.

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

Python 3.11+. Runtime has no third-party dependencies. `.env.example` lists future adapter settings; the package does not load `.env`.

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

## CI guards

CI builds/installs wheel and sdist on Python 3.11–3.14; pinned Ruff, mypy, detect-secrets and synthetic failure fixtures run offline. `python scripts/ci_guard.py artifacts` limits tracked blobs to 1 MiB and rejects private/media/model/credential paths, symlinks, and submodules; dependency changes require matching reviewed pins in `config/dependency-policy.json`.

Dependency review also queries advisory services in CI (network required; failures block the job), including GitHub's PR dependency review. Secret scanning covers current staged blobs, not full history; no scanner proves that credentials are absent.

## Layout

- `src/cineforge/`: contracts, device declarations, fake execution provider, local storage, and CLI.
- `tests/`: offline smoke tests; `.github/workflows/ci.yml`: CI.
- `AGENTS.md`: contributor rules and agent review expectations.
- [Design overview](docs/design/overview.md): vision, file structure, and subsystem contracts.
- [Milestone plan](docs/plans/milestones.md): tasks, dependencies, and acceptance criteria.
- [Agent workflow](docs/development/agent-workflow.md): implementation, review, testing, and coordination.
- `docs/design/`: architecture, GPU/model providers, characters, audio, skills, and licensing.

Code is proprietary pending an explicit open-source license decision. Keep credentials, model weights, recordings, and generated media outside Git; review third-party code, skill, model, and asset rights separately before reuse.
