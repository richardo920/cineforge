# Cineforge

Private foundation for a modular AI filmmaking platform: brief → screenplay → storyboard → shots → edit → sound → review → delivery.

**Status:** strict versioned production metadata, declared device/model selection, and a deterministic offline mock runner. No media generation, paid API calls, or GPU provisioning.

## Quick start

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cineforge --version
cineforge validate-plan examples/synthetic-plan.json
cineforge mock-run examples/synthetic-plan.json --run-id demo-run
python -m unittest discover -s tests -v
```

Python 3.11+. Runtime has no third-party dependencies. `.env.example` lists future adapter settings; the package does not load `.env`.

The example wraps a `ProjectPlan` in `{schema_version: 1, plan, models, devices}` with explicit synthetic declarations; see its fields for the strict input schema. `validate-plan` checks the entire wrapper and emits its canonical domain `ProjectPlan`; `mock-run` consumes the wrapper and emits a canonical `RunManifest` only after all shots succeed.

Both commands reject malformed UTF-8/JSON, unknown fields/references, invalid model metadata, and incompatible devices before submission. Mock artifacts are ephemeral metadata (`memory://`); output files and resume are not implemented. Identical inputs produce identical output; `--run-id` overrides the default input digest.

## CI guards

CI builds/installs wheel and sdist on Python 3.11–3.14; pinned Ruff, mypy, detect-secrets and synthetic failure fixtures run offline. `python scripts/ci_guard.py artifacts` limits tracked blobs to 1 MiB and rejects private/media/model/credential paths, symlinks, and submodules; dependency changes require matching reviewed pins in `config/dependency-policy.json`.

Dependency review also queries advisory services in CI (network required; failures block the job), including GitHub's PR dependency review. Secret scanning covers current staged blobs, not full history; no scanner proves that credentials are absent.

## Layout

- `src/cineforge/`: contracts, device declarations, fake execution provider, and CLI.
- `tests/`: offline smoke tests; `.github/workflows/ci.yml`: CI.
- `AGENTS.md`: contributor rules and agent review expectations.
- [Design overview](docs/design/overview.md): vision, file structure, and subsystem contracts.
- [Milestone plan](docs/plans/milestones.md): tasks, dependencies, and acceptance criteria.
- [Agent workflow](docs/development/agent-workflow.md): implementation, review, testing, and coordination.
- `docs/design/`: architecture, GPU/model providers, characters, audio, skills, and licensing.

Code is proprietary pending an explicit open-source license decision. Keep credentials, model weights, recordings, and generated media outside Git; review third-party code, skill, model, and asset rights separately before reuse.
