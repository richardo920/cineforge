# Cineforge

Private foundation for a modular AI filmmaking platform: brief → screenplay → storyboard → shots → edit → sound → review → delivery.

**Status:** project scaffold only; no media generation, paid API calls, or GPU provisioning. Architecture and implementation milestones are proposed in the planning PR; implementation starts after owner review and merge.

## Quick start

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cineforge --version
python -m unittest discover -s tests -v
```

Python 3.11+. Runtime has no third-party dependencies. `.env.example` lists future adapter settings; the scaffold does not load `.env`.

## Layout

- `src/cineforge/`: package and CLI scaffold.
- `tests/`: offline smoke tests; `.github/workflows/ci.yml`: CI.
- `AGENTS.md`: contributor rules and agent review expectations.
- Planning PR: `DESIGN.md`, `MILESTONES.md`, and subsystem documentation under `docs/`.

Code is proprietary pending an explicit open-source license decision. Keep credentials, model weights, recordings, and generated media outside Git; review third-party code, skill, model, and asset rights separately before reuse.
