# Architecture proposal

`Auteur (planner) → production manifest → scheduler → model adapter → execution runner → artifact store → editorial/audio review`.

GPU provisioners acquire/release workers independently from execution runners. Persist shot revisions, jobs, model/workflow versions, costs and artifact hashes; retries use idempotency keys.

M6 adds a private internal review UI after CLI/backend validation, covering storyboard, source images/video inputs, prompts, audio, characters, GPU/provider, pinned open models, and estimated cost/budget. Review flow: `draft comments → owner finishes review → agent captures comments and creates a new revision → owner reviews again → ApprovePlan(exact snapshot) → production execution`.

Plan/model/provider/GPU/asset/budget changes invalidate plan approval; missing or stale approval blocks production execution. Final picture/audio approval remains separate; previews use existing/private assets or require a separately approved generation budget.

`contracts.py` validates immutable v1 `ProjectPlan` and self-contained `RunManifest` records with strict dict and canonical JSON codecs. `tests/fixtures/production-v1.json` is synthetic metadata; URI/license/revision declarations do not verify storage, rights or reproducibility.

`providers.py` executes metadata-only fake jobs independently of the protocol-only `provisioning.py`; storage remains a later [milestone](../plans/milestones.md) task.
