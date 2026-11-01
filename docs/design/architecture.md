# Architecture proposal

`Auteur (planner) → production manifest → scheduler → model adapter → execution runner → artifact store → editorial/audio review`.

GPU provisioners acquire/release workers independently from execution runners. Persist shot revisions, jobs, model/workflow versions, costs and artifact hashes; retries use idempotency keys.

`contracts.py` validates immutable v1 `ProjectPlan` and self-contained `RunManifest` records with strict dict and canonical JSON codecs. `tests/fixtures/production-v1.json` is synthetic metadata; URI/license/revision declarations do not verify storage, rights or reproducibility.

`providers.py` executes metadata-only fake jobs independently of the protocol-only `provisioning.py`; storage remains a later [milestone](../plans/milestones.md) task.
