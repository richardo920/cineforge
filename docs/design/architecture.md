# Architecture proposal

`Auteur (planner) → production manifest → scheduler → model adapter → execution runner → artifact store → editorial/audio review`.

GPU provisioners acquire/release workers independently from runners such as ComfyUI. Persist shot revisions, job state, workflow/model versions, costs, and artifact hashes; retries resume individual jobs with idempotency keys. All components beyond the CLI scaffold are proposed; see [design](overview.md) and [milestones](../plans/milestones.md).
