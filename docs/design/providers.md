# GPU and model proposal

| Boundary | Contract |
| --- | --- |
| Provisioner | acquire(requirements, budget), inspect(), release() |
| Execution provider | submit(request, idempotency_key), status(), cancel(), result() |
| Model adapter | validate input, translate request, normalize output |
| Device | backend, VRAM, identity, supported precision |
| Model | task, pinned revision, allowed backends, memory, license |

Support CUDA, ROCm, Metal, and CPU only where the selected model/workflow declares compatibility. Start with a fake provider, then a local ComfyUI runner; cloud lifecycle adapters require bounded spend, timeouts, redacted logs, and cleanup tests.

`cineforge.devices` now provides declared `GPUDevice`/`ModelSpec` contracts and pure offline `select_device(model, devices)`. Selection requires backend, integer MiB capacity and precision compatibility; ties prefer least sufficient memory, then backend value, then device ID.

All fixtures are synthetic. Revision/license/commercial-use fields are declarations, not verified provenance or permission; this module performs no hardware detection, provisioning or execution.

`providers.py` exposes `ExecutionRequest(shot, model, attempt=1)` and `ExecutionProvider`: `submit(request, idempotency_key) -> ShotRun`, `status(job_id) -> ShotRun`, `cancel(job_id) -> ShotRun`, `result(job_id) -> ExecutionResult(run, artifact, payload)`.

`FakeExecutionProvider` submits queued jobs; each status poll advances queued → running → succeeded (`fail=True`: failed). Terminal jobs stay terminal, duplicate keys return the existing snapshot, conflicting requests fail, and only succeeded jobs return results.

Results contain synthetic JSON bytes with full shot/model provenance, actual SHA256/size, zero cost and fixed synthetic timestamps; `memory://` URIs do not write storage. IDs/bytes repeat for the same request/key across instances, but jobs are held only in memory; callers persist payloads separately. `provisioning.Provisioner` defines acquire/inspect/release only, with no implementation.
