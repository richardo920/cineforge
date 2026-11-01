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
