# GPU and model proposal

| Boundary | Contract |
| --- | --- |
| Provisioner | acquire(requirements, budget), inspect(), release() |
| Execution provider | submit(request, idempotency_key), status(), cancel(), result() |
| Model adapter | validate input, translate request, normalize output |
| Device | backend, VRAM, identity, supported precision |
| Model | task, pinned revision, allowed backends, memory, license |

Support CUDA, ROCm, Metal, and CPU only where the selected model/workflow declares compatibility. Start with a fake provider, then a local ComfyUI runner; cloud lifecycle adapters require bounded spend, timeouts, redacted logs, and cleanup tests.
