"""Reviewed worker capability data; these constants grant no approval or execution authority."""

from typing import NamedTuple


class ReferenceRuntime(NamedTuple):
    torch_version: str
    maximum_memory_mib: int
    capability: tuple[int, int]
    architecture: str
    minimum_effective_ram_bytes: int
    minimum_available_ram_bytes: int
    minimum_filesystem_free_bytes: int


class WorkerProfile(NamedTuple):
    profile_id: str  # Empty means the existing unnamed/default T2V worker.
    runner_revision: str
    model_revision: str
    distribution: str
    gpu: str
    minimum_memory_mib: int
    cuda_version: str
    maximum_asset_bytes: int
    files: tuple[tuple[str, int, str], ...]
    reference_runtime: ReferenceRuntime | None


T2V_WORKER = WorkerProfile(
    profile_id="",
    runner_revision="6d352802d8f8b9133f664a1757d6c6d26390c335",  # pragma: allowlist secret - public source revision
    model_revision="123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",  # pragma: allowlist secret - public model revision
    distribution="Wan-2.1", gpu="NVIDIA GeForce RTX 4090",
    minimum_memory_mib=24082, cuda_version="12.8",
    maximum_asset_bytes=8 * 1024**3,
    # Legacy downloads are selection/approval-bound on the host, not an exact worker manifest.
    files=(), reference_runtime=None,
)

I2V_WORKER = WorkerProfile(
    profile_id="m4-wan22-ti2v5b-5090-v1",
    runner_revision="6d352802d8f8b9133f664a1757d6c6d26390c335",  # pragma: allowlist secret - public source revision
    model_revision="ee6f4a40737a995bf5818954cfce6d59443b0f04",  # pragma: allowlist secret - public model revision
    distribution="Wan-2.2", gpu="NVIDIA GeForce RTX 5090",
    minimum_memory_mib=30720, cuda_version="12.8",
    maximum_asset_bytes=10_000_000_000,
    files=(
        ("split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", 9999658848,
         "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e"),  # pragma: allowlist secret - public weight digest
        ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897,
         "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"),  # pragma: allowlist secret - public weight digest
        ("split_files/vae/wan2.2_vae.safetensors", 1409400960,
         "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"),  # pragma: allowlist secret - public weight digest
    ),
    reference_runtime=ReferenceRuntime("2.8.0+cu128", 32768, (12, 0), "sm_120",
                                       64_000_000_000, 32 * 1024**3, 30 * 1024**3),
)

S2V_WORKER = WorkerProfile(
    profile_id="m5-wan22-s2v14b-5090-v1",
    runner_revision="6d352802d8f8b9133f664a1757d6c6d26390c335",  # pragma: allowlist secret - public source revision
    model_revision="ee6f4a40737a995bf5818954cfce6d59443b0f04",  # pragma: allowlist secret - public model revision
    distribution="Wan_2.2_ComfyUI_Repackaged", gpu="NVIDIA GeForce RTX 5090",
    minimum_memory_mib=30720, cuda_version="12.8", maximum_asset_bytes=17_000_000_000,
    files=(
    ("split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", 16394832474,
     "140e75af5534ac3d91e710d9df756f7032addd64b341ba2c1c70e3e6da9aa216"),  # pragma: allowlist secret - public weight digest
    ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897,
     "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"),  # pragma: allowlist secret - public weight digest
    ("split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", 630997322,
     "f0017a43ea57ef6b3d4866be607844bbd8cada6d30966f7d70044ed0d63d3f9e"),  # pragma: allowlist secret - public weight digest
    ("split_files/vae/wan_2.1_vae.safetensors", 253815318,
     "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),  # pragma: allowlist secret - public weight digest
),
    reference_runtime=ReferenceRuntime("2.8.0+cu128", 32768, (12, 0), "sm_120",
                                       64_000_000_000, 32 * 1024**3, 30 * 1024**3),
)

S2V_L40S_WORKER = S2V_WORKER._replace(
    profile_id="m5-wan22-s2v14b-l40s-v1", gpu="NVIDIA L40S",
    reference_runtime=ReferenceRuntime("2.8.0+cu128", 49152, (8, 9), "sm_89",
                                       64_000_000_000, 32 * 1024**3, 30 * 1024**3),
)
S2V_PROFILE_IDS = (S2V_WORKER.profile_id, S2V_L40S_WORKER.profile_id)


def s2v_worker(profile_id: str = S2V_WORKER.profile_id) -> WorkerProfile:
    if profile_id == S2V_WORKER.profile_id:
        return S2V_WORKER
    if profile_id == S2V_L40S_WORKER.profile_id:
        return S2V_L40S_WORKER
    raise ValueError("unsupported closed M5 worker profile")


def require_s2v_hardware(profile_id: str, name: str, memory: int,
                         capability: tuple[int, int]) -> None:
    selected = s2v_worker(profile_id)
    expected = selected.reference_runtime
    if (expected is None or name != selected.gpu or capability != expected.capability
            or any(type(x) is not int for x in capability) or type(memory) is not int
            or not selected.minimum_memory_mib <= memory <= expected.maximum_memory_mib):
        raise ValueError("hardware differs from selected M5 profile")

# END_REVIEWED_WORKER_PROFILES
