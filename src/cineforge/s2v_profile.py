"""Reviewed single-chunk core S2V recipe data; no runtime, rights or spending authority."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from cineforge.conditioning_profile import IMAGE as IMAGE
from cineforge.conditioning_profile import WORKFLOW_LICENSE as WORKFLOW_LICENSE
from cineforge.devices import Backend, ModelSpec, Precision
from cineforge.worker_profiles import S2V_WORKER
from cineforge.workflows import canonical, strict_object

PROFILE_ID = "m5-wan22-s2v14b-5090-v1"
MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
MODEL_REVISION = "ee6f4a40737a995bf5818954cfce6d59443b0f04"  # pragma: allowlist secret - public model revision
RUNNER_REVISION = "6d352802d8f8b9133f664a1757d6c6d26390c335"  # pragma: allowlist secret - public source revision
SOURCE_REVISION = "c067929d7e9cc0f3fd15be348fa1a1d85da15946"  # pragma: allowlist secret - public source revision
SOURCE_PATH = "templates/video_wan2_2_14B_s2v.json"
SOURCE_REPOSITORY = "https://github.com/Comfy-Org/workflow_templates"
SOURCE_SHA256 = "9b017cde7f0564968cc7d2d2d3d044741a991e6662674b3913d5d77f92e9af71"  # pragma: allowlist secret - public source digest
CANDIDATE_SHA256 = "2c1f389725e55eea0a9a3525eaffcc6723a31b0eec336fb62a70b14b881ac69b"  # pragma: allowlist secret - public source digest
MODEL_CARD_URI = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/raw/ee6f4a40737a995bf5818954cfce6d59443b0f04/README.md"
MODEL_CARD_SHA256 = "c5f31703b82a568ca7aaa49482d1698e74f665e1cf92b47148ad22c01305084d"  # pragma: allowlist secret - public source digest
MODEL_TERMS_URI = "https://raw.githubusercontent.com/Wan-Video/Wan2.2/1ea34ff48f87168174e12956e200b1d908b1c5ff/LICENSE.txt"
MODEL_TERMS_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"  # pragma: allowlist secret - public terms digest
GPU_ID = "NVIDIA GeForce RTX 5090"
MEMORY_MIB = 30720  # Conservative admission floor, not measured model fit.
MIN_RAM_PER_GPU_GB = 64
DURATION = 77 / 16
WIDTH, HEIGHT, FRAMES, FPS = 832, 480, 77, 16
STORAGE_PRECISION = "fp8-scaled diffusion/text; fp16 audio encoder; original VAE"
COMPUTE_PRECISION = "bf16"
FILES = S2V_WORKER.files

_CANDIDATE = '{\n  "147": {\n    "class_type": "CLIPLoader",\n    "inputs": {\n      "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",\n      "device": "cpu",\n      "type": "wan"\n    }\n  },\n  "148": {\n    "class_type": "VAEDecode",\n    "inputs": {\n      "samples": [\n        "152",\n        0\n      ],\n      "vae": [\n        "150",\n        0\n      ]\n    }\n  },\n  "149": {\n    "class_type": "CLIPTextEncode",\n    "inputs": {\n      "clip": [\n        "147",\n        0\n      ],\n      "text": "blurred face, covered mouth, profile, duplicate actor, subtitles"\n    }\n  },\n  "150": {\n    "class_type": "VAELoader",\n    "inputs": {\n      "vae_name": "wan_2.1_vae.safetensors"\n    }\n  },\n  "151": {\n    "class_type": "AudioEncoderLoader",\n    "inputs": {\n      "audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"\n    }\n  },\n  "152": {\n    "class_type": "KSampler",\n    "inputs": {\n      "cfg": 6,\n      "denoise": 1.0,\n      "latent_image": [\n        "153",\n        2\n      ],\n      "model": [\n        "154",\n        0\n      ],\n      "negative": [\n        "153",\n        1\n      ],\n      "positive": [\n        "153",\n        0\n      ],\n      "sampler_name": "uni_pc",\n      "scheduler": "simple",\n      "seed": 5201,\n      "steps": 20\n    }\n  },\n  "153": {\n    "class_type": "WanSoundImageToVideo",\n    "inputs": {\n      "audio_encoder_output": [\n        "173",\n        0\n      ],\n      "batch_size": 1,\n      "height": 480,\n      "length": 77,\n      "negative": [\n        "149",\n        0\n      ],\n      "positive": [\n        "159",\n        0\n      ],\n      "ref_image": [\n        "160",\n        0\n      ],\n      "vae": [\n        "150",\n        0\n      ],\n      "width": 832\n    }\n  },\n  "154": {\n    "class_type": "ModelSamplingSD3",\n    "inputs": {\n      "model": [\n        "161",\n        0\n      ],\n      "shift": 8\n    }\n  },\n  "159": {\n    "class_type": "CLIPTextEncode",\n    "inputs": {\n      "clip": [\n        "147",\n        0\n      ],\n      "text": "An original adult fictional actor faces the camera and speaks calmly. The mouth remains visible. Minimal head movement."\n    }\n  },\n  "160": {\n    "class_type": "LoadImage",\n    "inputs": {\n      "image": "m52_actor_pending_approval.png"\n    }\n  },\n  "161": {\n    "class_type": "UNETLoader",\n    "inputs": {\n      "unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors",\n      "weight_dtype": "default"\n    }\n  },\n  "162": {\n    "class_type": "LoadAudio",\n    "inputs": {\n      "audio": "m52_speech_pending_approval.wav"\n    }\n  },\n  "168": {\n    "class_type": "SaveWEBM",\n    "inputs": {\n      "codec": "vp9",\n      "crf": 18.0,\n      "filename_prefix": "m52_5bf322ca_candidate",\n      "fps": 16.0,\n      "images": [\n        "148",\n        0\n      ]\n    }\n  },\n  "173": {\n    "class_type": "AudioEncoderEncode",\n    "inputs": {\n      "audio": [\n        "162",\n        0\n      ],\n      "audio_encoder": [\n        "151",\n        0\n      ]\n    }\n  }\n}\n'


def model() -> ModelSpec:
    return ModelSpec(MODEL_ID, MODEL_REVISION, "speech-to-video", frozenset({Backend.CUDA}),
                     MEMORY_MIB, "Apache-2.0", Precision.BF16, True)


def model_files() -> list[dict[str, Any]]:
    return [{"path": path, "size_bytes": size, "sha256": sha, "revision": MODEL_REVISION,
             "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/"
                    + MODEL_REVISION + "/" + path,
             "license": "Apache-2.0 (repository declaration)"} for path, size, sha in FILES]


def graph(actor_sha256: str, speech_sha256: str, language: str) -> dict[str, Any]:
    if (any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (actor_sha256, speech_sha256)) or language not in ("en", "zh")):
        raise ValueError("invalid checked S2V graph input")
    value = strict_object(_CANDIDATE)
    value["160"]["inputs"]["image"] = "actor-" + actor_sha256 + ".png"
    value["162"]["inputs"]["audio"] = "speech-" + speech_sha256 + ".wav"
    value["152"]["inputs"]["seed"] = {"en": 5201, "zh": 5202}[language]
    value["168"]["inputs"]["filename_prefix"] = "cineforge-s2v-" + language
    return value


def service_declarations() -> dict[str, Any]:
    """No workflow/unknown speech placeholder; exact installed service capability only."""
    return {"schema_version": 1, "profile_id": PROFILE_ID,
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "distribution": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "task": "speech-to-video",
            "allowed_backends": ["cuda"], "minimum_memory_mib": MEMORY_MIB,
            "required_precision": COMPUTE_PRECISION, "storage_precision": STORAGE_PRECISION,
            "license": "Apache-2.0", "commercial_use": True, "files": model_files()},
        "runner": {"revision": RUNNER_REVISION, "custom_nodes": []},
        "hardware": {"gpu": GPU_ID, "min_ram_per_gpu_gb": MIN_RAM_PER_GPU_GB},
        "component_notices": {"wan_code_weights_vae": "Apache-2.0; retain origin/repack notices",
            "text_encoder": "google/umt5-xxl; Apache-2.0; scaled repack",
            "audio_encoder": "jonatasgrosman/wav2vec2-large-xlsr-53-english; Apache-2.0 repack",
            "runtime": "ComfyUI GPL-3.0; separate runtime source retained",
            "workflow": "MIT; derived candidate; no demo media",
            "codec": "Existing PyAV/FFmpeg build-specific notices; no codec certification"}}


def recipe_sha256() -> str:
    return hashlib.sha256(canonical({"service": service_declarations(),
        "candidate_sha256": CANDIDATE_SHA256, "template_sha256": SOURCE_SHA256,
        "qwen_sdk_revision": "022e286b98fbec7e1e916cb940cdf532cd9f488e",  # pragma: allowlist secret - public source revision
        "qwen_base_revision": "fd4b254389122332181a7c3db7f27e918eec64e3",  # pragma: allowlist secret - public model revision
        "qwen_design_revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",  # pragma: allowlist secret - public model revision
        "native_speech": "separately reviewed installed-source and receipt verification",
        "video": {"width": WIDTH, "height": HEIGHT, "frames": FRAMES, "fps": FPS,
            "single_chunk": True, "reference_transform": "pinned core bilinear center",
            "audio_timing": "50Hz->30Hz->16fps; zero feature buckets beyond source"}})).hexdigest()
