"""One proposed core-node I2V profile; constants grant no runtime or spending authority."""

from __future__ import annotations

import hashlib
from typing import Any

from cineforge.devices import Backend, ModelSpec, Precision
from cineforge.workflows import canonical

PROFILE_ID = "m4-wan22-ti2v5b-5090-v1"
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
GPU_ID = "NVIDIA GeForce RTX 5090"
REGION = "EU-RO-1"
REGIONS = frozenset({"EU-RO-1", "EU-CZ-1"})
MIN_RAM_PER_GPU_GB = 64  # RESTv2 placement filter in GB, not a measured RAM/VRAM quantity.
# Conservative admission allowance, not a measured minimum for the model.
MEMORY_MIB = 30 * 1024
DURATION = 49 / 24
MODE = "start_image"
RESOURCE_POLICY = {"minimum_effective_ram_limit_bytes": 64_000_000_000,
                   "minimum_available_ram_bytes": 32 * 1024**3,
                   "minimum_filesystem_free_bytes": 30 * 1024**3,
                   "note": "Conservative pre-download planning allowances, not measured model minima; "
                           "64GB is decimal, available RAM/disk thresholds use GiB; no swap counted."}
MODEL_REVISION = 'ee6f4a40737a995bf5818954cfce6d59443b0f04'  # pragma: allowlist secret - public provenance pin
RUNNER_REVISION = '6d352802d8f8b9133f664a1757d6c6d26390c335'  # pragma: allowlist secret - public provenance pin
IMAGE = 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'  # pragma: allowlist secret - public provenance pin
SOURCE_REVISION = 'c067929d7e9cc0f3fd15be348fa1a1d85da15946'  # pragma: allowlist secret - public provenance pin
SOURCE_SHA256 = '23c93b88adecbdf5bcf4c5438117070fe757f30aca373dd6a2eb488d2c6a585a'  # pragma: allowlist secret - public provenance pin
SOURCE_REPOSITORY = "https://github.com/Comfy-Org/workflow_templates"
SOURCE_PATH = "templates/video_wan2_2_5B_ti2v.json"

MODEL_CARD_URI = 'https://huggingface.co/Comfy-Org/Wan-2.2/raw/ee6f4a40737a995bf5818954cfce6d59443b0f04/README.md'  # pragma: allowlist secret - public source/terms pin
MODEL_CARD_SHA256 = 'c5f31703b82a568ca7aaa49482d1698e74f665e1cf92b47148ad22c01305084d'  # pragma: allowlist secret - public source/terms pin
MODEL_TERMS_URI = 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/1ea34ff48f87168174e12956e200b1d908b1c5ff/LICENSE.txt'  # pragma: allowlist secret - public source/terms pin
MODEL_TERMS_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'  # pragma: allowlist secret - public source/terms pin
WORKFLOW_LICENSE = b'MIT License\n\nCopyright (c) 2023-present Comfy Org\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'
NEGATIVE = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
FILES = (
    ('split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors', 9999658848,
     '456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e'),  # pragma: allowlist secret - public weight digest
    ('split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', 6735906897,
     'c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68'),  # pragma: allowlist secret - public weight digest
    ('split_files/vae/wan2.2_vae.safetensors', 1409400960,
     'e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156'),  # pragma: allowlist secret - public weight digest
)

IDENTITY = ("An original teal square-headed robot with one white circular eye and an orange scarf, "
            "matching the input image.")
PROMPTS = (
    IDENTITY + " Medium view with a locked camera. The robot slowly waves one hand. "
    "Soft even light, simple pale background.",
    IDENTITY + " The camera gently pushes in. The robot slowly tilts its head. "
    "Soft even light, simple pale background.",
    IDENTITY + " The camera moves slightly sideways. The robot turns its head slightly toward "
    "the camera. Soft even light, simple pale background.",
)
SEEDS = (4201, 4202, 4203)

def model() -> ModelSpec:
    return ModelSpec(MODEL_ID, MODEL_REVISION, "image-to-video", frozenset({Backend.CUDA}),
                     MEMORY_MIB, "Apache-2.0", Precision.FP16, True)


def model_files() -> list[dict[str, Any]]:
    return [{"path": path, "size_bytes": size, "sha256": digest, "revision": MODEL_REVISION,
             "url": f"https://huggingface.co/Comfy-Org/Wan-2.2/resolve/{MODEL_REVISION}/{path}",
             "license": "Apache-2.0 (repository declaration)"} for path, size, digest in FILES]


def graph(reference_sha256: str) -> dict[str, Any]:
    if (len(reference_sha256) != 64
            or any(c not in "0123456789abcdef" for c in reference_sha256)):
        raise ValueError("invalid reference digest")
    return {
        "37": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan",
            "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": PROMPTS[0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": NEGATIVE}},
        "48": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["37", 0], "shift": 8}},
        "56": {"class_type": "LoadImage", "inputs": {"image": "ref-" + reference_sha256 + ".png"}},
        "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {
            "vae": ["39", 0], "width": 1280, "height": 704, "length": 49, "batch_size": 1,
            "start_image": ["56", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["48", 0], "seed": SEEDS[0],
            "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple",
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "denoise": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "47": {"class_type": "SaveWEBM", "inputs": {"images": ["8", 0],
            "filename_prefix": "cineforge-m4", "codec": "vp9", "fps": 24, "crf": 32}},
    }


def declarations(reference_sha256: str) -> dict[str, Any]:
    compiled = graph(reference_sha256)
    return {"schema_version": 1, "profile_id": PROFILE_ID,
            "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "distribution": "Comfy-Org/Wan-2.2", "task": "image-to-video",
                "allowed_backends": ["cuda"], "minimum_memory_mib": MEMORY_MIB,
                "required_precision": "fp16", "license": "Apache-2.0",
                "commercial_use": True, "files": model_files()},
            "runner": {"revision": RUNNER_REVISION, "custom_nodes": []},
            "workflow": {"canonical_sha256": hashlib.sha256(canonical(compiled)).hexdigest(),
                "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
                "source_path": SOURCE_PATH, "source_raw_sha256": SOURCE_SHA256,
                "license": "MIT", "license_sha256": hashlib.sha256(WORKFLOW_LICENSE).hexdigest(),
                "allowed_classes": sorted({n["class_type"] for n in compiled.values()}),
                "bindings": {"prompt": {"node_id": "6", "input_name": "text"},
                             "seed": {"node_id": "3", "input_name": "seed"}},
                "output_node": "47", "output_kind": "images", "output_index": 0,
                "artifact_license": "LicenseRef-Generated-Unreviewed",
                "media_type": "video/webm", "duration_seconds": DURATION},
            "hardware": {"gpu": GPU_ID, "vram_gib": 32, "min_ram_per_gpu_gb": MIN_RAM_PER_GPU_GB,
                         "resource_policy": dict(RESOURCE_POLICY)}}
