"""Public immutable speech-source declarations; never installation or execution authority."""

SDK_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"  # pragma: allowlist secret - public source revision
DESIGN_REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"  # pragma: allowlist secret - public model revision

# (relative path, bytes, hash algorithm, published hash). Text SHA256 was rehashed.
# Weight hashes are upstream LFS declarations; local bodies must match before use.
SDK_FILES = (
    ('LICENSE', 11343, 'sha256', 'a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655'),  # pragma: allowlist secret - public source hash
    ('README.md', 60501, 'sha256', '51f4497e70e9ae024b10ee4bbcb470366f10592edeab56c58d2194ba2dadb745'),  # pragma: allowlist secret - public source hash
    ('pyproject.toml', 1099, 'sha256', '3e82fcb21b9d48bba789a7c0a2c2dcb88ad60c4d297d60ae4575ee3eb4e4d5aa'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/__init__.py', 839, 'sha256', 'ea52de59d070fde366467a6902d0edcfc1b0575b8c570a0c71020c41d6a593ed'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/__main__.py', 800, 'sha256', '7791bc1fe172af87902c52d83262aaf32f1029219a013db1df16dc6730501d97'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/cli/demo.py', 29160, 'sha256', 'ea28e0930753e05cbdd53ab7bd90b7be81ab859549902771db15f0b3abfcd5eb'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/__init__.py', 990, 'sha256', '1b380d9de843b6d585d938c339d066136567ca7125412674234204af4386679e'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/models/__init__.py', 807, 'sha256', '917d38d8fd7ed84de73703fd2394d535ca416e15694d1e50324e4a06d4022c82'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/models/configuration_qwen3_tts.py', 26428, 'sha256', 'f52867f14fde06a416dd14864d503ce6d13d0a08d5f5da30191e1c80c13f5d18'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/models/modeling_qwen3_tts.py', 100211, 'sha256', '25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/models/processing_qwen3_tts.py', 4022, 'sha256', '61472203188e46eda68d740c25f0f228ece24a61ecf942e57d6e49e78b4d6bcd'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py', 7946, 'sha256', '9e30c24394b00cb0366d7da3482b7436468acc1cd3da1a6fe614a1d34653a5e3'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py', 40519, 'sha256', '844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_25hz/configuration_qwen3_tts_tokenizer_v1.py', 14494, 'sha256', '28f94f715df7d96d365c9cea0021d9cca8c63d2f48f48423c43a3d3cae28ef02'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py', 56527, 'sha256', '8c6d5b7cfb133967f04286e069138ef7fe0cfca30e9acf3326106f6117135a7f'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_25hz/vq/assets/mel_filters.npz', 4271, 'sha256', '7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_25hz/vq/core_vq.py', 20074, 'sha256', 'd35fa9f00ef4863361493e102f4e28f092488497117df580819e93b5dff3b8de'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_25hz/vq/speech_vq.py', 14833, 'sha256', '7c68fcba8c508d8a18e2241b563302d1bf46f8ecb1515c59222263d8524bb298'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py', 14351, 'sha256', 'a1748b34968ba8effebf902c4a4399941b3e49bca3f6c0120fc673e3ba7d767d'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/inference/qwen3_tts_model.py', 37121, 'sha256', 'e1da450732857c1f5fe3e36ebab85db2f6dc6a48caaff6973f463384e30275e4'),  # pragma: allowlist secret - public source hash
    ('qwen_tts/inference/qwen3_tts_tokenizer.py', 15699, 'sha256', 'ac2d855022a1bd21d33ab7b267ec952eef71f81d3f8d969a306139ff1a929515'),  # pragma: allowlist secret - public source hash
)

BASE_FILES = (
    ('config.json', 4494, "sha256", 'b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9'),  # pragma: allowlist secret - public model hash
    ('generation_config.json', 245, "sha256", 'f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa'),  # pragma: allowlist secret - public model hash
    ('merges.txt', 1671839, "sha256", '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'),  # pragma: allowlist secret - public model hash
    ('model.safetensors', 3857413744, "sha256", '38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c'),  # pragma: allowlist secret - public model hash
    ('preprocessor_config.json', 127, "sha256", 'efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/config.json', 2336, "sha256", 'ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/configuration.json', 76, "sha256", '6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/model.safetensors', 682293092, "sha256", '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/preprocessor_config.json', 234, "sha256", 'fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb'),  # pragma: allowlist secret - public model hash
    ('tokenizer_config.json', 7344, "sha256", 'dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670'),  # pragma: allowlist secret - public model hash
    ('vocab.json', 2776833, "sha256", 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),  # pragma: allowlist secret - public model hash
)

DESIGN_FILES = (
    ('config.json', 4421, "sha256", 'aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0'),  # pragma: allowlist secret - public model hash
    ('generation_config.json', 245, "sha256", 'f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa'),  # pragma: allowlist secret - public model hash
    ('merges.txt', 1671839, "sha256", '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'),  # pragma: allowlist secret - public model hash
    ('model.safetensors', 3833402552, "sha256", '391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522'),  # pragma: allowlist secret - public model hash
    ('preprocessor_config.json', 127, "sha256", 'efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/config.json', 2336, "sha256", 'ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/configuration.json', 76, "sha256", '6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/model.safetensors', 682293092, "sha256", '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258'),  # pragma: allowlist secret - public model hash
    ('speech_tokenizer/preprocessor_config.json', 234, "sha256", 'fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb'),  # pragma: allowlist secret - public model hash
    ('tokenizer_config.json', 7344, "sha256", 'dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670'),  # pragma: allowlist secret - public model hash
    ('vocab.json', 2776833, "sha256", 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),  # pragma: allowlist secret - public model hash
)

# Exact distribution closure plus separately audited source-only sox. No resolver runs.
RUNTIME_DISTRIBUTIONS = (
    ('accelerate', '1.12.0'),
    ('aiofiles', '24.1.0'),
    ('annotated-doc', '0.0.5'),
    ('annotated-types', '0.8.0'),
    ('anyio', '4.16.0'),
    ('audioread', '3.1.0'),
    ('brotli', '1.2.1'),
    ('certifi', '2026.10.23'),
    ('cffi', '2.1.1'),
    ('charset-normalizer', '3.5.2'),
    ('click', '8.5.0'),
    ('cloudpickle', '3.1.2'),
    ('coloredlogs', '15.0.1'),
    ('decorator', '5.3.1'),
    ('einops', '0.8.1'),
    ('fastapi', '0.162.0'),
    ('ffmpy', '1.0.0'),
    ('filelock', '4.2.6'),
    ('flatbuffers', '25.12.19'),
    ('fsspec', '2026.9.0'),
    ('gradio', '5.49.1'),
    ('gradio-client', '1.13.3'),
    ('groovy', '0.1.2'),
    ('h11', '0.16.0'),
    ('hf-xet', '1.8.0'),
    ('httpcore', '1.0.9'),
    ('httpx', '0.28.1'),
    ('huggingface-hub', '0.36.2'),
    ('humanfriendly', '10.0'),
    ('idna', '3.20'),
    ('jinja2', '3.1.6'),
    ('joblib', '1.6.0'),
    ('lazy-loader', '0.6'),
    ('librosa', '0.11.0'),
    ('llvmlite', '0.50.0'),
    ('markdown-it-py', '4.2.0'),
    ('markupsafe', '3.0.4'),
    ('mdurl', '0.1.2'),
    ('mpmath', '1.3.0'),
    ('msgpack', '1.2.4'),
    ('narwhals', '2.28.0'),
    ('networkx', '3.6.1'),
    ('numba', '0.68.1'),
    ('numpy', '2.2.6'),
    ('nvidia-cublas-cu12', '12.8.4.1'),
    ('nvidia-cuda-cupti-cu12', '12.8.90'),
    ('nvidia-cuda-nvrtc-cu12', '12.8.93'),
    ('nvidia-cuda-runtime-cu12', '12.8.90'),
    ('nvidia-cudnn-cu12', '9.10.2.21'),
    ('nvidia-cufft-cu12', '11.3.3.83'),
    ('nvidia-cufile-cu12', '1.13.1.3'),
    ('nvidia-curand-cu12', '10.3.9.90'),
    ('nvidia-cusolver-cu12', '11.7.3.90'),
    ('nvidia-cusparse-cu12', '12.5.8.93'),
    ('nvidia-cusparselt-cu12', '0.7.1'),
    ('nvidia-nccl-cu12', '2.27.3'),
    ('nvidia-nvjitlink-cu12', '12.8.93'),
    ('nvidia-nvtx-cu12', '12.8.90'),
    ('onnxruntime', '1.23.2'),
    ('opentelemetry-api', '1.45.2'),
    ('orjson', '3.13.0'),
    ('packaging', '26.3'),
    ('pandas', '2.3.3'),
    ('pillow', '11.3.0'),
    ('platformdirs', '4.14.3'),
    ('pooch', '1.9.0'),
    ('protobuf', '7.36.2'),
    ('psutil', '7.2.2'),
    ('pycparser', '3.12'),
    ('pydantic', '2.11.10'),
    ('pydantic-core', '2.33.2'),
    ('pydub', '0.25.1'),
    ('pygments', '2.21.0'),
    ('python-dateutil', '4.0.0'),
    ('python-multipart', '0.0.33'),
    ('pytz', '2026.6'),
    ('pyyaml', '6.0.3'),
    ('regex', '2026.9.29'),
    ('requests', '2.34.2'),
    ('rich', '15.0.0'),
    ('ruff', '0.17.3'),
    ('safehttpx', '0.1.7'),
    ('safetensors', '0.8.0'),
    ('scikit-learn', '1.9.1'),
    ('scipy', '1.17.1'),
    ('semantic-version', '2.10.0'),
    ('setuptools', '84.2.1'),
    ('shellingham', '1.5.4'),
    ('soundfile', '0.13.1'),
    ('soxr', '1.1.0'),
    ('starlette', '0.52.1'),
    ('sympy', '1.14.0'),
    ('threadpoolctl', '3.7.0'),
    ('tokenizers', '0.22.2'),
    ('tomlkit', '0.13.3'),
    ('torch', '2.8.0'),
    ('torchaudio', '2.8.0'),
    ('tqdm', '4.70.1'),
    ('transformers', '4.57.3'),
    ('triton', '3.4.0'),
    ('typer', '0.27.3'),
    ('typing-extensions', '4.16.0'),
    ('typing-inspection', '0.4.4'),
    ('tzdata', '2026.6'),
    ('urllib3', '2.8.0'),
    ('uvicorn', '0.54.0'),
    ('websockets', '15.0.1'),
    ("sox", "1.5.0"),
)

# Pinned dependency advisories require a reviewed reachability disposition.
# No web/demo, training, checkpoint-conversion or arbitrary model/config paths are allowed.
ADVISORY_PACKAGES = ("accelerate", "gradio", "pillow", "starlette", "torch", "transformers")
SOX_SOURCE_SHA256 = "12c7be5bb1f548d891fe11e82c08cf5f1a1d74e225298f60082e5aeb2469ada0"  # pragma: allowlist secret - public source hash
SOX_SOURCE_BYTES = 63905
PCM_POLICY = "pcm16-rint-even-32768-reject-outside-unit-v1"
NETWORK_POLICY = "linux-seccomp-no-new-privs-unix-only-v1"
SDK_NPZ_GIT_BLOB = "28ea26909dbdfd608aef67afc4d74d7961ae4bb6"  # pragma: allowlist secret - public source identity

# Source and model notices are retained in private staging, not vendored SDK code.
SDK_SOURCE = "https://github.com/QwenLM/Qwen3-TTS"
SDK_LICENSE = "Apache-2.0"
MODEL_NOTICES = (
    ('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', '5ecdb67327fd37bb2e042aab12ff7391903235d3', "Apache-2.0", 3214, 'acfaf6c0d433866cb5e2bb73e7915ea3767f5a2e3abca413b485635a2c72b5e6'),  # pragma: allowlist secret - public card identity
    ('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'fd4b254389122332181a7c3db7f27e918eec64e3', "Apache-2.0", 57817, '621b7e88f1867bc03d817176fc1f7f55f4b5a70654b2a07fdcda1e169efb024b'),  # pragma: allowlist secret - public card identity
)
