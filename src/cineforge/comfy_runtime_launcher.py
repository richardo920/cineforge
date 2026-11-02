"""Fixed no-site M5 Comfy process entrypoint; never selects code from creative JSON."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from cineforge.comfy_runtime_plan import COMFY_ABI, COMFY_EXECUTION, comfy_plan
from cineforge.native_remote_bootstrap import nr_check, nr_read, nr_write
from cineforge.native_remote_plan import nr_digest, nr_encode, nr_json

_BOOT = ("import os,sys;sys.path.insert(0,sys.argv[1]);"
         "from cineforge.comfy_runtime_launcher import main;os._exit(main())")


def launch_command(plan: dict[str, Any], root: Path, native: Path, mode: str,
                   deadline: float, *, check_deadline: bool = True) -> tuple[list[str], dict[str, str]]:
    if mode not in ("probe", "server"):
        raise ValueError("unsupported fixed Comfy launch mode")
    if check_deadline:
        nr_check(deadline)
    environment = {"PATH": "", "HOME": str(root / "comfy-user"), "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0", "CUDA_CACHE_DISABLE": "1",
        "PYTORCH_JIT": "0",
        "TORCH_COMPILE_DISABLE": "1", "TRITON_CACHE_DIR": str(root / "comfy-temp"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "comfy-temp"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [plan["interpreter"]["path"], "-I", "-B", "-S", "-c", _BOOT,
               str(native / "firstparty"), str(root), mode, str(deadline)]
    return command, environment


def probe_result(value: dict[str, Any], root: Path, plan: dict[str, Any],
                 deadline: float) -> tuple[dict[str, Any], dict[str, Any]]:
    from cineforge.worker_bootstrap import resource_evidence, validate_s2v_runtime
    if (set(value) != {"runtime", "imports", "codecs", "abi", "native_files", "execution_sha256",
                      "jit_enabled_before", "jit_enabled_after"} or value["abi"] != COMFY_ABI
            or value["execution_sha256"] != nr_digest(plan["execution"])
            or value["jit_enabled_before"] is not False or value["jit_enabled_after"] is not False):
        raise ValueError("incomplete closed Comfy native probe")
    expected = {"torch", "torchvision", "torchaudio", "numpy", "transformers", "PIL", "av"}
    imports = value["imports"]
    if type(imports) is not dict or set(imports) != expected:
        raise ValueError("closed Comfy import provenance missing")
    prefix = str(root / "comfy-runtime/site") + "/"
    files = {str(root / "comfy-runtime" / p): sha for p, _, sha in plan["installed"]}
    if any(type(path) is not str or not path.startswith(prefix) or path not in files
           for path in imports.values()):
        raise ValueError("closed Comfy imported an ambient package")
    if value["codecs"] != {"png": True, "pcm16": True, "vp9": True}:
        raise ValueError("closed Comfy codec smoke unavailable")
    allowed = {p["path"]: p["sha256"] for p in [plan["interpreter"], *plan["native"]]}
    allowed.update(files)
    native = value["native_files"]
    if (type(native) is not dict or not native or len(native) > 1024
            or any(path not in allowed or sha != allowed[path] for path, sha in native.items())):
        raise ValueError("closed Comfy loaded unknown native libraries")
    runtime = value["runtime"]
    if type(runtime) is not dict:
        raise ValueError("closed Comfy runtime probe missing")
    runtime.update(resource_evidence(root))
    validate_s2v_runtime(runtime)
    if runtime["torchvision_version"] != "0.23.0+cu128":
        raise ValueError("closed Comfy torchvision differs")
    nr_check(deadline)
    return runtime, value


def _observe(root: Path, deadline: float) -> dict[str, Any]:
    import hashlib
    import importlib
    import io
    import platform
    import wave

    _require_abi()
    torch = importlib.import_module("torch")
    _require_jit_disabled(torch)
    av = importlib.import_module("av")
    numpy = importlib.import_module("numpy")
    PIL = importlib.import_module("PIL")
    Image = importlib.import_module("PIL.Image")
    torchaudio = importlib.import_module("torchaudio")
    torchvision = importlib.import_module("torchvision")
    transformers = importlib.import_module("transformers")
    _require_jit_disabled(torch)

    from cineforge.worker_bootstrap import bounded_process
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11)
            or sys.platform != "linux" or platform.machine() != "x86_64"
            or platform.libc_ver()[0] != "glibc"
            or tuple(int(p) for p in platform.libc_ver()[1].split(".")[:2]) < (2, 28)
            or torch.__version__ != "2.8.0+cu128" or torchvision.__version__ != "0.23.0+cu128"
            or torchaudio.__version__ != "2.8.0+cu128" or numpy.__version__ != "2.2.6"
            or transformers.__version__ != "4.57.3" or torch.version.cuda != "12.8"
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.device_count() != 1):
        raise ValueError("closed Comfy fixed GPU stack unavailable")
    props = torch.cuda.get_device_properties(0)
    if (props.name != "NVIDIA GeForce RTX 5090" or not 30720 <= props.total_memory // 1048576 <= 32768
            or tuple(torch.cuda.get_device_capability()) != (12, 0)):
        raise ValueError("closed Comfy observed GPU differs")
    x = torch.ones((1, 1, 2, 16), device="cuda", dtype=torch.bfloat16)
    y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
    nvml = bounded_process(["/usr/bin/nvidia-smi", "--id=0",
        "--query-gpu=memory.total,memory.reserved,driver_version", "--format=csv,noheader,nounits"],
        timeout=min(5, deadline-time.monotonic()), absolute_deadline=deadline,
        environment={"PATH": "", "LANG": "C"}).decode("ascii").strip().split(",")
    if len(nvml) != 3:
        raise ValueError("closed Comfy NVML evidence unavailable")
    runtime = dict(python_version=platform.python_version(), torch_version=torch.__version__,
        torchvision_version=torchvision.__version__, cuda_version=torch.version.cuda,
        gpu_raw_name=props.name, gpu_memory_mib=props.total_memory // 1048576,
        cuda_available=torch.cuda.is_available(), arch_list=torch.cuda.get_arch_list(),
        capability=list(torch.cuda.get_device_capability()), device_count=torch.cuda.device_count(),
        cuda_kernel_ok=bool(torch.isfinite(y).all().item()), compute_precision="bf16",
        nvml_total_mib=int(nvml[0]), nvml_reserved_mib=int(nvml[1]), driver_version=nvml[2].strip())
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    png.seek(0)
    image = Image.open(png)
    image.load()
    pcm = io.BytesIO()
    with wave.open(pcm, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 2400)
    pcm.seek(0)
    with av.open(pcm, format="wav") as audio:
        pcm_ok = sum(frame.samples for frame in audio.decode(audio=0)) == 2400
    video = io.BytesIO()
    with av.open(video, "w", format="webm") as container:
        stream = container.add_stream("libvpx-vp9", rate=16)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(numpy.zeros((16, 16, 3), dtype=numpy.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video.seek(0)
    with av.open(video, format="webm") as decoded:
        vp9_ok = len(list(decoded.decode(video=0))) == 1
    native = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split(maxsplit=5)[-1]
        if path.startswith("/") and not path.startswith("/dev/"):
            if path in native:
                continue
            file = Path(path)
            if not file.is_file() or file.is_symlink():
                raise ValueError("closed Comfy native mapping unavailable")
            sha = hashlib.sha256()
            with file.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    nr_check(deadline)
                    sha.update(block)
            native[path] = sha.hexdigest()
            nr_check(deadline)
    return {"runtime": runtime, "imports": {m.__name__: str(Path(m.__file__).resolve())
        for m in (torch, torchvision, torchaudio, numpy, transformers, PIL, av)},
        "codecs": {"png": image.mode == "RGB" and image.size == (8, 8), "pcm16": pcm_ok, "vp9": vp9_ok},
        "abi": COMFY_ABI, "native_files": native, "execution_sha256": nr_digest(COMFY_EXECUTION),
        "jit_enabled_before": False, "jit_enabled_after": False}


def _require_jit_disabled(torch: Any) -> None:
    if os.environ.get("PYTORCH_JIT") != "0" or bool(torch.jit._state._enabled):
        raise ValueError("fixed Comfy JIT-disabled import policy unavailable")


def _require_abi() -> None:
    import platform
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11)
            or sys.platform != "linux" or platform.machine() != "x86_64"
            or platform.libc_ver()[0] != "glibc"
            or tuple(int(v) for v in platform.libc_ver()[1].split(".")[:2]) < (2, 28)):
        raise ValueError("closed Comfy actual interpreter ABI unsupported")


def _startup_observer(root: Path, deadline: float) -> None:
    # Observe actual state after core imports and loopback readiness. This is a
    # narrow startup measurement, not a Torch version fix or hostile-code sandbox.
    import threading
    import urllib.request
    def observe() -> None:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                _require_jit_disabled(sys.modules["torch"])
                try:
                    with opener.open("http://127.0.0.1:8188/system_stats", timeout=min(1, deadline-time.monotonic())) as reply:
                        if reply.status == 200:
                            _require_jit_disabled(sys.modules["torch"])
                            nr_write(root / "comfy-jit-startup.json", nr_encode({
                                "execution_sha256": nr_digest(COMFY_EXECUTION),
                                "jit_enabled": False, "phase": "after_core_startup"}))
                            return
                except OSError:
                    pass
                time.sleep(.05)
        except BaseException:
            pass
        os._exit(2)
    threading.Thread(target=observe, daemon=True).start()


def main() -> int:
    # -I -B -S has already suppressed ambient .pth/site hooks and credentials.
    # Validate full staged files before adding any third-party import path.
    failed = False
    try:
        root, mode, deadline = Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
        if mode not in ("probe", "server"):
            raise ValueError
        _require_abi()
        plan = comfy_plan(nr_json(nr_read(root / "comfy-selection.json")))
        from cineforge.comfy_runtime_worker import verify_runtime, verify_source
        verify_runtime(plan, root, deadline)
        verify_source(plan, root, deadline)
        sys.path.insert(0, str(root / "comfy-runtime/site"))
        value = _observe(root, deadline)
        if mode == "probe":
            nr_check(deadline)
            sys.stdout.buffer.write(nr_encode(value))
            sys.stdout.buffer.flush()
            return 0
        # Production performs the same real import/kernel/codec proof, compares its
        # exact import/native identities with the bounded probe, then enters Comfy.
        previous = nr_json(nr_read(root / "comfy-launch-probe.json"))
        if any(value[key] != previous[key] for key in ("imports", "native_files", "codecs", "abi",
                                                      "execution_sha256", "jit_enabled_before", "jit_enabled_after")):
            raise ValueError
        sys.path.insert(0, str(root / "ComfyUI"))
        import runpy
        sys.argv = [str(root / "ComfyUI/main.py"), "--listen", "127.0.0.1", "--port", "8188",
            "--disable-all-custom-nodes", "--disable-api-nodes", "--disable-auto-launch",
            "--bf16-unet", "--disable-xformers", "--use-pytorch-cross-attention",
            "--disable-triton-backend", "--disable-comfy-compiler", "--disable-cuda-graphs",
            "--output-directory", str(root / "comfy-output"), "--user-directory", str(root / "comfy-user"),
            "--temp-directory", str(root / "comfy-temp")]
        sys.argv.extend(["--front-end-root", str(root / "comfy-runtime/site/comfyui_frontend_package/static")])
        nr_check(deadline)
        _startup_observer(root, deadline)
        runpy.run_path(sys.argv[0], run_name="__main__")
    except BaseException:
        failed = True
    return 2 if failed else 0
