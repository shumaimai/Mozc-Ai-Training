from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "extensions" / "plamo_ssd.cpp",
    ROOT / "extensions" / "plamo_ssd.cu",
)
_loaded_info: dict[str, str] = {}


def _find_rocm_root() -> Path:
    configured = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    candidates = [Path(configured)] if configured else []
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec and spec.submodule_search_locations:
        candidates.extend(Path(path) for path in spec.submodule_search_locations)
    candidates.append(Path(sys.executable).parents[1] / "Lib" / "site-packages" / "_rocm_sdk_devel")
    for candidate in candidates:
        if (candidate / "lib" / "llvm" / "bin" / "amdclang-cl.exe").is_file():
            return candidate.resolve()
    raise RuntimeError("ROCm development SDK was not found in this Python environment")


def _gpu_arch() -> str:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("An available ROCm GPU is required")
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


def _build_identity(arch: str) -> str:
    digest = hashlib.sha256()
    for source in SOURCES:
        digest.update(source.read_bytes())
    digest.update(torch.__version__.encode())
    digest.update(str(torch.version.hip).encode())
    digest.update(arch.encode())
    return digest.hexdigest()[:16]


@contextlib.contextmanager
def _build_environment(rocm_root: Path, arch: str):
    llvm_bin = rocm_root / "lib" / "llvm" / "bin"
    updates = {
        "ROCM_HOME": str(rocm_root),
        "ROCM_PATH": str(rocm_root),
        "HIP_HOME": str(rocm_root),
        "PYTORCH_ROCM_ARCH": arch,
        "CC": str(llvm_bin / "amdclang-cl.exe"),
        "CXX": str(llvm_bin / "amdclang-cl.exe"),
        "PATH": f"{rocm_root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_extension():
    arch = _gpu_arch()
    rocm_root = _find_rocm_root()
    identity = _build_identity(arch)
    module_name = f"plamo_ssd_hip_{identity}"
    build_dir = ROOT / "artifacts" / "plamo-ssd-extension" / identity
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / f"{module_name}.pyd"

    if binary.is_file():
        spec = importlib.util.spec_from_file_location(module_name, binary)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load extension binary: {binary}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        with _build_environment(rocm_root, arch):
            from torch.utils.cpp_extension import load

            module = load(
                name=module_name,
                sources=[str(source) for source in SOURCES],
                build_directory=str(build_dir),
                extra_cuda_cflags=[
                    "-O3",
                    f"--rocm-device-lib-path={rocm_root / 'lib' / 'llvm' / 'amdgcn' / 'bitcode'}",
                ],
                verbose=False,
            )

    _loaded_info.update(
        backend="hip",
        architecture=arch,
        build_identity=identity,
        binary=str(binary),
    )
    return module


def extension_info() -> dict[str, str]:
    return dict(_loaded_info)


def make_autograd_function():
    extension = load_extension()

    class PlamoSsdHip(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dt, a, b, c, d, z, dt_bias, seq_idx):
            empty_seq = torch.empty(0, device=x.device, dtype=torch.int32)
            output, states = extension.forward(
                x.contiguous(), dt.contiguous(), a.float().contiguous(),
                b.contiguous(), c.contiguous(), d.float().contiguous(),
                z.contiguous(), dt_bias.float().contiguous(),
                seq_idx.contiguous() if seq_idx.numel() else empty_seq,
            )
            ctx.save_for_backward(x, dt, a, b, c, d, z, dt_bias, seq_idx, states)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x, dt, a, b, c, d, z, dt_bias, seq_idx, states = ctx.saved_tensors
            gradients = extension.backward(
                grad_output.contiguous(), states, x, dt, a.float().contiguous(),
                b, c, d.float().contiguous(), z, dt_bias.float().contiguous(), seq_idx,
            )
            return (*gradients, None)

    return PlamoSsdHip.apply


def make_conv_autograd_function():
    extension = load_extension()

    class PlamoConvHip(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, seq_idx):
            empty_seq = torch.empty(0, device=x.device, dtype=torch.int32)
            ctx.save_for_backward(x, weight, seq_idx)
            return extension.conv_forward(
                x.contiguous(), weight.contiguous(),
                seq_idx.contiguous() if seq_idx.numel() else empty_seq,
            )

        @staticmethod
        def backward(ctx, grad_output):
            x, weight, seq_idx = ctx.saved_tensors
            grad_x, grad_weight = extension.conv_backward(
                grad_output, x, weight, seq_idx, ctx.needs_input_grad[1]
            )
            return grad_x, grad_weight if ctx.needs_input_grad[1] else None, None

    return PlamoConvHip.apply
