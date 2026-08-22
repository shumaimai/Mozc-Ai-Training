from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch

from tools.train.lora_sft import patch_plamo_rocm_kernel_fallbacks


def run(length: int, native: bool, iterations: int) -> dict:
    if native:
        os.environ.pop("MOZC_DISABLE_HIP_SSD", None)
    else:
        os.environ["MOZC_DISABLE_HIP_SSD"] = "1"
    module = types.ModuleType(f"benchmark_{native}.modeling_plamo")
    module.ssd_chunk_scan_combined = lambda *args, **kwargs: None
    sys.modules[module.__name__] = module
    patch_plamo_rocm_kernel_fallbacks(module, use_hip_kernels=native)

    torch.manual_seed(53)
    shape = (1, length, 32, 128)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dt = torch.randn(1, length, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    a = (-torch.rand(32, device="cuda")).requires_grad_()
    b = torch.randn(1, length, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn_like(b, requires_grad=True)
    d = torch.randn(32, device="cuda", requires_grad=True)
    z = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(32, device="cuda", requires_grad=True)
    seq = torch.zeros(1, length, device="cuda", dtype=torch.int32)
    tensors = (x, dt, a, b, c, d, z, bias)

    timings = []
    peaks = []
    checksum = 0.0
    for iteration in range(iterations + 1):
        for tensor in tensors:
            tensor.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = module.ssd_chunk_scan_combined(
            x, dt, a, b, c, 256, d, z, bias, True, False, seq, None
        )
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        checksum = output.float().sum().item()
        if iteration:
            timings.append(elapsed)
            peaks.append(torch.cuda.max_memory_allocated() / 1024**2)
    return {
        "native": native,
        "length": length,
        "median_seconds": round(statistics.median(timings), 4),
        "peak_vram_mb": round(max(peaks), 1),
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    results = []
    for length in args.lengths:
        results.append(run(length, False, args.iterations))
        results.append(run(length, True, args.iterations))
    report = {"results": results}
    path = Path(__file__).resolve().parents[1] / "artifacts" / "plamo-ssd-benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
