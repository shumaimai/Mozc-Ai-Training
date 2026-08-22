from __future__ import annotations

import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch

from tools.train.lora_sft import patch_plamo_rocm_kernel_fallbacks


def measure(native: bool, length: int, iterations: int = 5) -> dict:
    if native:
        os.environ.pop("MOZC_DISABLE_HIP_CONV", None)
    else:
        os.environ["MOZC_DISABLE_HIP_CONV"] = "1"
    os.environ["MOZC_DISABLE_HIP_SSD"] = "1"
    module = types.ModuleType(f"conv_{native}.modeling_plamo")
    module.ssd_chunk_scan_combined = lambda *args, **kwargs: None
    sys.modules[module.__name__] = module
    patch_plamo_rocm_kernel_fallbacks(module, use_hip_kernels=native)
    torch.manual_seed(59)
    x = torch.randn(1, 4096, length, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4096, 1, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    seq = torch.zeros(1, length, device="cuda", dtype=torch.int32)
    timings = []
    peaks = []
    for iteration in range(iterations + 1):
        x.grad = None
        weight.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output, _ = module._causal_conv1d(None, weight, x, seq)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        if iteration:
            timings.append(time.perf_counter() - started)
            peaks.append(torch.cuda.max_memory_allocated() / 1024**2)
    return {
        "native": native,
        "length": length,
        "median_seconds": round(statistics.median(timings), 5),
        "peak_vram_mb": round(max(peaks), 1),
    }


def main() -> None:
    results = []
    for length in (64, 256, 512):
        results.append(measure(False, length))
        results.append(measure(True, length))
    report = {"results": results}
    path = Path(__file__).resolve().parents[1] / "artifacts" / "plamo-conv-benchmark.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
