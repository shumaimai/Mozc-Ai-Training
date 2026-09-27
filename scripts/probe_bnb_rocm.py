from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"

import bitsandbytes as bnb  # noqa: E402


def main() -> None:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch GPU is required")

    torch.manual_seed(41)
    layer = bnb.nn.Linear4bit(
        128,
        32,
        bias=True,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
        quant_storage=torch.uint8,
    ).cuda()
    inputs = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = layer(inputs)
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    if inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise RuntimeError("bitsandbytes 4-bit backward produced invalid gradients")

    report = {
        "passed": True,
        "bitsandbytes": bnb.__version__,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "loss": loss.item(),
        "gradient_norm": inputs.grad.float().norm().item(),
    }
    output_path = Path(__file__).resolve().parents[1] / "artifacts" / "bnb-rocm-probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
