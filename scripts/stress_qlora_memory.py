from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"

import bitsandbytes as bnb  # noqa: E402


class RepeatedQuantizedBlock(torch.nn.Module):
    def __init__(self, hidden: int, repeats: int) -> None:
        super().__init__()
        self.up = bnb.nn.Linear4bit(
            hidden, hidden * 4, bias=False, compute_dtype=torch.bfloat16,
            quant_type="nf4", quant_storage=torch.uint8,
        )
        self.down = bnb.nn.Linear4bit(
            hidden * 4, hidden, bias=False, compute_dtype=torch.bfloat16,
            quant_type="nf4", quant_storage=torch.uint8,
        )
        self.adapter_a = torch.nn.Linear(hidden, 16, bias=False, dtype=torch.bfloat16)
        self.adapter_b = torch.nn.Linear(16, hidden, bias=False, dtype=torch.bfloat16)
        self.repeats = repeats

    def block(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = torch.nn.functional.silu(self.up(inputs))
        hidden = self.down(hidden)
        adapter = self.adapter_b(self.adapter_a(inputs)) * 0.01
        return residual + hidden + adapter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for _ in range(self.repeats):
            inputs = checkpoint(self.block, inputs, use_reentrant=False)
        return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--activation-offload", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(61)
    model = RepeatedQuantizedBlock(args.hidden, args.repeats).cuda()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("adapter_"))
    optimizer = bnb.optim.PagedAdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    inputs = torch.randn(
        1, args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    context = (
        torch.autograd.graph.save_on_cpu(pin_memory=False)
        if args.activation_offload else contextlib.nullcontext()
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with context:
        output = model(inputs)
        loss = output.float().square().mean()
        loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    report = {
        "passed": bool(torch.isfinite(inputs.grad).all()),
        "hidden": args.hidden,
        "tokens": args.tokens,
        "repeats": args.repeats,
        "activation_offload": args.activation_offload,
        "seconds": round(time.perf_counter() - started, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "loss": loss.item(),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("QLoRA memory stress produced invalid gradients")


if __name__ == "__main__":
    main()
