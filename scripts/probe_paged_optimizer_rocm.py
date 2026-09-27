from __future__ import annotations

import json
import io
import os
import sys
from pathlib import Path

import torch

os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"

import bitsandbytes as bnb  # noqa: E402


def main() -> None:
    torch.manual_seed(71)
    initial = torch.randn(512, 512, device="cuda")

    def create():
        parameter = torch.nn.Parameter(initial.clone())
        return parameter, bnb.optim.PagedAdamW8bit([parameter], lr=1e-3)

    def step(parameter, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.float().square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return loss.item()

    baseline_parameter, baseline_optimizer = create()
    baseline_losses = [step(baseline_parameter, baseline_optimizer) for _ in range(3)]

    resumed_parameter, resumed_optimizer = create()
    resumed_losses = [step(resumed_parameter, resumed_optimizer) for _ in range(2)]
    checkpoint = io.BytesIO()
    torch.save(
        {"parameter": resumed_parameter.detach(), "optimizer": resumed_optimizer.state_dict()},
        checkpoint,
    )
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cuda", weights_only=False)
    restored_parameter, restored_optimizer = create()
    restored_parameter.data.copy_(restored["parameter"])
    restored_optimizer.load_state_dict(restored["optimizer"])
    resumed_losses.append(step(restored_parameter, restored_optimizer))

    torch.testing.assert_close(restored_parameter, baseline_parameter, rtol=1e-6, atol=1e-7)
    if not torch.isfinite(restored_parameter).all() or resumed_losses[-1] >= resumed_losses[0]:
        raise RuntimeError(f"PagedAdamW8bit resume validation failed: {resumed_losses}")
    print(json.dumps({
        "passed": True,
        "baseline_losses": baseline_losses,
        "resumed_losses": resumed_losses,
        "checkpoint_resume": True,
    }, indent=2))


if __name__ == "__main__":
    main()
