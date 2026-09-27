from __future__ import annotations

import torch

from tools.train.plamo_ssd_hip import make_autograd_function


def closed_form(x, dt, a, b, c, d, z, bias):
    # Independent two-token, one-head, one-channel, one-state expansion.
    delta0 = torch.nn.functional.softplus(dt[0, 0, 0].float() + bias[0])
    state0 = delta0 * x[0, 0, 0, 0].float() * b[0, 0, 0, 0].float()
    raw0 = state0 * c[0, 0, 0, 0].float() + d[0] * x[0, 0, 0, 0].float()
    out0 = raw0 * torch.nn.functional.silu(z[0, 0, 0, 0].float())

    delta1 = torch.nn.functional.softplus(dt[0, 1, 0].float() + bias[0])
    state1 = (
        torch.exp(delta1 * a[0]) * state0
        + delta1 * x[0, 1, 0, 0].float() * b[0, 1, 0, 0].float()
    )
    raw1 = state1 * c[0, 1, 0, 0].float() + d[0] * x[0, 1, 0, 0].float()
    out1 = raw1 * torch.nn.functional.silu(z[0, 1, 0, 0].float())
    return torch.stack((out0, out1)).reshape(1, 2, 1, 1)


def check(dtype: torch.dtype) -> None:
    torch.manual_seed(67)
    values = {
        "x": torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True),
        "dt": torch.randn(1, 2, 1, device="cuda", dtype=dtype, requires_grad=True),
        "a": (-torch.rand(1, device="cuda")).requires_grad_(),
        "b": torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True),
        "c": torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True),
        "d": torch.randn(1, device="cuda", requires_grad=True),
        "z": torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True),
        "bias": torch.randn(1, device="cuda", requires_grad=True),
    }
    seq = torch.empty(0, device="cuda", dtype=torch.int32)
    hip = make_autograd_function()
    actual = hip(*values.values(), seq)
    expected = closed_form(**values)
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=8e-3)

    weights = torch.tensor([0.25, -0.75], device="cuda").reshape(1, 2, 1, 1)
    (actual.float() * weights).sum().backward(retain_graph=True)
    actual_grads = {name: tensor.grad.detach().float().clone() for name, tensor in values.items()}
    for tensor in values.values():
        tensor.grad = None
    (expected * weights).sum().backward()
    for name, tensor in values.items():
        torch.testing.assert_close(
            actual_grads[name], tensor.grad.float(), rtol=2e-2, atol=2e-2,
            msg=lambda message: f"{dtype} {name}: {message}",
        )


def main() -> None:
    check(torch.float16)
    check(torch.bfloat16)
    print("PLaMo independent FP16/BF16 oracle passed")


if __name__ == "__main__":
    main()
