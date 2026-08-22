from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as functional

from tools.train.lora_sft import patch_plamo_rocm_kernel_fallbacks


def reference_conv(x: torch.Tensor, weight: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(
        x.shape[0], x.shape[1], weight.shape[-1] - 1, device=x.device, dtype=x.dtype
    )
    outputs = []
    for index in range(x.shape[-1]):
        if index and seq_idx is not None:
            state = torch.where(
                (seq_idx[:, index - 1] != seq_idx[:, index])[:, None, None],
                torch.zeros_like(state),
                state,
            )
        window = torch.cat([state, x[:, :, index : index + 1]], dim=-1)
        state = window[:, :, 1:]
        value = (window * weight[:, 0, :][None]).sum(dim=-1)
        outputs.append(functional.silu(value).unsqueeze(-1))
    return torch.cat(outputs, dim=-1)


def reference_ssd(x, dt, A, B, C, D, z, dt_bias, seq_idx):
    state = torch.zeros(
        x.shape[0], x.shape[2], x.shape[3], B.shape[-1],
        device=x.device, dtype=torch.float32,
    )
    outputs = []
    for index in range(x.shape[1]):
        if index and seq_idx is not None:
            state = torch.where(
                (seq_idx[:, index - 1] != seq_idx[:, index])[:, None, None, None],
                torch.zeros_like(state), state,
            )
        delta = functional.softplus(dt[:, index].float() + dt_bias.float())
        decay = torch.exp(delta * A.float())
        state = (
            state * decay[..., None, None]
            + delta[..., None, None]
            * x[:, index].float()[..., None]
            * B[:, index].float()[..., None, :]
        )
        output = (state * C[:, index].float()[..., None, :]).sum(dim=-1)
        output += x[:, index].float() * D.float()[None, :, None]
        outputs.append((output * functional.silu(z[:, index].float())).to(x.dtype)[:, None])
    return torch.cat(outputs, dim=1)


def main() -> None:
    fake = types.ModuleType("test.modeling_plamo")
    fake.ssd_chunk_scan_combined = lambda *args, **kwargs: None
    sys.modules[fake.__name__] = fake
    patch_plamo_rocm_kernel_fallbacks(fake, use_hip_kernels=True)

    torch.manual_seed(47)
    device = "cuda"
    x_conv = torch.randn(2, 7, 11, device=device, requires_grad=True)
    weight = torch.randn(7, 1, 4, device=device, requires_grad=True)
    seq_idx = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
        device=device, dtype=torch.int32,
    )
    actual_conv, _ = fake._causal_conv1d(None, weight, x_conv, seq_idx)
    expected_conv = reference_conv(x_conv, weight, seq_idx)
    torch.testing.assert_close(actual_conv, expected_conv, rtol=1e-5, atol=1e-6)
    actual_conv.sum().backward(retain_graph=True)
    actual_conv_grads = (x_conv.grad.detach().clone(), weight.grad.detach().clone())
    x_conv.grad = None
    weight.grad = None
    expected_conv.sum().backward()
    torch.testing.assert_close(actual_conv_grads[0], x_conv.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_conv_grads[1], weight.grad, rtol=1e-5, atol=1e-6)

    shape = (2, 9, 3, 5)
    x = torch.randn(*shape, device=device, requires_grad=True)
    dt = torch.randn(2, 9, 3, device=device, requires_grad=True)
    A = (-torch.rand(3, device=device)).requires_grad_()
    B = torch.randn(2, 9, 3, 4, device=device, requires_grad=True)
    C = torch.randn(2, 9, 3, 4, device=device, requires_grad=True)
    D = torch.randn(3, device=device, requires_grad=True)
    z = torch.randn(*shape, device=device, requires_grad=True)
    dt_bias = torch.randn(3, device=device, requires_grad=True)
    ssd_seq = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1, 1]],
        device=device, dtype=torch.int32,
    )
    actual = fake.ssd_chunk_scan_combined(
        x, dt, A, B, C, 256, D, z, dt_bias, True, False, ssd_seq, None
    )
    expected = reference_ssd(x, dt, A, B, C, D, z, dt_bias, ssd_seq)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.sum().backward(retain_graph=True)
    gradient_tensors = (x, dt, A, B, C, D, z, dt_bias)
    actual_grads = [tensor.grad.detach().clone() for tensor in gradient_tensors]
    for tensor in gradient_tensors:
        tensor.grad = None
    expected.sum().backward()
    gradient_names = ("x", "dt", "A", "B", "C", "D", "z", "dt_bias")
    for name, actual_grad, tensor in zip(gradient_names, actual_grads, gradient_tensors):
        torch.testing.assert_close(
            actual_grad, tensor.grad, rtol=1e-5, atol=1e-6,
            msg=lambda message: f"{name} gradient mismatch: {message}",
        )

    # Exercise the compressed-state path without packed-sequence resets.
    for tensor in gradient_tensors:
        tensor.grad = None
    empty_seq = torch.empty(0, device=device, dtype=torch.int32)
    actual = fake.ssd_chunk_scan_combined(
        x, dt, A, B, C, 256, D, z, dt_bias, True, False, None, None
    )
    expected = reference_ssd(x, dt, A, B, C, D, z, dt_bias, None)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward(retain_graph=True)
    actual_grads = [tensor.grad.detach().clone() for tensor in gradient_tensors]
    for tensor in gradient_tensors:
        tensor.grad = None
    expected.sum().backward()
    for name, actual_grad, tensor in zip(gradient_names, actual_grads, gradient_tensors):
        torch.testing.assert_close(
            actual_grad, tensor.grad, rtol=5e-4, atol=5e-5,
            msg=lambda message: f"{name} compressed gradient mismatch: {message}",
        )
    print("PLaMo fallback forward/backward parity passed")


if __name__ == "__main__":
    main()
