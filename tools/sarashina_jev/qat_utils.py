"""Lightweight fake-quant modules for Sarashina-JEV QAT.

These wrappers intentionally keep the same parameter names as nn.Linear and
nn.Embedding so checkpoints can be reloaded by a normal Hugging Face Llama
model after training.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _blend_ste(x: torch.Tensor, dq: torch.Tensor, strength: float) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return x
    # Forward follows the quantize/dequantize value while gradients follow x.
    return x + (dq.to(dtype=x.dtype) - x).detach() * strength


def fake_quant_symmetric_rows(
    x: torch.Tensor,
    *,
    strength: float = 1.0,
    qmax: float = 127.0,
) -> torch.Tensor:
    """Signed INT8-like fake quantization independently along the last axis."""
    if strength <= 0.0 or x.numel() == 0:
        return x
    xf = x.float()
    max_abs = xf.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / qmax
    q = torch.round(xf / scale).clamp(-qmax, qmax)
    dq = q * scale
    return _blend_ste(x, dq, strength)


def fake_quant_affine_tensor(
    x: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Unsigned UINT8-like per-tensor activation fake quantization."""
    if strength <= 0.0 or x.numel() == 0:
        return x
    xf = x.float()
    xmin = xf.detach().amin()
    xmax = xf.detach().amax()
    span = (xmax - xmin).clamp_min(1e-8)
    scale = span / 255.0
    zero = torch.round(-xmin / scale).clamp(0.0, 255.0)
    q = torch.round(xf / scale + zero).clamp(0.0, 255.0)
    dq = (q - zero) * scale
    return _blend_ste(x, dq, strength)


class FakeQuantLinear(nn.Linear):
    """nn.Linear-compatible QAT wrapper with INT8-like weights/activations."""

    qat_strength: float

    @classmethod
    def from_linear(cls, module: nn.Linear) -> "FakeQuantLinear":
        out = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        out.weight = nn.Parameter(
            module.weight.detach().clone(),
            requires_grad=module.weight.requires_grad,
        )
        if module.bias is not None:
            out.bias = nn.Parameter(
                module.bias.detach().clone(),
                requires_grad=module.bias.requires_grad,
            )
        out.qat_strength = 1.0
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        xq = fake_quant_affine_tensor(input, strength=self.qat_strength)
        wq = fake_quant_symmetric_rows(self.weight, strength=self.qat_strength)
        out = F.linear(xq, wq, self.bias)
        return fake_quant_affine_tensor(out, strength=self.qat_strength)


class FakeQuantEmbedding(nn.Embedding):
    """Embedding wrapper that fake-quantizes gathered vectors.

    Quantizing gathered rows avoids scanning the full 102,400 x hidden embedding
    matrix on every training step while still injecting the error that the
    Gather INT8 path must tolerate.
    """

    qat_strength: float

    @classmethod
    def from_embedding(cls, module: nn.Embedding) -> "FakeQuantEmbedding":
        out = cls(
            module.num_embeddings,
            module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        out.weight = nn.Parameter(
            module.weight.detach().clone(),
            requires_grad=module.weight.requires_grad,
        )
        out.qat_strength = 1.0
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return fake_quant_symmetric_rows(out, strength=self.qat_strength)


def replace_modules_for_qat(module: nn.Module, *, quantize_embeddings: bool = True) -> int:
    """Recursively replace Linear/Embedding modules while preserving state keys."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, FakeQuantLinear) or isinstance(child, FakeQuantEmbedding):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear.from_linear(child))
            replaced += 1
            continue
        if quantize_embeddings and isinstance(child, nn.Embedding):
            setattr(module, name, FakeQuantEmbedding.from_embedding(child))
            replaced += 1
            continue
        replaced += replace_modules_for_qat(
            child,
            quantize_embeddings=quantize_embeddings,
        )
    return replaced


def set_qat_strength(module: nn.Module, strength: float) -> None:
    value = float(max(0.0, min(1.0, strength)))
    for child in module.modules():
        if isinstance(child, (FakeQuantLinear, FakeQuantEmbedding)):
            child.qat_strength = value


def qat_module_counts(module: nn.Module) -> dict[str, int]:
    return {
        "fake_quant_linear": sum(isinstance(x, FakeQuantLinear) for x in module.modules()),
        "fake_quant_embedding": sum(isinstance(x, FakeQuantEmbedding) for x in module.modules()),
    }
