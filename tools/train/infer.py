"""IME-style generation helpers for LoRA / base causal LMs."""

from __future__ import annotations

import re
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.dataset.export_train import build_ime_prompt
from tools.train.lora_sft import (
    _install_causal_conv1d_torch_stub,
    patch_plamo_rocm_kernel_fallbacks,
    patch_plamo_tied_weights_compat,
)


def load_model(
    base_model: str,
    adapter: str | None = None,
    device: str | None = None,
    *,
    trust_remote_code: bool | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if trust_remote_code is None:
        trust_remote_code = "plamo" in base_model.lower()
    if trust_remote_code:
        patch_plamo_tied_weights_compat()
        _install_causal_conv1d_torch_stub()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=False, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=trust_remote_code
    )
    if trust_remote_code:
        patch_plamo_rocm_kernel_fallbacks()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def parse_candidates(text: str, limit: int = 3) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[\n\r]+|,|、", text):
        cand = raw.strip().strip("・-•* ")
        if not cand or cand in seen:
            continue
        seen.add(cand)
        lines.append(cand)
        if len(lines) >= limit:
            break
    return lines


@torch.inference_mode()
def _greedy_decode(model, input_ids: torch.Tensor, *, max_new_tokens: int, eos_token_id: int | None) -> torch.Tensor:
    """Manual greedy decode to avoid PLaMo/transformers generate shape bugs."""
    out = input_ids
    for _ in range(max_new_tokens):
        outputs = model(input_ids=out)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # Expect [batch, seq, vocab]; squeeze accidental extra dims.
        while logits.dim() > 3:
            logits = logits.squeeze(1)
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1)
        if next_token.dim() == 0:
            next_token = next_token.unsqueeze(0)
        elif next_token.dim() > 1:
            next_token = next_token.view(next_token.shape[0], -1)[:, 0]
        out = torch.cat([out, next_token.unsqueeze(-1)], dim=-1)
        if eos_token_id is not None and int(next_token[0].item()) == int(eos_token_id):
            break
    return out


@torch.inference_mode()
def generate_candidates(
    model,
    tokenizer,
    *,
    reading: str,
    mozc_candidates: list[str] | None = None,
    context: list[str] | None = None,
    max_new_tokens: int = 48,
    device: str = "cpu",
) -> dict[str, Any]:
    prompt = build_ime_prompt(reading, mozc_candidates or [], context or [])
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    model_name = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
    use_manual = "plamo" in str(model_name).lower() or "plamo" in type(model).__name__.lower()
    if not use_manual:
        # PeftModel wraps base; inspect base config too.
        base = getattr(model, "base_model", None)
        base_name = getattr(getattr(base, "config", None), "_name_or_path", "") or ""
        use_manual = "plamo" in str(base_name).lower()
        if not use_manual:
            try:
                use_manual = "plamo" in type(model.get_base_model()).__name__.lower()
            except Exception:
                pass
    if use_manual:
        output = _greedy_decode(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    candidates = parse_candidates(completion, limit=3)
    return {
        "prompt": prompt,
        "raw": completion,
        "candidates": candidates,
    }
