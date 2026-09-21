"""Sarashina2 backbone pruning + IME page scorer.

This intentionally loads AutoModel rather than AutoModelForCausalLM:
the LM head and autoregressive generation stack are not part of the scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from tools.sarashina_jev.util import select_even_layers


DEFAULT_MODEL = "sbintuitions/sarashina2.2-0.5b"


def _layers(backbone: nn.Module) -> nn.ModuleList:
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(
            f"expected Llama-like backbone.layers ModuleList, got {type(layers).__name__}"
        )
    return layers


class SarashinaJevScorer(nn.Module):
    """Score one Mozc candidate from a pruned Sarashina2 decoder backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.score_head = nn.Linear(hidden, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        *,
        keep_layers: int = 12,
        torch_dtype: torch.dtype | None = None,
    ) -> tuple["SarashinaJevScorer", list[int]]:
        kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        backbone = AutoModel.from_pretrained(model_name, **kwargs)
        original = _layers(backbone)
        kept = select_even_layers(len(original), keep_layers)
        backbone.layers = nn.ModuleList([original[i] for i in kept])
        backbone.config.num_hidden_layers = keep_layers
        backbone.config.use_cache = False
        model = cls(backbone)
        return model, kept

    def configure_trainable(
        self,
        *,
        last_n_layers: int = 4,
        train_embeddings: bool = False,
        train_final_norm: bool = True,
    ) -> None:
        """Freeze most of the backbone while keeping the decision head trainable.

        last_n_layers=0 means train all retained transformer layers.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.score_head.parameters():
            param.requires_grad = True

        layers = _layers(self.backbone)
        chosen = layers if last_n_layers == 0 else layers[-min(last_n_layers, len(layers)) :]
        for layer in chosen:
            for param in layer.parameters():
                param.requires_grad = True

        if train_embeddings:
            embed = self.backbone.get_input_embeddings()
            for param in embed.parameters():
                param.requires_grad = True

        if train_final_norm:
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                for param in norm.parameters():
                    param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        last_index = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_index = last_index.view(-1, 1, 1).expand(
            -1, 1, hidden.shape[-1]
        )
        pooled = hidden.gather(1, gather_index).squeeze(1)
        # The pruned backbone may run in bf16/fp16 while the newly-created
        # scalar head intentionally keeps fp32 parameters. Match the input
        # to the head outside autocast too (evaluation/inference).
        pooled = pooled.to(dtype=self.score_head.weight.dtype)
        return self.score_head(pooled).squeeze(-1)

    def score_pages(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score [batch, page, seq] in one forward over batch*page candidates."""
        if input_ids.ndim != 3:
            raise ValueError("input_ids must be [batch, page, seq]")
        batch, page, seq = input_ids.shape
        flat_scores = self(
            input_ids.reshape(batch * page, seq),
            attention_mask.reshape(batch * page, seq),
        )
        scores = flat_scores.reshape(batch, page)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -1e4)
        return scores

    def parameter_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(
            p.numel() for p in self.backbone.get_input_embeddings().parameters()
        )
        return {"total": total, "trainable": trainable, "embedding": embedding}

    def save_artifact(
        self,
        out_dir: str | Path,
        tokenizer,
        *,
        source_model: str,
        kept_layer_indices: list[int],
        page_size: int = 5,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        out = Path(out_dir)
        (out / "backbone").mkdir(parents=True, exist_ok=True)
        (out / "tokenizer").mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "backbone", safe_serialization=True)
        tokenizer.save_pretrained(out / "tokenizer")
        torch.save(self.score_head.state_dict(), out / "score_head.pt")
        config = {
                    "source_model": source_model,
                    "kept_layer_indices": kept_layer_indices,
                    "page_size": page_size,
                    "pooling": "last_non_padding_token",
                    "input_format": "BOS + candidate + reading + context_tail + decision_marker",
                    "lm_head": False,
                    "autoregressive_generation": False,
                }
        if extra_meta:
            config.update(extra_meta)
        (out / "jev_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_artifact(
        cls,
        out_dir: str | Path,
        *,
        torch_dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
    ) -> tuple["SarashinaJevScorer", Any, dict[str, Any]]:
        out = Path(out_dir)
        kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        backbone = AutoModel.from_pretrained(out / "backbone", **kwargs)
        model = cls(backbone)
        state = torch.load(
            out / "score_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.score_head.load_state_dict(state)
        tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer")
        meta = json.loads((out / "jev_config.json").read_text(encoding="utf-8"))
        return model, tokenizer, meta
