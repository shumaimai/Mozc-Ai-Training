"""Export the frozen Phase 2 format_v2 cross-encoder checkpoint to ONNX fp32.

The exported graph is exactly the training forward pass:
    logits = Linear(encoder(input_ids, attention_mask).last_hidden_state[:, 0])

No tokenizer is baked in; tokenization parity is verified separately by
tools/rerank/token_parity_check.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="frozen cross_encoder.pt")
    parser.add_argument("--out", required=True, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seq-len", type=int, default=16, help="dummy seq length")
    args = parser.parse_args()

    import torch
    from torch import nn
    from transformers import AutoModel

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = blob["base_model"]
    if not blob.get("complete"):
        raise SystemExit("refusing to export an incomplete checkpoint")

    class CrossEncoder(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                name, trust_remote_code=True, torch_dtype=torch.float32
            )
            hidden = int(self.encoder.config.hidden_size)
            self.score = nn.Linear(hidden, 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            return self.score(cls).squeeze(-1)

    model = CrossEncoder(base)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()

    dummy_ids = torch.ones(2, args.seq_len, dtype=torch.int64)
    dummy_mask = torch.ones(2, args.seq_len, dtype=torch.int64)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["score"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "score": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        # The torch 2.14 dynamo exporter emits Split.num_outputs, which ORT
        # rejects for this opset. The TorchScript exporter emits a standard
        # Split with the split-length input and loads everywhere.
        dynamo=False,
    )

    meta = {
        "base_model": base,
        "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": sha256_file(Path(args.ckpt)),
        "ckpt_step": blob.get("step"),
        "onnx": str(out_path.resolve()),
        "onnx_sha256": sha256_file(out_path),
        "opset": args.opset,
        "torch": torch.__version__,
        "inputs": ["input_ids:int64[batch,seq]", "attention_mask:int64[batch,seq]"],
        "output": "score:fp32[batch]",
    }
    meta_path = out_path.with_suffix(".export_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
