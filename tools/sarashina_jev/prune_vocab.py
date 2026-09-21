"""Prune Sarashina-JEV's SentencePiece vocabulary by keeping a contiguous prefix.

Keeping token IDs 0..N-1 unchanged lets us truncate the embedding matrix
without remapping any retained token. Removed unigram pieces fall back to
smaller retained pieces / byte fallback.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

from tools.sarashina_jev.data import build_candidate_text, read_jsonl
from tools.sarashina_jev.model import SarashinaJevScorer


def find_sentencepiece_model(tokenizer_dir: Path) -> Path:
    candidates = [
        tokenizer_dir / "tokenizer.model",
        tokenizer_dir / "spiece.model",
        tokenizer_dir / "sentencepiece.bpe.model",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = list(tokenizer_dir.glob("*.model"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(f"SentencePiece model not found in {tokenizer_dir}")


def prune_sentencepiece_prefix(src: Path, dst: Path, target_vocab: int) -> dict:
    proto = sp_pb2.ModelProto()
    proto.ParseFromString(src.read_bytes())
    original = len(proto.pieces)
    if not (256 < target_vocab < original):
        raise ValueError(
            f"target vocab must be >256 and < original size {original}, got {target_vocab}"
        )

    # SentencePiece byte fallback requires all BYTE pieces to remain available.
    required_types = {
        sp_pb2.ModelProto.SentencePiece.UNKNOWN,
        sp_pb2.ModelProto.SentencePiece.CONTROL,
        sp_pb2.ModelProto.SentencePiece.USER_DEFINED,
        sp_pb2.ModelProto.SentencePiece.BYTE,
    }
    required_after_cut = [
        (i, piece.piece, int(piece.type))
        for i, piece in enumerate(proto.pieces)
        if i >= target_vocab and piece.type in required_types
    ]
    if required_after_cut:
        raise ValueError(
            "contiguous-prefix pruning would remove required special/byte pieces: "
            f"{required_after_cut[:20]}"
        )

    removed = original - target_vocab
    del proto.pieces[target_vocab:]
    proto.trainer_spec.vocab_size = target_vocab
    dst.write_bytes(proto.SerializeToString())

    return {
        "strategy": "contiguous_prefix",
        "original_vocab_size": original,
        "target_vocab_size": target_vocab,
        "removed_pieces": removed,
    }


def probe_tokenizer(old_tokenizer, new_tokenizer, rows: list[dict], limit: int) -> dict:
    old_total = new_total = samples = 0
    old_unk = new_unk = 0
    changed = 0
    max_ratio = 0.0
    examples = []

    unk_old = old_tokenizer.unk_token_id
    unk_new = new_tokenizer.unk_token_id

    for row in rows[:limit]:
        reading = str(row.get("reading") or "")
        context = str(row.get("context_prev") or row.get("context") or "")
        candidates = [str(x) for x in (row.get("mozc_nbest") or []) if x][:5]
        if not reading or not candidates:
            continue
        for candidate in candidates:
            text = build_candidate_text(reading, context, candidate)
            old_ids = old_tokenizer.encode(text, add_special_tokens=False)
            new_ids = new_tokenizer.encode(text, add_special_tokens=False)
            old_total += len(old_ids)
            new_total += len(new_ids)
            samples += 1
            old_unk += sum(int(x == unk_old) for x in old_ids) if unk_old is not None else 0
            new_unk += sum(int(x == unk_new) for x in new_ids) if unk_new is not None else 0
            if old_ids != new_ids:
                changed += 1
            ratio = len(new_ids) / max(1, len(old_ids))
            max_ratio = max(max_ratio, ratio)
            if ratio >= 1.5 and len(examples) < 10:
                examples.append(
                    {
                        "reading": reading,
                        "candidate": candidate,
                        "old_tokens": len(old_ids),
                        "new_tokens": len(new_ids),
                        "ratio": ratio,
                    }
                )

    return {
        "samples": samples,
        "changed_fraction": changed / samples if samples else 0.0,
        "old_avg_tokens": old_total / samples if samples else 0.0,
        "new_avg_tokens": new_total / samples if samples else 0.0,
        "token_length_ratio": new_total / max(1, old_total),
        "max_token_length_ratio": max_ratio,
        "old_unk_tokens": old_unk,
        "new_unk_tokens": new_unk,
        "large_expansion_examples": examples,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=64000)
    p.add_argument("--probe-data", default="")
    p.add_argument("--probe-limit", type=int, default=1000)
    args = p.parse_args()

    src = Path(args.artifact)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"load artifact={src}", flush=True)
    model, old_tokenizer, meta = SarashinaJevScorer.load_artifact(src)

    old_embed = model.backbone.get_input_embeddings()
    old_vocab, hidden = old_embed.weight.shape
    target = int(args.vocab_size)
    if target >= old_vocab:
        raise ValueError(f"target vocab {target} must be smaller than embedding vocab {old_vocab}")

    temp_tokenizer = out / "_tokenizer_build"
    if temp_tokenizer.exists():
        shutil.rmtree(temp_tokenizer)
    shutil.copytree(src / "tokenizer", temp_tokenizer)
    sp_path = find_sentencepiece_model(temp_tokenizer)
    sp_report = prune_sentencepiece_prefix(sp_path, sp_path, target)

    # Remove stale fast-tokenizer serialization if one exists. AutoTokenizer can
    # regenerate from the pruned SentencePiece model and tokenizer config.
    tokenizer_json = temp_tokenizer / "tokenizer.json"
    if tokenizer_json.exists():
        tokenizer_json.unlink()

    new_tokenizer = AutoTokenizer.from_pretrained(temp_tokenizer, use_fast=False)
    actual_vocab = int(new_tokenizer.vocab_size)
    if actual_vocab != target:
        raise RuntimeError(
            f"pruned tokenizer vocab mismatch: expected {target}, got {actual_vocab}"
        )

    if old_tokenizer.unk_token_id != new_tokenizer.unk_token_id:
        raise RuntimeError("unk token ID changed during contiguous-prefix pruning")
    if old_tokenizer.bos_token_id != new_tokenizer.bos_token_id:
        raise RuntimeError("bos token ID changed during contiguous-prefix pruning")
    if old_tokenizer.eos_token_id != new_tokenizer.eos_token_id:
        raise RuntimeError("eos token ID changed during contiguous-prefix pruning")
    if old_tokenizer.pad_token_id != new_tokenizer.pad_token_id:
        raise RuntimeError("pad token ID changed during contiguous-prefix pruning")

    new_embed = nn.Embedding(
        target,
        hidden,
        padding_idx=(
            old_embed.padding_idx
            if old_embed.padding_idx is not None and old_embed.padding_idx < target
            else None
        ),
        device=old_embed.weight.device,
        dtype=old_embed.weight.dtype,
    )
    with torch.no_grad():
        new_embed.weight.copy_(old_embed.weight[:target])
    new_embed.weight.requires_grad = old_embed.weight.requires_grad
    model.backbone.set_input_embeddings(new_embed)
    model.backbone.config.vocab_size = target

    kept = list(meta.get("kept_layer_indices", []))
    model.save_artifact(
        out,
        new_tokenizer,
        source_model=str(meta.get("source_model", "sbintuitions/sarashina2.2-0.5b")),
        kept_layer_indices=kept,
        page_size=int(meta.get("page_size", 5)),
    )

    # save_pretrained may produce a fresh tokenizer.model; make sure it is the
    # pruned model and clean the temporary folder.
    shutil.rmtree(temp_tokenizer, ignore_errors=True)

    report = {
        **sp_report,
        "embedding": {
            "original_params": int(old_vocab * hidden),
            "target_params": int(target * hidden),
            "removed_params": int((old_vocab - target) * hidden),
            "fp32_saved_mib": (old_vocab - target) * hidden * 4 / (1024**2),
            "fp16_saved_mib": (old_vocab - target) * hidden * 2 / (1024**2),
        },
        "special_ids": {
            "unk": new_tokenizer.unk_token_id,
            "bos": new_tokenizer.bos_token_id,
            "eos": new_tokenizer.eos_token_id,
            "pad": new_tokenizer.pad_token_id,
        },
    }

    if args.probe_data:
        rows = read_jsonl(args.probe_data)
        report["tokenizer_probe"] = probe_tokenizer(
            old_tokenizer,
            new_tokenizer,
            rows,
            args.probe_limit,
        )

    config_path = out / "jev_config.json"
    saved_meta = json.loads(config_path.read_text(encoding="utf-8"))
    saved_meta["vocab_pruning"] = report
    config_path.write_text(
        json.dumps(saved_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "vocab_prune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"VOCAB_PRUNE_DONE out={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
