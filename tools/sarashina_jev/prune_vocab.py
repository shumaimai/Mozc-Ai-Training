"""Prune Sarashina-JEV's SentencePiece vocabulary while preserving required pieces.

The lowest-ID ordinary pieces are retained so their token IDs stay unchanged.
Required special/byte pieces outside that prefix are moved to the new vocabulary
tail, and their corresponding embedding rows are copied to the remapped IDs.
Removed unigram pieces fall back to smaller retained pieces / byte fallback.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.utils import sentencepiece_model_pb2_new as sp_pb2

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


def prune_sentencepiece_prefix(
    src: Path,
    dst: Path,
    target_vocab: int,
) -> tuple[dict, list[int], dict[int, int]]:
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
    required_indices = [
        i for i, piece in enumerate(proto.pieces) if piece.type in required_types
    ]
    if len(required_indices) > target_vocab:
        raise ValueError(
            f"target vocab {target_vocab} is too small for "
            f"{len(required_indices)} required special/byte pieces"
        )

    # Reserve slots for every required piece, then fill the remaining slots with
    # the lowest-ID pieces. In Sarashina this keeps the ordinary contiguous
    # prefix unchanged and remaps only the three FIM control tokens at the tail.
    kept = set(required_indices)
    for i in range(original):
        if len(kept) >= target_vocab:
            break
        kept.add(i)
    kept_indices = sorted(kept)
    if len(kept_indices) != target_vocab:
        raise RuntimeError(
            f"failed to select {target_vocab} pieces; selected {len(kept_indices)}"
        )
    id_remap = {old_id: new_id for new_id, old_id in enumerate(kept_indices)}

    piece_blobs = [proto.pieces[i].SerializeToString() for i in kept_indices]
    preserved_tail = [
        {
            "old_id": old_id,
            "new_id": id_remap[old_id],
            "piece": proto.pieces[old_id].piece,
            "type": int(proto.pieces[old_id].type),
        }
        for old_id in required_indices
        if id_remap[old_id] != old_id
    ]
    removed = original - target_vocab
    del proto.pieces[:]
    for piece_blob in piece_blobs:
        proto.pieces.add().ParseFromString(piece_blob)
    proto.trainer_spec.vocab_size = target_vocab
    dst.write_bytes(proto.SerializeToString())

    report = {
        "strategy": "low_id_prefix_plus_required_tail",
        "original_vocab_size": original,
        "target_vocab_size": target_vocab,
        "removed_pieces": removed,
        "unchanged_prefix_size": next(
            (new_id for new_id, old_id in enumerate(kept_indices) if new_id != old_id),
            target_vocab,
        ),
        "remapped_required_pieces": preserved_tail,
    }
    return report, kept_indices, id_remap


def remap_added_tokens_decoder(tokenizer_dir: Path, id_remap: dict[int, int]) -> None:
    """Keep tokenizer_config added-token IDs aligned with the pruned SP model."""
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        return

    remapped_decoder = {}
    for old_id_text, token_config in decoder.items():
        old_id = int(old_id_text)
        if old_id not in id_remap:
            content = (
                token_config.get("content")
                if isinstance(token_config, dict)
                else token_config
            )
            raise ValueError(
                f"vocabulary pruning would remove configured added token "
                f"id={old_id} content={content!r}"
            )
        remapped_decoder[str(id_remap[old_id])] = token_config
    config["added_tokens_decoder"] = remapped_decoder
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    sp_report, kept_indices, id_remap = prune_sentencepiece_prefix(
        sp_path,
        sp_path,
        target,
    )
    remap_added_tokens_decoder(temp_tokenizer, id_remap)

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
    for remapped_piece in sp_report["remapped_required_pieces"]:
        actual_id = new_tokenizer.convert_tokens_to_ids(remapped_piece["piece"])
        if actual_id != remapped_piece["new_id"]:
            raise RuntimeError(
                f"required piece ID mismatch for {remapped_piece['piece']!r}: "
                f"expected {remapped_piece['new_id']}, got {actual_id}"
            )

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
        row_indices = torch.tensor(
            kept_indices,
            dtype=torch.long,
            device=old_embed.weight.device,
        )
        new_embed.weight.copy_(old_embed.weight.index_select(0, row_indices))
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
