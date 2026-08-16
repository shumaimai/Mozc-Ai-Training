"""Cross-encoder reranker training (IME Mozc N-best) for GPU/ROCm.

Example:

  python -m tools.rerank.train_cross_encoder train \\
    --train data/rerank_v2/train.jsonl \\
    --eval data/rerank_v2/holdout.jsonl \\
    --model cl-nagoya/ruri-v3-pt-70m \\
    --out artifacts/rerank/ruri70m_ce \\
    --epochs 2 --batch-size 512 --fp16 --require-cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl


def build_pair_text(reading: str, context_prev: str, candidate: str) -> str:
    parts = [f"読み: {reading}"]
    if context_prev:
        parts.append(f"文脈: {context_prev}")
    parts.append(f"候補: {candidate}")
    return " [SEP] ".join(parts)


@dataclass
class PairExample:
    text: str
    label: int
    group_id: str


def expand_groups(
    rows: list[dict[str, Any]],
    *,
    max_neg: int = 15,
    require_gold_in_nbest: bool = False,
) -> list[PairExample]:
    """One positive (gold) + hard negatives from Mozc N-best."""
    examples: list[PairExample] = []
    skipped = 0
    for i, row in enumerate(rows):
        if require_gold_in_nbest and not row.get("gold_in_nbest"):
            skipped += 1
            continue
        reading = row.get("reading") or ""
        gold = row.get("gold") or ""
        ctx = row.get("context_prev") or ""
        nbest = list(row.get("mozc_nbest") or [])
        if not reading or not gold:
            continue
        cands: list[str] = []
        seen: set[str] = set()
        for c in [gold, *nbest]:
            if not c or c in seen:
                continue
            seen.add(c)
            cands.append(c)
        if gold not in seen:
            continue
        group_id = f"{row.get('source','')}:{reading}:{gold}:{i}"
        negs = [c for c in cands if c != gold][:max_neg]
        examples.append(
            PairExample(
                text=build_pair_text(reading, ctx, gold),
                label=1,
                group_id=group_id,
            )
        )
        for neg in negs:
            examples.append(
                PairExample(
                    text=build_pair_text(reading, ctx, neg),
                    label=0,
                    group_id=group_id,
                )
            )
    if skipped:
        print(f"skipped_groups_not_in_nbest={skipped}", flush=True)
    return examples


def _vram_mb() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def command_dry_run(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.train)))
    pairs = expand_groups(
        rows,
        max_neg=args.max_neg,
        require_gold_in_nbest=args.require_gold_in_nbest,
    )
    pos = sum(1 for p in pairs if p.label == 1)
    neg = len(pairs) - pos
    report = {
        "train_groups": len(rows),
        "pair_examples": len(pairs),
        "positives": pos,
        "negatives": neg,
        "model": args.model,
        "require_gold_in_nbest": args.require_gold_in_nbest,
        "status": "dry-run-only",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def command_train(args: argparse.Namespace) -> int:
    import torch

    # ROCm/WSL: initialize CUDA before importing transformers/heavy libs.
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required (--require-cuda) but unavailable")
    if torch.cuda.is_available():
        if torch.cuda.memory_allocated() == 0:
            _ = torch.zeros(8, device="cuda")
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"device=cuda name={torch.cuda.get_device_name(0)} "
            f"vram_total_gb={props.total_memory/1024**3:.2f}",
            flush=True,
        )
    else:
        print("device=cpu (WARNING: not using VRAM)", flush=True)

    class PairDataset(Dataset):
        def __init__(self, items: list[PairExample], tokenizer, max_len: int):
            self.items = items
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.items[idx]
            enc = self.tokenizer(
                item.text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(item.label, dtype=torch.float),
            }

    class CrossEncoder(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            hidden = int(self.encoder.config.hidden_size)
            self.score = nn.Linear(hidden, 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            return self.score(cls).squeeze(-1)

    train_rows = list(read_jsonl(Path(args.train)))
    eval_rows = list(read_jsonl(Path(args.eval))) if args.eval else []
    train_pairs = expand_groups(
        train_rows,
        max_neg=args.max_neg,
        require_gold_in_nbest=args.require_gold_in_nbest,
    )
    eval_pairs = (
        expand_groups(
            eval_rows,
            max_neg=args.max_neg,
            require_gold_in_nbest=args.require_gold_in_nbest,
        )
        if eval_rows
        else []
    )
    print(
        f"train_groups={len(train_rows)} train_pairs={len(train_pairs)} "
        f"eval_pairs={len(eval_pairs)} batch_size={args.batch_size} fp16={args.fp16}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = CrossEncoder(args.model)
    if getattr(args, "init_ckpt", None):
        init_path = Path(args.init_ckpt)
        pt = init_path / "cross_encoder.pt" if init_path.is_dir() else init_path
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        print(f"init_ckpt_loaded={pt}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(getattr(args, "resume", "") or "")
    if getattr(args, "auto_resume", False) and not resume_path:
        cand = out_dir / "checkpoint_latest.pt"
        if cand.is_file():
            resume_path = cand
    resume_blob: dict[str, Any] | None = None
    if resume_path and resume_path.is_file():
        resume_blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_blob["model"], strict=True)
        print(f"resume_weights_loaded={resume_path} step={resume_blob.get('step')}", flush=True)
    if args.grad_checkpointing:
        enc = model.encoder
        if hasattr(enc, "config") and hasattr(enc.config, "use_cache"):
            enc.config.use_cache = False
        if hasattr(enc, "gradient_checkpointing_enable"):
            try:
                enc.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                enc.gradient_checkpointing_enable()
            print("gradient_checkpointing=ON", flush=True)
        else:
            print("gradient_checkpointing=UNSUPPORTED", flush=True)
    print("moving model to", device, flush=True)
    try:
        model = model.to(device)
    except Exception as exc:
        print(f"model.to({device}) failed: {exc}; retry after empty_cache", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        model = model.to(device)
    if device == "cuda":
        # Keep params in fp32; AMP handles compute dtype.
        print(f"model_loaded vram={_vram_mb()}", flush=True)

    pin = device == "cuda"
    train_loader = DataLoader(
        PairDataset(train_pairs, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    try:
        # fused AdamW has been flaky on this ROCm build; prefer standard.
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    except (TypeError, RuntimeError) as exc:
        print(f"AdamW init failed ({exc}); retry", flush=True)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * args.epochs)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(args.fp16 and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = out_dir / "train.log"
    save_every = int(getattr(args, "save_every", 0) or 0)
    on_checkpoint = getattr(args, "on_checkpoint", None)
    step = 0
    start_epoch = 0
    if resume_blob is not None:
        if "optimizer" in resume_blob:
            optim.load_state_dict(resume_blob["optimizer"])
        if "scheduler" in resume_blob:
            sched.load_state_dict(resume_blob["scheduler"])
        if "scaler" in resume_blob and use_amp:
            try:
                scaler.load_state_dict(resume_blob["scaler"])
            except Exception as exc:
                print(f"scaler_resume_skip={exc}", flush=True)
        step = int(resume_blob.get("step") or 0)
        start_epoch = min(args.epochs - 1, step // steps_per_epoch)
        print(
            f"resume_state step={step}/{total_steps} start_epoch={start_epoch+1}/{args.epochs}",
            flush=True,
        )
    model.train()
    t0 = time.perf_counter()

    def _save_ckpt(tag: str, *, complete: bool = False) -> Path:
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        path = out_dir / tag
        torch.save(
            {
                "model": cpu_state,
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "base_model": args.model,
                "step": step,
                "epochs": args.epochs,
                "total_steps": total_steps,
                "complete": complete,
            },
            path,
        )
        meta_ckpt = {
            "tag": tag,
            "step": step,
            "total_steps": total_steps,
            "epochs": args.epochs,
            "complete": complete,
            "path": str(path),
        }
        (out_dir / "checkpoint_meta.json").write_text(
            json.dumps(meta_ckpt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if callable(on_checkpoint):
            on_checkpoint(path)
        return path

    log_mode = "a" if resume_blob is not None and log_path.is_file() else "w"
    with log_path.open(log_mode, encoding="utf-8") as logf:
        if resume_blob is not None:
            logf.write(f"# resume_from_step={step}\n")
            logf.flush()
        # Resume keeps optim/sched/weights. Continue until total_steps.
        # Mid-epoch cancel → remaining steps are trained on freshly shuffled
        # epochs (acceptable; avoids full cold start).
        while step < total_steps:
            epoch = min(args.epochs - 1, step // steps_per_epoch)
            for batch in train_loader:
                if step >= total_steps:
                    break
                step += 1
                optim.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    logits = model(input_ids, attention_mask)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optim)
                scaler.update()
                # Only step LR when optimizer actually stepped (AMP may skip).
                if scaler.get_scale() >= scale_before:
                    sched.step()
                if step == 1 or step % args.log_every == 0 or step == total_steps:
                    elapsed = time.perf_counter() - t0
                    vram = _vram_mb()
                    line = (
                        f"epoch={epoch+1}/{args.epochs} step={step}/{total_steps} "
                        f"loss={loss.item():.4f} elapsed_s={elapsed:.1f} vram={vram}"
                    )
                    print(line, flush=True)
                    logf.write(line + "\n")
                    logf.flush()
                elif step <= 5:
                    print(
                        f"epoch={epoch+1}/{args.epochs} step={step}/{total_steps} "
                        f"loss={loss.item():.4f} vram={_vram_mb()}",
                        flush=True,
                    )
                if save_every > 0 and step % save_every == 0:
                    ckpt = _save_ckpt("checkpoint_latest.pt")
                    print(f"checkpoint_saved={ckpt} step={step}", flush=True)
            if save_every > 0 and step < total_steps:
                ckpt = _save_ckpt("checkpoint_latest.pt")
                print(f"checkpoint_epoch_boundary={ckpt} step={step}", flush=True)

    # Save
    _save_ckpt("cross_encoder.pt", complete=True)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    meta = {
        "base_model": args.model,
        "train_path": str(args.train),
        "eval_path": str(args.eval),
        "train_groups": len(train_rows),
        "train_pairs": len(train_pairs),
        "eval_groups": len(eval_rows),
        "eval_pairs": len(eval_pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "max_neg": args.max_neg,
        "fp16": use_amp,
        "require_gold_in_nbest": args.require_gold_in_nbest,
        "device": device,
        "vram_peak": _vram_mb(),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "init_ckpt": getattr(args, "init_ckpt", "") or "",
        "save_every": save_every,
        "resumed_from": str(resume_path) if resume_blob is not None else "",
        "complete": True,
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE wrote {out_dir}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="IME cross-encoder reranker")
    sub = p.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="expand pairs / write sample (no GPU train)")
    dry.add_argument("--train", default="data/rerank_v2/train.jsonl")
    dry.add_argument("--model", default="cl-nagoya/ruri-v3-pt-70m")
    dry.add_argument("--max-neg", type=int, default=15)
    dry.add_argument("--require-gold-in-nbest", action="store_true", default=True)
    dry.add_argument("--allow-gold-outside-nbest", action="store_true")
    dry.add_argument("--out", default="artifacts/rerank/dry_run.json")

    def _dry(args: argparse.Namespace) -> int:
        if args.allow_gold_outside_nbest:
            args.require_gold_in_nbest = False
        return command_dry_run(args)

    dry.set_defaults(func=_dry)

    tr = sub.add_parser("train", help="GPU train cross-encoder")
    tr.add_argument("--train", default="data/rerank_v2/train.jsonl")
    tr.add_argument("--eval", default="data/rerank_v2/holdout.jsonl")
    tr.add_argument("--model", default="cl-nagoya/ruri-v3-pt-70m")
    tr.add_argument(
        "--init-ckpt",
        default="",
        help="optional dir or cross_encoder.pt to continue from (Track B)",
    )
    tr.add_argument("--out", default="artifacts/rerank/ruri70m_ce")
    tr.add_argument("--epochs", type=int, default=2)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--max-len", type=int, default=128)
    tr.add_argument("--max-neg", type=int, default=15)
    tr.add_argument("--lr", type=float, default=2e-5)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--log-every", type=int, default=20)
    tr.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="if >0, write checkpoint_latest.pt every N steps (+ epoch boundary)",
    )
    tr.add_argument(
        "--resume",
        default="",
        help="resume from checkpoint_latest.pt (or explicit path)",
    )
    tr.add_argument(
        "--auto-resume",
        action="store_true",
        help="if out/checkpoint_latest.pt exists, resume from it",
    )
    tr.add_argument("--fp16", action="store_true", default=True)
    tr.add_argument("--no-fp16", action="store_true")
    tr.add_argument("--grad-checkpointing", action="store_true", default=True)
    tr.add_argument("--no-grad-checkpointing", action="store_true")
    tr.add_argument("--require-cuda", action="store_true", default=True)
    tr.add_argument("--allow-cpu", action="store_true")
    tr.add_argument("--require-gold-in-nbest", action="store_true", default=True)
    tr.add_argument("--allow-gold-outside-nbest", action="store_true")

    def _train(args: argparse.Namespace) -> int:
        if args.no_fp16:
            args.fp16 = False
        if args.no_grad_checkpointing:
            args.grad_checkpointing = False
        if args.allow_cpu:
            args.require_cuda = False
        if args.allow_gold_outside_nbest:
            args.require_gold_in_nbest = False
        return command_train(args)

    tr.set_defaults(func=_train)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
