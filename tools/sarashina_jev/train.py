"""Train a pruned Sarashina2 as a 5-choice Mozc page decision model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tools.sarashina_jev.data import ListwisePageDataset, build_pages, read_jsonl
from tools.sarashina_jev.model import DEFAULT_MODEL, SarashinaJevScorer


def choose_dtype(device: str, fp16: bool, bf16: bool) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.inference_mode()
def evaluate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    gold_total = 0
    gold_correct = 0
    anchor_total = 0
    anchor_kept = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cand_mask = batch["candidate_mask"].to(device)
        target = batch["target"].to(device)
        scores = model.score_pages(ids, mask, cand_mask)
        pred = scores.argmax(dim=1)
        ok = pred.eq(target)
        total += int(target.numel())
        correct += int(ok.sum().item())
        gold = batch["is_gold_page"].bool()
        if gold.any():
            gold_total += int(gold.sum().item())
            gold_correct += int(ok.cpu()[gold].sum().item())
        anchor = ~gold
        if anchor.any():
            anchor_total += int(anchor.sum().item())
            anchor_kept += int(pred.cpu()[anchor].eq(0).sum().item())
    return {
        "page_hit1": correct / total if total else 0.0,
        "gold_page_hit1": gold_correct / gold_total if gold_total else 0.0,
        "anchor_keep_rate": anchor_kept / anchor_total if anchor_total else 0.0,
        "pages": float(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="artifacts/sarashina_jev/12l")
    parser.add_argument("--keep-layers", type=int, default=12)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-last-n-layers", type=int, default=4)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = choose_dtype(device, args.fp16, args.bf16)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(args.train)
    if args.limit:
        train_rows = train_rows[: args.limit]
    train_pages = build_pages(
        train_rows,
        page_size=args.page_size,
        max_pages=args.max_pages,
        anchor_weight=args.anchor_weight,
    )
    if not train_pages:
        raise SystemExit("no training pages were produced")

    eval_rows = read_jsonl(args.eval) if args.eval else []
    eval_pages = build_pages(
        eval_rows,
        page_size=args.page_size,
        max_pages=args.max_pages,
        anchor_weight=args.anchor_weight,
    )

    model, kept = SarashinaJevScorer.from_pretrained(
        args.model,
        keep_layers=args.keep_layers,
        torch_dtype=dtype,
    )
    model.configure_trainable(
        last_n_layers=args.train_last_n_layers,
        train_embeddings=args.train_embeddings,
    )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.to(device)

    report = model.parameter_report()
    print(
        json.dumps(
            {
                "device": device,
                "dtype": str(dtype),
                "kept_layers": kept,
                "train_pages": len(train_pages),
                "eval_pages": len(eval_pages),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    train_ds = ListwisePageDataset(
        train_pages,
        tokenizer,
        page_size=args.page_size,
        max_length=args.max_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = (
        DataLoader(
            ListwisePageDataset(
                eval_pages,
                tokenizer,
                page_size=args.page_size,
                max_length=args.max_length,
            ),
            batch_size=args.batch_size,
        )
        if eval_pages
        else None
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and dtype == torch.float16),
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        running_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cand_mask = batch["candidate_mask"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=dtype if use_amp else torch.float32,
                enabled=use_amp,
            ):
                scores = model.score_pages(ids, mask, cand_mask)
                per_page = F.cross_entropy(scores, target, reduction="none")
                raw_loss = (per_page * weight).sum() / weight.sum().clamp_min(1e-6)
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()
            running += float(raw_loss.detach().cpu())
            running_count += 1

            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_index % 20 == 0:
                print(
                    f"epoch={epoch+1} batch={batch_index}/{len(train_loader)} "
                    f"loss={running/max(running_count, 1):.4f}",
                    flush=True,
                )
                running = 0.0
                running_count = 0

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # Persist the trained epoch before evaluation so an eval-only failure
        # does not discard the GPU work.
        model.save_artifact(
            out,
            tokenizer,
            source_model=args.model,
            kept_layer_indices=kept,
            page_size=args.page_size,
        )
        print(f"checkpoint_saved epoch={epoch+1} path={out}", flush=True)

        if eval_loader is not None:
            metrics = evaluate(model, eval_loader, device)
            print(
                f"eval epoch={epoch+1} {json.dumps(metrics)}",
                flush=True,
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_artifact(
        out,
        tokenizer,
        source_model=args.model,
        kept_layer_indices=kept,
        page_size=args.page_size,
    )
    (out / "train_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={out} steps={global_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
