"""FP/BF16 teacher distillation for a pruned Sarashina-JEV student."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tools.sarashina_jev.data import ListwisePageDataset, build_pages, read_jsonl
from tools.sarashina_jev.model import SarashinaJevScorer


@torch.inference_mode()
def evaluate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    total = correct = 0
    for batch in loader:
        scores = model.score_pages(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["candidate_mask"].to(device),
        )
        pred = scores.argmax(dim=1)
        target = batch["target"].to(device)
        total += int(target.numel())
        correct += int(pred.eq(target).sum().item())
    return {"hit1": correct / total if total else 0.0, "pages": total}


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-artifact", required=True)
    p.add_argument("--student-artifact", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--eval-holdout", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=3e-6)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=0.10)
    p.add_argument("--margin-weight", type=float, default=0.10)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("distillation requires CUDA in this experiment")
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    teacher, tokenizer, teacher_meta = SarashinaJevScorer.load_artifact(
        args.teacher_artifact, torch_dtype=dtype
    )
    student, student_tokenizer, student_meta = SarashinaJevScorer.load_artifact(
        args.student_artifact, torch_dtype=dtype
    )
    if int(teacher_meta.get("page_size", 5)) != int(student_meta.get("page_size", 5)):
        raise ValueError("teacher and student page_size differ")
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    student.configure_trainable(
        last_n_layers=0, train_embeddings=False, train_final_norm=True
    )
    student.to(device)
    if hasattr(student.backbone, "gradient_checkpointing_enable"):
        student.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    page_size = int(student_meta.get("page_size", 5))
    train_rows = read_jsonl(args.train)
    if args.limit > 0:
        train_rows = train_rows[: args.limit]
    eval_rows = read_jsonl(args.eval)
    train_ds = ListwisePageDataset(
        build_pages(train_rows, page_size=page_size, anchor_weight=0.25),
        student_tokenizer,
        page_size=page_size,
        shuffle_gold_candidates=True,
        shuffle_seed=args.seed,
    )
    eval_ds = ListwisePageDataset(
        build_pages(eval_rows, page_size=page_size, anchor_weight=0.25),
        student_tokenizer,
        page_size=page_size,
    )
    holdout_loader = None
    if args.eval_holdout:
        holdout_ds = ListwisePageDataset(
            build_pages(read_jsonl(args.eval_holdout), page_size=page_size, anchor_weight=0.25),
            student_tokenizer,
            page_size=page_size,
        )
        holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)
    teacher_eval = evaluate(teacher, eval_loader, device)
    teacher_holdout = evaluate(teacher, holdout_loader, device) if holdout_loader else None

    head_params = list(student.score_head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in student.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        student.train()
        running = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "mse": 0.0, "margin": 0.0}
        count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cand_mask = batch["candidate_mask"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                teacher_scores = teacher.score_pages(ids, mask, cand_mask)
            with torch.autocast("cuda", dtype=dtype):
                student_scores = student.score_pages(ids, mask, cand_mask)
                ce = weighted_mean(F.cross_entropy(student_scores, target, reduction="none"), weight)
                temp = float(args.temperature)
                kd = weighted_mean(
                    F.kl_div(
                        F.log_softmax(student_scores.float() / temp, dim=-1),
                        F.softmax(teacher_scores.float() / temp, dim=-1),
                        reduction="none",
                    ).sum(dim=-1),
                    weight,
                ) * (temp * temp)
                sc = student_scores.float() - student_scores.float().mean(dim=-1, keepdim=True)
                tc = teacher_scores.float() - teacher_scores.float().mean(dim=-1, keepdim=True)
                mse = weighted_mean((sc - tc).pow(2).mean(dim=-1), weight)
                ti = teacher_scores.float().argmax(dim=-1, keepdim=True)
                sm = student_scores.float().gather(1, ti) - student_scores.float()
                tm = teacher_scores.float().gather(1, ti) - teacher_scores.float()
                margin = weighted_mean(((sm - tm).pow(2) * cand_mask.float()).mean(dim=-1), weight)
                raw = ce + args.kd_weight * kd + args.mse_weight * mse + args.margin_weight * margin
                loss = raw / args.grad_accum
            scaler.scale(loss).backward()
            for key, value in (("loss", raw), ("ce", ce), ("kd", kd), ("mse", mse), ("margin", margin)):
                running[key] += float(value.detach().cpu())
            count += 1
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        student_eval = evaluate(student, eval_loader, device)
        student_holdout = evaluate(student, holdout_loader, device) if holdout_loader else None
        report = {
            "epoch": epoch + 1,
            "losses": {k: v / max(1, count) for k, v in running.items()},
            "student_eval": student_eval,
            "student_holdout": student_holdout,
        }
        history.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        out = Path(args.out)
        student.save_artifact(
            out,
            student_tokenizer,
            source_model=str(student_meta.get("source_model", teacher_meta.get("source_model", "sarashina2.2-0.5b"))),
            kept_layer_indices=list(student_meta.get("kept_layer_indices", [])),
            page_size=page_size,
            extra_meta={
                "layer_pruning": student_meta.get("layer_pruning"),
                "parameter_report": student.parameter_report(),
            },
        )
        (out / "distill_checkpoint.json").write_text(
            json.dumps({"args": vars(args), "teacher_meta": teacher_meta, "student_init_meta": student_meta,
                        "teacher_eval": teacher_eval, "teacher_holdout": teacher_holdout, "history": history},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"out": args.out, "teacher_eval": teacher_eval, "teacher_holdout": teacher_holdout, "history": history}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
