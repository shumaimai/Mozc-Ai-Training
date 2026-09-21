"""QAT + teacher distillation for Sarashina-JEV INT8 recovery."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tools.sarashina_jev.data import ListwisePageDataset, build_pages, read_jsonl
from tools.sarashina_jev.model import SarashinaJevScorer
from tools.sarashina_jev.qat_utils import (
    qat_module_counts,
    replace_modules_for_qat,
    set_qat_activation_quantization,
    set_qat_strength,
)


@torch.inference_mode()
def evaluate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    total = correct = first_gold = predicted_first = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cand_mask = batch["candidate_mask"].to(device)
        target = batch["target"].to(device)
        scores = model.score_pages(ids, mask, cand_mask)
        pred = scores.argmax(dim=1)
        total += int(target.numel())
        correct += int(pred.eq(target).sum().item())
        first_gold += int(target.eq(0).sum().item())
        predicted_first += int(pred.eq(0).sum().item())
    return {
        "hit1": correct / total if total else 0.0,
        "candidate0_baseline": first_gold / total if total else 0.0,
        "predicted_candidate0_rate": predicted_first / total if total else 0.0,
        "pages": float(total),
    }


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-artifact", required=True)
    p.add_argument("--student-artifact", default="")
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--page-size", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--train-last-n-layers", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--embedding-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--kd-weight", type=float, default=0.7)
    p.add_argument("--mse-weight", type=float, default=0.05)
    p.add_argument("--margin-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument(
        "--activation-quantization",
        action="store_true",
        help="Also fake-quantize Linear activations (QDQ-style). Default is weight-only dynamic INT8 style.",
    )
    p.add_argument("--quant-start", type=float, default=0.25)
    p.add_argument("--quant-ramp-fraction", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("QAT requires CUDA in this experiment")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    teacher_dir = Path(args.teacher_artifact)
    student_dir = Path(args.student_artifact or args.teacher_artifact)
    print(
        f"load teacher={teacher_dir} student_init={student_dir} dtype={dtype}",
        flush=True,
    )

    teacher, tokenizer, teacher_meta = SarashinaJevScorer.load_artifact(
        teacher_dir,
        torch_dtype=dtype,
    )
    student, student_tokenizer, student_meta = SarashinaJevScorer.load_artifact(
        student_dir,
        torch_dtype=dtype,
    )

    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student.configure_trainable(
        last_n_layers=args.train_last_n_layers,
        train_embeddings=True,
        train_final_norm=True,
    )
    replaced = replace_modules_for_qat(student.backbone, quantize_embeddings=True)
    set_qat_activation_quantization(student, args.activation_quantization)
    student.to(device)
    if hasattr(student.backbone, "gradient_checkpointing_enable"):
        student.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    print(
        json.dumps(
            {
                "replaced_modules": replaced,
                **qat_module_counts(student),
                "student_params": student.parameter_report(),
                "activation_quantization": args.activation_quantization,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    train_rows = read_jsonl(args.train)
    if args.limit > 0:
        train_rows = train_rows[: args.limit]
    eval_rows = read_jsonl(args.eval)
    train_pages = build_pages(train_rows, page_size=args.page_size, anchor_weight=0.25)
    eval_pages = build_pages(eval_rows, page_size=args.page_size, anchor_weight=0.25)

    train_ds = ListwisePageDataset(
        train_pages,
        student_tokenizer,
        page_size=args.page_size,
        max_length=args.max_length,
        shuffle_gold_candidates=True,
        shuffle_seed=args.seed,
    )
    eval_ds = ListwisePageDataset(
        eval_pages,
        student_tokenizer,
        page_size=args.page_size,
        max_length=args.max_length,
        shuffle_gold_candidates=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    teacher_baseline = evaluate(teacher, eval_loader, device)
    print(f"teacher_eval {json.dumps(teacher_baseline)}", flush=True)

    head_params = [p for p in student.score_head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    embedding_params = [
        p for p in student.backbone.get_input_embeddings().parameters()
        if p.requires_grad
    ]
    embedding_ids = {id(p) for p in embedding_params}
    backbone_params = [
        p
        for p in student.parameters()
        if p.requires_grad and id(p) not in head_ids and id(p) not in embedding_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": embedding_params, "lr": args.embedding_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_batches = max(1, len(train_loader) * args.epochs)
    ramp_batches = max(1, int(total_batches * args.quant_ramp_fraction))
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    global_batch = 0
    optimizer.zero_grad(set_to_none=True)
    history: list[dict] = []

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        student.train()
        running = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "mse": 0.0, "margin": 0.0}
        running_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            global_batch += 1
            ramp = min(1.0, global_batch / ramp_batches)
            strength = args.quant_start + (1.0 - args.quant_start) * ramp
            set_qat_strength(student, strength)

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cand_mask = batch["candidate_mask"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                teacher_scores = teacher.score_pages(ids, mask, cand_mask)

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                student_scores = student.score_pages(ids, mask, cand_mask)
                ce_each = F.cross_entropy(student_scores, target, reduction="none")
                ce = weighted_mean(ce_each, weight)

                temp = float(args.temperature)
                teacher_prob = F.softmax(teacher_scores.float() / temp, dim=-1)
                student_log_prob = F.log_softmax(student_scores.float() / temp, dim=-1)
                kd_each = F.kl_div(
                    student_log_prob,
                    teacher_prob,
                    reduction="none",
                ).sum(dim=-1)
                kd = weighted_mean(kd_each, weight) * (temp * temp)

                student_center = student_scores.float() - student_scores.float().mean(
                    dim=-1,
                    keepdim=True,
                )
                teacher_center = teacher_scores.float() - teacher_scores.float().mean(
                    dim=-1,
                    keepdim=True,
                )
                mse_each = (student_center - teacher_center).pow(2).mean(dim=-1)
                mse = weighted_mean(mse_each, weight)

                # Match the teacher's top-1 separation from every other candidate.
                # This directly optimizes the ranking geometry used by the IME.
                teacher_top_idx = teacher_scores.float().argmax(dim=-1, keepdim=True)
                teacher_top = teacher_scores.float().gather(1, teacher_top_idx)
                student_top = student_scores.float().gather(1, teacher_top_idx)
                teacher_margin = teacher_top - teacher_scores.float()
                student_margin = student_top - student_scores.float()
                valid = cand_mask.float()
                margin_each = (
                    (student_margin - teacher_margin).pow(2) * valid
                ).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
                margin = weighted_mean(margin_each, weight)

                raw_loss = (
                    ce
                    + args.kd_weight * kd
                    + args.mse_weight * mse
                    + args.margin_weight * margin
                )
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()

            running["loss"] += float(raw_loss.detach().cpu())
            running["ce"] += float(ce.detach().cpu())
            running["kd"] += float(kd.detach().cpu())
            running["mse"] += float(mse.detach().cpu())
            running["margin"] += float(margin.detach().cpu())
            running_count += 1

            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if batch_index % 20 == 0:
                avg = {k: v / max(1, running_count) for k, v in running.items()}
                print(
                    f"qat epoch={epoch+1} batch={batch_index}/{len(train_loader)} "
                    f"strength={strength:.3f} "
                    f"loss={avg['loss']:.4f} ce={avg['ce']:.4f} "
                    f"kd={avg['kd']:.4f} mse={avg['mse']:.4f} "
                    f"margin={avg['margin']:.4f}",
                    flush=True,
                )
                running = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "mse": 0.0, "margin": 0.0}
                running_count = 0

        set_qat_strength(student, 1.0)
        student_eval = evaluate(student, eval_loader, device)
        epoch_report = {
            "epoch": epoch + 1,
            "student_fake_int8_eval": student_eval,
        }
        history.append(epoch_report)
        print(f"qat_eval {json.dumps(epoch_report)}", flush=True)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        student.save_artifact(
            out,
            student_tokenizer,
            source_model=str(teacher_meta.get("source_model", "sarashina-jeV")),
            kept_layer_indices=list(student_meta.get("kept_layer_indices", [])),
            page_size=args.page_size,
            extra_meta={
                "layer_pruning": student_meta.get("layer_pruning"),
                "parameter_report": student.parameter_report(),
            },
        )
        (out / "qat_checkpoint.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "teacher_meta": teacher_meta,
                    "student_init_meta": student_meta,
                    "teacher_eval": teacher_baseline,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"qat_checkpoint_saved epoch={epoch+1} path={out}", flush=True)

    print(
        json.dumps(
            {
                "teacher_eval": teacher_baseline,
                "history": history,
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
