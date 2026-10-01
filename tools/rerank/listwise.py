"""Candidate-set learning on the frozen public Dataset v2 contract.

Gold is never injected into candidates. Duplicate surfaces have one label.
Validation is used for model selection; final_test is never read.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import nn
from transformers import AutoModel

from tools.dataset.jsonl import read_jsonl
from tools.rerank.listwise_data import prepare
from tools.rerank.policy_replay import daemon_skip_reason
from tools.rerank.usage_guard import is_junk_surface


class Ranker(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            base, trust_remote_code=True, torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        self.encoder.config.reference_compile = False
        self.score = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        return self.score(self.encoder(input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state[:, 0]).squeeze(-1)


def collate(groups, device):
    tokens = [t for g in groups for t in g["tokens"]]
    width = max(map(len, tokens))
    ids = np.full((len(tokens), width), 3, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, row in enumerate(tokens):
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def packs(groups, budget=384):
    batch, count = [], 0
    for group in groups:
        n = len(group["tokens"])
        if batch and count + n > budget:
            yield batch
            batch, count = [], 0
        batch.append(group)
        count += n
    if batch:
        yield batch


@torch.inference_mode()
def predict(model, groups, budget=512):
    model.eval()
    results = []
    for batch in packs(groups, budget):
        ids, mask = collate(batch, "cuda")
        scores = model(ids, mask).float().cpu().tolist()
        offset = 0
        for g in batch:
            n = len(g["tokens"])
            results.append(scores[offset:offset+n])
            offset += n
    return results


def quality(groups, scores):
    n = len(groups)
    base = sum(g["gold_index"] == 0 for g in groups)
    raw = sum(int(np.argmax(s)) == g["gold_index"] for g,s in zip(groups,scores))
    neural = [(g,s) for g,s in zip(groups,scores) if g["eligibility_status"] == "NEURAL_ELIGIBLE"]
    sweep = []
    for tau in (0, .25, .5, .75, 1, 1.5, 2, 2.5, 3, 4, 5):
        helped = hurt = overwritten = 0
        for g,s in zip(groups,scores):
            best = int(np.argmax(s))
            allowed = not daemon_skip_reason(g["reading"], g["context"])
            if (allowed and not g["hard_protect"] and best != 0 and
                s[best]-s[0] >= tau and not is_junk_surface(g["candidates"][best])):
                overwritten += 1
                helped += g["gold_index"] == best
                hurt += g["gold_index"] == 0
        sweep.append({"tau": tau, "helped": helped, "hurt": hurt,
            "overwritten": overwritten, "final_hit1": (base+helped-hurt)/n})
    return {"n": n, "mozc_hit1": base/n, "raw_hit1": raw/n,
        "neural_eligible_n": len(neural), "neural_eligible_raw_hit1":
        sum(int(np.argmax(s))==g["gold_index"] for g,s in neural)/len(neural),
        "policy_sweep": sweep}


def write_scores(path, groups, scores):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for g,s in zip(groups,scores):
            record = {k:v for k,v in g.items() if k != "tokens"}
            record["scores"] = s
            f.write(json.dumps(record, ensure_ascii=False)+"\n")


def run_experiment(train_path, validation_path, baseline, out, variant,
                   epochs=2, lr=1e-5, distill_weight=0., layer_indices=None):
    torch.manual_seed(20260930)
    random.seed(20260930)
    np.random.seed(20260930)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    base_dir = Path(baseline)
    blob = torch.load(base_dir / "cross_encoder.pt", map_location="cpu", weights_only=False)
    if not blob.get("complete"):
        raise RuntimeError("refusing an incomplete initialization checkpoint")
    tok = spm.SentencePieceProcessor(model_file=str(base_dir / "tokenizer/tokenizer.model"))
    train = prepare(list(read_jsonl(Path(train_path))), tok, train=True)
    validation = prepare(list(read_jsonl(Path(validation_path))), tok)
    if len(validation) != 5974:
        raise RuntimeError("validation count mismatch")
    model = Ranker(blob["base_model"])
    if variant != "fresh":
        model.load_state_dict(blob["model"], strict=True)
    if layer_indices:
        model.encoder.layers = nn.ModuleList([model.encoder.layers[i] for i in layer_indices])
        model.encoder.config.num_hidden_layers = len(layer_indices)
    model.cuda()
    if distill_weight:
        teacher = Ranker(blob["base_model"])
        teacher.load_state_dict(blob["model"], strict=True)
        teacher.cuda()
        teacher_scores = predict(teacher, train)
        for g,s in zip(train,teacher_scores):
            g["teacher_scores"] = s
        del teacher
        torch.cuda.empty_cache()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    steps = sum(1 for _ in packs(train)) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda s:
        min(1., (s+1)/max(1,steps*.05))*max(0., 1.-s/steps))
    initial = quality(validation, predict(model, validation))
    print("INITIAL", variant, json.dumps(initial), flush=True)
    history = []
    start = time.monotonic()
    step = 0
    for epoch in range(epochs):
        random.shuffle(train)
        model.train()
        for batch in packs(train):
            ids, mask = collate(batch, "cuda")
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(ids, mask)
                losses, offset = [], 0
                for g in batch:
                    k = len(g["tokens"])
                    s = scores[offset:offset+k].float()
                    y = torch.tensor([g["gold_index"]], device="cuda")
                    ce = nn.functional.cross_entropy(s[None],y)
                    target = nn.functional.one_hot(y[0],k).float()
                    aux = nn.functional.binary_cross_entropy_with_logits(s,target)
                    loss = ce + .15*aux
                    if distill_weight:
                        ts = torch.tensor(g["teacher_scores"],device="cuda")
                        kl = nn.functional.kl_div(nn.functional.log_softmax(s/2,0),
                            nn.functional.softmax(ts/2,0),reduction="sum")*4
                        loss = loss + distill_weight*kl
                    losses.append(loss)
                    offset += k
                loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            scheduler.step()
            step += 1
            if step%100 == 0:
                print(f"TRAIN variant={variant} epoch={epoch+1} step={step} loss={loss.item():.5f} elapsed={time.monotonic()-start:.1f}",flush=True)
        scores = predict(model, validation)
        metrics = quality(validation, scores)
        metrics.update(epoch=epoch+1,step=step)
        history.append(metrics)
        epdir = root / f"epoch{epoch+1}"
        epdir.mkdir(exist_ok=True)
        torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
            "base_model":blob["base_model"],"complete":True,"step":step,
            "layer_indices":layer_indices,"objective":"listwise_ce+0.15_bce",
            "variant":variant}, epdir/"cross_encoder.pt")
        import shutil
        shutil.copytree(base_dir/"tokenizer",epdir/"tokenizer",dirs_exist_ok=True)
        write_scores(epdir/"validation_scores.jsonl.gz",validation,scores)
        (epdir/"validation_metrics.json").write_text(json.dumps(metrics,indent=2))
        print("VALIDATION", variant, json.dumps(metrics),flush=True)
    manifest = {"variant":variant,"seed":20260930,"epochs":epochs,"lr":lr,
        "distill_weight":distill_weight,"layer_indices":layer_indices,
        "train_groups":len(train),"validation_groups":len(validation),
        "final_test_mounted":False,"format":"contextual-ranking-v2-format-v1",
        "train_sha256":hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "validation_sha256":hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
        "baseline_sha256":hashlib.sha256((base_dir/"cross_encoder.pt").read_bytes()).hexdigest(),
        "initial":initial,"history":history,"elapsed_s":time.monotonic()-start,
        "torch":torch.__version__}
    import transformers
    manifest["transformers"] = transformers.__version__
    manifest["numpy"] = np.__version__
    manifest["sentencepiece"] = spm.__version__
    manifest["base_revision"] = getattr(model.encoder.config,"_commit_hash",None)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest
