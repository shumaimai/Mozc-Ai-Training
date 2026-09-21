"""Phase 0 closure: same-container production-faithful 30M latency matrix.

Uses the existing 80-group/1608-score parity fixture from the v1 artifact;
does not create Dataset v2 or train a model.  Both ORT versions are installed
into isolated target directories in the same Modal container and benchmarked
against the same persistent ONNX session settings.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("mozc-v2-phase0-latency")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "sentencepiece")
volume = modal.Volume.from_name("mozc-artifacts", create_if_missing=False)

CHILD = r'''
import json, os, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

fixture = json.loads(Path(sys.argv[1]).read_text())
onnx_path, tok_path, out_path = sys.argv[2:5]
model = spm.SentencePieceProcessor(model_file=tok_path)

def pct(xs, p):
    ys=sorted(xs); k=(len(ys)-1)*p/100; a=int(k); b=min(a+1,len(ys)-1)
    return ys[a]+(ys[b]-ys[a])*(k-a)

def run(candidates, padding, max_len, intra):
    so=ort.SessionOptions(); so.intra_op_num_threads=intra; so.inter_op_num_threads=1
    so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    batches=[]
    for group in fixture:
        ctx=group["context"]
        cs=group["candidates"][:candidates]
        batches.append([f"読み: {group['reading']}\n文脈: {ctx}\n候補: {c}" for c in cs])
    def forward(texts):
        rows=[]
        for text in texts:
            ids=[1]+model.encode(text, out_type=int)[:max_len-2]+[2]
            rows.append(ids)
        width=max_len if padding=="fixed128" else max(map(len,rows))
        ids=np.full((len(rows),width),3,dtype=np.int64); mask=np.zeros_like(ids)
        for i,row in enumerate(rows): ids[i,:len(row)]=row; mask[i,:len(row)]=1
        session.run(None,{"input_ids":ids,"attention_mask":mask})
        return width
    for batch in batches[:5]: forward(batch)
    times=[]; seq=[]
    for batch in batches:
        t=time.perf_counter(); seq.append(forward(batch)); times.append((time.perf_counter()-t)*1000)
    return {"candidate_count":candidates,"padding":padding,"intra":intra,
            "effective_seq":{"p50":pct(seq,50),"p95":pct(seq,95),"max":max(seq)},
            "latency":{"n":len(times),"p50_ms":pct(times,50),"p95_ms":pct(times,95),"max_ms":max(times),"mean_ms":statistics.fmean(times)}}

cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.lower().startswith('model name:')),platform.processor()) or "unavailable_in_modal_proc"
report={"ort_version":ort.__version__,"cpu_model":cpu_model,"logical_cores":os.cpu_count(),"configs":[]}
for intra in [1,8]:
  for candidates in [5,30]:
    for padding in ["longest","fixed128"]:
      report["configs"].append(run(candidates,padding,128,intra))
Path(out_path).write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
'''


@app.function(image=image, volumes={"/artifacts": volume}, cpu=8, timeout=1800)
def benchmark() -> None:
    root = Path("/artifacts/track30m_ctx/onnx")
    parity = json.loads((root / "parity_scores.json").read_text())
    groups = []
    for row in parity["all_scores"]:
        if not groups or groups[-1]["reading"] != row["reading"]:
            groups.append({"reading": row["reading"], "context": "昨日の文章を確認しながら駅に停まった列車について入力を続けています。", "candidates": []})
        groups[-1]["candidates"].append(row["candidate"])
    if len(groups) != 80 or sum(map(lambda x: len(x["candidates"]), groups)) != 1608:
        raise RuntimeError(f"unexpected parity fixture shape: groups={len(groups)} scores={sum(map(lambda x: len(x['candidates']), groups))}")
    fixture = Path("/tmp/phase0_old_eval_fixture.json")
    fixture.write_text(json.dumps(groups, ensure_ascii=False))
    results = []
    for version, target in (("1.22.1", "/tmp/ort122"), ("1.30.0", "/tmp/ort130")):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--target", target, f"onnxruntime=={version}"])
        out = Path(f"/tmp/phase0_latency_{version}.json")
        env = dict(os.environ, PYTHONPATH=target)
        subprocess.check_call([sys.executable, "-c", CHILD, str(fixture), str(root / "cross_encoder_fp32.onnx"), str(root / "tokenizer" / "tokenizer.model"), str(out)], env=env)
        results.append(json.loads(out.read_text()))
    report = {"phase": "0-D-closure", "status": "complete", "fixture": {"source": "/artifacts/track30m_ctx/onnx/parity_scores.json", "groups": 80, "scores": 1608, "context": "fixed realistic Japanese preceding text for sequence-shape control", "dataset_v2_generated": False}, "same_container": True, "model": "sbintuitions/modernbert-ja-30m", "model_size_bytes": (root / "cross_encoder_fp32.onnx").stat().st_size, "results": results}
    out = Path("/artifacts/contextual_ranking_v2/phase0d_latency_production.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    volume.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2))


@app.local_entrypoint()
def main() -> None:
    benchmark.remote()
