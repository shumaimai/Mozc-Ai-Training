"""Uncensored CPU accuracy/parity evaluation, with production guards replayed.

Outputs per-group scores for inspection and future threshold calibration.
The data path is explicit. No final-test defaults or network calls exist.
"""
from __future__ import annotations
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
from tools.dataset.jsonl import read_jsonl
from tools.rerank.cpu_optimization import load_runtime, percentiles


def summarize(records, runtime):
    base=sum(r["gold_index"]==0 for r in records)
    raw=sum(int(np.argmax(r["scores"]))==r["gold_index"] for r in records)
    neural=[r for r in records if r["eligibility_status"]=="NEURAL_ELIGIBLE"]
    sweep=[]
    for tau in (0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5):
        helped=hurt=overwritten=0
        for r in records:
            s=r["scores"]
            best=int(np.argmax(s))
            if (not runtime.skip_reason(r["reading"],r["context"]) and
                not r["hard_protect"] and best!=0 and s[best]-s[0]>=tau and
                not runtime.is_junk_surface(r["candidates"][best])):
                overwritten+=1
                helped+=best==r["gold_index"]
                hurt+=r["gold_index"]==0
        sweep.append(dict(tau=tau,helped=helped,hurt=hurt,overwritten=overwritten,
            final_hit1=(base+helped-hurt)/len(records)))
    return {"n":len(records),"mozc_hit1":base/len(records),"raw_hit1":raw/len(records),
        "neural_eligible_n":len(neural),"neural_eligible_raw_hit1":
        sum(int(np.argmax(r["scores"]))==r["gold_index"] for r in neural)/len(neural),
        "policy_sweep":sweep}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--reference",default="")
    args=p.parse_args()
    runtime=load_runtime(args.runtime)
    scorer=runtime.OrtScorer(Path(args.model),Path(args.tokenizer),128,args.threads)
    rows=list(read_jsonl(Path(args.data)))
    if args.limit: rows=rows[:args.limit]
    records,latencies=[],[]
    for i,row in enumerate(rows):
        cands=list(dict.fromkeys(c["surface"] for c in row["candidates"][:30] if c.get("surface")))
        reading=runtime.normalize_reading(row["reading"])
        ctx=runtime.clean_context(row["context_prev"])
        texts=[runtime.build_pair_text(reading,ctx,c) for c in cands]
        start=time.perf_counter()
        scores=scorer.score(texts)
        latency=(time.perf_counter()-start)*1000
        if i>=20 and not runtime.skip_reason(reading,ctx):
            latencies.append(latency)
        records.append({"row_index":i,"source_id":row.get("source_id"),
            "reading":reading,"context":ctx,"candidates":cands,
            "gold_index":cands.index(row["gold"]) if row["gold"] in cands else -1,
            "eligibility_status":row["eligibility_status"],"hard_protect":
            row["candidates"][0].get("protection")=="HARD_PROTECT",
            "scores":scores})
        if (i+1)%1000==0: print(f"scored={i+1}/{len(rows)}",flush=True)
    report=summarize(records,runtime)
    report.update(model=args.model,model_sha256=runtime.file_sha256(Path(args.model)),
        ort=runtime.ort.__version__,threads=args.threads,
        scoring_ms=percentiles(latencies),final_test_used=False)
    if args.reference:
        ref=list(read_jsonl(Path(args.reference)))
        if len(ref)!=len(records): raise RuntimeError("reference row count mismatch")
        diffs=[];argmax=final=final15=0
        context_mismatches = 0
        for a,b in zip(ref,records):
            if a["candidates"]!=b["candidates"]: raise RuntimeError("candidate mismatch")
            context_mismatches += a["context"] != b["context"]
            diffs.extend(abs(np.array(a["scores"])-b["scores"]))
            ia,ib=int(np.argmax(a["scores"])),int(np.argmax(b["scores"]))
            argmax+=ia==ib
            def select(r,j):
                return j if j!=0 and r["scores"][j]-r["scores"][0]>=2.5 else 0
            final+=select(a,ia)==select(b,ib)
            def select15(r,j):
                return j if j!=0 and r["scores"][j]-r["scores"][0]>=1.5 else 0
            final15+=select15(a,ia)==select15(b,ib)
        report["parity"]={"argmax_agreement":argmax/len(records),
            "tau2_5_selection_agreement":final/len(records),
            "tau1_5_selection_agreement":final15/len(records),
            "context_mismatches":context_mismatches,
            "max_abs_diff":float(max(diffs)),"mean_abs_diff":float(np.mean(diffs))}
    root=Path(args.out_dir)
    root.mkdir(parents=True,exist_ok=True)
    with gzip.open(root/"scores.jsonl.gz","wt",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    (root/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
