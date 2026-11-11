"""Post-freeze scoring, identical-device cost measurement and local occlusion."""
from __future__ import annotations

import argparse
import gc
import re
import time

from .common import (HANDOFF_COLUMNS, RESULTS, ROOT, config, environment, event,
                     load_frame, metrics, now, read_json, score_texts, selected,
                     selected_model, setup, sha256, source_hashes, token_stats,
                     write_csv, write_json)
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score():
    setup()
    model,tok,s=selected_model()
    output=RESULTS/"handoff/roberta_scores.parquet"
    if output.exists():
        raise RuntimeError("Raw scores already exist. Use verify to recompute without replacing them.")
    event("test_scoring_started",checkpoint_rev=s["checkpoint_rev"],frozen_at=s["frozen_at_utc"])
    rows,metric_rows,stats=[],[],[]
    for name,expected in config()["evaluation"]["expected_rows"].items():
        frame=load_frame(name)
        assert len(frame)==expected
        scores=score_texts(model,tok,frame.text.tolist())
        r=pd.DataFrame({"dataset":frame.dataset,"split":frame.split,"sample_id":frame.sample_id,
                        "gold":frame.label,"score_attack":scores,"model_id":s["model_id"],
                        "checkpoint_rev":s["checkpoint_rev"],"seed":s["seed"],
                        "transform_id":"clean","parent_id":None})[HANDOFF_COLUMNS]
        rows.append(r)
        metric_rows.append({"dataset_split":name,"model_id":s["model_id"],
                            "threshold_policy":"default_0.5_reference_only",**metrics(frame.label,scores)})
        stats.append(token_stats(tok,frame,name))
        event("split_scored",split=name,n=len(frame))
    output.parent.mkdir(parents=True,exist_ok=True)
    raw=pd.concat(rows,ignore_index=True)
    raw.to_parquet(output,index=False)
    predictions=raw.assign(threshold_policy="default_0.5_reference_only",threshold=0.5,
                           pred=(raw.score_attack>=0.5).astype(int))
    (RESULTS/"predictions").mkdir(parents=True,exist_ok=True)
    predictions.to_parquet(RESULTS/"predictions/roberta_predictions.parquet",index=False)
    write_csv(RESULTS/"metrics_at_0.5.csv",metric_rows)
    train_stats=pd.read_csv(RESULTS/"training_truncation.csv")
    write_csv(RESULTS/"truncation_stats.csv",pd.concat([train_stats.iloc[:1],pd.DataFrame(stats)],ignore_index=True))
    metadata={"experiment_id":config()["experiment_id"],"status":"scores_generated",
              "environment":environment(),"selected_model":s,"scores":{
                "file":output.relative_to(ROOT).as_posix(),"sha256":sha256(output),"rows":len(raw),
                "schema":HANDOFF_COLUMNS,"higher_means":"more attack-like",
                "score_type":"softmax_class_1_uncalibrated"},
              "scored_at_utc":now(),"source_sha256":source_hashes(),
              "reproduction":{"train":"python -m experiments.e4.train train",
                              "score":"python -m experiments.e4.evaluate score",
                              "verify":"python -m experiments.e4.verify"},
              "threshold_owner":"E6; 0.5 metrics are reference-only, not final group results"}
    write_json(RESULTS/"run_metadata.json",metadata)
    write_json(RESULTS/"handoff/run_metadata.json",metadata)
    cp=s["checkpoint_dir"]
    (RESULTS/"handoff/checkpoint_path.txt").write_text(
        f"checkpoint_dir = {cp}\nweights_sha256 = {s['checkpoint_rev']}\n"
        f"model_revision = {s['model_revision']}\nseed = {s['seed']}\n"
        "Run the following Python from the project root (CPU inference):\n"
        "import torch\nfrom transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
        "texts = ['Can you explain Python decorators?']\n"
        f"tokenizer = AutoTokenizer.from_pretrained('{cp}', local_files_only=True)\n"
        f"model = AutoModelForSequenceClassification.from_pretrained('{cp}', local_files_only=True).eval()\n"
        "with torch.inference_mode():\n"
        "    score_attack = torch.softmax(model(**tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')).logits, dim=-1)[:, 1]\n"
        "print(score_attack.tolist())\n",
        encoding="utf-8")
    event("scores_exported",rows=len(raw),sha256=sha256(output))


@torch.inference_mode()
def latency(model,tok,texts,device):
    model.to(device).eval()
    cfg=config()["latency"]
    def one(text):
        features=tok(text,truncation=True,max_length=256,return_token_type_ids=False,return_tensors="pt").to(device)
        return torch.softmax(model(**features).logits.float(),dim=-1)[0,1].item()
    for text in texts[:cfg["warmup_trials"]]:one(text)
    if device=="cuda":
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    values=[]
    for text in texts[cfg["warmup_trials"]:cfg["warmup_trials"]+cfg["measured_trials"]]:
        if device=="cuda":torch.cuda.synchronize()
        started=time.perf_counter();one(text)
        if device=="cuda":torch.cuda.synchronize()
        values.append((time.perf_counter()-started)*1000)
    summary={"device":device,"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),
             "mean_ms":float(np.mean(values)),"batch_size":1,"precision":"fp32",
             "warmup_trials":cfg["warmup_trials"],"measured_trials":len(values),
             "cpu_threads":torch.get_num_threads(),
             "includes_tokenization":True,"includes_transfer_softmax_cpu_scalar":True,
             "excludes_model_loading":True,
             "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2 if device=="cuda" else None,
             "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2 if device=="cuda" else None}
    return summary,values


def benchmark():
    setup()
    event("benchmark_started",note="Explicitly omit token_type_ids; the supplied E3 Bert tokenizer otherwise emits a field DistilBERT does not accept.")
    _,cp=selected()
    specs=[("e4_roberta_base",cp)]
    e3=ROOT/"models/e3/lr2e-5"
    if e3.exists():
        declared=read_json(ROOT/"results/e3/handoff/selected_config.json")
        assert sha256(e3/"model.safetensors")==declared["checkpoint_rev"],"E3 checkpoint hash mismatch"
        specs.append(("e3_distilbert_base_uncased",e3))
    frame=load_frame("d1_validation").iloc[:120]
    summaries,observations=[],[]
    write_csv(RESULTS/"latency_sample_ids.csv",frame[["sample_id"]].assign(
        role=["warmup"]*20+["measured"]*100))
    for name,path in specs:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
        for device in ["cuda","cpu"]:
            gc.collect();torch.cuda.empty_cache()
            model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,attn_implementation="sdpa")
            summary,values=latency(model,tok,frame.text.tolist(),device)
            summary.update(model_id=name,n_parameters=sum(p.numel() for p in model.parameters()),
                           checkpoint_bytes=sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                           weights_bytes=(path/"model.safetensors").stat().st_size,
                           checkpoint_rev=sha256(path/"model.safetensors"),hardware_gpu=torch.cuda.get_device_name(0))
            summaries.append(summary)
            observations.extend({"model_id":name,"device":device,"trial":i+1,
                                 "sample_id":frame.iloc[i+20].sample_id,"latency_ms":v} for i,v in enumerate(values))
            event("benchmark_complete",**summary)
            del model;gc.collect();torch.cuda.empty_cache()
    write_csv(RESULTS/"cost_comparison.csv",summaries)
    own=[r for r in summaries if r["model_id"]=="e4_roberta_base"]
    write_csv(RESULTS/"latency.csv",own)
    write_csv(RESULTS/"model_size.csv",[{k:own[0][k] for k in
              ["model_id","n_parameters","checkpoint_bytes","weights_bytes","checkpoint_rev"]}])
    write_csv(RESULTS/"latency_trials.csv",observations)
    write_json(RESULTS/"benchmark_environment.json",environment())
    s,_=selected()
    gpu=next(r for r in own if r["device"]=="cuda")
    s["cost"]={"n_parameters":gpu["n_parameters"],"checkpoint_bytes":gpu["checkpoint_bytes"],
                "batch1_latency_ms_median":gpu["median_ms"],"batch1_latency_ms_p95":gpu["p95_ms"],
                "inference_peak_vram_mb":gpu["peak_allocated_mib"],
                "training_peak_vram_mb":float(pd.read_csv(RESULTS/"tuning.csv").peak_allocated_mib.max()),
                "same_device_comparison":"results/e4/cost_comparison.csv"}
    write_json(RESULTS/"selected_config.json",s)
    write_json(RESULTS/"handoff/selected_config.json",s)


@torch.inference_mode()
def explain():
    setup()
    model,tok,s=selected_model()
    frame=load_frame("d1_test").set_index("sample_id")
    spans,case_rows=[],[]
    for sid in config()["occlusion"]["case_ids"]:
        row=frame.loc[sid];text=row.text
        encoded=tok(text,truncation=True,max_length=256,return_offsets_mapping=True,return_tensors="pt")
        offsets=encoded.pop("offset_mapping")[0].tolist()
        features={k:v.cuda() for k,v in encoded.items()}
        base=float(torch.softmax(model(**features).logits.float(),dim=-1)[0,1].item())
        case_rows.append({"sample_id":sid,"gold":int(row.label),"pred_at_0.5":int(base>=0.5),
                          "score_attack":base,"text":text,"checkpoint_rev":s["checkpoint_rev"]})
        for word_index,match in enumerate(re.finditer(r"\S+",text)):
            positions=[i for i,(a,b) in enumerate(offsets) if b>a and a<match.end() and b>match.start()]
            if not positions:continue
            altered={k:v.clone() for k,v in features.items()}
            altered["input_ids"][0,positions]=tok.mask_token_id
            masked=float(torch.softmax(model(**altered).logits.float(),dim=-1)[0,1].item())
            spans.append({"sample_id":sid,"span_index":word_index,"span":match.group(),
                          "char_start":match.start(),"char_end":match.end(),
                          "token_positions":" ".join(map(str,positions)),"masked_token_count":len(positions),
                          "original_score":base,"masked_score":masked,"delta_score":base-masked,
                          "gold":int(row.label),"checkpoint_rev":s["checkpoint_rev"]})
        event("case_occluded",sample_id=sid,score_attack=base)
    write_csv(RESULTS/"occlusion_cases.csv",case_rows)
    write_csv(RESULTS/"occlusion_spans.csv",spans)
    write_json(RESULTS/"occlusion_protocol.json",{
        **config()["occlusion"],"checkpoint_rev":s["checkpoint_rev"],"precision":"fp32",
        "time_utc":now(),"case_count":len(case_rows),"span_count":len(spans),
        "limitations":["Masking can produce out-of-distribution inputs.",
                       "Spans are tested individually, so interactions are not measured.",
                       "Cases came from the E2 case list; they need not be E4 errors.",
                       "These six selected cases do not represent the test distribution.",
                       "Gold labels and text are re-read from the frozen dataset; E2 interpretations are not copied."]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["score","benchmark","explain"])
    args=parser.parse_args()
    {"score":score,"benchmark":benchmark,"explain":explain}[args.command]()


if __name__=="__main__":main()
