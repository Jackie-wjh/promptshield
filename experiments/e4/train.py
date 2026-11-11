"""Train-only smoke test and six-candidate E4 experiment.

Gradient accumulation averages the loss over the actual effective batch, including
the final incomplete batch. Both learning rates start from identical weights and
use identical epoch permutations. No test split is read by this module.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import time

from .common import (CACHE, MODELS, RESULTS, ROOT, base_model, check_space, config,
                     config_hash, environment, event, load_frame, metrics, now,
                     read_json, score_texts, seed_all, setup, sha256, source_hashes,
                     token_stats, tokenizer, write_csv, write_json)
import numpy as np
import pandas as pd
import torch


def optimizer_for(model, lr):
    t = config()["training"]
    groups = [{"params": [p for n,p in model.named_parameters() if p.requires_grad
                          and not (n.endswith("bias") or "LayerNorm.weight" in n)],
               "weight_decay": t["weight_decay"]},
              {"params": [p for n,p in model.named_parameters() if p.requires_grad
                          and (n.endswith("bias") or "LayerNorm.weight" in n)],
               "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(t["adam_betas"]),
                             eps=t["adam_eps"], foreach=False)


def autocast_context(precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


def encoded_batch(encodings, labels, indices, tok):
    rows = [{k: encodings[k][int(i)] for k in ("input_ids", "attention_mask")} for i in indices]
    features = tok.pad(rows, padding=True, return_tensors="pt").to("cuda")
    gold = torch.as_tensor(labels[indices], dtype=torch.long, device="cuda")
    return features, gold


def update(model, optimizer, scaler, encodings, labels, indices, tok, micro, precision):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for start in range(0, len(indices), micro):
        part = indices[start:start+micro]
        inputs, gold = encoded_batch(encodings, labels, part, tok)
        with autocast_context(precision):
            logits = model(**inputs).logits
            summed = torch.nn.functional.cross_entropy(logits, gold, reduction="sum")
            loss = summed / len(indices)
        scaler.scale(loss).backward()
        loss_sum += float(summed.detach().float().item())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config()["training"]["max_grad_norm"])
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite gradient norm; do not silently continue")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < old_scale:
        raise RuntimeError("AMP skipped an optimizer update; inspect resource settings before formal training")
    return loss_sum / len(indices)


def smoke(micro, precision, steps):
    setup()
    check_space(5)
    seed_all(config()["seed"])
    train = load_frame("d1_train")
    tok = tokenizer()
    all_enc = tok(train.text.tolist(), truncation=True, max_length=256, padding=False)
    lengths = np.array([len(x) for x in all_enc["input_ids"]])
    rng = np.random.default_rng(config()["seed"])
    longest = np.argsort(lengths)[-16:]
    random_ids = rng.choice(np.setdiff1d(np.arange(len(train)), longest), size=84, replace=False)
    ids = np.concatenate([longest, random_ids])
    enc = {k: [v[i] for i in ids] for k,v in all_enc.items()}
    y = train.label.to_numpy()[ids]
    model = base_model().cuda().train()
    if config()["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = optimizer_for(model, 2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16")
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for step in range(steps):
        batch = np.arange(16) if step == 0 else rng.permutation(len(ids))[:16]
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = update(model,opt,scaler,enc,y,batch,tok,micro,precision)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
        if step % 10 == 0:
            event("smoke_progress", step=step+1, loss=loss)
    report = {"status": "passed", "scope": "D1 train only; these weights are discarded",
              "sample_count":100, "steps":steps, "micro_batch_size":micro,
              "effective_batch_size":16, "precision":precision,
              "first_step_max_token_length":int(lengths.max()),
              "median_update_seconds_after_warmup":float(np.median(durations[min(5,len(durations)-1):])),
              "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
              "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2,
              "environment":environment(), "time_utc":now()}
    report["estimated_six_epoch_training_seconds"] = report["median_update_seconds_after_warmup"]*math.ceil(len(train)/16)*6
    write_json(RESULTS/f"smoke_micro{micro}_{precision}.json",report)
    event("smoke_complete", **{k:v for k,v in report.items() if k!="environment"})


def train_all():
    setup()
    check_space(config()["storage"]["minimum_free_gib"])
    c, t = config(), config()["training"]
    assert t["effective_batch_size"] % t["micro_batch_size"] == 0
    freeze = RESULTS / "training_protocol.json"
    if freeze.exists():
        protocol = read_json(freeze)
        assert protocol["config_sha256"] == config_hash(), "Training config changed since freeze"
        for file, digest in protocol["training_source_sha256"].items():
            assert sha256(ROOT/file)==digest, f"Training implementation changed: {file}"
    else:
        write_json(freeze, {"time_utc":now(), "config":c, "config_sha256":config_hash(),
            "training_source_sha256":{p:sha256(ROOT/p) for p in
               ["experiments/e4/train.py", "experiments/e4/common.py"]},
            "environment":environment(),
            "data_manifest_sha256":sha256(ROOT/"data/audit/manifest.json")})
    if (RESULTS/"selected_config.json").exists():
        event("training_already_selected",note="No retraining after freeze")
        return
    train, dev = load_frame("d1_train"), load_frame("d1_validation")
    tok = tokenizer()
    write_csv(RESULTS/"training_truncation.csv",[token_stats(tok,train,"d1_train"),token_stats(tok,dev,"d1_validation")])
    enc = tok(train.text.tolist(), truncation=True,max_length=c["model"]["max_length"],padding=False)
    y = train.label.to_numpy(dtype=np.int64)
    effective = t["effective_batch_size"]
    per_epoch = math.ceil(len(train)/effective)
    total = per_epoch*t["epochs"]
    summaries, all_records = [], []
    for lr in t["learning_rates"]:
        tag = f"lr{lr:.0e}".replace("e-0","e-")
        checkpoint = MODELS/tag
        summary_path = RESULTS/f"training/{tag}_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            assert summary["config_sha256"] == config_hash()
            assert sha256(checkpoint/"model.safetensors") == summary["checkpoint_rev"]
            summaries.append(summary)
            all_records.extend(read_json(RESULTS/f"training/{tag}_epochs.json"))
            continue
        seed_all(c["seed"])
        model = base_model().cuda()
        if t["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        optimizer = optimizer_for(model,lr)
        warmup = round(t["warmup_ratio"]*total)
        def schedule(step):
            return step/max(1,warmup) if step < warmup else max(0.0,(total-step)/max(1,total-warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        scaler = torch.amp.GradScaler("cuda",enabled=t["precision"]=="fp16")
        records, best_f1, first_epoch = [], -1.0, 1
        resume = CACHE/f"resume_{tag}.pt"
        if resume.exists():
            state = torch.load(resume, map_location="cpu", weights_only=False)
            assert state["config_sha256"]==config_hash()
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            records,best_f1,first_epoch=state["records"],state["best_f1"],state["epoch"]+1
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            del state
            event("resumed",learning_rate=lr,epoch=first_epoch)
        steps_csv = RESULTS/f"training/{tag}_steps.csv"
        steps_csv.parent.mkdir(parents=True,exist_ok=True)
        # Step records from an interrupted epoch are kept separately, never treated as a completed candidate.
        with steps_csv.open("a",encoding="utf-8",newline="") as log:
            if log.tell()==0:
                log.write("epoch,update,learning_rate_used,train_loss\n")
            for epoch in range(first_epoch,t["epochs"]+1):
                check_space(5)
                order = np.random.default_rng(c["seed"]+epoch).permutation(len(train))
                order_hash = __import__("hashlib").sha256(order.astype("<i8").tobytes()).hexdigest()
                model.train()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                running, seen = 0.0,0
                for step,start in enumerate(range(0,len(train),effective),1):
                    ids=order[start:start+effective]
                    lr_used=optimizer.param_groups[0]["lr"]
                    loss=update(model,optimizer,scaler,enc,y,ids,tok,t["micro_batch_size"],t["precision"])
                    scheduler.step()
                    running+=loss*len(ids);seen+=len(ids)
                    log.write(f"{epoch},{step},{lr_used:.12g},{loss:.12g}\n")
                    if step%100==0:
                        log.flush()
                        event("training_progress",learning_rate=lr,epoch=epoch,update=step,total_updates=per_epoch,
                              train_loss=running/seen,seconds=time.perf_counter()-started)
                torch.cuda.synchronize()
                seconds=time.perf_counter()-started
                peak=torch.cuda.max_memory_allocated()/1024**2
                peak_reserved=torch.cuda.max_memory_reserved()/1024**2
                eval_started=time.perf_counter()
                scores,dev_loss=score_texts(model,tok,dev.text.tolist(),return_loss=True,labels=dev.label.tolist())
                met=metrics(dev.label.to_numpy(),scores)
                rec={"learning_rate":lr,"seed":c["seed"],"epoch":epoch,"train_loss":running/seen,
                     "dev_loss":dev_loss,**{f"dev_{k}":v for k,v in met.items()},
                     "optimizer_updates":per_epoch,"train_examples_seen":seen,
                     "epoch_train_seconds":seconds,"validation_seconds":time.perf_counter()-eval_started,
                     "peak_allocated_mib":peak,"peak_reserved_mib":peak_reserved,
                     "permutation_sha256":order_hash,"is_selected":False}
                records.append(rec)
                if met["macro_f1"] > best_f1:
                    best_f1=met["macro_f1"]
                    checkpoint.mkdir(parents=True,exist_ok=True)
                    model.save_pretrained(checkpoint,safe_serialization=True)
                    tok.save_pretrained(checkpoint)
                    write_csv(RESULTS/f"training/{tag}_best_validation.csv",
                              pd.DataFrame({"sample_id":dev.sample_id,"gold":dev.label,"score_attack":scores}))
                write_json(RESULTS/f"training/{tag}_epochs.json",records)
                write_csv(RESULTS/"tuning.csv",all_records+records)
                CACHE.mkdir(parents=True,exist_ok=True)
                state={"config_sha256":config_hash(),"epoch":epoch,"model":model.state_dict(),
                       "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                       "scaler":scaler.state_dict(),"records":records,"best_f1":best_f1,
                       "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
                torch.save(state,str(resume)+".tmp")
                os.replace(str(resume)+".tmp",resume)
                del state
                event("epoch_complete",**rec)
        best=sorted(records,key=lambda r:(-r["dev_macro_f1"],r["epoch"]))[0]
        summary={"learning_rate":lr,"epoch":best["epoch"],"dev_macro_f1":best["dev_macro_f1"],
                 "checkpoint_dir":checkpoint.relative_to(ROOT).as_posix(),
                 "checkpoint_rev":sha256(checkpoint/"model.safetensors"),
                 "config_sha256":config_hash()}
        write_json(summary_path,summary)
        summaries.append(summary);all_records.extend(records)
        del optimizer,scheduler,model,scaler
        gc.collect();torch.cuda.empty_cache()
    winner=sorted(summaries,key=lambda r:(-r["dev_macro_f1"],r["epoch"],r["learning_rate"]))[0]
    for rec in all_records:
        rec["is_selected"]=(rec["learning_rate"],rec["epoch"])==(winner["learning_rate"],winner["epoch"])
    write_csv(RESULTS/"tuning.csv",all_records)
    final={**winner,"selected_learning_rate":winner["learning_rate"],"selected_checkpoint_epoch":winner["epoch"],
           "model_id":c["model"]["model_id"],"hub_id":c["model"]["hub_id"],
           "model_revision":c["model"]["revision"],"tokenizer_revision":c["model"]["revision"],
           "seed":c["seed"],"selection":c["selection"],"training":t,
           "candidate_count":len(all_records),"frozen_at_utc":now(),
           "data_manifest_sha256":sha256(ROOT/"data/audit/manifest.json"),
           "source_sha256":source_hashes(),"status":"frozen_before_test_scoring"}
    write_json(RESULTS/"selected_config.json",final)
    write_json(RESULTS/"handoff/selected_config.json",final)
    write_csv(RESULTS/"handoff/tuning.csv",all_records)
    event("checkpoint_frozen",**winner)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["smoke","train"])
    parser.add_argument("--micro-batch",type=int,default=4)
    parser.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp32")
    parser.add_argument("--steps",type=int,default=50)
    args=parser.parse_args()
    if args.command=="smoke":smoke(args.micro_batch,args.precision,args.steps)
    else:train_all()


if __name__=="__main__":main()
