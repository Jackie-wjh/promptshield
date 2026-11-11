"""Shared protocol utilities. Test data are loaded only by post-freeze commands."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("E4_CACHE", str(ROOT / ".cache/e4")))
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RESULTS = ROOT / "results/e4"
MODELS = ROOT / "models/e4"
HANDOFF_COLUMNS = ["dataset", "split", "sample_id", "gold", "score_attack",
                   "model_id", "checkpoint_rev", "seed", "transform_id", "parent_id"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config():
    return read_json(ROOT / "configs/e4.json")


def config_hash():
    return sha256(ROOT / "configs/e4.json")


def event(name, **kwargs):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": now(), "event": name, **kwargs}) + "\n")
    print(name, json.dumps(kwargs), flush=True)


def check_space(required_gib=5):
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < required_gib:
        raise RuntimeError(f"Storage gate: {free:.2f} GiB free; need {required_gib} GiB. Ask user to clear space.")
    return free


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def setup():
    torch.set_num_threads(config()["training"]["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local training run; no silent CPU fallback.")


def environment():
    import sklearn
    import tokenizers
    import pyarrow
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0),
                    total_vram_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2,
                    cudnn=torch.backends.cudnn.version())
    try:
        info["git_base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_base_commit"] = "unavailable"
    return info


def load_frame(name):
    """Check byte hash, row count, IDs and class counts before returning a split."""
    path = ROOT / f"data/processed/{name}.parquet"
    manifest = read_json(ROOT / "data/audit/manifest.json")
    rec = next(r for r in manifest["processed_files"] if Path(r["path"]).name == path.name)
    assert sha256(path) == rec["sha256"], f"Data hash mismatch: {name}"
    df = pd.read_parquet(path)
    assert len(df) == rec["rows"] and df.sample_id.is_unique
    assert {str(k): int(v) for k, v in df.label.value_counts().items()} == rec["labels"]
    assert df.text.notna().all() and df.label.isin([0, 1]).all()
    return df


def tokenizer(checkpoint=None):
    c = config()["model"]
    if checkpoint is not None:
        return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    return AutoTokenizer.from_pretrained(c["hub_id"], revision=c["revision"], use_fast=True)


def base_model():
    c = config()["model"]
    return AutoModelForSequenceClassification.from_pretrained(
        c["hub_id"], revision=c["revision"], num_labels=2,
        id2label={0: "benign", 1: "attack"}, label2id={"benign": 0, "attack": 1},
        attn_implementation=c["attn_implementation"], use_safetensors=True)


def selected():
    s = read_json(RESULTS / "selected_config.json")
    assert s["config_sha256"] == config_hash(), "Configuration changed after model selection"
    cp = ROOT / s["checkpoint_dir"]
    assert sha256(cp / "model.safetensors") == s["checkpoint_rev"], "Checkpoint hash mismatch"
    return s, cp


def selected_model(device="cuda"):
    s, cp = selected()
    m = AutoModelForSequenceClassification.from_pretrained(cp, local_files_only=True,
          attn_implementation=config()["model"]["attn_implementation"]).to(device).eval()
    return m, tokenizer(cp), s


@torch.inference_mode()
def score_texts(model, tok, texts, device="cuda", batch_size=None, return_loss=False, labels=None):
    model.eval()
    b = batch_size or config()["evaluation"]["batch_size"]
    scores, total_loss = [], 0.0
    for start in range(0, len(texts), b):
        features = tok(texts[start:start+b], padding=True, truncation=True,
                       max_length=config()["model"]["max_length"], return_tensors="pt").to(device)
        logits = model(**features).logits.float()
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(p.tolist())
        if return_loss:
            y = torch.tensor(labels[start:start+b], device=device, dtype=torch.long)
            total_loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
    out = np.asarray(scores, dtype=np.float64)
    assert np.isfinite(out).all()
    return (out, total_loss / len(texts)) if return_loss else out


def metrics(gold, scores, threshold=0.5):
    gold, scores = np.asarray(gold), np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[0, 1]).ravel()
    result = dict(n=len(gold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                  threshold=threshold, fpr=float(fp / (fp + tn)))
    if len(np.unique(gold)) == 2:
        result.update(macro_f1=float(f1_score(gold, pred, average="macro", zero_division=0)),
                      attack_recall=float(tp / (tp + fn)), auroc=float(roc_auc_score(gold, scores)))
    return result


def token_stats(tok, frame, name):
    ids = tok(frame.text.tolist(), truncation=False, padding=False, return_attention_mask=False)["input_ids"]
    lengths = np.asarray([len(x) for x in ids])
    return {"dataset_split": name, "n": len(frame), "tokens_median": float(np.median(lengths)),
            "tokens_p95": float(np.percentile(lengths, 95)), "tokens_max": int(lengths.max()),
            "truncated_rows": int((lengths > config()["model"]["max_length"]).sum()),
            "truncation_rate": float((lengths > config()["model"]["max_length"]).mean())}


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted((ROOT / "experiments/e4").glob("*.py"))}
