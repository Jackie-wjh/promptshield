#!/usr/bin/env python3
"""Run E3: DistilBERT-base-uncased learning-rate comparison (LR 2e-5 versus 5e-5).

Protocol (see configs/e3.json):
  * Train on cleaned D1 train only, at most three epochs per learning rate.
  * After every epoch, evaluate D1 validation and record Macro F1 at threshold 0.5.
  * Select the single checkpoint by D1 validation Macro F1 only. D1/D2/D3 test and
    D4 NotInject are scored once, after that checkpoint is frozen.
  * Optionally re-run the frozen configuration with two extra seeds and report the
    spread; the primary seed always stays the reported pipeline.

Usage:
    python experiments/e3/run_e3.py
    python experiments/e3/run_e3.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/e3.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: one learning rate, one epoch, a few hundred training rows, no seed study.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("  ! CUDA requested but unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------------------
# data integrity gate (the handoff package does not ship scripts/check_data_integrity.py)
# --------------------------------------------------------------------------------------


def verify_data_integrity(project_root: Path, expected_rows: dict[str, int]) -> dict[str, Any]:
    """Re-derive file identity from data/audit/manifest.json before any training happens."""

    manifest_path = project_root / "data/audit/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {Path(entry["path"]).name: entry for entry in manifest["processed_files"]}
    checks: dict[str, Any] = {}
    status = "passed"

    for split, rows in expected_rows.items():
        filename = f"{split}.parquet"
        entry = declared.get(filename)
        if entry is None:
            checks[split] = {"status": "missing_from_manifest"}
            status = "failed"
            continue
        path = project_root / "data/processed" / filename
        actual_sha = sha256_file(path)
        frame = pd.read_parquet(path)
        label_counts = {str(int(k)): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
        declared_labels = {str(k): int(v) for k, v in entry["labels"].items()}
        ok = (
            actual_sha == entry["sha256"]
            and len(frame) == rows
            and len(frame) == entry["rows"]
            and label_counts == declared_labels
            and frame["sample_id"].is_unique
        )
        checks[split] = {
            "status": "passed" if ok else "failed",
            "manifest_sha256": entry["sha256"],
            "actual_sha256": actual_sha,
            "manifest_rows": int(entry["rows"]),
            "actual_rows": int(len(frame)),
            "label_counts": label_counts,
            "sample_id_unique": bool(frame["sample_id"].is_unique),
        }
        if not ok:
            status = "failed"

    return {
        "status": status,
        "manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
        "note": "Computed here because the handoff package does not include scripts/check_data_integrity.py.",
    }


# --------------------------------------------------------------------------------------
# metrics (identical definitions to experiments/e1/run_e1.py)
# --------------------------------------------------------------------------------------


def metric_record(gold: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(gold, prediction, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(gold, prediction, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(gold, prediction, pos_label=1, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "auroc": float(roc_auc_score(gold, scores)) if len(np.unique(gold)) == 2 else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(gold)),
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores.astype(float))
    return np.concatenate([unique, [np.nextafter(unique.max(), np.inf)]])


def select_thresholds(gold: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    """Same rule as E1. E3 only uses this to show a *provisional* dev-selected operating
    point in the demo app; the authoritative thresholds are produced by E6."""

    records = [(float(t), metric_record(gold, scores, float(t))) for t in threshold_candidates(scores)]
    tf1_threshold, tf1_metrics = max(records, key=lambda item: (item[1]["macro_f1"], item[0]))
    eligible = [item for item in records if item[1]["fpr"] is not None and item[1]["fpr"] <= 0.05]
    t5_threshold, t5_metrics = max(eligible, key=lambda item: (item[1]["attack_recall"], item[0]))
    return {
        "default_0.5": {"threshold": 0.5, **metric_record(gold, scores, 0.5)},
        "tF1": {"threshold": tf1_threshold, **tf1_metrics},
        "t5": {"threshold": t5_threshold, **t5_metrics},
    }


# --------------------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------------------


class EncodedDataset(Dataset):
    """Pre-tokenized texts without padding; padding happens per batch (dynamic padding)."""

    def __init__(self, input_ids: list[list[int]], attention_mask: list[list[int]], labels: np.ndarray) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[dict[str, list[int]], int]:
        return (
            {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index]},
            int(self.labels[index]),
        )


def collate_dynamic(pad_token_id: int):
    def collate(batch: list[tuple[dict[str, list[int]], int]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features, labels = zip(*batch)
        longest = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((len(features), longest), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), longest), dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}, torch.tensor(labels, dtype=torch.long)

    return collate


def encode_texts(tokenizer: Any, texts: list[str], max_length: int) -> tuple[list[list[int]], list[list[int]]]:
    encoded = tokenizer(texts, max_length=max_length, truncation=True, padding=False, return_attention_mask=True)
    return encoded["input_ids"], encoded["attention_mask"]


def token_length_stats(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, Any]:
    lengths = np.array(
        [len(ids) for ids in tokenizer(texts, truncation=False, padding=False, return_attention_mask=False)["input_ids"]],
        dtype=int,
    )
    truncated = int((lengths > max_length).sum())
    return {
        "n": int(len(lengths)),
        "max_length": int(max_length),
        "token_min": int(lengths.min()),
        "token_median": float(np.median(lengths)),
        "token_mean": float(lengths.mean()),
        "token_p95": float(np.percentile(lengths, 95)),
        "token_max": int(lengths.max()),
        "truncated_rows": truncated,
        "truncation_rate": float(truncated / len(lengths)) if len(lengths) else 0.0,
    }


# --------------------------------------------------------------------------------------
# model helpers
# --------------------------------------------------------------------------------------


def load_base_model(config: dict[str, Any], device: torch.device) -> Any:
    model_config = config["model"]
    kwargs: dict[str, Any] = {
        "revision": model_config["revision"],
        "num_labels": int(model_config["num_labels"]),
    }
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config["hub_id"], attn_implementation=model_config["attn_implementation"], **kwargs
        )
    except (TypeError, ValueError) as error:
        print(f"  ! attn_implementation not accepted ({error}); using the default attention path", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_config["hub_id"], **kwargs)
    return model.to(device)


def build_optimizer(model: Any, lr: float, training: dict[str, Any]) -> torch.optim.Optimizer:
    exclusions = tuple(training["weight_decay_exclusions"])
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if any(token in name for token in exclusions) else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": float(training["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=tuple(training["adam_betas"]),
        eps=float(training["adam_eps"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_ratio: float, total_steps: int) -> Any:
    warmup_steps = int(round(warmup_ratio * total_steps))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


@torch.no_grad()
def score_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: dict[str, Any],
    device: torch.device,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return the uncalibrated class-1 (attack) score for every text."""

    model.eval()
    model_config = config["model"]
    batch = int(batch_size or config["evaluation"]["eval_batch_size"])
    scores = np.empty(len(texts), dtype=np.float64)
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        encoded = tokenizer(
            chunk,
            max_length=int(model_config["max_length"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits.float()
        scores[start : start + len(chunk)] = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    return scores


def measure_latency(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Batch-1 latency including tokenization and host-to-device transfer."""

    model_config = config["model"]
    latency_config = config["latency"]
    warmup = int(latency_config["warmup_trials"])
    measured = int(latency_config["measured_trials"])
    if len(texts) < warmup + measured:
        measured = max(1, len(texts) - warmup)

    model.eval()

    def once(text: str) -> float:
        encoded = tokenizer(
            text,
            max_length=int(model_config["max_length"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits.float()
        return float(torch.softmax(logits, dim=-1)[0, 1].item())

    with torch.no_grad():
        for text in texts[:warmup]:
            once(text)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations: list[float] = []
        for text in texts[warmup : warmup + measured]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            once(text)
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000.0)

    return {
        "device": str(device),
        "batch_size": 1,
        "warmup_trials": warmup,
        "measured_trials": measured,
        "includes_tokenization": True,
        "excludes_model_loading": True,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "mean_ms": float(np.mean(durations)),
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


def train_one_configuration(
    lr: float,
    seed: int,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    tokenizer: Any,
    train_loader: DataLoader,
    dev_scores_fn: Any,
    dev_gold: np.ndarray,
    checkpoint_dir: Path,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Fine-tune one learning rate and return its best-epoch record plus the step curve."""

    training = config["training"]
    model = load_base_model(config, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, lr, training)
    total_steps = int(math.ceil(len(train_loader) / int(training["gradient_accumulation_steps"]))) * epochs
    scheduler = build_scheduler(optimizer, float(training["warmup_ratio"]), total_steps)

    curves: list[dict[str, Any]] = []
    best: dict[str, Any] = {"macro_f1": -1.0, "epoch": 0, "state": None, "record": None}
    epoch_records: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        running_loss = 0.0
        seen = 0
        epoch_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step, (features, labels) in enumerate(train_loader, start=1):
            features = {key: value.to(device) for key, value in features.items()}
            labels = labels.to(device)
            logits = model(**features).logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            curves.append(
                {
                    "learning_rate": lr,
                    "seed": seed,
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "train_loss": float(loss.item()),
                    "lr_used": float(scheduler.get_last_lr()[0]),
                }
            )

        epoch_seconds = time.perf_counter() - epoch_started
        peak_vram = (
            float(torch.cuda.max_memory_allocated(device)) / (1024**2) if device.type == "cuda" else None
        )
        dev_scores = dev_scores_fn(model)
        metrics = metric_record(dev_gold, dev_scores, 0.5)
        record = {
            "learning_rate": lr,
            "seed": seed,
            "epoch": epoch,
            "train_loss_epoch": running_loss / max(1, seen),
            "dev_macro_f1": metrics["macro_f1"],
            "dev_attack_recall": metrics["attack_recall"],
            "dev_fpr": metrics["fpr"],
            "dev_auroc": metrics["auroc"],
            "dev_tn": metrics["tn"],
            "dev_fp": metrics["fp"],
            "dev_fn": metrics["fn"],
            "dev_tp": metrics["tp"],
            "selection_threshold": 0.5,
            "steps": len(train_loader),
            "total_global_steps": total_steps,
            "epoch_seconds": epoch_seconds,
            "samples_per_second": float(seen * 1.0 / epoch_seconds) if epoch_seconds else None,
            "peak_vram_mb": peak_vram,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "n_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "is_selected": False,
        }
        epoch_records.append(record)
        if verbose:
            print(
                f"    epoch {epoch}: train_loss={record['train_loss_epoch']:.4f} "
                f"dev_macro_f1={metrics['macro_f1']:.4f} dev_recall={metrics['attack_recall']:.4f} "
                f"dev_fpr={metrics['fpr']:.4f} ({epoch_seconds:.1f}s, peak {peak_vram} MB)",
                flush=True,
            )
        if metrics["macro_f1"] > best["macro_f1"]:
            best = {
                "macro_f1": metrics["macro_f1"],
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "record": record,
            }

    if best["state"] is None:
        raise RuntimeError(f"No checkpoint recorded for learning rate {lr}")

    model.load_state_dict(best["state"])
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    weights = checkpoint_dir / "model.safetensors"
    summary = {
        "learning_rate": lr,
        "seed": seed,
        "selected_epoch": int(best["epoch"]),
        "dev_macro_f1_at_0.5": float(best["macro_f1"]),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_weights": str(weights),
        "checkpoint_rev": sha256_file(weights),
        "checkpoint_bytes": directory_bytes(checkpoint_dir),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, epoch_records, {"curves": curves, "config": summary}


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def plot_learning_curves(curves: pd.DataFrame, records: pd.DataFrame, selected: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    palette = {2e-05: "#3B6EA8", 5e-05: "#B23A48"}
    for lr, frame in curves.groupby("learning_rate"):
        frame = frame.sort_values("global_step")
        smooth = frame["train_loss"].rolling(window=20, min_periods=1).mean()
        axes[0].plot(frame["global_step"], smooth, linewidth=1.8, color=palette[lr], label=f"LR {lr:.0e}")
    axes[0].set_xlabel("Optimiser step")
    axes[0].set_ylabel("Training loss (20-step moving average)")
    axes[0].set_title("E3 training loss")
    axes[0].grid(color="#E3E3E3", linewidth=0.7)
    axes[0].legend(frameon=False)

    for lr, frame in records.groupby("learning_rate"):
        frame = frame.sort_values("epoch")
        axes[1].plot(frame["epoch"], frame["dev_macro_f1"], marker="o", linewidth=2, color=palette[lr], label=f"LR {lr:.0e}")
    axes[1].scatter(
        [selected["selected_epoch"]],
        [selected["dev_macro_f1_at_0.5"]],
        s=180,
        facecolors="none",
        edgecolors="#1B7F5B",
        linewidths=2.2,
        zorder=5,
        label="Selected checkpoint",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("D1 validation Macro F1 at threshold 0.5")
    axes[1].set_title("E3 checkpoint selection (D1 validation only)")
    axes[1].set_xticks(sorted(records["epoch"].unique()))
    axes[1].grid(color="#E3E3E3", linewidth=0.7)
    axes[1].legend(frameon=False)
    figure.suptitle("DistilBERT-base-uncased fine-tuning: learning rate 2e-5 versus 5e-5")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_e3_vs_e1(comparison: pd.DataFrame, path: Path) -> None:
    metrics = [("macro_f1", "Macro F1"), ("attack_recall", "Attack Recall"), ("fpr", "FPR")]
    splits = list(dict.fromkeys(comparison["split_label"]))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    models = list(dict.fromkeys(comparison["pipeline"]))
    colors = {"E1 Combined TF-IDF LR": "#9AA5B1", "E3 DistilBERT": "#3B6EA8"}
    width = 0.36
    positions = np.arange(len(splits))
    for axis, (metric, label) in zip(axes, metrics):
        for offset, model in enumerate(models):
            frame = comparison[comparison["pipeline"].eq(model)].set_index("split_label").reindex(splits)
            values = frame[metric].to_numpy(dtype=float) * 100.0
            axis.bar(positions + (offset - 0.5) * width, values, width, color=colors.get(model, "#666666"), label=model)
        axis.set_xticks(positions, [s.replace(" ", "\n") for s in splits])
        axis.set_ylabel(f"{label} (%)")
        axis.set_title(label)
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("E3 DistilBERT versus E1 Combined baseline, shared threshold 0.5 (operating thresholds are E6's)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_seed_variation(seed_frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    seeds = list(dict.fromkeys(seed_frame["seed"]))
    indexed = seed_frame.set_index("seed").reindex(seeds)
    dev = indexed["dev_macro_f1"].to_numpy(dtype=float) * 100.0
    test = indexed["d1_test_macro_f1"].to_numpy(dtype=float) * 100.0
    positions = np.arange(len(seeds))
    width = 0.35
    bars_dev = axis.bar(positions - width / 2, dev, width, color="#3B6EA8", label="D1 validation Macro F1")
    bars_test = axis.bar(positions + width / 2, test, width, color="#7FA7CC", label="D1 test Macro F1")
    for bars in (bars_dev, bars_test):
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8.5,
            )
    # Zoom on the realised range so the seed spread is visible instead of three identical bars.
    low = min(dev.min(), test.min())
    high = max(dev.max(), test.max())
    pad = max(0.25, (high - low) * 0.35)
    axis.set_ylim(low - pad, high + pad)
    axis.set_xticks(positions, [f"seed {seed}" for seed in seeds])
    axis.set_xlabel("Random seed (frozen learning rate and epoch budget)")
    axis.set_ylabel("Macro F1 (%) at threshold 0.5")
    axis.set_title(
        f"E3 seed spread for the frozen configuration\n"
        f"validation range {dev.max() - dev.min():.2f} pp, test range {test.max() - test.min():.2f} pp",
        fontsize=11,
    )
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=9,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    dry_run = bool(args.dry_run)
    data_dir = project_root / "data/processed"
    output_root = project_root / ("results/e3_dry_run" if dry_run else "results/e3")
    models_root = project_root / ("models/e3_dry_run" if dry_run else "models/e3")
    figures_dir = output_root / "figures"
    handoff_dir = output_root / "handoff"

    epochs = 1 if dry_run else int(config["training"]["epochs"])
    learning_rates = [float(config["training"]["learning_rates"][0])] if dry_run else [
        float(value) for value in config["training"]["learning_rates"]
    ]
    expected_rows = {key: int(value) for key, value in config["evaluation"]["expected_rows"].items()}
    device = resolve_device(config["latency"]["device"])

    print(f"E3 DistilBERT learning-rate comparison{' [DRY RUN]' if dry_run else ''}", flush=True)
    print(f"  project root : {project_root}", flush=True)
    print(f"  device       : {device}", flush=True)

    integrity = verify_data_integrity(project_root, expected_rows)
    if integrity["status"] != "passed":
        raise RuntimeError(f"Data integrity gate failed: {json.dumps(integrity['checks'], indent=2)}")
    print("  data gate    : passed", flush=True)

    train_frame = pd.read_parquet(data_dir / "d1_train.parquet")
    if dry_run:
        train_frame = train_frame.groupby("label", group_keys=False).head(150).reset_index(drop=True)
    evaluation_frames = {
        "d1_validation": pd.read_parquet(data_dir / "d1_validation.parquet"),
        "d1_test": pd.read_parquet(data_dir / "d1_test.parquet"),
        "d2_test": pd.read_parquet(data_dir / "d2_test.parquet"),
        "d3_test": pd.read_parquet(data_dir / "d3_test.parquet"),
        "d4_test": pd.read_parquet(data_dir / "d4_test.parquet"),
    }

    model_config = config["model"]
    training = config["training"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["hub_id"], revision=model_config["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    print(f"  tokenizer    : {type(tokenizer).__name__} pad_id={tokenizer.pad_token_id}", flush=True)

    # ---- truncation accounting -------------------------------------------------------
    truncation_rows = []
    for split, frame in {"d1_train": train_frame, **evaluation_frames}.items():
        stats = token_length_stats(tokenizer, frame["text"].astype(str).tolist(), int(model_config["max_length"]))
        truncation_rows.append({"split": split, **stats})
    truncation_frame = pd.DataFrame(truncation_rows)
    print("  truncation   :", flush=True)
    print(
        truncation_frame[["split", "n", "token_median", "token_max", "truncated_rows", "truncation_rate"]].to_string(
            index=False
        ),
        flush=True,
    )

    # ---- encode once ------------------------------------------------------------------
    train_ids, train_mask = encode_texts(
        tokenizer, train_frame["text"].astype(str).tolist(), int(model_config["max_length"])
    )
    train_labels = train_frame["label"].to_numpy(dtype=int)
    train_dataset = EncodedDataset(train_ids, train_mask, train_labels)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["micro_batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        collate_fn=collate_dynamic(int(tokenizer.pad_token_id)),
        drop_last=False,
        generator=generator,
    )

    dev_frame = evaluation_frames["d1_validation"]
    dev_gold = dev_frame["label"].to_numpy(dtype=int)
    dev_texts = dev_frame["text"].astype(str).tolist()

    # ---- train every learning rate ----------------------------------------------------
    primary_seed = int(config["seed"])
    tuning_records: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    config_summaries: list[dict[str, Any]] = []

    for position, lr in enumerate(learning_rates, start=1):
        print(f"  training LR {lr:.0e} (seed {primary_seed}) ...", flush=True)
        checkpoint_dir = models_root / f"lr{lr:.0e}".replace("e-0", "e-")
        summary, records, extras = train_one_configuration(
            lr=lr,
            seed=primary_seed,
            epochs=epochs,
            config=config,
            device=device,
            tokenizer=tokenizer,
            train_loader=train_loader,
            dev_scores_fn=lambda model: score_texts(model, tokenizer, dev_texts, config, device),
            dev_gold=dev_gold,
            checkpoint_dir=checkpoint_dir,
        )
        tuning_records.extend(records)
        curve_frames.append(pd.DataFrame(extras["curves"]))
        config_summaries.append(summary)

    tuning_frame = pd.DataFrame(tuning_records)

    # ---- checkpoint selection (D1 validation only) -------------------------------------
    ordered = tuning_frame.sort_values(
        ["dev_macro_f1", "epoch", "learning_rate"], ascending=[False, True, True]
    ).reset_index(drop=True)
    chosen = ordered.iloc[0]
    selected_lr = float(chosen["learning_rate"])
    selected_epoch = int(chosen["epoch"])
    selected_summary = next(
        summary for summary in config_summaries if summary["learning_rate"] == selected_lr
    )
    tuning_frame["is_selected"] = (
        tuning_frame["learning_rate"].eq(selected_lr) & tuning_frame["epoch"].eq(selected_epoch)
    )
    print(
        f"  selected     : LR {selected_lr:.0e}, epoch {selected_epoch}, "
        f"dev Macro F1 {chosen['dev_macro_f1']:.4f}",
        flush=True,
    )

    # ---- frozen evaluation ------------------------------------------------------------
    weights = Path(selected_summary["checkpoint_weights"])
    checkpoint_rev = selected_summary["checkpoint_rev"]
    model = AutoModelForSequenceClassification.from_pretrained(
        selected_summary["checkpoint_dir"], num_labels=int(model_config["num_labels"])
    ).to(device)
    model.eval()

    score_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for split, frame in evaluation_frames.items():
        dataset = split.split("_")[0]
        part = "validation" if split.endswith("validation") else "test"
        texts = frame["text"].astype(str).tolist()
        gold = frame["label"].to_numpy(dtype=int)
        scores = score_texts(model, tokenizer, texts, config, device)
        keep = np.isfinite(scores)
        if not keep.all():
            raise RuntimeError(f"Non-finite scores on {split}")

        score_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "split": part,
                    "sample_id": frame["sample_id"].tolist(),
                    "gold": gold,
                    "score_attack": scores,
                    "model_id": model_config["model_id"],
                    "checkpoint_rev": checkpoint_rev,
                    "seed": primary_seed,
                    "transform_id": "clean",
                    "parent_id": pd.Series([None] * len(frame), dtype="string"),
                }
            )
        )

        if split == "d1_validation":
            thresholds = select_thresholds(gold, scores)
        for policy, values in (thresholds.items() if thresholds else [("default_0.5", {"threshold": 0.5})]):
            threshold = float(values["threshold"])
            metrics_for_split = metric_record(gold, scores, threshold)
            if dataset == "d4":
                # D4 is benign-only: report counts and FPR, never balanced F1, recall or AUROC.
                metrics_for_split = {
                    "macro_f1": None,
                    "attack_recall": None,
                    "fpr": metrics_for_split["fpr"],
                    "auroc": None,
                    "tn": metrics_for_split["tn"],
                    "fp": metrics_for_split["fp"],
                    "fn": metrics_for_split["fn"],
                    "tp": metrics_for_split["tp"],
                    "n": metrics_for_split["n"],
                }
            record = {
                "pipeline": "e3_distilbert",
                "model_id": model_config["model_id"],
                "learning_rate": selected_lr,
                "checkpoint_epoch": selected_epoch,
                "dataset": dataset,
                "split": part,
                "threshold_policy": policy,
                "threshold": threshold,
                **metrics_for_split,
            }
            metric_rows.append(record)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "split": part,
                        "sample_id": frame["sample_id"].tolist(),
                        "gold": gold,
                        "score_attack": scores,
                        "score_type": config["evaluation"]["score_type"],
                        "threshold_policy": policy,
                        "threshold": threshold,
                        "pred": (scores >= threshold).astype(int),
                        "model_id": model_config["model_id"],
                        "checkpoint_rev": checkpoint_rev,
                        "seed": primary_seed,
                        "transform_id": "clean",
                        "parent_id": pd.Series([None] * len(frame), dtype="string"),
                    }
                )
            )
        print(f"  scored       : {split} ({len(frame):,} rows)", flush=True)

    inference_peak_vram = (
        float(torch.cuda.max_memory_allocated(device)) / (1024**2) if device.type == "cuda" else None
    )

    # ---- latency and size --------------------------------------------------------------
    latency_rows = []
    latency_targets = [device] if device.type == "cpu" else [device, torch.device("cpu")]
    for target in latency_targets:
        probe = model if target == device else model.to(target)
        latency = measure_latency(probe, tokenizer, dev_texts, config, target)
        latency_rows.append({"model_id": model_config["model_id"], "component": "full_pipeline", **latency})
        if target != device:
            model.to(device)
    latency_frame = pd.DataFrame(latency_rows)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    size_record = {
        "model_id": model_config["model_id"],
        "n_parameters": parameter_count,
        "dtype": "float32",
        "param_bytes": parameter_count * 4,
        "checkpoint_bytes": int(selected_summary["checkpoint_bytes"]),
        "checkpoint_dir": selected_summary["checkpoint_dir"],
        "weights_sha256": checkpoint_rev,
        "n_features_or_vocab": int(getattr(model.config, "vocab_size", 0)),
    }

    # ---- optional seed stability -------------------------------------------------------
    seed_frames: list[pd.DataFrame] = []
    extra_seeds = [int(value) for value in config["seed_stability"]["seeds"]] if config["seed_stability"]["enabled"] else []
    extra_seeds = [seed for seed in extra_seeds if seed != primary_seed]
    if not dry_run:
        seed_frames.append(
            pd.DataFrame(
                [
                    {
                        "seed": primary_seed,
                        "learning_rate": selected_lr,
                        "epochs": selected_epoch,
                        "dev_macro_f1": float(chosen["dev_macro_f1"]),
                        "dev_attack_recall": float(chosen["dev_attack_recall"]),
                        "dev_fpr": float(chosen["dev_fpr"]),
                        "d1_test_macro_f1": float(
                            next(row for row in metric_rows if row["split"] == "test" and row["dataset"] == "d1" and row["threshold_policy"] == "default_0.5")["macro_f1"]
                        ),
                        "role": "primary",
                    }
                ]
            )
        )
        for seed in extra_seeds:
            print(f"  seed study   : retraining frozen configuration with seed {seed}", flush=True)
            seed_summary, seed_records, _ = train_one_configuration(
                lr=selected_lr,
                seed=seed,
                epochs=selected_epoch,
                config=config,
                device=device,
                tokenizer=tokenizer,
                train_loader=train_loader,
                dev_scores_fn=lambda model: score_texts(model, tokenizer, dev_texts, config, device),
                dev_gold=dev_gold,
                checkpoint_dir=models_root / f"seed{seed}_lr{selected_lr:.0e}".replace("e-0", "e-"),
                verbose=False,
            )
            seed_model = AutoModelForSequenceClassification.from_pretrained(
                seed_summary["checkpoint_dir"], num_labels=int(model_config["num_labels"])
            ).to(device)
            seed_model.eval()
            seed_dev_scores = score_texts(seed_model, tokenizer, dev_texts, config, device)
            seed_test_scores = score_texts(
                seed_model,
                tokenizer,
                evaluation_frames["d1_test"]["text"].astype(str).tolist(),
                config,
                device,
            )
            seed_dev = metric_record(dev_gold, seed_dev_scores, 0.5)
            seed_test = metric_record(
                evaluation_frames["d1_test"]["label"].to_numpy(dtype=int), seed_test_scores, 0.5
            )
            seed_frames.append(
                pd.DataFrame(
                    [
                        {
                            "seed": seed,
                            "learning_rate": selected_lr,
                            "epochs": selected_epoch,
                            "dev_macro_f1": seed_dev["macro_f1"],
                            "dev_attack_recall": seed_dev["attack_recall"],
                            "dev_fpr": seed_dev["fpr"],
                            "d1_test_macro_f1": seed_test["macro_f1"],
                            "role": "supplementary_seed",
                        }
                    ]
                )
            )
            del seed_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    seed_frame = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()

    # ---- write tables -------------------------------------------------------------------
    output_root.mkdir(parents=True, exist_ok=True)
    curve_frame = pd.concat(curve_frames, ignore_index=True)
    tuning_sorted = tuning_frame.sort_values(["learning_rate", "epoch"]).reset_index(drop=True)
    atomic_csv(tuning_sorted, output_root / "tuning.csv")
    atomic_csv(curve_frame, output_root / "train_curves.csv")
    atomic_csv(truncation_frame, output_root / "truncation_stats.csv")
    atomic_csv(latency_frame, output_root / "latency.csv")
    atomic_csv(pd.DataFrame([size_record]), output_root / "model_size.csv")
    if not seed_frame.empty:
        atomic_csv(seed_frame, output_root / "seed_variation.csv")
    metric_frame = pd.DataFrame(metric_rows).sort_values(["dataset", "split", "threshold_policy"])
    atomic_csv(metric_frame, output_root / "metrics.csv")

    scores_frame = pd.concat(score_frames, ignore_index=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    atomic_parquet(scores_frame, output_root / "predictions" / "distilbert_scores.parquet")
    atomic_parquet(predictions_frame, output_root / "predictions" / "distilbert_predictions.parquet")

    handoff_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(scores_frame, handoff_dir / "distilbert_scores.parquet")
    atomic_csv(tuning_sorted, handoff_dir / "tuning.csv")
    (handoff_dir / "checkpoint_path.txt").write_text(
        "\n".join(
            [
                f"checkpoint_dir   = {Path(selected_summary['checkpoint_dir']).relative_to(project_root).as_posix()}",
                f"checkpoint_file  = {weights.relative_to(project_root).as_posix()}",
                f"checkpoint_rev   = {checkpoint_rev}",
                f"model_hub_id     = {model_config['hub_id']}",
                f"model_revision   = {model_config['revision']}",
                f"tokenizer        = AutoTokenizer.from_pretrained({model_config['hub_id']!r}, revision={model_config['revision']!r})",
                f"learning_rate    = {selected_lr}",
                f"checkpoint_epoch = {selected_epoch}",
                f"seed             = {primary_seed}",
                "",
                "Load with:",
                "  from transformers import AutoTokenizer, AutoModelForSequenceClassification",
                f"  tok = AutoTokenizer.from_pretrained('{Path(selected_summary['checkpoint_dir']).relative_to(project_root).as_posix()}')",
                f"  mdl = AutoModelForSequenceClassification.from_pretrained('{Path(selected_summary['checkpoint_dir']).relative_to(project_root).as_posix()}')",
                "  enc = tok(text, max_length=256, truncation=True, return_tensors='pt')",
                "  score_attack = torch.softmax(mdl(**enc).logits, -1)[0, 1].item()   # uncalibrated",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # ---- E1 comparison at the shared default threshold ----------------------------------
    comparison_rows: list[dict[str, Any]] = []
    split_labels = {
        "d1_validation": "D1 dev",
        "d1_test": "D1 test",
        "d2_test": "D2 test",
        "d3_test": "D3 test",
    }
    e1_handoff = project_root / "results/e1/handoff/combined_scores.parquet"
    e1_scores = pd.read_parquet(e1_handoff) if e1_handoff.exists() else None
    for split, label in split_labels.items():
        dataset, part = split.split("_", 1)
        part = "validation" if part == "validation" else "test"
        e3_row = next(
            row
            for row in metric_rows
            if row["dataset"] == dataset and row["split"] == part and row["threshold_policy"] == "default_0.5"
        )
        comparison_rows.append(
            {
                "pipeline": "E3 DistilBERT",
                "split_label": label,
                "macro_f1": e3_row["macro_f1"],
                "attack_recall": e3_row["attack_recall"],
                "fpr": e3_row["fpr"],
            }
        )
        if e1_scores is not None:
            subset = e1_scores[e1_scores["dataset"].eq(dataset) & e1_scores["split"].eq(part)]
            baseline = metric_record(subset["gold"].to_numpy(dtype=int), subset["score_attack"].to_numpy(dtype=float), 0.5)
            comparison_rows.append(
                {
                    "pipeline": "E1 Combined TF-IDF LR",
                    "split_label": label,
                    "macro_f1": baseline["macro_f1"],
                    "attack_recall": baseline["attack_recall"],
                    "fpr": baseline["fpr"],
                }
            )
    comparison_frame = pd.DataFrame(comparison_rows)
    atomic_csv(comparison_frame, output_root / "e3_vs_e1_at_0.5.csv")

    # ---- figures --------------------------------------------------------------------------
    plot_learning_curves(curve_frame, tuning_sorted, selected_summary, figures_dir / "fig1_learning_curves.png")
    if e1_scores is not None:
        plot_e3_vs_e1(comparison_frame, figures_dir / "fig2_e3_vs_e1.png")
    if not seed_frame.empty:
        plot_seed_variation(seed_frame, figures_dir / "fig3_seed_variation.png")

    # ---- deliverable metadata -------------------------------------------------------------
    selected_config = {
        "experiment_id": config["experiment_id"],
        "model_id": model_config["model_id"],
        "hub_id": model_config["hub_id"],
        "model_revision": model_config["revision"],
        "tokenizer_revision": model_config["revision"],
        "architecture": model_config["architecture"],
        "max_length": int(model_config["max_length"]),
        "selected_learning_rate": selected_lr,
        "selected_checkpoint_epoch": selected_epoch,
        "checkpoint_dir": selected_summary["checkpoint_dir"],
        "checkpoint_rev": checkpoint_rev,
        "checkpoint_epochs_trained": epochs,
        "seed": primary_seed,
        "selection": config["selection"],
        "dev_metrics_at_0.5": {
            key: float(chosen[key])
            for key in ("dev_macro_f1", "dev_attack_recall", "dev_fpr", "dev_auroc")
        },
        "candidate_grid": [
            {
                "learning_rate": float(row["learning_rate"]),
                "epoch": int(row["epoch"]),
                "dev_macro_f1": float(row["dev_macro_f1"]),
                "selected": bool(row["is_selected"]),
            }
            for _, row in tuning_sorted.iterrows()
        ],
        "training_hyperparameters": training,
        "batch_accounting": {
            "micro_batch_size": int(training["micro_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "effective_batch_size": int(training["effective_batch_size"]),
            "steps_per_epoch": int(tuning_sorted["steps"].iloc[0]),
        },
        "truncation": {
            row["split"]: {
                "n": int(row["n"]),
                "truncated_rows": int(row["truncated_rows"]),
                "truncation_rate": float(row["truncation_rate"]),
                "token_max": int(row["token_max"]),
            }
            for row in truncation_frame.to_dict("records")
        },
        "cost": {
            "n_parameters": parameter_count,
            "checkpoint_bytes": int(selected_summary["checkpoint_bytes"]),
            "batch1_latency_ms_median": float(
                latency_frame.loc[latency_frame["device"].eq(str(device)), "median_ms"].iloc[0]
            ),
            "batch1_latency_ms_p95": float(
                latency_frame.loc[latency_frame["device"].eq(str(device)), "p95_ms"].iloc[0]
            ),
            "inference_peak_vram_mb": inference_peak_vram,
        },
        "provisional_dev_thresholds": thresholds,
        "provisional_threshold_note": "Computed on D1 validation for the demo app only. E6 owns the authoritative tF1 and t5 thresholds.",
        "seed_stability": seed_frame.to_dict("records") if not seed_frame.empty else [],
    }
    atomic_json(selected_config, output_root / "selected_config.json")
    atomic_json(selected_config, handoff_dir / "selected_config.json")

    total_training_seconds = float(tuning_sorted["epoch_seconds"].sum())
    run_metadata = {
        "status": "complete" if not dry_run else "dry_run",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "data_integrity": integrity,
        "data_manifest_sha256": sha256_file(project_root / "data/audit/manifest.json"),
        "selection_rule": config["selection"],
        "selected_model": {
            "model_id": model_config["model_id"],
            "learning_rate": selected_lr,
            "checkpoint_epoch": selected_epoch,
            "checkpoint_rev": checkpoint_rev,
            "seed": primary_seed,
        },
        "scores": {
            "file": "results/e3/handoff/distilbert_scores.parquet",
            "rows": int(len(scores_frame)),
            "rows_per_split": {
                f"{dataset}/{part}": int(count)
                for (dataset, part), count in scores_frame.groupby(["dataset", "split"]).size().items()
            },
            "score_type": config["evaluation"]["score_type"],
            "higher_means": "more attack-like",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only",
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "gpu_total_memory_mb": (
                float(torch.cuda.get_device_properties(0).total_memory) / (1024**2) if torch.cuda.is_available() else None
            ),
            "device_used": str(device),
        },
        "timing": {
            "total_training_seconds": total_training_seconds,
            "epochs_trained": len(tuning_sorted),
            "seconds_per_epoch": total_training_seconds / max(1, len(tuning_sorted)),
        },
        "git": "n/a - the handoff package is not a git repository; provenance is carried by file hashes in this metadata",
        "reproduction": {
            "command": "python experiments/e3/run_e3.py",
            "verify_command": "python experiments/e3/verify_e3.py",
            "app_command": "streamlit run app/promptshield_app.py",
        },
        "notes": [
            "Vocabulary, model weights and all optimisation decisions use cleaned D1 train only.",
            "The checkpoint was selected on D1 validation Macro F1 at threshold 0.5; ties prefer the earlier epoch, then the smaller learning rate.",
            "D1/D2/D3 test and D4 were scored once, after the checkpoint was frozen; no test-informed revision was performed.",
            "D4 contains benign texts only, so it is reported as false alarms in E6, not as a balanced classification score.",
            "Class-1 scores are uncalibrated softmax outputs, not probabilities of maliciousness.",
            "Learning-rate candidates 2e-5 and 5e-5 use the same tokenizer, batch, schedule, epoch budget, seed and evaluation code; only the learning rate differs.",
            "Extra seeds are reported as spread for the frozen configuration and never replace the primary seed 6405.",
        ],
    }
    atomic_json(run_metadata, output_root / "run_metadata.json")
    atomic_json(run_metadata, handoff_dir / "run_metadata.json")

    print("", flush=True)
    print("E3 complete", flush=True)
    print(
        tuning_sorted[
            ["learning_rate", "epoch", "train_loss_epoch", "dev_macro_f1", "dev_attack_recall", "dev_fpr", "is_selected"]
        ].to_string(index=False),
        flush=True,
    )
    print("", flush=True)
    print(
        metric_frame[metric_frame["threshold_policy"].eq("default_0.5")][
            ["dataset", "split", "macro_f1", "attack_recall", "fpr", "tp", "fp", "tn", "fn"]
        ].to_string(index=False),
        flush=True,
    )
    print("", flush=True)
    print(latency_frame[["device", "median_ms", "p95_ms"]].to_string(index=False), flush=True)
    if not seed_frame.empty:
        print("", flush=True)
        print(seed_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
