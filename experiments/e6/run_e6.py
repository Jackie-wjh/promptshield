"""E6 - operating threshold selection and false-alarm analysis (M6).

Reads every member's frozen raw-score handoff file (results/e*/handoff/*_scores.parquet),
selects tF1 and t5 on D1 Validation only with E1's rule, freezes them, then evaluates D1/D2/D3
Test and the benign-only D4 NotInject set. Writes the files the README asks E6 to return:

    results/e6/thresholds.csv
    results/e6/metrics.csv
    results/e6/predictions.parquet
    results/e6/notinject_false_alarms.csv
    results/e6/run_metadata.json

plus supplementary tables, figures and summary.md. Run from the promptshield/ root:

    python experiments/e6/run_e6.py
    python experiments/e6/run_e6.py --verify-e1-joblib      # also re-score D1 Validation with the E1 joblibs
    python experiments/e6/run_e6.py --source e4=path/to/roberta_scores.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs/e6.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = CONFIG["seed"]
N_BOOT = CONFIG["uncertainty"]["bootstrap_resamples"]
Z95 = 1.959963984540054
T5_MAX_FPR = CONFIG["thresholds"]["t5_max_validation_fpr"]
POLICIES = tuple(CONFIG["thresholds"]["policies"])
MAIN_EXPERIMENTS = tuple(CONFIG["main_pipeline_experiments"])  # the six main pipelines; E5 rows are supplementary


def check_config_matches_e1() -> dict[str, Any]:
    """E6 must apply exactly the threshold rule frozen in configs/e1.json."""
    e1 = json.loads((ROOT / "configs/e1.json").read_text(encoding="utf-8"))["thresholds"]
    mine = CONFIG["thresholds"]
    pairs = {"t5_max_validation_fpr": (e1["t5_max_validation_fpr"], mine["t5_max_validation_fpr"]),
             "tF1_tie_break": (e1["tF1_tie_break"], mine["tF1_tie_break"]),
             "t5_tie_break": (e1["t5_tie_break"], mine["t5_tie_break"]),
             "prediction_rule": (e1["prediction_rule"], mine["prediction_rule"])}
    bad = {k: v for k, v in pairs.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"[e6] configs/e6.json disagrees with configs/e1.json thresholds: {bad}")
    return {k: v[1] for k, v in pairs.items()}
SCHEMA = ["dataset", "split", "sample_id", "gold", "score_attack",
          "model_id", "checkpoint_rev", "seed", "transform_id", "parent_id"]
SPLITS = {  # (dataset, split) -> processed file, expected rows, required?
    ("d1", "validation"): ("d1_validation", 1985, True),
    ("d1", "test"): ("d1_test", 2101, True),
    ("d2", "test"): ("d2_test", 116, True),
    ("d3", "test"): ("d3_test", 545, True),
    ("d4", "test"): ("d4_test", 339, False),
}
EVAL_SPLITS = [("d1", "validation"), ("d1", "test"), ("d2", "test"), ("d3", "test")]
DISPLAY = {
    "e1_word_tfidf_lr": "E1 Word TF-IDF+LR",
    "e1_char_tfidf_lr": "E1 Char TF-IDF+LR",
    "e1_combined_tfidf_lr": "E1 Combined TF-IDF+LR",
    "bilstm_random": "E2 BiLSTM (random init)",
    "e2_bilstm_random": "E2 BiLSTM (random init)",
    "e3_distilbert_base_uncased": "E3 DistilBERT",
    "e4_roberta_base": "E4 RoBERTa",
    "e5_clean_combined_tfidf_lr": "E5 Clean (= E1 Combined)",
    "e5_augmented_combined_tfidf_lr": "E5 Typo-augmented",
}


# --------------------------------------------------------------------------- utilities

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    tmp.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write(path, lambda p: frame.to_csv(p, index=False, lineterminator="\n"))


def write_json(value: Any, path: Path) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8", newline="\n"))


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def clopper_pearson(k: int, n: int) -> tuple[float | None, float | None]:
    from scipy.stats import beta
    if n == 0:
        return None, None
    low = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return low, high


def macro_f1_from_counts(tp, fp, tn, fn):
    """Vectorised sklearn macro F1 (zero_division=0) for binary counts."""
    tp, fp, tn, fn = (np.asarray(x, dtype=float) for x in (tp, fp, tn, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_attack = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_benign = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1_attack + f1_benign) / 2


# --------------------------------------------------------------------------- inputs

def experiment_of(path: Path, override: str | None = None) -> str:
    if override:
        return override
    match = re.search(r"results/(e\d+)/handoff", path.as_posix())
    return match.group(1) if match else "unknown"


def discover(extra: list[str]) -> list[tuple[str, Path]]:
    found = [(experiment_of(p), p) for p in sorted(ROOT.glob("results/e*/handoff/*_scores.parquet"))
             if not p.is_relative_to(ROOT / "results/e6")]
    for item in extra:
        name, _, raw = item.partition("=")
        path = Path(raw).expanduser().resolve()
        found.append((name, path))
    return found


def load_processed() -> dict[tuple[str, str], pd.DataFrame]:
    return {key: pd.read_parquet(ROOT / f"data/processed/{fname}.parquet")
            for key, (fname, _, _) in SPLITS.items()}


def validate(frame: pd.DataFrame, processed) -> tuple[list[str], list[str]]:
    problems, warnings = [], []
    missing = [c for c in SCHEMA if c not in frame.columns]
    if missing:
        return [f"missing columns {missing}"], warnings
    if frame["model_id"].nunique() != 1:
        problems.append(f"expected one model_id, found {frame['model_id'].unique().tolist()}")
    scores = frame["score_attack"].astype(float)
    if not np.isfinite(scores).all():
        problems.append(f"{int((~np.isfinite(scores)).sum())} non-finite scores")
    if not set(frame["gold"].unique()) <= {0, 1}:
        problems.append("gold is not 0/1")
    dups = int(frame.duplicated(["model_id", "dataset", "split", "sample_id", "transform_id"]).sum())
    if dups:
        problems.append(f"{dups} duplicate (model_id, dataset, split, sample_id, transform_id) rows")
    if set(frame["transform_id"].astype(str)) != {"clean"}:
        problems.append(f"non-clean transform_id values {sorted(set(frame['transform_id'].astype(str)))}")
    for key, (fname, n_expected, required) in SPLITS.items():
        rows = frame[(frame.dataset == key[0]) & (frame.split == key[1])]
        if rows.empty:
            (problems if required else warnings).append(f"{fname}: no rows")
            continue
        if len(rows) != n_expected:
            problems.append(f"{fname}: {len(rows)} rows, expected {n_expected}")
        ref = processed[key][["sample_id", "label"]]
        if set(rows.sample_id) != set(ref.sample_id):
            problems.append(f"{fname}: sample_id set differs from data/processed")
            continue
        merged = rows.merge(ref, on="sample_id")
        bad = int((merged.gold.astype(int) != merged.label.astype(int)).sum())
        if bad:
            problems.append(f"{fname}: {bad} gold labels differ from data/processed")
    rev = str(frame["checkpoint_rev"].iloc[0])
    if not re.fullmatch(r"[0-9a-f]{64}", rev):
        warnings.append(f"checkpoint_rev is not a SHA-256 ({rev!r})")
    if frame["parent_id"].notna().any():
        warnings.append("parent_id is not null on clean rows")
    return problems, warnings


# --------------------------------------------------------------------------- threshold selection

def sweep(gold: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Every candidate threshold of E1's rule (unique scores + one above the max), exact counts."""
    unique = np.unique(scores.astype(float))
    candidates = np.concatenate([unique, [np.nextafter(unique.max(), np.inf)]])
    pos = np.sort(scores[gold == 1])
    neg = np.sort(scores[gold == 0])
    tp = len(pos) - np.searchsorted(pos, candidates, side="left")
    fp = len(neg) - np.searchsorted(neg, candidates, side="left")
    fn, tn = len(pos) - tp, len(neg) - fp
    return pd.DataFrame({
        "threshold": candidates, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "macro_f1": macro_f1_from_counts(tp, fp, tn, fn),
        "attack_recall": tp / len(pos), "fpr": fp / len(neg),
    })


def select(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    best_f1 = table["macro_f1"].max()
    tied_f1 = table[table["macro_f1"] == best_f1]
    tf1 = tied_f1.loc[tied_f1["threshold"].idxmax()]
    eligible = table[table["fpr"] <= T5_MAX_FPR]
    best_recall = eligible["attack_recall"].max()
    tied_t5 = eligible[eligible["attack_recall"] == best_recall]
    t5 = tied_t5.loc[tied_t5["threshold"].idxmax()]
    return {
        "tF1": {"threshold": float(tf1.threshold), "n_tied": len(tied_f1),
                "tie_low": float(tied_f1.threshold.min()), "tie_high": float(tied_f1.threshold.max())},
        "t5": {"threshold": float(t5.threshold), "n_tied": len(tied_t5),
               "tie_low": float(tied_t5.threshold.min()), "tie_high": float(tied_t5.threshold.max())},
        "default_0.5": {"threshold": 0.5, "n_tied": None, "tie_low": None, "tie_high": None},
    }


# --------------------------------------------------------------------------- metrics

def metric_row(gold: np.ndarray, scores: np.ndarray, threshold: float, rng: np.random.Generator | None) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    pred = (scores >= threshold).astype(int)
    pos, neg = gold == 1, gold == 0
    tp, fn = int(pred[pos].sum()), int((1 - pred[pos]).sum())
    fp, tn = int(pred[neg].sum()), int((1 - pred[neg]).sum())
    n_pos, n_neg = tp + fn, fp + tn
    both = n_pos > 0 and n_neg > 0
    row: dict[str, Any] = {
        "n": len(gold), "n_attack": n_pos, "n_benign": n_neg, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "macro_f1": float(macro_f1_from_counts(tp, fp, tn, fn)) if both else None,
        "attack_recall": tp / n_pos if n_pos else None,
        "fpr": fp / n_neg if n_neg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "auroc": float(roc_auc_score(gold, scores)) if both else None,
    }
    row["attack_recall_ci_low"], row["attack_recall_ci_high"] = wilson(tp, n_pos)
    row["fpr_ci_low"], row["fpr_ci_high"] = wilson(fp, n_neg)
    row["macro_f1_ci_low"] = row["macro_f1_ci_high"] = None
    if both and rng is not None:
        pred_pos, pred_neg = pred[pos], pred[neg]
        boot_tp = pred_pos[rng.integers(0, n_pos, (N_BOOT, n_pos))].sum(axis=1)
        boot_fp = pred_neg[rng.integers(0, n_neg, (N_BOOT, n_neg))].sum(axis=1)
        boot = macro_f1_from_counts(boot_tp, boot_fp, n_neg - boot_fp, n_pos - boot_tp)
        row["macro_f1_ci_low"], row["macro_f1_ci_high"] = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return row


def d4_rows(scores: pd.DataFrame, threshold: float, d4: pd.DataFrame) -> list[dict[str, Any]]:
    frame = scores.merge(d4[["sample_id", "subset", "trigger_count", "category"]], on="sample_id")
    pred = frame["score_attack"].to_numpy(float) >= threshold
    groups = [("overall", "all", np.ones(len(frame), bool))]
    for count in sorted(frame["trigger_count"].unique()):
        groups.append(("trigger_count", str(count), (frame["trigger_count"] == count).to_numpy()))
    for category in sorted(frame["category"].unique()):
        groups.append(("category", category, (frame["category"] == category).to_numpy()))
    # report Table 9 rows; "Non-Multilingual" is a source category, not validated English identification
    is_multi = (frame["category"] == "Multilingual").to_numpy()
    groups.append(("multilingual", "Non-Multilingual", ~is_multi))
    groups.append(("multilingual", "Multilingual", is_multi))
    out = []
    for group_by, group, mask in groups:
        n, fp = int(mask.sum()), int(pred[mask].sum())
        w_lo, w_hi = wilson(fp, n)
        cp_lo, cp_hi = clopper_pearson(fp, n)
        out.append({"group_by": group_by, "group": group, "fp": fp, "n": n, "fpr": fp / n,
                    "ci_low": w_lo, "ci_high": w_hi, "ci_method": "wilson_95",
                    "cp_ci_low": cp_lo, "cp_ci_high": cp_hi})
    return out


def d3_rows(scores: pd.DataFrame, threshold: float, d3: pd.DataFrame) -> list[dict[str, Any]]:
    frame = scores.merge(d3[["sample_id", "attack_family", "technique"]], on="sample_id")
    frame["pred"] = (frame["score_attack"] >= threshold).astype(int)
    out = []
    for family, group in frame.groupby("attack_family"):
        is_benign = family == "benign"
        k = int(group["pred"].sum())
        lo, hi = wilson(k, len(group))
        out.append({"attack_family": family, "n": len(group),
                    "metric": "fpr" if is_benign else "attack_recall",
                    "flagged": k, "rate": k / len(group), "ci_low": lo, "ci_high": hi})
    return out


# --------------------------------------------------------------------------- E1 cross-checks

def check_against_e1(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """E6's vectorised search must equal E1's exhaustive select_thresholds exactly."""
    from experiments.e6.e1_reference import select_thresholds
    report: dict[str, Any] = {}
    for model_id, info in models.items():
        val = info["frame"][(info["frame"].dataset == "d1") & (info["frame"].split == "validation")]
        reference = select_thresholds(val["gold"].to_numpy(int), val["score_attack"].to_numpy(float))
        report[model_id] = {p: {"e6": info["thresholds"][p]["threshold"], "e1_rule": reference[p]["threshold"],
                                "equal": info["thresholds"][p]["threshold"] == reference[p]["threshold"]}
                            for p in ("tF1", "t5")}
    ref_path = ROOT / "results/e1/reference_only/thresholds.csv"
    if ref_path.is_file():
        ref = pd.read_csv(ref_path)
        for _, row in ref[ref["policy"].isin(["tF1", "t5"])].iterrows():
            if row["model_id"] in report:
                entry = report[row["model_id"]][row["policy"]]
                entry["e1_reference_only"] = float(row["threshold"])
                entry["abs_diff_vs_reference_only"] = abs(entry["e6"] - float(row["threshold"]))
    return report


def verify_e1_joblib(models: dict[str, dict[str, Any]], processed) -> dict[str, Any]:
    import joblib
    from experiments.e6.e1_reference import attack_scores
    val = processed[("d1", "validation")]
    report = {}
    for name in ("word", "char", "combined"):
        model_id = f"e1_{name}_tfidf_lr"
        if model_id not in models:
            continue
        path = ROOT / f"models/e1/{name}.joblib"
        rescored = pd.DataFrame({"sample_id": val["sample_id"],
                                 "rescored": attack_scores(joblib.load(path), val["text"].astype(str))})
        frame = models[model_id]["frame"]
        merged = rescored.merge(frame[(frame.dataset == "d1") & (frame.split == "validation")], on="sample_id")
        diff = (merged["rescored"] - merged["score_attack"]).abs()
        flips = {p: int(((merged["rescored"] >= t["threshold"]) != (merged["score_attack"] >= t["threshold"])).sum())
                 for p, t in models[model_id]["thresholds"].items()}
        # D4 texts with no in-vocabulary feature get the intercept-only score; record how thresholds treat them
        model = joblib.load(path)
        d4 = processed[("d4", "test")]
        features = model[:-1].transform(d4["text"].astype(str))
        empty = np.asarray((features != 0).sum(axis=1)).ravel() == 0
        intercept_score = float(1 / (1 + np.exp(-model.named_steps["classifier"].intercept_[0])))
        report[model_id] = {
            "rows": len(merged), "max_abs_diff": float(diff.max()), "prediction_flips": flips,
            "joblib_sha256": sha256_file(path),
            "d4_zero_feature_texts": {
                "total": int(empty.sum()),
                "multilingual": int((empty & (d4["category"] == "Multilingual").to_numpy()).sum()),
                "intercept_only_score": intercept_score,
                "flagged_at": {p: bool(intercept_score >= t["threshold"]) for p, t in models[model_id]["thresholds"].items()},
            },
        }
    return report


# --------------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=[],
                        help="extra handoff file as experiment=path (e.g. e4=/path/roberta_scores.parquet)")
    parser.add_argument("--verify-e1-joblib", action="store_true",
                        help="re-score D1 Validation with models/e1/*.joblib (needs scikit-learn 1.9.1)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results/e6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    out: Path = args.out
    processed = load_processed()

    inputs, excluded = [], []
    models: dict[str, dict[str, Any]] = {}
    for experiment, path in discover(args.source):
        frame = pd.read_parquet(path)
        problems, warnings = validate(frame, processed)
        record = {"experiment": experiment, "path": (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix(),
                  "sha256": sha256_file(path), "rows": len(frame),
                  "split_rows": ({f"{d}_{s}": int(n) for (d, s), n in frame.groupby(["dataset", "split"]).size().items()}
                                 if {"dataset", "split"} <= set(frame.columns) else {}),
                  "model_id": frame["model_id"].iloc[0] if "model_id" in frame and len(frame) else None,
                  "problems": problems, "warnings": warnings}
        inputs.append(record)
        if problems:
            excluded.append(record)
            print(f"[e6] EXCLUDED {record['path']}: {problems}", flush=True)
            continue
        model_id = record["model_id"]
        if model_id in models:
            raise SystemExit(f"[e6] duplicate model_id {model_id} in {path} and {models[model_id]['path']}")
        frame = frame.copy()
        frame["gold"] = frame["gold"].astype(int)
        frame["score_attack"] = frame["score_attack"].astype(float)
        models[model_id] = {"experiment": experiment, "path": record["path"], "frame": frame, "warnings": warnings}
        for w in warnings:
            print(f"[e6] warning {model_id}: {w}", flush=True)

    known = list(DISPLAY)
    order = sorted(models, key=lambda m: (models[m]["experiment"], known.index(m) if m in known else len(known), m))
    print(f"[e6] {len(order)} models: {order}", flush=True)

    # ---- Phase A: thresholds from D1 Validation only
    threshold_rows, sweeps = [], []
    for model_id in order:
        info = models[model_id]
        val = info["frame"][(info["frame"].dataset == "d1") & (info["frame"].split == "validation")]
        table = sweep(val["gold"].to_numpy(int), val["score_attack"].to_numpy(float))
        info["thresholds"] = select(table)
        sweeps.append(table.assign(model_id=model_id))
        for policy in POLICIES:
            sel = info["thresholds"][policy]
            m = metric_row(val["gold"].to_numpy(int), val["score_attack"].to_numpy(float), sel["threshold"], None)
            threshold_rows.append({
                "experiment": info["experiment"], "model_id": model_id, "display_name": DISPLAY.get(model_id, model_id),
                "is_main_pipeline": info["experiment"] in MAIN_EXPERIMENTS,
                "threshold_policy": policy, "threshold": sel["threshold"],
                "selection_data": "d1_validation",
                "selection_rule": {"tF1": "max Macro F1", "t5": f"max Attack Recall s.t. FPR <= {T5_MAX_FPR}",
                                   "default_0.5": "fixed 0.5"}[policy],
                "tie_break": "higher_threshold" if policy != "default_0.5" else None,
                "n_tied_candidates": sel["n_tied"], "tie_threshold_low": sel["tie_low"], "tie_threshold_high": sel["tie_high"],
                "n_candidates": len(table),
                "val_macro_f1": m["macro_f1"], "val_attack_recall": m["attack_recall"], "val_fpr": m["fpr"],
                "val_tp": m["tp"], "val_fp": m["fp"], "val_tn": m["tn"], "val_fn": m["fn"],
                "checkpoint_rev": str(info["frame"]["checkpoint_rev"].iloc[0]),
            })
    thresholds = pd.DataFrame(threshold_rows)
    config_check = check_config_matches_e1()
    e1_check = check_against_e1(models)
    mismatched = [m for m, v in e1_check.items() if not all(x["equal"] for x in v.values())]
    if mismatched:
        raise SystemExit(f"[e6] vectorised threshold search disagrees with E1's select_thresholds for {mismatched}")
    joblib_check = verify_e1_joblib(models, processed) if args.verify_e1_joblib else None

    # ---- Phase B: frozen thresholds on D1/D2/D3 Test (plus validation for reference) and D4
    metric_rows, d4_table, d3_table, prediction_parts = [], [], [], []
    for model_id in order:
        info = models[model_id]
        frame = info["frame"]
        for policy in POLICIES:
            threshold = info["thresholds"][policy]["threshold"]
            base = {"experiment": info["experiment"], "model_id": model_id,
                    "display_name": DISPLAY.get(model_id, model_id),
                    "is_main_pipeline": info["experiment"] in MAIN_EXPERIMENTS,
                    "threshold_policy": policy, "threshold": threshold}
            for dataset, split in EVAL_SPLITS:
                rows = frame[(frame.dataset == dataset) & (frame.split == split)]
                # one bootstrap stream per (model, policy, split): adding a model never changes other rows' CIs
                rng = np.random.default_rng([SEED, zlib.crc32(f"{model_id}|{policy}|{dataset}|{split}".encode())])
                metric_rows.append({**base, "dataset": dataset, "split": split,
                                    **metric_row(rows["gold"].to_numpy(int), rows["score_attack"].to_numpy(float),
                                                 threshold, rng)})
            d4 = frame[frame.dataset == "d4"]
            if d4.empty:
                d4_table.append({**base, "group_by": "overall", "group": "all", "fp": None, "n": 0, "fpr": None,
                                 "note": "no D4 scores in handoff file"})
            else:
                d4_table.extend({**base, **r} for r in d4_rows(d4, threshold, processed[("d4", "test")]))
            d3 = frame[frame.dataset == "d3"]
            d3_table.extend({**base, **r} for r in d3_rows(d3, threshold, processed[("d3", "test")]))
            part = frame.copy()
            part["experiment"] = info["experiment"]
            part["threshold_policy"] = policy
            part["threshold"] = threshold
            part["pred"] = (part["score_attack"] >= threshold).astype(int)
            part["threshold_source"] = "e6"
            prediction_parts.append(part)

    metrics = pd.DataFrame(metric_rows)
    notinject = pd.DataFrame(d4_table)
    d3_slices = pd.DataFrame(d3_table)
    predictions = pd.concat(prediction_parts, ignore_index=True)[
        ["experiment", "dataset", "split", "sample_id", "gold", "score_attack", "threshold_policy", "threshold",
         "pred", "model_id", "checkpoint_rev", "seed", "transform_id", "parent_id", "threshold_source"]]

    # ---- write
    write_csv(thresholds, out / "thresholds.csv")
    write_csv(metrics, out / "metrics.csv")
    atomic_write(out / "predictions.parquet", lambda p: predictions.to_parquet(p, index=False))
    write_csv(notinject, out / "notinject_false_alarms.csv")
    write_csv(d3_slices, out / "supplementary/d3_by_attack_family.csv")
    write_csv(pd.concat(sweeps, ignore_index=True), out / "supplementary/validation_threshold_sweep.csv")

    figure_hashes = {}
    if not args.no_figures:
        from experiments.e6.figures import make_figures
        for path in make_figures(out, thresholds, metrics, notinject, pd.concat(sweeps, ignore_index=True), order):
            figure_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)

    from experiments.e6.summary import write_summary
    write_summary(out / "summary.md", thresholds, metrics, notinject, d3_slices, order, models)

    verification = {
        "status": "passed" if not excluded else "passed_with_exclusions",
        "config": {"path": "configs/e6.json", "sha256": sha256_file(CONFIG_PATH), "matches_configs_e1_thresholds": config_check},
        "input_acceptance": [{k: v for k, v in record.items()} | {"accepted": not record["problems"]} for record in inputs],
        "checks_per_input": ["10-column schema", "single model_id", "finite scores", "gold in {0,1}",
                             "unique (model_id, dataset, split, sample_id, transform_id)", "transform_id == clean",
                             "row counts 1985/2101/116/545 (+339 D4 optional)",
                             "sample_id set and gold identical to data/processed"],
        "threshold_search_equals_e1_select_thresholds": e1_check,
        "e1_joblib_rescore": joblib_check,
        "thresholds_frozen_before_evaluation": True,
        "d2_d3_d4_used_for_selection": False,
    }
    write_json(verification, out / "verification.json")

    import matplotlib, scipy, sklearn
    metadata = {
        "experiment": "E6 operating threshold selection and false-alarm analysis",
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "protocol": {
            "selection_data": "D1 Validation rows of each handoff file only (1,985 rows)",
            "candidates": "every unique validation score plus nextafter(max) (E1 threshold_candidates)",
            "tF1": "maximise Macro F1; ties -> higher threshold",
            "t5": f"maximise Attack Recall among candidates with validation FPR <= {T5_MAX_FPR}; ties -> higher threshold",
            "prediction_rule": "pred = 1 iff score_attack >= threshold",
            "evaluation": "thresholds frozen before D1/D2/D3 Test and D4 metrics are computed; sources reported separately",
            "d4": "benign only: FP/N, FPR, Wilson 95% CI (Clopper-Pearson also given); no F1 or AUROC",
            "uncertainty": (f"Wilson 95% CI for recall and FPR; class-stratified percentile bootstrap ({N_BOOT} resamples) "
                            f"for Macro F1, seeded per row with [{SEED}, crc32('model_id|policy|dataset|split')]"),
            "scores": "uncalibrated; not probabilities of maliciousness",
        },
        "inputs": inputs,
        "excluded_inputs": [x["path"] for x in excluded],
        "models": {m: {"experiment": models[m]["experiment"], "is_main_pipeline": models[m]["experiment"] in MAIN_EXPERIMENTS,
                       "has_d4": bool((models[m]["frame"].dataset == "d4").any()), "warnings": models[m]["warnings"]}
                   for m in order},
        "missing_main_experiments": [e for e in MAIN_EXPERIMENTS if e not in {models[m]["experiment"] for m in order}],
        "config": CONFIG,
        "verification": "results/e6/verification.json",
        "outputs": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.suffix in {".csv", ".parquet", ".md"}},
        "figures": figure_hashes,
    }
    write_json(metadata, out / "run_metadata.json")
    print(f"[e6] wrote {out.relative_to(ROOT).as_posix()}/ ({len(order)} models, {len(predictions):,} prediction rows)", flush=True)
    if metadata["missing_main_experiments"]:
        print(f"[e6] main pipelines still missing: {metadata['missing_main_experiments']}", flush=True)


if __name__ == "__main__":
    main()
