#!/usr/bin/env python3
"""E3 supplementary analysis: paired bootstrap for the prespecified E3-versus-E1 comparison.

The plan prespecifies two paired, class-stratified bootstrap comparisons at 1,000 resamples:
E3 versus E1 Combined, and E5 augmented versus clean. This script supplies the E3 half so the
report's statistical section can cite a reproducible artefact. It only reads the two frozen
handoff score files; it never re-selects a model or a threshold.

Outputs:
    results/e3/evidence/bootstrap_e3_vs_e1.csv
    results/e3/evidence/d4_reference_at_0.5.csv

Usage:
    python experiments/e3/analyse_e3.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
E3_SCORES = PROJECT_ROOT / "results/e3/handoff/distilbert_scores.parquet"
E1_SCORES = PROJECT_ROOT / "results/e1/handoff/combined_scores.parquet"
OUT_DIR = PROJECT_ROOT / "results/e3/evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6405)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def macro_f1(gold: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(gold, prediction, average="macro", zero_division=0))


def fpr(gold: np.ndarray, prediction: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(gold, prediction, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if fp + tn else float("nan")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(
    gold: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Class-stratified, paired resampling. Both predictions are indexed with the same draws."""

    if metric == "fpr":
        pool = np.flatnonzero(gold == 0)
        strata = [pool]
    else:
        strata = [np.flatnonzero(gold == 1), np.flatnonzero(gold == 0)]

    def score(gold_slice: np.ndarray, prediction: np.ndarray) -> float:
        return macro_f1(gold_slice, prediction) if metric == "macro_f1" else fpr(gold_slice, prediction)

    point = score(gold, prediction_b) - score(gold, prediction_a)
    deltas = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        drawn = np.concatenate([rng.choice(stratum, len(stratum), replace=True) for stratum in strata])
        deltas[index] = score(gold[drawn], prediction_b[drawn]) - score(gold[drawn], prediction_a[drawn])
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_mean": float(deltas.mean()),
        "n_boot": int(n_boot),
        "excludes_zero": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    e3 = pd.read_parquet(E3_SCORES)
    e1 = pd.read_parquet(E1_SCORES)

    rows: list[dict[str, Any]] = []
    splits = [
        ("d1", "test", "D1 test"),
        ("d2", "test", "D2 test"),
        ("d3", "test", "D3 test"),
    ]
    for dataset, part, label in splits:
        left = e3[e3["dataset"].eq(dataset) & e3["split"].eq(part)][["sample_id", "gold", "score_attack"]]
        right = e1[e1["dataset"].eq(dataset) & e1["split"].eq(part)][["sample_id", "gold", "score_attack"]]
        merged = left.merge(right, on="sample_id", suffixes=("_e3", "_e1"), validate="one_to_one")
        if not np.array_equal(merged["gold_e3"].to_numpy(), merged["gold_e1"].to_numpy()):
            raise RuntimeError(f"Gold labels disagree between E3 and E1 on {dataset}/{part}")

        gold = merged["gold_e3"].to_numpy(dtype=int)
        prediction_e3 = (merged["score_attack_e3"].to_numpy(dtype=float) >= args.threshold).astype(int)
        prediction_e1 = (merged["score_attack_e1"].to_numpy(dtype=float) >= args.threshold).astype(int)

        for metric in ("macro_f1", "fpr"):
            result = paired_bootstrap(gold, prediction_e1, prediction_e3, metric, args.boot, rng)
            rows.append(
                {
                    "split_label": label,
                    "comparison": "E3 DistilBERT minus E1 Combined TF-IDF LR",
                    "metric": metric,
                    "threshold": args.threshold,
                    "n_samples": int(len(gold)),
                    "e3_value": macro_f1(gold, prediction_e3) if metric == "macro_f1" else fpr(gold, prediction_e3),
                    "e1_value": macro_f1(gold, prediction_e1) if metric == "macro_f1" else fpr(gold, prediction_e1),
                    **result,
                }
            )
        print(f"  {label}: n={len(gold):,} bootstrapped", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "bootstrap_e3_vs_e1.csv", index=False)

    # D4 reference row for the benign-only stress set. E6 owns the full trigger/category slices;
    # E3 needs the overall denominator to describe its own false-alarm behaviour honestly.
    d4_rows: list[dict[str, Any]] = []
    for name, frame_scores in (("E3 DistilBERT", e3), ("E1 Combined TF-IDF LR", e1)):
        subset = frame_scores[frame_scores["dataset"].eq("d4") & frame_scores["split"].eq("test")]
        if subset.empty:
            continue
        gold = subset["gold"].to_numpy(dtype=int)
        prediction = (subset["score_attack"].to_numpy(dtype=float) >= args.threshold).astype(int)
        false_positives = int(((prediction == 1) & (gold == 0)).sum())
        negatives = int((gold == 0).sum())
        low, high = wilson_interval(false_positives, negatives)
        d4_rows.append(
            {
                "model": name,
                "threshold": args.threshold,
                "n": int(len(subset)),
                "gold_benign": negatives,
                "fp": false_positives,
                "fpr": false_positives / negatives if negatives else float("nan"),
                "fpr_ci_low": low,
                "fpr_ci_high": high,
                "note": "Benign-only stress set: F1, recall and AUROC are not defined here and are not reported. "
                "E6 owns the authoritative NotInject table, including trigger-count and category slices.",
            }
        )
    pd.DataFrame(d4_rows).to_csv(OUT_DIR / "d4_reference_at_0.5.csv", index=False)

    summary = {
        "status": "complete",
        "comparisons": [
            "E3 versus E1 Combined TF-IDF LR (prespecified in the plan, section 7)",
        ],
        "design": {
            "resampling": "paired, class-stratified (attack and benign strata resampled independently)",
            "n_boot": int(args.boot),
            "rng_seed": int(args.seed),
            "threshold": args.threshold,
            "note": "Both models were scored on identical sample_ids, so each draw is shared. "
            "Intervals describe sampling variation on the evaluated sets, not variation across training seeds.",
        },
        "files": {
            "bootstrap": "results/e3/evidence/bootstrap_e3_vs_e1.csv",
            "d4_reference": "results/e3/evidence/d4_reference_at_0.5.csv",
        },
    }
    (OUT_DIR / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("", flush=True)
    print(
        frame[["split_label", "metric", "e3_value", "e1_value", "point_estimate", "ci_low", "ci_high", "excludes_zero"]]
        .to_string(index=False),
        flush=True,
    )
    print("", flush=True)
    print(pd.DataFrame(d4_rows)[["model", "fp", "gold_benign", "fpr", "fpr_ci_low", "fpr_ci_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
