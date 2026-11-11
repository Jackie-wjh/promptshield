"""Verbatim copies of E1's scoring and threshold functions (experiments/e1/run_e1.py, lines 128-173).

experiments/e1/run_e1.py cannot be imported from the handoff package because it imports
scripts.check_data_integrity, which is not shipped. E6 uses these copies only as an independent
reference: its own vectorised search must reproduce select_thresholds exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score


def attack_scores(model, texts: pd.Series | list[str]) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    attack_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    return model.predict_proba(list(texts))[:, attack_index]


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
    all_negative = np.nextafter(unique.max(), np.inf)
    return np.concatenate([unique, [all_negative]])


def select_thresholds(gold: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    candidates = threshold_candidates(scores)
    records = [(float(threshold), metric_record(gold, scores, float(threshold))) for threshold in candidates]

    tf1_threshold, tf1_metrics = max(
        records,
        key=lambda item: (item[1]["macro_f1"], item[0]),
    )
    eligible_t5 = [item for item in records if item[1]["fpr"] is not None and item[1]["fpr"] <= 0.05]
    t5_threshold, t5_metrics = max(
        eligible_t5,
        key=lambda item: (item[1]["attack_recall"], item[0]),
    )
    return {
        "default_0.5": {"threshold": 0.5, **metric_record(gold, scores, 0.5)},
        "tF1": {"threshold": tf1_threshold, **tf1_metrics},
        "t5": {"threshold": t5_threshold, **t5_metrics},
    }
