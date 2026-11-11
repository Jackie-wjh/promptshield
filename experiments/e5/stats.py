"""Statistics helpers for E5: confusion counts, Macro F1 from counts, class-stratified bootstrap.

All resampling is done on row indices so that the same draw can be applied to both models and to
both the original and the perturbed text of a pair (paired bootstrap).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def counts(gold: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((gold == 1) & (prediction == 1)))
    fp = int(np.sum((gold == 0) & (prediction == 1)))
    tn = int(np.sum((gold == 0) & (prediction == 0)))
    fn = int(np.sum((gold == 1) & (prediction == 0)))
    return tp, fp, tn, fn


def macro_f1_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    """Macro F1 with sklearn's zero_division=0 convention."""

    positive = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    negative = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return float((positive + negative) / 2.0)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def stratified_indices(strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One bootstrap draw: every stratum is resampled with replacement to its own size."""

    return np.concatenate([rng.choice(stratum, len(stratum)) for stratum in strata if len(stratum)])


def bootstrap_standard_error(gold: np.ndarray, scores: np.ndarray, threshold: float, n_boot: int, seed: int) -> dict[str, Any]:
    """Class-stratified bootstrap SE of Macro F1 at ``threshold`` (used for the selection guard)."""

    gold = np.asarray(gold, dtype=int)
    prediction = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    strata = [np.flatnonzero(gold == 1), np.flatnonzero(gold == 0)]
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        index = stratified_indices(strata, rng)
        values[k] = macro_f1_from_counts(*counts(gold[index], prediction[index]))
    point = macro_f1_from_counts(*counts(gold, prediction))
    return {"point_estimate": point, "standard_error": float(values.std(ddof=1)), "n_boot": int(n_boot), "seed": int(seed), "resampling": "class-stratified"}


def paired_bootstrap(
    gold: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    n_boot: int,
    seed: int,
    strata_by: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired, stratified bootstrap of (B minus A) for Macro F1, FPR and attack recall.

    Both models (or both texts) are evaluated on the same resampled rows, so each draw is shared.
    Strata default to the gold classes (attack and benign rows resampled independently); for a
    benign-only set a different grouping (e.g. trigger_count) can be passed via ``strata_by``.
    Macro F1 and recall are undefined when a class is absent and are reported as NaN.
    """

    gold = np.asarray(gold, dtype=int)
    pred_a = (np.asarray(score_a, dtype=float) >= threshold_a).astype(int)
    pred_b = (np.asarray(score_b, dtype=float) >= threshold_b).astype(int)
    grouping = gold if strata_by is None else np.asarray(strata_by)
    strata = [np.flatnonzero(grouping == level) for level in np.unique(grouping)]
    both_classes = bool((gold == 1).any() and (gold == 0).any())
    rng = np.random.default_rng(seed)

    def deltas(index: np.ndarray) -> tuple[float, float, float]:
        g = gold[index]
        tp_a, fp_a, tn_a, fn_a = counts(g, pred_a[index])
        tp_b, fp_b, tn_b, fn_b = counts(g, pred_b[index])
        d_f1 = macro_f1_from_counts(tp_b, fp_b, tn_b, fn_b) - macro_f1_from_counts(tp_a, fp_a, tn_a, fn_a) if both_classes else float("nan")
        d_recall = safe_ratio(tp_b, tp_b + fn_b) - safe_ratio(tp_a, tp_a + fn_a) if both_classes else float("nan")
        d_fpr = safe_ratio(fp_b, fp_b + tn_b) - safe_ratio(fp_a, fp_a + tn_a)
        return d_f1, d_fpr, d_recall

    point = deltas(np.arange(len(gold)))
    draws = np.empty((n_boot, 3), dtype=float)
    for k in range(n_boot):
        draws[k] = deltas(stratified_indices(strata, rng))
    result: dict[str, Any] = {"n_samples": int(len(gold)), "n_boot": int(n_boot), "n_attack": int((gold == 1).sum()), "n_benign": int((gold == 0).sum())}
    for position, name in enumerate(("macro_f1", "fpr", "attack_recall")):
        column = draws[:, position]
        defined = column[np.isfinite(column)]
        if len(defined) == 0:
            result[name] = {"point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_mean": float("nan"), "excludes_zero": False}
            continue
        low, high = np.percentile(defined, [2.5, 97.5])
        result[name] = {"point_estimate": float(point[position]), "ci_low": float(low), "ci_high": float(high), "bootstrap_mean": float(np.mean(defined)), "excludes_zero": bool(low > 0 or high < 0)}
    return result
