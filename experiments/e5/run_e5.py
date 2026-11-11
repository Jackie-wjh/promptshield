#!/usr/bin/env python3
"""Run E5: controlled typo augmentation on the frozen E1 Combined TF-IDF + Logistic Regression pipeline.

Threat model: accidental single-character typos in alphabetic words of at least four letters.

Protocol (configs/e5.json, experiments/e5/README.md):
  Phase A (selection; uses only D1 Train, D1 Validation, the manifest, the E1 joblib and the
  validation rows of E1's handoff scores for modelling and selection):
    * Data gate: SHA-256, row counts and label counts of the six processed files are re-derived
      from data/audit/manifest.json before anything else (this is the only Phase-A access to the
      test files; nothing from them is scored or compared until Phase B).
    * The fitted word+char TF-IDF FeatureUnion is loaded from models/e1/combined.joblib (hash
      cross-checked against results/e1/selected_models.csv) and never refit. Only the Logistic
      Regression (exactly E1's settings) is retrained, once per training-set composition.
    * A model-agnostic lexicon of attack- and benign-indicative words is built from D1 Train labels.
    * Conditions differ only in the copies appended to D1 Train: none (clean), exact duplicates
      (count-matched control) or typo-perturbed copies under three policies.
    * Every condition is scored on clean D1 Validation and on five typo-perturbed copies of it; the
      primary augmented condition is chosen by the pre-registered rule and frozen
      (selected_config.json, frozen = true) BEFORE any test split or D4 is opened.
  Phase B (evaluation):
    * The clean and the primary augmented model are scored once on D1 Validation, D1/D2/D3 Test and
      D4, and on typo-perturbed copies of D1 Validation and D1/D2/D3 Test (canonical evaluation
      seed for the handoff pairs, four extra draws for the spread). All conditions are also scored
      once on the test splits as pre-registered secondary evidence.
    * Paired class-stratified bootstrap intervals, D4 false alarms with Wilson intervals,
      representation-change rates, a mechanism analysis, figures, the handoff files and the
      human-QA sample are written.

Usage:
    python experiments/e5/run_e5.py                         # official run
    python experiments/e5/run_e5.py --stop-after-selection  # Phase A only
    python experiments/e5/run_e5.py --vectorizer-source refit --n-boot 200   # development only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "promptshield_mpl"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:  # Windows consoles default to a legacy code page; keep every print ASCII-safe anyway
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from experiments.e5 import common, figures
from experiments.e5.common import (
    HANDOFF_COLUMNS,
    PREDICTION_COLUMNS,
    atomic_csv,
    atomic_joblib,
    atomic_json,
    atomic_parquet,
    metric_record,
    select_thresholds,
    sha256_file,
    wilson_interval,
    write_text,
)
from experiments.e5.stats import bootstrap_standard_error, counts, paired_bootstrap, safe_ratio
from experiments.e5.typo import OPERATORS, augment_frame, build_lexicon

SPLIT_META: dict[str, tuple[str, str]] = {
    "d1_train": ("d1", "train"),
    "d1_validation": ("d1", "validation"),
    "d1_test": ("d1", "test"),
    "d2_test": ("d2", "test"),
    "d3_test": ("d3", "test"),
    "d4_test": ("d4", "test"),
}
PHASE_B_SPLITS = ("d1_test", "d2_test", "d3_test", "d4_test")
POLICY_ORDER = {"random1": 0, "random_rate": 1, "targeted_lexical": 2}
THRESHOLD_SUFFIX = {"default_0.5": "0.5", "tF1": "tF1", "t5": "t5"}


# --------------------------------------------------------------------------------------
# arguments, logging, small helpers
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/e5.json")
    parser.add_argument(
        "--vectorizer-source",
        choices=("joblib", "refit"),
        default="joblib",
        help="joblib = load the frozen E1 FeatureUnion (official). refit = rebuild it from configs/e1.json on D1 Train; "
        "development only, the run is marked NON-OFFICIAL everywhere and must not be handed to E6.",
    )
    parser.add_argument("--n-boot", type=int, default=None, help="override the bootstrap resample count (development)")
    parser.add_argument("--no-figures", action="store_true", help="skip figure rendering (development)")
    parser.add_argument("--stop-after-selection", action="store_true", help="run Phase A only and stop once selected_config.json is frozen")
    parser.add_argument("--strict-gate", action="store_true", help="abort when the clean retrain does not reproduce E1's validation scores (flips at 0.5 or max |diff| >= 1e-3); honoured in every mode")
    return parser.parse_args()


class RunLog:
    """Timestamped step log that ends up in run_metadata.json (contribution evidence, README §11.8)."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.steps: list[dict[str, Any]] = []

    def step(self, name: str) -> None:
        now = time.perf_counter()
        self.steps.append({"step": name, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": round(now - self.started, 2)})
        print(f"[E5 {now - self.started:7.1f}s] {name}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split(root: Path, stem: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / f"data/processed/{stem}.parquet").reset_index(drop=True)
    frame["text"] = frame["text"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame


def hashes_of(frame: pd.DataFrame) -> set[str]:
    return {common.normalized_sha256(text) for text in frame["text"].tolist()}


def posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def content_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes() + str(array.dtype).encode()).hexdigest()


def classifier_hashes(classifier: Any) -> dict[str, str]:
    return {"coef_sha256": content_hash(np.asarray(classifier.coef_, dtype=np.float64)), "intercept_sha256": content_hash(np.asarray(classifier.intercept_, dtype=np.float64))}


def branch_norms(classifier: Any, n_word: int) -> dict[str, float]:
    weights = np.asarray(classifier.coef_[0], dtype=float)
    return {"intercept": float(classifier.intercept_[0]), "w_norm_total": float(np.linalg.norm(weights)), "w_norm_word": float(np.linalg.norm(weights[:n_word])), "w_norm_char": float(np.linalg.norm(weights[n_word:]))}


def handoff_frame(dataset: str, split: str, sample_ids: list[str], gold: np.ndarray, scores: np.ndarray, model_id: str, checkpoint_rev: str, seed: int, transform_id: Any, parent_id: Any) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset": pd.array([dataset] * len(sample_ids), dtype="string"),
            "split": pd.array([split] * len(sample_ids), dtype="string"),
            "sample_id": pd.array(list(sample_ids), dtype="string"),
            "gold": np.asarray(gold, dtype=np.int64),
            "score_attack": np.asarray(scores, dtype=np.float64),
            "model_id": pd.array([model_id] * len(sample_ids), dtype="string"),
            "checkpoint_rev": pd.array([checkpoint_rev] * len(sample_ids), dtype="string"),
            "seed": np.full(len(sample_ids), int(seed), dtype=np.int64),
            "transform_id": pd.array(transform_id if isinstance(transform_id, list) else [transform_id] * len(sample_ids), dtype="string"),
            "parent_id": pd.array(parent_id if isinstance(parent_id, list) else [parent_id] * len(sample_ids), dtype="string"),
        }
    )
    return frame[HANDOFF_COLUMNS]


def pair_summary(gold: np.ndarray, score_original: np.ndarray, score_perturbed: np.ndarray, threshold: float) -> dict[str, Any]:
    """Paired robustness numbers for one (model, split, policy, threshold) on eligible pairs."""

    pred_o = (score_original >= threshold).astype(int)
    pred_p = (score_perturbed >= threshold).astype(int)
    attack = gold == 1
    benign = gold == 0
    delta = score_perturbed - score_original
    original = metric_record(gold, score_original, threshold)
    perturbed = metric_record(gold, score_perturbed, threshold)
    n_attack_detected = int(np.sum(attack & (pred_o == 1)))  # attacks the model caught on the clean text
    n_benign_rejected = int(np.sum(benign & (pred_o == 0)))  # benign texts the model passed on the clean text
    n_lost = int(np.sum(attack & (pred_o == 1) & (pred_p == 0)))
    n_gained = int(np.sum(benign & (pred_o == 0) & (pred_p == 1)))
    record: dict[str, Any] = {
        "n_pairs": int(len(gold)), "n_attack": int(attack.sum()), "n_benign": int(benign.sum()),
        "orig_macro_f1": original["macro_f1"], "pert_macro_f1": perturbed["macro_f1"],
        "orig_attack_recall": original["attack_recall"], "pert_attack_recall": perturbed["attack_recall"],
        "orig_fpr": original["fpr"], "pert_fpr": perturbed["fpr"], "orig_auroc": original["auroc"], "pert_auroc": perturbed["auroc"],
        "flip_rate": float(np.mean(pred_o != pred_p)),
        "n_attack_detected_original": n_attack_detected,
        "n_flip_attack_to_benign": n_lost,
        "flip_attack_to_benign_rate": safe_ratio(n_lost, n_attack_detected),  # conditional: lost / detected on the clean text
        "attack_lost_rate_of_all_attacks": safe_ratio(n_lost, int(attack.sum())),  # unconditional
        "n_benign_rejected_original": n_benign_rejected,
        "n_flip_benign_to_attack": n_gained,
        "flip_benign_to_attack_rate": safe_ratio(n_gained, n_benign_rejected),  # conditional: new false alarms / benign passed on the clean text
        "benign_gained_rate_of_all_benign": safe_ratio(n_gained, int(benign.sum())),  # unconditional
        "mean_abs_delta_score": float(np.mean(np.abs(delta))),
        "mean_delta_score_attack": float(np.mean(delta[attack])) if attack.any() else None,
        "mean_delta_score_benign": float(np.mean(delta[benign])) if benign.any() else None,
    }
    for label, mask in (("attack", attack), ("benign", benign)):
        for q in (5, 25, 50, 75, 95):
            record[f"delta_{label}_p{q}"] = float(np.percentile(delta[mask], q)) if mask.any() else None
    return record


def representation_change(features: Any, n_word: int, originals: list[str], perturbed: list[str]) -> dict[str, Any]:
    """Did the typo change the frozen TF-IDF vector, and in which branch? Cosine per branch and overall."""

    matrix_o = features.transform(originals).tocsr()
    matrix_p = features.transform(perturbed).tocsr()
    difference = (matrix_o - matrix_p).tocsr()
    difference.eliminate_zeros()
    result: dict[str, Any] = {"n_pairs": int(len(originals))}
    for branch, columns in (("any", slice(None)), ("word", slice(0, n_word)), ("char", slice(n_word, None))):
        changed = difference[:, columns].getnnz(axis=1) > 0
        o = matrix_o[:, columns]
        p = matrix_p[:, columns]
        dot = np.asarray(o.multiply(p).sum(axis=1)).ravel()
        norm_o = np.sqrt(np.asarray(o.multiply(o).sum(axis=1)).ravel())
        norm_p = np.sqrt(np.asarray(p.multiply(p).sum(axis=1)).ravel())
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.where((norm_o > 0) & (norm_p > 0), dot / (norm_o * norm_p), np.nan)
        result[f"changed_{branch}"] = changed
        result[f"changed_{branch}_rate"] = float(changed.mean())
        result[f"cosine_{branch}"] = cosine
        result[f"mean_cosine_{branch}"] = float(np.nanmean(cosine))
    return result


def denominator_record(rows: pd.DataFrame, train_aug_hashes: set[str] | None = None) -> dict[str, Any]:
    """Final paired denominators for one perturbed copy, overall and by gold class."""

    record: dict[str, Any] = {"role": rows["role"].iloc[0], "dataset": rows["dataset"].iloc[0], "split": rows["split"].iloc[0], "policy": rows["policy"].iloc[0], "perturbation_seed": int(rows["seed"].iloc[0]), "transform_id": rows["transform_id"].iloc[0]}
    for label_name, mask in (("all", np.ones(len(rows), dtype=bool)), ("attack", rows["label"].to_numpy() == 1), ("benign", rows["label"].to_numpy() == 0)):
        subset = rows[mask]
        record[f"n_{label_name}"] = int(len(subset))
        record[f"n_{label_name}_eligible_pairs"] = int(subset["valid"].sum())
        record[f"n_{label_name}_no_eligible_word"] = int(subset["status"].eq("ineligible_no_word").sum())
        record[f"n_{label_name}_invalid_collision"] = int(subset["status"].eq("invalid_collision").sum())
        record[f"n_{label_name}_invalid_unchanged"] = int(subset["status"].eq("invalid_unchanged").sum())
        record[f"n_{label_name}_fallback_random"] = int(subset["fallback_random"].fillna(False).astype(bool).sum())
        if train_aug_hashes is not None:
            record[f"n_{label_name}_collides_with_train"] = int(subset[subset["valid"]]["normalized_sha256"].isin(train_aug_hashes).sum())
    return record


QA_COLUMNS = ["qa_id", "qa_stratum", "stage", "dataset", "split", "sample_id", "parent_id", "label", "transform_id", "policy", "n_ops", "operators", "all_typos",
              "operator", "original_word", "perturbed_word", "char_index", "perturbed_in_word_vocab", "perturbed_text_sha256", "original_text", "perturbed_text"]


def draw_qa_sample(manifest: pd.DataFrame, ops_all: pd.DataFrame, parent_text: pd.Series, selected_condition: str, selection_seed: int, canonical_seed: int, qa_config: dict[str, Any], seed: int) -> pd.DataFrame:
    """Stratified human-QA sample: per stage x first operator, plus real-word outcomes. Deterministic given ``seed``.

    (sample_id, transform_id) is the unique key of a perturbed row: evaluation rows reuse the processed
    sample_id across policies and seeds, so every join uses both columns. Multi-typo rows carry all their
    changes in ``all_typos``; the stratum uses the first operator only.
    """

    rng = np.random.default_rng(int(seed))
    first_ops = ops_all[ops_all["op_index"] == 0].set_index(["sample_id", "transform_id"])[["operator", "original_word", "perturbed_word", "char_index"]]
    grouped = ops_all.sort_values("op_index").groupby(["sample_id", "transform_id"])
    all_changes = grouped.apply(lambda g: "; ".join(f"{o}->{p}" for o, p in zip(g["original_word"], g["perturbed_word"])), include_groups=False).rename("all_typos")
    any_real_word = grouped["perturbed_in_word_vocab"].max().rename("perturbed_in_word_vocab")
    pool = manifest[manifest["valid"] & (manifest["policy"] != "dup")].copy()
    pool = pool[(pool["role"] != "train") | ((pool["condition"] == selected_condition) & (pool["copy_index"] == 0))]
    pool = pool[(pool["role"] != "selection") | (pool["seed"] == int(selection_seed))]
    pool = pool[(pool["role"] != "evaluation") | (pool["seed"] == int(canonical_seed))]
    pool = pool.join(first_ops, on=["sample_id", "transform_id"]).join(all_changes, on=["sample_id", "transform_id"]).join(any_real_word, on=["sample_id", "transform_id"])
    parts = []
    for stage in qa_config["stages"]:
        for operator in OPERATORS:
            group = pool[(pool["role"] == stage) & (pool["operator"] == operator)]
            take = min(int(qa_config["per_operator_per_stage"]), len(group))
            if take:
                parts.append(group.iloc[np.sort(rng.choice(len(group), take, replace=False))].assign(qa_stratum=f"{stage}/{operator}"))
    real_word = pool[pool["perturbed_in_word_vocab"].fillna(False).astype(bool)]
    take = min(int(qa_config["extra_real_word_rows"]), len(real_word))
    if take:
        parts.append(real_word.iloc[np.sort(rng.choice(len(real_word), take, replace=False))].assign(qa_stratum="real_word_outcome"))
    sample = pd.concat(parts, ignore_index=True).drop_duplicates(["sample_id", "transform_id"])
    sample["original_text"] = sample["parent_id"].map(parent_text)
    sample.insert(0, "qa_id", np.arange(1, len(sample) + 1))
    sample = sample.rename(columns={"role": "stage", "text": "perturbed_text", "text_sha256": "perturbed_text_sha256"})
    return sample[QA_COLUMNS]


def measure_latency(model: Any, texts: list[str], warmup_samples: int, measured_samples: int) -> tuple[float, float]:
    """Batch-1 latency in ms including vectorization (identical procedure to E1)."""

    for text in texts[:warmup_samples]:
        common.attack_scores(model, [text])
    durations = []
    for text in texts[warmup_samples : warmup_samples + measured_samples]:
        started = time.perf_counter()
        common.attack_scores(model, [text])
        durations.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(durations)), float(np.percentile(durations, 95))


CLEAN_ONLY_SORT_KEYS = ["clean_val_macro_f1", "copies", "rate", "policy_order", "order"]
TIE_BREAK_KEYS = ["copies", "rate", "policy_order", "order"]


def with_policy_order(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    frame["policy_order"] = frame["policy"].map(POLICY_ORDER).fillna(9)
    return frame


def clean_only_winner(candidates: pd.DataFrame) -> str:
    """The group's plain rule (best clean-validation Macro F1, same tie-breaks), reported for comparison."""

    frame = with_policy_order(candidates)
    return str(frame.sort_values(CLEAN_ONLY_SORT_KEYS, ascending=[False, True, True, True, True]).iloc[0]["condition"])


def select_primary(candidates: pd.DataFrame, guard_value: float, objective_column: str, sd_column: str | None) -> dict[str, Any]:
    """Pre-registered rule: best mean perturbed-validation Macro F1 subject to the clean-validation guard.

    Returns the winner plus everything needed to audit the decision: which candidates passed the
    guard, the objective ranking, the tie group (within the leader's realization SD), the objective
    leader, and the margin of the WINNER over the best other candidate (negative when the tie rule
    promoted a non-leader). Both branches return the same keys.
    """

    frame = with_policy_order(candidates)
    feasible = frame[frame["clean_val_macro_f1"] >= guard_value]
    guard_failed = feasible.empty
    pool = frame if guard_failed else feasible
    by_objective = pool.sort_values([objective_column] + TIE_BREAK_KEYS, ascending=[False, True, True, True, True])
    leader = by_objective.iloc[0]
    if guard_failed:
        fallback = pool.sort_values(CLEAN_ONLY_SORT_KEYS, ascending=[False, True, True, True, True])
        winner_row = fallback.iloc[0]
        tie_group: list[str] = []
        tolerance = None
        fallback_ranking = fallback["condition"].tolist()
    else:
        tolerance = float(leader[sd_column]) if sd_column else 0.0
        tied = pool[pool[objective_column] >= float(leader[objective_column]) - tolerance]
        ranked = tied.sort_values(TIE_BREAK_KEYS, ascending=[True, True, True, True])
        winner_row = ranked.iloc[0]
        tie_group = ranked["condition"].tolist()
        fallback_ranking = []
    winner = str(winner_row["condition"])
    others = by_objective[by_objective["condition"] != winner]
    runner_up = others.iloc[0] if len(others) else None
    return {
        "winner": winner,
        "guard_failed": bool(guard_failed),
        "feasible": feasible["condition"].tolist(),
        "ranked_by_objective": by_objective["condition"].tolist(),
        "fallback_ranking_by_clean_f1": fallback_ranking,
        "tie_group": tie_group,
        "tie_tolerance": tolerance,
        "objective_leader": str(leader["condition"]),
        "winner_minus_leader": float(winner_row[objective_column] - leader[objective_column]),
        "runner_up": None if runner_up is None else str(runner_up["condition"]),
        "margin": None if runner_up is None else float(winner_row[objective_column] - runner_up[objective_column]),
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = load_json(config_path)
    config_e1 = load_json(root / config["frozen"]["e1_config"])
    results_dir = root / "results/e5"
    handoff_dir = results_dir / "handoff"
    supplementary_dir = results_dir / "supplementary"  # analyses beyond the README §7 E5 checklist
    predictions_dir = results_dir / "predictions"
    figures_dir = results_dir / "figures"
    models_dir = root / "models/e5"
    seeds = config["seeds"]
    classifier_seed = int(config_e1["seed"])  # E1's seed is the single source of truth
    if classifier_seed != int(seeds["classifier"]):
        raise RuntimeError("configs/e5.json seeds.classifier must equal configs/e1.json seed")
    c_value = float(config["frozen"]["C"])
    n_boot = int(args.n_boot) if args.n_boot is not None else int(config["bootstrap"]["n_boot"])
    max_attempts = int(config["typo"]["max_attempts"])
    official = args.vectorizer_source == "joblib" and args.n_boot is None  # any development override makes the run non-official
    log = RunLog()
    captured_warnings: list[str] = []

    # ================================ Phase A: selection ================================
    log.step("data gate: SHA-256, rows and labels of the six processed files")
    integrity = common.verify_data_integrity(root)
    if integrity["status"] != "passed":
        raise RuntimeError(f"Data integrity gate failed: {json.dumps(integrity, indent=2)}")
    train = load_split(root, "d1_train")
    validation = load_split(root, "d1_validation")
    expected_rows = config["evaluation"]["expected_rows"]
    if len(validation) != int(expected_rows["d1_validation"]):
        raise RuntimeError(f"d1_validation: expected {expected_rows['d1_validation']} rows, found {len(validation)}")
    y_train = train["label"].to_numpy(dtype=int)
    y_validation = validation["label"].to_numpy(dtype=int)
    forbidden_prefreeze = hashes_of(train) | hashes_of(validation)

    log.step("frozen features: " + ("loading models/e1/combined.joblib" if official else "DEVELOPMENT MODE, refitting from configs/e1.json"))
    e1_selected = pd.read_csv(root / config["frozen"]["e1_selected_models"])
    e1_row = e1_selected[e1_selected["pipeline"] == "combined"].iloc[0]
    expected_sha = str(e1_row["checkpoint_rev"])
    if expected_sha != config["frozen"]["e1_model_sha256"]:
        raise RuntimeError("configs/e5.json e1_model_sha256 disagrees with results/e1/selected_models.csv")
    if official:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            features, e1_classifier, frozen_info = common.load_frozen_e1(root, expected_sha)
        captured_warnings.extend(str(w.message) for w in caught)
        if float(e1_classifier.C) != float(e1_row["selected_C"]) or int(e1_classifier.coef_.shape[1]) != int(e1_row["features"]):
            raise RuntimeError("The loaded E1 Combined model does not match results/e1/selected_models.csv")
    else:
        features = common.build_features_refit(config_e1, train["text"].tolist())
        e1_classifier = None
        frozen_info = {"path": None, "sha256": None, "note": "refit from configs/e1.json; NON-OFFICIAL"}
    features_provenance = f"frozen E1 Combined FeatureUnion loaded from models/e1/combined.joblib ({frozen_info['sha256']})" if frozen_info.get("path") else "FeatureUnion REFIT from configs/e1.json on D1 Train (NON-OFFICIAL development run; not the frozen E1 vectorizer)"
    development_overrides = ([f"--vectorizer-source {args.vectorizer_source}"] if args.vectorizer_source != "joblib" else []) + ([f"--n-boot {n_boot} (configured {config['bootstrap']['n_boot']})"] if args.n_boot is not None else [])
    fingerprint_before = common.feature_fingerprint(features)
    n_word = int(fingerprint_before["word"]["n_features"])
    for branch, expected in config["frozen"]["expected_branch_features"].items():
        if fingerprint_before[branch]["n_features"] != int(expected):
            raise RuntimeError(f"{branch} branch has {fingerprint_before[branch]['n_features']} features, expected {expected}")
    word_vocabulary = features.transformer_list[0][1].vocabulary_

    log.step("lexicon from D1 Train labels (targeted policy, model-agnostic)")
    lexicon_frame, lexicon, lexicon_info = build_lexicon(train["text"].tolist(), y_train, int(config["lexicon"]["df_min"]), [int(k) for k in config["lexicon"]["top_k_candidates"]], float(config["lexicon"]["coverage_target"]))
    atomic_csv(lexicon_frame, results_dir / "lexicon.csv")
    print(f"[E5]   lexicon: k = {lexicon_info['chosen_k']} words per class (coverage target {lexicon_info['coverage_target']:.0%} {'met' if lexicon_info['coverage_target_met'] else 'NOT met'}; {lexicon_info['candidate_words']} candidate words with df >= {config['lexicon']['df_min']})", flush=True)

    log.step("transforming D1 Train and D1 Validation with the frozen features")
    matrices: dict[str, Any] = {"d1_train": features.transform(train["text"].tolist()).tocsr(), "d1_validation": features.transform(validation["text"].tolist()).tocsr()}

    def fit_classifier(matrix: Any, labels: np.ndarray) -> tuple[Any, float]:
        classifier = common.make_classifier(config_e1, c_value, classifier_seed)
        started = time.perf_counter()
        classifier.fit(matrix, labels)
        return classifier, time.perf_counter() - started

    log.step("clean condition: retraining E1's classifier on the frozen features")
    clean_classifier, clean_fit_seconds = fit_classifier(matrices["d1_train"], y_train)
    clean_val_scores = common.scores_from_matrix(clean_classifier, matrices["d1_validation"])
    selection = config["selection"]
    selection_threshold = float(selection["threshold"])
    delta_se = bootstrap_standard_error(y_validation, clean_val_scores, selection_threshold, 1000, int(seeds["delta_se"]))
    guard_value = float(delta_se["point_estimate"]) - float(selection["guard_multiplier"]) * float(delta_se["standard_error"])

    # Clean-reproduction gate on validation rows only. E1's handoff file is read in full here (it is a
    # score file, not a processed split); its test and D4 rows are first consulted after the freeze.
    e1_handoff_all = pd.read_parquet(root / config["frozen"]["e1_handoff_scores"])
    e1_validation = e1_handoff_all[(e1_handoff_all["dataset"] == "d1") & (e1_handoff_all["split"] == "validation")].set_index("sample_id")["score_attack"].reindex(validation["sample_id"]).to_numpy(dtype=float)
    gate = {
        "max_abs_diff_validation": float(np.max(np.abs(clean_val_scores - e1_validation))),
        "prediction_flips_at_0.5_validation": int(np.sum((clean_val_scores >= 0.5) != (e1_validation >= 0.5))),
    }
    gate["passed"] = bool(gate["prediction_flips_at_0.5_validation"] == 0 and gate["max_abs_diff_validation"] < 1e-3)
    print(f"[E5]   clean retrain vs E1 validation scores: max |diff| {gate['max_abs_diff_validation']:.2e}, flips at 0.5: {gate['prediction_flips_at_0.5_validation']} -> {'ok' if gate['passed'] else 'MISMATCH'}", flush=True)
    if not gate["passed"]:
        message = "The clean retrain does not reproduce E1's validation scores; " + ("expected when the vectorizer was refit (development mode)." if not frozen_info.get("path") else "investigate the scikit-learn / liblinear version before handing anything to E6.")
        if args.strict_gate:
            raise RuntimeError(message)
        print("[E5]   WARNING: " + message, flush=True)

    log.step("typo-perturbed copies of D1 Validation for selection (five random1 draws + one lexicon-targeted draw)")
    manifest_parts: list[pd.DataFrame] = []
    ops_parts: list[pd.DataFrame] = []
    denominator_rows: list[dict[str, Any]] = []
    selection_sets: list[dict[str, Any]] = []
    for seed in [int(s) for s in seeds["augment_select"]]:
        rows, ops, _stats = augment_frame(validation, role="selection", dataset="d1", split="validation", condition="selection_random1", policy="random1", rate=0.0, seed=seed, copy_index=0, transform_id=f"typo_random1_s{seed}", forbidden_hashes=forbidden_prefreeze, lexicon=lexicon, word_vocabulary=word_vocabulary, max_attempts=max_attempts, id_suffix=f"sel:random1:{seed}")
        manifest_parts.append(rows)
        ops_parts.append(ops)
        denominator_rows.append(denominator_record(rows))
        valid = rows[rows["valid"]].reset_index(drop=True)
        positions = valid["row_position"].to_numpy(dtype=int)
        selection_sets.append({"policy": "random1", "seed": seed, "positions": positions, "gold": y_validation[positions], "matrix": features.transform(valid["text"].tolist()).tocsr()})
    targeted_seed = int(seeds["augment_select"][0])
    rows, ops, _stats = augment_frame(validation, role="selection", dataset="d1", split="validation", condition="selection_targeted_lexical", policy="targeted_lexical", rate=0.0, seed=targeted_seed, copy_index=0, transform_id=f"typo_targeted_lexical_s{targeted_seed}", forbidden_hashes=forbidden_prefreeze, lexicon=lexicon, word_vocabulary=word_vocabulary, max_attempts=max_attempts, id_suffix=f"sel:targeted_lexical:{targeted_seed}")
    manifest_parts.append(rows)
    ops_parts.append(ops)
    denominator_rows.append(denominator_record(rows))
    valid = rows[rows["valid"]].reset_index(drop=True)
    positions = valid["row_position"].to_numpy(dtype=int)
    targeted_set = {"policy": "targeted_lexical", "seed": targeted_seed, "positions": positions, "gold": y_validation[positions], "matrix": features.transform(valid["text"].tolist()).tocsr()}

    log.step("training every condition and scoring it on clean and perturbed D1 Validation")
    conditions = config["conditions"]
    classifiers: dict[str, Any] = {}
    fit_info: dict[str, dict[str, Any]] = {}
    training_copies: dict[str, dict[str, Any]] = {}
    augmentation_summary: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for order, spec in enumerate(conditions):
        name = spec["name"]
        if name == "clean":
            classifier, seconds = clean_classifier, clean_fit_seconds
            n_rows, copies_texts, copies_labels, copy_positions = int(len(train)), [], [], np.array([], dtype=int)
            n_invalid_collision = n_invalid_unchanged = n_ineligible = n_fallback = 0
            n_ops_values: list[int] = []
        else:
            copies_texts, copies_labels, n_ops_values = [], [], []
            copy_positions_list: list[np.ndarray] = []
            n_invalid_collision = n_invalid_unchanged = n_ineligible = n_fallback = 0
            seen_in_condition: set[str] = set()  # shared across copies: no two copies may hold the same text
            for copy_index in range(int(spec["copies"])):
                seed = int(seeds["augment_train_copies"][copy_index])
                transform_id = f"dup_c{copy_index}" if spec["policy"] == "dup" else f"typo_{spec['policy']}_s{seed}_c{copy_index}"
                rows, ops, stats = augment_frame(train, role="train", dataset="d1", split="train", condition=name, policy=spec["policy"], rate=float(spec["rate"]), seed=seed, copy_index=copy_index, transform_id=transform_id, forbidden_hashes=forbidden_prefreeze, lexicon=lexicon, word_vocabulary=word_vocabulary, max_attempts=max_attempts, seen=seen_in_condition)
                manifest_parts.append(rows)
                ops_parts.append(ops)
                valid = rows[rows["valid"]]
                copies_texts.extend(valid["text"].tolist())
                copies_labels.extend(valid["label"].tolist())
                copy_positions_list.append(valid["row_position"].to_numpy(dtype=int))
                n_ops_values.extend(valid["n_ops"].tolist())
                n_invalid_collision += stats["n_invalid_collision"]
                n_invalid_unchanged += stats["n_invalid_unchanged"]
                n_ineligible += stats["n_no_eligible_word"]
                n_fallback += stats["n_fallback_random"]
                augmentation_summary.append({"condition": name, "copy_index": copy_index, "seed": seed, "policy": spec["policy"], "rate": float(spec["rate"]), "transform_id": transform_id, **stats})
            copy_positions = np.concatenate(copy_positions_list) if copy_positions_list else np.array([], dtype=int)
            matrix = sp.vstack([matrices["d1_train"], features.transform(copies_texts).tocsr()]).tocsr()
            labels = np.concatenate([y_train, np.asarray(copies_labels, dtype=int)])
            n_rows = int(matrix.shape[0])
            classifier, seconds = fit_classifier(matrix, labels)
        classifiers[name] = classifier
        training_copies[name] = {"texts": copies_texts, "positions": copy_positions}
        fit_info[name] = {"fit_seconds": float(seconds), "n_iter": int(np.max(classifier.n_iter_)), **branch_norms(classifier, n_word), **classifier_hashes(classifier)}

        val_scores = common.scores_from_matrix(classifier, matrices["d1_validation"])
        clean_metrics = metric_record(y_validation, val_scores, selection_threshold)
        eligible_positions = selection_sets[0]["positions"]  # eligibility does not depend on the seed
        clean_eligible = metric_record(y_validation[eligible_positions], val_scores[eligible_positions], selection_threshold)
        realization_f1, realization_recall, realization_fpr, realization_a2b, realization_b2a = [], [], [], [], []
        for item in selection_sets:
            pert_scores = common.scores_from_matrix(classifier, item["matrix"])
            summary = pair_summary(item["gold"], val_scores[item["positions"]], pert_scores, selection_threshold)
            realization_f1.append(summary["pert_macro_f1"])
            realization_recall.append(summary["pert_attack_recall"])
            realization_fpr.append(summary["pert_fpr"])
            realization_a2b.append(summary["n_flip_attack_to_benign"])
            realization_b2a.append(summary["n_flip_benign_to_attack"])
        targeted_scores = common.scores_from_matrix(classifier, targeted_set["matrix"])
        targeted_summary = pair_summary(targeted_set["gold"], val_scores[targeted_set["positions"]], targeted_scores, selection_threshold)
        added_labels = np.asarray(copies_labels, dtype=int)
        tuning_rows.append(
            {
                "condition": name, "order": order, "policy": spec["policy"], "rate": float(spec["rate"]), "copies": int(spec["copies"]), "is_control": bool(spec.get("control", False)),
                "n_training_rows": n_rows, "n_augmented_rows": int(len(copies_texts)), "n_ineligible": int(n_ineligible), "n_invalid_collision": int(n_invalid_collision),
                "n_invalid_unchanged": int(n_invalid_unchanged), "n_fallback_random": int(n_fallback), "attack_fraction_added": float(added_labels.mean()) if len(added_labels) else None,
                "n_ops_mean": float(np.mean(n_ops_values)) if n_ops_values else 0.0, "n_ops_median": float(np.median(n_ops_values)) if n_ops_values else 0.0,
                "fit_seconds": fit_info[name]["fit_seconds"], "n_iter": fit_info[name]["n_iter"], "intercept": fit_info[name]["intercept"],
                "w_norm_total": fit_info[name]["w_norm_total"], "w_norm_word": fit_info[name]["w_norm_word"], "w_norm_char": fit_info[name]["w_norm_char"],
                "selection_threshold": selection_threshold,
                "clean_val_macro_f1": clean_metrics["macro_f1"], "clean_val_attack_recall": clean_metrics["attack_recall"], "clean_val_fpr": clean_metrics["fpr"], "clean_val_auroc": clean_metrics["auroc"],
                "clean_val_macro_f1_eligible": clean_eligible["macro_f1"], "clean_val_attack_recall_eligible": clean_eligible["attack_recall"], "clean_val_fpr_eligible": clean_eligible["fpr"],
                "pert_val_n_pairs": int(len(eligible_positions)), "pert_val_macro_f1_mean": float(np.mean(realization_f1)), "pert_val_macro_f1_sd": float(np.std(realization_f1, ddof=1)),
                **{f"pert_val_macro_f1_r{k + 1}": float(v) for k, v in enumerate(realization_f1)},
                "pert_val_attack_recall_mean": float(np.mean(realization_recall)), "pert_val_fpr_mean": float(np.mean(realization_fpr)),
                "pert_val_flip_attack_to_benign_mean": float(np.mean(realization_a2b)), "pert_val_flip_benign_to_attack_mean": float(np.mean(realization_b2a)),
                "targeted_val_macro_f1": targeted_summary["pert_macro_f1"], "targeted_val_attack_recall": targeted_summary["pert_attack_recall"], "targeted_val_fpr": targeted_summary["pert_fpr"],
            }
        )
        print(f"[E5]   {name:18s} rows {n_rows:6,d} | clean-val F1 {clean_metrics['macro_f1']:.4f} | perturbed-val F1 {np.mean(realization_f1):.4f} ± {np.std(realization_f1, ddof=1):.4f} | targeted-val recall {targeted_summary['pert_attack_recall']:.4f}", flush=True)

    fingerprint_after = common.feature_fingerprint(features)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError("The frozen FeatureUnion changed during training; this must never happen")
    tuning = pd.DataFrame(tuning_rows)

    log.step("pre-registered selection on D1 Validation only")
    candidates = tuning[~tuning["is_control"]].copy()
    decision = select_primary(candidates, guard_value, "pert_val_macro_f1_mean", "pert_val_macro_f1_sd")
    selected_condition, guard_failed = decision["winner"], decision["guard_failed"]
    clean_only = clean_only_winner(candidates)
    tuning["satisfies_guard"] = tuning["clean_val_macro_f1"] >= guard_value
    tuning["is_selected"] = tuning["condition"].eq(selected_condition)
    tuning["clean_only_rule_winner"] = tuning["condition"].eq(clean_only)
    sensitivity_rows = []
    for multiplier in selection["sensitivity"]["guard_multipliers"]:
        variant = select_primary(candidates, float(delta_se["point_estimate"]) - float(multiplier) * float(delta_se["standard_error"]), "pert_val_macro_f1_mean", "pert_val_macro_f1_sd")
        sensitivity_rows.append({"variant": f"guard_multiplier={multiplier}", "objective": "mean of 5 realizations", "winner": variant["winner"], "guard_failed": variant["guard_failed"], "tie_group": "+".join(variant["tie_group"])})
    if selection["sensitivity"]["single_realizations"]:
        for k in range(len(selection_sets)):
            variant = select_primary(candidates, guard_value, f"pert_val_macro_f1_r{k + 1}", None)
            sensitivity_rows.append({"variant": f"single_realization_seed={selection_sets[k]['seed']}", "objective": f"realization {k + 1} only", "winner": variant["winner"], "guard_failed": variant["guard_failed"], "tie_group": "+".join(variant["tie_group"])})
    sensitivity_rows.append({"variant": "clean_only_rule", "objective": "clean validation Macro F1 (group default)", "winner": clean_only, "guard_failed": False, "tie_group": ""})
    sensitivity = pd.DataFrame(sensitivity_rows)
    print(f"[E5]   selected primary augmented condition: {selected_condition} (guard clean-val F1 >= {guard_value:.4f}, delta = {delta_se['standard_error']:.4f}; guard failed: {guard_failed}; tie group: {decision['tie_group']}; objective leader {decision['objective_leader']}; winner minus next-best {decision['runner_up']}: {decision['margin']:+.4f})", flush=True)

    log.step("saving every condition's model and freezing selected_config.json")
    checkpoint_revs: dict[str, str] = {}
    model_paths: dict[str, Path] = {}
    for name, classifier in classifiers.items():
        pipeline = common.assemble_pipeline(features, classifier)
        path = models_dir / f"{name}.joblib"
        atomic_joblib(pipeline, path)
        checkpoint_revs[name] = sha256_file(path)
        model_paths[name] = path
    model_ids = {"clean": config["model_ids"]["clean"], "augmented": config["model_ids"]["augmented"]}
    condition_of = {"clean": "clean", "augmented": selected_condition}
    frozen_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    selected_config: dict[str, Any] = {
        "experiment_id": config["experiment_id"], "official_run": official, "vectorizer_source": args.vectorizer_source, "frozen": True, "frozen_at_utc": frozen_at,
        "threat_model": config["threat_model"], "score_type": config["score_type"],
        "frozen_e1_model": {**frozen_info, "expected_sha256_from_selected_models": expected_sha}, "features_provenance": features_provenance, "development_overrides": development_overrides,
        "frozen_feature_fingerprint": fingerprint_before, "frozen_features_unchanged_after_training": True,
        "classifier": {"C": c_value, "solver": config_e1["classifier"]["solver"], "l1_ratio": config_e1["classifier"]["l1_ratio"], "class_weight": config_e1["classifier"]["class_weight"], "max_iter": config_e1["classifier"]["max_iter"], "random_state": classifier_seed},
        "seeds": seeds, "lexicon": {**config["lexicon"], **lexicon_info, "n_attack_words": int((lexicon_frame["lexicon_class"] == "attack").sum()), "n_benign_words": int((lexicon_frame["lexicon_class"] == "benign").sum())},
        "clean_reproduction_gate_validation": gate,
        "conditions": [{**spec, "n_training_rows": int(tuning.loc[tuning["condition"] == spec["name"], "n_training_rows"].iloc[0]), "model_path": posix(model_paths[spec["name"]], root), "checkpoint_rev": checkpoint_revs[spec["name"]], **fit_info[spec["name"]]} for spec in conditions],
        "selection": {**selection, "clean_reference_macro_f1": float(delta_se["point_estimate"]), "delta_standard_error": float(delta_se["standard_error"]), "delta_in_validation_errors": float(delta_se["standard_error"] / (1.0 / len(validation))), "guard_value": guard_value, "guard_failed": bool(guard_failed), "selected_condition": selected_condition, "decision": decision, "clean_only_rule_winner": clean_only, "candidates": candidates[["condition", "copies", "rate", "policy", "clean_val_macro_f1", "clean_val_macro_f1_eligible", "pert_val_macro_f1_mean", "pert_val_macro_f1_sd"]].assign(satisfies_guard=lambda f: f["clean_val_macro_f1"] >= guard_value).to_dict(orient="records"), "realization_seeds": [int(s) for s in seeds["augment_select"]], "perturbed_validation_pairs": int(len(selection_sets[0]["positions"]))},
        "frozen_models": {key: {"model_id": model_ids[key], "condition": condition_of[key], "model_path": posix(model_paths[condition_of[key]], root), "checkpoint_rev": checkpoint_revs[condition_of[key]]} for key in ("clean", "augmented")},
        "augmentation_summary": augmentation_summary,
    }
    atomic_json(selected_config, results_dir / "selected_config.json")
    atomic_csv(tuning, results_dir / "tuning.csv")
    atomic_csv(sensitivity, supplementary_dir / "selection_sensitivity.csv")
    atomic_csv(pd.DataFrame(augmentation_summary), results_dir / "augmentation_summary.csv")
    if args.stop_after_selection:
        atomic_json({"status": "selected_only", "frozen_at_utc": frozen_at, "steps": log.steps}, results_dir / "run_metadata.json")
        print("[E5] stopped after selection (Phase A); selected_config.json is frozen", flush=True)
        return

    # ================================ Phase B: evaluation ================================
    log.step("Phase B: first use of D1/D2/D3 Test and D4 texts for scoring (the data gate only hashed them)")
    frames: dict[str, pd.DataFrame] = {"d1_train": train, "d1_validation": validation}
    for stem in PHASE_B_SPLITS:
        frames[stem] = load_split(root, stem)
        if len(frames[stem]) != int(expected_rows[stem]):
            raise RuntimeError(f"{stem}: expected {expected_rows[stem]} rows, found {len(frames[stem])}")
        matrices[stem] = features.transform(frames[stem]["text"].tolist()).tocsr()
    split_hashes = {stem: hashes_of(frames[stem]) for stem in SPLIT_META}
    all_hashes: set[str] = set().union(*split_hashes.values())
    manifest_a = pd.concat(manifest_parts, ignore_index=True)
    train_aug_hashes = set(manifest_a[(manifest_a["role"] == "train") & manifest_a["valid"] & (manifest_a["policy"] != "dup")]["normalized_sha256"].tolist())

    log.step("post-freeze duplicate-equivalence audit against all six splits")
    audit: dict[str, Any] = {"note": "Pre-freeze, copies were filtered against D1 Train and D1 Validation only. This audit compares every augmented or perturbed text with every processed split after the freeze; nothing is altered.", "groups": []}
    for (role, condition, copy_index), group in manifest_a[manifest_a["valid"]].groupby(["role", "condition", "copy_index"]):
        if group["policy"].iloc[0] == "dup":
            continue
        entry: dict[str, Any] = {"role": role, "condition": condition, "copy_index": int(copy_index), "n_texts": int(len(group))}
        for stem, hashes in split_hashes.items():
            hits = group[group["normalized_sha256"].isin(hashes)]
            entry[f"collisions_with_{stem}"] = int(len(hits))
            if len(hits):
                entry[f"colliding_sample_ids_{stem}"] = hits["sample_id"].tolist()[:50]
        audit["groups"].append(entry)
    near_duplicates = {stem: int(len(split_hashes["d1_train"] & split_hashes[stem])) for stem in PHASE_B_SPLITS}
    audit["normalized_overlap_d1_train_vs_split"] = near_duplicates
    audit["total_collisions"] = int(sum(v for g in audit["groups"] for k, v in g.items() if k.startswith("collisions_with_")))

    log.step("provisional thresholds (E1 rule on clean D1 Validation) for every condition")
    thresholds_by_condition = {name: select_thresholds(y_validation, common.scores_from_matrix(classifier, matrices["d1_validation"])) for name, classifier in classifiers.items()}
    thresholds = {key: thresholds_by_condition[condition_of[key]] for key in ("clean", "augmented")}
    d4 = frames["d4_test"]
    slice_columns = [column for column in ("trigger_count", "category") if column in d4.columns]

    log.step("scoring all conditions once on every split (secondary evidence) and the two frozen models for the handoff")
    all_conditions_rows: list[dict[str, Any]] = []
    scores_by: dict[tuple[str, str], np.ndarray] = {}
    for name, classifier in classifiers.items():
        for stem, (dataset, split) in SPLIT_META.items():
            if stem == "d1_train":
                continue
            scores = common.scores_from_matrix(classifier, matrices[stem])
            scores_by[(name, stem)] = scores
            gold = frames[stem]["label"].to_numpy(dtype=int)
            for policy_name, values in thresholds_by_condition[name].items():
                threshold = float(values["threshold"])
                if dataset == "d4":
                    fp = int((scores >= threshold).sum())
                    all_conditions_rows.append({"condition": name, "dataset": dataset, "split": split, "threshold_policy": policy_name, "threshold": threshold, "macro_f1": None, "attack_recall": None, "fpr": fp / len(gold), "auroc": None, "tn": int(len(gold) - fp), "fp": fp, "fn": 0, "tp": 0, "n": int(len(gold))})
                else:
                    all_conditions_rows.append({"condition": name, "dataset": dataset, "split": split, "threshold_policy": policy_name, "threshold": threshold, **metric_record(gold, scores, threshold)})
    all_conditions = pd.DataFrame(all_conditions_rows)
    all_conditions.insert(0, "role", "secondary, pre-registered, never used for selection")

    metric_rows: list[dict[str, Any]] = []
    false_alarm_rows: list[dict[str, Any]] = []
    for key in ("clean", "augmented"):
        name = condition_of[key]
        model_id = model_ids[key]
        parts: list[pd.DataFrame] = []
        for stem, (dataset, split) in SPLIT_META.items():
            if stem == "d1_train":
                continue
            frame = frames[stem]
            gold = frame["label"].to_numpy(dtype=int)
            scores = scores_by[(name, stem)]
            parts.append(handoff_frame(dataset, split, frame["sample_id"].tolist(), gold, scores, model_id, checkpoint_revs[name], classifier_seed, "clean", pd.NA))
            for policy_name, values in thresholds[key].items():
                threshold = float(values["threshold"])
                if dataset == "d4":
                    prediction = (scores >= threshold).astype(int)
                    slices = [("overall", np.ones(len(frame), dtype=bool))]
                    for column in slice_columns:
                        slices += [(f"{column}={level}", (frame[column].to_numpy() == level)) for level in sorted(frame[column].dropna().unique(), key=str)]
                    for slice_name, mask in slices:
                        fp, n = int(prediction[mask].sum()), int(mask.sum())
                        low, high = wilson_interval(fp, n)
                        false_alarm_rows.append({"model_id": model_id, "model_key": key, "condition": name, "dataset": "d4", "split": "test", "slice": slice_name, "threshold_policy": policy_name, "threshold": threshold, "n": n, "fp": fp, "fpr": fp / n if n else None, "ci_low": low, "ci_high": high})
                    metric_rows.append({"model_id": model_id, "model_key": key, "condition": name, "dataset": dataset, "split": split, "threshold_policy": policy_name, "threshold": threshold, "threshold_source": config["evaluation"]["threshold_source"] if policy_name != "default_0.5" else "shared default", "macro_f1": None, "attack_recall": None, "fpr": float((scores >= threshold).mean()), "auroc": None, "tn": int((scores < threshold).sum()), "fp": int((scores >= threshold).sum()), "fn": 0, "tp": 0, "n": int(len(frame))})
                else:
                    metric_rows.append({"model_id": model_id, "model_key": key, "condition": name, "dataset": dataset, "split": split, "threshold_policy": policy_name, "threshold": threshold, "threshold_source": config["evaluation"]["threshold_source"] if policy_name != "default_0.5" else "shared default", **metric_record(gold, scores, threshold)})
        atomic_parquet(pd.concat(parts, ignore_index=True), handoff_dir / f"{model_id}_scores.parquet")
    metrics = pd.DataFrame(metric_rows)
    false_alarms = pd.DataFrame(false_alarm_rows)
    e1_full = e1_handoff_all.set_index(["dataset", "split", "sample_id"])["score_attack"]
    full_rows = pd.concat([handoff_frame(d, s, frames[stem]["sample_id"].tolist(), frames[stem]["label"].to_numpy(dtype=int), scores_by[("clean", stem)], model_ids["clean"], checkpoint_revs["clean"], classifier_seed, "clean", pd.NA) for stem, (d, s) in SPLIT_META.items() if stem != "d1_train"], ignore_index=True)
    reference = e1_full.reindex(pd.MultiIndex.from_frame(full_rows[["dataset", "split", "sample_id"]].astype(str))).to_numpy(dtype=float)
    reproduction_full = {"max_abs_diff_all_5086_rows": float(np.max(np.abs(full_rows["score_attack"].to_numpy() - reference))), "prediction_flips_at_0.5": int(np.sum((full_rows["score_attack"].to_numpy() >= 0.5) != (reference >= 0.5)))}

    log.step("typo-perturbed evaluation copies: canonical draw for the handoff pairs, extra draws for the spread")
    evaluation = config["evaluation"]
    canonical_seed = int(seeds["augment_eval_canonical"])
    extra_seeds = [int(s) for s in seeds["augment_eval_extra"]]
    pair_parts: list[pd.DataFrame] = []
    paired_metric_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    realization_rows: list[dict[str, Any]] = []
    bootstrap_inputs: list[dict[str, Any]] = []
    subset_names = ("all_eligible", "word_branch_changed", "word_branch_unchanged", "no_fallback", "excluding_train_collisions")
    for policy in evaluation["perturbation_policies"]:
        policy_seeds = [canonical_seed] + (extra_seeds if policy == "random1" else [])
        for seed in policy_seeds:
            for stem in evaluation["perturbed_splits"]:
                dataset, split = SPLIT_META[stem]
                frame = frames[stem]
                transform_id = f"typo_{policy}_s{seed}"
                rows, ops, stats = augment_frame(frame, role="evaluation", dataset=dataset, split=split, condition=f"eval_{policy}", policy=policy, rate=0.0, seed=seed, copy_index=0, transform_id=transform_id, forbidden_hashes=all_hashes, lexicon=lexicon, word_vocabulary=word_vocabulary, max_attempts=max_attempts, id_suffix=f"eval:{policy}:{seed}")
                rows["collides_with_train"] = rows["normalized_sha256"].isin(train_aug_hashes) & rows["valid"]
                manifest_parts.append(rows)
                ops_parts.append(ops)
                denominator_rows.append(denominator_record(rows, train_aug_hashes))
                valid = rows[rows["valid"]].reset_index(drop=True)
                if valid.empty:
                    continue
                positions = valid["row_position"].to_numpy(dtype=int)
                gold = frame["label"].to_numpy(dtype=int)[positions]
                pert_matrix = features.transform(valid["text"].tolist()).tocsr()
                pert_scores = {key: common.scores_from_matrix(classifiers[condition_of[key]], pert_matrix) for key in ("clean", "augmented")}
                pert_scores["dup"] = common.scores_from_matrix(classifiers["dup_x1"], pert_matrix) if "dup_x1" in classifiers else None
                for key in ("clean", "augmented"):
                    summary = pair_summary(gold, scores_by[(condition_of[key], stem)][positions], pert_scores[key], 0.5)
                    realization_rows.append({"policy": policy, "perturbation_seed": seed, "canonical": seed == canonical_seed, "dataset": dataset, "split": split, "model_id": model_ids[key], "model_key": key, "threshold": 0.5, "n_pairs": summary["n_pairs"], "orig_attack_recall": summary["orig_attack_recall"], "pert_attack_recall": summary["pert_attack_recall"], "orig_fpr": summary["orig_fpr"], "pert_fpr": summary["pert_fpr"], "flip_rate": summary["flip_rate"], "flip_attack_to_benign_rate": summary["flip_attack_to_benign_rate"], "n_flip_attack_to_benign": summary["n_flip_attack_to_benign"], "n_flip_benign_to_attack": summary["n_flip_benign_to_attack"]})
                if seed != canonical_seed:
                    continue
                first_ops = ops[ops["op_index"] == 0].set_index("sample_id").reindex(valid["sample_id"])
                change = representation_change(features, n_word, frame["text"].to_numpy()[positions].tolist(), valid["text"].tolist())
                representation_rows.append({"stage": "evaluation", "condition_or_policy": policy, "dataset": dataset, "split": split, "seed": seed, "n_pairs": change["n_pairs"], "changed_any_rate": change["changed_any_rate"], "changed_word_branch_rate": change["changed_word_rate"], "changed_char_branch_rate": change["changed_char_rate"], "mean_cosine_any": change["mean_cosine_any"], "mean_cosine_word": change["mean_cosine_word"], "mean_cosine_char": change["mean_cosine_char"], "perturbed_word_in_vocab_rate": float(first_ops["perturbed_in_word_vocab"].fillna(False).astype(bool).mean()), "original_word_in_vocab_rate": float(first_ops["original_word"].str.lower().map(lambda w: w in word_vocabulary).fillna(False).mean()), "fallback_random_rate": float(valid["fallback_random"].mean())})
                for key in ("clean", "augmented"):
                    name = condition_of[key]
                    score_original = scores_by[(name, stem)][positions]
                    pair = pd.DataFrame(
                        {
                            "dataset": dataset, "split": split, "sample_id": valid["parent_id"].astype(str).to_numpy(), "gold": gold, "score_attack": pert_scores[key].astype(float), "model_id": model_ids[key], "checkpoint_rev": checkpoint_revs[name], "seed": classifier_seed, "transform_id": transform_id, "parent_id": valid["parent_id"].astype(str).to_numpy(),
                            "condition": name, "policy": policy, "perturbation_seed": seed, "score_original": score_original.astype(float), "operator": first_ops["operator"].to_numpy(), "n_ops": valid["n_ops"].to_numpy(), "word_index": first_ops["word_index"].to_numpy(), "char_index": first_ops["char_index"].to_numpy(), "original_word": first_ops["original_word"].to_numpy(), "perturbed_word": first_ops["perturbed_word"].to_numpy(), "perturbed_in_word_vocab": first_ops["perturbed_in_word_vocab"].fillna(False).astype(bool).to_numpy(), "target_word": valid["target_word"].to_numpy(), "fallback_random": valid["fallback_random"].astype(bool).to_numpy(),
                            "representation_changed_any": change["changed_any"], "representation_changed_word": change["changed_word"], "representation_changed_char": change["changed_char"], "cosine_similarity": change["cosine_any"], "cosine_word_branch": change["cosine_word"], "cosine_char_branch": change["cosine_char"], "collides_with_train": valid["collides_with_train"].astype(bool).to_numpy(),
                        }
                    )
                    for policy_name, values in thresholds[key].items():
                        threshold = float(values["threshold"])
                        suffix = THRESHOLD_SUFFIX[policy_name]
                        pair[f"pred_original_{suffix}"] = (score_original >= threshold).astype(int)
                        pair[f"pred_perturbed_{suffix}"] = (pert_scores[key] >= threshold).astype(int)
                        pair[f"flipped_{suffix}"] = pair[f"pred_original_{suffix}"] != pair[f"pred_perturbed_{suffix}"]
                        masks = {"all_eligible": np.ones(len(pair), dtype=bool), "word_branch_changed": pair["representation_changed_word"].to_numpy(), "word_branch_unchanged": ~pair["representation_changed_word"].to_numpy(), "no_fallback": ~pair["fallback_random"].to_numpy(), "excluding_train_collisions": ~pair["collides_with_train"].to_numpy()}
                        for subset in subset_names:
                            mask = masks[subset]
                            if mask.sum() == 0 or (subset in ("no_fallback", "excluding_train_collisions") and mask.all()):
                                continue
                            paired_metric_rows.append({"model_id": model_ids[key], "model_key": key, "condition": name, "dataset": dataset, "split": split, "policy": policy, "perturbation_seed": seed, "threshold_policy": policy_name, "threshold": threshold, "subset": subset, "n_ineligible": int(stats["n_no_eligible_word"]), **pair_summary(gold[mask], score_original[mask], pert_scores[key][mask], threshold)})
                        if policy_name == "default_0.5":
                            for operator, group in pair.groupby("operator"):
                                g_gold = group["gold"].to_numpy()
                                operator_rows.append({"model_id": model_ids[key], "model_key": key, "dataset": dataset, "split": split, "policy": policy, "operator": operator, "n_pairs": int(len(group)), "flip_rate": float(group["flipped_0.5"].mean()), "mean_abs_delta_score": float((group["score_attack"] - group["score_original"]).abs().mean()), "mean_delta_score_attack": float((group["score_attack"] - group["score_original"]).to_numpy()[g_gold == 1].mean()) if (g_gold == 1).any() else None, "word_branch_changed_rate": float(group["representation_changed_word"].mean()), "perturbed_in_word_vocab_rate": float(group["perturbed_in_word_vocab"].mean())})
                    pair_parts.append(pair)
                if split == "test":
                    bootstrap_inputs.append({"kind": "perturbed", "dataset": dataset, "stem": stem, "policy": policy, "seed": seed, "gold": gold, "clean": pert_scores["clean"], "augmented": pert_scores["augmented"], "dup": pert_scores["dup"], "strata": None})
    pairs = pd.concat(pair_parts, ignore_index=True)
    paired_metrics = pd.DataFrame(paired_metric_rows)
    operator_metrics = pd.DataFrame(operator_rows)
    realizations = pd.DataFrame(realization_rows)
    summary_rows = []
    for (policy, dataset, split, model_key), group in realizations[realizations["policy"] == "random1"].groupby(["policy", "dataset", "split", "model_key"]):
        for statistic in ("mean", "sd"):
            aggregate = group.mean(numeric_only=True) if statistic == "mean" else group.std(numeric_only=True, ddof=1)
            summary_rows.append({"policy": policy, "perturbation_seed": statistic, "canonical": False, "dataset": dataset, "split": split, "model_id": group["model_id"].iloc[0], "model_key": model_key, "threshold": 0.5, "n_pairs": int(group["n_pairs"].iloc[0]), **{c: float(aggregate[c]) for c in ("orig_attack_recall", "pert_attack_recall", "orig_fpr", "pert_fpr", "flip_rate", "flip_attack_to_benign_rate", "n_flip_attack_to_benign", "n_flip_benign_to_attack")}})
    realizations = pd.concat([realizations, pd.DataFrame(summary_rows)], ignore_index=True)

    log.step("representation change of the training copies per condition")
    for name, info in training_copies.items():
        if not info["texts"]:
            continue
        change = representation_change(features, n_word, train["text"].to_numpy()[info["positions"]].tolist(), info["texts"])
        representation_rows.append({"stage": "train", "condition_or_policy": name, "dataset": "d1", "split": "train", "seed": None, "n_pairs": change["n_pairs"], "changed_any_rate": change["changed_any_rate"], "changed_word_branch_rate": change["changed_word_rate"], "changed_char_branch_rate": change["changed_char_rate"], "mean_cosine_any": change["mean_cosine_any"], "mean_cosine_word": change["mean_cosine_word"], "mean_cosine_char": change["mean_cosine_char"], "perturbed_word_in_vocab_rate": None, "original_word_in_vocab_rate": None, "fallback_random_rate": None})
    representation = pd.DataFrame(representation_rows)
    denominators = pd.DataFrame(denominator_rows)

    log.step(f"paired class-stratified bootstrap ({n_boot} resamples)")
    boot_seed = int(seeds["bootstrap"])
    for stem in ("d1_test", "d2_test", "d3_test", "d4_test"):
        dataset, _split = SPLIT_META[stem]
        strata = frames[stem]["trigger_count"].to_numpy() if (dataset == "d4" and "trigger_count" in frames[stem].columns) else None
        bootstrap_inputs.insert(0, {"kind": "clean", "dataset": dataset, "stem": stem, "policy": "clean", "seed": None, "gold": frames[stem]["label"].to_numpy(dtype=int), "clean": scores_by[("clean", stem)], "augmented": scores_by[(selected_condition, stem)], "dup": scores_by[("dup_x1", stem)] if "dup_x1" in classifiers else None, "strata": strata})
    bootstrap_rows: list[dict[str, Any]] = []
    for item in bootstrap_inputs:
        text_label = "clean text" if item["kind"] == "clean" else f"{item['policy']} typo"
        comparisons = [("augmented_minus_clean", "clean", "augmented", thresholds["clean"], thresholds["augmented"])]
        if item["dup"] is not None and item["dataset"] == "d1":
            comparisons.append(("augmented_minus_dup", "dup", "augmented", thresholds_by_condition["dup_x1"], thresholds["augmented"]))
        for comparison, a_key, b_key, thr_a, thr_b in comparisons:
            policies = ["default_0.5", "tF1"] + (["t5"] if (item["dataset"] == "d1" and item["policy"] in ("clean", "random1")) else [])
            for policy_name in policies:
                threshold_a = float(thr_a[policy_name]["threshold"])
                threshold_b = float(thr_b[policy_name]["threshold"])
                result = paired_bootstrap(item["gold"], item[a_key], item[b_key], threshold_a, threshold_b, n_boot, boot_seed, strata_by=item["strata"])
                for metric_name in ("macro_f1", "fpr", "attack_recall"):
                    if item["dataset"] == "d4" and metric_name != "fpr":
                        continue
                    is_primary = comparison == "augmented_minus_clean" and item["kind"] == "perturbed" and item["dataset"] == "d1" and item["policy"] == "random1" and policy_name == "default_0.5" and metric_name == "attack_recall"
                    bootstrap_rows.append({"role": "primary" if is_primary else "secondary", "comparison": comparison, "dataset": item["dataset"], "split": "test", "text_condition": item["kind"], "policy": item["policy"], "perturbation_seed": item["seed"], "text_label": text_label, "metric": metric_name, "strata": ("trigger_count" if item["strata"] is not None else ("benign rows" if metric_name == "fpr" else "attack rows" if metric_name == "attack_recall" else "attack and benign rows")), "threshold_policy": policy_name, "threshold_a": threshold_a, "threshold_b": threshold_b, "n_samples": result["n_samples"], "n_attack": result["n_attack"], "n_benign": result["n_benign"], "n_boot": result["n_boot"], "bootstrap_seed": boot_seed, **result[metric_name]})
    bootstrap = pd.DataFrame(bootstrap_rows)

    log.step("mechanism: coefficient shifts, branch contribution share, flips by word-branch change")
    feature_names = features.get_feature_names_out()
    coef_clean = np.asarray(classifiers["clean"].coef_[0], dtype=float)
    coef_aug = np.asarray(classifiers[selected_condition].coef_[0], dtype=float)
    delta_coef = coef_aug - coef_clean
    shift_rows: list[dict[str, Any]] = []
    for branch, columns in (("word", np.arange(0, n_word)), ("char", np.arange(n_word, len(coef_clean)))):
        for selection_name, ordering in (("largest_shift", columns[np.argsort(-np.abs(delta_coef[columns]))][:30]), ("top_attack_clean", columns[np.argsort(-coef_clean[columns])][:30]), ("top_benign_clean", columns[np.argsort(coef_clean[columns])][:30])):
            for rank, index in enumerate(ordering, start=1):
                shift_rows.append({"branch": branch, "selection": selection_name, "rank": rank, "feature": str(feature_names[index]).split("__", 1)[-1], "coef_clean": float(coef_clean[index]), "coef_augmented": float(coef_aug[index]), "delta": float(delta_coef[index])})
    coefficient_shift = pd.DataFrame(shift_rows)
    share_rows = []
    x_val = matrices["d1_validation"]
    for key in ("clean", "augmented"):
        weights = np.asarray(classifiers[condition_of[key]].coef_[0], dtype=float)
        contribution = np.abs(x_val.multiply(weights)).sum(axis=0)
        contribution = np.asarray(contribution).ravel()
        total = float(contribution.sum())
        for branch, columns in (("word", slice(0, n_word)), ("char", slice(n_word, None))):
            share_rows.append({"model_id": model_ids[key], "model_key": key, "branch": branch, "abs_contribution_sum": float(contribution[columns].sum()), "share": float(contribution[columns].sum() / total) if total else None})
    branch_share = pd.DataFrame(share_rows)

    log.step("latency, selected_models.csv, prediction files, QA sample")
    frozen_models = {key: common.assemble_pipeline(features, classifiers[condition_of[key]]) for key in ("clean", "augmented")}
    latency = {key: measure_latency(model, frames["d1_test"]["text"].tolist(), int(config["latency"]["warmup_samples"]), int(config["latency"]["measured_samples"])) for key, model in frozen_models.items()}
    selected_models = pd.DataFrame([{"pipeline": f"e5_{key}", "model_id": model_ids[key], "condition": condition_of[key], "selected_C": c_value, "dev_macro_f1_at_0.5": float(tuning.loc[tuning["condition"] == condition_of[key], "clean_val_macro_f1"].iloc[0]), "dev_perturbed_macro_f1_at_0.5": float(tuning.loc[tuning["condition"] == condition_of[key], "pert_val_macro_f1_mean"].iloc[0]), "features": int(fingerprint_before["total_features"]), "model_path": posix(model_paths[condition_of[key]], root), "checkpoint_rev": checkpoint_revs[condition_of[key]], "model_size_bytes": int(model_paths[condition_of[key]].stat().st_size), "batch1_latency_ms_median": latency[key][0], "batch1_latency_ms_p95": latency[key][1], "latency_device": "cpu", "latency_warmup": int(config["latency"]["warmup_samples"]), "latency_measured": int(config["latency"]["measured_samples"]), "latency_includes_vectorization": True} for key in ("clean", "augmented")])

    prediction_parts: list[pd.DataFrame] = []
    pair_prediction_parts: list[pd.DataFrame] = []
    for key in ("clean", "augmented"):
        name = condition_of[key]
        for policy_name, values in thresholds[key].items():
            threshold = float(values["threshold"])
            source = "shared default" if policy_name == "default_0.5" else evaluation["threshold_source"]
            for stem in ("d1_test", "d2_test", "d3_test"):
                dataset, split = SPLIT_META[stem]
                frame = frames[stem]
                scores = scores_by[(name, stem)]
                base = handoff_frame(dataset, split, frame["sample_id"].tolist(), frame["label"].to_numpy(dtype=int), scores, model_ids[key], checkpoint_revs[name], classifier_seed, "clean", pd.NA)
                base["threshold_policy"], base["threshold"], base["pred"] = policy_name, threshold, (scores >= threshold).astype(np.int64)
                base["score_type"], base["threshold_source"] = config["score_type"], source
                prediction_parts.append(base[PREDICTION_COLUMNS + ["score_type", "threshold_source"]])
            subset = pairs[pairs["model_id"] == model_ids[key]]
            block = subset[HANDOFF_COLUMNS].copy()
            block["threshold_policy"], block["threshold"], block["pred"] = policy_name, threshold, (subset["score_attack"].to_numpy() >= threshold).astype(np.int64)
            block["score_original"], block["pred_original"] = subset["score_original"].to_numpy(), (subset["score_original"].to_numpy() >= threshold).astype(np.int64)
            block["score_type"], block["threshold_source"] = config["score_type"], source
            pair_prediction_parts.append(block[PREDICTION_COLUMNS + ["score_original", "pred_original", "score_type", "threshold_source"]])
        atomic_parquet(pd.concat([p for p in prediction_parts if p["model_id"].iloc[0] == model_ids[key]], ignore_index=True), predictions_dir / f"{model_ids[key]}_predictions.parquet")
    atomic_parquet(pd.concat(pair_prediction_parts, ignore_index=True), predictions_dir / "e5_paired_predictions.parquet")

    manifest = pd.concat(manifest_parts, ignore_index=True)
    ops_all = pd.concat(ops_parts, ignore_index=True)
    parent_text = pd.concat([frames[s][["sample_id", "text"]] for s in SPLIT_META]).drop_duplicates("sample_id").set_index("sample_id")["text"]
    qa_config = config["qa"]
    qa_rng = np.random.default_rng(int(seeds["qa_sample"]))
    qa_sample = draw_qa_sample(manifest, ops_all, parent_text, selected_condition, int(seeds["augment_select"][0]), canonical_seed, qa_config, int(seeds["qa_sample"]))
    for column in ("qa_plausible_typo", "qa_word_recognizable", "qa_meaning_unchanged", "qa_ok", "qa_note"):
        qa_sample[column] = ""
    common.write_csv_for_humans(qa_sample, results_dir / "qa/typo_qa_sample.csv")  # UTF-8 with BOM so Excel/WPS keep the text intact

    log.step("writing tables, manifests and handoff files")
    atomic_csv(metrics, results_dir / "metrics.csv")
    atomic_csv(paired_metrics, results_dir / "paired_metrics.csv")
    atomic_csv(operator_metrics, supplementary_dir / "paired_by_operator.csv")
    atomic_csv(representation, results_dir / "representation_change.csv")
    atomic_csv(bootstrap, results_dir / "bootstrap.csv")
    atomic_csv(false_alarms, results_dir / "notinject_false_alarms.csv")
    atomic_csv(denominators, results_dir / "paired_denominators.csv")
    atomic_csv(selected_models, results_dir / "selected_models.csv")
    atomic_json(audit, results_dir / "duplicate_equivalence_check.json")
    atomic_csv(all_conditions, supplementary_dir / "all_conditions_test.csv")
    atomic_csv(realizations, supplementary_dir / "eval_realizations.csv")
    atomic_csv(coefficient_shift, supplementary_dir / "coefficient_shift.csv")
    atomic_csv(branch_share, supplementary_dir / "branch_share.csv")
    atomic_parquet(pairs, handoff_dir / "e5_typo_pairs.parquet")
    atomic_parquet(manifest, handoff_dir / "augmentation_manifest.parquet")
    atomic_parquet(ops_all, results_dir / "augmentation_ops.parquet")
    atomic_csv(tuning, handoff_dir / "tuning.csv")
    checkpoint_lines = []
    for key in ("clean", "augmented"):
        name = condition_of[key]
        checkpoint_lines += [f"[{key}]", f"model_id         = {model_ids[key]}", f"condition        = {name}", f"checkpoint_file  = {posix(model_paths[name], root)}", f"checkpoint_rev   = {checkpoint_revs[name]}", f"coef_sha256      = {fit_info[name]['coef_sha256']}", ""]
    environment = common.environment_info()
    checkpoint_lines += [f"features         = {features_provenance}; vocabulary/IDF hashes in selected_config.json", f"environment      = python {environment['python']}, scikit-learn {environment['scikit_learn']}, joblib {environment['joblib']}, numpy {environment['numpy']}", "score_type       = predict_proba_class_1_uncalibrated"]
    if not official:
        checkpoint_lines += ["", "NON-OFFICIAL development run (" + "; ".join(development_overrides) + "): do not hand these files to E6."]
    checkpoint_lines += ["", "Load with (from the project root):", "  import joblib", "  model = joblib.load('models/e5/clean.joblib')  # or the augmented condition's file", "  classes = list(model.named_steps['classifier'].classes_)", "  score_attack = model.predict_proba(texts)[:, classes.index(1)]   # uncalibrated", "", "Never call fit, fit_transform or clone on these pipelines: fit would refit the frozen E1 vectorizers."]
    write_text("\n".join(checkpoint_lines) + "\n", handoff_dir / "checkpoint_path.txt")

    figure_record: dict[str, Any]
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale in figures_dir.glob("*.png"):
        stale.unlink()  # never leave a figure from an earlier run next to this run's tables
    if args.no_figures:
        figure_record = {"skipped": True}
    else:
        log.step("rendering figures")
        footer = f"threat model: {config['threat_model']}  |  clean {checkpoint_revs['clean'][:12]}  augmented ({selected_condition}) {checkpoint_revs[selected_condition][:12]}"
        figures.plot_selection(tuning, selected_condition, guard_value, figures_dir / "fig1_selection.png", footer)
        figures.plot_robustness(paired_metrics, model_ids, figures_dir / "fig2_typo_robustness.png", footer)
        figures.plot_bootstrap(bootstrap, figures_dir / "fig3_bootstrap_deltas.png", footer)
        figures.plot_d4(false_alarms, figures_dir / "fig4_notinject_false_alarms.png", footer)
        figures.plot_mechanism(coefficient_shift, branch_share, figures_dir / "fig5_mechanism.png", footer)
        figure_record = {"skipped": False, "footer": footer, "n_boot": n_boot, "files": {path.name: sha256_file(path) for path in sorted(figures_dir.glob("*.png"))}}

    log.step("selected_config.json (final) and run_metadata.json")
    selected_config.update({
        "frozen_features_unchanged_after_training": fingerprint_after == fingerprint_before,
        "clean_reproduction_full": reproduction_full,
        "provisional_thresholds": {model_ids[key]: thresholds[key] for key in ("clean", "augmented")},
        "provisional_thresholds_by_condition": thresholds_by_condition,
        "provisional_threshold_note": evaluation["threshold_source"],
        "evaluation_perturbation": {"canonical_seed": canonical_seed, "extra_seeds": extra_seeds, "policies": evaluation["perturbation_policies"], "splits": evaluation["perturbed_splits"]},
        "d4_slice_columns": slice_columns,
        "latency": {key: {"batch1_latency_ms_median": latency[key][0], "batch1_latency_ms_p95": latency[key][1], **config["latency"]} for key in ("clean", "augmented")},
        "duplicate_equivalence_total_collisions": audit["total_collisions"],
    })
    atomic_json(selected_config, results_dir / "selected_config.json")
    atomic_json(selected_config, handoff_dir / "selected_config.json")
    run_metadata = {
        "experiment_id": config["experiment_id"], "status": "complete", "official_run": official, "frozen_at_utc": frozen_at, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_seconds": round(time.perf_counter() - log.started, 1),
        "config_sha256": sha256_file(config_path), "e1_config_sha256": sha256_file(root / config["frozen"]["e1_config"]), "data_manifest_sha256": integrity["manifest_sha256"], "data_integrity": integrity, "dataset_revisions": load_json(root / "data/audit/manifest.json").get("dataset_revisions"),
        "environment": environment, "sklearn_warnings_on_load": captured_warnings, "vectorizer_source": args.vectorizer_source, "frozen_e1_model": frozen_info, "features_provenance": features_provenance, "development_overrides": development_overrides, "figures": figure_record, "bootstrap": {"n_boot": n_boot, "seed": boot_seed}, "threat_model": config["threat_model"], "score_type": config["score_type"],
        "selected_condition": selected_condition, "models": {key: {"model_id": model_ids[key], "condition": condition_of[key], "checkpoint_rev": checkpoint_revs[condition_of[key]], "model_path": posix(model_paths[condition_of[key]], root)} for key in ("clean", "augmented")},
        "scores": {
            "clean_rows": {model_ids[key]: {"file": f"results/e5/handoff/{model_ids[key]}_scores.parquet", "rows": 5086, "schema": HANDOFF_COLUMNS, "transform_id": "clean"} for key in ("clean", "augmented")},
            "typo_pairs": {"file": "results/e5/handoff/e5_typo_pairs.parquet", "rows": int(len(pairs)), "note": "perturbed rows only; sample_id = processed id, parent_id = sample_id, score_attack = perturbed score, score_original = clean-row score; never concatenate with the clean files", "transform_ids": sorted(pairs["transform_id"].unique().tolist())},
            "predictions": {"files": [f"results/e5/predictions/{model_ids[k]}_predictions.parquet" for k in ("clean", "augmented")] + ["results/e5/predictions/e5_paired_predictions.parquet"], "schema": PREDICTION_COLUMNS + ["score_type", "threshold_source"], "note": "README section 9 schema with provisional tF1/t5; regenerate from E6's thresholds when they exist"},
            "manifest": {"file": "results/e5/handoff/augmentation_manifest.parquet", "rows": int(len(manifest)), "operations_file": "results/e5/augmentation_ops.parquet"},
            "higher_means": "more attack-like",
        },
        "reproduction": {"command": "python experiments/e5/run_e5.py", "verify_command": "python experiments/e5/verify_e5.py", "report_command": "python experiments/e5/write_report.py", "qa_command": "python experiments/e5/qa_summary.py --reviewer <name>", "main_table_command": "python experiments/e5/build_main_table.py"},
        "steps": log.steps,
        "notes": [
            f"Features: {features_provenance}. Only the Logistic Regression was retrained, with E1's hyperparameters and seed.",
            "Conditions differ only in the copies appended to D1 Train (none, exact duplicates, or typo-perturbed copies).",
            "The primary augmented condition was selected on D1 Validation only (clean and five typo-perturbed draws); selected_config.json was frozen before any test split or D4 text was used for scoring (the data gate hashes all six files first).",
            "Training (6701/6702), selection (6501-6505) and evaluation (6511-6515) perturbation seeds are disjoint; the classifier seed 6405 is E1's.",
            "Class-1 scores are uncalibrated Logistic Regression predict_proba outputs, not probabilities of maliciousness.",
            "tF1 and t5 reported here are provisional; E6 owns the authoritative thresholds.",
        ],
    }
    if not official:
        reason = "; ".join(development_overrides)
        run_metadata["notes"].insert(0, f"NON-OFFICIAL development run ({reason}). Do not hand these files to E6.")
        for target in (results_dir / "NON_OFFICIAL_RUN.txt", handoff_dir / "NON_OFFICIAL_RUN.txt"):
            write_text(f"This directory was produced by a development run with overrides: {reason}. {features_provenance}. Re-run `python experiments/e5/run_e5.py` without overrides before handing anything to E6.\n", target)
    else:
        for target in (results_dir / "NON_OFFICIAL_RUN.txt", handoff_dir / "NON_OFFICIAL_RUN.txt"):
            if target.exists():
                target.unlink()
    atomic_json(run_metadata, results_dir / "run_metadata.json")
    atomic_json(run_metadata, handoff_dir / "run_metadata.json")

    log.step("complete")
    print(tuning[["condition", "n_training_rows", "clean_val_macro_f1", "pert_val_macro_f1_mean", "pert_val_macro_f1_sd", "targeted_val_attack_recall", "satisfies_guard", "is_selected"]].to_string(index=False), flush=True)
    print(metrics[metrics["threshold_policy"] == "default_0.5"][["model_key", "dataset", "split", "macro_f1", "attack_recall", "fpr"]].to_string(index=False), flush=True)
    primary = bootstrap[bootstrap["role"] == "primary"].iloc[0]
    print(f"primary comparison: delta attack recall (augmented - clean), D1 test, random typo, threshold 0.5 = {100 * primary['point_estimate']:+.2f} pp (95% CI {100 * primary['ci_low']:+.2f} to {100 * primary['ci_high']:+.2f})", flush=True)


if __name__ == "__main__":
    main()
