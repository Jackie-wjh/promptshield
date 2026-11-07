#!/usr/bin/env python3
"""Run a bounded exploratory E1 tuning sweep without changing the frozen main run."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e1.run_e1 import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    attack_scores,
    metric_record,
    select_thresholds,
)
from scripts.check_data_integrity import run_checks


SEED = 6405
C_GRID = [0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.5]


VARIANTS: dict[str, list[dict[str, Any]]] = {
    "word": [
        {"name": "baseline", "min_df": 1, "sublinear_tf": False},
        {"name": "sublinear", "min_df": 1, "sublinear_tf": True},
        {"name": "min_df2", "min_df": 2, "sublinear_tf": False},
        {"name": "min_df2_sublinear", "min_df": 2, "sublinear_tf": True},
    ],
    "char": [
        {"name": "baseline", "analyzer": "char", "min_df": 1, "sublinear_tf": False},
        {"name": "sublinear", "analyzer": "char", "min_df": 1, "sublinear_tf": True},
        {"name": "min_df2", "analyzer": "char", "min_df": 2, "sublinear_tf": False},
        {"name": "char_wb", "analyzer": "char_wb", "min_df": 1, "sublinear_tf": False},
        {"name": "char_wb_sublinear", "analyzer": "char_wb", "min_df": 1, "sublinear_tf": True},
    ],
    "combined": [
        {
            "name": "baseline",
            "min_df": 1,
            "sublinear_tf": False,
            "char_analyzer": "char",
        },
        {
            "name": "sublinear",
            "min_df": 1,
            "sublinear_tf": True,
            "char_analyzer": "char",
        },
        {
            "name": "min_df2",
            "min_df": 2,
            "sublinear_tf": False,
            "char_analyzer": "char",
        },
        {
            "name": "min_df2_sublinear",
            "min_df": 2,
            "sublinear_tf": True,
            "char_analyzer": "char",
        },
        {
            "name": "char_wb",
            "min_df": 1,
            "sublinear_tf": False,
            "char_analyzer": "char_wb",
        },
        {
            "name": "char_wb_sublinear",
            "min_df": 1,
            "sublinear_tf": True,
            "char_analyzer": "char_wb",
        },
    ],
}


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def word_vectorizer(min_df: int, sublinear_tf: bool) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=min_df,
        lowercase=True,
        norm="l2",
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )


def char_vectorizer(
    analyzer: str, min_df: int, sublinear_tf: bool
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(3, 5),
        max_features=50000,
        min_df=min_df,
        lowercase=True,
        norm="l2",
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )


def make_features(representation: str, variant: dict[str, Any]) -> Any:
    if representation == "word":
        return word_vectorizer(variant["min_df"], variant["sublinear_tf"])
    if representation == "char":
        return char_vectorizer(
            variant["analyzer"], variant["min_df"], variant["sublinear_tf"]
        )
    if representation == "combined":
        return FeatureUnion(
            [
                (
                    "word",
                    word_vectorizer(variant["min_df"], variant["sublinear_tf"]),
                ),
                (
                    "char",
                    char_vectorizer(
                        variant["char_analyzer"],
                        variant["min_df"],
                        variant["sublinear_tf"],
                    ),
                ),
            ]
        )
    raise ValueError(representation)


def make_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        solver="liblinear",
        l1_ratio=0.0,
        class_weight=None,
        max_iter=1000,
        random_state=SEED,
    )


def paired_stratified_bootstrap(
    gold: np.ndarray,
    baseline_prediction: np.ndarray,
    tuned_prediction: np.ndarray,
    iterations: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    class_indexes = [np.flatnonzero(gold == label) for label in (0, 1)]
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indexes, size=len(indexes), replace=True) for indexes in class_indexes]
        )
        baseline = f1_score(
            gold[sampled], baseline_prediction[sampled], average="macro", zero_division=0
        )
        tuned = f1_score(
            gold[sampled], tuned_prediction[sampled], average="macro", zero_division=0
        )
        deltas[iteration] = tuned - baseline
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    root = PROJECT_ROOT
    integrity = run_checks(root)
    if integrity["status"] != "passed":
        raise RuntimeError("Data integrity gate failed")

    main_results_dir = root / "results/e1"
    main_hash_before = tree_hash(main_results_dir)
    processed = root / "data/processed"
    output = root / "results/e1_exploratory"
    train = pd.read_parquet(processed / "d1_train.parquet")
    validation = pd.read_parquet(processed / "d1_validation.parquet")
    x_train = train["text"].tolist()
    y_train = train["label"].to_numpy(dtype=int)
    x_validation = validation["text"].tolist()
    y_validation = validation["label"].to_numpy(dtype=int)

    tuning_rows: list[dict[str, Any]] = []
    selected_models: dict[str, Pipeline] = {}
    selected_metadata: dict[str, dict[str, Any]] = {}

    # Model selection phase. No test file is loaded before all choices are frozen.
    for representation, variants in VARIANTS.items():
        best_key: tuple[float, int, int, float] | None = None
        print(f"Exploring {representation}", flush=True)
        for variant_index, variant in enumerate(variants):
            features = make_features(representation, variant)
            started = time.perf_counter()
            train_matrix = features.fit_transform(x_train)
            validation_matrix = features.transform(x_validation)
            vectorizer_seconds = time.perf_counter() - started
            n_features = int(train_matrix.shape[1])
            for c_value in C_GRID:
                classifier = make_classifier(c_value)
                started = time.perf_counter()
                classifier.fit(train_matrix, y_train)
                fit_seconds = time.perf_counter() - started
                attack_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
                scores = classifier.predict_proba(validation_matrix)[:, attack_index]
                metrics = metric_record(y_validation, scores, 0.5)
                tuning_rows.append(
                    {
                        "pipeline": representation,
                        "variant": variant["name"],
                        "C": c_value,
                        "dev_macro_f1": metrics["macro_f1"],
                        "dev_attack_recall": metrics["attack_recall"],
                        "dev_fpr": metrics["fpr"],
                        "dev_auroc": metrics["auroc"],
                        "features": n_features,
                        "vectorizer_seconds": vectorizer_seconds,
                        "classifier_fit_seconds": fit_seconds,
                    }
                )
                # Exact score ties prefer fewer features, then the baseline variant,
                # then smaller C. This is fixed before loading any test file.
                selection_key = (
                    float(metrics["macro_f1"]),
                    -n_features,
                    -variant_index,
                    -float(c_value),
                )
                if best_key is None or selection_key > best_key:
                    best_key = selection_key
                    selected_models[representation] = Pipeline(
                        [("features", features), ("classifier", classifier)]
                    )
                    selected_metadata[representation] = {
                        "pipeline": representation,
                        "variant": variant["name"],
                        "variant_parameters": variant,
                        "C": c_value,
                        "dev_macro_f1_at_0.5": metrics["macro_f1"],
                        "features": n_features,
                    }
            print(
                f"  {variant['name']}: features={n_features:,} "
                f"best_so_far={best_key[0]:.6f}",
                flush=True,
            )

    # Evaluation phase starts only after all exploratory configurations are frozen.
    evaluation = {
        "d1": pd.read_parquet(processed / "d1_test.parquet"),
        "d2": pd.read_parquet(processed / "d2_test.parquet"),
        "d3": pd.read_parquet(processed / "d3_test.parquet"),
    }
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for representation, model in selected_models.items():
        validation_scores = attack_scores(model, validation["text"])
        thresholds = select_thresholds(y_validation, validation_scores)
        metadata = selected_metadata[representation]
        for policy, values in thresholds.items():
            threshold_rows.append(
                {
                    "pipeline": representation,
                    "variant": metadata["variant"],
                    "C": metadata["C"],
                    "policy": policy,
                    **values,
                }
            )

        prediction_parts: list[pd.DataFrame] = []
        baseline_predictions = pd.read_parquet(
            main_results_dir / "predictions" / f"{representation}_predictions.parquet"
        )
        for dataset, frame in evaluation.items():
            gold = frame["label"].to_numpy(dtype=int)
            scores = attack_scores(model, frame["text"])
            threshold = float(thresholds["tF1"]["threshold"])
            prediction = (scores >= threshold).astype(int)
            metrics = metric_record(gold, scores, threshold)
            metric_rows.append(
                {
                    "pipeline": representation,
                    "variant": metadata["variant"],
                    "C": metadata["C"],
                    "dataset": dataset,
                    "threshold_policy": "tF1",
                    "threshold": threshold,
                    **metrics,
                }
            )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "split": "test",
                        "sample_id": frame["sample_id"].tolist(),
                        "gold": gold,
                        "score_attack": scores,
                        "threshold_policy": "tF1",
                        "threshold": threshold,
                        "pred": prediction,
                        "pipeline": representation,
                        "variant": metadata["variant"],
                        "C": metadata["C"],
                        "seed": SEED,
                    }
                )
            )

            baseline = baseline_predictions[
                baseline_predictions["dataset"].eq(dataset)
                & baseline_predictions["threshold_policy"].eq("tF1")
            ].reset_index(drop=True)
            if baseline["sample_id"].tolist() != frame["sample_id"].tolist():
                raise AssertionError("Baseline prediction IDs do not align")
            baseline_metrics = metric_record(
                gold,
                baseline["score_attack"].to_numpy(dtype=float),
                float(baseline["threshold"].iloc[0]),
            )
            lower, upper = paired_stratified_bootstrap(
                gold,
                baseline["pred"].to_numpy(dtype=int),
                prediction,
            )
            comparison_rows.append(
                {
                    "pipeline": representation,
                    "dataset": dataset,
                    "baseline_macro_f1": baseline_metrics["macro_f1"],
                    "tuned_macro_f1": metrics["macro_f1"],
                    "delta_macro_f1": metrics["macro_f1"] - baseline_metrics["macro_f1"],
                    "delta_macro_f1_ci_low": lower,
                    "delta_macro_f1_ci_high": upper,
                    "baseline_attack_recall": baseline_metrics["attack_recall"],
                    "tuned_attack_recall": metrics["attack_recall"],
                    "delta_attack_recall": metrics["attack_recall"]
                    - baseline_metrics["attack_recall"],
                    "baseline_fpr": baseline_metrics["fpr"],
                    "tuned_fpr": metrics["fpr"],
                    "delta_fpr": metrics["fpr"] - baseline_metrics["fpr"],
                }
            )

        atomic_parquet(
            pd.concat(prediction_parts, ignore_index=True),
            output / "predictions" / f"{representation}_predictions.parquet",
        )

    tuning = pd.DataFrame(tuning_rows).sort_values(["pipeline", "variant", "C"])
    selected = pd.DataFrame(
        [
            {
                **{key: value for key, value in metadata.items() if key != "variant_parameters"},
                "variant_parameters": json.dumps(
                    metadata["variant_parameters"], sort_keys=True
                ),
            }
            for metadata in selected_metadata.values()
        ]
    ).sort_values("pipeline")
    metrics = pd.DataFrame(metric_rows).sort_values(["pipeline", "dataset"])
    comparisons = pd.DataFrame(comparison_rows).sort_values(["pipeline", "dataset"])
    thresholds_frame = pd.DataFrame(threshold_rows).sort_values(["pipeline", "policy"])
    atomic_csv(tuning, output / "tuning.csv")
    atomic_csv(selected, output / "selected.csv")
    atomic_csv(metrics, output / "metrics_tF1.csv")
    atomic_csv(comparisons, output / "comparison_to_main.csv")
    atomic_csv(thresholds_frame, output / "thresholds.csv")

    main_hash_after = tree_hash(main_results_dir)
    if main_hash_after != main_hash_before:
        raise AssertionError("Frozen main E1 results changed during exploratory tuning")
    summary = {
        "status": "complete",
        "scope": "exploratory_only",
        "selection_data": "D1 validation only",
        "external_test_usage": "evaluation after all configurations were frozen",
        "c_grid": C_GRID,
        "candidate_fits": len(tuning),
        "selected": selected.to_dict(orient="records"),
        "main_e1_tree_sha256_before": main_hash_before,
        "main_e1_tree_sha256_after": main_hash_after,
        "main_e1_unchanged": True,
        "notes": [
            "The original E1 run remains the primary prespecified result.",
            "This sweep must be labeled exploratory because it was added after the main run.",
            "Exact validation-score ties prefer fewer features, then the baseline variant, then smaller C.",
            "No D2 or D3 label was used for model or threshold selection.",
        ],
    }
    atomic_json(summary, output / "summary.json")
    print("Exploratory tuning complete", flush=True)
    print(selected.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
