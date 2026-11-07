#!/usr/bin/env python3
"""Run E1 Word, Char, and Combined TF-IDF Logistic Regression experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_data_integrity import run_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/e1.json")
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)


def make_vectorizer(spec: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=spec["analyzer"],
        ngram_range=tuple(spec["ngram_range"]),
        max_features=int(spec["max_features"]),
        lowercase=bool(spec["lowercase"]),
        norm=spec["norm"],
        sublinear_tf=bool(spec["sublinear_tf"]),
        dtype=np.float32,
    )


def make_pipeline(name: str, c_value: float, config: dict[str, Any]) -> Pipeline:
    specs = config["vectorizers"]
    if name == "word":
        features: Any = make_vectorizer(specs["word"])
    elif name == "char":
        features = make_vectorizer(specs["char"])
    elif name == "combined":
        features = FeatureUnion(
            [
                ("word", make_vectorizer(specs["word"])),
                ("char", make_vectorizer(specs["char"])),
            ]
        )
    else:
        raise ValueError(f"Unknown representation: {name}")
    classifier = LogisticRegression(
        C=float(c_value),
        solver=config["classifier"]["solver"],
        l1_ratio=float(config["classifier"]["l1_ratio"]),
        class_weight=config["classifier"]["class_weight"],
        max_iter=int(config["classifier"]["max_iter"]),
        random_state=int(config["seed"]),
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def attack_scores(model: Pipeline, texts: pd.Series | list[str]) -> np.ndarray:
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


def feature_dimensions(model: Pipeline, representation: str) -> list[dict[str, Any]]:
    features = model.named_steps["features"]
    if representation == "combined":
        rows = []
        total = 0
        for branch_name, branch in features.transformer_list:
            count = len(branch.vocabulary_)
            total += count
            rows.append({"pipeline": representation, "branch": branch_name, "features": count})
        rows.append({"pipeline": representation, "branch": "total", "features": total})
        return rows
    count = len(features.vocabulary_)
    return [{"pipeline": representation, "branch": representation, "features": count}]


def global_coefficients(
    model: Pipeline, representation: str, top_n: int = 30
) -> list[dict[str, Any]]:
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    rows: list[dict[str, Any]] = []
    for direction, indexes in (
        ("attack", np.argsort(coefficients)[-top_n:][::-1]),
        ("benign", np.argsort(coefficients)[:top_n]),
    ):
        for rank, index in enumerate(indexes, start=1):
            rows.append(
                {
                    "pipeline": representation,
                    "direction": direction,
                    "rank": rank,
                    "feature": str(names[index]),
                    "coefficient": float(coefficients[index]),
                }
            )
    return rows


def fixed_explanation_rows(test: pd.DataFrame) -> pd.DataFrame:
    parts = [test[test["label"].eq(label)].sort_values("source_row").head(3) for label in (0, 1)]
    return pd.concat(parts, ignore_index=True).sort_values(["label", "source_row"]).reset_index(drop=True)


def local_contributions(
    model: Pipeline,
    samples: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    names = features.get_feature_names_out()
    coefficients = classifier.coef_[0]
    matrix = features.transform(samples["text"].tolist()).tocsr()
    scores = attack_scores(model, samples["text"])
    rows: list[dict[str, Any]] = []
    for sample_position, sample in samples.iterrows():
        vector = matrix.getrow(sample_position)
        values = vector.data
        indexes = vector.indices
        contributions = values * coefficients[indexes]
        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        for rank, position in enumerate(order, start=1):
            feature_index = indexes[position]
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "source_row": int(sample["source_row"]),
                    "gold": int(sample["label"]),
                    "score_attack": float(scores[sample_position]),
                    "rank": rank,
                    "feature": str(names[feature_index]),
                    "feature_value": float(values[position]),
                    "coefficient": float(coefficients[feature_index]),
                    "contribution": float(contributions[position]),
                }
            )
    return pd.DataFrame(rows)


def plot_local_contributions(local: pd.DataFrame, output_path: Path) -> None:
    attack_sample = local[local["gold"].eq(1)]["sample_id"].iloc[0]
    plot_data = local[local["sample_id"].eq(attack_sample)].copy()
    plot_data = plot_data.sort_values("contribution")
    labels = [feature[:48] for feature in plot_data["feature"]]
    colors = ["#B23A48" if value > 0 else "#3B6EA8" for value in plot_data["contribution"]]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.barh(labels, plot_data["contribution"], color=colors)
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_xlabel("Local contribution  TF IDF value x logistic coefficient")
    score = plot_data["score_attack"].iloc[0]
    axis.set_title(f"Combined LR local explanation\n{attack_sample}  gold 1  attack score {score:.3f}")
    axis.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_tuning(tuning: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for pipeline, frame in tuning.groupby("pipeline", sort=False):
        frame = frame.sort_values("C")
        axis.plot(frame["C"], frame["dev_macro_f1"], marker="o", linewidth=2, label=pipeline)
    axis.set_xscale("log")
    axis.set_xlabel("Logistic regression C")
    axis.set_ylabel("D1 validation Macro F1 at threshold 0.5")
    axis.set_title("E1 controlled representation comparison")
    axis.set_xticks([0.1, 1.0, 10.0], labels=["0.1", "1", "10"])
    axis.grid(color="#E0E0E0", linewidth=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def measure_latency(
    model: Pipeline, texts: list[str], warmup_samples: int, measured_samples: int
) -> tuple[float, float]:
    for text in texts[:warmup_samples]:
        attack_scores(model, [text])
    durations = []
    for text in texts[warmup_samples : warmup_samples + measured_samples]:
        started = time.perf_counter()
        attack_scores(model, [text])
        durations.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(durations)), float(np.percentile(durations, 95))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processed = project_root / "data/processed"
    results_dir = project_root / "results/e1"
    predictions_dir = results_dir / "predictions"
    models_dir = project_root / "models/e1"

    integrity = run_checks(project_root)
    require_status = integrity.get("status")
    if require_status != "passed":
        raise RuntimeError(f"Data integrity gate failed: {require_status}")

    train = pd.read_parquet(processed / "d1_train.parquet")
    validation = pd.read_parquet(processed / "d1_validation.parquet")
    evaluation = {
        "d1": pd.read_parquet(processed / "d1_test.parquet"),
        "d2": pd.read_parquet(processed / "d2_test.parquet"),
        "d3": pd.read_parquet(processed / "d3_test.parquet"),
    }
    x_train = train["text"].tolist()
    y_train = train["label"].to_numpy(dtype=int)
    x_validation = validation["text"].tolist()
    y_validation = validation["label"].to_numpy(dtype=int)

    tuning_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    main_results: list[dict[str, Any]] = []
    local_frame: pd.DataFrame | None = None

    data_manifest_sha = sha256_file(project_root / "data/audit/manifest.json")
    config_sha = sha256_file(config_path)
    experiment_revision = hashlib.sha256(
        f"{config_sha}:{data_manifest_sha}".encode("utf-8")
    ).hexdigest()

    for representation in ("word", "char", "combined"):
        best_model: Pipeline | None = None
        best_key: tuple[float, float] | None = None
        best_c: float | None = None
        print(f"Tuning {representation}", flush=True)
        for c_value in config["c_grid"]:
            model = make_pipeline(representation, float(c_value), config)
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            validation_scores = attack_scores(model, x_validation)
            metrics = metric_record(y_validation, validation_scores, 0.5)
            n_features = int(model.named_steps["classifier"].coef_.shape[1])
            n_iter = int(model.named_steps["classifier"].n_iter_.max())
            tuning_rows.append(
                {
                    "pipeline": representation,
                    "C": float(c_value),
                    "dev_macro_f1": metrics["macro_f1"],
                    "dev_attack_recall": metrics["attack_recall"],
                    "dev_fpr": metrics["fpr"],
                    "dev_auroc": metrics["auroc"],
                    "fit_seconds": fit_seconds,
                    "n_features": n_features,
                    "n_iter": n_iter,
                    "selection_threshold": 0.5,
                }
            )
            selection_key = (float(metrics["macro_f1"]), -float(c_value))
            print(
                f"  C={c_value:g} dev_macro_f1={metrics['macro_f1']:.6f} "
                f"features={n_features:,} fit={fit_seconds:.2f}s",
                flush=True,
            )
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_model = model
                best_c = float(c_value)

        if best_model is None or best_c is None:
            raise RuntimeError(f"No model selected for {representation}")

        validation_scores = attack_scores(best_model, x_validation)
        thresholds = select_thresholds(y_validation, validation_scores)
        model_path = models_dir / f"{representation}.joblib"
        atomic_joblib(best_model, model_path)
        checkpoint_revision = sha256_file(model_path)
        model_id = f"e1_{representation}_tfidf_lr"

        for policy, values in thresholds.items():
            threshold_rows.append(
                {
                    "pipeline": representation,
                    "model_id": model_id,
                    "C": best_c,
                    "policy": policy,
                    **values,
                }
            )

        dimension_rows.extend(feature_dimensions(best_model, representation))
        coefficient_rows.extend(global_coefficients(best_model, representation))

        latency_config = config["latency"]
        latency_texts = evaluation["d1"]["text"].tolist()
        latency_median, latency_p95 = measure_latency(
            best_model,
            latency_texts,
            int(latency_config["warmup_samples"]),
            int(latency_config["measured_samples"]),
        )
        selected_rows.append(
            {
                "pipeline": representation,
                "model_id": model_id,
                "selected_C": best_c,
                "dev_macro_f1_at_0.5": best_key[0],
                "features": int(best_model.named_steps["classifier"].coef_.shape[1]),
                "model_path": str(model_path.relative_to(project_root)),
                "checkpoint_rev": checkpoint_revision,
                "model_size_bytes": model_path.stat().st_size,
                "batch1_latency_ms_median": latency_median,
                "batch1_latency_ms_p95": latency_p95,
            }
        )

        prediction_parts: list[pd.DataFrame] = []
        for dataset, frame in evaluation.items():
            gold = frame["label"].to_numpy(dtype=int)
            scores = attack_scores(best_model, frame["text"])
            for policy, values in thresholds.items():
                threshold = float(values["threshold"])
                predictions = (scores >= threshold).astype(int)
                metrics = metric_record(gold, scores, threshold)
                record = {
                    "pipeline": representation,
                    "model_id": model_id,
                    "C": best_c,
                    "dataset": dataset,
                    "split": "test",
                    "threshold_policy": policy,
                    "threshold": threshold,
                    **metrics,
                }
                metric_rows.append(record)
                if policy == "tF1":
                    main_results.append(record)

                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset,
                            "split": "test",
                            "sample_id": frame["sample_id"].tolist(),
                            "gold": gold,
                            "score_attack": scores,
                            "score_type": "predict_proba_class_1_uncalibrated",
                            "threshold_policy": policy,
                            "threshold": threshold,
                            "pred": predictions,
                            "model_id": model_id,
                            "checkpoint_rev": checkpoint_revision,
                            "seed": int(config["seed"]),
                            "transform_id": "clean",
                            "parent_id": pd.Series([None] * len(frame), dtype="string"),
                        }
                    )
                )
        atomic_parquet(
            pd.concat(prediction_parts, ignore_index=True),
            predictions_dir / f"{representation}_predictions.parquet",
        )

        if representation == "combined":
            samples = fixed_explanation_rows(evaluation["d1"])
            local_frame = local_contributions(
                best_model,
                samples,
                int(config["local_explanations"]["top_features_per_sample"]),
            )

    tuning = pd.DataFrame(tuning_rows).sort_values(["pipeline", "C"]).reset_index(drop=True)
    thresholds_frame = pd.DataFrame(threshold_rows).sort_values(["pipeline", "policy"])
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["pipeline", "dataset", "threshold_policy"]
    )
    main_results_frame = pd.DataFrame(main_results).sort_values(["pipeline", "dataset"])
    selected_frame = pd.DataFrame(selected_rows).sort_values("pipeline")
    dimensions_frame = pd.DataFrame(dimension_rows)
    coefficients_frame = pd.DataFrame(coefficient_rows)
    if local_frame is None:
        raise RuntimeError("Combined local explanations were not generated")

    atomic_csv(tuning, results_dir / "tuning.csv")
    atomic_csv(thresholds_frame, results_dir / "thresholds.csv")
    atomic_csv(metrics_frame, results_dir / "metrics.csv")
    atomic_csv(main_results_frame, results_dir / "main_results_tF1.csv")
    atomic_csv(selected_frame, results_dir / "selected_models.csv")
    atomic_csv(dimensions_frame, results_dir / "feature_dimensions.csv")
    atomic_csv(coefficients_frame, results_dir / "global_coefficients.csv")
    atomic_csv(local_frame, results_dir / "local_explanations.csv")
    plot_tuning(tuning, results_dir / "tuning_macro_f1.png")
    plot_local_contributions(local_frame, results_dir / "local_contribution.png")

    summary = {
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "experiment_revision": experiment_revision,
        "config_sha256": config_sha,
        "data_manifest_sha256": data_manifest_sha,
        "data_integrity_status": integrity["status"],
        "selection_rule": config["selection"],
        "primary_threshold_policy": "tF1",
        "selected_models": selected_rows,
        "main_results_tF1": main_results,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "notes": [
            "All vocabularies and IDF values were fitted on cleaned D1 train only.",
            "C was selected on D1 validation Macro F1 at threshold 0.5; ties prefer smaller C.",
            "tF1 and t5 thresholds were selected only from D1 validation scores; ties prefer higher thresholds.",
            "Class-1 scores are uncalibrated Logistic Regression predict_proba outputs.",
            "D2 and D3 were used only after model and threshold selection.",
        ],
    }
    atomic_json(summary, results_dir / "run_summary.json")

    print("E1 complete", flush=True)
    print(selected_frame.to_string(index=False), flush=True)
    print("Primary tF1 results", flush=True)
    print(
        pd.DataFrame(main_results)[
            ["pipeline", "dataset", "macro_f1", "attack_recall", "fpr", "auroc"]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
