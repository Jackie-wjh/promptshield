#!/usr/bin/env python3
"""Verify saved E1 models, predictions, thresholds, metrics, and explanations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e1.run_e1 import (
    atomic_json,
    attack_scores,
    metric_record,
    select_thresholds,
    sha256_file,
)
from scripts.check_data_integrity import run_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / "configs/e1.json").read_text(encoding="utf-8"))
    results = root / "results/e1"
    processed = root / "data/processed"
    tuning = pd.read_csv(results / "tuning.csv")
    selected = pd.read_csv(results / "selected_models.csv")
    thresholds = pd.read_csv(results / "thresholds.csv")
    saved_metrics = pd.read_csv(results / "metrics.csv")
    local = pd.read_csv(results / "local_explanations.csv")
    run_summary = json.loads((results / "run_summary.json").read_text(encoding="utf-8"))

    integrity = run_checks(root)
    require(integrity["status"] == "passed", "Data integrity gate failed")
    require(len(tuning) == 9, "E1 must contain exactly nine tuning fits")
    require(
        set(tuning["pipeline"]) == {"word", "char", "combined"},
        "Unexpected E1 pipelines",
    )

    validation = pd.read_parquet(processed / "d1_validation.parquet")
    evaluation = {
        "d1": pd.read_parquet(processed / "d1_test.parquet"),
        "d2": pd.read_parquet(processed / "d2_test.parquet"),
        "d3": pd.read_parquet(processed / "d3_test.parquet"),
    }
    checks: dict[str, str] = {"data_integrity": "passed"}

    for pipeline in ("word", "char", "combined"):
        pipeline_tuning = tuning[tuning["pipeline"].eq(pipeline)].copy()
        require(
            pipeline_tuning["C"].tolist() == [0.1, 1.0, 10.0],
            f"{pipeline}: C grid mismatch",
        )
        best = pipeline_tuning.sort_values(
            ["dev_macro_f1", "C"], ascending=[False, True]
        ).iloc[0]
        selected_row = selected[selected["pipeline"].eq(pipeline)].iloc[0]
        require(
            close(float(best["C"]), float(selected_row["selected_C"])),
            f"{pipeline}: saved model does not match selection rule",
        )

        model_path = root / selected_row["model_path"]
        require(model_path.is_file(), f"{pipeline}: model file is missing")
        require(
            sha256_file(model_path) == selected_row["checkpoint_rev"],
            f"{pipeline}: checkpoint hash mismatch",
        )
        model = joblib.load(model_path)
        require(
            close(float(model.named_steps["classifier"].C), float(selected_row["selected_C"])),
            f"{pipeline}: classifier C mismatch",
        )
        require(
            int(model.named_steps["classifier"].coef_.shape[1]) == int(selected_row["features"]),
            f"{pipeline}: feature dimension mismatch",
        )

        validation_scores = attack_scores(model, validation["text"])
        independently_selected = select_thresholds(
            validation["label"].to_numpy(dtype=int), validation_scores
        )
        pipeline_thresholds = thresholds[thresholds["pipeline"].eq(pipeline)]
        for policy, values in independently_selected.items():
            saved = pipeline_thresholds[pipeline_thresholds["policy"].eq(policy)].iloc[0]
            require(
                close(float(saved["threshold"]), float(values["threshold"])),
                f"{pipeline}/{policy}: threshold mismatch",
            )
            require(
                close(float(saved["macro_f1"]), float(values["macro_f1"])),
                f"{pipeline}/{policy}: validation metric mismatch",
            )
        require(
            independently_selected["t5"]["fpr"] <= 0.05,
            f"{pipeline}: t5 violates validation FPR constraint",
        )

        prediction_path = results / "predictions" / f"{pipeline}_predictions.parquet"
        predictions = pd.read_parquet(prediction_path)
        require(len(predictions) == (2101 + 116 + 545) * 3, f"{pipeline}: prediction row count")
        require(
            predictions["checkpoint_rev"].nunique() == 1
            and predictions["checkpoint_rev"].iloc[0] == selected_row["checkpoint_rev"],
            f"{pipeline}: prediction checkpoint mismatch",
        )

        for dataset, frame in evaluation.items():
            recomputed_scores = attack_scores(model, frame["text"])
            for policy, threshold_values in independently_selected.items():
                group = predictions[
                    predictions["dataset"].eq(dataset)
                    & predictions["threshold_policy"].eq(policy)
                ].reset_index(drop=True)
                require(
                    group["sample_id"].tolist() == frame["sample_id"].tolist(),
                    f"{pipeline}/{dataset}/{policy}: sample IDs differ",
                )
                require(
                    np.allclose(group["score_attack"], recomputed_scores, rtol=0.0, atol=1e-12),
                    f"{pipeline}/{dataset}/{policy}: saved scores differ from model",
                )
                threshold = float(threshold_values["threshold"])
                expected_prediction = (recomputed_scores >= threshold).astype(int)
                require(
                    np.array_equal(group["pred"].to_numpy(dtype=int), expected_prediction),
                    f"{pipeline}/{dataset}/{policy}: predictions do not follow score >= threshold",
                )
                recomputed_metrics = metric_record(
                    frame["label"].to_numpy(dtype=int), recomputed_scores, threshold
                )
                saved_row = saved_metrics[
                    saved_metrics["pipeline"].eq(pipeline)
                    & saved_metrics["dataset"].eq(dataset)
                    & saved_metrics["threshold_policy"].eq(policy)
                ].iloc[0]
                for metric in ("macro_f1", "attack_recall", "fpr", "auroc"):
                    require(
                        close(float(saved_row[metric]), float(recomputed_metrics[metric])),
                        f"{pipeline}/{dataset}/{policy}: {metric} mismatch",
                    )
                for count in ("tn", "fp", "fn", "tp", "n"):
                    require(
                        int(saved_row[count]) == int(recomputed_metrics[count]),
                        f"{pipeline}/{dataset}/{policy}: {count} mismatch",
                    )

        checks[f"{pipeline}_model_predictions_metrics"] = "passed"

    require(
        np.allclose(
            local["contribution"],
            local["feature_value"] * local["coefficient"],
            rtol=0.0,
            atol=1e-12,
        ),
        "Local contribution values are not TF-IDF value times coefficient",
    )
    require(local["sample_id"].nunique() == 6, "Expected six fixed local explanation samples")
    checks["local_explanations"] = "passed"
    require(run_summary["status"] == "complete", "Run summary is not complete")
    require(run_summary["data_integrity_status"] == "passed", "Run summary data gate failed")
    checks["run_summary"] = "passed"

    report = {
        "status": "passed",
        "checks": checks,
        "models_verified": 3,
        "tuning_fits_verified": 9,
        "prediction_rows_verified": 3 * (2101 + 116 + 545) * 3,
        "metric_rows_verified": 3 * 3 * 3,
    }
    atomic_json(report, results / "verification.json")
    print("E1 verification passed")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
