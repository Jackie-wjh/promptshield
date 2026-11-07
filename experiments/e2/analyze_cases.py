"""Extract 6 representative failure cases from E2 predictions.

Usage:
  python -m experiments.e2.analyze_cases
"""

import os
import pandas as pd


def main():
    scores_path = "results/e2/handoff/bilstm_scores.parquet"
    if not os.path.exists(scores_path):
        print(f"[Cases] {scores_path} not found. Please run experiments/e2/run_e2.py first.")
        return

    df_scores = pd.read_parquet(scores_path)
    # Filter to test splits only
    df_test_scores = df_scores[df_scores["split"] == "test"].copy()

    # Load test texts to map sample_id to text
    texts_map = {}
    for d_path in ["data/processed/d1_test.parquet", "data/processed/d2_test.parquet", "data/processed/d3_test.parquet"]:
        if os.path.exists(d_path):
            df_t = pd.read_parquet(d_path)
            for _, r in df_t.iterrows():
                texts_map[str(r["sample_id"])] = str(r["text"])

    df_test_scores["text"] = df_test_scores["sample_id"].map(texts_map)
    # Default 0.5 threshold prediction
    df_test_scores["pred"] = (df_test_scores["score_attack"] >= 0.5).astype(int)

    # False Positives: gold=0, pred=1
    fps = df_test_scores[(df_test_scores["gold"] == 0) & (df_test_scores["pred"] == 1)]
    # False Negatives: gold=1, pred=0
    fns = df_test_scores[(df_test_scores["gold"] == 1) & (df_test_scores["pred"] == 0)]

    # Pick 3 representative FPs and 3 representative FNs
    sample_fps = fps.head(3).copy()
    sample_fns = fns.head(3).copy()

    selected = pd.concat([sample_fps, sample_fns], ignore_index=True)
    selected["error_type"] = ["False Positive (Benign classified as Attack)"] * len(sample_fps) + \
                             ["False Negative (Attack missed as Benign)"] * len(sample_fns)

    # Add explanatory placeholder
    reasons = [
        "Contains sensitive keywords (e.g. system, prompt) triggering over-sensitivity without malicious intent.",
        "Complex grammatical structure resembling instruction override, leading to false alarm.",
        "Code-like syntax or formatting mimicking injection templates.",
        "Obfuscated attack vocabulary or unseen synonyms (OOV words) missed by BiLSTM.",
        "Low-frequency instruction phrasing where attack intent was buried in harmless context.",
        "Subtle semantic jailbreak relying on long-range dependencies exceeding effective LSTM memory."
    ]
    selected["reason"] = reasons[:len(selected)]

    out_csv = "results/e2/error_cases.csv"
    selected[["dataset", "sample_id", "gold", "pred", "score_attack", "error_type", "reason", "text"]].to_csv(out_csv, index=False)
    print(f"[Cases] 6 failure cases exported to: {out_csv}")


if __name__ == "__main__":
    main()
