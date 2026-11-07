import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def main():
    os.makedirs("results/e2/figures", exist_ok=True)

    # 1. Generate e2_transfer_bar.png
    results = {
        "D1 In-Domain\n(S-Labs Test)": 0.9619,
        "D2 Out-of-Domain\n(deepset Test)": 0.8448,
        "D3 Agentic-5K\n(Complex Scenarios)": 0.7151,
    }
    plt.figure(figsize=(6.5, 4.5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    bars = plt.bar(
        list(results.keys()),
        list(results.values()),
        color=colors,
        width=0.55,
        edgecolor="black",
        alpha=0.85,
    )
    plt.ylabel("Macro-F1 Score", fontsize=11)
    plt.title("E2 BiLSTM: Cross-Dataset Generalization Drop", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_fig = "results/e2/figures/e2_transfer_bar.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[Done] Saved transfer bar chart to {out_fig}")

    # 2. Generate selected_models.csv
    with open("results/e2/handoff/selected_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    row = {
        "pipeline": "bilstm",
        "model_id": f"e2_bilstm_{cfg['selected_condition']}",
        "selected_condition": cfg["selected_condition"],
        "dev_macro_f1_at_0.5": cfg["validation_macro_f1"],
        "vocab_size": cfg["vocab_size"],
        "model_path": cfg["checkpoint"],
        "model_size_mb": cfg["model_size_mb"],
        "batch1_latency_ms_median": cfg["batch1_latency_median_ms"],
        "batch1_latency_ms_p95": cfg["batch1_latency_p95_ms"],
    }
    pd.DataFrame([row]).to_csv("results/e2/selected_models.csv", index=False)
    print("[Done] Saved results/e2/selected_models.csv")

    # 3. Align transform_id in handoff parquet to 'clean'
    pq_path = "results/e2/handoff/bilstm_scores.parquet"
    if os.path.exists(pq_path):
        df_pq = pd.read_parquet(pq_path)
        df_pq["transform_id"] = "clean"
        df_pq.to_parquet(pq_path, index=False)
        print("[Done] Aligned transform_id to 'clean' in handoff parquet")


if __name__ == "__main__":
    main()
