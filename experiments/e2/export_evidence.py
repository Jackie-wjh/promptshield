import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def main():
    os.makedirs("results/e2/figures", exist_ok=True)

    # 1. Generate e2_transfer_bar.png with clean sample size annotations
    results = {
        "D1 In-Domain\n(S-Labs, N=2,101)": 0.9619,
        "D2 Out-of-Domain\n(deepset, N=116)": 0.8448,
        "D3 Agentic\n(Deniz, N=545)": 0.7151,
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

    # 2. Generate selected_models.csv with forward slashes and exact SHA-256
    with open("results/e2/handoff/selected_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    row = {
        "pipeline": "bilstm",
        "model_id": f"e2_bilstm_{cfg['selected_condition']}",
        "selected_condition": cfg["selected_condition"],
        "dev_macro_f1_at_0.5": cfg["validation_macro_f1"],
        "vocab_size": cfg["vocab_size"],
        "model_path": cfg["checkpoint"].replace("\\", "/"),
        "checkpoint_rev": cfg.get("checkpoint_rev", "6ec73987d1c64bedbd75cc3dfb5a734751f11900eb5818b6851cdb5e943f9cdd"),
        "model_size_mb": cfg["model_size_mb"],
        "batch1_latency_ms_median": cfg["batch1_latency_median_ms"],
        "batch1_latency_ms_p95": cfg["batch1_latency_p95_ms"],
    }
    pd.DataFrame([row]).to_csv("results/e2/selected_models.csv", index=False)
    print("[Done] Saved results/e2/selected_models.csv")


if __name__ == "__main__":
    main()
