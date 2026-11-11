"""Figures for E5. Every figure carries a footer with the threat model and both checkpoint hashes.

Palette: the validated default categorical palette (light mode). In figures 1-4 blue = clean model /
clean text and orange = augmented model / perturbed text; grey = control conditions. Figure 5 is a
different encoding and uses neutral ink for the coefficient shifts and aqua / violet for the two
feature branches, each with its own legend. Thin marks, hairline grid, legend whenever two series
are shown, selective direct labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_CLEAN = "#2a78d6"
COLOR_AUG = "#eb6834"
COLOR_CONTROL = "#898781"
COLOR_WORD_BRANCH = "#1baf7a"  # slot 3 aqua
COLOR_CHAR_BRANCH = "#6250d6"  # slot 7 violet
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    axis.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    axis.set_axisbelow(True)


def finish(figure: plt.Figure, path: Path, footer: str, rect: tuple[float, float, float, float] = (0, 0.03, 1, 1)) -> None:
    figure.text(0.01, 0.005, footer, fontsize=7, color=INK_MUTED, ha="left", va="bottom")
    figure.tight_layout(rect=rect)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def plot_selection(tuning: pd.DataFrame, selected: str, guard_value: float, path: Path, footer: str) -> None:
    frame = tuning.reset_index(drop=True)
    y = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(9.0, 0.6 * len(frame) + 2.2), facecolor=SURFACE)
    style_axis(axis, grid_axis="x")
    from matplotlib.lines import Line2D

    axis.hlines(y, frame["pert_val_macro_f1_mean"], frame["clean_val_macro_f1_eligible"], color=GRID, linewidth=2, zorder=1)
    axis.axvline(guard_value, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
    controls = frame["is_control"].astype(bool).to_numpy()
    axis.scatter(frame["clean_val_macro_f1"], y, s=70, facecolors="none", edgecolors=np.where(controls, COLOR_CONTROL, COLOR_CLEAN), linewidths=1.6, zorder=3)
    axis.scatter(frame["clean_val_macro_f1_eligible"], y, s=64, color=np.where(controls, COLOR_CONTROL, COLOR_CLEAN), zorder=3)
    axis.errorbar(frame["pert_val_macro_f1_mean"], y, xerr=frame["pert_val_macro_f1_sd"], fmt="o", markersize=8, color=COLOR_AUG, ecolor=COLOR_AUG, elinewidth=1.2, capsize=3, zorder=3)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COLOR_CLEAN, markeredgewidth=1.6, markersize=8, label="clean text, full D1 validation (1,985 rows; the guard applies here)"),
        Line2D([], [], marker="o", linestyle="none", color=COLOR_CLEAN, markersize=8, label="clean text, eligible rows only (same rows as the typo copies)"),
        Line2D([], [], marker="o", linestyle="-", color=COLOR_AUG, markersize=8, label="typo-perturbed eligible rows, mean ± SD over 5 draws"),
        Line2D([], [], marker="o", linestyle="none", color=COLOR_CONTROL, markersize=8, label="controls (grey markers): never eligible as primary"),
        Line2D([], [], linestyle="--", color=INK_MUTED, label="guard: clean − δ on the full split"),
    ]
    labels = []
    for _, row in frame.iterrows():
        label = row["condition"]
        if row["condition"] == selected:
            label += "  ✔ selected"
        elif bool(row["is_control"]):
            label += "  (control)"
        labels.append(label)
    axis.set_yticks(y, labels=labels, color=INK)
    axis.invert_yaxis()
    axis.set_xlabel("Macro F1 at threshold 0.5 (axis zoomed to the observed range)", color=INK_SECONDARY)
    axis.set_title("E5 selection stage: clean versus typo-perturbed D1 validation per condition", color=INK, loc="left")
    axis.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8)
    finish(figure, path, footer)


def plot_robustness(paired: pd.DataFrame, model_ids: dict[str, str], path: Path, footer: str) -> None:
    datasets = [("d1", "D1 test (S-Labs)"), ("d2", "D2 test (deepset)"), ("d3", "D3 test (agentic)")]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True, facecolor=SURFACE)
    width = 0.36
    for axis, (dataset, title) in zip(axes, datasets):
        style_axis(axis)
        values = {}
        for key, model_id in model_ids.items():
            rows = paired[(paired["model_id"] == model_id) & (paired["dataset"] == dataset) & (paired["split"] == "test") & (paired["threshold_policy"] == "default_0.5") & (paired["subset"] == "all_eligible")]
            random_row = rows[rows["policy"] == "random1"].iloc[0]
            targeted_row = rows[rows["policy"] == "targeted_lexical"].iloc[0]
            values[key] = [float(random_row["orig_attack_recall"]), float(random_row["pert_attack_recall"]), float(targeted_row["pert_attack_recall"])]
        x = np.arange(3)
        bars_clean = axis.bar(x - width / 2, values["clean"], width - 0.02, color=COLOR_CLEAN, label="clean model")
        bars_aug = axis.bar(x + width / 2, values["augmented"], width - 0.02, color=COLOR_AUG, label="augmented model")
        for bars in (bars_clean, bars_aug):
            for bar in bars:
                axis.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK_SECONDARY)
        axis.set_xticks(x, labels=["clean\ntext", "one random\ntypo", "lexicon-targeted\ntypo"], color=INK_SECONDARY)
        axis.set_ylim(0, 1.08)
        axis.set_title(title, color=INK, loc="left", fontsize=11)
    axes[0].set_ylabel("Attack recall at threshold 0.5 (eligible pairs)", color=INK_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.99, 0.99))
    figure.suptitle("E5: attack recall on clean and typo-perturbed test text (canonical evaluation draw)", color=INK, x=0.01, ha="left")
    finish(figure, path, footer, rect=(0, 0.03, 1, 0.93))


def plot_bootstrap(bootstrap: pd.DataFrame, path: Path, footer: str) -> None:
    frame = bootstrap[(bootstrap["threshold_policy"] == "default_0.5") & (bootstrap["comparison"] == "augmented_minus_clean")].copy().reset_index(drop=True)
    frame["label"] = frame.apply(lambda row: f"{row['dataset'].upper()} {row['text_label']} · Δ {row['metric'].replace('_', ' ')}" + ("  ◀ primary" if row["role"] == "primary" else ""), axis=1)
    y = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(8.8, 0.3 * len(frame) + 1.8), facecolor=SURFACE)
    style_axis(axis, grid_axis="x")
    axis.axvline(0, color=AXIS, linewidth=0.8)
    primary = frame["role"].eq("primary").to_numpy()
    axis.hlines(y, frame["ci_low"], frame["ci_high"], color=COLOR_AUG, linewidth=np.where(primary, 3.2, 1.8))
    axis.scatter(frame["point_estimate"], y, s=np.where(primary, 70, 34), color=COLOR_AUG, zorder=3)
    axis.set_yticks(y, labels=frame["label"], fontsize=8, color=INK)
    for tick, is_primary in zip(axis.get_yticklabels(), primary):
        if is_primary:
            tick.set_fontweight("bold")
    axis.invert_yaxis()
    axis.set_xlabel("augmented minus clean at threshold 0.5, 95% paired bootstrap interval (augmented minus duplicate control: see bootstrap.csv)", color=INK_SECONDARY)
    axis.set_title(f"E5: paired class-stratified bootstrap, {int(frame['n_boot'].iloc[0]):,} resamples", color=INK, loc="left")
    finish(figure, path, footer)


def plot_d4(false_alarms: pd.DataFrame, path: Path, footer: str) -> None:
    frame = false_alarms[false_alarms["threshold_policy"] == "default_0.5"].copy()
    groups = [g for g in frame["slice"].unique() if g == "overall" or g.startswith("trigger_count=")]
    figure, axis = plt.subplots(figsize=(8.0, 4.2), facecolor=SURFACE)
    style_axis(axis)
    x = np.arange(len(groups))
    width = 0.36
    for offset, (label, color, key) in enumerate([("clean model", COLOR_CLEAN, "clean"), ("augmented model", COLOR_AUG, "augmented")]):
        rows = [frame[(frame["slice"] == g) & (frame["model_key"] == key)].iloc[0] for g in groups]
        centers = x + (offset - 0.5) * width
        heights = [float(r["fpr"]) for r in rows]
        errors = np.array([[float(r["fpr"]) - float(r["ci_low"]) for r in rows], [float(r["ci_high"]) - float(r["fpr"]) for r in rows]])
        axis.bar(centers, heights, width - 0.02, color=color, label=label)
        axis.errorbar(centers, heights, yerr=errors, fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=3)
        for center, row in zip(centers, rows):
            axis.annotate(f"{int(row['fp'])}/{int(row['n'])}", (center, float(row["ci_high"])), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK_SECONDARY)
    axis.set_xticks(x, labels=[g.replace("trigger_count=", "trigger words = ") for g in groups], color=INK_SECONDARY)
    axis.set_ylim(0, min(1.0, float(frame["ci_high"].max()) * 1.2))
    axis.set_ylabel("False-positive rate (benign-only D4), Wilson 95% CI", color=INK_SECONDARY)
    axis.set_title("E5: NotInject false alarms at threshold 0.5", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9)
    finish(figure, path, footer)


def plot_mechanism(coefficient_shift: pd.DataFrame, branch_share: pd.DataFrame, path: Path, footer: str, top_n: int = 12) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    for axis, branch in zip(axes[:2], ("word", "char")):
        rows = coefficient_shift[(coefficient_shift["branch"] == branch) & (coefficient_shift["selection"] == "largest_shift")].head(top_n).iloc[::-1]
        style_axis(axis, grid_axis="x")
        axis.axvline(0, color=AXIS, linewidth=0.8)
        axis.barh(rows["feature"].str.slice(0, 28), rows["delta"], color=INK_SECONDARY, height=0.6)
        axis.set_title(f"{branch} branch: largest coefficient shifts (augmented − clean; right = more attack-like)", color=INK, loc="left", fontsize=10)
        axis.set_xlabel("Δ logistic coefficient", color=INK_SECONDARY)
        axis.tick_params(axis="y", labelsize=8)
    axis = axes[2]
    style_axis(axis)
    models = list(branch_share["model_key"].unique())
    x = np.arange(len(models))
    word = [float(branch_share[(branch_share["model_key"] == m) & (branch_share["branch"] == "word")]["share"].iloc[0]) for m in models]
    char = [float(branch_share[(branch_share["model_key"] == m) & (branch_share["branch"] == "char")]["share"].iloc[0]) for m in models]
    axis.bar(x, word, 0.55, color=COLOR_WORD_BRANCH, label="word branch")
    axis.bar(x, char, 0.55, bottom=word, color=COLOR_CHAR_BRANCH, label="char branch", edgecolor=SURFACE, linewidth=2)
    for position, (w, c) in enumerate(zip(word, char)):
        axis.annotate(f"{100 * w:.0f}%", (position, w / 2), ha="center", va="center", fontsize=9, color="#ffffff")
        axis.annotate(f"{100 * c:.0f}%", (position, w + c / 2), ha="center", va="center", fontsize=9, color="#ffffff")
    axis.set_xticks(x, labels=[m.replace("_", " ") for m in models], color=INK_SECONDARY)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("share of Σ|x·w| on D1 validation", color=INK_SECONDARY)
    axis.set_title("branch contribution share", color=INK, loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    figure.suptitle("E5 mechanism: what the augmented Logistic Regression re-weighted on the frozen features", color=INK, x=0.01, ha="left")
    finish(figure, path, footer, rect=(0, 0.03, 1, 0.93))
