"""E6 figures: validation trade-off, cross-source operating points, NotInject false alarms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
# Categorical slots in fixed order; colour follows the model, never its rank.
MODEL_COLORS = {
    "e1_word_tfidf_lr": "#2a78d6",
    "e1_char_tfidf_lr": "#eb6834",
    "e1_combined_tfidf_lr": "#1baf7a",
    "bilstm_random": "#eda100",
    "e2_bilstm_random": "#eda100",
    "e3_distilbert_base_uncased": "#e87ba4",
    "e5_clean_combined_tfidf_lr": "#6250d6",
    "e5_augmented_combined_tfidf_lr": "#e34948",
}
E4_COLOR = "#008300"
FALLBACK = "#52514e"
METRIC_COLORS = {"macro_f1": "#2a78d6", "attack_recall": "#eb6834", "fpr": "#1baf7a"}
METRIC_LABELS = {"macro_f1": "Macro F1", "attack_recall": "Attack recall", "fpr": "FPR"}
SEQ = ["#86b6ef", "#2a78d6", "#104281"]  # sequential blue steps for trigger_count 1/2/3


def color_of(model_id: str) -> str:
    if model_id in MODEL_COLORS:
        return MODEL_COLORS[model_id]
    return E4_COLOR if model_id.startswith("e4") or "roberta" in model_id else FALLBACK


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new_figure(rows: int, cols: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)
    return fig, axes


def footer(fig, text: str, y: float = 0.005) -> None:
    fig.text(0.01, y, text, fontsize=7, color=INK_2, ha="left", va="top" if y < 0 else "bottom")


def fig_tradeoff(out: Path, thresholds: pd.DataFrame, sweeps: pd.DataFrame, order: list[str], names) -> Path:
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = new_figure(rows, cols, (3.1 * cols, 2.6 * rows + 0.8))
    for ax, model_id in zip(axes.flat, order):
        table = sweeps[sweeps.model_id == model_id].sort_values("threshold")
        table = table[table.threshold <= 1.0]
        for metric, color in METRIC_COLORS.items():
            ax.plot(table.threshold, table[metric], color=color, linewidth=2, label=METRIC_LABELS[metric])
        sel = thresholds[thresholds.model_id == model_id].set_index("threshold_policy")["threshold"]
        ax.axvline(sel["tF1"], color=INK, linewidth=1.2, linestyle="-")
        ax.axvline(sel["t5"], color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
        ax.axvline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{names(model_id)}\ntF1 = {sel['tF1']:.3f}   t5 = {sel['t5']:.3f}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("threshold", fontsize=8, color=INK_2)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK, linewidth=1.2), plt.Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 2))),
                plt.Line2D([], [], color=INK_2, linewidth=0.8, linestyle=":")]
    labels += ["tF1 (max Macro F1)", "t5 (max recall, FPR ≤ 5%)", "default 0.5"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("D1 Validation threshold trade-off (selection data only)", fontsize=11, color=INK, x=0.01, ha="left", y=1.05)
    footer(fig, "Scores are uncalibrated. Thresholds are chosen on D1 Validation and frozen before any test set is scored.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = out / "figures/fig1_validation_tradeoff.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_cross_source(out: Path, metrics: pd.DataFrame, notinject: pd.DataFrame, order: list[str], names) -> Path:
    sources = [("d1", "D1 test\n(in-domain)"), ("d2", "D2 test"), ("d3", "D3 test\n(agentic)"), ("d4", "D4 NotInject\n(benign only)")]
    fig, axes = new_figure(2, 2, (11, 7))
    offsets = np.linspace(-0.25, 0.25, len(order)) if len(order) > 1 else [0.0]
    for col, policy in enumerate(("tF1", "t5")):
        for row, metric in enumerate(("fpr", "attack_recall")):
            ax = axes[row, col]
            for offset, model_id in zip(offsets, order):
                xs, ys = [], []
                for i, (dataset, _) in enumerate(sources):
                    if dataset == "d4":
                        if metric != "fpr":
                            continue
                        hit = notinject[(notinject.model_id == model_id) & (notinject.threshold_policy == policy)
                                        & (notinject.group_by == "overall")]
                        value = hit["fpr"].iloc[0] if len(hit) else None
                    else:
                        hit = metrics[(metrics.model_id == model_id) & (metrics.threshold_policy == policy)
                                      & (metrics.dataset == dataset) & (metrics.split == "test")]
                        value = hit[metric].iloc[0] if len(hit) else None
                    if value is not None and not pd.isna(value):
                        xs.append(i + offset)
                        ys.append(value)
                ax.plot(xs, ys, color=color_of(model_id), linewidth=1, alpha=0.5)
                ax.scatter(xs, ys, s=36, color=color_of(model_id), edgecolor=SURFACE, linewidth=1.5, zorder=3,
                           label=names(model_id))
            n_sources = 4 if metric == "fpr" else 3
            ax.set_xticks(range(n_sources), [s[1] for s in sources[:n_sources]], fontsize=8)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.02, 1.02)
            if metric == "fpr":
                ax.axhline(0.05, color=INK_2, linewidth=0.8, linestyle=":")
                ax.text(3.45, 0.06, "5% FPR budget (D1 val)", fontsize=7, color=INK_2, ha="right")
            ax.set_title(f"{METRIC_LABELS[metric]} at {policy}", fontsize=9, color=INK, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Frozen D1 thresholds applied to other sources", fontsize=11, color=INK, x=0.01, ha="left")
    footer(fig, "Each source reported separately; D4 has no attacks, so only FPR is shown. Thresholds were never re-tuned on D2/D3/D4.", y=-0.01)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    path = out / "figures/fig2_cross_source_operating_points.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_notinject(out: Path, notinject: pd.DataFrame, order: list[str], names) -> Path:
    with_d4 = [m for m in order if notinject[(notinject.model_id == m) & (notinject.n > 0)].shape[0]]
    fig, axes = new_figure(1, 3, (13, 0.55 * len(with_d4) + 1.8))
    for ax, policy in zip(axes.flat, ("default_0.5", "tF1", "t5")):
        for y, model_id in enumerate(reversed(with_d4)):
            rows = notinject[(notinject.model_id == model_id) & (notinject.threshold_policy == policy)]
            overall = rows[rows.group_by == "overall"].iloc[0]
            ax.plot([overall.ci_low, overall.ci_high], [y, y], color=color_of(model_id), linewidth=2, solid_capstyle="round")
            ax.scatter([overall.fpr], [y], s=60, color=color_of(model_id), edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(min(overall.ci_high + 0.02, 0.98), y, f"{int(overall.fp)}/{int(overall.n)}", fontsize=7,
                    color=INK_2, va="center")
            for k, (_, sub) in enumerate(rows[rows.group_by == "trigger_count"].sort_values("group").iterrows()):
                ax.scatter([sub.fpr], [y - 0.28], s=14, color=SEQ[k % 3], marker="|", linewidth=2, zorder=2)
        ax.set_yticks(range(len(with_d4)), [names(m) for m in reversed(with_d4)] if policy == "default_0.5" else [""] * len(with_d4),
                      fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, len(with_d4) - 0.4)
        ax.set_title(f"threshold = {policy}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("false-alarm rate on 339 benign prompts", fontsize=8, color=INK_2)
    handles = [plt.Line2D([], [], color=INK_2, marker="o", linewidth=2, markersize=6, label="overall FPR, Wilson 95% CI")]
    handles += [plt.Line2D([], [], color=SEQ[k], marker="|", linestyle="", markersize=8, markeredgewidth=2,
                           label=f"{k + 1} trigger word{'s' if k else ''}") for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("NotInject (D4) false alarms by threshold policy", fontsize=11, color=INK, x=0.01, ha="left")
    missing = [m for m in order if m not in with_d4]
    note = "Benign-only stress test: no F1 or AUROC. "
    if missing:
        note += "No D4 scores delivered for: " + ", ".join(names(m) for m in missing) + "."
    footer(fig, note, y=-0.01)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path = out / "figures/fig3_notinject_false_alarms.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def make_figures(out: Path, thresholds, metrics, notinject, sweeps, order) -> list[Path]:
    (out / "figures").mkdir(parents=True, exist_ok=True)
    names = dict(zip(thresholds.model_id, thresholds.display_name)).get
    return [fig_tradeoff(out, thresholds, sweeps, order, names),
            fig_cross_source(out, metrics, notinject, order, names),
            fig_notinject(out, notinject, order, names)]
