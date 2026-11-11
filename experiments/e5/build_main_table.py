#!/usr/bin/env python3
"""Cross-source main table (M5 responsibility): E1-E4 main pipelines x D1/D2/D3, plus E5, D4 and cost tables.

The script discovers every raw-score handoff file under results/e*/handoff/ (plus any
``--source name=path``), validates each one against the README contract (section 8 schema,
section 13 row counts, unique key, finite scores, sample_id and gold identical to data/processed),
and reports Macro F1, attack recall and FPR per source at the default 0.5 threshold and at
tF1 / t5. Sources are never pooled.

Thresholds come from results/e6/thresholds.csv when E6 has delivered it (long layout model_id /
policy / threshold, or wide layout model_id / tF1 / t5). Any model missing from that file falls
back to a PROVISIONAL threshold (tF1 / t5 re-derived from the model's own D1 validation rows with
E1's rule); every row carries its own ``threshold_source`` and every table says when sources are
mixed. Files that violate the contract are listed and excluded, never silently patched.

Usage:
    python experiments/e5/build_main_table.py
    python experiments/e5/build_main_table.py --source e4=D:/from_teammate/roberta_scores.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import numpy as np
import pandas as pd

from experiments.e5 import common
from experiments.e5.common import EVAL_SPLITS, HANDOFF_COLUMNS, atomic_csv, atomic_json, metric_record, select_thresholds, wilson_interval, write_text

MAIN_PIPELINE_EXPERIMENTS = ("e1", "e2", "e3", "e4")  # the six main pipelines; E5 rows are supplementary
EXPERIMENT_ORDER = {name: index for index, name in enumerate(MAIN_PIPELINE_EXPERIMENTS + ("e5", "e6"))}
DISPLAY_ALIASES = {"bilstm_random": "e2_bilstm_random", "bilstm_glove": "e2_bilstm_glove"}  # E2's file does not follow the e<N>_ convention
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BYTES_PER_MIB = 1024 ** 2
PIVOT_DATASETS = ("d1", "d2", "d3")


def parse_source(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--source expects NAME=PATH, e.g. e4=results/e4/handoff/roberta_scores.parquet")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source", action="append", default=[], type=parse_source, metavar="NAME=PATH", help="extra raw-score parquet file(s) from a teammate")
    parser.add_argument("--e6-thresholds", type=Path, default=None, help="override the location of E6's thresholds.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/e5/main_table")
    return parser.parse_args()


def discover_sources(root: Path, extra: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for path in sorted((root / "results").glob("e*/handoff/*_scores.parquet")):
        sources.append((path.parent.parent.name, path))
    for name, path in extra:
        sources.append((name, path if path.is_absolute() else root / path))
    return sources


def describe(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def validate_file(frame: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one raw-score file against the README contract."""

    errors: list[str] = []
    warnings: list[str] = []
    missing = [c for c in HANDOFF_COLUMNS if c not in frame.columns]
    if missing:
        return [f"missing columns {missing}"], warnings
    extra = [c for c in frame.columns if c not in HANDOFF_COLUMNS]
    if extra:
        warnings.append(f"extra columns {extra} (ignored)")
    if frame["model_id"].nunique() != 1:
        return [f"expected one model_id per file, found {sorted(frame['model_id'].astype(str).unique())}"], warnings
    model_id = str(frame["model_id"].iloc[0])
    if not re.match(r"^e[1-6]_", model_id):
        warnings.append(f"model_id '{model_id}' does not follow the e<N>_ naming convention" + (f"; displayed as '{DISPLAY_ALIASES[model_id]}'" if model_id in DISPLAY_ALIASES else ""))
    if frame["checkpoint_rev"].nunique() != 1:
        errors.append("checkpoint_rev is not constant within the file")
    elif not HEX64.match(str(frame["checkpoint_rev"].iloc[0])):
        warnings.append(f"checkpoint_rev '{frame['checkpoint_rev'].iloc[0]}' is not a SHA-256 hex digest")
    if frame.duplicated(["model_id", "dataset", "split", "sample_id", "transform_id"]).any():
        errors.append("duplicate (model_id, dataset, split, sample_id, transform_id) keys")
    scores = pd.to_numeric(frame["score_attack"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        errors.append("non-finite or non-numeric score_attack values")
    gold = pd.to_numeric(frame["gold"], errors="coerce")
    if gold.isna().any() or not gold.isin([0, 1]).all():
        errors.append("gold must be 0/1")
    if not (frame["transform_id"].astype(str) == "clean").all():
        warnings.append(f"transform_id values other than 'clean': {sorted(set(frame['transform_id'].astype(str)))}")
    if frame["parent_id"].notna().any():
        warnings.append("parent_id is non-null on clean rows (contract: null unless the row is a transformed copy)")
    counts = frame.groupby(["dataset", "split"]).size().to_dict()
    unexpected = sorted(str(k) for k in counts if k not in set(EVAL_SPLITS.values()))
    if unexpected:
        errors.append(f"unexpected (dataset, split) groups {unexpected}: a handoff file holds only D1 validation, D1/D2/D3 test and optionally D4 test (README section 8)")
    for stem, (dataset, split) in EVAL_SPLITS.items():
        actual = int(counts.get((dataset, split), 0))
        expected = len(frames[stem])
        if dataset == "d4" and actual == 0:
            warnings.append("d4/test: no rows (D4 is optional for E2-E4, required for E5/E6 analyses)")
        elif actual != expected:
            errors.append(f"{dataset}/{split}: {actual} rows, expected {expected}")
    if errors:
        return errors, warnings
    for stem, (dataset, split) in EVAL_SPLITS.items():
        rows = frame[(frame["dataset"] == dataset) & (frame["split"] == split)]
        if rows.empty:
            continue
        processed = frames[stem].set_index("sample_id")["label"]
        ids = rows["sample_id"].astype(str)
        if set(ids) != set(processed.index.astype(str)):
            errors.append(f"{dataset}/{split}: sample_id set differs from processed data")
            continue
        if not np.array_equal(pd.to_numeric(rows["gold"]).to_numpy(dtype=int), processed.loc[ids].to_numpy(dtype=int)):
            errors.append(f"{dataset}/{split}: gold labels differ from processed data")
    return errors, warnings


def load_e6_thresholds(path: Path | None) -> pd.DataFrame | None:
    """Accept E6's thresholds.csv in long (model_id, policy|threshold_policy, threshold) or wide (model_id, tF1, t5) layout."""

    if path is None or not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "model_id" not in frame.columns:
        print(f"[main table] {path} has no model_id column; ignoring it", flush=True)
        return None
    policy_column = next((c for c in ("policy", "threshold_policy") if c in frame.columns), None)
    if policy_column and "threshold" in frame.columns:
        long = frame.rename(columns={policy_column: "policy"})[["model_id", "policy", "threshold"]].copy()
    else:
        wide = [c for c in frame.columns if str(c).strip().casefold() in {"tf1", "t_f1", "t5", "t_5"}]
        if not wide:
            print(f"[main table] {path} has neither a long nor a wide threshold layout; ignoring it", flush=True)
            return None
        long = frame.melt(id_vars=["model_id"], value_vars=wide, var_name="policy", value_name="threshold")
    aliases = {"tf1": "tF1", "t_f1": "tF1", "f1": "tF1", "t5": "t5", "t_5": "t5", "fpr5": "t5", "default_0.5": "default_0.5"}
    long["policy"] = long["policy"].astype(str).str.strip().map(lambda p: aliases.get(p.casefold(), p))
    long["threshold"] = pd.to_numeric(long["threshold"], errors="coerce")
    bad = long[~long["threshold"].between(0.0, 1.0)]
    if len(bad):
        print(f"[main table] warning: {len(bad)} E6 threshold rows are non-numeric or outside [0, 1] and are ignored: {bad.to_dict(orient='records')[:5]}", flush=True)
        long = long.drop(bad.index)
    duplicates = long[long.duplicated(["model_id", "policy"], keep=False)]
    if len(duplicates):
        print(f"[main table] warning: duplicate (model_id, policy) rows in E6 thresholds; the first is used: {sorted(set(zip(duplicates['model_id'], duplicates['policy'])))}", flush=True)
        long = long.drop_duplicates(["model_id", "policy"], keep="first")
    unknown = sorted(set(long["policy"]) - {"tF1", "t5", "default_0.5"})
    if unknown:
        print(f"[main table] warning: unrecognised policy names in E6 thresholds (ignored): {unknown}", flush=True)
    return long


def pct(value: Any, digits: int = 1) -> str:
    return "" if value is None or pd.isna(value) else f"{100 * float(value):.{digits}f}%"


def f4(value: Any) -> str:
    return "" if value is None or pd.isna(value) else f"{float(value):.4f}"


def pivot_markdown(long: pd.DataFrame, policy: str, title: str) -> str:
    rows = long[(long["threshold_policy"] == policy) & (long["split"] == "test") & (long["dataset"].isin(PIVOT_DATASETS))]
    sources = sorted(set(rows["threshold_source"]))
    label = {"default": "shared default threshold 0.5", "e6": "authoritative E6 thresholds", "provisional": "PROVISIONAL thresholds (each model's own D1 validation, E1 rule)"}
    heading = " / ".join(label.get(s, s) for s in sources) + ("  — MIXED SOURCES, see the source column" if len(sources) > 1 else "")
    lines = [f"### {title}: `{policy}` ({heading})", "", "| Experiment | model_id | threshold source | D1 Macro F1 | D1 Recall | D1 FPR | D2 Macro F1 | D2 Recall | D2 FPR | D3 Macro F1 | D3 Recall | D3 FPR |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (experiment, display), group in rows.groupby(["experiment", "display_model_id"], sort=False):
        cells = [experiment, f"`{display}`", str(group["threshold_source"].iloc[0])]
        for dataset in PIVOT_DATASETS:
            row = group[group["dataset"] == dataset]
            cells += ["", "", ""] if row.empty else [f4(row["macro_f1"].iloc[0]), pct(row["attack_recall"].iloc[0]), pct(row["fpr"].iloc[0])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def collect_cost_rows(root: Path) -> pd.DataFrame:
    """Best-effort cost table from each member's files (layouts differ; NA where a member did not report)."""

    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(p for p in (root / "results").glob("e*") if p.is_dir()):
        experiment = experiment_dir.name
        selected_models = experiment_dir / "selected_models.csv"
        latency_csv = experiment_dir / "latency.csv"
        model_size_csv = experiment_dir / "model_size.csv"
        selected_config = next((p for p in (experiment_dir / "selected_config.json", experiment_dir / "handoff/selected_config.json") if p.is_file()), None)
        config_blob = common.json.loads(selected_config.read_text(encoding="utf-8")) if selected_config else {}
        if selected_models.is_file():
            # E1 records its latency scope in configs/e1.json rather than in selected_models.csv
            scope = {}
            experiment_config = root / f"configs/{experiment}.json"
            if experiment_config.is_file():
                scope = common.json.loads(experiment_config.read_text(encoding="utf-8")).get("latency", {})
            for _, row in pd.read_csv(selected_models).iterrows():
                size_bytes, size_mb = row.get("model_size_bytes"), row.get("model_size_mb")
                includes = row.get("latency_includes_vectorization", scope.get("includes_vectorization"))
                if pd.notna(row.get("features")):
                    count, kind = row.get("features"), "tfidf_features"
                elif pd.notna(row.get("parameters", config_blob.get("parameters"))):
                    count, kind = row.get("parameters", config_blob.get("parameters")), "parameters"
                else:
                    count, kind = row.get("vocab_size"), "vocabulary_size"
                device = row.get("latency_device")
                rows.append({"experiment": experiment, "model_id": row.get("model_id"), "count": count, "count_kind": kind, "size_mb": float(size_mb) if pd.notna(size_mb) else (float(size_bytes) / BYTES_PER_MIB if pd.notna(size_bytes) else None), "device": device if isinstance(device, str) else "not reported", "warmup": row.get("latency_warmup", scope.get("warmup_samples")), "timed": row.get("latency_measured", scope.get("measured_samples")), "includes_tokenization_or_vectorization": None if includes is None or pd.isna(includes) else bool(includes), "median_ms": row.get("batch1_latency_ms_median"), "p95_ms": row.get("batch1_latency_ms_p95"), "peak_vram_mb": None, "source_file": selected_models.relative_to(root).as_posix()})
        elif latency_csv.is_file():
            sizes = pd.read_csv(model_size_csv).set_index("model_id") if model_size_csv.is_file() else pd.DataFrame()
            for _, row in pd.read_csv(latency_csv).iterrows():
                model_id = row.get("model_id")
                size_bytes = sizes["checkpoint_bytes"].get(model_id) if "checkpoint_bytes" in sizes.columns else None
                device = str(row.get("device", "not reported"))
                rows.append({"experiment": experiment, "model_id": model_id, "count": sizes["n_parameters"].get(model_id) if "n_parameters" in sizes.columns else config_blob.get("cost", {}).get("n_parameters"), "count_kind": "parameters", "size_mb": float(size_bytes) / BYTES_PER_MIB if size_bytes is not None and pd.notna(size_bytes) else None, "device": device, "warmup": row.get("warmup_trials"), "timed": row.get("measured_trials"), "includes_tokenization_or_vectorization": row.get("includes_tokenization"), "median_ms": row.get("median_ms"), "p95_ms": row.get("p95_ms"), "peak_vram_mb": config_blob.get("cost", {}).get("inference_peak_vram_mb") if device.startswith("cuda") else None, "source_file": latency_csv.relative_to(root).as_posix()})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "results/e5/main_table").resolve()
    frames = common.load_processed(root)
    for frame in frames.values():
        frame["sample_id"] = frame["sample_id"].astype(str)
    e6_path = args.e6_thresholds or root / "results/e6/thresholds.csv"
    e6 = load_e6_thresholds(e6_path)
    print(f"[main table] E6 thresholds: {'found at ' + str(e6_path) if e6 is not None else 'not found (results/e6/thresholds.csv); tF1/t5 are provisional'}", flush=True)

    sources = discover_sources(root, args.source)
    report: dict[str, Any] = {"e6_thresholds_file": str(e6_path) if e6 is not None else None, "files": {}, "missing_experiments": [], "provisional_thresholds_for": []}
    long_rows: list[dict[str, Any]] = []
    d4_rows: list[dict[str, Any]] = []
    present: set[str] = set()
    for experiment, path in sources:
        relative = describe(path, root)
        try:
            frame = pd.read_parquet(path)
            errors, warnings = validate_file(frame, frames)
        except Exception as exc:  # a teammate's file must never kill the whole table
            frame, errors, warnings = pd.DataFrame(), [f"could not read/validate: {type(exc).__name__}: {exc}"], []
        model_id = str(frame["model_id"].iloc[0]) if "model_id" in frame.columns and len(frame) else None
        report["files"][relative] = {"experiment": experiment, "rows": int(len(frame)), "model_id": model_id, "errors": errors, "warnings": warnings}
        print(f"[main table] {relative}: {len(frame)} rows, {len(errors)} errors, {len(warnings)} warnings", flush=True)
        for message in errors:
            print(f"    error: {message}", flush=True)
        for message in warnings:
            print(f"    warning: {message}", flush=True)
        if errors or model_id is None:
            continue
        present.add(experiment)
        display = DISPLAY_ALIASES.get(model_id, model_id)
        is_main = experiment in MAIN_PIPELINE_EXPERIMENTS
        dev = frame[(frame["dataset"] == "d1") & (frame["split"] == "validation")]
        provisional = select_thresholds(pd.to_numeric(dev["gold"]).to_numpy(dtype=int), dev["score_attack"].to_numpy(dtype=float))
        policies: dict[str, tuple[float, str]] = {"default_0.5": (0.5, "default")}
        for policy in ("tF1", "t5"):
            match = e6[(e6["model_id"] == model_id) & (e6["policy"] == policy)] if e6 is not None else pd.DataFrame()
            if len(match):
                policies[policy] = (float(match["threshold"].iloc[0]), "e6")
            else:
                policies[policy] = (float(provisional[policy]["threshold"]), "provisional")
                if e6 is not None:
                    report["provisional_thresholds_for"].append(f"{display}/{policy}")
        for (dataset, split), group in frame.groupby(["dataset", "split"], sort=True):
            gold = pd.to_numeric(group["gold"]).to_numpy(dtype=int)
            scores = group["score_attack"].to_numpy(dtype=float)
            for policy, (threshold, source) in policies.items():
                base = {"experiment": experiment, "model_id": model_id, "display_model_id": display, "is_main_pipeline": is_main, "threshold_policy": policy, "threshold": threshold, "threshold_source": source}
                if dataset == "d4":
                    fp = int((scores >= threshold).sum())
                    low, high = wilson_interval(fp, len(group))
                    d4_rows.append({**base, "n": int(len(group)), "fp": fp, "fpr": fp / len(group), "ci_low": low, "ci_high": high})
                    continue
                long_rows.append({**base, "dataset": dataset, "split": split, **metric_record(gold, scores, threshold)})
    report["missing_experiments"] = [e for e in MAIN_PIPELINE_EXPERIMENTS if e not in present]
    if not long_rows:
        raise SystemExit("[main table] no valid raw-score file found")
    long = pd.DataFrame(long_rows)
    long["experiment_order"] = long["experiment"].map(lambda e: EXPERIMENT_ORDER.get(e, 99))
    long = long.sort_values(["experiment_order", "display_model_id", "dataset", "split", "threshold_policy"]).drop(columns="experiment_order")
    d4 = pd.DataFrame(d4_rows)
    cost = collect_cost_rows(root)
    mixed = len(set(long["threshold_source"]) - {"default"}) > 1

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(long, output_dir / "main_table_long.csv")
    if not d4.empty:
        atomic_csv(d4, output_dir / "d4_false_alarms.csv")
    if not cost.empty:
        atomic_csv(cost, output_dir / "cost_table.csv")
    atomic_json(report, output_dir / "validation_report.json")

    main_rows = long[long["is_main_pipeline"]]
    supplementary = long[~long["is_main_pipeline"]]
    status = "all tF1/t5 thresholds come from E6" if e6 is not None and not report["provisional_thresholds_for"] else ("MIXED: " + ", ".join(report["provisional_thresholds_for"]) + " fall back to provisional thresholds" if e6 is not None else "PROVISIONAL: E6 has not delivered results/e6/thresholds.csv yet; tF1/t5 are re-derived per model from its own D1 validation rows with E1's rule and must be replaced")
    parts = [
        "# Cross-source main table (generated by experiments/e5/build_main_table.py)",
        "",
        f"Threshold status: **{status}**.",
        f"Missing main-pipeline experiments: {', '.join(report['missing_experiments']) or 'none'}.",
        "Each row is one frozen model scored on each test source separately; sources are never pooled. The `threshold source` column is per row" + (" (sources are mixed in this build)." if mixed else "."),
        "",
        "## Main pipelines (E1-E4)",
        "",
    ]
    for policy in ("tF1", "t5", "default_0.5"):
        parts.append(pivot_markdown(main_rows, policy, "Threshold policy"))
    if not supplementary.empty:
        parts += ["## Supplementary: E5 clean versus typo-augmented (same frozen E1 Combined features)", ""]
        for policy in ("tF1", "default_0.5"):
            parts.append(pivot_markdown(supplementary, policy, "Threshold policy"))
    if not d4.empty:
        parts += ["## D4 NotInject false alarms (benign-only; FP/N with Wilson 95% CI)", "", "| Experiment | model_id | policy | threshold source | threshold | FP / N | FPR | 95% CI |", "|---|---|---|---|---:|---:|---:|---|"]
        for _, row in d4.sort_values(["experiment", "display_model_id", "threshold_policy"]).iterrows():
            parts.append(f"| {row['experiment']} | `{row['display_model_id']}` | {row['threshold_policy']} | {row['threshold_source']} | {row['threshold']:.4f} | {int(row['fp'])} / {int(row['n'])} | {pct(row['fpr'])} | {pct(row['ci_low'])} to {pct(row['ci_high'])} |")
        parts.append("")
    if not cost.empty:
        parts += ["## Cost (as reported by each member; batch size 1, but hardware and timing scope differ per experiment)", "", "| Experiment | model_id | count | count kind | size (MiB) | device | warm-up | timed | tokenization or vectorization included | median (ms) | P95 (ms) | peak VRAM (MiB) |", "|---|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|"]
        def cell(value: Any, fmt: str = "{:.2f}") -> str:
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                return ""
            if isinstance(value, (bool, np.bool_)):
                return "yes" if value else "no"
            if isinstance(value, (int, float, np.integer, np.floating)):
                return fmt.format(float(value))
            return str(value)

        for _, row in cost.iterrows():
            parts.append(f"| {row['experiment']} | `{row['model_id']}` | {cell(row['count'], '{:,.0f}')} | {cell(row['count_kind'])} | {cell(row['size_mb'])} | {cell(row['device'])} | {cell(row['warmup'], '{:.0f}')} | {cell(row['timed'], '{:.0f}')} | {cell(row['includes_tokenization_or_vectorization'])} | {cell(row['median_ms'])} | {cell(row['p95_ms'])} | {cell(row['peak_vram_mb'], '{:.0f}')} |")
        parts.append("")
    parts += ["## Validation of the input files", ""]
    for file_name, info in report["files"].items():
        status_word = "rejected" if info["errors"] else "accepted"
        parts.append(f"- `{file_name}` ({info['experiment']}, {info['rows']} rows): {status_word}" + (f"; errors: {info['errors']}" if info["errors"] else "") + (f"; warnings: {info['warnings']}" if info["warnings"] else ""))
    write_text("\n".join(parts) + "\n", output_dir / "main_table.md")
    print(f"[main table] written to {output_dir}", flush=True)
    print(pivot_markdown(main_rows, "default_0.5", "Threshold policy"), flush=True)


if __name__ == "__main__":
    main()
