#!/usr/bin/env python3
"""Independently verify PromptShield raw, processed, audit, and manifest data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "data/audit/integrity_report.json",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_common_rows(
    processed: pd.DataFrame,
    raw: pd.DataFrame,
    dataset: str,
    split: str,
    text_column: str = "text",
    sample_split_column: str | None = None,
) -> None:
    require(processed["sample_id"].is_unique, f"{dataset}/{split}: duplicate sample_id")
    require(processed["source_row"].is_unique, f"{dataset}/{split}: duplicate source_row")
    for row in processed.itertuples(index=False):
        source_row = int(row.source_row)
        require(0 <= source_row < len(raw), f"{row.sample_id}: source_row out of range")
        source = raw.iloc[source_row]
        text = source[text_column]
        require(row.text == text, f"{row.sample_id}: text differs from raw source")
        require(int(row.label) == int(source["label"]), f"{row.sample_id}: label mismatch")
        sample_split = (
            str(getattr(row, sample_split_column)) if sample_split_column else split
        )
        expected_id = f"{dataset}:{sample_split}:{source_row:06d}"
        require(row.sample_id == expected_id, f"{row.sample_id}: unstable sample_id")
        require(row.text_sha256 == sha256_text(text), f"{row.sample_id}: text hash mismatch")
        require(
            row.normalized_sha256 == sha256_text(normalize(text)),
            f"{row.sample_id}: normalized hash mismatch",
        )


def expected_d1_rows(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, list[int]], dict[tuple[str, int], tuple[str, str]]]:
    conflicting = set(
        train.groupby("text", sort=False)["label"].nunique().loc[lambda values: values > 1].index
    )
    validation_test_overlap = set(validation["text"]) & set(test["text"])
    retained_validation_rows = [
        index for index, text in enumerate(validation["text"]) if text not in validation_test_overlap
    ]
    retained_validation_texts = set(validation.iloc[retained_validation_rows]["text"])
    test_texts = set(test["text"])

    retained_train_rows: list[int] = []
    seen: set[tuple[str, int]] = set()
    decisions: dict[tuple[str, int], tuple[str, str]] = {}
    for index, row in train.iterrows():
        text = row["text"]
        label = int(row["label"])
        if text in conflicting:
            decisions[("train", int(index))] = ("exclude", "conflicting_labels")
        elif text in test_texts:
            decisions[("train", int(index))] = ("exclude", "overlap_with_test")
        elif text in retained_validation_texts:
            decisions[("train", int(index))] = ("exclude", "overlap_with_validation")
        elif (text, label) in seen:
            decisions[("train", int(index))] = ("exclude", "duplicate_same_label")
        else:
            seen.add((text, label))
            retained_train_rows.append(int(index))
            decisions[("train", int(index))] = ("keep", "retained")

    for index, text in enumerate(validation["text"]):
        decisions[("validation", index)] = (
            ("exclude", "overlap_with_test")
            if text in validation_test_overlap
            else ("keep", "retained")
        )
    for index in range(len(test)):
        decisions[("test", index)] = ("keep", "retained")

    return (
        {
            "train": retained_train_rows,
            "validation": retained_validation_rows,
            "test": list(range(len(test))),
        },
        decisions,
    )


def run_checks(project_root: Path) -> dict[str, Any]:
    data_root = project_root / "data"
    raw_root = data_root / "raw"
    processed_root = data_root / "processed"
    audit_root = data_root / "audit"
    config_path = project_root / "configs/data_sources.json"
    manifest_path = audit_root / "manifest.json"
    summary_path = audit_root / "summary.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}

    require(
        manifest["config"]["sha256"] == sha256_file(config_path),
        "Config hash does not match manifest",
    )
    expected_revisions = {
        key: value["revision"] for key, value in config["datasets"].items()
    }
    require(
        manifest["dataset_revisions"] == expected_revisions,
        "Pinned revisions differ between config and manifest",
    )

    for group in ("raw_files", "processed_files", "audit_files"):
        for record in manifest[group]:
            path = data_root / record["path"]
            require(path.is_file(), f"Manifest file is missing: {path}")
            require(
                sha256_file(path) == record["sha256"],
                f"Manifest hash mismatch: {path}",
            )
    checks["manifest_hashes"] = "passed"

    d1_raw = {
        "train": pd.read_csv(raw_root / "d1_train.csv"),
        "validation": pd.read_csv(raw_root / "d1_validation.csv"),
        "test": pd.read_csv(raw_root / "d1_test.csv"),
    }
    expected_rows, expected_decisions = expected_d1_rows(
        d1_raw["train"], d1_raw["validation"], d1_raw["test"]
    )
    d1_processed: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(processed_root / f"d1_{split}.parquet")
        d1_processed[split] = frame
        require(
            frame["source_row"].astype(int).tolist() == expected_rows[split],
            f"D1 {split}: retained row identities do not match independent cleaning",
        )
        verify_common_rows(frame, d1_raw[split], "d1", split)
    checks["d1_raw_mapping"] = "passed"

    audit = pd.read_csv(audit_root / "d1_decisions.csv")
    require(len(audit) == sum(map(len, d1_raw.values())), "D1 audit row count mismatch")
    require(
        not audit.duplicated(["split", "source_row"]).any(),
        "D1 audit has duplicate split/source_row keys",
    )
    for row in audit.itertuples(index=False):
        expected = expected_decisions[(str(row.split), int(row.source_row))]
        require(
            (row.decision, row.reason) == expected,
            f"D1 audit decision mismatch for {row.split}/{row.source_row}",
        )
        raw_row = d1_raw[str(row.split)].iloc[int(row.source_row)]
        require(row.raw_text == raw_row["text"], f"D1 audit text mismatch: {row.sample_id}")
        require(int(row.label) == int(raw_row["label"]), f"D1 audit label mismatch: {row.sample_id}")
    checks["d1_audit_decisions"] = "passed"

    d2_raw = pd.read_parquet(raw_root / "d2_test.parquet")
    d2 = pd.read_parquet(processed_root / "d2_test.parquet")
    verify_common_rows(d2, d2_raw, "d2", "test")
    checks["d2_raw_mapping"] = "passed"

    d3_raw = pd.read_parquet(raw_root / "d3_test.parquet")
    d3 = pd.read_parquet(processed_root / "d3_test.parquet")
    verify_common_rows(d3, d3_raw, "d3", "test")
    metadata_map = {
        "source_id": "id",
        "class": "class",
        "attack_family": "attack_family",
        "technique": "technique",
        "severity": "severity",
        "language": "language",
        "source_name": "source",
    }
    for output_column, raw_column in metadata_map.items():
        expected = d3["source_row"].map(d3_raw[raw_column])
        require(
            d3[output_column].reset_index(drop=True).equals(expected.reset_index(drop=True)),
            f"D3 metadata mismatch: {output_column}",
        )
    checks["d3_raw_and_metadata_mapping"] = "passed"

    d4 = pd.read_parquet(processed_root / "d4_test.parquet")
    d4_raw_files = {
        "one": pd.read_parquet(raw_root / "d4_one.parquet"),
        "two": pd.read_parquet(raw_root / "d4_two.parquet"),
        "three": pd.read_parquet(raw_root / "d4_three.parquet"),
    }
    for subset, raw in d4_raw_files.items():
        part = d4[d4["subset"].eq(subset)].reset_index(drop=True)
        raw_with_label = raw.copy()
        raw_with_label["label"] = 0
        verify_common_rows(
            part,
            raw_with_label,
            "d4",
            "test",
            text_column="prompt",
            sample_split_column="subset",
        )
        for row in part.itertuples(index=False):
            source = raw.iloc[int(row.source_row)]
            words = source["word_list"].tolist()
            expected_words = json.dumps([str(word) for word in words], ensure_ascii=False)
            require(row.category == source["category"], f"D4 category mismatch: {row.sample_id}")
            require(row.trigger_words_json == expected_words, f"D4 words mismatch: {row.sample_id}")
    require((d4["label"] == 0).all(), "D4 must be all benign")
    require(d4["category"].eq("Multilingual").sum() == 84, "D4 Multilingual count mismatch")
    checks["d4_raw_and_metadata_mapping"] = "passed"

    train_text = set(d1_processed["train"]["text"])
    validation_text = set(d1_processed["validation"]["text"])
    test_text = set(d1_processed["test"]["text"])
    require(not train_text & validation_text, "D1 train/validation exact leakage")
    require(not train_text & test_text, "D1 train/test exact leakage")
    require(not validation_text & test_text, "D1 validation/test exact leakage")
    normalized_train = {normalize(text) for text in train_text}
    external_overlaps = {
        "d2": len(normalized_train & {normalize(text) for text in d2["text"]}),
        "d3": len(normalized_train & {normalize(text) for text in d3["text"]}),
        "d4": len(normalized_train & {normalize(text) for text in d4["text"]}),
    }
    require(not any(external_overlaps.values()), f"Normalized external overlap: {external_overlaps}")
    checks["post_clean_overlap"] = {
        "d1_pairwise_exact": 0,
        "d1_train_external_nfkc_casefold": external_overlaps,
    }

    observed_counts = {
        "d1_train": len(d1_processed["train"]),
        "d1_validation": len(d1_processed["validation"]),
        "d1_test": len(d1_processed["test"]),
        "d2_test": len(d2),
        "d3_test": len(d3),
        "d4_test": len(d4),
    }
    require(
        observed_counts
        == {
            "d1_train": 11046,
            "d1_validation": 1985,
            "d1_test": 2101,
            "d2_test": 116,
            "d3_test": 545,
            "d4_test": 339,
        },
        f"Unexpected prepared counts: {observed_counts}",
    )
    require(
        summary["d1"]["exclusions"]
        == {
            "conflicting_labels": 2,
            "duplicate_same_label": 37,
            "overlap_with_test": 116,
            "overlap_with_validation": 4,
        },
        "Summary exclusion counts are incorrect",
    )

    return {
        "status": "passed",
        "checks": checks,
        "row_counts": observed_counts,
        "d1_exclusions": summary["d1"]["exclusions"],
        "scope_note": (
            "This verifies file identity, schema, row mapping, labels, hashes, exact cleaning, "
            "metadata preservation, and specified exact overlaps. It does not prove semantic "
            "label correctness, absence of paraphrased near-duplicates, or absence of pretraining contamination."
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_checks(args.project_root.resolve())
    atomic_json(report, args.report.resolve())
    print("PromptShield data integrity verification passed")
    print(json.dumps(report["row_counts"], sort_keys=True))
    print(json.dumps(report["d1_exclusions"], sort_keys=True))


if __name__ == "__main__":
    main()
