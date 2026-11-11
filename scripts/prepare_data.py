#!/usr/bin/env python3
"""Prepare frozen PromptShield datasets without modifying raw inputs.

The cleaning order follows the project plan exactly:
1. remove all conflicting-label D1 training groups;
2. preserve D1 test rows;
3. remove D1 validation rows whose exact text occurs in test;
4. remove D1 training rows whose exact text occurs in retained validation or test;
5. deduplicate remaining D1 training rows by exact text and label, keeping the
   first original row.

NFKC/casefold normalization is used for auditing only. Model input text is
always preserved byte-for-byte as decoded from the source table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROCESSOR_VERSION = "1.0.0"
COMMON_COLUMNS = [
    "dataset",
    "split",
    "sample_id",
    "source_row",
    "text",
    "label",
    "text_sha256",
    "normalized_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/audit"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.json"))
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_for_audit(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_text(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{name}.{column} contains null values")
    non_strings = frame[column].map(lambda value: not isinstance(value, str))
    if non_strings.any():
        raise ValueError(f"{name}.{column} contains non-string values")
    if frame[column].str.strip().eq("").any():
        raise ValueError(f"{name}.{column} contains blank values")


def validate_labels(frame: pd.DataFrame, name: str) -> None:
    require_columns(frame, ["label"], name)
    labels = set(frame["label"].dropna().astype(int).tolist())
    if labels != {0, 1}:
        raise ValueError(f"{name}.label must contain both 0 and 1; found {sorted(labels)}")
    if frame["label"].isna().any():
        raise ValueError(f"{name}.label contains null values")


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def check_expected_raw(
    raw_dir: Path, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset in config["datasets"].values():
        for filename, expected in dataset["files"].items():
            path = raw_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing pinned raw file: {path}")
            frame = read_table(path)
            frames[filename] = frame
            if len(frame) != expected["rows"]:
                raise ValueError(
                    f"{filename}: expected {expected['rows']} rows, found {len(frame)}"
                )
            if "labels" in expected:
                require_columns(frame, ["text", "label"], filename)
                validate_text(frame, "text", filename)
                validate_labels(frame, filename)
                observed = label_counts(frame)
                if observed != expected["labels"]:
                    raise ValueError(
                        f"{filename}: expected labels {expected['labels']}, found {observed}"
                    )
            else:
                require_columns(frame, ["prompt", "word_list", "category"], filename)
                validate_text(frame, "prompt", filename)
    return frames


def sample_ids(dataset: str, split: str, rows: Iterable[int]) -> list[str]:
    return [f"{dataset}:{split}:{int(row):06d}" for row in rows]


def common_frame(
    source: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    text_column: str = "text",
    sample_split: str | None = None,
) -> pd.DataFrame:
    rows = source["_source_row"].astype(int).tolist()
    texts = source[text_column].tolist()
    labels = source["label"].astype(int).tolist()
    normalized = [normalize_for_audit(text) for text in texts]
    return pd.DataFrame(
        {
            "dataset": dataset,
            "split": split,
            "sample_id": sample_ids(dataset, sample_split or split, rows),
            "source_row": rows,
            "text": texts,
            "label": labels,
            "text_sha256": [sha256_text(text) for text in texts],
            "normalized_sha256": [sha256_text(text) for text in normalized],
        }
    )


def _attach_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_source_row"] = range(len(result))
    return result


def clean_d1(
    train_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    train = _attach_source_rows(train_raw)
    validation = _attach_source_rows(validation_raw)
    test = _attach_source_rows(test_raw)

    conflicting_counts = train.groupby("text", sort=False)["label"].nunique()
    conflicting_texts = set(conflicting_counts[conflicting_counts > 1].index)

    validation_test_overlap = set(validation["text"]) & set(test["text"])
    retained_validation = validation[~validation["text"].isin(validation_test_overlap)].copy()
    retained_held_out_texts = set(retained_validation["text"]) | set(test["text"])

    train_conflict_mask = train["text"].isin(conflicting_texts)
    train_after_conflicts = train[~train_conflict_mask].copy()
    train_held_out_mask = train_after_conflicts["text"].isin(retained_held_out_texts)
    train_after_held_out = train_after_conflicts[~train_held_out_mask].copy()
    duplicate_mask = train_after_held_out.duplicated(["text", "label"], keep="first")
    retained_train = train_after_held_out[~duplicate_mask].copy()

    cleaned = {
        "train": common_frame(retained_train, dataset="d1", split="train"),
        "validation": common_frame(
            retained_validation, dataset="d1", split="validation"
        ),
        "test": common_frame(test, dataset="d1", split="test"),
    }

    audit_parts: list[pd.DataFrame] = []
    for split, frame in (("train", train), ("validation", validation), ("test", test)):
        part = common_frame(frame, dataset="d1", split=split).rename(
            columns={"text": "raw_text"}
        )
        part["decision"] = "keep"
        part["reason"] = "retained"
        audit_parts.append(part)
    audit = pd.concat(audit_parts, ignore_index=True)

    train_audit = audit["split"].eq("train")
    conflict_rows = train.loc[train_conflict_mask, "_source_row"].astype(int)
    conflict_audit = train_audit & audit["source_row"].isin(conflict_rows)
    audit.loc[conflict_audit, ["decision", "reason"]] = [
        "exclude",
        "conflicting_labels",
    ]

    held_out_rows = train_after_conflicts.loc[train_held_out_mask, "_source_row"].astype(int)
    for row in held_out_rows:
        text = train.loc[train["_source_row"].eq(row), "text"].iloc[0]
        reason = "overlap_with_test" if text in set(test["text"]) else "overlap_with_validation"
        mask = train_audit & audit["source_row"].eq(row)
        audit.loc[mask, ["decision", "reason"]] = ["exclude", reason]

    duplicate_rows = train_after_held_out.loc[duplicate_mask, "_source_row"].astype(int)
    duplicate_audit = train_audit & audit["source_row"].isin(duplicate_rows)
    audit.loc[duplicate_audit, ["decision", "reason"]] = [
        "exclude",
        "duplicate_same_label",
    ]

    validation_audit = audit["split"].eq("validation")
    validation_overlap_rows = validation.loc[
        validation["text"].isin(validation_test_overlap), "_source_row"
    ].astype(int)
    validation_overlap_audit = validation_audit & audit["source_row"].isin(
        validation_overlap_rows
    )
    audit.loc[validation_overlap_audit, ["decision", "reason"]] = [
        "exclude",
        "overlap_with_test",
    ]

    audit = audit[
        [
            "dataset",
            "split",
            "sample_id",
            "source_row",
            "raw_text",
            "label",
            "text_sha256",
            "normalized_sha256",
            "decision",
            "reason",
        ]
    ]

    raw_normalized = train["text"].map(normalize_for_audit)
    stats = {
        "raw": {
            "train": {"rows": len(train), "labels": label_counts(train)},
            "validation": {
                "rows": len(validation),
                "labels": label_counts(validation),
            },
            "test": {"rows": len(test), "labels": label_counts(test)},
        },
        "observations": {
            "train_conflicting_text_groups": len(conflicting_texts),
            "train_conflicting_rows": int(train_conflict_mask.sum()),
            "train_exact_text_duplicate_excess": int(train.duplicated("text").sum()),
            "train_same_label_duplicate_excess": int(
                train.duplicated(["text", "label"]).sum()
            ),
            "train_nfkc_casefold_duplicate_excess": int(raw_normalized.duplicated().sum()),
            "train_validation_exact_overlap_unique_texts": len(
                set(train["text"]) & set(validation["text"])
            ),
            "validation_test_exact_overlap_unique_texts": len(validation_test_overlap),
            "train_test_exact_overlap_unique_texts": len(
                set(train["text"]) & set(test["text"])
            ),
        },
        "exclusions": {
            reason: int(count)
            for reason, count in audit.loc[audit["decision"].eq("exclude"), "reason"]
            .value_counts()
            .sort_index()
            .items()
        },
        "processed": {
            split: {"rows": len(frame), "labels": label_counts(frame)}
            for split, frame in cleaned.items()
        },
    }
    return cleaned, audit, stats


def prepare_d2(frame: pd.DataFrame) -> pd.DataFrame:
    source = _attach_source_rows(frame)
    return common_frame(source, dataset="d2", split="test")


def prepare_d3(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "text",
        "label",
        "class",
        "attack_family",
        "technique",
        "severity",
        "language",
        "source",
        "id",
    ]
    require_columns(frame, required, "d3_test.parquet")
    source = _attach_source_rows(frame)
    common = common_frame(source, dataset="d3", split="test")
    metadata = source[
        [
            "id",
            "class",
            "attack_family",
            "technique",
            "severity",
            "language",
            "source",
        ]
    ].reset_index(drop=True)
    metadata = metadata.rename(columns={"id": "source_id", "source": "source_name"})
    return pd.concat([common, metadata], axis=1)


def _word_list_json(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return json.dumps([str(item) for item in value], ensure_ascii=False)


def prepare_d4(frames: list[tuple[str, int, pd.DataFrame]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for subset, trigger_count, frame in frames:
        source = _attach_source_rows(frame)
        source["label"] = 0
        common = common_frame(
            source,
            dataset="d4",
            split="test",
            text_column="prompt",
            sample_split=subset,
        )
        common["subset"] = subset
        common["trigger_count"] = trigger_count
        common["category"] = source["category"].astype(str).tolist()
        common["trigger_words_json"] = source["word_list"].map(_word_list_json).tolist()
        parts.append(common)
    return pd.concat(parts, ignore_index=True)


def atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    frame = read_table(path)
    record: dict[str, Any] = {
        "path": str(path.relative_to(relative_to)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    if "label" in frame.columns:
        record["labels"] = label_counts(frame)
    return record


def assert_processed_expectations(
    cleaned_d1: dict[str, pd.DataFrame], config: dict[str, Any]
) -> None:
    expected = config["datasets"]["d1"]["expected_processed"]
    for split, frame in cleaned_d1.items():
        observed = {"rows": len(frame), "labels": label_counts(frame)}
        if observed != expected[split]:
            raise AssertionError(
                f"D1 {split}: expected {expected[split]}, found {observed}"
            )


def verify_no_overlap(
    cleaned_d1: dict[str, pd.DataFrame], d2: pd.DataFrame, d3: pd.DataFrame, d4: pd.DataFrame
) -> dict[str, Any]:
    train = cleaned_d1["train"]
    validation = cleaned_d1["validation"]
    test = cleaned_d1["test"]
    train_text = set(train["text"])
    train_normalized = set(train["text"].map(normalize_for_audit))
    checks = {
        "d1_train_validation_exact": len(train_text & set(validation["text"])),
        "d1_train_test_exact": len(train_text & set(test["text"])),
        "d1_validation_test_exact": len(set(validation["text"]) & set(test["text"])),
        "d1_train_d2_test_nfkc_casefold": len(
            train_normalized & set(d2["text"].map(normalize_for_audit))
        ),
        "d1_train_d3_test_nfkc_casefold": len(
            train_normalized & set(d3["text"].map(normalize_for_audit))
        ),
        "d1_train_d4_test_nfkc_casefold": len(
            train_normalized & set(d4["text"].map(normalize_for_audit))
        ),
    }
    nonzero = {name: count for name, count in checks.items() if count != 0}
    if nonzero:
        raise AssertionError(f"Unexpected post-cleaning overlap: {nonzero}")
    return checks


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_frames = check_expected_raw(args.raw_dir, config)

    cleaned_d1, d1_audit, d1_stats = clean_d1(
        raw_frames["d1_train.csv"],
        raw_frames["d1_validation.csv"],
        raw_frames["d1_test.csv"],
    )
    assert_processed_expectations(cleaned_d1, config)

    d2 = prepare_d2(raw_frames["d2_test.parquet"])
    d3 = prepare_d3(raw_frames["d3_test.parquet"])
    d4 = prepare_d4(
        [
            ("one", 1, raw_frames["d4_one.parquet"]),
            ("two", 2, raw_frames["d4_two.parquet"]),
            ("three", 3, raw_frames["d4_three.parquet"]),
        ]
    )
    overlap_checks = verify_no_overlap(cleaned_d1, d2, d3, d4)

    outputs = {
        "d1_train.parquet": cleaned_d1["train"],
        "d1_validation.parquet": cleaned_d1["validation"],
        "d1_test.parquet": cleaned_d1["test"],
        "d2_test.parquet": d2,
        "d3_test.parquet": d3,
        "d4_test.parquet": d4,
    }
    for filename, frame in outputs.items():
        atomic_write_parquet(frame, args.processed_dir / filename)
    atomic_write_csv(d1_audit, args.audit_dir / "d1_decisions.csv")

    summary = {
        "schema_version": 1,
        "processor_version": PROCESSOR_VERSION,
        "text_policy": {
            "model_input": "original_text_unchanged",
            "audit_normalization": "Unicode NFKC followed by casefold",
        },
        "d1": d1_stats,
        "external_tests": {
            "d2": {"rows": len(d2), "labels": label_counts(d2)},
            "d3": {"rows": len(d3), "labels": label_counts(d3)},
            "d4": {
                "rows": len(d4),
                "labels": label_counts(d4),
                "trigger_count_rows": {
                    str(int(key)): int(value)
                    for key, value in d4["trigger_count"].value_counts().sort_index().items()
                },
                "category_rows": {
                    str(key): int(value)
                    for key, value in d4["category"].value_counts().sort_index().items()
                },
            },
        },
        "post_clean_overlap_checks": overlap_checks,
        "limitations": [
            "Near-duplicate and semantic-overlap review is not automated in this stage.",
            "NFKC/casefold hashes are audit aids and are never substituted for model input.",
            "D2 train and D3 train/validation are retained only as pinned raw files and are not exported for model fitting.",
        ],
    }
    atomic_write_json(summary, args.audit_dir / "summary.json")

    raw_records = [
        file_record(args.raw_dir / filename, args.raw_dir.parent)
        for filename in sorted(raw_frames)
    ]
    processed_records = [
        file_record(args.processed_dir / filename, args.processed_dir.parent)
        for filename in sorted(outputs)
    ]
    manifest = {
        "schema_version": 1,
        "processor_version": PROCESSOR_VERSION,
        "config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
        },
        "dataset_revisions": {
            key: value["revision"] for key, value in config["datasets"].items()
        },
        "raw_files": raw_records,
        "processed_files": processed_records,
        "audit_files": [
            {
                "path": str((args.audit_dir / "d1_decisions.csv").relative_to(args.audit_dir.parent)),
                "sha256": sha256_file(args.audit_dir / "d1_decisions.csv"),
                "bytes": (args.audit_dir / "d1_decisions.csv").stat().st_size,
                "rows": len(d1_audit),
            },
            {
                "path": str((args.audit_dir / "summary.json").relative_to(args.audit_dir.parent)),
                "sha256": sha256_file(args.audit_dir / "summary.json"),
                "bytes": (args.audit_dir / "summary.json").stat().st_size,
            },
        ],
    }
    atomic_write_json(manifest, args.audit_dir / "manifest.json")

    print("Prepared PromptShield data successfully")
    for filename, frame in outputs.items():
        print(f"  {filename}: {len(frame):,} rows")
    print(f"  audit decisions: {len(d1_audit):,} rows")
    print("  all frozen count and overlap checks passed")


if __name__ == "__main__":
    main()
