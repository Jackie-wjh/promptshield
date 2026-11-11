"""Shared helpers for E5 (controlled typo augmentation on the frozen E1 Combined pipeline).

Everything that must be *identical* to E1 (metric_record, threshold selection, atomic writers,
vectorizer/classifier construction) is re-implemented here verbatim, because
experiments/e1/run_e1.py imports a ``scripts`` package that is not part of the handoff package
and therefore cannot be imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_SPLITS: tuple[tuple[str, str, str], ...] = (
    # (file stem, dataset, split)
    ("d1_train", "d1", "train"),
    ("d1_validation", "d1", "validation"),
    ("d1_test", "d1", "test"),
    ("d2_test", "d2", "test"),
    ("d3_test", "d3", "test"),
    ("d4_test", "d4", "test"),
)
SPLIT_META: dict[str, tuple[str, str]] = {stem: (dataset, split) for stem, dataset, split in PROCESSED_SPLITS}
EVAL_SPLITS: dict[str, tuple[str, str]] = {stem: meta for stem, meta in SPLIT_META.items() if stem != "d1_train"}

HANDOFF_COLUMNS = [
    "dataset",
    "split",
    "sample_id",
    "gold",
    "score_attack",
    "model_id",
    "checkpoint_rev",
    "seed",
    "transform_id",
    "parent_id",
]

# README section 9: the prediction schema E6 returns (raw score + threshold policy + prediction).
PREDICTION_COLUMNS = [
    "dataset",
    "split",
    "sample_id",
    "gold",
    "score_attack",
    "threshold_policy",
    "threshold",
    "pred",
    "model_id",
    "checkpoint_rev",
    "seed",
    "transform_id",
    "parent_id",
]


# --------------------------------------------------------------------------------------
# hashing and normalisation
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_WHITESPACE = re.compile(r"\s+")


def normalized_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse: the equivalence used for duplicate checks."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def normalized_sha256(text: str) -> str:
    return sha256_text(normalized_text(text))


# --------------------------------------------------------------------------------------
# atomic writers (identical behaviour to E1)
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_joblib(model: Any, path: Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)


def write_text(text: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv_for_humans(frame: pd.DataFrame, path: Path) -> None:
    """CSV meant to be opened in Excel/WPS: UTF-8 with BOM so non-ASCII text survives the round trip."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", encoding="utf-8-sig")
    os.replace(temporary, path)


def read_csv_from_humans(path: Path, text_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read a CSV a person may have re-saved from Excel: UTF-8 (with or without BOM) first, then GBK."""

    dtype = {column: "string" for column in text_columns}
    for encoding in ("utf-8-sig", "cp936"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=dtype, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"{path} is neither UTF-8 nor GBK")


QA_TRUE = {"1", "1.0", "TRUE", "T", "Y", "YES", "OK", "是", "对"}
QA_FALSE = {"0", "0.0", "FALSE", "F", "N", "NO", "否", "错"}


def parse_qa_marks(series: pd.Series) -> tuple[pd.Series, pd.Series, list[str]]:
    """Map a human-filled yes/no column to (flag 1/0 or NA, reviewed mask, unrecognised values).

    Accepts 1/0, 1.0/0.0, TRUE/FALSE (what Excel writes for booleans), Y/N, YES/NO and 是/否.
    """

    raw = series.astype("string").fillna("").str.strip()
    upper = raw.str.upper()
    flag = pd.Series(pd.NA, index=series.index, dtype="Int64")
    flag[upper.isin(QA_TRUE)] = 1
    flag[upper.isin(QA_FALSE)] = 0
    reviewed = flag.notna()
    unrecognised = sorted(set(raw[(raw != "") & ~reviewed].tolist()))
    return flag, reviewed, unrecognised


# --------------------------------------------------------------------------------------
# data gate and loading
# --------------------------------------------------------------------------------------


def verify_data_integrity(project_root: Path) -> dict[str, Any]:
    """Recompute SHA-256, row counts and label counts of the six processed files.

    The handoff package does not ship ``scripts/check_data_integrity.py``, so this gate
    re-derives the checks from ``data/audit/manifest.json`` (same approach as E3).
    """

    project_root = Path(project_root)
    manifest_path = project_root / "data/audit/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    status = "passed"
    for item in manifest["processed_files"]:
        path = project_root / "data" / item["path"]
        frame = pd.read_parquet(path)
        actual_sha = sha256_file(path)
        labels = {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
        record = {
            "manifest_sha256": item["sha256"],
            "actual_sha256": actual_sha,
            "manifest_rows": int(item["rows"]),
            "actual_rows": int(len(frame)),
            "manifest_labels": {str(k): int(v) for k, v in item["labels"].items()},
            "actual_labels": labels,
            "sample_id_unique": bool(frame["sample_id"].is_unique),
        }
        ok = (
            record["manifest_sha256"] == record["actual_sha256"]
            and record["manifest_rows"] == record["actual_rows"]
            and record["manifest_labels"] == record["actual_labels"]
            and record["sample_id_unique"]
        )
        record["status"] = "passed" if ok else "failed"
        if not ok:
            status = "failed"
        checks[Path(item["path"]).stem] = record
    return {
        "status": status,
        "manifest_path": str(manifest_path.relative_to(project_root)).replace("\\", "/"),
        "manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
        "note": "Computed by experiments/e5/common.py because the handoff package does not include scripts/check_data_integrity.py.",
    }


def load_processed(project_root: Path) -> dict[str, pd.DataFrame]:
    processed = Path(project_root) / "data/processed"
    frames: dict[str, pd.DataFrame] = {}
    for stem, _dataset, _split in PROCESSED_SPLITS:
        frame = pd.read_parquet(processed / f"{stem}.parquet").reset_index(drop=True)
        frame["text"] = frame["text"].astype(str)
        frames[stem] = frame
    return frames


def all_normalized_hashes(frames: dict[str, pd.DataFrame]) -> set[str]:
    """Normalized hashes of every processed text (all splits) for the duplicate-equivalence check."""

    hashes: set[str] = set()
    for frame in frames.values():
        hashes.update(normalized_sha256(text) for text in frame["text"].tolist())
    return hashes


# --------------------------------------------------------------------------------------
# metrics and thresholds (verbatim E1 logic)
# --------------------------------------------------------------------------------------


def metric_record(gold: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    gold = np.asarray(gold, dtype=int)
    scores = np.asarray(scores, dtype=float)
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
    unique = np.unique(np.asarray(scores, dtype=float))
    all_negative = np.nextafter(unique.max(), np.inf)
    return np.concatenate([unique, [all_negative]])


def _fast_counts_per_candidate(gold: np.ndarray, scores: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """tp, fp, tn, fn for every candidate threshold (pred = score >= threshold), by sorting once."""

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_gold = gold[order]
    n_attack = int(gold.sum())
    n_benign = int(len(gold) - n_attack)
    # number of rows with score < candidate (searchsorted left) -> rows predicted benign
    below = np.searchsorted(sorted_scores, candidates, side="left")
    attack_below = np.concatenate([[0], np.cumsum(sorted_gold == 1)])[below]
    benign_below = below - attack_below
    tp = n_attack - attack_below
    fn = attack_below
    tn = benign_below
    fp = n_benign - benign_below
    return tp, fp, tn, fn


def select_thresholds(gold: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    """tF1 (max Macro F1, ties -> higher threshold) and t5 (max recall s.t. FPR <= 5%, ties -> higher).

    Same rule and same tie-breaks as experiments/e1/run_e1.py. The exhaustive E1 search calls
    sklearn once per candidate (minutes for six models), so the candidates are first ranked with
    exact confusion counts; the final comparison among the near-maximal candidates uses E1's own
    ``metric_record`` arithmetic, so the selected thresholds are identical to E1's search.
    """

    gold = np.asarray(gold, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidates = threshold_candidates(scores)
    tp, fp, tn, fn = _fast_counts_per_candidate(gold, scores, candidates)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_positive = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        f1_negative = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
        macro = (f1_positive + f1_negative) / 2.0
        fpr = np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), np.nan)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    near_best_f1 = np.flatnonzero(macro >= macro.max() - 1e-9)
    tf1_records = [(float(candidates[i]), metric_record(gold, scores, float(candidates[i]))) for i in near_best_f1]
    tf1_threshold, tf1_metrics = max(tf1_records, key=lambda item: (item[1]["macro_f1"], item[0]))
    eligible = np.flatnonzero(fpr <= 0.05)
    best_recall = recall[eligible].max()
    near_best_recall = eligible[recall[eligible] >= best_recall - 1e-9]
    t5_records = [(float(candidates[i]), metric_record(gold, scores, float(candidates[i]))) for i in near_best_recall]
    t5_records = [item for item in t5_records if item[1]["fpr"] is not None and item[1]["fpr"] <= 0.05]
    t5_threshold, t5_metrics = max(t5_records, key=lambda item: (item[1]["attack_recall"], item[0]))
    return {
        "default_0.5": {"threshold": 0.5, **metric_record(gold, scores, 0.5)},
        "tF1": {"threshold": tf1_threshold, **tf1_metrics},
        "t5": {"threshold": t5_threshold, **t5_metrics},
    }


def select_thresholds_exhaustive(gold: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    """E1's original exhaustive search, kept verbatim for cross-checking ``select_thresholds``."""

    candidates = threshold_candidates(scores)
    records = [(float(threshold), metric_record(gold, scores, float(threshold))) for threshold in candidates]
    tf1_threshold, tf1_metrics = max(records, key=lambda item: (item[1]["macro_f1"], item[0]))
    eligible_t5 = [item for item in records if item[1]["fpr"] is not None and item[1]["fpr"] <= 0.05]
    t5_threshold, t5_metrics = max(eligible_t5, key=lambda item: (item[1]["attack_recall"], item[0]))
    return {
        "default_0.5": {"threshold": 0.5, **metric_record(gold, scores, 0.5)},
        "tF1": {"threshold": tf1_threshold, **tf1_metrics},
        "t5": {"threshold": t5_threshold, **t5_metrics},
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (same formula as E3's analysis)."""

    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


# --------------------------------------------------------------------------------------
# E1 model construction (verbatim) and frozen-feature handling
# --------------------------------------------------------------------------------------


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


def make_classifier(config_e1: dict[str, Any], c_value: float, seed: int) -> LogisticRegression:
    """Exactly E1's classifier: the only hyperparameters are the frozen ones from configs/e1.json."""

    classifier = config_e1["classifier"]
    return LogisticRegression(
        C=float(c_value),
        solver=classifier["solver"],
        l1_ratio=float(classifier["l1_ratio"]),
        class_weight=classifier["class_weight"],
        max_iter=int(classifier["max_iter"]),
        random_state=int(seed),
    )


def build_features_refit(config_e1: dict[str, Any], train_texts: list[str]) -> FeatureUnion:
    """Development fallback: rebuild E1's Combined FeatureUnion from configs/e1.json on D1 Train.

    The official run loads the frozen FeatureUnion from models/e1/combined.joblib instead.
    Equivalence of the two is checked by comparing scores with results/e1/handoff.
    """

    specs = config_e1["vectorizers"]
    features = FeatureUnion([("word", make_vectorizer(specs["word"])), ("char", make_vectorizer(specs["char"]))])
    features.fit(list(train_texts))
    return features


def load_frozen_e1(project_root: Path, expected_sha256: str | None = None) -> tuple[FeatureUnion, LogisticRegression, dict[str, Any]]:
    """Load the frozen E1 Combined pipeline and return (fitted FeatureUnion, fitted LR, info)."""

    import joblib

    path = Path(project_root) / "models/e1/combined.joblib"
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"models/e1/combined.joblib hash {actual} does not match the expected {expected_sha256}")
    model = joblib.load(path)
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    return features, classifier, {"path": "models/e1/combined.joblib", "sha256": actual}


def feature_fingerprint(features: FeatureUnion) -> dict[str, Any]:
    """Hashes of each branch's vocabulary and IDF vector; proves the frozen features were not touched."""

    fingerprint: dict[str, Any] = {}
    total = 0
    for name, vectorizer in features.transformer_list:
        vocabulary = vectorizer.vocabulary_
        terms = [term for term, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
        fingerprint[name] = {
            "n_features": int(len(vocabulary)),
            "vocabulary_sha256": sha256_text("\n".join(terms)),
            "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype=np.float64).tobytes()).hexdigest(),
            "analyzer": vectorizer.analyzer,
            "ngram_range": list(vectorizer.ngram_range),
        }
        total += len(vocabulary)
    fingerprint["total_features"] = int(total)
    return fingerprint


def word_unigram_coefficients(features: FeatureUnion, classifier: LogisticRegression) -> dict[str, float]:
    """Map lower-cased word unigram -> logistic coefficient (word branch of the Combined model)."""

    names = features.get_feature_names_out()
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    mapping: dict[str, float] = {}
    prefix = "word__"
    for name, coefficient in zip(names, coefficients):
        if name.startswith(prefix):
            term = name[len(prefix) :]
            if " " not in term:
                mapping[term] = float(coefficient)
    return mapping


def assemble_pipeline(features: FeatureUnion, classifier: LogisticRegression) -> Pipeline:
    """A Pipeline of already-fitted steps; loads and scores exactly like models/e1/*.joblib."""

    return Pipeline([("features", features), ("classifier", classifier)])


def attack_scores(model: Pipeline, texts: pd.Series | list[str]) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    attack_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    return model.predict_proba(list(texts))[:, attack_index]


def scores_from_matrix(classifier: LogisticRegression, matrix: Any) -> np.ndarray:
    attack_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    return classifier.predict_proba(matrix)[:, attack_index]


# --------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------


def environment_info() -> dict[str, Any]:
    import joblib
    import matplotlib
    import scipy
    import sklearn

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import pyarrow

        info["pyarrow"] = pyarrow.__version__
    except ImportError:  # pragma: no cover
        info["pyarrow"] = None
    return info
