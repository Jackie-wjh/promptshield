"""Add reproducible D4 scores to the frozen E2 BiLSTM handoff.

This script does not retrain E2. It loads the selected checkpoint and its saved
training vocabulary, verifies that rescoring D1 Validation reproduces the
existing handoff, scores the frozen benign-only D4 set, and atomically updates
``results/e2/handoff/bilstm_scores.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2.dataset import PromptDataset, TextVocabulary, pad_collate_fn
from experiments.e2.model import BiLSTMClassifier


SCHEMA = [
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


def vocabulary_from_checkpoint(mapping: dict[str, int]) -> TextVocabulary:
    vocab = TextVocabulary()
    vocab.word2idx = {str(word): int(index) for word, index in mapping.items()}
    vocab.idx2word = {index: word for word, index in vocab.word2idx.items()}
    return vocab


def score_frame(
    model: BiLSTMClassifier,
    vocab: TextVocabulary,
    frame: pd.DataFrame,
    *,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    dataset = PromptDataset(frame, vocab, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )
    sample_ids: list[str] = []
    gold: list[int] = []
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["lengths"])
            scores.extend(torch.sigmoid(logits).cpu().numpy().astype(float).tolist())
            gold.extend(batch["labels"].cpu().numpy().astype(int).tolist())
            sample_ids.extend(str(value) for value in batch["sample_ids"])
    return pd.DataFrame(
        {"sample_id": sample_ids, "gold": gold, "score_attack": scores}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/e2/bilstm_random_best.pt",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=ROOT / "results/e2/handoff/bilstm_scores.parquet",
    )
    parser.add_argument(
        "--d4",
        type=Path,
        default=ROOT / "data/processed/d4_test.parquet",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data/processed/d1_validation.parquet",
    )
    parser.add_argument("--verify-atol", type=float, default=1e-6)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    vocab = vocabulary_from_checkpoint(checkpoint["vocab"])
    model_config = config["model"]
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_size=int(model_config["hidden_size"]),
        dropout=float(model_config["dropout"]),
        trainable_embedding=bool(model_config["trainable_embedding"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    max_length = int(config["training"]["max_length"])
    batch_size = int(config["training"]["batch_size"])
    existing = pd.read_parquet(args.handoff)

    validation_scores = score_frame(
        model,
        vocab,
        pd.read_parquet(args.validation),
        max_length=max_length,
        batch_size=batch_size,
    )
    reference = existing[
        existing["dataset"].eq("d1") & existing["split"].eq("validation")
    ][["sample_id", "score_attack"]].rename(columns={"score_attack": "reference"})
    checked = validation_scores.merge(reference, on="sample_id", validate="one_to_one")
    if len(checked) != len(validation_scores):
        raise RuntimeError("D1 Validation IDs do not match the existing E2 handoff")
    max_abs_diff = float(
        np.max(np.abs(checked["score_attack"].to_numpy() - checked["reference"].to_numpy()))
    )
    if max_abs_diff > args.verify_atol:
        raise RuntimeError(
            f"Checkpoint does not reproduce the E2 handoff: max abs diff {max_abs_diff:.3g}"
        )

    d4_scores = score_frame(
        model,
        vocab,
        pd.read_parquet(args.d4),
        max_length=max_length,
        batch_size=batch_size,
    )
    checkpoint_rev = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    seed = int(config["seed"])
    d4_scores.insert(0, "split", "test")
    d4_scores.insert(0, "dataset", "d4")
    d4_scores["model_id"] = "bilstm_random"
    d4_scores["checkpoint_rev"] = checkpoint_rev
    d4_scores["seed"] = seed
    d4_scores["transform_id"] = "clean"
    d4_scores["parent_id"] = pd.NA

    merged = existing[~existing["dataset"].eq("d4")].copy()
    merged["checkpoint_rev"] = checkpoint_rev
    merged["transform_id"] = "clean"
    merged["parent_id"] = pd.NA
    merged = pd.concat([merged[SCHEMA], d4_scores[SCHEMA]], ignore_index=True)
    temporary = args.handoff.with_suffix(args.handoff.suffix + ".tmp")
    merged.to_parquet(temporary, index=False)
    temporary.replace(args.handoff)

    metadata_path = args.handoff.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = checkpoint_rev
    metadata["d4_rows"] = int(len(d4_scores))
    metadata["d1_validation_rescore_max_abs_diff"] = max_abs_diff
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "passed",
                "d4_rows_added": int(len(d4_scores)),
                "handoff_rows": int(len(merged)),
                "d1_validation_rescore_max_abs_diff": max_abs_diff,
                "checkpoint_sha256": checkpoint_rev,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
