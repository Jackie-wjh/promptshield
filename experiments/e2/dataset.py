"""Vocabulary, Data loading, and Pretrained Vector handling for E2.

Rules enforced:
1. Vocabulary is strictly built ONLY from D1 Train text.
2. Special tokens: <PAD> = 0, <UNK> = 1.
3. Calculates GloVe coverage rate and Out-Of-Vocabulary (OOV) rate.
4. Tokenization uses clean whitespace/word-boundary splitting with lowercase.
"""

import os
import re
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def tokenize(text: str) -> List[str]:
    """Simple robust regex tokenizer converting to lowercase words."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())


class TextVocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build_from_texts(self, texts: List[str]) -> "TextVocabulary":
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))

        for word, count in counter.most_common():
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, text: str, max_length: int = 256) -> Tuple[List[int], int]:
        tokens = tokenize(text)
        if len(tokens) == 0:
            tokens = [PAD_TOKEN]

        ids = [self.word2idx.get(w, UNK_IDX) for w in tokens[:max_length]]
        true_len = max(1, len(ids))
        return ids, true_len


class PromptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: TextVocabulary, max_length: int = 256):
        self.sample_ids = df["sample_id"].tolist()
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist() if "label" in df.columns else [0] * len(df)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids, true_len = self.vocab.encode(self.texts[idx], max_length=self.max_length)
        return {
            "sample_id": self.sample_ids[idx],
            "input_ids": ids,
            "length": true_len,
            "label": self.labels[idx],
        }


def pad_collate_fn(batch: List[dict]):
    """Dynamic padding up to the maximum true length in the current mini-batch."""
    sample_ids = [item["sample_id"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    max_len = max(item["length"] for item in batch)
    padded_ids = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        padded_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return {
        "sample_ids": sample_ids,
        "input_ids": padded_ids,
        "lengths": lengths,
        "labels": labels,
    }


def load_or_download_glove(
    glove_path: str = "data/pretrained/glove.6B.100d.txt",
    dim: int = 100,
) -> Dict[str, np.ndarray]:
    """Loads GloVe embeddings from file.

    If not found, attempts to download the official zip file into data/pretrained.
    """
    if not os.path.exists(glove_path):
        os.makedirs(os.path.dirname(glove_path), exist_ok=True)
        zip_path = os.path.join(os.path.dirname(glove_path), "glove.6B.zip")
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        print(f"[GloVe] File not found at {glove_path}.")
        print(f"[GloVe] Attempting download from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract(f"glove.6B.{dim}d.txt", os.path.dirname(glove_path))
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print("[GloVe] Download and extraction complete.")
        except Exception as e:
            print(f"[GloVe] Warning: Auto-download failed: {e}")
            print(f"[GloVe] Please place glove.6B.{dim}d.txt manually in {os.path.dirname(glove_path)} if needed.")
            return {}

    embeddings = {}
    print(f"[GloVe] Loading vectors from {glove_path}...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype="float32")
            if len(vector) == dim:
                embeddings[word] = vector
    print(f"[GloVe] Successfully loaded {len(embeddings)} word vectors.")
    return embeddings


def build_embedding_matrix(
    vocab: TextVocabulary,
    glove_vectors: Dict[str, np.ndarray],
    dim: int = 100,
) -> Tuple[torch.Tensor, float, float]:
    """Constructs the embedding weight tensor for vocab.

    Returns:
        matrix: torch.Tensor of shape (vocab_size, dim)
        coverage_rate: float (percentage of vocab found in GloVe)
        oov_rate: float (percentage of vocab Out-Of-Vocabulary)
    """
    matrix = np.random.normal(scale=0.05, size=(len(vocab), dim)).astype("float32")
    matrix[PAD_IDX] = np.zeros(dim, dtype="float32")

    matched_count = 0
    total_tokens = len(vocab) - 1  # exclude PAD

    for word, idx in vocab.word2idx.items():
        if idx == PAD_IDX:
            continue
        if word in glove_vectors:
            matrix[idx] = glove_vectors[word]
            matched_count += 1

    coverage_rate = (matched_count / total_tokens) if total_tokens > 0 else 0.0
    oov_rate = 1.0 - coverage_rate
    return torch.tensor(matrix, dtype=torch.float32), coverage_rate, oov_rate
