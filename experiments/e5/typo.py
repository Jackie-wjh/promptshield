"""Deterministic, controlled typo generation for E5.

Threat model: accidental single-character typos in alphabetic words of at least four letters.

* A text is split into whitespace-delimited tokens; whitespace and punctuation are preserved
  exactly. A token is *eligible* when its letters form ONE contiguous run of at least four
  ASCII letters with only non-letters before and after it (``^[^A-Za-z]*([A-Za-z]{4,})[^A-Za-z]*$``).
  Anything with a non-letter between letters is therefore never touched: contractions
  (don't), hyphenated and snake_case words, URLs, e-mail addresses, most code and leetspeak.
  Leading or trailing digits or brackets do not protect a token: in ``user123`` the run ``user``
  is eligible and the digits are kept as its suffix.
* Four operators, drawn with equal probability, act on interior letters only (first and last
  letter stay fixed): adjacent swap, delete one letter, duplicate one letter, QWERTY-neighbour
  substitution (case preserved).
* Word-selection policies: ``random1`` (one uniformly random eligible word), ``random_rate``
  (``max(1, ceil(rate * n_eligible))`` distinct words) and ``targeted_lexical`` (the listed word
  with the largest |log-odds| in a lexicon built from D1 Train labels only: attack texts receive
  the typo in an attack-indicative word, benign texts in a benign-indicative word; random
  fallback when no listed word is present). ``dup`` makes an exact copy (count-matched control).
* Randomness is a per-text ``numpy.random.default_rng([seed, copy_index, hash(sample_id)])``
  stream, so every augmented row can be regenerated on its own and a retry never shifts another
  row.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiments.e5.common import normalized_sha256, sha256_text

TOKEN_RE = re.compile(r"\S+")
CORE_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]{4,})([^A-Za-z]*)$")

OPERATORS: tuple[str, ...] = ("swap", "delete", "duplicate", "qwerty")
TYPO_POLICIES: tuple[str, ...] = ("random1", "random_rate", "targeted_lexical")
STATUSES: tuple[str, ...] = ("eligible_valid", "ineligible_no_word", "invalid_unchanged", "invalid_collision", "duplicate_control")

QWERTY_NEIGHBOURS: dict[str, str] = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb",
    "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

OPS_COLUMNS = [
    "role", "sample_id", "parent_id", "condition", "copy_index", "seed", "policy", "transform_id", "op_index",
    "operator", "word_index", "char_index", "original_word", "perturbed_word", "perturbed_in_word_vocab",
]


@dataclass(frozen=True)
class EligibleToken:
    word_index: int  # position among whitespace tokens
    start: int  # character offset of the token in the text
    end: int
    prefix: str  # leading non-letters (quotes, brackets ...)
    core: str  # the letters that may be perturbed
    suffix: str  # trailing non-letters (commas, periods ...)


@dataclass(frozen=True)
class TypoOp:
    operator: str
    word_index: int
    char_index: int  # index inside the core where the operator acted
    original_word: str
    perturbed_word: str


def eligible_tokens(text: str) -> list[EligibleToken]:
    tokens: list[EligibleToken] = []
    for word_index, match in enumerate(TOKEN_RE.finditer(text)):
        core_match = CORE_RE.match(match.group(0))
        if core_match is None:
            continue
        prefix, core, suffix = core_match.groups()
        tokens.append(EligibleToken(word_index, match.start(), match.end(), prefix, core, suffix))
    return tokens


def apply_operator(core: str, operator: str, rng: np.random.Generator) -> tuple[str, int] | None:
    """Apply one operator to the interior of ``core``; return (new_core, char_index) or None if infeasible."""

    n = len(core)
    if n < 4:
        return None
    if operator == "swap":
        candidates = [i for i in range(1, n - 2) if core[i] != core[i + 1]]
        if not candidates:
            return None
        i = int(rng.choice(candidates))
        return core[:i] + core[i + 1] + core[i] + core[i + 2 :], i
    if operator == "delete":
        i = int(rng.integers(1, n - 1))
        return core[:i] + core[i + 1 :], i
    if operator == "duplicate":
        i = int(rng.integers(1, n - 1))
        return core[:i] + core[i] + core[i:], i
    if operator == "qwerty":
        candidates = [i for i in range(1, n - 1) if core[i].lower() in QWERTY_NEIGHBOURS]
        if not candidates:
            return None
        i = int(rng.choice(candidates))
        original = core[i]
        neighbours = QWERTY_NEIGHBOURS[original.lower()]
        replacement = neighbours[int(rng.integers(0, len(neighbours)))]
        if original.isupper():
            replacement = replacement.upper()
        return core[:i] + replacement + core[i + 1 :], i
    raise ValueError(f"Unknown operator {operator!r}")


# --------------------------------------------------------------------------------------
# lexicon for the targeted policy (built from D1 Train labels only, before any model is trained)
# --------------------------------------------------------------------------------------


def build_lexicon(
    texts: list[str],
    labels: list[int] | np.ndarray,
    df_min: int,
    top_k_candidates: list[int],
    coverage_target: float,
) -> tuple[pd.DataFrame, dict[str, tuple[str, float]], dict[str, Any]]:
    """Attack- and benign-indicative eligible words by smoothed log-odds of document frequency.

    log_odds(w) = log((df_attack + 1) / (n_attack + 2)) - log((df_benign + 1) / (n_benign + 2)).
    The top k words are 'attack', the bottom k are 'benign'. k is the smallest candidate whose
    top-k attack words occur in at least ``coverage_target`` of the eligible attack texts AND whose
    top-k benign words occur in at least ``coverage_target`` of the eligible benign texts (a D1
    Train statistic only); otherwise the largest candidate, which is recorded. Model-agnostic by
    construction: no classifier coefficient is involved.
    Returns (lexicon frame, word -> (class, log_odds), info with the chosen k and coverage table).
    """

    df_attack: Counter[str] = Counter()
    df_benign: Counter[str] = Counter()
    labels_array = np.asarray(labels, dtype=int)
    word_sets: list[set[str]] = []
    for text, label in zip(texts, labels_array):
        words = {token.core.lower() for token in eligible_tokens(text)}
        word_sets.append(words)
        (df_attack if label == 1 else df_benign).update(words)
    n_attack = int((labels_array == 1).sum())
    n_benign = int((labels_array == 0).sum())
    rows = []
    for word in set(df_attack) | set(df_benign):
        da, db = df_attack[word], df_benign[word]
        if da + db < df_min:
            continue
        log_odds = math.log((da + 1) / (n_attack + 2)) - math.log((db + 1) / (n_benign + 2))
        rows.append({"word": word, "df_attack": da, "df_benign": db, "log_odds": log_odds})
    frame = pd.DataFrame(rows).sort_values(["log_odds", "word"], ascending=[False, True]).reset_index(drop=True)
    eligible_attack = [words for words, label in zip(word_sets, labels_array) if label == 1 and words]
    eligible_benign = [words for words, label in zip(word_sets, labels_array) if label == 0 and words]
    coverage_rows = []
    chosen_k = int(top_k_candidates[-1])
    target_met = False
    for k in sorted(int(c) for c in top_k_candidates):
        attack_words = set(frame.head(k)["word"])
        benign_words = set(frame.tail(k)["word"])
        attack_coverage = float(np.mean([bool(words & attack_words) for words in eligible_attack])) if eligible_attack else 0.0
        benign_coverage = float(np.mean([bool(words & benign_words) for words in eligible_benign])) if eligible_benign else 0.0
        coverage_rows.append({"k": k, "attack_coverage": attack_coverage, "benign_coverage": benign_coverage})
        if not target_met and attack_coverage >= coverage_target and benign_coverage >= coverage_target:
            chosen_k, target_met = k, True
    k_capped = chosen_k > len(frame) // 2
    chosen_k = min(chosen_k, len(frame) // 2)  # the two class lists must stay disjoint
    attack = frame.head(chosen_k).assign(lexicon_class="attack")
    benign = frame.tail(chosen_k).sort_values(["log_odds", "word"], ascending=[True, True]).assign(lexicon_class="benign")
    lexicon_frame = pd.concat([attack, benign], ignore_index=True)
    lexicon_frame.insert(0, "rank", np.concatenate([np.arange(1, len(attack) + 1), np.arange(1, len(benign) + 1)]))
    mapping = {row["word"]: (row["lexicon_class"], float(row["log_odds"])) for _, row in lexicon_frame.iterrows()}
    info = {"chosen_k": int(chosen_k), "k_capped_to_half_of_candidates": bool(k_capped), "coverage_target": float(coverage_target), "coverage_target_met": bool(target_met), "candidate_words": int(len(frame)), "coverage_by_k": coverage_rows, "n_attack_texts_eligible": len(eligible_attack), "n_benign_texts_eligible": len(eligible_benign)}
    return lexicon_frame, mapping, info


# --------------------------------------------------------------------------------------
# word selection and perturbation
# --------------------------------------------------------------------------------------


def choose_tokens(
    tokens: list[EligibleToken],
    policy: str,
    rate: float,
    rng: np.random.Generator,
    lexicon: dict[str, tuple[str, float]] | None,
    label: int | None,
) -> tuple[list[EligibleToken], dict[str, Any]]:
    """Pick which eligible tokens to perturb. Returns (tokens, info) with fallback/target details."""

    info: dict[str, Any] = {"fallback_random": False, "target_word": None, "target_occurrences": 0}
    if not tokens:
        return [], info
    if policy == "targeted_lexical":
        if lexicon is None or label is None:
            raise ValueError("targeted_lexical needs the lexicon and the gold label")
        wanted = "attack" if int(label) == 1 else "benign"
        listed = [(abs(lexicon[t.core.lower()][1]), t) for t in tokens if t.core.lower() in lexicon and lexicon[t.core.lower()][0] == wanted]
        if listed:
            best_weight = max(weight for weight, _ in listed)
            best_word = min((t for weight, t in listed if weight == best_weight), key=lambda t: t.word_index)
            info["target_word"] = best_word.core.lower()
            info["target_occurrences"] = sum(1 for t in tokens if t.core.lower() == best_word.core.lower())
            return [best_word], info
        info["fallback_random"] = True
        policy = "random1"
    if policy == "random1":
        count = 1
    elif policy == "random_rate":
        count = max(1, int(math.ceil(rate * len(tokens))))
    else:
        raise ValueError(f"Unknown policy {policy!r}")
    count = min(count, len(tokens))
    chosen_positions = rng.choice(len(tokens), size=count, replace=False)
    return [tokens[int(p)] for p in sorted(chosen_positions)], info


def perturb_text(
    text: str,
    policy: str,
    rate: float,
    rng: np.random.Generator,
    lexicon: dict[str, tuple[str, float]] | None = None,
    label: int | None = None,
    operators: tuple[str, ...] = OPERATORS,
) -> tuple[str, list[TypoOp], dict[str, Any]] | None:
    """Return (perturbed text, operations, info) or None when the text has no eligible word."""

    tokens = eligible_tokens(text)
    if not tokens:
        return None
    chosen, info = choose_tokens(tokens, policy, rate, rng, lexicon, label)
    operations: list[TypoOp] = []
    pieces: list[str] = []
    cursor = 0
    for token in chosen:
        order = [operators[int(k)] for k in rng.permutation(len(operators))]
        result = None
        operator = order[0]
        for operator in order:
            result = apply_operator(token.core, operator, rng)
            if result is not None:
                break
        if result is None:  # cannot happen for cores of 4+ letters, kept for safety
            continue
        new_core, char_index = result
        operations.append(TypoOp(operator, token.word_index, char_index, token.core, new_core))
        pieces.append(text[cursor : token.start])
        pieces.append(token.prefix + new_core + token.suffix)
        cursor = token.end
    pieces.append(text[cursor:])
    return "".join(pieces), operations, info


def row_rng(seed: int, copy_index: int, sample_id: str) -> np.random.Generator:
    """Independent stream per (seed, copy, sample_id); the id is hashed so row order is irrelevant."""

    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return np.random.default_rng([int(seed), int(copy_index), int.from_bytes(digest[:8], "big")])


def augment_frame(
    frame: pd.DataFrame,
    *,
    role: str,
    dataset: str,
    split: str,
    condition: str,
    policy: str,
    rate: float,
    seed: int,
    copy_index: int,
    transform_id: str,
    forbidden_hashes: set[str],
    lexicon: dict[str, tuple[str, float]] | None = None,
    word_vocabulary: dict[str, int] | None = None,
    max_attempts: int = 10,
    id_suffix: str | None = None,
    seen: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate one perturbed (or duplicated) copy of every eligible row of ``frame``.

    ``forbidden_hashes`` are the normalized hashes a perturbed text must not collide with
    (pre-freeze: D1 Train + D1 Validation originals; post-freeze evaluation: all six splits).
    ``seen`` holds the hashes already generated for the same condition; pass one set through
    every copy of a multi-copy condition so that copies never contain the same perturbed text.
    A text that still collides after ``max_attempts`` draws from its own RNG stream is recorded
    with status ``invalid_collision`` and never used. Policy ``dup`` copies eligible texts
    unchanged (count-matched control) and skips the collision rule on purpose.
    Returns (manifest rows, operations, stats). Only rows with ``valid == True`` may be used.
    """

    suffix = id_suffix if id_suffix is not None else f"aug:{condition}:{copy_index}"
    if seen is None:
        seen = set()
    rows: list[dict[str, Any]] = []
    ops: list[dict[str, Any]] = []
    stats = {"n_input": int(len(frame)), "n_valid": 0, "n_no_eligible_word": 0, "n_invalid_unchanged": 0, "n_invalid_collision": 0, "n_fallback_random": 0, "n_attempts_total": 0, "n_ops_total": 0}
    sample_ids = frame["sample_id"].astype(str).tolist()
    texts = frame["text"].tolist()
    labels = frame["label"].tolist() if "label" in frame.columns else [0] * len(frame)
    vocabulary = word_vocabulary or {}
    for position, (sample_id, text, label) in enumerate(zip(sample_ids, texts, labels)):
        rng = row_rng(seed, copy_index, sample_id)
        base = {
            "role": role, "dataset": dataset, "split": split, "source_split": f"{dataset}_{split}",
            "sample_id": f"{sample_id}:{suffix}" if role == "train" else sample_id,
            "parent_id": sample_id, "label": int(label), "condition": condition, "copy_index": int(copy_index),
            "seed": int(seed), "policy": policy, "rate": float(rate), "transform_id": transform_id, "row_position": int(position),
        }
        empty = {"n_ops": 0, "operators": "", "ops_json": "[]", "target_word": None, "target_occurrences": 0, "fallback_random": False,
                 "text": None, "text_sha256": None, "normalized_sha256": None, "attempts": 0, "retry_count": 0, "collides_with_train": None}
        tokens = eligible_tokens(text)
        if not tokens:
            stats["n_no_eligible_word"] += 1
            rows.append({**base, **empty, "status": "ineligible_no_word", "valid": False, "invalid_reason": "no_eligible_word"})
            continue
        if policy == "dup":
            stats["n_valid"] += 1
            rows.append({**base, **empty, "text": text, "text_sha256": sha256_text(text), "normalized_sha256": normalized_sha256(text),
                         "status": "duplicate_control", "valid": True, "invalid_reason": None})
            continue
        parent_norm = normalized_sha256(text)
        accepted = False
        attempts = 0
        last_reason = "invalid_unchanged"
        perturbed, operations, info = text, [], {"fallback_random": False, "target_word": None, "target_occurrences": 0}
        while attempts < max_attempts:
            attempts += 1
            stats["n_attempts_total"] += 1
            result = perturb_text(text, policy, rate, rng, lexicon, int(label))
            if result is None:  # cannot happen: eligibility was checked above
                break
            perturbed, operations, info = result
            norm = normalized_sha256(perturbed)
            if perturbed == text or norm == parent_norm:
                last_reason = "invalid_unchanged"
                continue
            if norm in forbidden_hashes or norm in seen:
                last_reason = "invalid_collision"
                continue
            accepted = True
            break
        record = {**base, "n_ops": len(operations), "operators": "+".join(op.operator for op in operations),
                  "ops_json": json.dumps([asdict(op) for op in operations], ensure_ascii=False), "target_word": info["target_word"],
                  "target_occurrences": int(info["target_occurrences"]), "fallback_random": bool(info["fallback_random"]), "text": perturbed,
                  "text_sha256": sha256_text(perturbed), "normalized_sha256": normalized_sha256(perturbed), "attempts": attempts,
                  "retry_count": attempts - 1, "collides_with_train": None}
        if not accepted:
            stats["n_invalid_unchanged" if last_reason == "invalid_unchanged" else "n_invalid_collision"] += 1
            rows.append({**record, "status": last_reason, "valid": False, "invalid_reason": last_reason})
            continue
        seen.add(record["normalized_sha256"])
        stats["n_valid"] += 1
        stats["n_ops_total"] += len(operations)
        if info["fallback_random"]:
            stats["n_fallback_random"] += 1
        rows.append({**record, "status": "eligible_valid", "valid": True, "invalid_reason": None})
        for op_index, op in enumerate(operations):
            ops.append({"role": role, "sample_id": base["sample_id"], "parent_id": sample_id, "condition": condition, "copy_index": int(copy_index),
                        "seed": int(seed), "policy": policy, "transform_id": transform_id, "op_index": op_index, **asdict(op),
                        "perturbed_in_word_vocab": bool(op.perturbed_word.lower() in vocabulary)})
    rows_frame = pd.DataFrame(rows)
    ops_frame = pd.DataFrame(ops, columns=OPS_COLUMNS)
    return rows_frame, ops_frame, stats
