"""Validation harness for Name That LLM (18-class authorship attribution).

Everything downstream (baselines, ablations, model selection) must go through
these functions, otherwise its numbers do not track the leaderboard.

Public API
----------
LABELS                       canonical label order (18 strings, competition order)
prompt_folds(...)            GroupKFold-style split over prompt groups
sim_subset(...)              simulate the hidden test: 1-2 passages per prompt
evaluate(...)                balanced accuracy + per-class recall
board_like_score(...)        the metric that actually tracks the leaderboard

Run `python cv.py` for a self-test of the harness on official.parquet.
"""
from __future__ import annotations

import os
import sys

# The Anaconda pandas build ships numexpr/bottleneck compiled against numpy 1.x;
# under numpy 2.x they print C-level tracebacks on import. Marking them as
# unavailable makes pandas skip the optional acceleration without noise.
for _m in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_m, None)

import numpy as np
import pandas as pd

# The 18 labels in competition order. This is the single source of truth: the
# corpus builder and MANIFEST.json import it from here.
LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def prompt_folds(official: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """GroupKFold over prompt groups, greedily balanced by class counts.

    Returns an object array of length ``n_splits``; element k is an int array of
    row positions into ``official`` that belong to fold k. The union of all
    elements is exactly ``np.arange(len(official))`` and no prompt_group ever
    appears in two folds.

    The greedy step assigns the largest prompt blocks first (random tiebreak,
    so the split is deterministic for a given seed but not tied to row order),
    each time to the fold whose per-class load vector grows the least.
    """
    official = official.reset_index(drop=True)
    n = len(official)
    if n == 0:
        raise ValueError("empty official frame")

    prompts = official["prompt_group"].astype(str).to_numpy()
    y = official["model"].astype(str).to_numpy()
    labels = sorted(set(y.tolist()))

    uniq, inv = np.unique(prompts, return_inverse=True)
    g = len(uniq)
    if g < n_splits:
        raise ValueError(f"only {g} prompt groups for {n_splits} folds")

    # M[group, class] = how many rows of that class the group carries
    lab_ix = {lab: i for i, lab in enumerate(labels)}
    y_ix = np.array([lab_ix[v] for v in y], dtype=np.int64)
    M = np.zeros((g, len(labels)), dtype=np.float64)
    np.add.at(M, (inv, y_ix), 1.0)

    sizes = M.sum(axis=1)
    rng = np.random.default_rng(seed)
    tie = rng.random(g)
    # primary key: size descending; secondary: random tiebreak
    order = np.lexsort((tie, -sizes))

    loads = np.zeros((n_splits, len(labels)), dtype=np.float64)
    assign = np.zeros(g, dtype=np.int64)
    fidx = np.arange(n_splits)
    for gid in order:
        cand = loads + M[gid]  # (n_splits, C)
        worst = cand.max(axis=1)  # max per-class load after the move
        total = cand.sum(axis=1)
        pick = np.lexsort((fidx, total, worst))[0]
        assign[gid] = pick
        loads[pick] += M[gid]

    row_fold = assign[inv]
    folds = np.empty(n_splits, dtype=object)
    for k in range(n_splits):
        folds[k] = np.where(row_fold == k)[0]
    return folds


def prompt_fold_ids(official: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> list:
    """Same partition, but as lists of prompt_group strings (for caching)."""
    folds = prompt_folds(official, n_splits=n_splits, seed=seed)
    pg = official["prompt_group"].astype(str).to_numpy()
    return [sorted(set(pg[idx].tolist())) for idx in folds]


# --------------------------------------------------------------------------- #
# hidden-test simulation
# --------------------------------------------------------------------------- #
def sim_subset(idx: np.ndarray, official: pd.DataFrame, lo: int = 1, hi: int = 2,
               seed: int = 0) -> np.ndarray:
    """Keep 1-2 random passages per prompt_group from ``idx``.

    The hidden test gives a prompt only a handful of answers, which is why the
    full-18-answer OOF score does not track the leaderboard. Returns the kept
    positions (a subset of ``idx``, sorted, no duplicates).
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    official = official.reset_index(drop=True)
    prompts = official["prompt_group"].astype(str).to_numpy()
    uniq = np.unique(prompts[idx])  # sorted -> deterministic group order
    rng = np.random.default_rng(seed)

    keep = []
    for p in uniq:
        rows = idx[prompts[idx] == p]
        k = int(rng.integers(lo, hi + 1))
        k = min(k, rows.size)
        keep.append(rng.choice(rows, size=k, replace=False))
    return np.sort(np.concatenate(keep).astype(np.int64))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred) -> dict:
    """Macro recall (balanced accuracy) + per-class recall.

    Recall is averaged over classes *present in y_true*; classes with zero
    support are reported as 0.0 in ``per_class_recall`` but excluded from
    ``bacc`` (and counted in ``n_classes``). On the competition test set all 18
    classes are present, so this equals the official metric.
    """
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")

    per_class = {}
    for lab in LABELS:
        m = y_true == lab
        sup = int(m.sum())
        if sup == 0:
            per_class[lab] = 0.0
            continue
        per_class[lab] = float((y_pred[m] == lab).mean())

    present = [lab for lab in LABELS if int((y_true == lab).sum()) > 0]
    bacc = float(np.mean([per_class[lab] for lab in present])) if present else float("nan")
    acc = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    return {
        "bacc": bacc,
        "per_class_recall": per_class,
        "n": int(y_true.size),
        "n_classes": len(present),
        "accuracy": acc,
    }


def board_like_score(y_true, y_pred, prompt_group, lo: int = 1, hi: int = 2,
                     seed: int = 0, n_reps: int = 20, labels=None) -> dict:
    """Leaderboard-like score: 1-2 passages per prompt, averaged over resamples.

    For every repetition, each prompt group contributes a random ``lo..hi``
    passages; macro recall is computed on the union of those passages and then
    averaged over repetitions. This is the number to optimise -- full-18 OOF
    balanced accuracy correlates far worse with the leaderboard.

    ``y_true``/``y_pred`` are the out-of-fold predictions; ``prompt_group`` the
    matching group id per row.
    """
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    pg = np.asarray([str(v) for v in prompt_group])
    if not (len(y_true) == len(y_pred) == len(pg)):
        raise ValueError("y_true / y_pred / prompt_group must be the same length")

    lab_list = list(LABELS if labels is None else labels)
    uniq = np.unique(pg)
    groups = [np.where(pg == p)[0] for p in uniq]  # deterministic order
    rows_by_group = [np.arange(len(y_true))[pg == p] for p in uniq]

    rng = np.random.default_rng(seed)
    baccs, recalls = [], []
    ns = []
    for _ in range(n_reps):
        sel = []
        for rows in rows_by_group:
            k = min(int(rng.integers(lo, hi + 1)), rows.size)
            sel.append(rng.choice(rows, size=k, replace=False))
        sel = np.sort(np.concatenate(sel))
        yt, yp = y_true[sel], y_pred[sel]
        present = [lab for lab in lab_list if int((yt == lab).sum()) > 0]
        if not present:
            continue
        rec = {}
        for lab in lab_list:
            m = yt == lab
            rec[lab] = float((yp[m] == lab).mean()) if int(m.sum()) else 0.0
        baccs.append(float(np.mean([rec[lab] for lab in present])))
        recalls.append(rec)
        ns.append(int(sel.size))

    if not baccs:
        return {"bacc": float("nan"), "bacc_sd": float("nan"),
                "per_class_recall": {}, "n": 0, "n_reps": 0}
    per_class = {lab: float(np.mean([r[lab] for r in recalls])) for lab in lab_list}
    full = evaluate(y_true, y_pred)
    return {
        "bacc": float(np.mean(baccs)),
        "bacc_sd": float(np.std(baccs)),
        "bacc_min": float(np.min(baccs)),
        "bacc_max": float(np.max(baccs)),
        "per_class_recall": per_class,
        "n": int(np.mean(ns)),
        "n_reps": len(baccs),
        "full_bacc": full["bacc"],
    }


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def print_eval(tag: str, res: dict, show_classes: bool = True) -> None:
    print(f"[{tag}] bacc={res['bacc']:.4f}  n={res['n']}  "
          f"classes={res.get('n_classes', '?')}  acc={res.get('accuracy', float('nan')):.4f}")
    if show_classes:
        items = sorted(res["per_class_recall"].items(), key=lambda kv: kv[1])
        for lab, r in items:
            print(f"    {lab:<24s} {r:.3f}")


def _load_official(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(HERE, "official.parquet")
    return pd.read_parquet(path)


def _self_test() -> None:
    path = os.path.join(HERE, "official.parquet")
    if not os.path.exists(path):
        print(f"official.parquet not found at {path} -- run build_corpus.py first.")
        return
    official = _load_official(path)
    print(f"official rows={len(official)}  prompts={official['prompt_group'].nunique()}  "
          f"labels={official['model'].nunique()}")

    for seed in (0, 1):
        folds = prompt_folds(official, n_splits=5, seed=seed)
        sizes = [len(f) for f in folds]
        # no prompt group may straddle two folds
        pg = official["prompt_group"].to_numpy()
        overlap = sum(len(set(pg[folds[a]]) & set(pg[folds[b]]))
                      for a in range(5) for b in range(a + 1, 5))
        print(f"seed={seed} fold row sizes={sizes}  prompt-overlap across folds={overlap}")

    folds = prompt_folds(official, n_splits=5, seed=0)
    ct = pd.crosstab(official["model"], [np.zeros(len(official), int)])  # placeholder
    tab = pd.DataFrame({k: official.iloc[folds[k]]["model"].value_counts() for k in range(5)})
    tab = tab.reindex(LABELS).fillna(0).astype(int)
    print("\nper-fold class counts:")
    print(tab.to_string())

    sub = sim_subset(folds[0], official, lo=1, hi=2, seed=0)
    print(f"\nsim_subset(fold0): {len(folds[0])} rows / "
          f"{official.iloc[folds[0]]['prompt_group'].nunique()} prompts -> "
          f"{len(sub)} rows / {official.iloc[sub]['prompt_group'].nunique()} prompts")
    sub2 = sim_subset(folds[0], official, lo=1, hi=2, seed=0)
    print(f"deterministic across calls: {np.array_equal(sub, sub2)}")

    y = official["model"].to_numpy()
    perfect = evaluate(y, y)
    print(f"\nevaluate(perfect) bacc={perfect['bacc']:.4f} n={perfect['n']}")
    rng = np.random.default_rng(0)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    rand = evaluate(y, shuffled)
    print(f"evaluate(shuffled) bacc={rand['bacc']:.4f} (expect ~0.055 = 1/18)")
    bl = board_like_score(y, shuffled, official["prompt_group"].to_numpy(), seed=0, n_reps=10)
    print(f"board_like(shuffled) bacc={bl['bacc']:.4f} sd={bl['bacc_sd']:.4f} "
          f"n/rep={bl['n']}  (expect wider spread than full)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _self_test()
