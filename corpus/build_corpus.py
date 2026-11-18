"""Build the training corpus for Name That LLM (18-class authorship attribution).

Produces, under corpus/ :
    union.parquet          text, model, src, prompt_group, n_words
    union_capped.parquet   same schema, per-label cap applied
    official.parquet       train.csv as text, model, prompt_group="tr:"+prompt_id
    test3.parquet          the 3 visible test rows
    MANIFEST.json          sha256 / row counts / chosen cap / label order

Source priority follows pool_report.md section 3.2:
    1  crtproduction/name-that-llm-open-pool            ntl_open_pool.parquet
    2  olegzholobov/name-that-llm-response-corpus       external_all_wide.parquet (MELTED)
    3  prvsiyan/prvsiyans-extended-name-that-llm-dataset passages_clean.csv
    4  nuridudak/ntl-130k-clean-passages                ntl_merged_external.parquet

Deliberately excluded (pool_report section 3.6): quarantine_luna_*, external_short_wide,
melvindenishl (byte-identical v13), morginalium, samarth1006, riccapucho (as bulk),
gemini37_unverified (unverified routing), seyitkaangunes, zbigniewkirchner,
tadeuhayashida, pjmathematician (prompts only).

Idempotent: every source extraction is cached under corpus/.cache/ as parquet and
reused on the next run. Delete .cache/ (or pass --refresh) to force a rebuild.
"""
from __future__ import annotations

import sys

# The Anaconda pandas build here ships numexpr/bottleneck compiled against numpy 1.x,
# which spams the console with C-level tracebacks under numpy 2.x. Marking them as
# unavailable makes pandas skip the optional acceleration cleanly.
for _m in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_m, None)

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
ROOT = r"C:\Users\eleve\Desktop\Kaggle\nllm"
POOLS = os.path.join(ROOT, "pools")
DATA = os.path.join(ROOT, "data")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")

sys.path.insert(0, HERE)
from cv import LABELS  # noqa: E402  single source of truth for label order

# src code, priority (lower = higher priority), human label
SRC = {
    "op": (1, "crtproduction/name-that-llm-open-pool"),
    "oz": (2, "olegzholobov/name-that-llm-response-corpus"),
    "prv": (3, "prvsiyan/prvsiyans-extended-name-that-llm-dataset"),
    "nur": (4, "nuridudak/ntl-130k-clean-passages"),
}
PRIO = {k: v[0] for k, v in SRC.items()}

WORD_LO, WORD_HI = 40, 250
CAP_RATIO = 1.6
SEED = 0


def hr(title: str = "") -> None:
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)


def step(done: str, n: int, extra: str = "") -> None:
    """Report the row count at one build step."""
    print(f"  {done:<58s} {n:>8,d} rows{('  ' + extra) if extra else ''}")


def normalize_ws(s: pd.Series) -> pd.Series:
    """Collapse whitespace runs to one space, strip both ends."""
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


# --------------------------------------------------------------------------- #
# source loaders -- each returns a long frame with text / model / prompt_group
# --------------------------------------------------------------------------- #
def load_op(refresh: bool) -> pd.DataFrame:
    path = os.path.join(POOLS, "crtproduction__name-that-llm-open-pool",
                        "ntl_open_pool.parquet")
    df = pd.read_parquet(path, columns=["text", "model", "prompt_group",
                                        "flag_train_prompt_regen", "flag_kb_proxy"])
    n0 = len(df)
    # Cheap insurance: prvsiyan-style rows written against reconstructed train prompts.
    df = df[~df["flag_train_prompt_regen"].fillna(False).astype(bool)]
    kb = int(df["flag_kb_proxy"].fillna(False).astype(bool).sum())
    step("op   ntl_open_pool.parquet", len(df),
         f"(dropped {n0 - len(df):,} flag_train_prompt_regen; {kb:,} rows carry flag_kb_proxy)")
    return df[["text", "model", "prompt_group"]].assign(src="op")


def load_oz(refresh: bool) -> pd.DataFrame:
    path = os.path.join(POOLS, "olegzholobov__name-that-llm-response-corpus",
                        "external_all_wide.parquet")
    cache = os.path.join(CACHE, "oz_long.parquet")
    if os.path.exists(cache) and not refresh:
        df = pd.read_parquet(cache)
        step("oz   external_all_wide.parquet (cached melt)", len(df),
             f"from {os.path.basename(cache)}")
        return df
    wide = pd.read_parquet(path)
    model_cols = [c for c in wide.columns if c not in ("id", "prompt")]
    missing = [lab for lab in LABELS if lab not in model_cols]
    if missing:
        raise RuntimeError(f"oz wide table is missing label columns: {missing}")
    n_prompts = wide["prompt"].nunique()
    # One row per prompt x model cell; the label is the COLUMN NAME, not a value.
    long = (wide.melt(id_vars=["id", "prompt"], value_vars=model_cols,
                      var_name="model", value_name="text")
                 .dropna(subset=["text"]))
    long = long[long["text"].astype(str).str.strip() != ""]
    # Group key: the prompt string itself, hashed. `id` varies per response row
    # (the same prompt appears under several ids), so it is NOT a group key.
    long["prompt_group"] = ("oz" + pd.util.hash_pandas_object(
        long["prompt"].astype(str), index=False).astype("uint64").astype(str))
    out = long[["text", "model", "prompt_group"]].assign(src="oz")
    step("oz   external_all_wide.parquet (melted)", len(out),
         f"from {len(wide):,} wide rows / {n_prompts:,} distinct prompts")
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def load_prv(refresh: bool) -> pd.DataFrame:
    path = os.path.join(POOLS, "prvsiyan__prvsiyans-extended-name-that-llm-dataset",
                        "passages_clean.csv")
    df = pd.read_csv(path, usecols=["prompt_id", "model", "text", "split"])
    n0 = len(df)
    # `train_regen` passages answer the reconstructed competition prompts.
    df = df[df["split"].astype(str) != "train_regen"]
    step("prv  passages_clean.csv", len(df),
         f"(dropped {n0 - len(df):,} split=train_regen)")
    return df[["text", "model", "prompt_id"]].rename(
        columns={"prompt_id": "prompt_group"}).assign(src="prv")


def load_nur(refresh: bool) -> pd.DataFrame:
    path = os.path.join(POOLS, "nuridudak__ntl-130k-clean-passages",
                        "ntl_merged_external.parquet")
    df = pd.read_parquet(path, columns=["text", "model", "prompt_group",
                                        "style_ok", "label_conflict", "neardup_drop"])
    n0 = len(df)
    keep = (df["style_ok"].fillna(False).astype(bool)
            & ~df["label_conflict"].fillna(False).astype(bool)
            & ~df["neardup_drop"].fillna(False).astype(bool))
    df = df[keep]
    step("nur  ntl_merged_external.parquet", len(df),
         f"(dropped {n0 - len(df):,} style_ok/label_conflict/neardup)")
    return df[["text", "model", "prompt_group"]].assign(src="nur")


LOADERS = {"op": load_op, "oz": load_oz, "prv": load_prv, "nur": load_nur}


# --------------------------------------------------------------------------- #
# official / test
# --------------------------------------------------------------------------- #
def build_official() -> pd.DataFrame:
    t = pd.read_csv(os.path.join(DATA, "train.csv"))
    out = pd.DataFrame({
        "text": t["text"].astype(str),
        "model": t["model"].astype(str),
        "prompt_group": "tr:" + t["prompt_id"].astype(str),
    })
    step("official  data/train.csv", len(out),
         f"({out['prompt_group'].nunique()} prompts, {out['model'].nunique()} labels)")
    return out


def build_test3() -> pd.DataFrame:
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    out = pd.DataFrame({
        "id": te["id"].astype(str),
        "text": te["text"].astype(str),
        "prompt_group": "te:" + te["id"].astype(str),
        "n_words": te["text"].astype(str).str.split().str.len().astype("int64"),
    })
    step("test3     data/test.csv", len(out), "(no labels - the visible test rows)")
    return out


# --------------------------------------------------------------------------- #
# union build
# --------------------------------------------------------------------------- #
def build_union(refresh: bool, drop_kb: bool = False) -> pd.DataFrame:
    hr("1. source ingestion (priority order, highest first)")
    frames = [LOADERS[k](refresh) for k, _ in sorted(PRIO.items(), key=lambda kv: kv[1])]
    raw = pd.concat(frames, ignore_index=True)
    step("RAW union (all sources concatenated)", len(raw))
    print(f"    per-source raw rows: " +
          ", ".join(f"{k}={int((raw['src'] == k).sum()):,}" for k in SRC))

    hr("2. cleaning, dedupe, train-hash removal, length filter")
    raw["text"] = raw["text"].astype(str)
    raw = raw[raw["text"].str.strip() != ""]
    step("drop empty / whitespace-only text", len(raw))

    # Unknown labels would silently add a 19th class downstream.
    unknown = ~raw["model"].isin(LABELS)
    if unknown.any():
        print(f"    WARNING: dropping {int(unknown.sum()):,} rows with labels outside "
              f"the 18: {sorted(raw.loc[unknown, 'model'].unique())[:10]}")
        raw = raw[~unknown]
        step("drop rows with an unrecognised label", len(raw))

    # Length is measured on whitespace tokens, the same way train.csv was profiled.
    raw["key"] = normalize_ws(raw["text"]).str.lower()

    # Namespace prompt_group so ids from different pools can never collide
    # ("op:x" / "oz:x" / "prv:x" / "nur:x").
    raw["prompt_group"] = raw["src"].astype(str) + ":" + raw["prompt_group"].astype(str)

    # Stable sort by priority, then keep="first" => the highest-priority pool keeps
    # the row. Never dedupe on prompt_group: different models on one prompt is signal.
    raw["_prio"] = raw["src"].map(PRIO).astype("int64")
    raw = raw.sort_values("_prio", kind="stable").reset_index(drop=True)

    n_before = len(raw)
    per_src_before = raw["src"].value_counts().to_dict()
    raw = raw.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    step("dedupe on normalize_ws(text).lower() (keep first)", len(raw),
         f"(removed {n_before - len(raw):,} duplicate texts)")
    per_src_after = raw["src"].value_counts().to_dict()
    for k in sorted(SRC, key=lambda k: PRIO[k]):
        lost = per_src_before.get(k, 0) - per_src_after.get(k, 0)
        print(f"    {k:<4s} kept {per_src_after.get(k, 0):>7,d} of "
              f"{per_src_before.get(k, 0):>7,d}  ({lost:,} lost to a higher-priority pool)")

    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    train_keys = set(normalize_ws(train["text"]).str.lower().tolist())
    hit = raw["key"].isin(train_keys)
    n_hit = int(hit.sum())
    raw = raw[~hit].reset_index(drop=True)
    step("drop rows whose hash matches a train.csv passage", len(raw),
         f"(removed {n_hit:,})")
    if n_hit:
        print("    NOTE: non-zero -- pools now overlap train.csv; check provenance.")

    raw["n_words"] = raw["text"].str.split().str.len().astype("int64")
    raw = raw[(raw["n_words"] >= WORD_LO) & (raw["n_words"] <= WORD_HI)]
    step(f"length filter {WORD_LO} <= n_words <= {WORD_HI}", len(raw))

    if drop_kb:
        # Re-read the pool with the flag, recompute the same keys, and drop the
        # survivors whose key is flagged. Keeps the main path free of a second join.
        kb = pd.read_parquet(
            os.path.join(POOLS, "crtproduction__name-that-llm-open-pool",
                         "ntl_open_pool.parquet"),
            columns=["text", "flag_kb_proxy"])
        kb_keys = set(normalize_ws(kb.loc[kb["flag_kb_proxy"].fillna(False)
                                   .astype(bool), "text"]).str.lower().tolist())
        m = raw["key"].isin(kb_keys)
        raw = raw[~m].reset_index(drop=True)
        step("--drop-kb-proxy: drop flag_kb_proxy rows", len(raw), f"(removed {int(m.sum()):,})")

    union = raw[["text", "model", "src", "prompt_group", "n_words"]].reset_index(drop=True)
    return union


# --------------------------------------------------------------------------- #
# per-label cap
# --------------------------------------------------------------------------- #
def _counts(df: pd.DataFrame) -> pd.Series:
    return df["model"].value_counts().reindex(LABELS).fillna(0).astype("int64")


def _apply_cap(union: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    """Keep at most `cap` rows per label, spread across sources proportionally.

    Taking the head of the priority-sorted frame would let `op` fill the whole
    quota for a label and discard the independent `oz` harvest. Allocating the
    quota across sources (largest remainder) keeps the source mix intact; the
    rows picked inside each source are a seeded random sample, not the head, so
    no single prompt_group dominates.
    """
    rng = np.random.default_rng(seed)
    keep_parts = []
    for lab in LABELS:
        sub = union[union["model"] == lab]
        if len(sub) <= cap:
            keep_parts.append(sub)
            continue
        src_counts = sub["src"].value_counts()
        srcs = sorted(src_counts.index, key=lambda s: PRIO[s])
        total = int(src_counts.sum())
        quota = {s: int(cap * int(src_counts[s]) // total) for s in srcs}
        rem = cap - sum(quota.values())
        # largest remainder, higher priority wins ties
        order = sorted(srcs, key=lambda s: (-(cap * int(src_counts[s]) / total - quota[s]),
                                            PRIO[s]))
        for s in order[:rem]:
            quota[s] += 1
        for s in srcs:
            q = min(quota[s], int(src_counts[s]))
            if q <= 0:
                continue
            part = sub[sub["src"] == s]
            if q < len(part):
                pick = rng.choice(len(part), size=q, replace=False)
                part = part.iloc[np.sort(pick)]
            keep_parts.append(part)
    out = pd.concat(keep_parts, ignore_index=True)
    out["_prio"] = out["src"].map(PRIO).astype("int64")
    out = (out.sort_values(["_prio", "model"], kind="stable")
              .drop(columns="_prio").reset_index(drop=True))
    return out[["text", "model", "src", "prompt_group", "n_words"]]


def cap_report(counts: pd.Series, cap: int, title: str) -> None:
    capped = counts.clip(upper=cap)
    mx, mn = int(capped.max()), int(capped.min())
    cv = float(capped.std(ddof=0) / capped.mean())
    print(f"\n  {title}")
    print(f"    {'label':<24s} {'before':>8s} {'after':>8s} {'lost':>8s}")
    for lab in LABELS:
        b, a = int(counts[lab]), int(capped[lab])
        print(f"    {lab:<24s} {b:>8,d} {a:>8,d} {b - a:>8,d}")
    print(f"    {'TOTAL':<24s} {int(counts.sum()):>8,d} {int(capped.sum()):>8,d} "
          f"{int(counts.sum() - capped.sum()):>8,d}")
    print(f"    min={mn:,}  max={mx:,}  max/min={mx / mn:.3f}  CV={cv:.4f}")


def source_report(df: pd.DataFrame, title: str) -> None:
    """Count rows and prompts per source, plus how many prompts are single-source."""
    ct = df["src"].value_counts()
    print(f"\n  {title}")
    print(f"    {'src':<6s} {'priority':>9s} {'rows':>9s} {'prompts':>9s}")
    for k in sorted(SRC, key=lambda k: PRIO[k]):
        n = int(ct.get(k, 0))
        gp = int(df.loc[df["src"] == k, "prompt_group"].nunique())
        print(f"    {k:<6s} {PRIO[k]:>9d} {n:>9,d} {gp:>9,d}")
    print(f"    {'TOTAL':<6s} {'':>9s} {len(df):>9,d} {df['prompt_group'].nunique():>9,d}")


def choose_cap(union: pd.DataFrame) -> dict:
    """Largest cap that keeps max/min <= 1.6, starting from the 25th percentile.

    After capping at c (c >= min), min stays `mn` and max becomes `min(mx, c)`,
    so the constraint is `min(mx, c) <= 1.6 * mn`. The largest feasible c is
    therefore `mx` when the corpus is already balanced enough, else
    `floor(1.6 * mn)`. We start at p25 (never below the smallest class, per the
    brief) and only ever lower the cap, so the chosen value is
    `min(max(mn, p25), largest_feasible)`.
    """
    counts = _counts(union)
    mn, mx = int(counts.min()), int(counts.max())
    p25 = int(np.floor(np.percentile(counts.to_numpy(), 25)))
    largest_feasible = mx if mx <= CAP_RATIO * mn else int(np.floor(CAP_RATIO * mn))
    start = max(mn, p25)
    chosen = min(start, largest_feasible)
    return {
        "cap": int(chosen),
        "start": int(start),
        "largest_feasible": int(largest_feasible),
        "counts": counts,
        "p25": p25,
        "mn": mn,
        "mx": mx,
        "lowered": bool(chosen < start),
    }


# --------------------------------------------------------------------------- #
# hashing / manifest
# --------------------------------------------------------------------------- #
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(HERE, name)
    df.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="ignore corpus/.cache and re-extract from the pools")
    ap.add_argument("--drop-kb-proxy", action="store_true",
                    help="also drop crtproduction rows flagged flag_kb_proxy")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(HERE, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    hr("NAME THAT LLM -- corpus build")
    print(f"root {ROOT}")
    print(f"out  {HERE}")
    print(f"labels {len(LABELS)}: {', '.join(LABELS)}")

    official = build_official()
    test3 = build_test3()

    union = build_union(args.refresh, drop_kb=args.drop_kb_proxy)

    hr("3. union summary")
    step("UNION (final, before capping)", len(union))
    uc = _counts(union)
    print(f"    prompts={union['prompt_group'].nunique():,}  "
          f"labels={union['model'].nunique()}  "
          f"min={int(uc.min()):,}  max={int(uc.max()):,}  "
          f"max/min={int(uc.max()) / int(uc.min()):.3f}  "
          f"CV={float(uc.std(ddof=0) / uc.mean()):.4f}")
    print(f"    n_words: mean={union['n_words'].mean():.1f} "
          f"p25={union['n_words'].quantile(.25):.0f} "
          f"median={union['n_words'].median():.0f} "
          f"p75={union['n_words'].quantile(.75):.0f} "
          f"max={union['n_words'].max()}")
    print(f"    per-source kept rows: " +
          ", ".join(f"{k}={int((union['src'] == k).sum()):,}" for k in SRC))

    hr("4. per-label cap")
    cinfo = choose_cap(union)
    cap, largest_feasible, counts = cinfo["cap"], cinfo["largest_feasible"], cinfo["counts"]
    print(f"    per-label counts: min={cinfo['mn']:,} p25={cinfo['p25']:,} "
          f"median={int(counts.median()):,} max={cinfo['mx']:,}")
    print(f"    largest cap with max/min <= {CAP_RATIO}: {largest_feasible:,} "
          f"(= floor({CAP_RATIO} * min={cinfo['mn']:,}))")
    print(f"    cap start = max(min, p25) = {cinfo['start']:,}"
          f"{'  -> lowered to satisfy the ratio' if cinfo['lowered'] else ''}")
    print(f"    CHOSEN CAP = {cap:,} rows per label")
    capped = _apply_cap(union, cap)
    cap_report(counts, cap, f"per-label counts at cap={cap:,}:")
    source_report(capped, "capped corpus composition:")

    hr("5. writing")
    p_union = write_parquet(union, "union.parquet")
    step("corpus/union.parquet", len(union))
    p_capped = write_parquet(capped, "union_capped.parquet")
    step("corpus/union_capped.parquet", len(capped))
    p_off = write_parquet(official, "official.parquet")
    step("corpus/official.parquet", len(official))
    p_test = write_parquet(test3, "test3.parquet")
    step("corpus/test3.parquet", len(test3))

    cc = _counts(capped)
    manifest = {
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "build_seconds": round(time.time() - t0, 1),
        "root": ROOT,
        "labels": LABELS,
        "n_labels": len(LABELS),
        "source_priority": {k: {"priority": PRIO[k], "pool": SRC[k][1]} for k in
                            sorted(SRC, key=lambda k: PRIO[k])},
        "word_filter": [WORD_LO, WORD_HI],
        "dedupe": "normalize_ws(text).lower(), keep first (source priority order)",
        "cap": int(cap),
        "cap_ratio_target": CAP_RATIO,
        "cap_method": ("start at p25 of per-label counts (never below the smallest "
                       "class), lower to the largest integer cap with max/min <= 1.6; "
                       "quota per label allocated across sources proportionally "
                       "(largest remainder), seeded random sample inside a source"),
        "cap_start_p25": int(cinfo["start"]),
        "cap_lowered_to_feasible": bool(cinfo["lowered"]),
        "largest_feasible_cap": int(largest_feasible),
        "drop_kb_proxy": bool(args.drop_kb_proxy),
        "files": {},
        "row_counts": {
            "official": int(len(official)),
            "test3": int(len(test3)),
            "union": int(len(union)),
            "union_capped": int(len(capped)),
        },
        "counts_per_label": {
            "union": {lab: int(uc[lab]) for lab in LABELS},
            "union_capped": {lab: int(cc[lab]) for lab in LABELS},
        },
        "max_min_ratio": {
            "union": float(int(uc.max()) / int(uc.min())),
            "union_capped": float(int(cc.max()) / int(cc.min())),
        },
        "prompt_groups": {
            "official": int(official["prompt_group"].nunique()),
            "union": int(union["prompt_group"].nunique()),
            "union_capped": int(capped["prompt_group"].nunique()),
        },
    }
    for path in (p_union, p_capped, p_off, p_test):
        manifest["files"][os.path.basename(path)] = {
            "sha256": sha256(path),
            "bytes": os.path.getsize(path),
        }
    mpath = os.path.join(HERE, "MANIFEST.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    step("corpus/MANIFEST.json", 1, f"(cap={cap:,})")

    hr(f"done in {time.time() - t0:.1f}s")
    for name, meta in manifest["files"].items():
        print(f"  {name:<24s} {meta['sha256'][:16]}...  {meta['bytes'] / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
