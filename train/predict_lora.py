#!/usr/bin/env python
"""Inference with a trained LoRA24 checkpoint -> log-probabilities (n x 18).

Output is ``float32`` .npy of **log-probabilities** in the fixed ``lora24.LABELS``
order, exactly what the Kaggle blend consumes (sum of w * log p across members).

    python predict_lora.py --out .../out/q14_stage12 --input .../data/test.csv \
        --save out/q14_test.npy

Options that mirror the reference blend:
  --groups <col>   prompt-group column (csv/parquet). Enables --twin_w > 0, the
                   centred-twin term (subtract the group mean feature, read the
                   centred head). Without it, singletons contribute nothing.
  --twin_w         weight of the centred term (recipe: 0.35 at blend time; the
                   checkpoint's own centred head is what is used).
  --sinkhorn       rescale class columns to a uniform prior (needs the full set).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lora24 as L  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict with a LoRA24 checkpoint.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", required=True, help="checkpoint dir written by train_lora.py")
    p.add_argument("--input", required=True, help="parquet/csv with a 'text' column")
    p.add_argument("--save", required=True, help="output .npy path (n x 18 log-probs)")
    p.add_argument("--base", default=None, help="override base model dir from meta.json")
    p.add_argument("--groups", default=None, help="column holding prompt-group ids")
    p.add_argument("--twin_w", type=float, default=0.0, help="centred-twin weight (0 = off)")
    p.add_argument("--sinkhorn", action="store_true",
                   help="rebalance class columns to a uniform prior before saving")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--max_len", type=int, default=0, help="override meta max_len (0 = use meta)")
    p.add_argument("--device", default=None)
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument("--dump_scores", default=None,
                   help="also write an .npy of argmax class indices for eyeballing")
    return p


def read_texts(path: str, groups_col: str | None):
    """Read only text[, groups] -- via pyarrow, so no pandas dependency."""
    cols = ["text"] + ([groups_col] if groups_col else [])
    tbl = L._arrow_table(path, cols)
    texts = L._col_strings(tbl, "text")
    groups = L._col_strings(tbl, groups_col) if groups_col else None
    if not texts:
        raise SystemExit(f"[input] {path} has no rows")
    return texts, groups


def sinkhorn(p: np.ndarray, n_iter: int = 100, target: float | None = None) -> np.ndarray:
    """Rebalance class columns to a uniform prior (the reference blend's last step)."""
    p = p.copy()
    tgt = target if target is not None else len(p) / p.shape[1]
    for _ in range(n_iter):
        p *= tgt / np.clip(p.sum(0), 1e-12, None)
        p /= np.clip(p.sum(1, keepdims=True), 1e-12, None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.twin_w > 0 and not args.groups:
        raise SystemExit("--twin_w > 0 needs --groups <column> (the centred head is per prompt)")
    device = L.pick_device(args.device)
    t0 = time.time()

    model, tok, meta = L.rebuild(args.out, device, base_dir=args.base)
    max_len = args.max_len or int(meta["cfg"].get("max_len", 384))
    texts, groups = read_texts(args.input, args.groups)
    print(f"[predict] {len(texts)} rows from {args.input}  device={device}  "
          f"keep={meta['cfg']['keep']} layers={meta['cfg']['layers']} max_len={max_len}", flush=True)

    tokens = L.encode_all(tok, texts, max_len)
    lp = L.predict_logprobs(model, tokens, device, bs=args.bs, groups=groups,
                            twin_w=args.twin_w)
    if args.sinkhorn:
        lp = np.log(np.clip(sinkhorn(np.exp(lp)), 1e-12, None)).astype(np.float32)
        print("[predict] sinkhorn rebalancing applied", flush=True)

    if lp.shape != (len(texts), len(L.LABELS)):
        raise SystemExit(f"[predict] bad output shape {lp.shape}")
    if not np.isfinite(lp).all():
        bad = int((~np.isfinite(lp)).any(1).sum())
        raise SystemExit(f"[predict] {bad} row(s) produced non-finite log-probs")
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
    np.save(args.save, lp)
    if args.dump_scores:
        np.save(args.dump_scores, lp.argmax(1).astype(np.int16))
    pred = lp.argmax(1)
    counts = np.bincount(pred, minlength=len(L.LABELS))
    top = int(counts.max()) if counts.size else 0
    print(f"[predict] saved {args.save} {lp.shape} {lp.dtype} in {time.time() - t0:.0f}s; "
          f"predicted classes {int((counts > 0).sum())}/18, most frequent "
          f"{L.LABELS[int(counts.argmax())]} ({top} rows)", flush=True)
    print(json.dumps({"labels": L.LABELS}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
