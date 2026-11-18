#!/usr/bin/env python
"""5-fold prompt-grouped CV on the official file, scored two ways.

For each held-out fold we report balanced accuracy on
  (a) all 18 answers per prompt (the standard OOF number), and
  (b) only 1-2 answers per prompt ("sim_k", resampled -- this is what tracks the
      leaderboard, because the hidden test gives each prompt only a few answers).

Two modes:
  --ckpt <dir>   evaluate an existing checkpoint (fast; no training)
  (default)      train a fresh model per fold, stage1 on the pool then stage2 on
                 the 4 training folds, so the number is a true OOF estimate.
                 The fold checkpoints are kept so the same run doubles as an
                 ensemble for inference.

    python oof_lora.py --model .../Qwen3-14B-Base --layers 27 \
        --stage1 .../data/union_capped.parquet --stage2 .../data/official.parquet \
        --out .../out/oof_q14 --folds 5 --epochs1 1 --epochs2 2 --bs 8
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
import train_lora as T  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt-grouped 5-fold CV for the LoRA24 recipe.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", required=True, help="output dir for fold checkpoints and reports")
    p.add_argument("--ckpt", default=None,
                   help="evaluate this checkpoint instead of training per fold")
    p.add_argument("--model", default=None, help="base model dir (required unless --ckpt)")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--pool_layers", default="2,4,6,8,12,16,20,24")
    p.add_argument("--stage1", default=None, help="pool parquet for stage 1")
    p.add_argument("--stage2", default=None, help="official parquet (text, model, prompt_group)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs1", type=float, default=1.0)
    p.add_argument("--epochs2", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=5e-5)
    p.add_argument("--head_lr2", type=float, default=5e-4)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--dtype", default="bf16", choices=sorted(L.DTYPES))
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold_seed", type=int, default=0, help="seed for the fold partition")
    p.add_argument("--sim_k", default="1-2", help="lo-hi, e.g. 1-2 or 2-3")
    p.add_argument("--sim_reps", type=int, default=20)
    p.add_argument("--limit1", type=int, default=0)
    p.add_argument("--class_weight_power", type=float, default=1.0)
    p.add_argument("--class_weight_cap", type=float, default=5.0)
    p.add_argument("--centred_weight", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=0.10)
    p.add_argument("--min_lr_ratio", type=float, default=0.10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--head_wd", type=float, default=0.01)
    p.add_argument("--device", default=None)
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--twin_w", type=float, default=0.0, help="centred-twin weight at eval (0 = plain only)")
    p.add_argument("--keep_folds", action="store_true", help="do not delete a fold's checkpoint")
    p.add_argument("--only_fold", type=int, default=None,
                   help="train and evaluate ONLY this fold, then exit (for card-parallel runs)")
    p.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--allow_unknown_arch", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 fold, 20 steps, tiny random backbone")
    return p


def parse_sim_k(spec: str):
    if "-" in str(spec):
        lo, hi = str(spec).split("-", 1)
        return int(lo), int(hi)
    return int(spec), int(spec)


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not args.stage2 and not args.smoke:
        p.error("--stage2 is required (or use --smoke)")
    if not args.ckpt and not args.model and not args.smoke:
        p.error("give --ckpt <dir> to evaluate a checkpoint, or --model to train per fold")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    lo, hi = parse_sim_k(args.sim_k)
    os.makedirs(args.out, exist_ok=True)
    L.seed_all(args.seed)

    smoke = None
    if args.smoke:
        smoke = L.make_smoke_assets(os.path.join(args.out, "_smoke_assets"))
        args.model = args.model or smoke["model_dir"]
        args.stage1 = args.stage1 or smoke["stage1"]
        args.stage2 = smoke["stage2"]
        args.layers = args.layers or smoke["keep"]
        args.max_len = smoke["max_len"]
        args.pool_layers = ",".join(str(x) for x in smoke["pool_layers"])
        args.folds = 2
        args.bs = 6
        args.limit1 = 60
        args.epochs1 = 0.3
        args.epochs2 = 0.5
        args.sim_reps = 3
        args.log_every = 5
        args.device = args.device or "cpu"
        args.allow_unknown_arch = True

    device = L.pick_device(args.device)
    print(f"[env] device={device} sim_k={lo}-{hi} folds={args.folds} "
          f"mode={'ckpt:' + args.ckpt if args.ckpt else 'train-per-fold'}", flush=True)

    d2 = L.read_table(args.stage2, verbose=True)
    if d2["labels"] is None or d2["groups"] is None:
        raise SystemExit("[cv] --stage2 needs 'model' and 'prompt_group'")
    texts, labels, groups = d2["texts"], d2["labels"], d2["groups"]
    folds = L.prompt_folds(len(texts), groups, labels, n_splits=args.folds, seed=args.fold_seed)
    for k, f in enumerate(folds):
        present = np.bincount(labels[f], minlength=len(L.LABELS))
        print(f"[cv] fold {k}: {len(f)} rows, {len(set(np.asarray(groups)[f].tolist()))} prompts, "
              f"classes {int((present > 0).sum())}/18", flush=True)

    oof = np.full((len(texts), len(L.LABELS)), np.nan, dtype=np.float32)
    fold_reports = []
    for k, hold in enumerate(folds):
        if args.only_fold is not None and k != args.only_fold:
            continue
        t_fold = time.time()
        out_k = os.path.join(args.out, f"fold{k}")
        if args.ckpt:
            print(f"[fold {k}] evaluating {args.ckpt}", flush=True)
            model, tok, meta = L.rebuild(args.ckpt, device, verbose=False)
        else:
            train_idx = np.setdiff1d(np.arange(len(texts)), hold)
            _, meta = train_one_fold(args, out_k, [texts[i] for i in train_idx],
                                     labels[train_idx], [groups[i] for i in train_idx],
                                     device, smoke)
            model, tok, meta = L.rebuild(out_k, device, verbose=False)
        max_len = int(meta["cfg"].get("max_len", args.max_len))
        hold_tokens = L.encode_all(tok, [texts[i] for i in hold], max_len, verbose=False)
        hold_groups = [groups[i] for i in hold]
        lp = L.predict_logprobs(model, hold_tokens, device, bs=args.bs,
                               groups=hold_groups if args.twin_w > 0 else None,
                               twin_w=args.twin_w, verbose=False)
        oof[hold] = lp
        pred = lp.argmax(1)
        y_hold = labels[hold]
        full = L.evaluate(y_hold, pred)
        sim = L.board_like(y_hold, pred, hold_groups, lo=lo, hi=hi,
                           seed=args.fold_seed * 100 + k, n_reps=args.sim_reps)
        fold_reports.append(dict(fold=k, rows=int(len(hold)), bacc_full=full["bacc"],
                                 acc_full=full["accuracy"], bacc_sim=sim["bacc"],
                                 bacc_sim_sd=sim["bacc_sd"], n_classes=full["n_classes"],
                                 minutes=round((time.time() - t_fold) / 60, 2)))
        print(f"[fold {k}] all-18 bacc={full['bacc']:.4f} acc={full['accuracy']:.4f} | "
              f"sim_k {lo}-{hi} bacc={sim['bacc']:.4f} (sd {sim['bacc_sd']:.4f}) | "
              f"{fold_reports[-1]['minutes']:.1f}m", flush=True)
        del model
        if args.only_fold is not None:
            return 0
        if not args.ckpt and not args.keep_folds:
            import shutil
            shutil.rmtree(out_k, ignore_errors=True)

    done = ~np.isnan(oof).any(1)
    y = labels[done]
    pred = oof[done].argmax(1)
    grp = [groups[i] for i in np.where(done)[0]]
    full = L.evaluate(y, pred)
    sim = L.board_like(y, pred, grp, lo=lo, hi=hi, seed=args.fold_seed, n_reps=max(20, args.sim_reps))
    bf = np.array([r["bacc_full"] for r in fold_reports])
    bs = np.array([r["bacc_sim"] for r in fold_reports])
    print("\n" + "=" * 78, flush=True)
    print(f"OOF over {int(done.sum())}/{len(texts)} rows, {args.folds} prompt-grouped folds", flush=True)
    print(f"  (a) all 18 per prompt : mean {bf.mean():.4f}  sd {bf.std():.4f}  "
          f"folds {[round(float(x), 4) for x in bf]}", flush=True)
    print(f"  (b) sim_k {lo}-{hi}       : mean {bs.mean():.4f}  sd {bs.std():.4f}  "
          f"folds {[round(float(x), 4) for x in bs]}", flush=True)
    print(f"  pooled (a) bacc {full['bacc']:.4f} acc {full['accuracy']:.4f} | "
          f"pooled (b) bacc {sim['bacc']:.4f} sd {sim['bacc_sd']:.4f} "
          f"range {sim.get('bacc_min', float('nan')):.4f}-{sim.get('bacc_max', float('nan')):.4f}",
          flush=True)
    worst = sorted(full["per_class_recall"].items(), key=lambda kv: kv[1])[:5]
    print("  weakest classes: " + ", ".join(f"{L.LABELS[c]} {r:.2f}" for c, r in worst), flush=True)
    print("=" * 78, flush=True)

    np.save(os.path.join(args.out, "oof_logprobs.npy"), oof)
    np.save(os.path.join(args.out, "oof_folds.npy"),
            np.concatenate([np.full(len(f), k, dtype=np.int16) for k, f in enumerate(folds)]))
    report = dict(
        rows=int(len(texts)), folds=args.folds, sim_k=[lo, hi], sim_reps=args.sim_reps,
        per_fold=fold_reports,
        mean_bacc_full=float(bf.mean()), sd_bacc_full=float(bf.std()),
        mean_bacc_sim=float(bs.mean()), sd_bacc_sim=float(bs.std()),
        pooled_bacc_full=full["bacc"], pooled_acc_full=full["accuracy"],
        pooled_bacc_sim=sim["bacc"], pooled_bacc_sim_sd=sim["bacc_sd"],
        per_class_recall={L.LABELS[c]: float(r) for c, r in full["per_class_recall"].items()},
        args=vars(args), labels=L.LABELS,
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(os.path.join(args.out, "oof_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.out, "oof_labels.npy"), labels)
    print(f"[done] {args.out}/oof_report.json, oof_logprobs.npy, oof_labels.npy", flush=True)
    return 0


def train_one_fold(args, out_k, texts, labels, groups, device, smoke):
    """Stage1 (pool) then stage2 (4 training folds) into out_k."""
    ns = argparse.Namespace(**vars(args))
    ns.out = out_k
    ns.max_steps = 20 if smoke else 0
    ns.allow_unknown_arch = bool(smoke)
    ns.log_every = args.log_every
    ns.pool_layers = args.pool_layers
    ns.dtype = args.dtype
    os.makedirs(out_k, exist_ok=True)

    pool_layers = [int(x) for x in str(args.pool_layers).split(",") if x.strip()]
    model, info = L.build_model(
        args.model, args.layers, pool_layers, L.DTYPES[args.dtype], device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        allow_unknown_arch=ns.allow_unknown_arch, verbose=False,
    )
    if not args.no_grad_ckpt:
        model.trunk.gradient_checkpointing_enable()
        try:
            model.trunk.enable_input_require_grads()
        except Exception:
            pass
    info["grad_ckpt"] = not bool(args.no_grad_ckpt)
    tok = L.load_tokenizer(args.model)
    info["max_len"] = int(args.max_len)
    info["centred_weight"] = float(args.centred_weight)
    meta = dict(format=L.FORMAT, labels=list(L.LABELS), cfg=info, args=vars(ns).copy(),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), stages={})

    if args.stage1 and args.epochs1 > 0:
        d1 = L.read_table(args.stage1, limit=args.limit1, seed=args.seed)
        t1 = L.encode_all(tok, d1["texts"], args.max_len, verbose=False)
        n = T.run_stage(model, tok, t1, d1["labels"], d1["groups"], ns,
                        epochs=args.epochs1, lr_lora=args.lr, lr_head=args.head_lr,
                        tag="s1", device=device, cw_power=args.class_weight_power,
                        cw_cap=args.class_weight_cap, log_every=args.log_every)
        meta["stages"]["stage1"] = dict(rows=len(t1), steps=n)
        L.save_checkpoint(model, out_k, meta)

    t2 = L.encode_all(tok, texts, args.max_len, verbose=False)
    n = T.run_stage(model, tok, t2, labels, groups, ns,
                    epochs=args.epochs2, lr_lora=args.lr2, lr_head=args.head_lr2,
                    tag="s2", device=device, cw_power=args.class_weight_power,
                    cw_cap=args.class_weight_cap, log_every=args.log_every)
    meta["stages"]["stage2"] = dict(rows=len(t2), steps=n)
    L.save_checkpoint(model, out_k, meta)
    del model
    return out_k, meta


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
