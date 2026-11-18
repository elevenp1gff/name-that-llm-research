"""Measure step time / HBM for the real 27-layer Qwen3-14B recipe on one 910B3.

Builds the actual truncated trunk (27 of 40 layers, hidden 5120), runs a few
optimizer steps at the requested batch size, and reports s/step, peak HBM, and the
extrapolated stage1/stage2 wall clock. Uses synthetic ids so no data or tokenizer
is needed -- the numbers are the transformer cost, which is what dominates.

    ASCEND_RT_VISIBLE_DEVICES=3 python bench_step.py --bs 8 --steps 6
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lora24 as L  # noqa: E402

REPO = "/path/to/workspace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{REPO}/models/Qwen3-14B-Base")
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--pool_layers", default="2,4,6,8,12,16,20,24")
    ap.add_argument("--bs", type=int, default=8, help="passages/step (rounded up to triplets)")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--steps", type=int, default=6, help="timed steps (after 2 warmup)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--epochs1", type=float, default=1.0)
    ap.add_argument("--epochs2", type=float, default=2.0)
    ap.add_argument("--n1", type=int, default=0, help="stage-1 rows (default: from data dir)")
    ap.add_argument("--n2", type=int, default=2699)
    ap.add_argument("--dtype", default="bf16", choices=sorted(L.DTYPES))
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = L.pick_device(args.device)
    print(f"[bench] device={device} model={args.model}", flush=True)
    pool_layers = [int(x) for x in args.pool_layers.split(",")]
    model, info = L.build_model(args.model, args.layers, pool_layers, L.DTYPES[args.dtype],
                                device, no_final_norm=True, verbose=True)
    if args.grad_ckpt:
        model.trunk.gradient_checkpointing_enable()
        model.trunk.enable_input_require_grads()
        print("[bench] gradient checkpointing ON", flush=True)

    n_seq = max(3, 3 * (-(-args.bs // 3)))
    attn_impl = getattr(model.trunk.config, "_attn_implementation", "?")
    print(f"[bench] {n_seq} passages/step x {args.max_len} tokens, "
          f"attn_implementation={attn_impl}", flush=True)
    if attn_impl in ("eager", "sdpa"):
        # Eager attention materialises (bs, heads, seq, seq) per layer:
        # 9 x 40 x 384 x 384 x 2B = 106 MB/layer x 27 = 2.9 GB, plus its saved copy.
        # SDPA on torch 2.1/NPU often falls back to the same math. That is the
        # difference between fitting at bs 8 and OOM at bs 6.
        print("[bench] NOTE: memory-efficient attention is NOT active; "
              "expect O(seq^2) activation cost", flush=True)

    rng = np.random.default_rng(0)
    labels = torch.from_numpy(rng.integers(0, 18, size=n_seq).astype(np.int64)).to(device)
    weights = torch.from_numpy(L.class_weights(rng.integers(0, 18, size=5000))).to(device)
    gids = [f"g{i // 3}" for i in range(n_seq)]

    groups = L.trainable_groups(model, 1e-4, 1e-3)
    opt = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
    model.train()

    times = []
    for step in range(args.warmup + args.steps):
        ids = torch.from_numpy(
            rng.integers(1000, 100000, size=(n_seq, args.max_len)).astype(np.int64)).to(device)
        att = torch.ones_like(ids)
        t0 = time.time()
        z = model.features(ids, att)
        zc, keep = L.centre_by_group(z, gids)
        plain, centred = model.head_logits(z, zc)
        loss = L.ce_masked(plain, labels, weights) + L.ce_masked(centred, labels, weights, keep)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tag = "warmup" if step < args.warmup else "timed "
        print(f"[bench] {tag} step {step}: {dt:.3f}s  loss {float(loss):.4f}", flush=True)
        if step >= args.warmup:
            times.append(dt)

    peak_mb = None
    if device.type == "npu":
        peak_mb = torch.npu.max_memory_allocated() / 2**20
    elif device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 2**20

    med = float(np.median(times))
    n1 = args.n1
    if not n1:
        cand = f"{REPO}/data/union_capped.parquet"
        if os.path.exists(cand):
            try:
                import pyarrow.parquet as pq
                n1 = pq.ParquetFile(cand).metadata.num_rows
            except Exception as exc:
                # A file mid-upload has no footer; fall back rather than fail a bench.
                print(f"[bench] {cand} unreadable ({type(exc).__name__}); "
                      f"assuming the recipe's 34,940 pool rows", flush=True)
    n1 = n1 or 34940
    steps1 = int(np.ceil(n1 / n_seq) * args.epochs1)
    steps2 = int(np.ceil(args.n2 / n_seq) * args.epochs2)
    out = dict(
        bs_requested=args.bs, passages_per_step=n_seq, max_len=args.max_len,
        median_step_s=round(med, 3), min_step_s=round(min(times), 3),
        max_step_s=round(max(times), 3), peak_hbm_mb=round(peak_mb, 1) if peak_mb else None,
        stage1_rows=n1, stage1_steps=steps1, stage1_min=round(steps1 * med / 60, 1),
        stage2_rows=args.n2, stage2_steps=steps2, stage2_min=round(steps2 * med / 60, 1),
        total_min=round((steps1 + steps2) * med / 60, 1),
        grad_ckpt=bool(args.grad_ckpt), cfg=info, device=str(device),
    )
    print("[bench] " + json.dumps({k: v for k, v in out.items() if k != "cfg"}, indent=2),
          flush=True)
    path = os.path.join(HERE, f"bench_bs{args.bs}{'_ckpt' if args.grad_ckpt else ''}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bench] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
