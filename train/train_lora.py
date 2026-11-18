#!/usr/bin/env python
"""Two-stage LoRA classifier on a truncated Qwen3 backbone ("LoRA24" recipe).

Stage 1: big public pool, learns what each model sounds like.
Stage 2: the 2,699 official rows, learns how it sounded to *these* prompts.

Usage (NPU, via run_card.sh so the Ascend env and HBM check are handled)::

    bash run_card.sh 3 --model .../Qwen3-14B-Base --layers 27 \
        --stage1 .../data/union_capped.parquet --stage2 .../data/official.parquet \
        --out .../out/q14_stage12 --epochs1 1 --epochs2 2 --bs 8

Smoke test on any machine (CPU/CUDA, tiny random backbone, no external data)::

    python train_lora.py --smoke --out ./_smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lora24 as L  # noqa: E402


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-stage truncated-Qwen3 LoRA classifier (LoRA24 recipe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m = p.add_argument_group("model")
    m.add_argument("--model", help="base model dir (HF safetensors)")
    m.add_argument("--layers", type=int, default=None,
                   help="decoder layers to keep (14B -> 27, 8B -> 24; asserted)")
    m.add_argument("--pool_layers", default="2,4,6,8,12,16,20,24",
                   help="comma list of 8 hidden-state depths to pool")
    m.add_argument("--lora_r", type=int, default=16)
    m.add_argument("--lora_alpha", type=float, default=32.0)
    m.add_argument("--lora_dropout", type=float, default=0.05)
    m.add_argument("--dtype", default="bf16", choices=sorted(L.DTYPES))
    m.add_argument("--allow_unknown_arch", action="store_true",
                   help="skip the hidden_size/num_layers assertion (not recommended)")
    m.add_argument("--final_norm", action="store_true",
                   help="keep the trunk's final RMSNorm (recipe disables it)")

    d = p.add_argument_group("data")
    d.add_argument("--stage1", help="stage-1 parquet/csv (text, model, prompt_group)")
    d.add_argument("--stage2", help="stage-2 parquet/csv (text, model, prompt_group)")
    d.add_argument("--max_len", type=int, default=384)
    d.add_argument("--min_words", type=int, default=0, help="drop stage-1 rows shorter than this")
    d.add_argument("--max_words", type=int, default=0, help="drop stage-1 rows longer than this")
    d.add_argument("--limit1", type=int, default=0, help="debug: cap stage-1 rows")
    d.add_argument("--limit2", type=int, default=0, help="debug: cap stage-2 rows")

    o = p.add_argument_group("optimisation")
    o.add_argument("--epochs1", type=float, default=1.0)
    o.add_argument("--epochs2", type=float, default=2.0)
    o.add_argument("--bs", type=int, default=8, help="passages per batch (must be a multiple of 3)")
    o.add_argument("--lr", type=float, default=1e-4, help="stage-1 LoRA lr")
    o.add_argument("--head_lr", type=float, default=1e-3, help="stage-1 head lr")
    o.add_argument("--lr2", type=float, default=5e-5, help="stage-2 LoRA lr")
    o.add_argument("--head_lr2", type=float, default=5e-4, help="stage-2 head lr")
    o.add_argument("--wd", type=float, default=0.0, help="weight decay on LoRA params")
    o.add_argument("--head_wd", type=float, default=0.01)
    o.add_argument("--warmup", type=float, default=0.10, help="warmup fraction of total steps")
    o.add_argument("--min_lr_ratio", type=float, default=0.10)
    o.add_argument("--grad_clip", type=float, default=1.0)
    o.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    o.add_argument("--class_weight_power", type=float, default=1.0,
                   help="0 = unweighted CE, 1 = fully balanced (recipe)")
    o.add_argument("--class_weight_cap", type=float, default=5.0)
    o.add_argument("--centred_weight", type=float, default=1.0,
                   help="weight of the centred-head CE (recipe: 1.0)")

    r = p.add_argument_group("runtime")
    r.add_argument("--out", required=True, help="output checkpoint dir")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default=None, help="force e.g. cpu / cuda:0 / npu:0")
    r.add_argument("--num_threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    r.add_argument("--log_every", type=int, default=20)
    r.add_argument("--save_every", type=int, default=0, help="also checkpoint every N steps (0 = off)")
    r.add_argument("--resume", default=None, help="resume from a checkpoint dir (adapter+heads+meta)")
    r.add_argument("--eval_every", type=int, default=0,
                   help="if >0 and --stage2 given: quick stage-2 holdout bacc every N steps")
    r.add_argument("--verify_rebuild", action="store_true",
                   help="after saving, reload the base model and run one forward pass "
                        "(needs 2x model HBM; always on for --smoke)")
    r.add_argument("--no_grad_ckpt", action="store_true",
                   help="disable gradient checkpointing (default ON). Measured on a 910B3: "
                        "27/40 layers, 384 tokens, bs 8 peaks at 21.8 GB WITH ckpt and OOMs "
                        "past 40.7 GB WITHOUT it, so ckpt is the difference between fitting "
                        "on a shared card and dying 4 steps in. Costs ~30%% step time.")
    r.add_argument("--max_steps", type=int, default=0,
                   help="hard cap on optimizer steps in stage 1 (debug)")
    r.add_argument("--smoke", action="store_true",
                   help="self-contained 20-step CPU/CUDA run on 200 synthetic rows")
    r.add_argument("--smoke_steps", type=int, default=20)
    return p


def parse_args(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if a.smoke:
        return a  # --smoke overrides every data/model arg; only --out/--device matter
    if not a.model:
        p.error("--model is required (or use --smoke)")
    if not a.stage1 and not a.stage2:
        p.error("at least one of --stage1 / --stage2 is required")
    if a.bs < 2:
        p.error("--bs must be >= 2")
    return a


def batch_shape(bs: int) -> int:
    """--bs is passages/step, rounded up to whole triplets.

    The reference recipe batches in same-prompt triplets, so bs 8 -> 9 passages.
    The 1-passage gap costs <2% activation HBM and keeps every batch structured.
    """
    import math as _m

    return max(3, 3 * int(_m.ceil(bs / 3.0)))


# --------------------------------------------------------------------------- #
# batch loop shared by both stages
# --------------------------------------------------------------------------- #
class StepCounter:
    def __init__(self, total: int, warmup: float, min_ratio: float, log_every: int):
        self.total, self.warmup, self.min_ratio, self.log_every = total, warmup, min_ratio, log_every
        self.step = 0
        self.t0 = time.time()
        self.loss_sum = 0.0
        self.loss_n = 0

    def factor(self) -> float:
        return L.warmup_cosine(self.step, self.total, self.warmup, self.min_ratio)

    def log(self, extra: str = "") -> None:
        if self.step % self.log_every:
            return
        el = time.time() - self.t0
        sps = el / max(1, self.step)
        eta = sps * max(0, self.total - self.step)
        avg = self.loss_sum / max(1, self.loss_n)
        print(f"  step {self.step:6d}/{self.total}  loss {avg:.4f}  "
              f"{sps:.2f}s/step  elapsed {el/60:.1f}m  eta {eta/60:.1f}m {extra}", flush=True)
        self.loss_sum, self.loss_n = 0.0, 0


def set_lrs(opt, lrs, factor):
    for group, lr in zip(opt.param_groups, lrs):
        group["lr"] = lr * factor


def run_stage(model, tok, tokens, labels, groups, args, *, epochs, lr_lora, lr_head,
              tag, device, cw_power, cw_cap, log_every, save_every=0,
              out_dir=None, meta=None, eval_fn=None, eval_every=0, max_steps=0):
    """One stage = `epochs` passes of triplet batches over (tokens, labels, groups)."""
    n = len(tokens)
    n_seq = batch_shape(args.bs)
    steps_per_epoch = max(1, math.ceil(n / float(n_seq)))
    total = max(1, int(round(steps_per_epoch * float(epochs) / max(1, args.accum))))
    if max_steps:
        total = min(total, int(max_steps))
    sampler = L.GroupSampler(groups, n_seq, seed=args.seed + (0 if tag == "stage1" else 1))
    weights = None
    if cw_power > 0:
        w = L.class_weights(labels, power=cw_power, cap=cw_cap)
        weights = torch.from_numpy(w).to(device)
        print(f"[{tag}] class weights (min/median/max) "
              f"{w.min():.2f}/{np.median(w):.2f}/{w.max():.2f}", flush=True)
    groups_params = L.trainable_groups(model, lr_lora, lr_head, wd_lora=args.wd, wd_head=args.head_wd)
    opt = torch.optim.AdamW(groups_params, lr=lr_lora, betas=(0.9, 0.95), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    counter = StepCounter(total, args.warmup, args.min_lr_ratio, log_every)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)

    print(f"[{tag}] {n} rows, {len(set(groups))} prompts, {steps_per_epoch} steps/epoch, "
          f"{total} optimizer steps, {n_seq} passages/step (--bs {args.bs}), "
          f"accum {args.accum}, lr {lr_lora:g}/{lr_head:g}, epochs {epochs}", flush=True)

    model.train()
    step_t0 = time.time()
    micro = 0
    opt.zero_grad(set_to_none=True)
    for _epoch in range(max(1, int(math.ceil(epochs)))):
        if counter.step >= total:
            break
        for _ in range(steps_per_epoch):
            if counter.step >= total:
                break
            idx, gids = sampler.batch()
            ids, att = L.collate(tokens, idx, device)
            tgt = y[torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)]
            z = model.features(ids, att)
            zc, keep = L.centre_by_group(z, gids)
            plain, centred = model.head_logits(z, zc)
            loss_plain = L.ce_masked(plain, tgt, weights)
            loss_cent = L.ce_masked(centred, tgt, weights, keep)
            loss = (loss_plain + args.centred_weight * loss_cent) / args.accum
            if not torch.isfinite(loss):
                print(f"  ! non-finite loss at step {counter.step}, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                micro = 0
                continue
            loss.backward()
            micro += 1
            counter.loss_sum += float(loss.detach()) * args.accum
            counter.loss_n += 1
            if micro < args.accum:
                continue
            micro = 0
            set_lrs(opt, base_lrs, counter.factor())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            counter.step += 1
            counter.log(f"(plain {float(loss_plain):.3f} centred {float(loss_cent):.3f}, "
                        f"{step_t0 and (time.time()-step_t0)/max(1,counter.step):.2f}s/step)")
            if save_every and out_dir and meta and counter.step % save_every == 0:
                meta["step"] = counter.step
                L.save_checkpoint(model, os.path.join(out_dir, f"step{counter.step}"), meta)
            if eval_every and eval_fn and counter.step % eval_every == 0:
                model.eval()
                eval_fn(model, counter.step)
                model.train()
    if micro:  # flush a partial accumulation window
        opt.step()
        opt.zero_grad(set_to_none=True)
        counter.step += 1
    print(f"[{tag}] done: {counter.step} steps in {(time.time() - counter.t0)/60:.1f} min", flush=True)
    return counter.step


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    L.seed_all(args.seed)
    device = L.pick_device(args.device)
    print(f"[env] device={device}  torch={torch.__version__}  "
          f"ascend_blocking={os.environ.get('ASCEND_LAUNCH_BLOCKING', 'unset')}", flush=True)
    if device.type == "cpu" and not args.smoke:
        print("[env] WARNING: running on CPU -- this is a real training run on the wrong device.",
              flush=True)

    smoke = None
    if args.smoke:
        args.out = os.path.abspath(args.out)
        smoke = L.make_smoke_assets(os.path.join(args.out, "_smoke_assets"))
        if args.model and os.path.exists(args.model):
            print(f"[smoke] using the real weights at {args.model}", flush=True)
        else:
            args.model = smoke["model_dir"]
        args.stage1 = smoke["stage1"]
        args.stage2 = smoke["stage2"]
        args.layers = args.layers if args.layers else smoke["keep"]
        args.max_len = smoke["max_len"]
        args.pool_layers = ",".join(str(x) for x in smoke["pool_layers"])
        args.bs = 6
        args.accum = 1
        args.epochs2 = 0.0
        args.limit1 = args.limit1 or 200
        args.log_every = 1
        args.allow_unknown_arch = True
        args.verify_rebuild = True
        args.device = args.device or "cpu"
        device = L.pick_device(args.device)
        if not args.max_steps:
            args.max_steps = args.smoke_steps
        print(f"[smoke] device={device}, backbone={args.model}, steps={args.max_steps}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    pool_layers = [int(x) for x in str(args.pool_layers).split(",") if x.strip()]
    dtype = L.DTYPES[args.dtype]
    model, info = L.build_model(
        args.model, args.layers, pool_layers, dtype, device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        no_final_norm=not args.final_norm, allow_unknown_arch=args.allow_unknown_arch,
    )
    if not args.no_grad_ckpt:
        # The frozen trunk must emit grads into the adapter inputs, otherwise
        # checkpointing would give all-zero LoRA grads.
        model.trunk.gradient_checkpointing_enable()
        try:
            model.trunk.enable_input_require_grads()
        except Exception as exc:  # very old transformers
            print(f"[model] enable_input_require_grads unavailable ({exc})", flush=True)
        print("[model] gradient checkpointing ON (disable with --no_grad_ckpt)", flush=True)
    info["grad_ckpt"] = not bool(args.no_grad_ckpt)
    tok = L.load_tokenizer(args.model)
    info["max_len"] = int(args.max_len)
    info["centred_weight"] = float(args.centred_weight)
    info["class_weight"] = dict(power=float(args.class_weight_power), cap=float(args.class_weight_cap))
    print(f"[env] tokenizer eos={tok.eos_token_id} pad={tok.pad_token_id} vocab={tok.vocab_size}",
          flush=True)

    if args.resume:
        print(f"[resume] loading {args.resume}", flush=True)
        L.load_into(model, args.resume)

    meta = dict(format=L.FORMAT, labels=list(L.LABELS), cfg=info, args=vars(args).copy(),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), stages={})
    for k in ("model", "stage1", "stage2", "out"):
        meta["args"][k] = os.path.abspath(str(meta["args"][k])) if meta["args"].get(k) else None

    # ---- stage 1 ---------------------------------------------------------- #
    if args.stage1 and args.epochs1 > 0:
        d1 = L.read_table(args.stage1, limit=args.limit1, seed=args.seed,
                          min_words=args.min_words, max_words=args.max_words)
        if d1["labels"] is None or d1["groups"] is None:
            raise SystemExit("[data] stage1 needs both 'model' and 'prompt_group' columns")
        t0 = time.time()
        tok1 = L.encode_all(tok, d1["texts"], args.max_len)
        print(f"[stage1] lengths {L.length_stats(tok1)}", flush=True)
        n = run_stage(model, tok, tok1, d1["labels"], d1["groups"], args,
                      epochs=args.epochs1, lr_lora=args.lr, lr_head=args.head_lr,
                      tag="stage1", device=device,
                      cw_power=args.class_weight_power, cw_cap=args.class_weight_cap,
                      log_every=args.log_every, save_every=args.save_every,
                      out_dir=args.out, meta=meta, max_steps=args.max_steps)
        meta["stages"]["stage1"] = dict(rows=len(tok1), steps=n, epochs=args.epochs1,
                                        lr=args.lr, head_lr=args.head_lr,
                                        lengths=L.length_stats(tok1),
                                        minutes=round((time.time() - t0) / 60, 2))
        L.save_checkpoint(model, args.out, meta)
        print(f"[stage1] checkpoint -> {args.out}", flush=True)

    # ---- stage 2 ---------------------------------------------------------- #
    if args.stage2 and args.epochs2 > 0:
        d2 = L.read_table(args.stage2, limit=args.limit2, seed=args.seed)
        if d2["labels"] is None or d2["groups"] is None:
            raise SystemExit("[data] stage2 needs both 'model' and 'prompt_group' columns")
        tok2 = L.encode_all(tok, d2["texts"], args.max_len)
        print(f"[stage2] lengths {L.length_stats(tok2)}", flush=True)

        eval_fn = None
        if args.eval_every > 0 and len(tok2) >= 90:
            folds = L.prompt_folds(len(tok2), d2["groups"], d2["labels"], n_splits=5, seed=0)
            hold = folds[0]
            tr = np.setdiff1d(np.arange(len(tok2)), hold)

            def eval_fn(mdl, step, _tok=tok2, _y=d2["labels"], _tr=tr, _hold=hold):
                mdl.eval()
                lp = L.predict_logprobs(mdl, [_tok[i] for i in _hold], device,
                                        bs=args.bs, verbose=False)
                r = L.evaluate(_y[_hold], lp.argmax(1))
                print(f"  [eval@{step}] heldout n={r['n']} bacc={r['bacc']:.4f} "
                      f"acc={r['accuracy']:.4f}", flush=True)

        t0 = time.time()
        n = run_stage(model, tok, tok2, d2["labels"], d2["groups"], args,
                      epochs=args.epochs2, lr_lora=args.lr2, lr_head=args.head_lr2,
                      tag="stage2", device=device,
                      cw_power=args.class_weight_power, cw_cap=args.class_weight_cap,
                      log_every=args.log_every, save_every=args.save_every,
                      out_dir=args.out, meta=meta, eval_fn=eval_fn, eval_every=args.eval_every)
        meta["stages"]["stage2"] = dict(rows=len(tok2), steps=n, epochs=args.epochs2,
                                        lr=args.lr2, head_lr=args.head_lr2,
                                        lengths=L.length_stats(tok2),
                                        minutes=round((time.time() - t0) / 60, 2))
        L.save_checkpoint(model, args.out, meta)
        print(f"[stage2] checkpoint -> {args.out}", flush=True)

    if not meta["stages"]:
        raise SystemExit("[main] nothing ran -- give --stage1/--epochs1 or --stage2/--epochs2")

    # ---- verify the artefact ---------------------------------------------- #
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    L.save_checkpoint(model, args.out, meta)
    print(f"[done] wrote {args.out} (adapter + heads + meta.json)", flush=True)
    verify_checkpoint(args.out, model, device, rebuild=bool(smoke) or args.verify_rebuild)
    if smoke:
        print("[smoke] OK -- artefact round-trips and a forward pass is finite", flush=True)
    return 0


@torch.no_grad()
def verify_checkpoint(out_dir: str, model, device, rebuild: bool = False) -> None:
    """Structural check of the written checkpoint (+ optional full rebuild).

    A full rebuild re-loads the 14B base, which does not fit twice on one card,
    so by default this only checks that every tensor on disk has the right shape
    and that no parameter of the live model is left unaccounted for. --verify_rebuild
    (and --smoke) additionally rebuild a fresh model and run one forward pass:
    that is the check which catches a meta.json unable to reconstruct the model.
    """
    from safetensors.torch import load_file

    ad_path = os.path.join(out_dir, "adapter", "adapter_model.safetensors")
    h_path = os.path.join(out_dir, "heads.safetensors")
    missing = [p for p in (ad_path, h_path, os.path.join(out_dir, "meta.json")) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[verify] checkpoint incomplete, missing {missing}")
    ad, hd = load_file(ad_path), load_file(h_path)
    mods = dict(model.named_modules())
    for key, val in ad.items():
        mod_name, which = L._split_lora_key(key)
        mod = mods.get(mod_name) if mod_name else None
        if not isinstance(mod, L.LoRALinear):
            raise SystemExit(f"[verify] adapter tensor {key} does not map to a LoRALinear")
        want = tuple(mod.lora_A.shape if which == "lora_A" else mod.lora_B.shape)
        if tuple(val.shape) != want:
            raise SystemExit(f"[verify] {mod_name}.{which}: on-disk {tuple(val.shape)} != model {want}")
        if not torch.isfinite(val).all():
            raise SystemExit(f"[verify] {mod_name}.{which}: non-finite weights on disk")
    want_keys = set(L.heads_state(model))
    if set(hd) != want_keys:
        raise SystemExit(f"[verify] heads.safetensors keys differ: "
                         f"extra {sorted(set(hd) - want_keys)[:4]} "
                         f"missing {sorted(want_keys - set(hd))[:4]}")
    print(f"[verify] keys ok: {len(ad)} adapter + {len(hd)} head tensors, shapes match", flush=True)
    if not rebuild:
        return
    fresh, tok, meta = L.rebuild(out_dir, device, verbose=False)
    probe = tok("the quick brown fox jumps over the lazy dog", add_special_tokens=False)["input_ids"]
    z = fresh.features(*L.collate([np.asarray(probe, dtype=np.int32)], [0], device))
    lp = F.log_softmax(fresh.head_logits(z)[0].float(), dim=-1)
    ok = bool(torch.isfinite(lp).all()) and tuple(lp.shape) == (1, len(L.LABELS))
    print(f"[verify] rebuilt: dim={meta['cfg']['dim']} keep={meta['cfg']['keep']} "
          f"layers={meta['cfg']['layers']} sum(exp(logp))={float(lp.exp().sum()):.4f} finite={ok}",
          flush=True)
    if not ok:
        raise SystemExit("[verify] rebuilt checkpoint produced a bad forward pass")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[main] interrupted", file=sys.stderr)
        sys.exit(130)
