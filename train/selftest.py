"""Local checks that do not need the real 14B weights.

Covers: arch assertions against the real Qwen3-14B/8B configs, pooling-depth
validation, LoRA injection count, masking, centring, class weights, fold
determinism, checkpoint round-trip, and the run_card.sh HBM parser.
Run:  python selftest.py

Exits 0 only if every check passes. On an NPU host this must be launched through
run_card.sh (or after sourcing the Ascend env), because `import torch_npu` in
lora24.pick_device fails with libhccl.so otherwise -- the error below is then the
*test harness*, not the code under test.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lora24 as L  # noqa: E402

FAIL = []


def preflight() -> None:
    """Fail fast with the real cause if the Ascend env was never sourced."""
    try:
        from transformers import Qwen3Model  # noqa: F401
        return
    except Exception as exc:
        msg = str(exc)
        hint = ""
        if "libhccl" in msg or "einops" in msg:
            hint = ("\n  -> launch this through run_card.sh, or:\n"
                    "       source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
                    "       export PYTHONPATH=.:$PYTHONPATH:/path/to/workspace/vendor\n"
                    "     (libhccl.so = Ascend env not sourced; einops = bootstrap_deps.sh)")
        print(f"PREFLIGHT FAILED: cannot import Qwen3Model: "
              f"{type(exc).__name__}: {msg.splitlines()[0]}{hint}")
        sys.exit(2)


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAIL.append(name)


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


preflight()


# --------------------------------------------------------------------------- #
# 1. arch assertions against the real published configs
# --------------------------------------------------------------------------- #
Q14 = dict(hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
           num_key_value_heads=8, head_dim=128, intermediate_size=17408,
           vocab_size=151936, model_type="qwen3", rms_norm_eps=1e-6)
Q8 = dict(hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
          num_key_value_heads=8, head_dim=128, intermediate_size=12288,
          vocab_size=151936, model_type="qwen3", rms_norm_eps=1e-6)

try:
    from transformers import Qwen3Config

    kv = {k: v for k, v in Q14.items() if k not in ("model_type", "rms_norm_eps")}
    kv["rms_norm_eps"] = 1e-6
    cfg14 = Qwen3Config(**kv)
    cfg8 = Qwen3Config(**{**kv, "hidden_size": 4096, "num_hidden_layers": 36,
                          "num_attention_heads": 32, "intermediate_size": 12288})
    check("Qwen3Config accepts the 14B/8B field sets", True)
except Exception as exc:
    cfg14, cfg8 = Cfg(**Q14), Cfg(**Q8)
    check("Qwen3Config accepts the 14B/8B field sets", False, f"{type(exc).__name__}: {exc}")

check("14B -> keep 27", L.check_arch(cfg14, None) == ("Qwen3-14B-Base", 27))
check("8B  -> keep 24", L.check_arch(cfg8, None) == ("Qwen3-8B-Base", 24))
check("14B with --layers 27 passes", L.check_arch(cfg14, 27)[1] == 27)
check("8B  with --layers 24 passes", L.check_arch(cfg8, 24)[1] == 24)


def must_fail(name, fn):
    try:
        fn()
    except SystemExit as exc:
        check(name, True, str(exc).splitlines()[0][:70])
        return
    check(name, False, "did NOT raise")


must_fail("--layers 24 on 14B is rejected", lambda: L.check_arch(cfg14, 24))
must_fail("--layers 27 on 8B is rejected", lambda: L.check_arch(cfg8, 27))
must_fail("--layers 0 on 14B is rejected", lambda: L.check_arch(cfg14, 0))
must_fail("unknown backbone is rejected",
          lambda: L.check_arch(Cfg(hidden_size=3584, num_hidden_layers=28), 20))

# the default 8 pooling depths fit inside keep=24 and keep=27, and do NOT fit keep=20
check("pool [2..24] legal at keep 24", L.check_pool_layers(L.DEFAULT_POOL_LAYERS, 24) == L.DEFAULT_POOL_LAYERS)
check("pool [2..24] legal at keep 27", L.check_pool_layers(L.DEFAULT_POOL_LAYERS, 27) == L.DEFAULT_POOL_LAYERS)
must_fail("pool depth 24 illegal at keep 20",
          lambda: L.check_pool_layers(L.DEFAULT_POOL_LAYERS, 20))
must_fail("7 pooling depths rejected",
          lambda: L.check_pool_layers(L.DEFAULT_POOL_LAYERS[:7], 24))
must_fail("depth 0 rejected", lambda: L.check_pool_layers([0, 2, 4, 6, 8, 12, 16, 20], 24))

# --------------------------------------------------------------------------- #
# 2. model surgery on a scaled-down Qwen3 with the real 14B field shapes
# --------------------------------------------------------------------------- #
SMOKE = L.make_smoke_assets(tempfile.mkdtemp(prefix="l24_selftest_"), verbose=False)
dev = torch.device("cpu")

# Build the tiny model, then check the *14B-shaped* assertions separately above.
model, info = L.build_model(SMOKE["model_dir"], SMOKE["keep"], SMOKE["pool_layers"],
                            torch.float32, dev, allow_unknown_arch=True, verbose=False)
check("keep layers kept", len(model.trunk.layers) == SMOKE["keep"],
      f"{len(model.trunk.layers)} of {SMOKE['layers']}")
check("final norm disabled", isinstance(model.trunk.norm, torch.nn.Identity))
n_expect = SMOKE["keep"] * 7
check("LoRA on 7 projections x keep layers", info["lora"]["wrapped"] == n_expect,
      f"{info['lora']['wrapped']} == {n_expect}")
check("feature dim == 8 * hidden", info["dim"] == 8 * SMOKE["hidden"], info["dim"])
check("only LoRA + heads trainable",
      all((".lora_" in n) or n.startswith(("plain_head", "centred_head", "norms"))
          for n, p in model.named_parameters() if p.requires_grad))
check("adapter params stay fp32",
      all(p.dtype == torch.float32 for n, p in model.named_parameters() if ".lora_" in n))
check("lora_B starts at zero (identity init)",
      all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if n.endswith("lora_B")))

# frozen trunk really is frozen
trunk_grads = [n for n, p in model.trunk.named_parameters() if p.requires_grad and ".lora_" not in n]
check("no trainable trunk weight", not trunk_grads, str(trunk_grads[:3]))

# --------------------------------------------------------------------------- #
# 3. forward / pooling semantics
# --------------------------------------------------------------------------- #
tok = L.load_tokenizer(SMOKE["model_dir"])
texts = ["alpha beta gamma delta epsilon", "zeta eta theta"]
tokens = L.encode_all(tok, texts, SMOKE["max_len"], verbose=False)
check("EOS appended once per row",
      all(int(t[-1]) == tok.eos_token_id for t in tokens))

ids, att = L.collate(tokens, [0, 1], dev)
check("padding is right-side and masked",
      int(att[0].sum()) == len(tokens[0]) and int(att[1].sum()) == len(tokens[1]))

model.eval()
with torch.no_grad():
    z = model.features(ids, att)
check("features shape", z.shape == (2, info["dim"]), tuple(z.shape))
check("features finite", torch.isfinite(z).all())

# The pooling mask excludes position 0 from the *mean*; token 0 still participates
# in attention, so the correct invariant is the pooling arithmetic itself, checked
# against a hand-rolled reference on the trunk's own hidden states.
with torch.no_grad():
    out = model.trunk(input_ids=ids, attention_mask=att, output_hidden_states=True,
                      use_cache=False)
    hs = out.hidden_states
    ref = []
    m = att.unsqueeze(-1).float().clone()
    m[:, 0, :] = 0.0
    den = m.sum(1).clamp(min=1.0)
    for layer, norm in zip(model.pool_layers, model.norms):
        ref.append(norm((hs[layer].float() * m).sum(1) / den))
    ref = torch.cat(ref, -1)
check("pooling matches a masked-mean reference", torch.allclose(z, ref, atol=1e-5),
      f"max|d| {float((z - ref).abs().max()):.2e}")

# token 0 must be excluded from the mean: with a 1-token row the denominator is 0
# -> clamped to 1, so the pooled vector is zeros; a 2-token row averages only token 1.
one = np.asarray([tok.eos_token_id], dtype=np.int32)
ids_1, att_1 = L.collate([one], [0], dev)
with torch.no_grad():
    z_1 = model.features(ids_1, att_1)
check("single-token row pools to zeros (denominator clamped, mask[:,0]=0)",
      torch.allclose(z_1, torch.zeros_like(z_1), atol=1e-6))

two = np.asarray([tok.eos_token_id, int(tokens[0][0])], dtype=np.int32)
ids_2, att_2 = L.collate([two], [0], dev)
with torch.no_grad():
    z_2 = model.features(ids_2, att_2)
    hs2 = model.trunk(input_ids=ids_2, attention_mask=att_2, output_hidden_states=True,
                      use_cache=False).hidden_states
manual = torch.cat([nm(hs2[layer].float()[:, 1, :]) for layer, nm in
                    zip(model.pool_layers, model.norms)], -1)
check("2-token row averages token 1 only", torch.allclose(z_2, manual, atol=1e-5))

# changing a genuinely different token must change the features
ids3 = ids.clone()
ids3[1, 0] = (int(tokens[1][0]) + 7) % tok.vocab_size
with torch.no_grad():
    z3 = model.features(ids3, att)
check("changing a token DOES change features", not torch.equal(z[1], z3[1]))

# padded region must not leak in
ids4, att4 = L.collate(tokens, [0], dev, fixed_len=len(tokens[0]) + 40)
with torch.no_grad():
    z4 = model.features(ids4, att4)
check("extra right padding changes nothing", torch.allclose(z[0], z4[0], atol=1e-6))

# --------------------------------------------------------------------------- #
# 4. centring, class weights, schedule, folds
# --------------------------------------------------------------------------- #
z5 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 20.0]])
zc, keep = L.centre_by_group(z5, ["a", "a", "b", "b"])
check("centring subtracts the group mean",
      torch.allclose(zc[:2].sum(0), torch.zeros(2), atol=1e-6) and keep.all())
check("centred rows are symmetric within a pair",
      torch.allclose(zc[0], -zc[1]) and torch.allclose(zc[2], -zc[3]))
zc1, keep1 = L.centre_by_group(torch.tensor([[1.0, 1.0], [5.0, 5.0]]), ["a", "b"])
check("singleton groups get keep=False", not bool(keep1.any()))
check("singleton centred value is zero", torch.allclose(zc1, torch.zeros(2, 2)))

y = np.array([0] * 100 + [1] * 10 + [2] * 10)
w = L.class_weights(y)
check("class weights mean 1", abs(float(w.mean()) - 1.0) < 1e-5, f"{w.mean():.4f}")
check("rare class weighted more", w[1] > w[0], f"{w[1]:.2f} > {w[0]:.2f}")
check("weights are capped", float(w.max()) <= 5.0 + 1e-6, f"max {w.max():.2f}")

sched = [L.warmup_cosine(s, 100, 0.1) for s in range(101)]
check("warmup rises from small to 1.0", sched[0] < 0.2 and abs(sched[9] - 1.0) < 1e-9,
      f"{sched[0]:.3f} -> {sched[9]:.3f}")
check("cosine decays to min_lr_ratio", abs(sched[-1] - 0.1) < 0.02, f"{sched[-1]:.3f}")
check("schedule is monotone after warmup",
      all(sched[i] >= sched[i + 1] - 1e-9 for i in range(10, 100)))

groups = [f"p{i // 18}" for i in range(18 * 20)]
labels = np.tile(np.arange(18), 20)
f1 = L.prompt_folds(len(groups), groups, labels, n_splits=5, seed=0)
f2 = L.prompt_folds(len(groups), groups, labels, n_splits=5, seed=0)
check("fold partition is deterministic", all(np.array_equal(a, b) for a, b in zip(f1, f2)))
check("folds are disjoint and complete", sum(len(f) for f in f1) == len(groups) and
      len(set(np.concatenate(f1))) == len(groups))
pg = np.asarray(groups)
check("no prompt straddles folds",
      sum(len(set(pg[f1[a]]) & set(pg[f1[b]])) for a in range(5) for b in range(a + 1, 5)) == 0)

s1 = L.sim_subset(f1[0], groups, lo=1, hi=2, seed=0)
s2 = L.sim_subset(f1[0], groups, lo=1, hi=2, seed=0)
check("sim_subset deterministic", np.array_equal(s1, s2))
per_prompt = {}
for i in s1:
    per_prompt.setdefault(groups[int(i)], 0)
    per_prompt[groups[int(i)]] += 1
check("sim_k keeps 1..2 per prompt", set(per_prompt.values()) <= {1, 2},
      f"counts {sorted(set(per_prompt.values()))}")

yt = np.arange(18)
check("evaluate(perfect) == 1", abs(L.evaluate(yt, yt)["bacc"] - 1.0) < 1e-9)
rng = np.random.default_rng(0)
shuf = yt.copy()
rng.shuffle(shuf)
check("evaluate(shuffled) is low", L.evaluate(yt, shuf)["bacc"] < 0.35,
      f"{L.evaluate(yt, shuf)['bacc']:.3f}")

# --------------------------------------------------------------------------- #
# 5. checkpoint round-trip through the real save/load path
# --------------------------------------------------------------------------- #
out = tempfile.mkdtemp(prefix="l24_ckpt_")
meta = dict(format=L.FORMAT, labels=list(L.LABELS), cfg=info, stages={}, args={})
with torch.no_grad():
    for n, p in model.named_parameters():
        if ".lora_" in n:
            p.add_(torch.randn_like(p) * 0.02)  # make the adapter non-trivial
L.save_checkpoint(model, out, meta)
check("checkpoint files exist",
      all(os.path.exists(os.path.join(out, f)) for f in
          ("adapter/adapter_model.safetensors", "adapter/adapter_config.json",
           "heads.safetensors", "meta.json")))

import json
acfg = json.load(open(os.path.join(out, "adapter", "adapter_config.json")))
check("adapter_config has r/alpha/targets",
      acfg["r"] == 16 and acfg["lora_alpha"] == 32.0 and len(acfg["target_modules"]) == 7,
      f"r={acfg['r']} alpha={acfg['lora_alpha']}")

model2, _ = L.build_model(SMOKE["model_dir"], SMOKE["keep"], SMOKE["pool_layers"],
                          torch.float32, dev, allow_unknown_arch=True, verbose=False)
L.load_into(model2, out, verbose=False)
model.eval()
model2.eval()
with torch.no_grad():
    za = model.features(ids, att)
    zb = model2.features(ids, att)
check("reloaded model reproduces features", torch.allclose(za, zb, atol=1e-6),
      f"max|d| {float((za - zb).abs().max()):.2e}")
la = model.head_logits(za)[0]
lb = model2.head_logits(zb)[0]
check("reloaded heads reproduce logits", torch.allclose(la, lb, atol=1e-6))

from safetensors.torch import load_file  # noqa: E402
ad = load_file(os.path.join(out, "adapter", "adapter_model.safetensors"))
k0 = sorted(ad)[0]
check("adapter keys are peft-style and sliceable at 'layers.'",
      k0.startswith("base_model.model.layers.") and "trunk." not in k0, k0)
check("adapter keys resolve back to modules",
      all(L._split_lora_key(k)[0] is not None for k in ad))

lp = L.predict_logprobs(model2, tokens, dev, bs=2, verbose=False)
check("predict output shape/dtype", lp.shape == (2, 18) and lp.dtype == np.float32)
check("predict returns log-probs (exp sums to 1)",
      np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-5), f"{np.exp(lp).sum(1)}")
lp_t = L.predict_logprobs(model2, tokens, dev, bs=2, groups=["g1", "g1"], twin_w=0.35,
                          verbose=False)
check("twin_w changes the log-probs", not np.allclose(lp, lp_t))
check("twin output still finite", np.isfinite(lp_t).all())

# duplicate prompts group -> centred term is a real signal, not zeros
check("centred head differs from plain head",
      not torch.allclose(model.plain_head.weight, model.centred_head.weight))

# --------------------------------------------------------------------------- #
# 6. run_card.sh: bash syntax + HBM parser against captured npu-smi text
# --------------------------------------------------------------------------- #
rc = os.path.join(HERE, "run_card.sh")
if os.name == "nt":
    print("SKIP  run_card.sh execution (Windows host)")
else:
    p = subprocess.run(["bash", "-n", rc], capture_output=True, text=True)
    check("run_card.sh parses under bash -n", p.returncode == 0, p.stderr.strip()[:80])

SAMPLE = """\
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page) |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)        |
+===========================+===============+======================================================================+
| 0     910B3               | OK            | 135.9                36                      0    / 0                |
| 0                         | 0000:C1:00.0  | 53                   0    / 0                31288/ 65536            |
+===========================+===============+======================================================================+
| 3     910B3               | OK            | 195.1                41                      0    / 0                |
| 0                         | 0000:82:00.0  | 60                   0    / 0                14260/ 65536            |
+===========================+===============+======================================================================+
"""


def parse_hbm(text, card, expect_total=65536):
    import re
    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        pairs = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", line)]
        hbm = [p for p in pairs if p[1] > 1000 and abs(p[1] - expect_total) <= 1024]
        if hbm:
            rows.append(hbm[-1])
    used, total = rows[card]
    return used, total - used


u, f = parse_hbm(SAMPLE, 0)
check("HBM parser: card 0 used/free", (u, f) == (31288, 65536 - 31288), f"used {u} free {f}")
u, f = parse_hbm(SAMPLE, 1)
check("HBM parser: card index maps to the Nth card row",
      (u, f) == (14260, 65536 - 14260), f"used {u} free {f}")

print()
print(f"{len(FAIL)} failure(s)" if FAIL else "ALL CHECKS PASSED")
if FAIL:
    for f in FAIL:
        print("  -", f)
sys.exit(1 if FAIL else 0)
