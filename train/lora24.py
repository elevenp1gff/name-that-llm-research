"""Shared core for the two-stage truncated-Qwen3 LoRA classifier ("LoRA24" recipe).

Why this file exists
--------------------
train_lora.py / predict_lora.py / oof_lora.py must agree *exactly* on the model
surgery, the pooling mask and the checkpoint layout.  Keeping that in one module
is the only way to guarantee the inference path reproduces training.

Deliberate choices
------------------
* **No peft.**  The NPU env has peft 0.21.2 installed but `import peft` fails
  (`No module named 'einops'`, raised from transformers.models.bloom).  A
  hand-rolled wrapper is 40 lines, has no version coupling, and produces the
  identical graph on npu / cuda / cpu.  Adapter keys are still written in peft
  naming so public merge code keeps working.
* **No autocast.**  The trunk is cast to bf16 once; LoRA A/B stay fp32 and are
  cast to the activation dtype inside forward (same numerics as autocast, one
  fewer NPU op pattern to go wrong).  Pooled features and heads are fp32.
* **No DataLoader / no workers.**  Tokens are pre-tokenised into RAM and batches
  are assembled by index.  Training on NFS with forked workers on Ascend is a
  known source of mystery hangs; this sidesteps it and avoids any need for
  flock-based sharding.

Checkpoint layout written by :func:`save_checkpoint`::

    out/
      adapter/adapter_model.safetensors   # peft-style keys, fp32
      adapter/adapter_config.json
      heads.safetensors                   # plain.*, centered.*, norms.{i}.*
      meta.json                           # everything needed to rebuild
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "ntl-lora24-v1"

#: Canonical label order -- identical to corpus/cv.py LABELS (competition order).
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

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: 8 pooling depths of the public recipe.  Index i of hidden_states == output of
#: decoder layer i-1, so index 24 needs keep >= 24.
DEFAULT_POOL_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]

#: (hidden_size, num_hidden_layers) -> (name, expected truncated layer count)
EXPECTED_ARCH = {
    (5120, 40): ("Qwen3-14B-Base", 27),
    (4096, 36): ("Qwen3-8B-Base", 24),
}

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


# --------------------------------------------------------------------------- #
# device / seeding
# --------------------------------------------------------------------------- #
def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Prefer NPU, then CUDA, then CPU.

    Refuses to fall back silently to CPU when an Ascend card was requested via
    ASCEND_RT_VISIBLE_DEVICES -- that failure mode otherwise looks like a very
    slow training run instead of a setup error.
    """
    if prefer:
        return torch.device(prefer)
    npu_err = None
    try:
        import torch_npu  # noqa: F401  (registers the "npu" backend)

        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception as exc:  # pragma: no cover - depends on host
        npu_err = exc
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") and npu_err is not None:
        raise SystemExit(
            "[device] ASCEND_RT_VISIBLE_DEVICES is set but torch_npu is unusable:\n"
            f"  {type(npu_err).__name__}: {npu_err}\n"
            "  Every python invocation must first:\n"
            "    source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        )
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def device_type(device: torch.device) -> str:
    return device.type


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# model surgery
# --------------------------------------------------------------------------- #
def read_config(base_dir: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(base_dir)


def check_arch(cfg, keep: Optional[int], allow_unknown: bool = False) -> Tuple[str, int]:
    """Assert the backbone matches the published recipe; return (name, keep).

    Fails loudly rather than training a wrong-width trunk for 20 hours: a
    `--layers 24` against a 14B config, or any unfamiliar Qwen variant, is a
    mistake until proven otherwise.
    """
    hidden = int(cfg.hidden_size)
    total = int(cfg.num_hidden_layers)
    if getattr(cfg, "model_type", "") == "qwen3_moe":
        # MoE backbones (Qwen3-30B-A3B): 48 layers, hidden 2048, 128 experts.
        expect = max(1, int(round(total * 0.67)))
        if keep is None:
            keep = expect
        if keep > total:
            raise SystemExit(f"[arch] --layers {keep} outside 1..{total}")
        return f"Qwen3-30B-A3B-Base(MoE)", int(keep)
    known = EXPECTED_ARCH.get((hidden, total))
    if known is None:
        if not allow_unknown:
            pretty = ", ".join(
                f"hidden={h}/layers={n} -> {nm} keep={k}" for (h, n), (nm, k) in EXPECTED_ARCH.items()
            )
            raise SystemExit(
                f"[arch] unrecognised backbone: hidden_size={hidden}, num_hidden_layers={total}, "
                f"model_type={getattr(cfg, 'model_type', '?')}.\n"
                f"       known: {pretty}\n"
                f"       Pass --allow_unknown_arch only if you really mean it "
                f"(pooling depths and layer truncation were tuned for Qwen3)."
            )
        name, expect = f"unknown(hidden={hidden},layers={total})", keep
    else:
        name, expect = known[0], known[1]
        if keep is not None and int(keep) != expect:
            raise SystemExit(
                f"[arch] {name}: the LoRA24 recipe truncates to {expect} of {total} layers, "
                f"but --layers {keep} was given.\n"
                f"       Use --layers {expect} (or --allow_unknown_arch to override)."
            )
    keep = int(expect if keep is None else keep)
    if not (1 <= keep <= total):
        raise SystemExit(f"[arch] --layers {keep} outside 1..{total}")
    return name, keep


def check_pool_layers(pool_layers: Sequence[int], keep: int) -> List[int]:
    layers = [int(x) for x in pool_layers]
    if len(layers) != 8:
        raise SystemExit(f"[arch] the recipe pools 8 depths, got {len(layers)}: {layers}")
    if len(set(layers)) != len(layers):
        raise SystemExit(f"[arch] duplicate pooling depths: {layers}")
    if min(layers) < 1:
        raise SystemExit(f"[arch] pooling depth 0 is the embedding output; use >= 1: {layers}")
    if max(layers) > keep:
        raise SystemExit(
            f"[arch] pooling depth {max(layers)} needs layer index <= keep={keep} "
            f"(hidden_states[{keep}] is the last available when truncating to {keep})."
        )
    return sorted(layers)


class LoRALinear(nn.Module):
    """y = W x + b + dropout(x) A^T B^T * alpha/r, W/b frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("lora r must be > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.dropout_p = float(dropout)
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)
        dt = x.dtype
        delta = (x @ self.lora_A.to(dt).t()) @ self.lora_B.to(dt).t()
        return out + delta * self.scale

    def extra_repr(self) -> str:
        return f"r={self.r}, scale={self.scale:g}, p={self.dropout_p:g}"


def inject_lora(model: nn.Module, keep: int, r: int, alpha: float, dropout: float,
                targets: Sequence[str] = TARGET_MODULES) -> int:
    """Wrap projections inside the first `keep` decoder layers. Returns count.

    For sparse-MoE layers the FFN is a Qwen3MoeSparseMoeBlock: `mlp.gate` is the
    router Linear and the real FFNs are `mlp.experts.<i>.{gate,up,down}_proj`. Putting
    LoRA on 128 experts x keep layers is not affordable, so on MoE we adapt the
    attention projections + the router only and leave every expert frozen.
    """
    layers = model.layers
    n = 0
    for i in range(keep):
        layer = layers[i]
        is_moe = hasattr(layer.mlp, "experts")
        holders = {
            "q_proj": layer.self_attn, "k_proj": layer.self_attn, "v_proj": layer.self_attn,
            "o_proj": layer.self_attn,
        }
        names = [t for t in targets if t in holders]
        if not is_moe:
            holders.update({"gate_proj": layer.mlp, "up_proj": layer.mlp,
                            "down_proj": layer.mlp})
            names = list(targets)
        for name in names:
            holder = holders[name]
            cur = getattr(holder, name)
            if not isinstance(cur, nn.Linear):
                raise SystemExit(f"[lora] layers.{i}.{name} is {type(cur).__name__}, expected nn.Linear")
            setattr(holder, name, LoRALinear(cur, r, alpha, dropout))
            n += 1
        if is_moe and hasattr(layer.mlp, "gate") and isinstance(layer.mlp.gate, nn.Linear):
            layer.mlp.gate = LoRALinear(layer.mlp.gate, r, alpha, dropout)
            n += 1
    return n


def freeze_trunk(model: nn.Module) -> int:
    """Freeze everything in the backbone except the injected LoRA factors.

    `from_pretrained` hands back every weight with requires_grad=True, including
    embed_tokens and the q/k RMSNorms. Leaving those trainable is both off-recipe
    and expensive (the 151936 x 5120 embedding would show up in AdamW state).
    """
    n = 0
    for name, p in model.named_parameters():
        if ".lora_" in name:
            continue
        if p.requires_grad:
            p.requires_grad_(False)
            n += 1
    return n


class PoolModel(nn.Module):
    """Truncated trunk -> masked mean at 8 depths -> per-depth LayerNorm -> concat -> 2 heads."""

    def __init__(self, trunk: nn.Module, pool_layers: Sequence[int], hidden: int,
                 n_labels: int, drop_first_token: bool = True):
        super().__init__()
        self.trunk = trunk
        self.pool_layers = [int(x) for x in pool_layers]
        self.hidden = int(hidden)
        self.n_labels = int(n_labels)
        self.drop_first_token = bool(drop_first_token)
        # fp32: pooling LayerNorms are part of the trained head, not the trunk.
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in self.pool_layers])
        dim = self.hidden * len(self.pool_layers)
        self.plain_head = nn.Linear(dim, self.n_labels)
        self.centred_head = nn.Linear(dim, self.n_labels)

    # -- features ---------------------------------------------------------- #
    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        need = max(self.pool_layers)
        if len(hs) <= need:
            raise RuntimeError(
                f"trunk returned {len(hs)} hidden states, pooling depth {need} unavailable "
                f"(keep={self.trunk.config.num_hidden_layers})"
            )
        mask = attention_mask.unsqueeze(-1).float().clone()
        if self.drop_first_token:
            mask[:, 0, :] = 0.0  # token 0 is the attention sink, not content
        den = mask.sum(1).clamp(min=1.0)
        feats = []
        for layer, norm in zip(self.pool_layers, self.norms):
            summed = (hs[layer].float() * mask).sum(1) / den
            feats.append(norm(summed))
        return torch.cat(feats, -1)

    # -- heads ------------------------------------------------------------- #
    def head_logits(self, z: torch.Tensor, z_centred: Optional[torch.Tensor] = None):
        plain = self.plain_head(z)
        centred = self.centred_head(z if z_centred is None else z_centred)
        return plain, centred


def build_model(base_dir: str, keep: Optional[int], pool_layers: Sequence[int],
                dtype: torch.dtype, device: torch.device, lora_r: int = 16,
                lora_alpha: float = 32.0, lora_dropout: float = 0.05,
                n_labels: int = len(LABELS), no_final_norm: bool = True,
                allow_unknown_arch: bool = False, verbose: bool = True):
    """Load `keep` decoder layers of `base_dir`, inject LoRA, attach heads.

    Returns (model, info).  `info` is the dict that lands in meta.json["cfg"].
    """
    from transformers import AutoModel

    cfg = read_config(base_dir)
    name, keep = check_arch(cfg, keep, allow_unknown_arch)
    pool_layers = check_pool_layers(pool_layers, keep)

    total_layers = int(cfg.num_hidden_layers)
    cfg.num_hidden_layers = keep
    cfg.use_cache = False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = list(cfg.layer_types)[:keep]
    if getattr(cfg, "max_window_layers", None) is not None:
        cfg.max_window_layers = min(int(cfg.max_window_layers), keep)

    t0 = time.time()
    kwargs = dict(config=cfg, low_cpu_mem_usage=True, output_loading_info=True)
    try:
        trunk, loading = AutoModel.from_pretrained(base_dir, torch_dtype=dtype, **kwargs)
    except TypeError:  # transformers >= 5 renamed torch_dtype -> dtype
        trunk, loading = AutoModel.from_pretrained(base_dir, dtype=dtype, **kwargs)
    if loading.get("missing_keys"):
        raise SystemExit(f"[load] missing weights in {base_dir}: {loading['missing_keys'][:8]}")
    if no_final_norm:
        trunk.norm = nn.Identity()  # checkpoints were trained with the final norm off
    trunk.to(device)

    n_wrapped = inject_lora(trunk, keep, lora_r, lora_alpha, lora_dropout)
    n_frozen = freeze_trunk(trunk)
    model = PoolModel(trunk, pool_layers, int(cfg.hidden_size), n_labels).to(device)
    # heads/norms were created on CPU by nn.Module children order -> move them too.
    model.norms.to(device, dtype=torch.float32)
    model.plain_head.to(device, dtype=torch.float32)
    model.centred_head.to(device, dtype=torch.float32)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    info = dict(
        base=os.path.abspath(base_dir),
        arch=name,
        model_type=str(getattr(cfg, "model_type", "")),
        total_layers=total_layers,
        keep=keep,
        layers=pool_layers,
        hidden=int(cfg.hidden_size),
        n_heads=int(getattr(cfg, "num_attention_heads", 0)),
        intermediate_size=int(getattr(cfg, "intermediate_size", 0)),
        n_labels=n_labels,
        dim=int(cfg.hidden_size) * len(pool_layers),
        dtype=str(dtype).replace("torch.", ""),
        no_final_norm=bool(no_final_norm),
        drop_first_token=True,
        lora=dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                  targets=list(TARGET_MODULES), wrapped=n_wrapped, dtype="float32",
                  frozen=bool(n_frozen)),
        trainable_params=int(n_train),
        total_params=int(n_total),
        built_in_s=round(time.time() - t0, 1),
    )
    if verbose:
        print(f"[model] {name}: {keep}/{total_layers} layers, hidden {cfg.hidden_size}, "
              f"pool {pool_layers}, dim {info['dim']}, lora modules {n_wrapped}, "
              f"frozen {n_frozen}, trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M "
              f"({time.time() - t0:.0f}s)", flush=True)
    return model, info


# --------------------------------------------------------------------------- #
# tokenisation / batching
# --------------------------------------------------------------------------- #
def load_tokenizer(base_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode_all(tok, texts: Sequence[str], max_len: int, chunk: int = 20000,
               verbose: bool = True) -> List[np.ndarray]:
    """Tokenise once into a list of int32 arrays, each ending with one EOS.

    add_special_tokens=False + explicit EOS matches the reference inference code.
    """
    eos = tok.eos_token_id
    out: List[np.ndarray] = []
    t0 = time.time()
    for s in range(0, len(texts), chunk):
        batch = [str(t) for t in texts[s:s + chunk]]
        enc = tok(batch, add_special_tokens=False, truncation=True, max_length=max_len - 1)["input_ids"]
        out.extend(np.asarray(list(x) + [eos], dtype=np.int32) for x in enc)
        if verbose:
            done = min(s + chunk, len(texts))
            print(f"\r[tokenise] {done}/{len(texts)} ({time.time() - t0:.0f}s)", end="", flush=True)
    if verbose and texts:
        print(f"\r[tokenise] {len(texts)} texts in {time.time() - t0:.0f}s", flush=True)
    return out


def length_stats(tokens: Sequence[np.ndarray]) -> dict:
    lens = np.asarray([len(t) for t in tokens])
    if not lens.size:
        return {}
    return {k: int(v) for k, v in zip(
        ("n", "min", "p50", "p90", "p99", "max"),
        (lens.size, lens.min(), np.percentile(lens, 50), np.percentile(lens, 90),
         np.percentile(lens, 99), lens.max()))}


def collate(tokens: Sequence[np.ndarray], idx: Sequence[int], device: torch.device,
            fixed_len: Optional[int] = None):
    seqs = [tokens[i] for i in idx]
    width = fixed_len or max(len(s) for s in seqs)
    ids = np.zeros((len(seqs), width), dtype=np.int64)
    att = np.zeros((len(seqs), width), dtype=np.int64)
    for j, s in enumerate(seqs):
        n = min(len(s), width)
        ids[j, :n] = s[:n]
        att[j, :n] = 1
    return (torch.from_numpy(ids).to(device), torch.from_numpy(att).to(device))


class GroupSampler:
    """Batches assembled as same-prompt groups, sized to `n_seq` passages.

    Why groups rather than rows: the centred head needs >= 2 passages from one
    prompt, and the reference recipe trains on triplets of same-prompt answers.

    Why not exactly 3 per group: `union_capped.parquet` has 54,151 prompts and
    51% of them are singletons. A strict triplet sampler would silently discard
    half the corpus. Here a group contributes min(3, size) distinct passages, and
    groups are drawn until the batch is full -- so singleton prompts still train
    the plain head and simply carry no centred signal (centre_by_group masks
    them out, and ce_masked renormalises over the rows that do).

    Groups are drawn with probability proportional to size **capped at 3**, so a
    prompt with 4,441 answers cannot monopolise the batches while big prompts
    still contribute their diversity.
    """

    def __init__(self, groups: Sequence, n_seq: int, seed: int = 0, max_per_group: int = 3,
                 verbose: bool = True):
        rng = np.random.default_rng(seed)
        by_group: dict = {}
        for i, g in enumerate(groups):
            by_group.setdefault(str(g), []).append(i)
        pool = {g: np.asarray(v, dtype=np.int64) for g, v in by_group.items()}
        keys = np.asarray(list(pool.keys()), dtype=object)
        weight = np.asarray([min(len(pool[str(k)]), max_per_group) for k in keys], dtype=np.float64)
        weight /= weight.sum()
        self.pool, self.keys, self.weight = pool, keys, weight
        self.n_seq, self.max_per_group, self.rng = int(n_seq), int(max_per_group), rng
        if verbose:
            n_single = sum(1 for v in pool.values() if len(v) == 1)
            n_rows = sum(len(v) for v in pool.values())
            used = sum(min(len(v), max_per_group) for v in pool.values())
            print(f"[sampler] {len(pool)} prompts / {n_rows} rows; "
                  f"{n_single} singleton prompts ({100.0*n_single/max(1,len(pool)):.0f}%) "
                  f"reach the plain head only; effective diversity {used} "
                  f"({100.0*used/max(1,n_rows):.0f}% of rows)", flush=True)

    def batch(self) -> Tuple[List[int], List[str]]:
        idx: List[int] = []
        gids: List[str] = []
        guard = 0
        while len(idx) < self.n_seq and guard < 1000:
            guard += 1
            k = str(self.keys[self.rng.choice(len(self.keys), p=self.weight)])
            rows = self.pool[k]
            take = self.rng.choice(rows, size=min(self.max_per_group, rows.size), replace=False)
            allow = self.n_seq - len(idx)
            for r in take[:allow]:
                idx.append(int(r))
                gids.append(k)
        return idx, gids


def centre_by_group(z: torch.Tensor, gids: Sequence[str]):
    """Subtract each group's mean feature. Returns (z_centred, keep_mask)."""
    uniq, inv = np.unique(np.asarray([str(g) for g in gids]), return_inverse=True)
    gi = torch.from_numpy(inv.astype(np.int64)).to(z.device)
    sums = torch.zeros(uniq.size, z.shape[1], dtype=z.dtype, device=z.device)
    sums.index_add_(0, gi, z)
    counts = torch.zeros(uniq.size, dtype=z.dtype, device=z.device)
    counts.index_add_(0, gi, torch.ones_like(gi, dtype=z.dtype))
    means = sums / counts.clamp(min=1.0).unsqueeze(1)
    keep = counts[gi] >= 2.0
    return z - means[gi], keep


# --------------------------------------------------------------------------- #
# trainable parameter groups / optimiser / schedule
# --------------------------------------------------------------------------- #
def trainable_groups(model: nn.Module, lr_lora: float, lr_head: float,
                     wd_lora: float = 0.0, wd_head: float = 0.01):
    lora, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if ".lora_" in name else head).append(p)
    return [
        dict(params=lora, lr=lr_lora, weight_decay=wd_lora, name="lora"),
        dict(params=head, lr=lr_head, weight_decay=wd_head, name="head"),
    ]


def warmup_cosine(step: int, total: int, warmup_frac: float, min_ratio: float = 0.1) -> float:
    if total <= 1:
        return 1.0
    warm = max(1, int(round(total * warmup_frac)))
    if step < warm:
        return (step + 1) / warm
    span = max(1, total - warm)
    t = min(1.0, (step - warm) / span)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def class_weights(labels: np.ndarray, n_labels: int = len(LABELS), power: float = 1.0,
                  cap: float = 5.0) -> np.ndarray:
    """Balanced CE weights N/(K*n_c), raised to `power`, capped, mean-normalised."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n_labels * counts)) ** float(power)
    w = np.clip(w, 1.0 / cap, cap)
    w = w / w.mean()
    return w.astype(np.float32)


def ce_masked(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor],
              keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    per = F.cross_entropy(logits, target, weight=weight, reduction="none", label_smoothing=0.0)
    if keep is not None:
        denom = keep.sum().clamp(min=1.0)
        return (per * keep).sum() / denom
    return per.mean()


# --------------------------------------------------------------------------- #
# data loading (pyarrow only)
# --------------------------------------------------------------------------- #
# pandas is deliberately avoided: on this task's inputs it buys nothing (we need
# three columns and string filtering) and it drags in a numpy-ABI-sensitive
# optional-dependency chain. pyarrow is a hard requirement of parquet anyway and
# the server has 21.0.0.
def _arrow_table(path: str, cols: Sequence[str]):
    """Read only `cols` from a parquet/csv, with a clear error if a column is absent."""
    if not os.path.exists(path):
        raise SystemExit(f"[data] missing file: {path}")
    if path.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq

        try:
            have = pq.read_schema(path).names
        except Exception as exc:
            # An in-flight upload (scp/sftp mid-copy) has no parquet footer yet.
            # Failing here with "magic bytes not found" sends people hunting for a
            # corrupt dataset instead of a transfer still in progress.
            size = os.path.getsize(path) if os.path.exists(path) else -1
            raise SystemExit(
                f"[data] cannot read {path} as parquet ({type(exc).__name__}: {exc}).\n"
                f"       size on disk = {size} bytes. If a copy is still running, wait for "
                f"it to finish (this error is 'no footer yet', not corruption)."
            )
        missing = [c for c in cols if c not in have]
        if missing:
            raise SystemExit(f"[data] {path} is missing column(s) {missing}; has {have}")
        # column projection matters: the pool parquet has 20+ columns / 100k+ rows.
        tbl = pq.read_table(path, columns=list(cols))
    else:
        import pyarrow.csv as pv

        tbl = pv.read_csv(path)
        missing = [c for c in cols if c not in tbl.schema.names]
        if missing:
            raise SystemExit(f"[data] {path} is missing column(s) {missing}; "
                             f"has {tbl.schema.names}")
        tbl = tbl.select(list(cols))
    return tbl


def _col_strings(tbl, name: str) -> List[str]:
    """Column -> list[str], with nulls as empty strings."""
    col = tbl.column(name)
    return [("" if v is None else str(v)) for v in col.to_pylist()]


def read_table(path: str, need_prompt: bool = True, need_label: bool = True,
               limit: int = 0, seed: int = 0, min_words: int = 0, max_words: int = 0,
               verbose: bool = True):
    """Read a parquet/csv with columns text[, model][, prompt_group].

    Returns dict(texts=list[str], labels=int64 ndarray or None, groups=list[str] or None).
    Rows with an unknown label are dropped, counted and reported -- never silently.
    """
    cols = ["text"] + (["model"] if need_label else []) + (["prompt_group"] if need_prompt else [])
    tbl = _arrow_table(path, cols)
    texts = _col_strings(tbl, "text")
    models = _col_strings(tbl, "model") if need_label else None
    groups = _col_strings(tbl, "prompt_group") if need_prompt else None

    lut = {lab: i for i, lab in enumerate(LABELS)}
    keep_idx: List[int] = []
    dropped_blank = 0
    dropped_label = 0
    unknown_examples: dict = {}
    for i, txt in enumerate(texts):
        if not txt.strip():
            dropped_blank += 1
            continue
        if min_words or max_words:
            n_words = len(txt.split())
            if (min_words and n_words < min_words) or (max_words and n_words > max_words):
                continue
        if need_label:
            lab = (models[i] or "").strip()
            if lab not in lut:
                dropped_label += 1
                unknown_examples[lab] = unknown_examples.get(lab, 0) + 1
                continue
        keep_idx.append(i)

    if verbose and dropped_blank:
        print(f"[data] dropped {dropped_blank} row(s) with empty text", flush=True)
    if verbose and dropped_label:
        top = dict(sorted(unknown_examples.items(), key=lambda kv: -kv[1])[:5])
        print(f"[data] dropped {dropped_label} row(s) whose label is not one of the 18: {top}",
              flush=True)

    labels = np.asarray([lut[(models[i] or "").strip()] for i in keep_idx], dtype=np.int64) \
        if need_label else None
    groups = [groups[i] for i in keep_idx] if need_prompt else None
    texts = [texts[i] for i in keep_idx]

    if limit and len(texts) > limit:
        rs = np.random.default_rng(seed)
        sub = np.sort(rs.choice(len(texts), size=limit, replace=False))
        texts = [texts[i] for i in sub]
        labels = None if labels is None else labels[sub]
        groups = None if groups is None else [groups[i] for i in sub]
        if verbose:
            print(f"[data] --limit {limit}: subsampled {len(texts)} rows from {path}", flush=True)

    if not texts:
        raise SystemExit(f"[data] {path}: no usable rows after filtering")
    if verbose:
        extra = ""
        if labels is not None:
            present = np.bincount(labels, minlength=len(LABELS))
            nz = present[present > 0]
            extra = (f", labels {int(nz.size)}/18, per class min/max {int(nz.min())}/{int(nz.max())}"
                     f", max/min {nz.max()/max(1, nz.min()):.2f}")
        if groups is not None:
            extra += f", prompts {len(set(groups))}"
        print(f"[data] {os.path.basename(path)}: {len(texts)} rows{extra}", flush=True)
    return dict(texts=texts, labels=labels, groups=groups, path=os.path.abspath(path))


def write_parquet(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """Minimal dict-rows -> parquet writer used by the smoke-asset builder."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays = {c: [r[c] for r in rows] for c in columns}
    tbl = pa.table({c: pa.array(v) for c, v in arrays.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pq.write_table(tbl, path)


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def _peft_key(name: str) -> str:
    """trunk.layers.3.self_attn.q_proj.lora_A
    -> base_model.model.layers.3.self_attn.q_proj.lora_A.weight

    The `trunk.` prefix is dropped and the peft prefix added so the file is also
    loadable by the public merge code, which slices from ``layers.`` onward.
    """
    if name.startswith("trunk."):
        name = name[len("trunk."):]
    return "base_model.model." + name + ".weight"


def _our_key(key: str) -> str:
    """Inverse of :func:`_peft_key`.

    Keys on disk look like ``base_model.model.trunk.layers.3.self_attn.q_proj.lora_A.weight``.
    Only the ``layers.N...`` suffix is normative: the public merge code does
    ``name[name.index("layers."):]`` so the same file also merges into a bare
    HF trunk without `peft` or our PoolModel wrapper.
    """
    key = key.replace("base_model.model.", "")
    if key.endswith(".weight") and (".lora_A" in key or ".lora_B" in key):
        key = key[: -len(".weight")]
    marker = "layers."
    if not key.startswith("trunk.") and marker in key:
        key = "trunk." + key[key.index(marker):]
    return key


def _split_lora_key(key: str):
    """'...q_proj.lora_A' -> ('trunk...q_proj', 'lora_A'); None if not an adapter key."""
    name = _our_key(key)
    for suffix in (".lora_A", ".lora_B"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return None, None


def adapter_state(model: nn.Module) -> dict:
    return {_peft_key(n): p.detach().float().cpu().contiguous()
            for n, p in model.named_parameters() if ".lora_" in n}


def heads_state(model: PoolModel) -> dict:
    sd = {f"plain.{k}": v.detach().float().cpu() for k, v in model.plain_head.state_dict().items()}
    sd.update({f"centered.{k}": v.detach().float().cpu() for k, v in model.centred_head.state_dict().items()})
    for i, norm in enumerate(model.norms):
        for k, v in norm.state_dict().items():
            sd[f"norms.{i}.{k}"] = v.detach().float().cpu()
    return sd


def load_into(model: PoolModel, out_dir: str, verbose: bool = True) -> dict:
    """Load adapter + heads (+ meta) from an `out` dir into an existing model."""
    from safetensors.torch import load_file

    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        want = meta.get("cfg", {})
        got = dict(keep=len(model.trunk.layers), layers=model.pool_layers, hidden=model.hidden)
        for k in ("keep", "layers", "hidden"):
            if k in want and want[k] != got[k]:
                raise SystemExit(f"[load] {out_dir} was trained with {k}={want[k]}, model has {got[k]}")
    else:
        meta = {}

    mods = dict(model.named_modules())
    ad_path = os.path.join(out_dir, "adapter", "adapter_model.safetensors")
    if os.path.exists(ad_path):
        sd = load_file(ad_path)
        loaded = 0
        for key, val in sd.items():
            mod_name, which = _split_lora_key(key)
            if mod_name is None:
                raise SystemExit(f"[load] {ad_path}: unexpected adapter key {key}")
            mod = mods.get(mod_name)
            if mod is None or not isinstance(mod, LoRALinear):
                raise SystemExit(f"[load] adapter key {key} -> {mod_name} is not a LoRALinear")
            p = mod.lora_A if which == "lora_A" else mod.lora_B
            if tuple(p.shape) != tuple(val.shape):
                raise SystemExit(f"[load] shape mismatch for {mod_name}.{which}: "
                                 f"{tuple(p.shape)} vs {tuple(val.shape)}")
            with torch.no_grad():
                p.copy_(val.to(p.dtype))
            loaded += 1
        if verbose:
            print(f"[load] adapter: {loaded} tensors from {ad_path}", flush=True)
    elif verbose:
        print(f"[load] no adapter at {ad_path} (heads only)", flush=True)

    h_path = os.path.join(out_dir, "heads.safetensors")
    if not os.path.exists(h_path):
        raise SystemExit(f"[load] missing {h_path}")
    hs = load_file(h_path)
    plain = {k.split(".", 1)[1]: v for k, v in hs.items() if k.startswith("plain.")}
    centred = {k.split(".", 1)[1]: v for k, v in hs.items() if k.startswith(("centered.", "centred."))}
    model.plain_head.load_state_dict(plain)
    model.centred_head.load_state_dict(centred)
    for i, norm in enumerate(model.norms):
        norm.load_state_dict({k.split(".", 2)[2]: v for k, v in hs.items() if k.startswith(f"norms.{i}.")})
    if verbose:
        print(f"[load] heads: {len(hs)} tensors from {h_path}", flush=True)
    return meta


def save_checkpoint(model: PoolModel, out_dir: str, meta: dict) -> None:
    """Atomic-ish write (NFS: no flock, so build in .tmp then rename into place)."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, ".tmp_write")
    os.makedirs(os.path.join(tmp, "adapter"), exist_ok=True)
    save_file(adapter_state(model), os.path.join(tmp, "adapter", "adapter_model.safetensors"))
    lora_cfg = dict(
        r=meta["cfg"]["lora"]["r"], lora_alpha=meta["cfg"]["lora"]["alpha"],
        lora_dropout=meta["cfg"]["lora"]["dropout"], bias="none", fan_in_fan_out=False,
        target_modules=list(meta["cfg"]["lora"]["targets"]), task_type="FEATURE_EXTRACTION",
        base_model_name_or_path=meta["cfg"]["base"], inference_mode=True,
        ntl_format=FORMAT,
    )
    with open(os.path.join(tmp, "adapter", "adapter_config.json"), "w", encoding="utf-8") as fh:
        json.dump(lora_cfg, fh, indent=2)
    save_file(heads_state(model), os.path.join(tmp, "heads.safetensors"))
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    for rel in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json",
                "heads.safetensors", "meta.json"):
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(os.path.join(tmp, rel), dst)
    os.rmdir(os.path.join(tmp, "adapter"))
    os.rmdir(tmp)


def load_meta(out_dir: str) -> dict:
    path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(path):
        raise SystemExit(f"[load] no meta.json in {out_dir} -- not a checkpoint dir?")
    with open(path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("labels") != LABELS:
        raise SystemExit(
            f"[load] {path} label order differs from LABELS (18-class competition order); "
            f"refusing to mix column semantics"
        )
    return meta


def rebuild(out_dir: str, device: torch.device, base_dir: Optional[str] = None,
            verbose: bool = True):
    """meta.json -> (model, tokenizer, meta). The single entry point for inference."""
    meta = load_meta(out_dir)
    cfg = meta["cfg"]
    base = base_dir or cfg["base"]
    dtype = DTYPES.get(cfg.get("dtype", "bf16"), torch.bfloat16)
    model, _ = build_model(
        base, cfg["keep"], cfg["layers"], dtype, device,
        lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], n_labels=len(meta["labels"]),
        no_final_norm=cfg.get("no_final_norm", True), allow_unknown_arch=True, verbose=verbose,
    )
    load_into(model, out_dir, verbose=verbose)
    model.eval()
    tok = load_tokenizer(base)
    return model, tok, meta


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_logprobs(model: PoolModel, tokens: Sequence[np.ndarray], device: torch.device,
                     bs: int = 16, groups: Optional[Sequence[str]] = None,
                     twin_w: float = 0.0, fixed_len: Optional[int] = None,
                     verbose: bool = True) -> np.ndarray:
    """log-probabilities (n, 18) in LABELS order; optional prompt-centred twin term.

    `twin_w` > 0 needs `groups`: within each prompt group the mean feature is
    subtracted and the centred head's log-softmax is added with weight twin_w.
    Rows in singleton groups contribute nothing (their centred features are 0).
    """
    model.eval()
    n = len(tokens)
    out = np.zeros((n, len(LABELS)), dtype=np.float32)
    if n == 0:
        return out
    order = np.argsort([len(t) for t in tokens], kind="stable")  # similar lengths batched
    t0 = time.time()
    for s in range(0, n, bs):
        sel = order[s:s + bs]
        ids, att = collate(tokens, sel, device, fixed_len=fixed_len)
        z = model.features(ids, att)
        plain, _ = model.head_logits(z)
        lp = F.log_softmax(plain.float(), dim=-1).cpu().numpy()
        if twin_w > 0.0 and groups is not None:
            gsel = [groups[i] for i in sel]
            zc, keep = centre_by_group(z, gsel)
            _, centred = model.head_logits(z, zc)
            lc = F.log_softmax(centred.float(), dim=-1).cpu().numpy()
            lp = lp + twin_w * lc * keep.cpu().numpy()[:, None]
        out[sel] = lp
        if verbose and (s // bs) % 20 == 0:
            done = min(s + bs, n)
            print(f"\r[predict] {done}/{n} ({time.time() - t0:.0f}s)", end="", flush=True)
    if verbose:
        print(f"\r[predict] {n} rows in {time.time() - t0:.0f}s", flush=True)
    return out


# --------------------------------------------------------------------------- #
# metrics (mirrors corpus/cv.py so numbers stay comparable)
# --------------------------------------------------------------------------- #
def prompt_folds(rows: int, groups: Sequence[str], labels: Sequence[int],
                 n_splits: int = 5, seed: int = 0) -> List[np.ndarray]:
    """GroupKFold over prompts, greedily balanced per class (cv.py's algorithm)."""
    prompts = np.asarray([str(g) for g in groups])
    y = np.asarray(labels, dtype=np.int64)
    uniq, inv = np.unique(prompts, return_inverse=True)
    g = uniq.size
    if g < n_splits:
        raise SystemExit(f"[cv] only {g} prompt groups for {n_splits} folds")
    n_cls = int(y.max()) + 1
    M = np.zeros((g, n_cls), dtype=np.float64)
    np.add.at(M, (inv, y), 1.0)
    sizes = M.sum(1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(g), -sizes))
    loads = np.zeros((n_splits, n_cls), dtype=np.float64)
    assign = np.zeros(g, dtype=np.int64)
    fidx = np.arange(n_splits)
    for gid in order:
        cand = loads + M[gid]
        pick = int(np.lexsort((fidx, cand.sum(1), cand.max(1)))[0])
        assign[gid] = pick
        loads[pick] += M[gid]
    row_fold = assign[inv]
    return [np.where(row_fold == k)[0] for k in range(n_splits)]


def sim_subset(idx: np.ndarray, groups: Sequence[str], lo: int = 1, hi: int = 2,
               seed: int = 0) -> np.ndarray:
    """Keep lo..hi random rows per prompt group (the board-like protocol)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    prompts = np.asarray([str(g) for g in groups])
    rng = np.random.default_rng(seed)
    keep = []
    for p in np.unique(prompts[idx]):
        rows = idx[prompts[idx] == p]
        k = min(int(rng.integers(lo, hi + 1)), rows.size)
        keep.append(rng.choice(rows, size=k, replace=False))
    return np.sort(np.concatenate(keep).astype(np.int64))


def evaluate(y_true: Sequence[int], y_pred: Sequence[int], n_labels: int = len(LABELS)) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    present = [c for c in range(n_labels) if int((y_true == c).sum()) > 0]
    rec = {c: float((y_pred[y_true == c] == c).mean()) if int((y_true == c).sum()) else 0.0
           for c in range(n_labels)}
    return dict(
        bacc=float(np.mean([rec[c] for c in present])) if present else float("nan"),
        accuracy=float((y_true == y_pred).mean()) if y_true.size else float("nan"),
        n=int(y_true.size), n_classes=len(present), per_class_recall=rec,
    )


def board_like(y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str],
               lo: int = 1, hi: int = 2, seed: int = 0, n_reps: int = 20,
               n_labels: int = len(LABELS)) -> dict:
    """Mean balanced accuracy over `n_reps` independent lo..hi-per-prompt subsamples."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    all_idx = np.arange(y_true.size)
    scores = []
    for r in range(n_reps):
        sel = sim_subset(all_idx, groups, lo=lo, hi=hi, seed=seed + r)
        if sel.size == 0:
            continue
        scores.append(evaluate(y_true[sel], y_pred[sel], n_labels)["bacc"])
    if not scores:
        return dict(bacc=float("nan"), bacc_sd=float("nan"), n_reps=0,
                    full_bacc=evaluate(y_true, y_pred, n_labels)["bacc"])
    return dict(bacc=float(np.mean(scores)), bacc_sd=float(np.std(scores)),
                bacc_min=float(np.min(scores)), bacc_max=float(np.max(scores)),
                n_reps=len(scores), full_bacc=evaluate(y_true, y_pred, n_labels)["bacc"])


# --------------------------------------------------------------------------- #
# smoke assets: tiny random Qwen3 + tokenizer + synthetic parquet
# --------------------------------------------------------------------------- #
_SMOKE_STYLES = [
    "— and — again", "can't won't don't", "“quoted” prose", "plain short clauses.",
    "Semicolons; colons: lists", "no capitals here at all", "Title Case Every Word",
    "very long, meandering, subordinate sentence structure that keeps going",
    "burst. short. fragments.", "emojis 🙂 and unicode", "Parenthetical (asides) everywhere",
    "numeric 42 3.14 1,000", "ALL CAPS SHOUTING", "e.g. i.e. viz. cf.",
    "lots of question marks?? really??", "dashes - hyphens -- en rules ---",
    "bulleted feel with free verse lines", "formal register; nominalisations abound",
]


def make_smoke_assets(root: str, n_stage1: int = 200, n_stage2: int = 180,
                      n_groups1: int = 50, n_groups2: int = 30, seed: int = 0,
                      vocab: int = 512, hidden: int = 128, layers: int = 12,
                      keep: int = 8, heads: int = 4, kv_heads: int = 2,
                      max_len: int = 64, verbose: bool = True):
    """Build model_dir/ (tiny random Qwen3 + trained tokenizer) and 2 parquet files.

    Everything downstream then runs unmodified: AutoConfig/AutoModel.from_pretrained,
    AutoTokenizer.from_pretrained, read_table. If --smoke is used with a *real*
    model dir, this is skipped.
    """
    from tokenizers import Tokenizer, decoders, models as tk_models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)

    def make_rows(n, n_groups, tag):
        rows = []
        for i in range(n):
            lab = int(rng.integers(0, len(LABELS)))
            prompt = f"{tag} prompt {int(rng.integers(0, n_groups)):03d} asked about topic {int(rng.integers(0, 40))}"
            body = " ".join(
                f"{_SMOKE_STYLES[lab]} answer number {int(rng.integers(0, 999))} "
                f"with words {int(rng.integers(0, 50))} and more text"
                for _ in range(2)
            )
            rows.append(dict(text=body, model=LABELS[lab], src="smoke",
                             prompt_group=prompt, n_words=len(body.split())))
        # guarantee >= 2 rows per prompt group so centring is always defined
        counts: dict = {}
        for r in rows:
            counts[r["prompt_group"]] = counts.get(r["prompt_group"], 0) + 1
        thin = {g for g, c in counts.items() if c < 2}
        if thin:
            rows += [dict(r) for r in rows if r["prompt_group"] in thin]
        return rows

    d1 = make_rows(n_stage1, n_groups1, "pool")
    d2 = make_rows(n_stage2, n_groups2, "official")
    p1 = os.path.join(root, "smoke_stage1.parquet")
    p2 = os.path.join(root, "smoke_stage2.parquet")
    cols = ["text", "model", "src", "prompt_group", "n_words"]
    write_parquet(p1, d1, cols)
    write_parquet(p2, d2, cols)

    # --- tokenizer ------------------------------------------------------- #
    mdir = os.path.join(root, "tiny_model")
    sig = dict(vocab=vocab, hidden=hidden, layers=layers, heads=heads, kv_heads=kv_heads)
    sig_path = os.path.join(mdir, "smoke_sig.json")
    cached = None
    if os.path.exists(sig_path):
        with open(sig_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
    if cached != sig:
        if os.path.isdir(mdir):
            shutil.rmtree(mdir)  # a stale tiny model with the wrong shape is worse than 2s of retraining
        tk = Tokenizer(tk_models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab, min_frequency=1,
            special_tokens=["<unk>", "<s>", "</s>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.train_from_iterator([r["text"] for r in d1] + [r["text"] for r in d2], trainer)
        fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>",
                                       bos_token="<s>", eos_token="</s>", pad_token="</s>")
        fast.save_pretrained(mdir)
        cfg = Qwen3Config(
            vocab_size=fast.vocab_size, hidden_size=hidden, num_hidden_layers=layers,
            num_attention_heads=heads, num_key_value_heads=kv_heads,
            head_dim=hidden // heads, intermediate_size=hidden * 3,
            max_position_embeddings=512, tie_word_embeddings=False,
            eos_token_id=fast.eos_token_id, bos_token_id=fast.bos_token_id,
            pad_token_id=fast.eos_token_id, torch_dtype="float32",
        )
        tiny = Qwen3Model(cfg).eval()
        tiny.save_pretrained(mdir)
        cfg.save_pretrained(mdir)
        fast.save_pretrained(mdir)
        with open(sig_path, "w", encoding="utf-8") as fh:
            json.dump(sig, fh)
    if verbose:
        print(f"[smoke] tiny model {mdir}, stage1 {len(d1)} rows -> {p1}, "
              f"stage2 {len(d2)} rows -> {p2}, keep={keep}/{layers}, max_len={max_len}", flush=True)
    # 8 evenly spaced depths that are always legal for this keep
    pool = sorted({max(1, int(round(keep * (i + 1) / 8.0))) for i in range(8)})
    while len(pool) < 8:  # tiny keeps can collide after rounding
        cand = [d for d in range(1, keep + 1) if d not in pool]
        if not cand:
            break
        pool = sorted(pool + [cand[-1]])
    return dict(model_dir=mdir, stage1=p1, stage2=p2, keep=keep, max_len=max_len,
                hidden=hidden, layers=layers, pool_layers=pool)
