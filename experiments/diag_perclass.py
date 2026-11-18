"""Where is balanced accuracy actually being lost, class by class?

Purpose (one run, no ablations): take the strongest *cheap CPU* predictor I have
(4-view n-gram TF-IDF + LinearSVC, official fold-train + union pool at weight
0.35, optional soft centring), run a prompt-grouped 5-fold OOF, and dissect the
error structure so the next hour of work is spent on the classes that are worth
the most metric points.

Design notes that matter
------------------------
1. Soft centring without dense matrices.
   `exp_soft_trans.py` centres the *feature* rows by their recovered-cluster mean,
   which needs a dense 61k x 265k matrix -- impossible with the union pool in the
   training matrix. For a linear model the two are identical on the score side:

       w . (x - mu_g) + b  ==  s(x) - mean_{i in g} s(x_i)

   so centring the scores and centring the features give the *same* decision
   values, at zero cost. This script subtracts the per-cluster mean decision
   score, which is the exact score-space image of exp_soft_trans's feature
   centring. (Clusters of size < 2 are left alone, as there.)

2. Parallelism. liblinear (dual) is single-threaded, so 5 folds x 4 views are run
   as 20 independent processes on the 192-core box. Wall time is one task, not 20.

3. Panels. The hidden test shows a prompt only a few answers. The same fitted
   models are therefore scored twice: on the full 18-answer fold panel, and on a
   `sim_subset` panel with 1-2 answers per prompt. Both are reported, because
   centring behaves differently when clusters are tiny.

Everything printed is computed; there are no hard-coded numbers.
"""
import argparse
import json
import os
import sys
import time

# Set before numpy/sklearn import: liblinear is single-threaded anyway, and every
# (fold, view) is its own process, so 1 thread each avoids 20-way BLAS thrash.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import multiprocessing as mp

# pandas' optional accelerators are compiled against numpy 1.x and throw C-level
# tracebacks on import under numpy 2.x.
sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)

import numpy as np
import pandas as pd

J = os.environ.get('NTL_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(J, 'jobs'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

LAB = list(CV.LABELS)
NL = len(LAB)
LAB_IX = {c: i for i, c in enumerate(LAB)}

# Same four views as exp_svc_scale.py, kept byte-identical because the fusion is
# tuned to them. View 3 keeps punctuation/case (lowercase=False, token_pattern=\S+)
# so it can see markdown/typography habits. View 4 is stop-words-REMOVED content
# n-grams capped at 20k features: a low-variance topical scaffold that keeps the
# other three from over-trusting topic-specific vocabulary.
VIEW_NAMES = ['char_wb35', 'word13', 'word26raw', 'funcword']
VIEWS = [
    dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=300000,
         sublinear_tf=True),
    dict(analyzer='word', ngram_range=(1, 3), min_df=3, max_features=200000,
         sublinear_tf=True),
    dict(analyzer='word', ngram_range=(2, 6), min_df=3, max_features=100000,
         sublinear_tf=True, lowercase=False, token_pattern=r'\S+'),
    dict(analyzer='word', ngram_range=(1, 3), min_df=5, max_features=20000,
         sublinear_tf=True, stop_words='english'),
]

FAMILY = {
    'claude-fable-5': 'anthropic', 'claude-opus-4-6': 'anthropic',
    'claude-opus-4-8': 'anthropic', 'claude-opus-5': 'anthropic',
    'claude-sonnet-5': 'anthropic',
    'gemini-3.1-pro-preview': 'google', 'gemini-3.6-flash': 'google',
    'gemini-3.7-flash': 'google', 'gemini-3.8-flash': 'google',
    'glm-5.3': 'zai', 'glm-5.3-flash': 'zai',
    'gpt-5.5': 'openai', 'gpt-5.6-luna': 'openai', 'gpt-5.6-sol': 'openai',
    'gpt-5.6-terra': 'openai', 'gpt-6-astra': 'openai',
    'kimi-k3': 'moonshot', 'qwen3.8-max': 'qwen',
}
FAMILIES = ['anthropic', 'google', 'zai', 'openai', 'moonshot', 'qwen']

# short codes for the 18x18 table; the full names are listed right above it
SHORT = {
    'claude-fable-5': 'cFB5', 'claude-opus-4-6': 'cO46', 'claude-opus-4-8': 'cO48',
    'claude-opus-5': 'cO5', 'claude-sonnet-5': 'cS5',
    'gemini-3.1-pro-preview': 'g31P', 'gemini-3.6-flash': 'g36F',
    'gemini-3.7-flash': 'g37F', 'gemini-3.8-flash': 'g38F',
    'glm-5.3': 'z53', 'glm-5.3-flash': 'z53F',
    'gpt-5.5': 'o55', 'gpt-5.6-luna': 'o56L', 'gpt-5.6-sol': 'o56S',
    'gpt-5.6-terra': 'o56T', 'gpt-6-astra': 'o6A',
    'kimi-k3': 'mK3', 'qwen3.8-max': 'q38M',
}

_G = {}


# --------------------------------------------------------------------------- #
# cluster recovery (verbatim semantics of exp_soft_trans.recover / capped_agglo)
# --------------------------------------------------------------------------- #
def capped_agglo(S, cap=18, thr=0.05):
    n = len(S)
    A = S.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32)
    alive = np.ones(n, bool)
    lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new
        A[:, i] = new
        A[i, i] = -np.inf
        A[j] = -np.inf
        A[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    return lab


def recover(texts, thr=0.05):
    vec = TfidfVectorizer(lowercase=True, stop_words=list(ENGLISH_STOP_WORDS),
                          ngram_range=(1, 2),
                          min_df=1 if len(texts) < 20 else 2,
                          max_df=0.9 if len(texts) < 20 else 0.4,
                          sublinear_tf=True)
    X = normalize(vec.fit_transform(texts))
    return capped_agglo(np.asarray((X @ X.T).todense()), cap=18, thr=thr)


# --------------------------------------------------------------------------- #
# worker: one (fold, view) fit
# --------------------------------------------------------------------------- #
def _work(task):
    fi, vi = task
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    off = _G['official']
    pool = _G['pool']
    folds = _G['folds']
    t0 = time.time()

    fi_rows = folds[fi]
    tr = np.setdiff1d(np.arange(len(off)), fi_rows)
    fit_txt = np.concatenate([off['text'].values[tr], pool['text'].values])
    fit_y = np.concatenate([off['model'].values[tr], pool['model'].values])
    w = np.concatenate([np.ones(len(tr)),
                        np.full(len(pool), _G['extra_w'], dtype=np.float64)])
    te = off['text'].values[fi_rows]

    v = TfidfVectorizer(**VIEWS[vi]).fit(fit_txt)
    Xtr = normalize(v.transform(fit_txt))
    Xte = normalize(v.transform(te))
    t1 = time.time()
    clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
    clf.fit(Xtr, fit_y, sample_weight=w)
    d = clf.decision_function(Xte)

    D = np.full((len(te), NL), -50.0)
    for j, c in enumerate(clf.classes_):
        D[:, LAB_IX[c]] = d[:, j]
    del Xtr, Xte
    return fi, vi, D, len(v.vocabulary_), time.time() - t1


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #
def centre_scores(D, g):
    """Score-space image of feature centring: subtract the per-cluster mean.

    exp_soft_trans centres the feature rows by the recovered-cluster mean; for a
    linear model that is exactly s(x) - mean_{i in g} s(x_i), so this reproduces
    the same decision values without ever building a dense feature matrix.
    Clusters of size < 2 are left untouched, as in exp_soft_trans.centre().
    """
    D = D.copy()
    for c in np.unique(g):
        m = g == c
        if m.sum() >= 2:
            D[m] -= D[m].mean(0)
    return D


def fuse_view_scores(per_view_D, temp=2.0):
    """Average log-probabilities across views, as exp_svc_scale.py does.

    exp_svc_scale accumulates `log_softmax(D * 2.0)` per view and divides by the
    number of views -- i.e. the geometric mean of the per-view probabilities, not
    the softmax of the mean score. Those give different argmaxes, so the fusion
    is reproduced literally here.
    """
    acc = np.zeros_like(per_view_D[0])
    for D in per_view_D:
        Z = D * temp
        Z = Z - Z.max(1, keepdims=True)          # numerical guard == log_softmax
        acc += Z - np.log(np.exp(Z).sum(1, keepdims=True))
    return acc / len(per_view_D)


def confusion(y_true_idx, y_pred_idx, n=NL):
    C = np.zeros((n, n), dtype=np.int64)  # rows = true, cols = predicted
    np.add.at(C, (y_true_idx, y_pred_idx), 1)
    return C


def points_table(recall, support):
    """(best_recall - recall_c)/18: metric points recoverable by fixing class c.

    `best_recall` is the max recall any class achieves (a single scalar), so the
    top class scores 0 and this is purely a ranking of headroom inside *this*
    model. (1 - recall)/18 is also reported as distance-to-perfect.
    """
    present = [c for c in LAB if support[c] > 0]
    best = max(recall[c] for c in present) if present else float('nan')
    rows = []
    for c in LAB:
        rows.append(dict(
            label=c,
            family=FAMILY[c],
            support=int(support[c]),
            recall=float(recall[c]),
            miss_rate=float(1.0 - recall[c]),
            points_available=float((best - recall[c]) / NL),
            points_to_perfect=float((1.0 - recall[c]) / NL),
        ))
    rows.sort(key=lambda r: -r['points_available'])
    return rows, float(best)


def family_stats(C):
    """C is true x pred with rows/cols in LAB order."""
    fam_ix = np.array([FAMILIES.index(FAMILY[c]) for c in LAB])
    # collapse classes to families: 18x18 -> 6x6
    F = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.int64)
    np.add.at(F, (fam_ix[:, None].repeat(NL, 1).ravel(),
                  fam_ix[None, :].repeat(NL, 0).ravel()), C.ravel())

    total = int(C.sum())
    correct = int(np.trace(C))
    errors = total - correct
    off = C.copy()
    np.fill_diagonal(off, 0)
    intra = int(sum(off[np.ix_(fam_ix == k, fam_ix == k)].sum() for k in range(len(FAMILIES))))

    per_family = {}
    for k, F_ in enumerate(FAMILIES):
        cls = [c for c in LAB if FAMILY[c] == F_]
        n_true = int(sum(int(C[LAB_IX[c]].sum()) for c in cls))
        n_fam_ok = int(F[k].sum())
        n_correct = int(sum(int(C[LAB_IX[c], LAB_IX[c]]) for c in cls))
        per_family[F_] = dict(
            n_classes=len(cls),
            n_rows=n_true,
            family_recall=float(F[k, k] / n_true) if n_true else float('nan'),
            n_family_correct=n_fam_ok,
            conditional_acc=float(n_correct / n_fam_ok) if n_fam_ok else float('nan'),
            overall_recall=float(n_correct / n_true) if n_true else float('nan'),
        )

    # Oracle-family projection: assume every row's family is known perfectly and
    # the within-family splitter keeps the accuracy it has on family-correct rows.
    # Assumes rows are exchangeable -- an upper bound, not a forecast.
    proj = []
    for c in LAB:
        a = LAB_IX[c]
        n_true = int(C[a].sum())
        n_ok = int(C[a][fam_ix == fam_ix[a]].sum())
        proj.append(n_ok / n_true if n_true else 0.0)
    return dict(
        n=total, n_correct=correct, n_errors=errors,
        accuracy=float(correct / total) if total else float('nan'),
        intra_family_errors=intra,
        cross_family_errors=int(errors - intra),
        intra_family_fraction=float(intra / errors) if errors else float('nan'),
        family_matrix=[[int(x) for x in row] for row in F],
        per_family=per_family,
        oracle_family_projected_bacc=float(np.mean(proj)),
    )


def top_pairs(C, k=12):
    out = []
    n = C.shape[0]
    for a in range(n):
        for b in range(n):
            if a != b and C[a, b] > 0:
                out.append((int(C[a, b]), LAB[a], LAB[b]))
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [dict(count=c, true=a, pred=b, family=FAMILY[a],
                 same_family=bool(FAMILY[a] == FAMILY[b])) for c, a, b in out[:k]]


def print_matrix(C, title):
    print(f'\n{title}  (rows = TRUE, cols = PREDICTED, counts)')
    print('         ' + ''.join(f'{SHORT[c]:>6s}' for c in LAB) + f'{"tot":>7s}')
    for i, c in enumerate(LAB):
        row = ''.join(f'{int(C[i, j]):>6d}' for j in range(NL))
        print(f'{SHORT[c]:>6s} | ' + row + f'{int(C[i].sum()):>7d}')
    print('         ' + ''.join(f'{SHORT[c]:>6s}' for c in LAB))
    print(f'{"pred":>6s}   ' + ''.join(f'{int(C[:, j].sum()):>6d}' for j in range(NL)))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centring', type=int, default=1, choices=[0, 1])
    ap.add_argument('--thr', type=float, default=0.05)
    ap.add_argument('--pool_cap', type=int, default=3400,
                    help='rows kept per label from union.parquet (0 = all 269120)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold_seed', type=int, default=0, help='prompt_folds seed')
    ap.add_argument('--panel_seed', type=int, default=0, help='sim_subset seed')
    ap.add_argument('--extra_w', type=float, default=0.35, help='union sample weight')
    ap.add_argument('--jobs', type=int, default=20)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--n_examples', type=int, default=8)
    ap.add_argument('--n_top_pairs', type=int, default=12)
    ap.add_argument('--quick', action='store_true', help='pool_cap=1200, jobs=8 (smoke test)')
    ap.add_argument('--out', default=os.path.join(J, 'out/diag_perclass.json'))
    a = ap.parse_args()
    if a.quick:
        a.pool_cap = 1200
        a.jobs = 8

    t_start = time.time()
    official = pd.read_parquet(os.path.join(J, 'data/official.parquet')).reset_index(drop=True)
    union = pd.read_parquet(os.path.join(J, 'data/union.parquet')).reset_index(drop=True)
    if a.pool_cap > 0:
        parts = [g.sample(min(len(g), a.pool_cap), random_state=0)
                 for _, g in union.groupby('model')]
        pool = pd.concat(parts, ignore_index=True)
        pool_sampled = True
    else:
        pool = union
        pool_sampled = False

    print('=' * 100)
    print('Name That LLM -- per-class error diagnosis')
    print('=' * 100)
    print(f'official rows={len(official)}  prompts={official["prompt_group"].nunique()} '
          f' classes={official["model"].nunique()}')
    print(f'union rows={len(union)}  -> pool used={len(pool)}'
          f'{" (cap %d/label, SUBSAMPLED)" % a.pool_cap if pool_sampled else " (full)"}')
    print(f'centring={a.centring}  thr={a.thr}  extra_w={a.extra_w}  '
          f'temp={a.temp}  folds={a.folds} seed={a.fold_seed}  jobs={a.jobs}')
    print(f'views={VIEW_NAMES}', flush=True)

    folds = CV.prompt_folds(official, a.folds, seed=a.fold_seed)
    print('fold row sizes: ' + str([len(f) for f in folds]))

    _G.update(official=official, pool=pool, folds=folds, extra_w=a.extra_w)
    tasks = [(fi, vi) for fi in range(len(folds)) for vi in range(len(VIEWS))]
    print(f'\nfitting {len(tasks)} models ({len(folds)} folds x {len(VIEWS)} views) '
          f'on {a.jobs} workers ...', flush=True)

    D = {fi: {} for fi in range(len(folds))}   # D[fold][view] = raw decisions
    meta = {}
    done = 0
    ctx = mp.get_context('fork')
    with ctx.Pool(a.jobs) as P:
        for fi, vi, Dv, nfeat, fit_s in P.imap_unordered(_work, tasks):
            assert Dv.shape == (len(folds[fi]), NL), (Dv.shape, len(folds[fi]), NL)
            D[fi][vi] = Dv
            meta[f'f{fi}/{VIEW_NAMES[vi]}'] = dict(n_features=int(nfeat),
                                                   fit_seconds=round(fit_s, 1))
            done += 1
            print(f'  [{done}/{len(tasks)}] fold{fi} {VIEW_NAMES[vi]:>10s} '
                  f'feats={nfeat:>7d} fit={fit_s:.0f}s  elapsed={time.time()-t_start:.0f}s',
                  flush=True)
    for fi in range(len(folds)):
        assert len(D[fi]) == len(VIEWS), f'fold {fi} missing views'

    print(f'\nall models fitted in {(time.time()-t_start)/60:.1f} min', flush=True)

    # ---------------- panels ----------------
    panels = {}
    SPEC = (('full18', None), ('partial2', (1, 2)))
    for name, span in SPEC:
        rows_by_fold, g_by_fold = {}, {}
        for fi in range(len(folds)):
            fi_rows = folds[fi]
            if span is None:
                sel_local = np.arange(len(fi_rows))
            else:
                lo, hi = span
                keep = CV.sim_subset(fi_rows, official, lo=lo, hi=hi, seed=a.panel_seed)
                # sim_subset returns sorted positions into `official`; map to fold-local
                sel_local = np.sort(np.searchsorted(fi_rows, keep))
                assert np.array_equal(fi_rows[sel_local], keep), 'panel mapping broken'
            rows_by_fold[fi] = sel_local
            g_by_fold[fi] = recover(official['text'].values[fi_rows[sel_local]], thr=a.thr)
        panels[name] = (rows_by_fold, g_by_fold)
        n_rows = sum(len(v) for v in rows_by_fold.values())
        n_grp = sum(len(np.unique(v)) for v in g_by_fold.values())
        sz = np.concatenate([np.bincount(v).tolist()[1:] for v in g_by_fold.values()])
        print(f'panel {name:>9s}: {n_rows} rows, {n_grp} recovered clusters, '
              f'cluster size mean={sz.mean():.2f} max={sz.max()}')

    def predict_panel(pname, centre):
        pred = np.empty(len(official), dtype=object)
        filled = np.zeros(len(official), dtype=bool)
        conf = {}
        for fi in range(len(folds)):
            fi_rows = folds[fi]
            sl = panels[pname][0][fi]
            g = panels[pname][1][fi]
            views = []
            for vi in range(len(VIEWS)):
                S = D[fi][vi][sl]
                if centre:
                    S = centre_scores(S, g)
                views.append(S)
            # fuse exactly like exp_svc_scale: mean of per-view log_softmax
            LP = fuse_view_scores(views, temp=a.temp)
            pi = LP.argmax(1)
            pred[fi_rows[sl]] = np.array(LAB, dtype=object)[pi]
            filled[fi_rows[sl]] = True
            conf[fi] = (official['model'].values[fi_rows[sl]], pi)
        sel = np.where(filled)[0]
        yt = np.asarray([str(v) for v in official['model'].values[sel]])
        yp = np.asarray([str(v) for v in pred[sel]])
        return dict(sel=sel, y_true=yt, y_pred=yp,
                    per_fold={fi: (np.asarray(ytf), np.array([LAB[i] for i in pif]))
                              for fi, (ytf, pif) in conf.items()})

    # ---------------- scoring ----------------
    variants = {}
    for pname in ('full18', 'partial2'):
        for centre in ([False, True] if a.centring else [False]):
            key = f'{pname}/{"centred" if centre else "plain"}'
            variants[key] = predict_panel(pname, centre)

    print('\n' + '=' * 100)
    print('BALANCED ACCURACY BY VARIANT')
    print('=' * 100)
    print(f'{"variant":<22s} {"bacc":>8s} {"acc":>8s} {"n":>6s}')
    base_for_delta = {}
    for key in ('full18/plain', 'partial2/plain'):
        if key in variants:
            base_for_delta[key.split('/')[0]] = CV.evaluate(
                variants[key]['y_true'], variants[key]['y_pred'])['bacc']
    for key, v in variants.items():
        r = CV.evaluate(v['y_true'], v['y_pred'])
        pn = key.split('/')[0]
        d = r['bacc'] - base_for_delta.get(pn, r['bacc'])
        print(f'{key:<22s} {r["bacc"]:8.4f} {r["accuracy"]:8.4f} {r["n"]:6d}'
              f'   delta_vs_plain={d:+.4f}')

    # primary = the variant we were asked to use
    prim_panel = 'full18'
    prim_centre = bool(a.centring)
    prim_key = f'{prim_panel}/{"centred" if prim_centre else "plain"}'
    V = variants[prim_key]
    res = CV.evaluate(V['y_true'], V['y_pred'])
    print(f'\nPRIMARY variant for the per-class dissection: {prim_key}')

    # board-like (1-2 answers/prompt resampling) -- the metric that tracks LB
    board = CV.board_like_score(V['y_true'], V['y_pred'],
                                official['prompt_group'].values[V['sel']],
                                lo=1, hi=2, seed=a.panel_seed, n_reps=20)
    print(f'board_like (1-2 per prompt, 20 reps) = {board["bacc"]:.4f} '
          f'sd={board["bacc_sd"]:.4f}  (full-panel bacc {res["bacc"]:.4f})')

    # ---------------- per fold ----------------
    per_fold = []
    print('\n' + '=' * 100)
    print('PER-FOLD BALANCED ACCURACY AND PER-CLASS RECALL (ascending)')
    print('=' * 100)
    for fi in sorted(V['per_fold']):
        ytf, ypf = V['per_fold'][fi]
        rf = CV.evaluate(ytf, ypf)
        print(f'\n--- fold {fi}: bacc={rf["bacc"]:.4f}  n={rf["n"]}  acc={rf["accuracy"]:.4f}')
        items = sorted(((c, rf['per_class_recall'][c], int((ytf == c).sum())) for c in LAB),
                       key=lambda t: t[1])
        print('    class                     recall  support')
        for c, r, s in items:
            print(f'    {c:<24s} {r:7.3f}  {s:5d}')
        per_fold.append(dict(fold=fi, bacc=float(rf['bacc']), n=int(rf['n']),
                             accuracy=float(rf['accuracy']),
                             per_class_recall={c: float(rf['per_class_recall'][c]) for c in LAB},
                             support={c: int((ytf == c).sum()) for c in LAB}))

    # ---------------- pooled per-class ----------------
    support = {c: int((V['y_true'] == c).sum()) for c in LAB}
    recall = {c: float(res['per_class_recall'][c]) for c in LAB}
    ptr, best = points_table(recall, support)

    print('\n' + '=' * 100)
    print(f'POOLED PER-CLASS RECALL -- {prim_key}  (bacc={res["bacc"]:.4f}, n={res["n"]})')
    print('sorted ascending by recall; points_available = (max_recall - recall)/18')
    print('=' * 100)
    print(f'{"#":>3s} {"class":<24s} {"family":<10s} {"recall":>7s} {"supp":>5s} '
          f'{"misses":>7s} {"pts_avail":>10s} {"pts_to_1.0":>11s}')
    for i, r in enumerate(ptr, 1):
        print(f'{i:>3d} {r["label"]:<24s} {r["family"]:<10s} {r["recall"]:7.3f} '
              f'{r["support"]:5d} {int(round(r["miss_rate"]*r["support"])):7d} '
              f'{r["points_available"]:10.4f} {r["points_to_perfect"]:11.4f}')
    print(f'\nmax_recall (the "best_in_column" reference) = {best:.4f}')
    print(f'sum of points_available over all classes = {sum(r["points_available"] for r in ptr):.4f}')
    print(f'(identity check: max_recall - bacc = {best - res["bacc"]:.4f})')

    # ---------------- confusion ----------------
    C = confusion(np.array([LAB_IX[v] for v in V['y_true']]),
                  np.array([LAB_IX[v] for v in V['y_pred']]))
    print('\nlegend: ' + '  '.join(f'{SHORT[c]}={c}' for c in LAB))
    print_matrix(C, f'CONFUSION MATRIX -- {prim_key}')

    pairs = top_pairs(C, a.n_top_pairs)
    print(f'\nTOP {len(pairs)} OFF-DIAGONAL CELLS (true -> predicted)')
    print(f'{"#":>3s} {"count":>6s} {"true":<24s} {"predicted":<24s} {"same_family":>11s}')
    for i, p in enumerate(pairs, 1):
        print(f'{i:>3d} {p["count"]:6d} {p["true"]:<24s} {p["pred"]:<24s} '
              f'{str(p["same_family"]):>11s}')

    # ---------------- families ----------------
    fam = family_stats(C)
    print('\n' + '=' * 100)
    print('FAMILY STRUCTURE')
    print('=' * 100)
    print(f'errors total={fam["n_errors"]}  intra-family={fam["intra_family_errors"]} '
          f'({fam["intra_family_fraction"]*100:.1f}%)  '
          f'cross-family={fam["cross_family_errors"]}')
    print(f'\n{"family":<12s} {"classes":>8s} {"rows":>6s} {"fam_recall":>11s} '
          f'{"cond_acc|fam_ok":>16s} {"overall_recall":>15s}')
    for F_ in FAMILIES:
        s = fam['per_family'][F_]
        print(f'{F_:<12s} {s["n_classes"]:8d} {s["n_rows"]:6d} {s["family_recall"]:11.3f} '
              f'{s["conditional_acc"]:16.3f} {s["overall_recall"]:15.3f}')
    print(f'\nfamily-level confusion (rows=true family, cols=predicted family):')
    print('            ' + ''.join(f'{F_[:8]:>10s}' for F_ in FAMILIES))
    for i, F_ in enumerate(FAMILIES):
        print(f'{F_:<12s}' + ''.join(f'{int(x):>10d}' for x in fam['family_matrix'][i]))
    print(f'\noracle-family projection (perfect gate, current within-family splitter): '
          f'bacc ~ {fam["oracle_family_projected_bacc"]:.4f}  '
          f'vs current {res["bacc"]:.4f}')

    # ---------------- examples of the 3 biggest pairs ----------------
    examples = []
    print('\n' + '=' * 100)
    print('EXAMPLES FROM THE 3 BIGGEST CONFUSION PAIRS (first 300 chars)')
    print('=' * 100)
    for p in pairs[:3]:
        m = np.where((V['y_true'] == p['true']) & (V['y_pred'] == p['pred']))[0]
        sel_g = np.sort(V['sel'][m])[:a.n_examples]
        print(f'\n### true={p["true"]}  pred={p["pred"]}  (n={p["count"]}, '
              f'same_family={p["same_family"]})')
        ex = []
        for j, gi in enumerate(sel_g, 1):
            t = str(official['text'].values[gi])[:300].replace('\n', ' ')
            print(f'  [{j}] {t}')
            ex.append(dict(idx=int(gi), prompt_group=str(official['prompt_group'].values[gi]),
                           text=t))
        examples.append(dict(true=p['true'], pred=p['pred'], count=p['count'], examples=ex))

    # ---------------- json ----------------
    out = dict(
        config=dict(centring=a.centring, thr=a.thr, pool_cap=a.pool_cap,
                    pool_rows=int(len(pool)), pool_subsampled=bool(pool_sampled),
                    extra_w=a.extra_w, folds=a.folds, fold_seed=a.fold_seed,
                    panel_seed=a.panel_seed, temp=a.temp, views=VIEW_NAMES,
                    n_jobs=a.jobs),
        labels=LAB,
        families=FAMILY,
        variants={k: dict(bacc=float(CV.evaluate(v['y_true'], v['y_pred'])['bacc']),
                          accuracy=float(CV.evaluate(v['y_true'], v['y_pred'])['accuracy']),
                          n=int(len(v['y_true'])))
                  for k, v in variants.items()},
        primary=prim_key,
        pooled=dict(bacc=float(res['bacc']), accuracy=float(res['accuracy']),
                    n=int(res['n']),
                    per_class_recall=recall, support=support,
                    board_like_bacc=float(board['bacc']),
                    board_like_sd=float(board['bacc_sd']),
                    board_like_per_class=board['per_class_recall']),
        max_recall_reference=best,
        points_ranking=ptr,
        per_fold=per_fold,
        confusion_matrix=[[int(x) for x in row] for row in C],
        confusion_axis='rows = true class, cols = predicted class, order = labels',
        top_pairs=pairs,
        families_stats=fam,
        examples=examples,
        fit_meta=meta,
        runtime_minutes=round((time.time() - t_start) / 60, 1),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.out}')
    print(f'total runtime {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
