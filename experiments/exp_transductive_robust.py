"""Is the +0.070 transductive gain real, robust, and does it survive a stronger base?

The +0.070 was measured on a deliberately weak base (TF-IDF SVC alone, 0.2615). Two
things must be checked before betting the competition on it:

  1. ROBUSTNESS - does it hold for different fold seeds and different panel shapes
     (k=18 full panels, and smaller), with clusters recovered out-of-fold?
  2. WHAT IT DOES TO A STRONG BASE - the public pipeline is at ~0.65 per-passage.
     Build the best single-passage classifier I can cheaply (SVC on official + capped
     pool, which the earlier run showed helps), then apply the same assignment step.
     If the gain collapses on a strong base, the whole idea is worthless.

Also reports the ORACLE gap (assignment with true groups) so we know the remaining
headroom, and how it decomposes.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax, log_softmax

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = np.array(CV.LABELS)


def capped_agglo(S, cap=18, thr=0.02):
    n = len(S)
    A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab


def recover_groups(texts, thr=0.02, cap=18):
    vec = TfidfVectorizer(lowercase=True, stop_words=list(ENGLISH_STOP_WORDS),
                          ngram_range=(1, 2), min_df=1 if len(texts) < 20 else 2,
                          max_df=0.9 if len(texts) < 20 else 0.4, sublinear_tf=True)
    X = normalize(vec.fit_transform(texts))
    return capped_agglo(np.asarray((X @ X.T).todense()), cap, thr)


def hungarian(lp, groups):
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if 2 <= len(idx) <= lp.shape[1]:
            r, c = linear_sum_assignment(-lp[idx])
            pred[idx[r]] = c
    return pred


def svc_member(official, pool, fold_idx, extra_w=0.35, cap_per_label=3000, seed=0):
    """4-view n-gram SVC trained on fold-train official + capped pool -> log-probs on fold."""
    tr = np.setdiff1d(np.arange(len(official)), fold_idx)
    # groupby.apply on this pandas version can drop the grouping column from the frame,
    # so sample per label explicitly instead.
    parts = []
    for lab_, g in pool.groupby('model'):
        parts.append(g.sample(min(len(g), cap_per_label), random_state=seed))
    p = pd.concat(parts, ignore_index=True)
    fit_txt = np.concatenate([official['text'].values[tr], p['text'].values])
    fit_y = np.concatenate([official['model'].values[tr], p['model'].values])
    w = np.concatenate([np.ones(len(tr)), np.full(len(p), extra_w)])
    te = official['text'].values[fold_idx]
    views = [
        dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True),
        dict(analyzer='word', ngram_range=(1, 3), min_df=3, max_features=200000, sublinear_tf=True),
    ]
    lp = np.zeros((len(te), 18))
    for spec in views:
        v = TfidfVectorizer(**spec).fit(fit_txt)
        Xtr = normalize(v.transform(fit_txt)); Xte = normalize(v.transform(te))
        clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
        clf.fit(Xtr, fit_y, sample_weight=w)
        d = clf.decision_function(Xte)
        D = np.full((len(te), 18), -50.0)
        for j, c in enumerate(clf.classes_):
            D[:, list(LAB).index(c)] = d[:, j]
        lp += log_softmax(D * 2.0, 1)
    return lp / len(views)


def views():
    return [0, 1]


def run(official, lp, y, pg, folds, tag, ks=(18, 12, 6)):
    out = {}
    # per-passage baseline
    base_full = CV.evaluate(y, LAB[lp.argmax(1)])['bacc']
    print(f'\n[{tag}] per-passage full={base_full:.4f}')
    out['perpassage'] = base_full

    for k in ks:
        # honest: recover clusters on the (sub)test set only, per fold
        base_b, hung_b, orac_b, sz = [], [], [], []
        for fi in folds:
            sub = fi if k >= 18 else CV.sim_subset(fi, official, lo=k, hi=k, seed=0)
            g = recover_groups(official.text.values[sub], thr=0.02)
            lp_s = lp[sub]; y_s = y[sub]; true_g = pg[sub]
            base_b.append(CV.evaluate(y_s, LAB[lp_s.argmax(1)])['bacc'])
            hung_b.append(CV.evaluate(y_s, LAB[hungarian(lp_s, g)])['bacc'])
            orac_b.append(CV.evaluate(y_s, LAB[hungarian(lp_s, true_g)])['bacc'])
            sz.append(len(np.unique(g)))
        print(f'  k={k:<2d} recovered_clusters={np.mean(sz):.1f}  '
              f'per-passage={np.mean(base_b):.4f}  hung(recovered)={np.mean(hung_b):.4f} '
              f'({np.mean(hung_b)-np.mean(base_b):+.4f})  hung(TRUE)={np.mean(orac_b):.4f} '
              f'({np.mean(orac_b)-np.mean(base_b):+.4f})')
        out[f'k{k}'] = dict(clusters=float(np.mean(sz)),
                            perpassage=float(np.mean(base_b)),
                            hung_rec=float(np.mean(hung_b)),
                            hung_true=float(np.mean(orac_b)))
    return out


def main():
    official = pd.read_parquet(os.path.join(HERE, 'official.parquet'))
    pool = pd.read_parquet(os.path.join(HERE, 'union_capped.parquet'))
    y = official.model.values
    pg = official.prompt_group.values
    res = {}

    lp_weak = np.load(os.path.join(HERE, 'oof_svc_official_only.npy'))
    lp_strong = None
    sp = os.path.join(HERE, 'oof_svc_pool_3k.npy')
    if os.path.exists(sp):
        lp_strong = np.load(sp)
        print('loaded precomputed strong member', lp_strong.shape)
    else:
        print('training strong member (SVC on official + capped pool)...')
        t0 = time.time()
        folds = CV.prompt_folds(official, 5, seed=0)
        lp_strong = np.zeros((len(official), 18))
        for k, fi in enumerate(folds):
            lp_strong[fi] = svc_member(official, pool, fi)
            print(f'  fold{k} done {time.time()-t0:.0f}s', flush=True)
        np.save(sp, lp_strong)

    for seed in (0, 1):
        folds = CV.prompt_folds(official, 5, seed=seed)
        print(f'\n{"="*70}\nfold seed {seed}')
        res[f'weak_seed{seed}'] = run(official, lp_weak, y, pg, folds, f'SVC official-only seed{seed}')
        res[f'strong_seed{seed}'] = run(official, lp_strong, y, pg, folds, f'SVC official+3k pool seed{seed}')

    json.dump(res, open(os.path.join(HERE, 'transductive_robust.json'), 'w'), indent=1)
    print('\nsaved transductive_robust.json')


if __name__ == '__main__':
    main()
