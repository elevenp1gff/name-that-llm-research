"""The realistic case: a prompt has only k of 18 answers (hidden test is not a full panel).

A recovered cluster with k members is NOT a complete assignment problem: at most k of the
18 labels are present, and we do not know which. Forcing an injective assignment over all
18 labels therefore mis-allocates. This measures what actually helps at each k:

  argmax          baseline
  hung18          injective over all 18 labels (what the public code does)
  hung_k          pick the best k labels via rectangular assignment, leave the rest argmax
  top1_gate       only reassign rows whose top-2 margin is small
  none_partial    leave clusters alone when their recovered size < 18

Measured out of fold with clusters recovered on the fold's own rows.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from scipy.special import softmax

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


def recover(texts):
    vec = TfidfVectorizer(lowercase=True, stop_words=list(ENGLISH_STOP_WORDS),
                          ngram_range=(1, 2), min_df=1 if len(texts) < 20 else 2,
                          max_df=0.9 if len(texts) < 20 else 0.4, sublinear_tf=True)
    X = normalize(vec.fit_transform(texts))
    return capped_agglo(np.asarray((X @ X.T).todense()))


def assign(lp, g, mode):
    """mode: 'argmax' | 'hung18' | 'hung_k' | 'margin' """
    pred = lp.argmax(1).copy()
    for c in np.unique(g):
        idx = np.where(g == c)[0]
        k = len(idx)
        if k < 2:
            continue
        sub = lp[idx]
        if mode == 'hung18':
            r, cc = linear_sum_assignment(-sub)
            pred[idx[r]] = cc
        elif mode == 'hung_k':
            # rectangular: k rows, 18 cols -> each row gets a distinct label
            r, cc = linear_sum_assignment(-sub)
            pred[idx[r]] = cc
        elif mode == 'margin':
            p = softmax(sub, 1)
            s = np.sort(p, 1)
            marg = s[:, -1] - s[:, -2]
            r, cc = linear_sum_assignment(-sub)
            take = r[marg[r] < 0.35]
            if len(take):
                m = dict(zip(r, cc))
                pred[idx[take]] = [m[t] for t in take]
    return pred


def main():
    official = pd.read_parquet(os.path.join(HERE, 'official.parquet'))
    lp_all = np.load(os.path.join(HERE, 'oof_svc_pool_3k.npy'))
    y = official.model.values
    pg = official.prompt_group.values
    folds = CV.prompt_folds(official, 5, seed=0)
    res = {}

    print(f'{"k":>3} {"recov_cl":>9} {"purity":>7} {"argmax":>8} {"hung18":>8} {"hung_k":>8} {"margin":>8}')
    for k in (18, 12, 8, 5, 3, 2):
        preds = {m: np.empty(len(official), dtype=object) for m in ('argmax', 'hung18', 'hung_k', 'margin')}
        pur, ncl = [], []
        for fi in folds:
            sub = fi if k >= 18 else CV.sim_subset(fi, official, lo=k, hi=k, seed=0)
            g = recover(official.text.values[sub])
            ncl.append(len(np.unique(g)))
            for c in np.unique(g):
                idx = sub[g == c]
                pur.append(pd.Series(pg[idx]).value_counts().iloc[0] / len(idx))
            for m in preds:
                preds[m][sub] = LAB[assign(lp_all[sub], g, m)]
        scores = {m: CV.evaluate(y, preds[m])['bacc'] for m in preds}
        print(f'{k:>3} {np.mean(ncl):9.1f} {np.mean(pur):7.3f} '
              + ' '.join(f'{scores[m]:8.4f}' for m in ('argmax', 'hung18', 'hung_k', 'margin')))
        res[k] = dict(n_clusters=float(np.mean(ncl)), purity=float(np.mean(pur)),
                      **{m: float(v) for m, v in scores.items()})

    json.dump(res, open(os.path.join(HERE, 'partial_panel.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
