"""Pick the soft-centring cluster threshold by measurement, not by copying.

The 0.672 notebook runs with `--adapt_w 1.0 --adapt_thr 0.15`. `adapt_thr` is the
average-linkage cosine-similarity cut used to recover prompt groups on the *test* rows,
and 0.15 is a very loose cut (my earlier calibration: at sim 0.15 recovered clusters
merge prompts, pair-precision drops). The soft path degrades gracefully, but a bad
grouping still adds noise to the centring.

This sweeps adapt_thr against the measured OOF gain, at the panel sizes that matter
(the hidden test is not a full 18-answer panel - proven by my 0.525 submission).
Decision rule: choose the threshold that is non-negative at EVERY k, not the one with
the best single number, because the true panel size is unknown.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)
J = os.environ.get('NTL_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(J, 'jobs'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

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


def recover_thr(texts, thr):
    vec = TfidfVectorizer(lowercase=True, stop_words=list(ENGLISH_STOP_WORDS),
                          ngram_range=(1, 2), min_df=1 if len(texts) < 20 else 2,
                          max_df=0.9 if len(texts) < 20 else 0.4, sublinear_tf=True)
    X = normalize(vec.fit_transform(texts))
    return capped_agglo(np.asarray((X @ X.T).todense()), 18, thr)


def centre(X, g):
    X = X.copy()
    for c in np.unique(g):
        m = g == c
        if m.sum() >= 2:
            X[m] -= X[m].mean(0)
    return X


def feats(tr_txt, te_txt):
    parts_a, parts_b = [], []
    for kw in (dict(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=80000),
               dict(analyzer='word', ngram_range=(1, 2), min_df=3, max_features=60000)):
        v = TfidfVectorizer(sublinear_tf=True, **kw).fit(tr_txt)
        parts_a.append(normalize(v.transform(tr_txt)))
        parts_b.append(normalize(v.transform(te_txt)))
    return hstack(parts_a).tocsr(), hstack(parts_b).tocsr()


def main():
    official = pd.read_parquet(os.path.join(J, 'data/official.parquet'))
    y = official['model'].values
    pg = official['prompt_group'].values
    folds = CV.prompt_folds(official, 5, seed=0)

    thrs = [0.02, 0.05, 0.08, 0.12, 0.15]
    ks = [18, 12, 8, 5, 3]
    table = {}
    print(f'{"thr":>6} ' + ' '.join(f'{"k"+str(k):>8}' for k in ks))
    for thr in thrs:
        row = {}
        for k in ks:
            pred = np.empty(len(official), dtype=object)
            for fi in folds:
                sub = fi if k >= 18 else CV.sim_subset(fi, official, lo=k, hi=k, seed=0)
                tr = np.setdiff1d(np.arange(len(official)), fi)
                Xtr, Xte = feats(official['text'].values[tr], official['text'].values[sub])
                g = recover_thr(official['text'].values[sub], thr)
                gtr = pg[tr]
                A = np.asarray(Xtr.todense(), dtype=np.float32)
                B = np.asarray(Xte.todense(), dtype=np.float32)
                clf = LogisticRegression(C=5, max_iter=2000).fit(centre(A, gtr), y[tr])
                pred[sub] = LAB[clf.predict_log_proba(centre(B, g)).argmax(1)]
            present = ~np.array([x is None for x in pred])
            row[k] = round(CV.evaluate(y[present], pred[present])['bacc'], 4)
        table[thr] = row
        print(f'{thr:>6} ' + ' '.join(f'{row[k]:>8.4f}' for k in ks), flush=True)

    # baseline (no centring) for reference
    base = {}
    for k in ks:
        pred = np.empty(len(official), dtype=object)
        for fi in folds:
            sub = fi if k >= 18 else CV.sim_subset(fi, official, lo=k, hi=k, seed=0)
            tr = np.setdiff1d(np.arange(len(official)), fi)
            Xtr, Xte = feats(official['text'].values[tr], official['text'].values[sub])
            clf = LogisticRegression(C=5, max_iter=2000).fit(
                np.asarray(Xtr.todense(), dtype=np.float32), y[tr])
            pred[sub] = LAB[clf.predict_log_proba(
                np.asarray(Xte.todense(), dtype=np.float32)).argmax(1)]
        present = ~np.array([x is None for x in pred])
        base[k] = round(CV.evaluate(y[present], pred[present])['bacc'], 4)
    print(f'{"base":>6} ' + ' '.join(f'{base[k]:>8.4f}' for k in ks))

    safe = {t: all(table[t][k] >= base[k] for k in ks) for t in thrs}
    print('\ndelta vs baseline:')
    for t in thrs:
        d = ' '.join(f'{table[t][k]-base[k]:>+8.4f}' for k in ks)
        print(f'{t:>6} {d}   all-non-negative={safe[t]}')
    json.dump({'table': {str(t): {str(k): v for k, v in r.items()} for t, r in table.items()},
               'base': {str(k): v for k, v in base.items()},
               'safe': {str(t): bool(s) for t, s in safe.items()}},
              open(os.path.join(J, 'out/adapt_thr_sweep.json'), 'w'), indent=1)
    print('\nwrote out/adapt_thr_sweep.json')


if __name__ == '__main__':
    main()
