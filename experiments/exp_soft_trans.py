"""Does the FAILURE of transductive assignment hide a WINNING variant?

Facts I have measured:
  * assignment with FULL 18-answer panels: 0.3989 -> 0.4960  (+0.097 OOF)
  * the same idea on the real hidden test:  0.668  -> 0.525   (-0.143 LB)
  * recovered-cluster purity collapses as the panel gets smaller:
    k=18 purity 1.000 | k=12 1.000 | k=8 0.666 | k=5 0.515 | k=2 0.310

So on a *partial* panel the hard 18-way constraint is wrong. But that does not mean all
transductive information is worthless: the SOFT version (centre the features by the
recovered cluster mean, then predict per passage) never imposes a one-to-one constraint,
so it degrades gracefully. The public notebook actually has this as `--adapt_w` /
`adapt_thr` and the forum's only "test-shaped CV" practitioner reported ordering
agreement in the m=2/3 columns for exactly this soft variant.

This measures, out of fold, at panel sizes 2..18:
    plain | soft-centring only | hard assignment | soft then hard
so I can pick a rule that is safe at every panel size instead of guessing.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)
J = os.environ.get('NTL_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(J, 'jobs'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

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


def centre(X, g):
    X = X.copy()
    for c in np.unique(g):
        m = g == c
        if m.sum() >= 2:
            X[m] -= X[m].mean(0)
    return X


def hung(lp, g):
    p = lp.argmax(1).copy()
    for c in np.unique(g):
        idx = np.where(g == c)[0]
        if 2 <= len(idx) <= lp.shape[1]:
            r, cc = linear_sum_assignment(-lp[idx])
            p[idx[r]] = cc
    return p


def build_features(train_txt, test_txt):
    views = []
    for kw in (dict(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=80000),
               dict(analyzer='word', ngram_range=(1, 2), min_df=3, max_features=60000)):
        v = TfidfVectorizer(sublinear_tf=True, **kw).fit(train_txt)
        views.append((normalize(v.transform(train_txt)), normalize(v.transform(test_txt))))
    from scipy.sparse import hstack
    return hstack([a for a, _ in views]).tocsr(), hstack([b for _, b in views]).tocsr()


def main():
    official = pd.read_parquet(os.path.join(J, 'data/official.parquet'))
    y = official['model'].values
    pg = official['prompt_group'].values
    folds = CV.prompt_folds(official, 5, seed=0)
    res = {}

    print(f'{"k":>3} {"purity":>7} {"plain":>7} {"soft":>7} {"hard":>7} {"soft+hard":>9}')
    for k in (18, 12, 8, 5, 3, 2):
        preds = {m: np.empty(len(official), dtype=object) for m in
                 ('plain', 'soft', 'hard', 'softhard')}
        pur = []
        for fi in folds:
            sub = fi if k >= 18 else CV.sim_subset(fi, official, lo=k, hi=k, seed=0)
            tr = np.setdiff1d(np.arange(len(official)), fi)
            Xtr, Xte = build_features(official['text'].values[tr], official['text'].values[sub])
            g = recover(official['text'].values[sub])
            gtr = pg[tr]
            for c in np.unique(g):
                idx = sub[g == c]
                pur.append(pd.Series(pg[idx]).value_counts().iloc[0] / len(idx))

            # plain
            clf = LogisticRegression(C=5, max_iter=2000).fit(Xtr, y[tr])
            lp_plain = clf.predict_log_proba(Xte)
            preds['plain'][sub] = LAB[lp_plain.argmax(1)]

            # soft: centre both sides by their group means, then predict
            Xtr_d = np.asarray(Xtr.todense(), dtype=np.float32)
            Xte_d = np.asarray(Xte.todense(), dtype=np.float32)
            clf_s = LogisticRegression(C=5, max_iter=2000).fit(centre(Xtr_d, gtr), y[tr])
            lp_soft = clf_s.predict_log_proba(centre(Xte_d, g))
            preds['soft'][sub] = LAB[lp_soft.argmax(1)]
            preds['hard'][sub] = LAB[hung(lp_plain, g)]
            preds['softhard'][sub] = LAB[hung(lp_soft, g)]

        row = {}
        for m, p in preds.items():
            present = ~np.array([x is None for x in p])
            row[m] = CV.evaluate(y[present], p[present])['bacc']
        row['purity'] = float(np.mean(pur))
        res[k] = row
        print(f'{k:>3} {row["purity"]:7.3f} {row["plain"]:7.4f} {row["soft"]:7.4f} '
              f'{row["hard"]:7.4f} {row["softhard"]:9.4f}', flush=True)

    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(J, 'out/transductive_soft.json'), 'w'), indent=1)
    print('\nwrote out/transductive_soft.json')


if __name__ == '__main__':
    main()
