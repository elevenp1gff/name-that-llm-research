"""Can a multi-view / multi-pool BAG of n-gram models beat the single 4-view SVC?

The n-gram side is the one place I hold a real data advantage (269k union pool, +0.028
over the public pool), and in the public blend it carries the single largest weight
(`svc_w = 12`, against a 3.0 LoRA member). But it is exactly ONE model: four TF-IDF
views concatenated into one LinearSVC.

Two cheap upgrades that the public recipe does not try:

  A) per-view BAG: train each view as its own LinearSVC, average their log-probs,
     instead of concatenating all views into one learner. Different views make
     different mistakes, so averaging them is free decorrelation.
  B) pool BAG: train on disjoint halves of the union pool (two seeds) and average.
     The pool has 54k prompts; two half-pools are nearly independent samples of the
     same distribution, which is textbook bagging on the axis where I have an edge.

Measured with the competition harness (prompt-grouped 5 folds, board-like scoring),
including the soft-centring variant that measured positive at every panel size.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)
J = os.environ.get('NTL_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(J, 'jobs'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from scipy.special import log_softmax
from scipy.sparse import hstack

LAB = np.array(CV.LABELS)

VIEWS = [
    ('char', dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=300000,
                  sublinear_tf=True)),
    ('word', dict(analyzer='word', ngram_range=(1, 3), min_df=3, max_features=200000,
                  sublinear_tf=True)),
    ('shape', dict(analyzer='word', ngram_range=(2, 6), min_df=3, max_features=100000,
                   sublinear_tf=True, lowercase=False, token_pattern=r'\S+')),
    ('func', dict(analyzer='word', ngram_range=(1, 3), min_df=5, max_features=20000,
                  sublinear_tf=True, stop_words='english')),
]


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


def recover(texts, thr=0.05):
    vec = TfidfVectorizer(lowercase=True, stop_words=list(ENGLISH_STOP_WORDS),
                          ngram_range=(1, 2), min_df=1 if len(texts) < 20 else 2,
                          max_df=0.9 if len(texts) < 20 else 0.4, sublinear_tf=True)
    X = normalize(vec.fit_transform(texts))
    return capped_agglo(np.asarray((X @ X.T).todense()), 18, thr)


def centre_sparse(X, g):
    X = X.tocsc(copy=True)
    for c in np.unique(g):
        idx = np.where(g == c)[0]
        if len(idx) >= 2:
            X[idx, :] = X[idx, :] - np.asarray(X[idx, :].mean(0))
    return X.tocsr()


def align(clf, d, n_lab=18):
    D = np.full((d.shape[0], n_lab), -50.0)
    for j, c in enumerate(clf.classes_):
        D[:, list(LAB).index(c)] = d[:, j]
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=os.path.join(J, 'data/union.parquet'))
    ap.add_argument('--pool_cap', type=int, default=60000, help='rows total, 0=all')
    ap.add_argument('--extra_w', type=float, default=0.35)
    ap.add_argument('--centring', type=int, default=1)
    ap.add_argument('--thr', type=float, default=0.05)
    ap.add_argument('--out', default=os.path.join(J, 'out/bag_svc.json'))
    a = ap.parse_args()

    official = pd.read_parquet(os.path.join(J, 'data/official.parquet'))
    pool = pd.read_parquet(a.pool, columns=['text', 'model', 'prompt_group'])
    if a.pool_cap and len(pool) > a.pool_cap:
        pool = pool.groupby('model', group_keys=False).apply(
            lambda g: g.sample(min(len(g), a.pool_cap // 18), random_state=0))
        pool = pool.reset_index(drop=True)
    print(f'official {len(official)} | pool {len(pool)}', flush=True)

    y = official['model'].values
    pg = official['prompt_group'].values
    folds = CV.prompt_folds(official, 5, seed=0)

    configs = ['concat', 'bag_views', 'bag_pool2', 'bag_both']
    results = {}
    for cfg in configs:
        t0 = time.time()
        pred = np.empty(len(official), dtype=object)
        lp_all = np.zeros((len(official), 18), dtype=np.float32)
        for fi in folds:
            tr = np.setdiff1d(np.arange(len(official)), fi)
            fit_txt = np.concatenate([official['text'].values[tr], pool['text'].values])
            fit_y = np.concatenate([official['model'].values[tr], pool['model'].values])
            w = np.concatenate([np.ones(len(tr)), np.full(len(pool), a.extra_w)])
            te_txt = official['text'].values[fi]
            g = recover(te_txt, a.thr) if a.centring else np.arange(len(te_txt))
            gtr = pg[tr] if a.centring else np.arange(len(tr))

            vecs, Xtr_list, Xte_list = {}, [], []
            for name, spec in VIEWS:
                v = TfidfVectorizer(**spec).fit(fit_txt)
                vecs[name] = v
                Xtr_list.append(normalize(v.transform(fit_txt)))
                Xte_list.append(normalize(v.transform(te_txt)))

            lp = np.zeros((len(te_txt), 18))
            if cfg in ('concat', 'bag_pool2'):
                # one learner over all views concatenated
                A = hstack(Xtr_list).tocsr()
                B = hstack(Xte_list).tocsr()
                if a.centring:
                    A = centre_sparse(A, gtr); B = centre_sparse(B, g)
                clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
                clf.fit(A, fit_y, sample_weight=w)
                lp += log_softmax(align(clf, clf.decision_function(B)) * 2.0, 1)
            elif cfg in ('bag_views', 'bag_both'):
                for vi in range(len(VIEWS)):
                    A, B = Xtr_list[vi], Xte_list[vi]
                    if a.centring:
                        A = centre_sparse(A, gtr); B = centre_sparse(B, g)
                    clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
                    clf.fit(A, fit_y, sample_weight=w)
                    lp += log_softmax(align(clf, clf.decision_function(B)) * 2.0, 1)
                lp /= len(VIEWS)

            if cfg in ('bag_pool2', 'bag_both'):
                # second bag: pool rows split by prompt_group hash so the two halves are
                # disjoint at the PROMPT level (not the row level)
                h = pool['prompt_group'].astype(str).str.len() % 2
                p2 = pool[h == 1].reset_index(drop=True)
                fit2 = np.concatenate([official['text'].values[tr], p2['text'].values])
                y2 = np.concatenate([official['model'].values[tr], p2['model'].values])
                w2 = np.concatenate([np.ones(len(tr)), np.full(len(p2), a.extra_w)])
                lp2 = np.zeros_like(lp)
                if cfg == 'bag_both':
                    for vi in range(len(VIEWS)):
                        v = TfidfVectorizer(**VIEWS[vi][1]).fit(fit2)
                        A = normalize(v.transform(fit2)); B = normalize(v.transform(te_txt))
                        if a.centring:
                            A = centre_sparse(A, gtr); B = centre_sparse(B, g)
                        clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
                        clf.fit(A, y2, sample_weight=w2)
                        lp2 += log_softmax(align(clf, clf.decision_function(B)) * 2.0, 1)
                    lp2 /= len(VIEWS)
                else:
                    A = hstack([normalize(v.transform(fit2)) for _, v in
                                [(n, TfidfVectorizer(**s).fit(fit2)) for n, s in VIEWS]]).tocsr()
                    B = hstack([normalize(TfidfVectorizer(**s).fit(fit2).transform(te_txt))
                                for n, s in VIEWS]).tocsr()
                    if a.centring:
                        A = centre_sparse(A, gtr); B = centre_sparse(B, g)
                    clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
                    clf.fit(A, y2, sample_weight=w2)
                    lp2 += log_softmax(align(clf, clf.decision_function(B)) * 2.0, 1)
                lp = 0.5 * (lp + lp2)

            lp_all[fi] = lp.astype(np.float32)
            pred[fi] = LAB[lp.argmax(1)]
            print(f'  [{cfg}] fold{fi[0] if len(fi) else "?"} done {time.time()-t0:.0f}s',
                  flush=True)

        full = CV.evaluate(y, pred)['bacc']
        board = CV.board_like_score(y, pred, pg, n_reps=20)
        results[cfg] = dict(full=float(full), board=float(board['bacc']),
                            board_sd=float(board['bacc_sd']),
                            minutes=round((time.time() - t0) / 60, 1))
        print(f'[{cfg}] FULL={full:.4f} BOARD={board["bacc"]:.4f} '
              f'(sd {board["bacc_sd"]:.4f}) {results[cfg]["minutes"]}m', flush=True)
        np.save(os.path.join(J, f'out/bagsvc_{cfg}.npy'), lp_all)
        json.dump(results, open(a.out, 'w'), indent=1)

    print('\n=== summary ===')
    for k, v in results.items():
        print(f'  {k:<10s} full={v["full"]:.4f} board={v["board"]:.4f}')
    json.dump(results, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
