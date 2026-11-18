"""Does MORE pool data keep helping, and where does it saturate?

Yesterday's local run measured, for the 4-view n-gram SVC family:
    official only            0.2615
    + pool capped 165,208    0.4608
That was a jump of +0.199, so the obvious next question is whether the *uncapped*
union (269,120 rows) and per-label sampling ratios go further, or whether 165k was
already past the point of diminishing returns.

This runs entirely on the server (192 cores) with the same cv.py harness. Each config
is a full prompt-grouped 5-fold OOF, so the numbers are directly comparable to the
0.4608 already recorded.

    python exp_svc_scale.py --cfgs 0,3000,6000,9000,15000,0   # 0 = uncapped
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
J = os.environ.get('NTL_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(J, 'jobs'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from scipy.special import log_softmax

LAB = np.array(CV.LABELS)
VIEWS = [
    dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True),
    dict(analyzer='word', ngram_range=(1, 3), min_df=3, max_features=200000, sublinear_tf=True),
    dict(analyzer='word', ngram_range=(2, 6), min_df=3, max_features=100000, sublinear_tf=True,
         lowercase=False, token_pattern=r'\S+'),
    dict(analyzer='word', ngram_range=(1, 3), min_df=5, max_features=20000, sublinear_tf=True,
         stop_words='english'),
]


def sample_per_label(pool, cap, seed=0):
    if cap <= 0:
        return pool
    parts = [g.sample(min(len(g), cap), random_state=seed) for _, g in pool.groupby('model')]
    return pd.concat(parts, ignore_index=True)


def one_fold(official, pool, fold_idx, extra_w=0.35):
    tr = np.setdiff1d(np.arange(len(official)), fold_idx)
    fit_txt = np.concatenate([official['text'].values[tr], pool['text'].values])
    fit_y = np.concatenate([official['model'].values[tr], pool['model'].values])
    w = np.concatenate([np.ones(len(tr)), np.full(len(pool), extra_w)])
    te = official['text'].values[fold_idx]

    lp = np.zeros((len(te), len(LAB)))
    for spec in VIEWS:
        v = TfidfVectorizer(**spec).fit(fit_txt)
        Xtr = normalize(v.transform(fit_txt))
        Xte = normalize(v.transform(te))
        clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
        clf.fit(Xtr, fit_y, sample_weight=w)
        d = clf.decision_function(Xte)
        D = np.full((len(te), len(LAB)), -50.0)
        for j, c in enumerate(clf.classes_):
            D[:, list(LAB).index(c)] = d[:, j]
        lp += log_softmax(D * 2.0, 1)
    return lp / len(VIEWS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfgs', default='0,3000,9000,0')
    ap.add_argument('--pool', default=os.path.join(J, 'data/union.parquet'))
    ap.add_argument('--official', default=os.path.join(J, 'data/official.parquet'))
    ap.add_argument('--out', default=os.path.join(J, 'out/svc_scale.json'))
    ap.add_argument('--fold_seed', type=int, default=0)
    a = ap.parse_args()

    official = pd.read_parquet(a.official)
    pool_all = pd.read_parquet(a.pool)
    print(f'official {len(official)} | pool {len(pool_all)}', flush=True)

    folds = CV.prompt_folds(official, 5, seed=a.fold_seed)
    results = {}
    for cap in [int(x) for x in a.cfgs.split(',')]:
        p = sample_per_label(pool_all, cap, seed=0)
        t0 = time.time()
        preds = np.empty(len(official), dtype=object)
        for k, fi in enumerate(folds):
            lp = one_fold(official, p, fi)
            preds[fi] = LAB[lp.argmax(1)]
            print(f'  cap={cap or "all":>6} fold{k} bacc='
                  f'{CV.evaluate(official["model"].values[fi], preds[fi])["bacc"]:.4f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        full = CV.evaluate(official['model'].values, preds)['bacc']
        board = CV.board_like_score(official['model'].values, preds,
                                    official['prompt_group'].values, n_reps=20)['bacc']
        key = 'all' if cap == 0 else str(cap)
        results[key] = dict(pool_rows=int(len(p)), full=float(full), board=float(board),
                            minutes=round((time.time() - t0) / 60, 1))
        print(f'[cap={key}] rows={len(p)} FULL={full:.4f} BOARD={board:.4f} '
              f'({results[key]["minutes"]}m)', flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)

    print('\n=== summary ===')
    for k, v in results.items():
        print(f'  cap={k:>6} rows={v["pool_rows"]:>7} full={v["full"]:.4f} board={v["board"]:.4f}')


if __name__ == '__main__':
    main()
