"""One config per process: which pool subset trains the 4-view SVC best?

Usage: exp_kb_one.py <all|no_kb|kb_only> [fold]
Writes a JSON per fold immediately, so a late crash cannot lose earlier folds and the
result is visible while it runs.

Why this matters: the public pool marks 72,030 of its rows as generated through the
Kaggle Benchmarks proxy (`flag_kb_proxy`), i.e. a different serving path with different
typography. If the classifier is partly learning "proxy formatting" instead of
authorship, those rows are poison for a hidden test generated the competition's way.
Nobody in the public write-ups has tested this split.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)
J = os.environ.get('NTL_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(J, 'jobs'))
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
OUT = os.path.join(J, 'out/kb')


def key(s):
    return ' '.join(str(s).lower().split())


def main():
    mode = sys.argv[1]
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(OUT, exist_ok=True)

    official = pd.read_parquet(os.path.join(J, 'data/official.parquet'))
    pool = pd.read_parquet(os.path.join(J, 'data/union.parquet'), columns=['text', 'model'])
    src = pd.read_parquet(os.path.join(J, 'data/open_pool_src.parquet'),
                          columns=['text', 'flag_kb_proxy'])
    flags = dict(zip(src.text.map(key), src.flag_kb_proxy))
    pool['kb'] = pool.text.map(key).map(flags)

    if mode == 'no_kb':
        pool = pool[pool.kb != True]
    elif mode == 'kb_only':
        pool = pool[pool.kb == True]
    pool = pool[['text', 'model']]
    print(f'[{mode}] pool rows {len(pool)}', flush=True)

    folds = CV.prompt_folds(official, 5, seed=0)
    for k, fi in enumerate(folds):
        if only is not None and k != only:
            continue
        fp = os.path.join(OUT, f'{mode}_f{k}.json')
        if os.path.exists(fp):
            print(f'  fold {k} already done', flush=True)
            continue
        t0 = time.time()
        tr = np.setdiff1d(np.arange(len(official)), fi)
        fit_txt = np.concatenate([official['text'].values[tr], pool['text'].values])
        fit_y = np.concatenate([official['model'].values[tr], pool['model'].values])
        w = np.concatenate([np.ones(len(tr)), np.full(len(pool), 0.35)])
        te = official['text'].values[fi]
        lp = np.zeros((len(te), len(LAB)))
        for vi, spec in enumerate(VIEWS):
            v = TfidfVectorizer(**spec).fit(fit_txt)
            clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
            clf.fit(normalize(v.transform(fit_txt)), fit_y, sample_weight=w)
            d = clf.decision_function(normalize(v.transform(te)))
            D = np.full((len(te), len(LAB)), -50.0)
            for j, c in enumerate(clf.classes_):
                D[:, list(LAB).index(c)] = d[:, j]
            lp += log_softmax(D * 2.0, 1)
            print(f'    view {vi} done {time.time()-t0:.0f}s', flush=True)
        lp /= len(VIEWS)
        pred = LAB[lp.argmax(1)]
        b = CV.evaluate(official['model'].values[fi], pred)
        bl = CV.board_like_score(official['model'].values[fi], pred,
                                 official['prompt_group'].values[fi], n_reps=20)
        rec = dict(mode=mode, fold=k, rows=int(len(pool)), bacc=float(b['bacc']),
                   board=float(bl['bacc']), board_sd=float(bl['bacc_sd']),
                   minutes=round((time.time() - t0) / 60, 1))
        json.dump(rec, open(fp, 'w'), indent=1)
        np.save(os.path.join(OUT, f'{mode}_f{k}_lp.npy'), lp.astype(np.float32))
        print(f'  fold {k}: bacc={b["bacc"]:.4f} board={bl["bacc"]:.4f} '
              f'({rec["minutes"]}m)', flush=True)


if __name__ == '__main__':
    main()
