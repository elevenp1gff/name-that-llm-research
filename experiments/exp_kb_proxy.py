"""Does the kb_proxy split contain a usable signal?

Hypothesis worth testing cheaply on the server (CPU, no GPU needed):
`flag_kb_proxy` marks passages generated through Kaggle Benchmarks, i.e. a different
serving path whose typography differs. A classifier that has learned "this text looks
like the proxy" may be keying on formatting rather than authorship, which would HURT on
the real hidden test (which comes from the competition's own generation path).

Two measurements, both with the real harness:
  A) 4-view SVC trained WITH vs WITHOUT kb_proxy rows -> full prompt-grouped OOF.
     If dropping them helps, the public pool is partly poisoned for this task and the
     fix is free.
  B) the same models evaluated separately on kb_proxy vs non-kb_proxy official-like
     rows, to see whether the proxy rows are just harder/easier.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)
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


def fit_predict(official, pool, fold_idx, extra_w=0.35):
    tr = np.setdiff1d(np.arange(len(official)), fold_idx)
    fit_txt = np.concatenate([official['text'].values[tr], pool['text'].values])
    fit_y = np.concatenate([official['model'].values[tr], pool['model'].values])
    w = np.concatenate([np.ones(len(tr)), np.full(len(pool), extra_w)])
    te = official['text'].values[fold_idx]
    lp = np.zeros((len(te), len(LAB)))
    for spec in VIEWS:
        v = TfidfVectorizer(**spec).fit(fit_txt)
        clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
        clf.fit(normalize(v.transform(fit_txt)), fit_y, sample_weight=w)
        d = clf.decision_function(normalize(v.transform(te)))
        D = np.full((len(te), len(LAB)), -50.0)
        for j, c in enumerate(clf.classes_):
            D[:, list(LAB).index(c)] = d[:, j]
        lp += log_softmax(D * 2.0, 1)
    return lp / len(VIEWS)


def main():
    official = pd.read_parquet(os.path.join(J, 'data/official.parquet'))
    pool = pd.read_parquet(os.path.join(J, 'data/union.parquet'))
    # union.parquet has no kb_proxy flag; rebuild it from the source pool file
    src = pd.read_parquet(os.path.join(J, 'data/open_pool_src.parquet'),
                          columns=['text', 'model', 'flag_kb_proxy'])
    key = lambda s: ' '.join(str(s).lower().split())
    flags = dict(zip(src.text.map(key), src.flag_kb_proxy))
    pool['kb'] = pool.text.map(key).map(flags)
    print('pool', len(pool), '| kb_proxy known', int(pool.kb.notna().sum()),
          '| kb rows', int((pool.kb == True).sum()), flush=True)

    folds = CV.prompt_folds(official, 5, seed=0)
    cfgs = {
        'union_all': pool,
        'union_no_kb': pool[pool.kb != True],
        'union_kb_only': pool[pool.kb == True],
    }
    res = {}
    for name, p in cfgs.items():
        t0 = time.time()
        preds = np.empty(len(official), dtype=object)
        for fi in folds:
            lp = fit_predict(official, p, fi)
            preds[fi] = LAB[lp.argmax(1)]
        full = CV.evaluate(official['model'].values, preds)['bacc']
        board = CV.board_like_score(official['model'].values, preds,
                                    official['prompt_group'].values, n_reps=20)['bacc']
        res[name] = dict(rows=int(len(p)), full=float(full), board=float(board),
                         min=round((time.time() - t0) / 60, 1))
        print(f'[{name}] rows={len(p)} FULL={full:.4f} BOARD={board:.4f} '
              f'({res[name]["min"]}m)', flush=True)

    print('\n=== summary ===')
    for k, v in res.items():
        print(f'  {k:<14s} rows={v["rows"]:>7} full={v["full"]:.4f} board={v["board"]:.4f}')
    import json
    json.dump(res, open(os.path.join(J, 'out/kb_proxy_study.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
