"""Gradient-boosted decision: does the 165k corpus actually help the n-gram members?

Trains the same 4-view n-gram LinearSVC family that the winning public pipeline uses,
but with OURS official rows only vs official + the 165k capped corpus, and reports
both full-18 prompt-grouped OOF and the board-like (1-2 answers/prompt) score.

Uses corpus/cv.py as the single validation harness.
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv as CV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import log_softmax

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '.cache')
os.makedirs(CACHE, exist_ok=True)


def views():
    """The 4 n-gram views used by the public 4-view LinearSVC."""
    return [
        ('char', dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=300000,
                      sublinear_tf=True, lowercase=True)),
        ('word', dict(analyzer='word', ngram_range=(1, 3), min_df=3, max_features=200000,
                      sublinear_tf=True, lowercase=True)),
        ('shape', dict(analyzer='word', ngram_range=(2, 6), min_df=3, max_features=100000,
                       sublinear_tf=True, lowercase=False, token_pattern=r'\S+')),
        ('func', dict(analyzer='word', ngram_range=(1, 3), min_df=5, max_features=20000,
                      sublinear_tf=True, lowercase=True, stop_words='english')),
    ]


def build_X(fit_texts, apply_texts, spec):
    v = TfidfVectorizer(**spec)
    v.fit(fit_texts)
    from sklearn.preprocessing import normalize
    return normalize(v.transform(apply_texts)), v


def one_fold(fold_idx, official, pool, seed=0, extra_weight=0.35, use_pool=True,
             max_pool_per_label=None):
    tr_idx = np.setdiff1d(np.arange(len(official)), fold_idx)
    tr_txt = official.text.values[tr_idx]
    tr_y = official.model.values[tr_idx]
    te_txt = official.text.values[fold_idx]
    te_y = official.model.values[fold_idx]

    if use_pool:
        p = pool
        if max_pool_per_label:
            p = p.groupby('model', group_keys=False).apply(
                lambda g: g.sample(min(len(g), max_pool_per_label), random_state=0))
        fit_txt = np.concatenate([tr_txt, p.text.values])
        fit_y = np.concatenate([tr_y, p.model.values])
        w = np.concatenate([np.ones(len(tr_txt)), np.full(len(p), extra_weight)])
    else:
        fit_txt, fit_y, w = tr_txt, tr_y, np.ones(len(tr_txt))

    lp = np.zeros((len(te_txt), len(CV.LABELS)))
    for name, spec in views():
        Xtr, vec = build_X(fit_txt, fit_txt, spec)
        Xte, _ = build_X(fit_txt, te_txt, spec)
        clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, dual=True)
        clf.fit(Xtr, fit_y, sample_weight=w)
        d = clf.decision_function(Xte)
        # align classes to LABELS order
        order = [list(clf.classes_).index(l) if l in clf.classes_ else None for l in CV.LABELS]
        D = np.full((len(te_txt), len(CV.LABELS)), -1e3)
        for j, o in enumerate(order):
            if o is not None:
                D[:, j] = d[:, o]
        lp += log_softmax(D * 2.0, axis=1)
    return lp / len(views()), te_y


def main():
    official = pd.read_parquet(os.path.join(HERE, 'official.parquet'))
    pool = pd.read_parquet(os.path.join(HERE, 'union_capped.parquet'))
    print('official', official.shape, 'pool', pool.shape)

    folds = CV.prompt_folds(official, 5, seed=0)
    configs = [
        ('official_only', dict(use_pool=False)),
        ('pool_all', dict(use_pool=True, max_pool_per_label=None)),
        ('pool_3k', dict(use_pool=True, max_pool_per_label=3000)),
    ]
    results = {}
    for name, kw in configs:
        t0 = time.time()
        preds = np.empty(len(official), dtype=object)
        lps = np.zeros((len(official), 18))
        for k, fi in enumerate(folds):
            lp, ty = one_fold(fi, official, pool, **kw)
            lps[fi] = lp
            preds[fi] = np.array(CV.LABELS)[lp.argmax(1)]
            print(f'  [{name}] fold{k} n={len(fi)} '
                  f'bacc={CV.evaluate(ty, preds[fi])["bacc"]:.4f}  ({time.time()-t0:.0f}s)',
                  flush=True)
        full = CV.evaluate(official.model.values, preds)
        bl = CV.board_like_score(official.model.values, preds,
                                 official.prompt_group.values, n_reps=20, seed=0)
        results[name] = dict(full=full, board=bl)
        print(f'[{name}] FULL bacc={full["bacc"]:.4f} | BOARD-LIKE bacc={bl["bacc"]:.4f} '
              f'sd={bl["bacc_sd"]:.4f}  total={time.time()-t0:.0f}s', flush=True)
        np.save(os.path.join(HERE, f'oof_svc_{name}.npy'), lps)

    print('\n=== summary ===')
    for k, v in results.items():
        print(f'{k:<15s} full={v["full"]["bacc"]:.4f}  board={v["board"]["bacc"]:.4f} '
              f'(sd {v["board"]["bacc_sd"]:.4f})')
    json.dump({k: {'full': v['full']['bacc'], 'board': v['board']['bacc'],
                   'board_sd': v['board']['bacc_sd'],
                   'per_class': v['board']['per_class_recall']} for k, v in results.items()},
              open(os.path.join(HERE, 'svc_pool_experiment.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
