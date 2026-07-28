# -*- coding: utf-8 -*-
"""
paper/icd_representation_ablation.py — Disease representation ablation (Item 5)
===========================================================================
WHY: Finding A showed across three phases that ALL of HINT/MEXA's predictive
power comes from the ICD codes (drug structure and protocol text add nothing).
So the critical question is:

    How good does the disease representation need to be? Is MEXA's icd2vec
    (node2vec, no longer recommended by its author) really better than a simple
    one-hot?

Representations compared (all with the SAME classifier + SAME harness):
  icd2vec_64   : MEXA's published representation (node2vec, 64-d) — reference
  chapter      : ICD chapter (first letter, ~22 dims)   -> COARSEST
  category3    : ICD 3-character category (e.g. 'C50', ~1-2k dims)
  full_code    : full ICD code (e.g. 'C50.911', ~5k dims) -> FINEST

This ablation distinguishes three possible results:
  (a) icd2vec >> one-hot   -> the embedding method matters, worth trying a modern one
  (b) icd2vec ~= one-hot   -> the embedding method is irrelevant; the signal is
                              coarse, not architecture/input. PROBLEM-limiting
  (c) chapter ~= full_code -> the signal is very coarse: just "which disease family"

Usage:
    cd paper
    python icd_representation_ablation.py
    CTP_PHASE=II python icd_representation_ablation.py
"""
import os
import ast
import csv


def _parse_codes(cell):
    """HINT icdcodes cell -> flat list of ICD codes."""
    try:
        outer = ast.literal_eval(cell)
    except Exception:
        return []
    out = []
    for grp in outer:
        try:
            out += (ast.literal_eval(grp) if isinstance(grp, str) else list(grp))
        except Exception:
            pass
    return [c.strip() for c in out if isinstance(c, str) and c.strip()]


def load_raw(phase, split, C):
    """HINT raw CSV -> (nctid list, code lists, labels)."""
    p = os.path.join(C.HINT_DIR, 'data', f'phase_{phase}_{split}.csv')
    ids, codes, y = [], [], []
    for r in csv.DictReader(open(p, newline='')):
        ids.append(r['nctid']); codes.append(_parse_codes(r['icdcodes']))
        y.append(int(r['label']))
    return ids, codes, y


def _key(code, level):
    if level == 'chapter':
        return code[0]
    if level == 'category3':
        return code.split('.')[0][:3]
    return code                      # full_code


def build_multihot(train_codes, other_codes_list, level):
    """The vocabulary is built ONLY from train (no leakage)."""
    import numpy as np
    vocab = sorted({_key(c, level) for codes in train_codes for c in codes})
    idx = {k: i for i, k in enumerate(vocab)}

    def enc(codes_list):
        X = np.zeros((len(codes_list), len(vocab)), dtype=np.float32)
        for i, codes in enumerate(codes_list):
            for c in codes:
                j = idx.get(_key(c, level))
                if j is not None:
                    X[i, j] = 1.0
        return X

    return [enc(cl) for cl in [train_codes] + other_codes_list], len(vocab)


def build_icd2vec(phase, C, id_order):
    """ONLY the ICD part (first 64 dims) from the MEXA data — the published representation.

    --- FIX (row alignment) ---
    The MEXA dataset returns the .pkl files in GLOB ORDER; the raw HINT CSV is in
    its own row order. These two orders are NOT the SAME. In a previous version the
    features were taken from the MEXA order and the labels/nctids from the CSV order
    -> labels did not match the features and ROC collapsed to chance (~0.50). Now a
    nctid -> vector dictionary is built and indexed in the CSV order.
    """
    import numpy as np
    import baseline_models as B
    out = []
    for split, ids_wanted in zip(['train', 'valid', 'test'], id_order):
        X, y, ids = B.build_features(phase, split, C)
        lut = {nct: X[i, :64] for i, nct in enumerate(ids)}     # first 64 = icd2vec
        missing = [n for n in ids_wanted if n not in lut]
        if missing:
            print(f'    [warning] {split}: {len(missing)} nctid not in the MEXA data')
        dim = X.shape[1] and 64
        out.append(np.array([lut.get(n, np.zeros(dim, dtype=np.float32))
                             for n in ids_wanted]))
    return out


def main(phase=None, seeds=None, n_boot=None):
    """seeds: list (e.g. [2023,2024,2025,2026,2027]). A single element -> a single run.

    CONTROL 1 (multi-seed): a single seed can be misleading;
      the main finding is repeated over 5 seeds to check robustness to seed noise.
    CONTROL 2 (apples-to-apples): HINT is a NEURAL NETWORK; comparing one-hot only
      with tree/linear models is open to the 'different classifier family' objection.
      So an MLP is also trained on the 22 binary features -> neural vs neural.
    """
    import numpy as np
    import pandas as pd
    import config as C
    import eval_harness as H
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    phase = phase or C.PHASE
    seeds = seeds or [C.SEED]
    n_boot = n_boot if n_boot is not None else (200 if len(seeds) > 1 else C.BOOTSTRAP_N)
    print(f'phase {phase} | seeds {seeds} | n_boot {n_boot}')

    tr_ids, tr_codes, ytr = load_raw(phase, 'train', C)
    va_ids, va_codes, yva = load_raw(phase, 'valid', C)
    te_ids, te_codes, yte = load_raw(phase, 'test', C)
    ytr, yva, yte = map(np.array, (ytr, yva, yte))
    print(f'  train {len(ytr)} | valid {len(yva)} | test {len(yte)}')

    reps = {}
    for level in ['chapter', 'category3', 'full_code']:
        (Xtr, Xva, Xte), n = build_multihot(tr_codes, [va_codes, te_codes], level)
        reps[level] = (Xtr, Xva, Xte)
        print(f'  {level:11s}: {n} dims')

    print('  icd2vec_64 : extracting from the MEXA data (aligned by nctid)...')
    a, b, c = build_icd2vec(phase, C, [tr_ids, va_ids, te_ids])
    reps['icd2vec_64'] = (a, b, c)

    def make_clfs(sd):
        return {
            'LogReg': make_pipeline(SimpleImputer(), StandardScaler(with_mean=False),
                                    LogisticRegression(max_iter=3000, random_state=sd)),
            'RandForest': make_pipeline(SimpleImputer(),
                                        RandomForestClassifier(n_estimators=300,
                                                               random_state=sd, n_jobs=-1)),
            # CONTROL 2: neural classifier -> same family as HINT
            'MLP': make_pipeline(SimpleImputer(), StandardScaler(with_mean=False),
                                 MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                               early_stopping=True, random_state=sd)),
        }

    rows = []
    for sd in seeds:
        for rep, (Xtr, Xva, Xte) in reps.items():
            for cname, m in make_clfs(sd).items():
                m.fit(Xtr, ytr)
                out = os.path.join(
                    C.PREDS_DIR,
                    f'preds_icdrep_{rep}_{cname.lower()}_s{sd}_phase_{phase}.csv')
                with open(out, 'w', newline='') as f:
                    w = csv.writer(f); w.writerow(['nctid', 'score', 'label', 'split'])
                    for ids, X, yy, sp in [(va_ids, Xva, yva, 'valid'),
                                           (te_ids, Xte, yte, 'test')]:
                        s = m.predict_proba(X)[:, 1]
                        for i, sc, lb in zip(ids, s, yy):
                            w.writerow([i, float(sc), int(lb), sp])
                r = H.evaluate_model(out, select_metric=C.SELECT_METRIC,
                                     n_boot=n_boot, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
                rows.append(dict(representation=rep, classifier=cname, seed=sd,
                                 test_roc=round(r['point']['roc_auc'], 4),
                                 test_pr=round(r['point']['pr_auc'], 4),
                                 test_f1=round(r['point']['f1'], 4)))
        print(f'  seed {sd} done')

    df = pd.DataFrame(rows)
    agg = (df.groupby(['representation', 'classifier'])
             .agg(roc_mean=('test_roc', 'mean'), roc_std=('test_roc', 'std'),
                  pr_mean=('test_pr', 'mean'), f1_mean=('test_f1', 'mean'),
                  n=('test_roc', 'size'))
             .reset_index().sort_values('roc_mean', ascending=False))
    agg[['roc_mean', 'roc_std', 'pr_mean', 'f1_mean']] = \
        agg[['roc_mean', 'roc_std', 'pr_mean', 'f1_mean']].round(4)

    print(f'\n=========== ICD REPRESENTATION ABLATION (phase {phase}, '
          f'{len(seeds)} seed) ===========')
    print(agg.to_string(index=False))

    # --- CONTROL 1: does the representation matter? (across seeds) ---
    print('\n--- CONTROL 1: does the representation matter? (per classifier) ---')
    for cname in ['LogReg', 'RandForest', 'MLP']:
        s = agg[agg.classifier == cname]
        if len(s) == 0:
            continue
        i2v = s[s.representation == 'icd2vec_64']
        ch = s[s.representation == 'chapter']
        if len(i2v) and len(ch):
            g = float(i2v.roc_mean.iloc[0]) - float(ch.roc_mean.iloc[0])
            pooled = float(np.hypot(i2v.roc_std.iloc[0] or 0, ch.roc_std.iloc[0] or 0))
            print(f'  {cname:11s} icd2vec={float(i2v.roc_mean.iloc[0]):.4f}  '
                  f'chapter={float(ch.roc_mean.iloc[0]):.4f}  diff={g:+.4f}  '
                  f'(pooled seed sd {pooled:.4f})')

    # --- CONTROL 2: neural vs neural ---
    print('\n--- CONTROL 2: neural-vs-neural (chapter one-hot + MLP) ---')
    mlp_ch = agg[(agg.representation == 'chapter') & (agg.classifier == 'MLP')]
    if len(mlp_ch):
        v = float(mlp_ch.roc_mean.iloc[0]); sdv = float(mlp_ch.roc_std.iloc[0] or 0)
        ref = {'I': 0.5740, 'II': 0.6210, 'III': 0.6851}[phase]
        print(f'  chapter one-hot + MLP (22 binary features): {v:.4f} ± {sdv:.4f}')
        print(f'  HINT (full architecture, same harness)    : {ref:.4f}')
        print(f'  diff (HINT - one-hot MLP) = {ref - v:+.4f}')
        print('  -> Same classifier family; if a gap remains, the architecture contributes.')

    p = os.path.join(C.PAPER_DIR, f'icd_representation_phase_{phase}.csv')
    agg.to_csv(p, index=False)
    praw = os.path.join(C.PAPER_DIR, f'icd_representation_raw_phase_{phase}.csv')
    df.to_csv(praw, index=False)
    print(f'\nsaved -> {p}\n      -> {praw} (per run)')
    return agg


if __name__ == '__main__':
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument('--phase', default=None)
    _p.add_argument('--seeds', type=int, default=5, help='how many seeds (2023..2023+n-1)')
    _p.add_argument('--n_boot', type=int, default=None)
    _a = _p.parse_args()
    main(_a.phase, [2023 + i for i in range(_a.seeds)], _a.n_boot)
