# -*- coding: utf-8 -*-
"""
paper/baseline_models.py — Simple ML baselines (real baseline)
=========================================================================
WHY: HINT and MEXA have ROC ~0.68. So what comes out if we apply logistic
regression to the same data? If that also gets ~0.68, it means neither deep
model adds anything over a simple linear model — this could be the project's
broadest result.

From MEXA's embedded data (icd 64-d + smiles 15-d + criteria 768-d) it builds a
SINGLE feature vector per trial (mean over tokens, respecting the masks), then
trains classic models. The output is in the same {nctid,score,label,split}
format, so it passes through the COMMON HARNESS — measured with exactly the same
rules as HINT/MEXA.

Usage:
    cd paper
    python baseline_models.py                 # config.PHASE
    CTP_PHASE=I python baseline_models.py
"""
import os
import sys
import csv
import numpy as np


def _pool(tokens, masks):
    """Averages [L, D] tokens, respecting the mask -> [D]."""
    keep = ~masks
    if keep.sum() == 0:
        return tokens.mean(axis=0)
    return tokens[keep].mean(axis=0)


def build_features(phase, split, C):
    """From the MEXA dataset: [N, 64+15+768] feature matrix + label + nctid."""
    sys.path.insert(0, C.MEXA_DIR)
    cwd = os.getcwd()
    os.chdir(C.MEXA_DIR)
    try:
        import json, argparse
        from ctp.dataset import ClinicalTtrialsPredictionDatasetH as CTPDataset

        args = argparse.Namespace(**json.load(
            open(os.path.join(os.path.dirname(C._MEXA_CKPTS[phase]), 'args.json'))))
        max_length = {'icds': args.imax_length, 'smiless': args.smax_length,
                      'in_criteria': args.in_cmax_length, 'ex_criteria': args.ex_cmax_length}
        reduce = {'icds': args.ireduce, 'criteria': args.creduce}

        ds = CTPDataset(C.MEXA_DATA_FOLDER, subset=split, phase=phase,
                        max_length=max_length, reduce=reduce)
        X, y, ids = [], [], []
        for i in range(len(ds)):
            d = ds[i]
            f = np.concatenate([
                _pool(d['itokens'].numpy(),    d['imasks'].numpy()),
                _pool(d['stokens'].numpy(),    d['smasks'].numpy()),
                _pool(d['in_ctokens'].numpy(), d['in_cmasks'].numpy()),
                _pool(d['ex_ctokens'].numpy(), d['ex_cmasks'].numpy()),
            ])
            X.append(f); y.append(int(d['label'])); ids.append(d['nctid'])
        return np.array(X), np.array(y), ids
    finally:
        os.chdir(cwd)


def run(phase=None, seed=None):
    import config as C
    import eval_harness as H
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    phase = phase or C.PHASE
    seed = seed if seed is not None else C.SEED
    print(f'phase {phase} | seed {seed}')

    print('extracting features...')
    Xtr, ytr, _ = build_features(phase, 'train', C)
    Xva, yva, iva = build_features(phase, 'valid', C)
    Xte, yte, ite = build_features(phase, 'test', C)
    print(f'  train {Xtr.shape}  valid {Xva.shape}  test {Xte.shape}')

    models = {
        'LogReg': make_pipeline(SimpleImputer(), StandardScaler(),
                                LogisticRegression(max_iter=2000, random_state=seed)),
        'RandomForest': make_pipeline(SimpleImputer(),
                                      RandomForestClassifier(n_estimators=300,
                                                             random_state=seed, n_jobs=-1)),
        'GradBoost': make_pipeline(SimpleImputer(),
                                   GradientBoostingClassifier(random_state=seed)),
    }

    results = {}
    for name, m in models.items():
        print(f'\n--- training {name}...')
        m.fit(Xtr, ytr)
        out = os.path.join(C.PREDS_DIR, f'preds_{name.lower()}_phase_{phase}.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['nctid', 'score', 'label', 'split'])
            for ids, X, yy, sp in [(iva, Xva, yva, 'valid'), (ite, Xte, yte, 'test')]:
                s = m.predict_proba(X)[:, 1]
                for a, b, c in zip(ids, s, yy):
                    w.writerow([a, float(b), int(c), sp])
        r = H.evaluate_model(out, select_metric=C.SELECT_METRIC,
                             n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=seed)
        results[name] = r
        print(f"  test ROC={r['point']['roc_auc']:.4f}  PR={r['point']['pr_auc']:.4f}  "
              f"F1={r['point']['f1']:.4f}")

    print('\n================ BASELINE TABLE (common harness) ================')
    print(H.compare_table(results, test_labels=yte).to_string(index=False))
    print('\n  Reference (phase III): HINT ROC=0.6851 | MEXA-fixed=0.6776 | '
          'MEXA-buggy=0.6807 | TRIVIAL=0.500')
    print('  -> If simple models also get similar ROC, the deep architectures add nothing.')

    p = os.path.join(C.PAPER_DIR, f'baselines_phase_{phase}.csv')
    H.compare_table(results, test_labels=yte).to_csv(p, index=False)
    print(f'\nsaved -> {p}')
    return results


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
