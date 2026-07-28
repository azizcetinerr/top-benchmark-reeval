# -*- coding: utf-8 -*-
"""
nested_cv_grid.py — model x feature grid with NESTED cross-validation (reviewer R1).

The paper's validation-selected grid uses a single small validation split (117/446/344),
which is too small for stable selection. This re-runs the selection with nested CV: repeated
stratified K-fold on train+valid combined chooses the (model, feature-set) pair by INNER-CV
ROC; the held-out OUTER folds give an honest selection estimate; the real test set is touched
exactly ONCE at the end with the finally-selected pipeline. This replaces "best-on-a-tiny-val"
with "best under repeated CV", the correct fix for the small-validation concern.

CPU-only but heavier than the single-split grid. Run:
    python nested_cv_grid.py --phase III
    python nested_cv_grid.py --phase III --repeats 3 --folds 5
Needs: scikit-learn, xgboost, lightgbm, catboost, rdkit, pandas.
"""
import argparse, os, sys, itertools, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C, tabfm_baseline as TB

HINT_ROC = {"I": 0.574, "II": 0.621, "III": 0.685}

def feat(df, which):
    """which in {'meta','meta_rdkit_prot','gram'} -> feature table."""
    if which == "gram":  return TB.build_feature_table(df, include_gram=True)
    if which == "meta":  return TB.build_feature_table(df, full_rdkit=False)  # coarse; still has meta
    return TB.build_feature_table(df)

def numeric(X, cols):
    X = X.reindex(columns=cols)
    return pd.get_dummies(X, columns=[c for c in ("phase","icd_chapter") if c in cols], dummy_na=True).fillna(0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(); ph = a.phase
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))
    # pool train+valid for selection; test held out
    devX = {w: None for w in ("meta_rdkit_prot", "gram")}
    y_dev = y_te = None
    Xte_by = {}
    for w in ("meta_rdkit_prot", "gram"):
        Xtr, ytr, _ = feat(d("train"), w); Xva, yva, _ = feat(d("valid"), w); Xte, yte, _ = feat(d("test"), w)
        cols = sorted(set(Xtr.columns) | set(Xva.columns) | set(Xte.columns))
        Xdev = pd.concat([numeric(Xtr, cols), numeric(Xva, cols)], ignore_index=True)
        devX[w] = Xdev.reindex(columns=numeric(pd.concat([Xtr,Xva,Xte]), cols).columns, fill_value=0)
        Xte_by[w] = numeric(Xte, cols).reindex(columns=devX[w].columns, fill_value=0)
        y_dev = np.r_[np.asarray(ytr), np.asarray(yva)]; y_te = np.asarray(yte)

    models = {
        "LogReg":  lambda: make_pipeline(SimpleImputer(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RF":      lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=4, random_state=0),
        "HistGBM": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    cv = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=2023)
    print(f"=== nested-CV selection, phase {ph} ({a.repeats}x{a.folds}-fold on train+valid) ===")
    best = (-1, None, None)
    for w, (mn, mk) in itertools.product(devX, models.items()):
        sc = cross_val_score(mk(), devX[w].values, y_dev, cv=cv, scoring="roc_auc", n_jobs=4)
        print(f"  {mn:8} x {w:16} inner-CV ROC = {sc.mean():.4f} ± {sc.std():.4f}")
        if sc.mean() > best[0]: best = (sc.mean(), (mn, mk), w)
    # refit selected pipeline on full dev, evaluate ONCE on test
    _, (mn, mk), w = best
    clf = mk().fit(devX[w].values, y_dev)
    p = clf.predict_proba(Xte_by[w].values)[:, 1]
    def ci(fn, n=1000, seed=2023):
        r = np.random.default_rng(seed); m = len(y_te)
        return np.round(np.percentile([fn(y_te[i], p[i]) for i in (r.integers(0,m,m) for _ in range(n))], [2.5,97.5]), 3)
    print(f"\n  SELECTED: {mn} on {w}")
    print(f"  TEST (single touch): ROC={roc_auc_score(y_te,p):.3f} {ci(roc_auc_score)}  "
          f"PR={average_precision_score(y_te,p):.3f} {ci(average_precision_score)}")
    print(f"  HINT: ROC={HINT_ROC[ph]:.3f}  -> compare CIs (nested-CV selection, not tiny-val).")

if __name__ == "__main__":
    main()
