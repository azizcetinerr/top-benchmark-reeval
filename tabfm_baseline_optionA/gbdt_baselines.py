# -*- coding: utf-8 -*-
"""
gbdt_baselines.py — tuned XGBoost / LightGBM / CatBoost on the Option-A feature table
with the disease-ontology (GRAM) features (the winning set).

Answers reviewer Q1 (tuned, not sklearn defaults), Q2 (the actual tabular SOTA:
XGBoost/LightGBM/CatBoost), Q7 (class-imbalance handling on). Discipline matches the paper:
a small hyper-parameter grid is selected on the VALIDATION split (never on test), then read
once on test with bootstrap 95% CIs and compared to HINT.

Class imbalance: XGB scale_pos_weight = n_neg/n_pos; LGBM class_weight='balanced';
CatBoost auto_class_weights='Balanced'. CatBoost gets phase + ICD chapter as NATIVE
high-cardinality categoricals (Prokhorenkova et al., 2018), the reviewer's specific point.

Run (any phase; CPU is fine, small data is GBDT's home turf):
    python gbdt_baselines.py --phase III
    python gbdt_baselines.py --phase I
    python gbdt_baselines.py --phase II
Needs: pip install xgboost lightgbm catboost rdkit scikit-learn pandas
"""
import argparse, os, sys, itertools, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C
import tabfm_baseline as TB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

HINT = {"I": (0.574, 0.638), "II": (0.621, 0.674), "III": (0.685, 0.852)}  # ROC, PR (run A)
CAT = ["phase", "icd_chapter"]

def boot_ci(y, s, fn, n=1000, seed=2023):
    rng = np.random.default_rng(seed); m = len(y)
    v = [fn(y[i], s[i]) for i in (rng.integers(0, m, m) for _ in range(n))]
    return np.round(np.percentile(v, [2.5, 97.5]), 3)

def align(frames, cols):
    """Reindex all splits to a common column set; GRAM columns differ per split."""
    out = []
    for X in frames:
        X = X.reindex(columns=cols)
        for c in cols:
            X[c] = (X[c].astype(str).fillna("NA") if c in CAT
                    else pd.to_numeric(X[c], errors="coerce").fillna(0))
        out.append(X)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE)
    ph = ap.parse_args().phase
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))
    bf = lambda df: TB.build_feature_table(df, include_gram=True)     # winning feature set
    Xtr, ytr, _ = bf(d("train")); Xva, yva, _ = bf(d("valid")); Xte, yte, itest = bf(d("test"))
    ytr, yva, yte = map(np.asarray, (ytr, yva, yte))
    predir = os.path.join(HERE, "preds"); os.makedirs(predir, exist_ok=True)
    cols = sorted(set(Xtr.columns) | set(Xva.columns) | set(Xte.columns))
    Xtr, Xva, Xte = align([Xtr, Xva, Xte], cols)
    # one-hot for XGB/LGBM (they need numeric); CatBoost uses the raw frames
    A = pd.get_dummies(pd.concat([Xtr, Xva, Xte], keys=["t", "v", "e"]), columns=CAT)
    Etr, Eva, Ete = A.xs("t").values, A.xs("v").values, A.xs("e").values
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    print(f"=== phase {ph} + GRAM ({Etr.shape[1]} feat) | test n={len(yte)} "
          f"pos={yte.mean():.3f} scale_pos_weight={spw:.2f} ===")

    rows = {}
    def report(name, p):
        roc, pr = roc_auc_score(yte, p), average_precision_score(yte, p)
        rl, pl = boot_ci(yte, p, roc_auc_score), boot_ci(yte, p, average_precision_score)
        rows[name] = (roc, rl, pr, pl)
        print(f"  {name:22} ROC={roc:.3f} [{rl[0]},{rl[1]}]   PR={pr:.3f} [{pl[0]},{pl[1]}]")
        # dump per-trial scores so an exact DeLong test vs HINT is possible
        tag = name.split()[0].lower()
        pd.DataFrame({"nctid": itest, "score": p, "label": yte, "split": "test"}).to_csv(
            os.path.join(predir, f"preds_{tag}_phase_{ph}.csv"), index=False)

    # XGBoost -----------------------------------------------------------------
    best = (-1, None, None)
    for md, lr in itertools.product([3, 6], [0.05, 0.1]):
        m = xgb.XGBClassifier(max_depth=md, n_estimators=300, learning_rate=lr,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              tree_method="hist", n_jobs=4, verbosity=0).fit(Etr, ytr)
        va = roc_auc_score(yva, m.predict_proba(Eva)[:, 1])
        if va > best[0]: best = (va, m, (md, lr))
    report("XGBoost (tuned)", best[1].predict_proba(Ete)[:, 1])
    print(f"      [val-selected cfg (md,lr)={best[2]}, val ROC={best[0]:.3f}]")

    # LightGBM ----------------------------------------------------------------
    best = (-1, None, None)
    for nl, lr in itertools.product([31, 63], [0.05, 0.1]):
        m = lgb.LGBMClassifier(num_leaves=nl, n_estimators=300, learning_rate=lr,
                               subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                               n_jobs=4, verbose=-1).fit(Etr, ytr)
        va = roc_auc_score(yva, m.predict_proba(Eva)[:, 1])
        if va > best[0]: best = (va, m, (nl, lr))
    report("LightGBM (tuned)", best[1].predict_proba(Ete)[:, 1])
    print(f"      [val-selected cfg (num_leaves,lr)={best[2]}, val ROC={best[0]:.3f}]")

    # CatBoost (native categoricals) ------------------------------------------
    best = (-1, None, None)
    for dp, lr in itertools.product([4, 6], [0.05, 0.1]):
        m = CatBoostClassifier(depth=dp, iterations=400, learning_rate=lr,
                               auto_class_weights="Balanced", cat_features=CAT,
                               verbose=0, thread_count=4).fit(Xtr, ytr)
        va = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if va > best[0]: best = (va, m, (dp, lr))
    report("CatBoost (tuned)", best[1].predict_proba(Xte)[:, 1])
    print(f"      [val-selected cfg (depth,lr)={best[2]}, val ROC={best[0]:.3f}]")

    print(f"\n  HINT (run A)           ROC={HINT[ph][0]:.3f}           PR={HINT[ph][1]:.3f}")
    print("  Reading: overlapping CIs with HINT => tuned tabular SOTA is statistically "
          "indistinguishable from HINT; the equivalence is not an untuned-baseline artefact.")

if __name__ == "__main__":
    main()
