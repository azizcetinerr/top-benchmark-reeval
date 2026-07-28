# -*- coding: utf-8 -*-
"""
leakage_audit.py — Where does the metadata signal come from? Is there leakage?
============================================================
The ablation showed that just 7 metadata columns give PR-AUC ~0.857 — i.e. all
of the signal is here. In phase II TabFM significantly beat HINT. Decision matrix:
"do not accept it as is, a feature label may be leaking."

This script does NOT REQUIRE a GPU (fast, runs anywhere). Two independent angles:

  1) UNIVARIATE AUC: how well each feature ALONE separates the test label. If a
     feature alone gives AUC ~0.75+, put it under the microscope — either a very
     strong real signal or leakage.

  2) PERMUTATION IMPORTANCE (GBM): shuffles each feature and measures the PR-AUC
     drop. Shows which column the model actually leans on.

It also reports the relationship between the 'phase' category and the label (a
value like phase IV can imply approval — this is the classic TOP leakage).

Run:
  python leakage_audit.py --phase III
  python leakage_audit.py --phase II     # the main suspect phase
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import tabfm_baseline as TB  # noqa: E402

META_COLS = ["phase", "n_drugs", "n_smiles", "n_diseases", "n_icdcodes",
             "icd_chapter", "n_icd_chapters"]


def univariate_auc(Xtr, ytr, Xte, yte):
    """Test ROC-AUC of each numeric feature ALONE (direction-independent)."""
    rows = []
    for c in Xte.columns:
        if Xte[c].dtype == object:            # categorical -> skip (handled separately below)
            continue
        v = Xte[c].to_numpy(dtype=float)
        if np.all(np.isnan(v)) or np.nanstd(v) == 0:
            continue
        v = np.nan_to_num(v, nan=np.nanmedian(v))
        try:
            auc = roc_auc_score(yte, v)
        except ValueError:
            continue
        rows.append({"feature": c, "uni_AUC": round(max(auc, 1 - auc), 4)})
    return pd.DataFrame(rows).sort_values("uni_AUC", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    args = ap.parse_args()
    phase = args.phase

    d = lambda s: os.path.join(C.HINT_DIR, "data", f"phase_{phase}_{s}.csv")
    df_tr, df_te = pd.read_csv(d("train")), pd.read_csv(d("test"))

    Xtr, ytr, _ = TB.build_feature_table(df_tr)
    Xte, yte, _ = TB.build_feature_table(df_te)
    me = [c for c in META_COLS if c in Xtr.columns]
    Xtr, Xte = Xtr[me], Xte[me]

    print(f"\n=== phase {phase} | metadata leakage audit ===")
    print(f"test positive rate = {yte.mean():.3f}")

    # --- phase value vs label relationship (phase IV leakage is classic) ---
    if "phase" in Xtr.columns:
        g = (pd.DataFrame({"phase": df_te["phase"].astype(str), "label": yte})
             .groupby("phase")["label"].agg(["size", "mean"]))
        print("\n[phase -> label] (a single value + an extremely high/low rate is suspicious)")
        print(g.to_string())

    # --- 1) univariate AUC ---
    uni = univariate_auc(Xtr, ytr, Xte, yte)
    print("\n[univariate test AUC] (0.70+ = put under the microscope)")
    print(uni.to_string(index=False))

    # --- 2) GBM permutation importance (categoricals one-hot) ---
    cats = [c for c in ["phase", "icd_chapter"] if c in Xtr.columns]
    Xtr_e = pd.get_dummies(Xtr, columns=cats, dummy_na=False)
    Xte_e = pd.get_dummies(Xte, columns=cats, dummy_na=False)
    cols = Xtr_e.columns.union(Xte_e.columns)
    Xtr_e = Xtr_e.reindex(columns=cols, fill_value=0)
    Xte_e = Xte_e.reindex(columns=cols, fill_value=0)

    clf = HistGradientBoostingClassifier(random_state=0).fit(Xtr_e.values, ytr)
    base = average_precision_score(yte, clf.predict_proba(Xte_e.values)[:, 1])
    print(f"\n[GBM meta-only] test PR-AUC = {base:.4f}  ROC-AUC = "
          f"{roc_auc_score(yte, clf.predict_proba(Xte_e.values)[:, 1]):.4f}")

    r = permutation_importance(clf, Xte_e.values, yte, n_repeats=20,
                               random_state=0,
                               scoring="average_precision")
    imp = (pd.DataFrame({"feature": cols, "PR_drop": r.importances_mean})
           .sort_values("PR_drop", ascending=False).head(15))
    print("\n[permutation importance] (PR-AUC drop — what the model leans on)")
    print(imp.to_string(index=False))

    print("\nReading:")
    print("  * If a single feature alone gives AUC ~0.75+: is it a real strong "
          "signal or leakage? Question its meaning.")
    print("  * If counters like n_diseases/n_icdcodes lead, it is probably a "
          "REAL signal (large/complex trial -> different success).")
    print("  * If the ICD chapter (icd_chapter) leads: a disease-area effect "
          "(e.g. oncology fails more) — this is also real, not leakage.")
    print("  * If an unexpected technical field (a counting quirk) leads, "
          "inspect the data curation.")


if __name__ == "__main__":
    main()
