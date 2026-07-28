# -*- coding: utf-8 -*-
"""
tabfm_single_member.py — how much of TabFM's stability comes from ensembling? (reviewer Q5)

The paper reports TabFM's phase-III ROC-AUC seed sd as 0.0011, but that is with 32 ensemble
members; ensembling shrinks variance, so a fair "stability" comparison must also report the
SINGLE-member sd. This runs TabFM with n_estimators=1 across 5 random_states and reports the
single-member sd next to the 32-member sd.

Needs the jax[cuda12] `tabfm` env + GPU (see D5_QUANTIFY.md / ADAPT_KURULUM.md). Run on g124:
    python tabfm_single_member.py --phase III
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C, tabfm_baseline as TB

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--seeds", default="0,1,2,3,4"); a = ap.parse_args()
    ph = a.phase; seeds = [int(s) for s in a.seeds.split(",")]
    from tabfm import TabFMClassifier, tabfm_v1_0_0_jax as v1
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))
    Xtr, ytr, _ = TB.build_feature_table(d("train"))
    Xte, yte, _ = TB.build_feature_table(d("test"))
    ytr, yte = np.asarray(ytr), np.asarray(yte)

    def roc_at(n_est, seed):
        clf = TabFMClassifier(model=v1.load(), n_estimators=n_est, random_state=seed)
        clf.fit(Xtr, ytr)
        return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    # We already have the 32-member sd (0.0011, tabfm_seeds.py); only the SINGLE-member sd is
    # missing. n_estimators=1 is ~32x cheaper, so this runs even forced onto CPU.
    ens32 = {"I": None, "II": None, "III": 0.0011}.get(ph)   # known 32-member sd (phase III)
    r = np.array([roc_at(1, s) for s in seeds])
    print(f"  n_estimators= 1: ROC mean={r.mean():.4f} sd={r.std(ddof=1):.4f} "
          f"range=[{r.min():.4f},{r.max():.4f}] over seeds {seeds}")
    print(f"  n_estimators=32 (from tabfm_seeds.py): sd={ens32}")
    print("\nReading: report BOTH sds in Section 'Stability'. If the single-member sd is far "
          "larger than the 32-member sd, TabFM's headline stability is largely an ensembling "
          "effect and the comparison to the deep models must be stated at matched ensemble size.\n"
          "If GPU cuDNN fails on this node, force CPU:  JAX_PLATFORMS=cpu python tabfm_single_member.py --phase III")

if __name__ == "__main__":
    main()
