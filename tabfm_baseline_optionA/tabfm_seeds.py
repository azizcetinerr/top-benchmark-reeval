# -*- coding: utf-8 -*-
"""
tabfm_seeds.py — E4: noise floor (GPU)
=========================================
The paper's central claim: on TOP, effect sizes lie below the seed noise.
For HINT a seed sd of ~0.023 ROC is reported. What is TabFM's own seed variance?
And is the phase-III HINT+TabFM fusion gain (~0.03 ROC) above the seed noise,
or inside it?

This script runs TabFM with different random_states (context/OOF change),
measures ROC/PR per seed and reports the seed sd. It also computes, per seed,
the HINT+TabFM fusion gain (stacking) and gives the gain's seed distribution.

Requires GPU (TabFM). Run:
  python tabfm_seeds.py --phase III --seeds 0,1,2,3,4
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
TABFM_DIR = os.path.join(ROOT, "tabfm-main")
PREDS_DIR = os.path.join(HERE, "preds")
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import eval_harness as H    # noqa: E402
import tabfm_baseline as TB  # noqa: E402


def hint_test_valid(phase):
    p = os.path.join(PAPER_DIR, "preds", f"preds_hint_phase_{phase}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p); d["nctid"] = d["nctid"].astype(str)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    phase = args.phase
    seeds = [int(s) for s in args.seeds.split(",")]

    d = lambda s: os.path.join(C.HINT_DIR, "data", f"phase_{phase}_{s}.csv")
    df_tr, df_va, df_te = (pd.read_csv(d(s)) for s in ("train", "valid", "test"))
    Xtr, ytr, _ = TB.build_feature_table(df_tr)
    Xva, yva, idv = TB.build_feature_table(df_va)
    Xte, yte, idt = TB.build_feature_table(df_te)
    cols = Xtr.columns.union(Xva.columns).union(Xte.columns)
    Xtr, Xva, Xte = (X.reindex(columns=cols) for X in (Xtr, Xva, Xte))

    hint = hint_test_valid(phase)

    # Robust to time limits: append each seed to the CSV IMMEDIATELY (don't lose to timeout)
    out_csv = os.path.join(HERE, f"tabfm_seeds_phase_{phase}.csv")
    import csv as _csv
    _hdr = ["seed", "TabFM_ROC", "TabFM_PR"] + (
        ["fusion_ROC", "gain_over_TabFM"] if hint is not None else [])
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="") as f:
            _csv.writer(f).writerow(_hdr)

    rows = []
    for seed in seeds:
        sv, st = TB.fit_predict_tabfm(Xtr, ytr, [Xva, Xte], tabfm_dir=TABFM_DIR,
                                      random_state=seed, verbose=False)
        roc = roc_auc_score(yte, st); pr = average_precision_score(yte, st)
        row = {"seed": seed, "TabFM_ROC": round(roc, 4), "TabFM_PR": round(pr, 4)}

        # HINT+TabFM fusion gain for this seed (stacking)
        if hint is not None:
            hv = hint[hint.split == "valid"].set_index("nctid")["score"]
            ht = hint[hint.split == "test"].set_index("nctid")["score"]
            iv = [n for n in idv if n in hv.index]
            it = [n for n in idt if n in ht.index]
            posv = {n: i for i, n in enumerate(idv)}
            post = {n: i for i, n in enumerate(idt)}
            Xv = np.column_stack([sv[[posv[n] for n in iv]], hv.loc[iv].to_numpy()])
            Xt = np.column_stack([st[[post[n] for n in it]], ht.loc[it].to_numpy()])
            yv2 = yva[[posv[n] for n in iv]]; yt2 = yte[[post[n] for n in it]]
            sc = StandardScaler().fit(Xv)
            lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xv), yv2)
            fus = lr.predict_proba(sc.transform(Xt))[:, 1]
            tabfm_only = st[[post[n] for n in it]]
            row["fusion_ROC"] = round(roc_auc_score(yt2, fus), 4)
            row["gain_over_TabFM"] = round(
                roc_auc_score(yt2, fus) - roc_auc_score(yt2, tabfm_only), 4)
        rows.append(row)
        with open(out_csv, "a", newline="") as f:      # write immediately
            _csv.writer(f).writerow([row.get(k, "") for k in _hdr])
        print(f"[seed {seed}] TabFM ROC={roc:.4f} PR={pr:.4f}"
              + (f" | fusion gain={row.get('gain_over_TabFM')}" if hint is not None else ""))

    res = pd.DataFrame(rows)
    print(f"\n=== TabFM seed variance (phase {phase}, {len(seeds)} seeds) ===")
    print(res.to_string(index=False))
    print(f"\nTabFM ROC: ort={res.TabFM_ROC.mean():.4f}  sd={res.TabFM_ROC.std():.4f}")
    print(f"TabFM PR : ort={res.TabFM_PR.mean():.4f}  sd={res.TabFM_PR.std():.4f}")
    if "gain_over_TabFM" in res:
        g = res["gain_over_TabFM"]
        print(f"Fusion gain (HINT+TabFM - TabFM): mean={g.mean():+.4f}  "
              f"sd={g.std():.4f}  min={g.min():+.4f}  max={g.max():+.4f}")
        print("Reading: if the gain mean > seed sd it is real; if the gain range "
              "spans 0 it is at the noise floor (supports the paper's thesis).")
    print(f"[written] {out_csv} (each seed appended immediately)")


if __name__ == "__main__":
    main()
