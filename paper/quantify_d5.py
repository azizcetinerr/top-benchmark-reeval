# -*- coding: utf-8 -*-
"""
quantify_d5.py — how much does HINT's dead eligibility-criteria branch (defect D5) cost?

D5: the shipped data/sentence2embedding.pkl is empty, so protocol2feature feeds a zero
768-vector to HINT's criteria branch for every trial. This script quantifies the effect by
comparing HINT evaluated with the EMPTY cache (the current preds_hint_phase_X.csv) against
HINT evaluated with a POPULATED cache (preds_hint_criteria_phase_X.csv, produced by the steps
in D5_QUANTIFY.md), and asks whether restoring the criteria modality moves HINT toward MEXA.

This comparison step is CPU-only. Generating preds_hint_criteria_* requires BioBERT + the HINT
stack on the cluster (see D5_QUANTIFY.md); it is inference-only (released checkpoint, no
retraining) for the quick signal, or a full retrain for the definitive number.

Usage:  python quantify_d5.py --phase III
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C
PR = os.path.join(HERE, "preds")

def load(path):
    d = pd.read_csv(path); d["nctid"] = d["nctid"].astype(str)
    d = d[d.split == "test"].drop_duplicates("nctid").set_index("nctid")
    return d["score"].astype(float), d["label"].astype(int)

def pboot(y, sa, sb, metric, n=2000, seed=2023):
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    rng = np.random.default_rng(seed); m = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, m, m); yy = y[i]
        if yy.min() == yy.max(): continue
        d.append(fn(yy, sa[i]) - fn(yy, sb[i]))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(fn(y, sa) - fn(y, sb)), lo, hi, min(p, 1.0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE)
    ph = ap.parse_args().phase
    p_empty = os.path.join(PR, f"preds_hint_phase_{ph}.csv")            # current (D5 active)
    p_crit  = os.path.join(PR, f"preds_hint_criteria_phase_{ph}.csv")   # populated cache
    p_mexa  = os.path.join(PR, f"preds_mexa_phase_{ph}.csv")
    if not os.path.exists(p_crit):
        sys.exit(f"[!] {p_crit} not found. First produce it with the D5_QUANTIFY.md steps (cluster).")

    s_e, y = load(p_empty); s_c, _ = load(p_crit)
    common = s_e.index.intersection(s_c.index)
    y = y.loc[common].to_numpy(); s_e = s_e.loc[common].to_numpy(); s_c = s_c.loc[common].to_numpy()
    print(f"=== D5 quantification, phase {ph} (n={len(y)}) ===")
    for metric in ("roc_auc", "pr_auc"):
        fn = roc_auc_score if metric == "roc_auc" else average_precision_score
        d, lo, hi, p = pboot(y, s_c, s_e, metric)
        print(f"  HINT(cache) - HINT(empty)  {metric}: {fn(y,s_c):.4f} vs {fn(y,s_e):.4f}  "
              f"Δ={d:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.3f}")

    if os.path.exists(p_mexa):
        s_m, y_m = load(p_mexa)
        cc = common.intersection(s_m.index)                 # HINT∩MEXA common test set
        se = pd.Series(s_e, index=common).loc[cc].to_numpy()
        sc = pd.Series(s_c, index=common).loc[cc].to_numpy()
        sm = s_m.loc[cc].to_numpy()
        yc = y_m.loc[cc].to_numpy()
        print(f"\n  --- vs MEXA on common set (n={len(cc)}) ---")
        for tag, s in [("HINT(empty)", se), ("HINT(cache)", sc)]:
            d, lo, hi, p = pboot(yc, s, sm, "roc_auc")
            print(f"  {tag} - MEXA roc_auc: Δ={d:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.3f}")
        print("\nReading: if HINT(cache) >> HINT(empty), D5 materially degraded HINT in our eval; "
              "if the HINT(cache)-MEXA gap widens, part of the published MEXA>HINT edge is D5.")

if __name__ == "__main__":
    main()
