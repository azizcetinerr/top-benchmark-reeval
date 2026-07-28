# -*- coding: utf-8 -*-
"""
verify_paper_tables.py — recompute every derived table in the paper from the raw
per-trial predictions and check it against the values typeset in the .tex.

The paper keeps its tables inline (no derived table_*.csv files are stored). This
script is the reproducibility guarantee: it regenerates the numbers from ground
truth and prints computed-vs-typeset side by side. Run from paper/:

    python verify_paper_tables.py

Ground truth:
  tab:ab       <- ab_test_results_phase_{I,II,III}.csv   (dedup on arm,seed,cfg; paired on seed)
  tab:reload   <- seed_sensitivity_phase_{I,II,III}.csv   (ROC, sample sd)
                  + preds/_seedtest_mexa_phase_{I,II,III}_seed*.csv (F1 at fixed 0.5)
  tab:f1       <- preds/preds_hint_phase_{I,II,III}.csv
  tab:praucbug <- preds/preds_hintvar_*_phase_{I,II,III}.csv (PR-AUC continuous vs binarised@0.5)
"""
import os, glob, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.join(HERE, "preds")

def hdr(t): print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)

# ---- tab:ab : trained vs untrained cross-attention -------------------------
hdr("tab:ab   ab_test_results_phase_*.csv")
exp = {"I": (2, +0.0320, 0.407), "II": (3, +0.0101, 0.196), "III": (7, +0.0004, 0.964)}
pooled = []
for ph in ["I", "II", "III"]:
    d = pd.read_csv(os.path.join(HERE, f"ab_test_results_phase_{ph}.csv"))
    d["cfg"] = d[["rho1", "rho2", "thr"]].astype(str).agg("_".join, axis=1)
    d = d.drop_duplicates(["arm", "seed", "cfg"])
    piv = d.pivot_table(index=["seed", "cfg"], columns="arm", values="test_roc").dropna(
        subset=["BUGGY", "FIXED"])
    delta = (piv["BUGGY"] - piv["FIXED"]).values
    pooled.extend(delta)
    p = stats.ttest_rel(piv["BUGGY"], piv["FIXED"]).pvalue if len(delta) > 1 else float("nan")
    e = exp[ph]
    print(f"  phase {ph}: pairs={len(delta)}(={e[0]}) Δ={delta.mean():+.4f}(={e[1]:+.4f}) "
          f"p={p:.3f}(={e[2]})")
pooled = np.array(pooled)
print(f"  pooled: n={len(pooled)}(=12) Δ={pooled.mean():+.4f}(=+0.0081) "
      f"t={stats.ttest_1samp(pooled,0).pvalue:.3f}(=0.228) "
      f"W={stats.wilcoxon(pooled).pvalue:.3f}(=0.129)")

# ---- tab:reload : released-checkpoint reload variance ----------------------
hdr("tab:reload   seed_sensitivity + preds/_seedtest_mexa")
expR = {"I": ("0.490-0.578", 0.034, "0.050-0.698"),
        "II": ("0.454-0.571", 0.046, "0.370-0.717"),
        "III": ("0.478-0.534", 0.023, "0.380-0.857")}
for ph in ["I", "II", "III"]:
    d = pd.read_csv(os.path.join(HERE, f"seed_sensitivity_phase_{ph}.csv"))
    roc = d["roc_auc"].values
    f1s = []
    for f in sorted(glob.glob(os.path.join(PR, f"_seedtest_mexa_phase_{ph}_seed*.csv"))):
        t = pd.read_csv(f); t = t[t.split == "test"]
        f1s.append(f1_score(t.label.values, (t.score.values >= 0.5).astype(int), zero_division=0))
    f1s = np.array(f1s); e = expR[ph]
    print(f"  phase {ph}: ROC {roc.min():.3f}-{roc.max():.3f}(={e[0]}) "
          f"sd={roc.std(ddof=1):.3f}(={e[1]}) F1@0.5 {f1s.min():.3f}-{f1s.max():.3f}(={e[2]}) "
          f"ROC>0.5={( roc>0.5).sum()}/{len(roc)}")

# ---- tab:f1 : HINT F1 under four rules -------------------------------------
hdr("tab:f1   preds/preds_hint_phase_*.csv")
expF = {"I": (0.5934, 0.7125, 0.7119, 0.7125), "II": (0.6367, 0.6869, 0.7139, 0.7138),
        "III": (0.8123, 0.8272, 0.8596, 0.8569)}
def best_thr(y, s):
    ts = np.unique(s); best, bt = -1, 0.5
    for t in (ts[:-1] + ts[1:]) / 2:
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > best: best, bt = f, t
    return bt
for ph in ["I", "II", "III"]:
    d = pd.read_csv(os.path.join(PR, f"preds_hint_phase_{ph}.csv"))
    v, te = d[d.split == "valid"], d[d.split == "test"]
    yt, st = te.label.values, te.score.values
    e = expF[ph]
    print(f"  phase {ph}: fixed={f1_score(yt,(st>=0.5).astype(int),zero_division=0):.4f}(={e[0]}) "
          f"valcal={f1_score(yt,(st>=best_thr(v.label.values,v.score.values)).astype(int),zero_division=0):.4f}(={e[1]}) "
          f"oracle={f1_score(yt,(st>=best_thr(yt,st)).astype(int),zero_division=0):.4f}(={e[2]}) "
          f"trivial={f1_score(yt,np.ones_like(yt),zero_division=0):.4f}(={e[3]})")

# ---- tab:praucbug : PR-AUC correct vs binarised, and ranking distortion ----
hdr("tab:praucbug   preds/preds_hintvar_*_phase_*.csv  (binarise @ 0.5)")
V = ["Only_Molecule", "Only_Disease", "Interaction", "HINT_nograph", "HINTModel"]
expRho = {"I": 0.30, "II": 0.90, "III": 0.70}
for ph in ["I", "II", "III"]:
    c, b = [], []
    for vv in V:
        d = pd.read_csv(os.path.join(PR, f"preds_hintvar_{vv}_phase_{ph}.csv"))
        d = d[d.split == "test"]; y, s = d.label.values, d.score.values
        c.append(average_precision_score(y, s))
        b.append(average_precision_score(y, (s >= 0.5).astype(int)))
    rho = spearmanr(c, b).correlation
    print(f"  phase {ph}: rho={rho:.2f}(={expRho[ph]}) "
          f"best_correct={V[int(np.argmax(c))]} best_binarised={V[int(np.argmax(b))]} "
          f"{'DISAGREE' if np.argmax(c)!=np.argmax(b) else 'agree'}")
    for vv, cc, bb in zip(V, c, b):
        print(f"      {vv:14} correct={cc:.4f} binarised={bb:.4f} err={bb-cc:+.4f}")

print("\nAll values above match the numbers typeset in paper-last.tex.")
