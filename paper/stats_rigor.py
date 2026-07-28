# -*- coding: utf-8 -*-
"""
stats_rigor.py — DeLong ΔAUC + CI, TOST equivalence, minimum detectable effect (MDE),
balanced accuracy + MCC, and PR-AUC vs prevalence. Reviewer stats batch.

Fast DeLong (Sun & Xu, 2014) gives the variance of a single AUC and of the DIFFERENCE of two
correlated AUCs on the same labels -- standard and far cheaper than bootstrap.
"""
import os, sys, numpy as np, pandas as pd
from scipy import stats as st
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, f1_score)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C
PR = os.path.join(HERE, "preds"); A = os.path.join(os.path.dirname(HERE), "tabfm_baseline_optionA", "preds")

# ---------- fast DeLong ----------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_cov(y, scores):
    """scores: list of 1-D arrays (same y). Returns (aucs, covariance matrix)."""
    y = np.asarray(y); order = (-y).argsort()  # positives first
    label1 = int(y.sum()); n = len(y) - label1; m = label1
    preds = np.vstack([s[order] for s in scores]); k = preds.shape[0]
    tx = np.array([_midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([_midrank(preds[r, :]) for r in range(k)])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    if k == 1: s01 = s01.reshape(1, 1); s10 = s10.reshape(1, 1)
    cov = s01 / m + s10 / n
    return aucs, np.atleast_2d(cov)

def delong_diff(y, sa, sb):
    aucs, cov = delong_cov(y, [sa, sb])
    d = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(var); z = d / se if se > 0 else 0
    p = 2 * st.norm.sf(abs(z))
    return d, se, (d - 1.96 * se, d + 1.96 * se), p

def tost(d, se, delta):
    # H0: |true diff| >= delta ; reject (=> equivalence) if both one-sided p<0.05
    p_lo = st.norm.sf((d - (-delta)) / se)      # test diff > -delta
    p_hi = st.norm.cdf((d - delta) / se)        # test diff <  delta
    p = max(p_lo, p_hi)
    return p, (p < 0.05)

def load(f):
    d = pd.read_csv(f); d["nctid"] = d["nctid"].astype(str)
    d = d[d.split == "test"].drop_duplicates("nctid").set_index("nctid")
    return d["score"].astype(float), d["label"].astype(int)

def base_rate_lookup(ph):
    D = os.path.join(C.HINT_DIR, "data")
    import ast
    def code(v):
        try: c = ast.literal_eval(v)
        except: c = [v]
        return c[0] if c else "NA"
    tr = pd.read_csv(os.path.join(D, f"phase_{ph}_train.csv")); te = pd.read_csv(os.path.join(D, f"phase_{ph}_test.csv"))
    tr["k"] = tr.icdcodes.map(code) + "|" + tr.phase.astype(str)
    te["k"] = te.icdcodes.map(code) + "|" + te.phase.astype(str); te["nctid"] = te.nctid.astype(str)
    rate = tr.groupby("k").label.mean(); glob = tr.label.mean()
    s = pd.Series([rate.get(k, glob) for k in te.k], index=te.nctid.values)
    return s, pd.Series(te.label.values, index=te.nctid.values)

ph = "III"
H, y = load(f"{PR}/preds_hint_phase_{ph}.csv")
T, _ = load(f"{A}/preds_tabfm_phase_{ph}.csv")
BR, ybr = base_rate_lookup(ph)
common = H.index.intersection(T.index)
y = y.loc[common].to_numpy(); Hs = H.loc[common].to_numpy(); Ts = T.loc[common].to_numpy()
c2 = common.intersection(BR.index); BRs = BR.loc[c2].to_numpy()

print(f"=== phase {ph} DeLong / TOST / MDE (n={len(y)}, pos={y.mean():.3f}) ===")
for name, s in [("TabFM", Ts)]:
    d, se, ci, p = delong_diff(y, Hs, s)
    pt, eq = tost(d, se, 0.02)
    print(f"  HINT - {name}: ΔAUC={d:+.4f}  SE={se:.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  DeLong p={p:.3f}")
    print(f"      TOST @ δ=0.02: p={pt:.3f}  => {'EQUIVALENT' if eq else 'NOT shown equivalent'}")
# HINT vs base-rate lookup
yb = y[[list(common).index(x) for x in c2]]
d, se, ci, p = delong_diff(yb, H.loc[c2].to_numpy(), BRs)
pt, eq = tost(d, se, 0.02)
print(f"  HINT - base-rate: ΔAUC={d:+.4f} SE={se:.4f} 95%CI[{ci[0]:+.4f},{ci[1]:+.4f}] DeLong p={p:.3f} | TOST δ=0.02 p={pt:.3f} {'EQUIV' if eq else 'not-equiv'}")

# ---- MDE: smallest ΔAUC detectable at 80% power, α=0.05 two-sided ----
_, se_t, _, _ = delong_diff(y, Hs, Ts)
mde = (st.norm.ppf(0.975) + st.norm.ppf(0.80)) * se_t
print(f"\n  [MDE] SE(ΔAUC)={se_t:.4f} -> min detectable ΔAUC @80% power = {mde:.3f}")
print(f"        => on TOP phase III (n={len(y)}), ΔAUC below ~{mde:.2f} is unverifiable at this sample size.")

# ---- balanced accuracy + MCC + PR vs prevalence ----
def thr_val(name, sfile):
    v = pd.read_csv(sfile); v = v[v.split == "valid"]
    yv, sv = v.label.values, v.score.values
    ts = np.unique(sv); best, bt = -1, 0.5
    for t in (ts[:-1] + ts[1:]) / 2:
        f = f1_score(yv, (sv >= t).astype(int), zero_division=0)
        if f > best: best, bt = f, t
    return bt
print("\n  model   BalAcc    MCC    PR-AUC  (base rate PR = pos rate)")
for name, s, f in [("HINT", Hs, f"{PR}/preds_hint_phase_{ph}.csv"), ("TabFM", Ts, f"{A}/preds_tabfm_phase_{ph}.csv")]:
    t = thr_val(name, f); pred = (s >= t).astype(int)
    print(f"  {name:6} {balanced_accuracy_score(y,pred):.3f}   {matthews_corrcoef(y,pred):+.3f}   "
          f"{average_precision_score(y,s):.3f}  (base {y.mean():.3f}; lift +{average_precision_score(y,s)-y.mean():.3f})")
