# -*- coding: utf-8 -*-
"""
option_B_stress_phaseIII.py — robustness check for the phase-III marginal edge
===============================================================================
Two questions about the small phase-III fusion edges:
  (A) Are the low p-values real, or Monte-Carlo noise from a single bootstrap?
      -> recompute at high n_boot across several bootstrap seeds; p should be stable.
  (B) We scanned many contrasts (4 methods x 3 combos x 2 metrics = 24). Under
      multiple testing, do any edges survive correction?
      -> Benjamini-Hochberg FDR and Bonferroni over the whole phase-III family.

Reads the combined-score CSVs in preds/ (produced by option_B_blend.py).
Writes: optionB_phaseIII_multiplecomparison.csv

Result (this repo): p-values are stable (A), but ZERO contrasts survive BH-FDR or
Bonferroni (B). The phase-III edge is consistent with chance given multiplicity;
the near-null conclusion holds. NNLS is never significant even uncorrected.
"""
import os, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "preds")
PH = "III"
METHODS = ["stack", "zavg", "rank", "nnls"]
COMBOS = ["HINT_TabFM", "MEXA_TabFM", "HINT_MEXA_TabFM"]

def load_test(method, combo):
    d = pd.read_csv(os.path.join(P, f"preds_ensB_{method}_{combo}_phase_{PH}.csv"))
    d = d[d.split == "test"]
    return d["score"].to_numpy(), d["label"].to_numpy()

def fast_auc(y, s):
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def fast_ap(y, s):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

FN = {"pr_auc": fast_ap, "roc_auc": fast_auc}

singles = {nm: load_test("stack", nm) for nm in ("HINT", "MEXA", "TabFM")}
y = singles["HINT"][1]
assert abs(fast_auc(y, singles["HINT"][0]) - roc_auc_score(y, singles["HINT"][0])) < 1e-6
best = {"pr_auc": max(singles, key=lambda n: fast_ap(y, singles[n][0])),
        "roc_auc": max(singles, key=lambda n: fast_auc(y, singles[n][0]))}
print(f"best single: ROC={best['roc_auc']}, PR={best['pr_auc']}", flush=True)

def pboot(sa, sb, metric, n, seed):
    fn = FN[metric]; rng = np.random.default_rng(seed); m = len(y); d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m); yy = y[idx]
        if yy.min() == yy.max(): d[i] = 0.0; continue
        d[i] = fn(yy, sa[idx]) - fn(yy, sb[idx])
    return float(fn(y, sa) - fn(y, sb)), min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)

# (A) p stability for the two raw-significant HINT+TabFM contrasts
print("\n=== (A) p stability (n_boot=4000, seeds 0..2) ===", flush=True)
for method, combo, metric in [("stack", "HINT_TabFM", "roc_auc"),
                              ("zavg", "HINT_TabFM", "pr_auc")]:
    sa, _ = load_test(method, combo); sb, _ = singles[best[metric]]
    ps = [pboot(sa, sb, metric, 4000, s)[1] for s in range(3)]
    print(f"{method:5} {combo} {metric:7} p/seed={[round(x,4) for x in ps]} "
          f"mean={np.mean(ps):.4f}", flush=True)

# (B) full family + correction
print("\n=== (B) family + BH-FDR + Bonferroni (n_boot=3000) ===", flush=True)
rows = []
for method, combo, metric in itertools.product(METHODS, COMBOS, ["pr_auc", "roc_auc"]):
    sa, _ = load_test(method, combo); sb, _ = singles[best[metric]]
    diff, p = pboot(sa, sb, metric, 3000, 2023)
    rows.append([f"{method}:{combo}", metric, best[metric], round(diff, 4), round(p, 4)])
df = pd.DataFrame(rows, columns=["contrast", "metric", "vs_best_single", "diff", "p_raw"])
df = df.sort_values("p_raw").reset_index(drop=True)
m = len(df); r = np.arange(1, m + 1); bh = df["p_raw"].values * m / r
for i in range(m - 2, -1, -1): bh[i] = min(bh[i], bh[i + 1])
df["p_BH"] = np.round(np.clip(bh, 0, 1), 4)
df["p_Bonf"] = np.round(np.clip(df["p_raw"] * m, 0, 1), 4)
df["win_BH"] = (df.p_BH < 0.05) & (df["diff"] > 0)
df["win_Bonf"] = (df.p_Bonf < 0.05) & (df["diff"] > 0)

out = os.path.join(HERE, "optionB_phaseIII_multiplecomparison.csv")
df.to_csv(out, index=False)
pd.set_option("display.width", 170)
print(df.to_string(index=False), flush=True)
print(f"\nfamily m={m} | raw wins={int(((df.p_raw<0.05)&(df['diff']>0)).sum())} | "
      f"survive BH-FDR@0.05={int(df.win_BH.sum())} | survive Bonferroni={int(df.win_Bonf.sum())}")
print(f"[written] {out}")
