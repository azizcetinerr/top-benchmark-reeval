# -*- coding: utf-8 -*-
"""calibration.py — Brier score and ECE for HINT/MEXA/TabFM (Table tab:calib)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PR = os.path.join(HERE, "preds")
A = os.path.join(os.path.dirname(HERE), "tabfm_baseline_optionA", "preds")

def load(f):
    d = pd.read_csv(f); d = d[d.split == "test"]
    return d.label.values.astype(float), d.score.values.astype(float)

def ece(y, p, bins=10):
    """equal-MASS (quantile) bins."""
    idx = np.argsort(p); y, p = y[idx], p[idx]; e = 0.0; n = len(y)
    for b in np.array_split(np.arange(n), bins):
        if len(b): e += abs(p[b].mean() - y[b].mean()) * len(b)
    return e / n

brier = lambda y, p: float(np.mean((p - y) ** 2))

def boot(y, p, fn, n=1000, seed=2023):
    r = np.random.default_rng(seed); m = len(y)
    return np.round(np.percentile([fn(y[i], p[i]) for i in (r.integers(0, m, m) for _ in range(n))], [2.5, 97.5]), 3)

print(f"{'phase':5} {'model':6} {'Brier[95% CI]':>22} {'ECE(eq-mass)[95% CI]':>24}")
for ph in ["I", "II", "III"]:
    fs = {"HINT": f"{PR}/preds_hint_phase_{ph}.csv", "TabFM": f"{A}/preds_tabfm_phase_{ph}.csv"}
    if ph == "III": fs["MEXA"] = f"{PR}/preds_mexa_phase_III.csv"
    for n, f in fs.items():
        y, p = load(f)
        if n == "MEXA": p = 1 / (1 + np.exp(-p))          # logits -> prob
        print(f"{ph:5} {n:6} {brier(y,p):.3f} {list(boot(y,p,brier))}   {ece(y,p):.3f} {list(boot(y,p,ece))}")
