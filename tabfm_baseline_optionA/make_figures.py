# -*- coding: utf-8 -*-
"""
make_figures.py — the paper's two figures (reviewer item 14).
  fig_forest.pdf     : phase-III ROC-AUC with 95% CIs for every model, trivial baseline marked.
  fig_noisefloor.pdf : observed / claimed effect sizes vs the minimum detectable effect (0.05).
Numbers are the paper's (Tables tab:fair/tab:gbdt/tab:tabfm + stats_rigor.py).
Run: python make_figures.py   ->  writes both PDFs next to the .tex.
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "svg.fonttype": "none"})
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- Figure 1: phase-III forest plot ----------
# (label, ROC, lo, hi)  -- lo/hi = bootstrap/DeLong 95% CI; None -> point
rows = [
    ("Trivial (all-positive)", 0.500, None, None),
    ("MEXA-CTP (released ckpt)", 0.518, 0.484, 0.553),
    ("LightGBM (tuned)",        0.647, 0.610, 0.680),
    ("TabPFN v2 (0-train)",     0.647, 0.609, 0.683),
    ("TabFM (0-train)",         0.667, 0.631, 0.702),
    ("CatBoost (tuned)",        0.680, 0.642, 0.713),
    ("HINT",                    0.685, 0.647, 0.720),
    ("XGBoost (tuned)",         0.691, 0.653, 0.724),
    ("ICD-code base-rate lookup", 0.694, 0.660, 0.724),
]
rows = sorted(rows, key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(5.4, 3.2))
y = np.arange(len(rows))
for i, (lab, m, lo, hi) in enumerate(rows):
    is_hint = lab == "HINT"
    c = "#c0392b" if is_hint else ("#7f8c8d" if "Trivial" in lab or "released" in lab else "#2c3e50")
    if lo is not None:
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(m, i, "o", color=c, ms=7 if is_hint else 5, zorder=3)
ax.axvline(0.5, ls="--", color="#95a5a6", lw=1)
ax.text(0.5, len(rows)-0.3, " chance", color="#95a5a6", va="top", fontsize=8)
# shade HINT's CI band across the plot to show overlap
hint = [r for r in rows if r[0] == "HINT"][0]
ax.axvspan(hint[2], hint[3], color="#c0392b", alpha=0.07, zorder=0)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Test ROC-AUC (phase III), 95\\% CI"); ax.set_xlim(0.46, 0.76)
ax.set_title("Every model overlaps HINT (shaded band)", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_forest.pdf")); plt.close(fig)

# ---------- Figure 2: effect size vs noise floor ----------
# (label, effect_lo, effect_hi)  -- |ROC-AUC| effect magnitude
eff = [
    ("A/B: training cross-attention (III)", 0.0004, 0.0004),
    ("Fusion HINT+TabFM vs best single",    0.016, 0.016),
    ("HINT $-$ TabFM (observed)",           0.018, 0.018),
    ("HINT $-$ base-rate lookup",           0.009, 0.009),
    ("HINT run-to-run seed noise",          0.023, 0.023),
    ("HINT $-$ GradBoost (III)",            0.034, 0.034),
    ("MEXA-CTP \\emph{claimed} ROC gain",   0.011, 0.035),
]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
MDE = 0.05
ax.axvspan(0, MDE, color="#e74c3c", alpha=0.10, zorder=0)
ax.axvline(MDE, color="#e74c3c", lw=1.5)
ax.text(MDE+0.001, len(eff)-0.4, " MDE = 0.05\n (80\\% power, n=1146)", color="#c0392b", fontsize=8, va="top")
for i, (lab, lo, hi) in enumerate(eff):
    if lo == hi:
        ax.plot(lo, i, "o", color="#2c3e50", ms=6, zorder=3)
    else:
        ax.plot([lo, hi], [i, i], color="#2c3e50", lw=4, solid_capstyle="round", zorder=3)
        ax.plot([lo, hi], [i, i], "|", color="#2c3e50", ms=9, zorder=3)
ax.set_yticks(range(len(eff))); ax.set_yticklabels([e[0] for e in eff])
ax.set_xlabel("Effect size (|$\\Delta$ ROC-AUC|)"); ax.set_xlim(-0.002, 0.07)
ax.set_title("Every observed and claimed effect sits below the noise floor", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_noisefloor.pdf")); plt.close(fig)
print("wrote fig_forest.pdf and fig_noisefloor.pdf")
