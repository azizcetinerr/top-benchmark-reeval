# -*- coding: utf-8 -*-
"""
option_B_blend.py — Option B: Prediction-level blending (including NNLS)
========================================================================
Question: does blending the scores of HINT + MEXA + TabFM (weighted average /
stacking) significantly beat the BEST SINGLE model?

This script keeps exactly the discipline of ensemble_compare.py in Option A and
adds on top the **NNLS weighted average** we discussed:

    methods: stack (logistic meta) | zavg | rank | nnls

Common protocol (faithful to the paper's discipline):
  * Weights/meta-model are learned ONLY on validation; test is never looked at.
  * The validation combined score is produced with OOF (K-fold cross-fit) — no
    in-sample optimism. This is the "NNLS + OOF machine inside TabFM" generalized
    to three models: the same cross-fitting pattern, now for the blend weights.
  * Member scores are converted to rank-percentile to be scale-independent
    (HINT sigmoid, MEXA logit, TabFM probability — so the scale difference does
    not break the blend).
  * ALL models are evaluated on the SAME common nctid set (intersection of members).
  * Evaluation is done with the common harness (threshold@val + PR/ROC + bootstrap CI).

MEXA exists only in phase III; in I/II it automatically falls back to HINT+TabFM.

Run:
    python option_B_blend.py --phase III
    python option_B_blend.py --phase all         # I, II, III in order

Empirical finding (in this repo): no fusion, including NNLS, significantly beats
the best single model. Details: option_B_blend.md.
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

# --- Get the common harness from paper/ (repos don't import each other; we use the harness) ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
SECENEK_A = os.path.join(ROOT, "tabfm_baseline_optionA")
MY_PREDS = os.path.join(HERE, "preds")
os.makedirs(MY_PREDS, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import eval_harness as H    # noqa: E402

METHODS = ["stack", "zavg", "rank", "nnls"]

COMBOS = [
    ("HINT",            ["hint"]),
    ("MEXA",            ["mexa"]),
    ("TabFM",           ["tabfm"]),
    ("HINT+TabFM",      ["hint", "tabfm"]),
    ("MEXA+TabFM",      ["mexa", "tabfm"]),
    ("HINT+MEXA+TabFM", ["hint", "mexa", "tabfm"]),
]


def member_path(member, phase):
    """TabFM scores are in Option A, HINT/MEXA in paper/preds."""
    if member == "tabfm":
        return os.path.join(SECENEK_A, "preds", f"preds_tabfm_phase_{phase}.csv")
    return os.path.join(PAPER_DIR, "preds", f"preds_{member}_phase_{phase}.csv")


def load_member(member, phase):
    df = pd.read_csv(member_path(member, phase))
    df["nctid"] = df["nctid"].astype(str)
    df = df.drop_duplicates(["nctid", "split"])
    sc, lb = {}, {}
    for sp in ("valid", "test"):
        s = df[df["split"] == sp]
        sc[sp] = s.set_index("nctid")["score"].astype(float)
        lb[sp] = s.set_index("nctid")["label"].astype(int)
    return sc, lb


def _rankpct(a):
    """Converts scores to a [0,1] rank-percentile (scale-independent)."""
    order = a.argsort().argsort()
    return order / max(len(a) - 1, 1)


# =============================================================================
# NNLS + OOF: learn the blend weights without leakage
# =============================================================================
def _fit_nnls(R, y):
    """min_w ||y - R w||^2, w>=0, normalized to sum 1. R: (n, M) rank-pct."""
    w, _ = nnls(R, y.astype(float))
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _oof_nnls(Rv, yv, n_splits=5, seed=2023):
    """
    OOF (out-of-fold) NNLS score for validation. On each fold learn the weights
    on the OTHER folds and apply them to the held-out fold. This way the valid
    combined score has not seen its own weights -> threshold selection carries no
    in-sample optimism.
    """
    n = len(yv)
    oof = np.zeros(n)
    k = min(n_splits, np.bincount(yv).min())
    if k < 2:                       # too few positives/negatives -> OOF not possible
        w = _fit_nnls(Rv, yv)
        return Rv @ w
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, va in skf.split(Rv, yv):
        w = _fit_nnls(Rv[tr], yv[tr])
        oof[va] = Rv[va] @ w
    return oof


def build_combined(members, data, common, phase, name, method):
    """Produces a combined valid+test score CSV for the given members."""
    yv = data[members[0]][1]["valid"].loc[common["valid"]].to_numpy()
    yt = data[members[0]][1]["test"].loc[common["test"]].to_numpy()
    Xv = np.column_stack([data[m][0]["valid"].loc[common["valid"]].to_numpy()
                          for m in members])
    Xt = np.column_stack([data[m][0]["test"].loc[common["test"]].to_numpy()
                          for m in members])

    if len(members) == 1:
        sv, st = Xv[:, 0], Xt[:, 0]                     # single: raw score
    elif method == "zavg":
        scaler = StandardScaler().fit(Xv)
        sv = scaler.transform(Xv).mean(axis=1)
        st = scaler.transform(Xt).mean(axis=1)
    elif method == "rank":
        sv = np.column_stack([_rankpct(Xv[:, i]) for i in range(Xv.shape[1])]).mean(1)
        st = np.column_stack([_rankpct(Xt[:, i]) for i in range(Xt.shape[1])]).mean(1)
    elif method == "nnls":
        Rv = np.column_stack([_rankpct(Xv[:, i]) for i in range(Xv.shape[1])])
        Rt = np.column_stack([_rankpct(Xt[:, i]) for i in range(Xt.shape[1])])
        w = _fit_nnls(Rv, yv)                           # test: on all of valid
        st = Rt @ w
        sv = _oof_nnls(Rv, yv, seed=C.SEED)             # valid: OOF (leakage-free)
        build_combined.last_weights = dict(zip(members, np.round(w, 3)))
    else:  # stack
        scaler = StandardScaler().fit(Xv)
        Xv_s, Xt_s = scaler.transform(Xv), scaler.transform(Xt)
        lr = LogisticRegression(max_iter=1000).fit(Xv_s, yv)
        st = lr.predict_proba(Xt_s)[:, 1]
        sv = cross_val_predict(LogisticRegression(max_iter=1000), Xv_s, yv,
                               cv=min(5, np.bincount(yv).min()),
                               method="predict_proba")[:, 1]

    out = os.path.join(MY_PREDS, f"preds_ensB_{method}_{name}_phase_{phase}.csv")
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["nctid", "score", "label", "split"])
        for nid, s, l in zip(common["valid"], sv, yv):
            w_.writerow([nid, float(s), int(l), "valid"])
        for nid, s, l in zip(common["test"], st, yt):
            w_.writerow([nid, float(s), int(l), "test"])
    return out, yt


build_combined.last_weights = None


def paired_bootstrap(y, sa, sb, metric="pr_auc", n=1000, seed=2023):
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    rng = np.random.default_rng(seed)
    m = len(y); diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m); yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(fn(yy, sa[idx]) - fn(yy, sb[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


def run_phase(phase, method, n_boot):
    have = {}
    for m in ["hint", "mexa", "tabfm"]:
        if os.path.exists(member_path(m, phase)):
            have[m] = load_member(m, phase)
        else:
            print(f"[note] {m} missing (phase {phase}) -> combinations containing it are skipped")

    common = {}
    for sp in ("valid", "test"):
        idx = None
        for m in have:
            s = set(have[m][0][sp].index)
            idx = s if idx is None else (idx & s)
        common[sp] = sorted(idx)
    print(f"\n########## PHASE {phase} | method={method} ##########")
    print(f"[common set] valid={len(common['valid'])} test={len(common['test'])} "
          f"(members: {sorted(have)})")

    results, yt_ref, nnls_w = {}, None, {}
    for name, members in COMBOS:
        if not all(m in have for m in members):
            continue
        safe = name.replace("+", "_")
        # single models are method-independent; fusions use the selected method
        use = "stack" if len(members) == 1 else method
        path, yt = build_combined(members, have, common, phase, safe, use)
        if use == "nnls" and build_combined.last_weights:
            nnls_w[name] = build_combined.last_weights
        results[name] = H.evaluate_model(path, select_metric=C.SELECT_METRIC,
                                         n_boot=C.BOOTSTRAP_N,
                                         alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
        yt_ref = yt

    table = H.compare_table(results, test_labels=yt_ref)
    print(f"\n=== Single + fusion (phase {phase}, method={method}) ===")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(HERE, f"optionB_{method}_phase_{phase}.csv"),
                 index=False)
    if nnls_w:
        print("NNLS weights:", nnls_w)

    # --- Determine the best single model (by test ROC) and compare with fusions ---
    def test_scores(nm):
        safe = nm.replace("+", "_")
        use = "stack" if nm in ("HINT", "MEXA", "TabFM") else method
        d = pd.read_csv(os.path.join(MY_PREDS,
            f"preds_ensB_{use}_{safe}_phase_{phase}.csv"))
        d = d[d.split == "test"]
        return d["score"].to_numpy(), d["label"].to_numpy()

    singles = [nm for nm in ("HINT", "MEXA", "TabFM") if nm in results]
    fusions = [nm for nm in results if nm not in ("HINT", "MEXA", "TabFM")]

    # FAIR comparison: for EACH METRIC compare the fusion with the BEST single for
    # that metric. (The ROC-best and PR-best may be different models; fixing a
    # single "best" produces a misleading 'gain'.)
    metric_fn = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}
    best_by_metric = {}
    for metric, fn in metric_fn.items():
        best_by_metric[metric] = max(
            singles, key=lambda nm: fn(test_scores(nm)[1], test_scores(nm)[0]))

    rows = []
    for a in fusions:
        for metric in ("pr_auc", "roc_auc"):
            ref = best_by_metric[metric]
            sa, y = test_scores(a); sb, _ = test_scores(ref)
            d, lo, hi, p = paired_bootstrap(y, sa, sb, metric, n=n_boot, seed=C.SEED)
            rows.append({"contrast": f"{a} − {ref}", "metric": metric,
                         "diff": round(d, 4), "CI95": f"[{lo:+.4f}, {hi:+.4f}]",
                         "p": round(p, 4),
                         "gain": "YES" if (p < 0.05 and d > 0) else "no"})
    dfp = pd.DataFrame(rows)
    print(f"\n=== Does the fusion beat each metric's best single? "
          f"(ROC-best={best_by_metric['roc_auc']}, "
          f"PR-best={best_by_metric['pr_auc']}; paired bootstrap) ===")
    print(dfp.to_string(index=False))
    dfp.to_csv(os.path.join(HERE, f"optionB_{method}_significance_phase_{phase}.csv"),
               index=False)
    print("\nReading: if no row has 'gain=YES', blending adds no significant value "
          "over the best single model (this supports the paper's 'shared signal / "
          "noise floor' thesis).")
    return dfp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE, help="I | II | III | all")
    ap.add_argument("--method", default="nnls", choices=METHODS + ["all"],
                    help="fusion method (default nnls)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    phases = ["I", "II", "III"] if args.phase == "all" else [args.phase]
    methods = METHODS if args.method == "all" else [args.method]
    for ph in phases:
        for mt in methods:
            run_phase(ph, mt, args.n_boot)


if __name__ == "__main__":
    main()
