# -*- coding: utf-8 -*-
"""
ensemble_compare.py — Single + fusion comparison
====================================================
The 6 models compared:
    HINT, MEXA, TabFM,
    HINT+TabFM, MEXA+TabFM, HINT+MEXA+TabFM

Question: is COMBINING two (or three) models better than the best single model?
If the paper's "TabFM ≈ HINT, signal is in the disease" finding is correct, the
fusion is expected to bring no notable gain (they share the same signal).

Method (faithful to the paper's discipline):
  * Fusion = a LOGISTIC meta-learner over the members' scores (stacking).
    Weights are learned ONLY on validation; test is never looked at.
  * The meta-learner is applied to test. The validation combined score is
    produced with OOF (cross_val_predict) for threshold selection — no in-sample
    optimism.
  * Scores are standardized per member (HINT sigmoid, MEXA logit, TabFM
    probability — so the scale difference does not break stacking).
  * ALL models are evaluated on the SAME common nctid set (the intersection of
    all available members) so the paired bootstrap is valid.
  * Evaluation is again the common harness (threshold@val + PR/ROC on score +
    bootstrap CI).

MEXA is only available in phase III; in I/II combinations with MEXA are skipped
automatically.

Run:  python ensemble_compare.py --phase III
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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
MY_PREDS = os.path.join(HERE, "preds")
ENS_DIR = os.path.join(HERE, "preds")
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import eval_harness as H    # noqa: E402

# Member -> preds file locator (hint/mexa in paper, tabfm is ours)
def member_path(member, phase):
    if member == "tabfm":
        return os.path.join(MY_PREDS, f"preds_tabfm_phase_{phase}.csv")
    return os.path.join(PAPER_DIR, "preds", f"preds_{member}_phase_{phase}.csv")


COMBOS = [
    ("HINT",            ["hint"]),
    ("MEXA",            ["mexa"]),
    ("TabFM",           ["tabfm"]),
    ("HINT+TabFM",      ["hint", "tabfm"]),
    ("MEXA+TabFM",      ["mexa", "tabfm"]),
    ("HINT+MEXA+TabFM", ["hint", "mexa", "tabfm"]),
]


def load_member(member, phase):
    """Returns {split: Series(score indexed by nctid)}, {split: Series(label)}."""
    df = pd.read_csv(member_path(member, phase))
    df["nctid"] = df["nctid"].astype(str)
    df = df.drop_duplicates(["nctid", "split"])
    sc, lb = {}, {}
    for sp in ("valid", "test"):
        s = df[df["split"] == sp]
        sc[sp] = s.set_index("nctid")["score"].astype(float)
        lb[sp] = s.set_index("nctid")["label"].astype(int)
    return sc, lb


def paired_bootstrap(y, sa, sb, metric="pr_auc", n=2000, seed=2023):
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    rng = np.random.default_rng(seed)
    m = len(y); diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(fn(yy, sa[idx]) - fn(yy, sb[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


def _rankpct(a):
    """Converts scores to a [0,1] rank-percentile (scale-independent)."""
    order = a.argsort().argsort()
    return order / max(len(a) - 1, 1)


def build_combined(members, data, common, phase, name, method="stack"):
    """Produces a combined score CSV for the given members (on the common nctid set).
    method: stack (logistic meta-learner) | zavg (z-average) | rank (rank-average)"""
    yv = data[members[0]][1]["valid"].loc[common["valid"]].to_numpy()
    yt = data[members[0]][1]["test"].loc[common["test"]].to_numpy()

    # member score matrices (in the common index order)
    Xv = np.column_stack([data[m][0]["valid"].loc[common["valid"]].to_numpy()
                          for m in members])
    Xt = np.column_stack([data[m][0]["test"].loc[common["test"]].to_numpy()
                          for m in members])

    if len(members) == 1:
        sv, st = Xv[:, 0], Xt[:, 0]                 # single: raw score
    elif method == "zavg":
        scaler = StandardScaler().fit(Xv)
        sv = scaler.transform(Xv).mean(axis=1)
        st = scaler.transform(Xt).mean(axis=1)
    elif method == "rank":
        sv = np.column_stack([_rankpct(Xv[:, i]) for i in range(Xv.shape[1])]).mean(1)
        st = np.column_stack([_rankpct(Xt[:, i]) for i in range(Xt.shape[1])]).mean(1)
    else:  # stack
        scaler = StandardScaler().fit(Xv)
        Xv_s, Xt_s = scaler.transform(Xv), scaler.transform(Xt)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xv_s, yv)                            # test: fit on all of valid
        st = lr.predict_proba(Xt_s)[:, 1]
        sv = cross_val_predict(LogisticRegression(max_iter=1000), Xv_s, yv,
                               cv=min(5, np.bincount(yv).min()),
                               method="predict_proba")[:, 1]  # valid OOF

    out = os.path.join(ENS_DIR, f"preds_ens_{name}_phase_{phase}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["nctid", "score", "label", "split"])
        for nid, s, l in zip(common["valid"], sv, yv):
            w.writerow([nid, float(s), int(l), "valid"])
        for nid, s, l in zip(common["test"], st, yt):
            w.writerow([nid, float(s), int(l), "test"])
    return out, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--method", default="stack", choices=["stack", "zavg", "rank"],
                    help="fusion method: stack (default) | zavg | rank")
    args = ap.parse_args()
    phase = args.phase

    # available members
    have = {}
    for m in ["hint", "mexa", "tabfm"]:
        if os.path.exists(member_path(m, phase)):
            have[m] = load_member(m, phase)
        else:
            print(f"[note] {m} missing (phase {phase}) -> combinations containing it are skipped")

    # common nctid set of ALL available members (fair, pairable test)
    common = {}
    for sp in ("valid", "test"):
        idx = None
        for m in have:
            s = set(have[m][0][sp].index)
            idx = s if idx is None else (idx & s)
        common[sp] = sorted(idx)
    print(f"[common set] valid={len(common['valid'])} test={len(common['test'])} "
          f"(members: {sorted(have)})")

    # label consistency check
    if len(have) > 1:
        base = have[list(have)[0]][1]["test"].loc[common["test"]].to_numpy()
        for m in have:
            if not np.array_equal(base, have[m][1]["test"].loc[common["test"]].to_numpy()):
                print(f"[WARNING] {m} test labels do not match!")

    results, yt_ref = {}, None
    for name, members in COMBOS:
        if not all(m in have for m in members):
            continue
        safe = name.replace("+", "_")
        path, yt = build_combined(members, have, common, phase, safe,
                                  method=args.method)
        results[name] = H.evaluate_model(path, select_metric=C.SELECT_METRIC,
                                         n_boot=C.BOOTSTRAP_N,
                                         alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
        yt_ref = yt

    table = H.compare_table(results, test_labels=yt_ref)
    print(f"\n=== Single + fusion (phase {phase}, common set) ===")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(HERE, f"ensemble_phase_{phase}.csv"), index=False)

    # --- Does fusion provide a gain? Paired bootstrap ---
    def test_scores(name):
        d = pd.read_csv(os.path.join(ENS_DIR,
            f"preds_ens_{name.replace('+','_')}_phase_{phase}.csv"))
        d = d[d.split == "test"]
        return d["score"].to_numpy(), d["label"].to_numpy()

    kontrastlar = [
        ("HINT+TabFM", "HINT"), ("HINT+TabFM", "TabFM"),
        ("MEXA+TabFM", "TabFM"),
        ("HINT+MEXA+TabFM", "TabFM"), ("HINT+MEXA+TabFM", "HINT+TabFM"),
    ]
    rows = []
    for a, b in kontrastlar:
        if a not in results or b not in results:
            continue
        sa, y = test_scores(a); sb, _ = test_scores(b)
        for metric in ("pr_auc", "roc_auc"):
            d, lo, hi, p = paired_bootstrap(y, sa, sb, metric, n=args.n_boot,
                                            seed=C.SEED)
            rows.append({"contrast": f"{a} − {b}", "metric": metric,
                         "diff": round(d, 4), "CI95": f"[{lo:+.4f}, {hi:+.4f}]",
                         "p": round(p, 4),
                         "gain": "YES" if (p < 0.05 and d > 0) else "no"})
    if rows:
        dfp = pd.DataFrame(rows)
        print(f"\n=== Fusion gain (paired bootstrap, phase {phase}) ===")
        print(dfp.to_string(index=False))
        dfp.to_csv(os.path.join(HERE, f"ensemble_significance_phase_{phase}.csv"),
                   index=False)
        print("\nReading: if 'diff' > 0 and p<0.05, the fusion significantly beats that "
              "single model. If the CI includes 0 the fusion brings NO gain — this is "
              "evidence that the models share the same (disease) signal.")


if __name__ == "__main__":
    main()
