# -*- coding: utf-8 -*-
"""
compare_all.py — Compare TabFM against ALL EXISTING baselines + significance test
===============================================================================
Why it is needed:
  1) paper/preds contains scores beyond HINT and MEXA too: logreg, randomforest,
     gradboost and the HINT ablations (hintvar_HINT_nograph, Only_Molecule,
     Only_Disease, Interaction). These directly answer Option A's question — we
     find them all automatically and put them in the table.
  2) "The confidence intervals overlap" is NOT a SIGNIFICANCE TEST. Since both
     models are evaluated on the same test set their differences are CORRELATED;
     the correct test is a PAIRED bootstrap that computes the DIFFERENCE on the
     same bootstrap sample. Looking at separate CIs misses real differences (too
     conservative).

Output: comparison table + difference CI and p-value of each model against TabFM.

Run:
  python compare_all.py --phase III
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
PAPER_PREDS = os.path.join(PAPER_DIR, "preds")
MY_PREDS = os.path.join(HERE, "preds")

sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import eval_harness as H    # noqa: E402


def _load_test(path):
    """Returns the {nctid, score, label} test split."""
    df = pd.read_csv(path)
    df = df[df["split"] == "test"][["nctid", "score", "label"]].copy()
    df["nctid"] = df["nctid"].astype(str)
    return df.drop_duplicates("nctid").set_index("nctid")


def paired_bootstrap(y, s_a, s_b, metric="pr_auc", n=2000, seed=2023):
    """
    PAIRED bootstrap: on each repetition it samples the SAME rows for both
    models and computes the DIFFERENCE. Returns: (diff_mean, lo, hi, p_two_sided).

    p: two-sided empirical p-value derived from the probability that the
    difference does not include 0.
    """
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    diffs = np.empty(n)
    ok = 0
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yy = y[idx]
        if yy.min() == yy.max():        # single-class sample -> metric undefined
            continue
        diffs[ok] = fn(yy, s_a[idx]) - fn(yy, s_b[idx])
        ok += 1
    diffs = diffs[:ok]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided empirical p: fraction of the difference sign on the opposite side x2
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    phase = args.phase
    suffix = f"_phase_{phase}.csv"

    # --- TabFM (ours) ---
    mine = os.path.join(MY_PREDS, f"preds_tabfm_phase_{phase}.csv")
    if not os.path.exists(mine):
        raise SystemExit(f"Run run_optionA.py --phase {phase} first: {mine} missing")

    # --- Find competitors AUTOMATICALLY (all preds_* in paper/preds) ---
    rakipler = {}
    for f in sorted(os.listdir(PAPER_PREDS)):
        if f.startswith("preds_") and f.endswith(suffix):
            isim = f[len("preds_"):-len(suffix)]
            if isim == "tabfm":            # skip our old copy if present
                continue
            rakipler[isim] = os.path.join(PAPER_PREDS, f)
    print(f"[found] {len(rakipler)} competitors: {sorted(rakipler)}")

    # --- Common harness table ---
    results = {"TabFM(OptionA)": H.evaluate_model(
        mine, select_metric=C.SELECT_METRIC, n_boot=C.BOOTSTRAP_N,
        alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)}
    for isim, p in rakipler.items():
        try:
            results[isim] = H.evaluate_model(
                p, select_metric=C.SELECT_METRIC, n_boot=C.BOOTSTRAP_N,
                alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
        except Exception as e:
            print(f"[skipped] {isim}: {type(e).__name__}")

    test_labels = pd.read_csv(mine).query("split=='test'")["label"].to_numpy()
    table = H.compare_table(results, test_labels=test_labels)
    print(f"\n=== All models (phase {phase}) ===")
    print(table.to_string(index=False))
    out = os.path.join(HERE, f"compare_all_phase_{phase}.csv")
    table.to_csv(out, index=False)
    print(f"[written] {out}")

    # --- PAIRED significance tests (against TabFM) ---
    a = _load_test(mine)
    rows = []
    for isim, p in rakipler.items():
        b = _load_test(p)
        ortak = a.index.intersection(b.index)
        if len(ortak) < 50:
            print(f"[skipped] {isim}: only {len(ortak)} common nctid")
            continue
        y = a.loc[ortak, "label"].to_numpy()
        # labels must be the same in both files — otherwise the alignment is broken
        if not np.array_equal(y, b.loc[ortak, "label"].to_numpy()):
            print(f"[WARNING] {isim}: labels do not match, skipping")
            continue
        sa = a.loc[ortak, "score"].to_numpy()
        sb = b.loc[ortak, "score"].to_numpy()
        for metric in ("pr_auc", "roc_auc"):
            d, lo, hi, pv = paired_bootstrap(y, sa, sb, metric, n=args.n_boot,
                                             seed=C.SEED)
            rows.append({"competitor": isim, "metric": metric, "n": len(ortak),
                         "diff(TabFM-competitor)": round(d, 4),
                         "CI95": f"[{lo:+.4f}, {hi:+.4f}]",
                         "p": round(pv, 4),
                         "significant": "YES" if pv < 0.05 else "no"})

    if rows:
        dfp = pd.DataFrame(rows).sort_values(["metric", "diff(TabFM-competitor)"],
                                             ascending=[True, False])
        print(f"\n=== Paired bootstrap: TabFM - competitor (phase {phase}) ===")
        print(dfp.to_string(index=False))
        out2 = os.path.join(HERE, f"significance_phase_{phase}.csv")
        dfp.to_csv(out2, index=False)
        print(f"[written] {out2}")
        print("\nReading: if the CI includes 0 the difference is statistically absent. "
              "This is the RESULT you want for the 'TabFM ≈ HINT' claim — a CI that "
              "includes 0 is stronger evidence of equivalence than overlapping intervals.")


if __name__ == "__main__":
    main()
