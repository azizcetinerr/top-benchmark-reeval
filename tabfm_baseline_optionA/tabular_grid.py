# -*- coding: utf-8 -*-
"""
tabular_grid.py — E1 (model panel) × E2 (feature grid), CPU
================================================================
Answers two questions at once:
  E1: Is TabFM special, or does ANY tabular model match HINT?
  E2: Which feature set lifts the tabular model highest? Does encoding the
      molecule differently (Morgan fingerprint, full-RDKit, GRAM ancestors)
      make a difference?

Design:
  * The feature table is built ONCE per phase (metadata + full-RDKit(mean,max)
    + protocol + Morgan + GRAM), then variants are derived by column subset — to
    avoid recomputing RDKit over and over.
  * Each (phase × feature-set × model) cell is evaluated with the common harness
    rules: threshold@validation, ROC/PR on continuous score. (No bootstrap since
    the grid is large; point estimate. The headline comparisons in compare_all /
    ensemble scripts use bootstrap.)
  * TabFM is NOT here (GPU); its results come separately. This script places HINT
    and all classic models side by side with the same features.

Output: tabular_grid_phase_{X}.csv (long form) + on-screen summary.

Run:  python tabular_grid.py --phase III
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import eval_harness as H    # noqa: E402
import tabfm_baseline as TB  # noqa: E402

META = ["phase", "n_drugs", "n_smiles", "n_diseases", "n_icdcodes",
        "icd_chapter", "n_icd_chapters"]


def col_groups(X):
    rd = [c for c in X.columns if c.startswith("rdkit_")]
    rd_key = [c for c in rd if any(f"rdkit_{k}_" in c for k in TB._KEY_DESCRIPTORS)]
    pr = [c for c in X.columns if c.startswith("crit_")]
    mo = [c for c in X.columns if c.startswith("morgan_")]
    gr = [c for c in X.columns if c.startswith("gram_") or c == "n_gram_ancestors"]
    me = [c for c in META if c in X.columns]
    return me, pr, rd, rd_key, mo, gr


def feature_sets(X):
    me, pr, rd, rd_key, mo, gr = col_groups(X)
    return {
        "meta":              me,
        "meta+prot":         me + pr,
        "meta+rdkit_key":    me + rd_key,
        "meta+rdkit_full":   me + rd,
        "meta+morgan":       me + mo,
        "meta+gram":         me + gr,
        "meta+rdkit+prot (default)": me + rd_key + pr,
        "meta+rdkit+morgan+gram":    me + rd_key + mo + gr,
        "ALL":               me + pr + rd + mo + gr,
        "icd_chapter_only":  ["phase", "icd_chapter"],
        "icd_count_only":    ["phase", "n_icdcodes", "n_icd_chapters"],
    }


def models():
    return {
        "LogReg":       make_pipeline(SimpleImputer(), StandardScaler(),
                                      LogisticRegression(max_iter=2000)),
        "RandomForest": make_pipeline(SimpleImputer(),
                                      RandomForestClassifier(n_estimators=300,
                                                             n_jobs=-1,
                                                             random_state=0)),
        "ExtraTrees":   make_pipeline(SimpleImputer(),
                                      ExtraTreesClassifier(n_estimators=300,
                                                           n_jobs=-1,
                                                           random_state=0)),
        "HistGBM":      make_pipeline(HistGradientBoostingClassifier(random_state=0)),
        "kNN":          make_pipeline(SimpleImputer(), StandardScaler(),
                                      KNeighborsClassifier(n_neighbors=25)),
        "GaussianNB":   make_pipeline(SimpleImputer(), GaussianNB()),
        "MLP":          make_pipeline(SimpleImputer(), StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=(64, 32),
                                                    max_iter=500, random_state=0)),
    }


def onehot_align(Xtr, Xva, Xte, cats):
    cats = [c for c in cats if c in Xtr.columns]
    a = pd.get_dummies(Xtr, columns=cats, dummy_na=False)
    b = pd.get_dummies(Xva, columns=cats, dummy_na=False)
    c = pd.get_dummies(Xte, columns=cats, dummy_na=False)
    cols = a.columns.union(b.columns).union(c.columns)
    return (a.reindex(columns=cols, fill_value=0),
            b.reindex(columns=cols, fill_value=0),
            c.reindex(columns=cols, fill_value=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--full-rdkit", action="store_true",
                    help="build all ~200 RDKit descriptors (requires a stable environment)")
    ap.add_argument("--quick", action="store_true",
                    help="quick check: light feature sets + fast models")
    args = ap.parse_args()
    phase = args.phase

    d = lambda s: os.path.join(C.HINT_DIR, "data", f"phase_{phase}_{s}.csv")
    df_tr, df_va, df_te = (pd.read_csv(d(s)) for s in ("train", "valid", "test"))

    # Build the full table once (all layers)
    opts = dict(full_rdkit=args.full_rdkit, include_morgan=True, include_gram=True)
    print("[build] full feature table (metadata+full-RDKit+protocol+Morgan+GRAM)...")
    Xtr, ytr, _ = TB.build_feature_table(df_tr, **opts)
    Xva, yva, idv = TB.build_feature_table(df_va, **opts)
    Xte, yte, idt = TB.build_feature_table(df_te, **opts)
    cols = Xtr.columns.union(Xva.columns).union(Xte.columns)
    Xtr, Xva, Xte = (X.reindex(columns=cols) for X in (Xtr, Xva, Xte))
    for X in (Xtr, Xva, Xte):
        sp = [c for c in X.columns if c.startswith(("morgan_", "gram_"))]
        X[sp] = X[sp].fillna(0)
    print(f"[build] {len(cols)} total columns")

    fsets = feature_sets(Xtr)
    mdls = models()
    if args.quick:              # quick check: light sets + fast models
        fsets = {k: fsets[k] for k in
                 ["meta", "meta+prot", "meta+rdkit_key", "icd_chapter_only"]}
        mdls = {k: mdls[k] for k in ["LogReg", "HistGBM"]}
    rows = []
    scores = {}                 # (fset,model) -> (sv, st)  cache for the paired test
    for fname, fcols in fsets.items():
        fcols = [c for c in fcols if c in cols]
        Atr, Ava, Ate = onehot_align(Xtr[fcols], Xva[fcols], Xte[fcols],
                                     TB.CAT_COLS)
        for mname, mdl in mdls.items():
            try:
                mdl.fit(Atr.values, ytr)
                sv = mdl.predict_proba(Ava.values)[:, 1]
                st = mdl.predict_proba(Ate.values)[:, 1]
            except Exception as e:
                print(f"[skip] {fname}/{mname}: {type(e).__name__}")
                continue
            thr = H.select_threshold(sv, yva, "f1")
            pt = H.ranking_metrics(st, yte)
            tm = H.threshold_metrics(st, yte, thr)
            vr = H.ranking_metrics(sv, yva)          # VALIDATION metrics (for selection)
            scores[(fname, mname)] = (sv, st)
            rows.append({"feature_set": fname, "n_feat": Atr.shape[1],
                         "model": mname,
                         "val_ROC": round(vr["roc_auc"], 4),
                         "val_PR": round(vr["pr_auc"], 4),
                         "ROC": round(pt["roc_auc"], 4),
                         "PR": round(pt["pr_auc"], 4), "F1": round(tm["f1"], 4)})
        print(f"  [{fname}] {len(fcols)} columns done")

    res = pd.DataFrame(rows)
    out = os.path.join(HERE, f"tabular_grid_phase_{phase}.csv")
    res.to_csv(out, index=False)

    # HINT reference + test scores (for the paired bootstrap)
    hint_p = os.path.join(PAPER_DIR, "preds", f"preds_hint_phase_{phase}.csv")
    hint_roc = hint_pr = None
    hint_test = None
    if os.path.exists(hint_p):
        r = H.evaluate_model(hint_p, n_boot=200, seed=C.SEED)
        hint_roc, hint_pr = r["point"]["roc_auc"], r["point"]["pr_auc"]
        hd = pd.read_csv(hint_p); hd["nctid"] = hd["nctid"].astype(str)
        hint_test = hd[hd.split == "test"].set_index("nctid")["score"]

    print(f"\n=== Top 12 cells (by TEST ROC — optimistic!) phase {phase} ===")
    print(res.sort_values("ROC", ascending=False).head(12).to_string(index=False))

    # --- RIGOROUS: select on VALIDATION, report on TEST (paper discipline) ---
    def paired_vs_hint(fname, mname, metric):
        sv, st = scores[(fname, mname)]
        s = pd.Series(st, index=[str(x) for x in idt])
        common = s.index.intersection(hint_test.index)
        y = pd.Series(yte, index=[str(x) for x in idt]).loc[common].to_numpy()
        a = s.loc[common].to_numpy(); b = hint_test.loc[common].to_numpy()
        fn = (H.average_precision_score if metric == "pr" else H.roc_auc_score) \
            if hasattr(H, "average_precision_score") else None
        from sklearn.metrics import roc_auc_score, average_precision_score
        fn = average_precision_score if metric == "pr" else roc_auc_score
        rng = np.random.default_rng(C.SEED); diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            diffs.append(fn(yy, a[idx]) - fn(yy, b[idx]))
        diffs = np.array(diffs)
        p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), min(p, 1.0)

    print(f"\n=== RIGOROUS selection: best VALIDATION, then TEST (phase {phase}) ===")
    for sel_metric, val_col, test_col in [("ROC", "val_ROC", "ROC"),
                                          ("PR", "val_PR", "PR")]:
        best_val = res.loc[res[val_col].idxmax()]
        best_test = res.loc[res[test_col].idxmax()]
        print(f"\n[{sel_metric}] validation-selected: {best_val['feature_set']} / "
              f"{best_val['model']}  -> TEST {sel_metric}={best_val[test_col]}")
        print(f"      (compare: TEST-selected winner {best_test['feature_set']}/"
              f"{best_test['model']} TEST {sel_metric}={best_test[test_col]}"
              + ("  <-- SAME" if best_val['feature_set'] == best_test['feature_set']
                 and best_val['model'] == best_test['model'] else "  <-- DIFFERENT (test-peeking was misleading)")
              + ")")
        if hint_test is not None:
            met = "pr" if sel_metric == "PR" else "roc"
            d, lo, hi, p = paired_vs_hint(best_val["feature_set"],
                                          best_val["model"], met)
            hv = hint_pr if sel_metric == "PR" else hint_roc
            verdict = ("BEATS HINT" if p < 0.05 and d > 0 else
                       "MATCHES HINT" if p >= 0.05 else "BELOW HINT")
            print(f"      HINT {sel_metric}={hv:.4f} | diff(selected−HINT)={d:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] p={p:.3f} -> {verdict}")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
