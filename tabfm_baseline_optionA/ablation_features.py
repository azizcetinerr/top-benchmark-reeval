# -*- coding: utf-8 -*-
"""
ablation_features.py — Sinyal hangi katmandan geliyor?
======================================================
Option A's main finding is "the tabular baseline matches HINT". The natural next
question: what does that parity rest on?

We turn the three layers on and off separately:
  meta          : HINT metadata only (phase, drug/disease/ICD counts)
  meta+prot     : + protocol scalars (criterion counts, text length)
  meta+rdkit    : + RDKit physicochemical descriptors (molecule)
  full          : all

Why it matters:
  * If the RDKit layer adds nothing, the finding hardens: molecule information
    (neither graph nor physicochemistry) is useless on this task — a much
    stronger claim about HINT's D-MPNN.
  * If all signal comes from 'meta', the real predictor is non-structural
    variables like phase/scale; this should then be examined for leakage.

NOTE: each subset is a separate TabFM run (n_estimators=32 fixed — TabFM is
not weakened). Should be run on GPU.

Run:
  python ablation_features.py --phase III
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
TABFM_DIR = os.path.join(ROOT, "tabfm-main")
PREDS_DIR = os.path.join(HERE, "preds")
os.makedirs(PREDS_DIR, exist_ok=True)

sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import eval_harness as H    # noqa: E402
import tabfm_baseline as TB  # noqa: E402

META_COLS = ["phase", "n_drugs", "n_smiles", "n_diseases", "n_icdcodes",
             "icd_chapter", "n_icd_chapters"]


def _subset(X, kume):
    rd = [c for c in X.columns if c.startswith("rdkit_")]
    pr = [c for c in X.columns if c.startswith("crit_")]
    me = [c for c in META_COLS if c in X.columns]
    secim = {"meta": me,
             "meta+prot": me + pr,
             "meta+rdkit": me + rd,
             "full": me + pr + rd}[kume]
    return X[secim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--sets", default="meta,meta+prot,meta+rdkit,full")
    args = ap.parse_args()
    phase = args.phase

    d = lambda s: os.path.join(C.HINT_DIR, "data", f"phase_{phase}_{s}.csv")
    df_tr, df_va, df_te = (pd.read_csv(d(s)) for s in ("train", "valid", "test"))
    print(f"[data] train={len(df_tr)} valid={len(df_va)} test={len(df_te)}")

    Xtr, ytr, _ = TB.build_feature_table(df_tr)
    Xva, yva, id_va = TB.build_feature_table(df_va)
    Xte, yte, id_te = TB.build_feature_table(df_te)
    cols = Xtr.columns.union(Xva.columns).union(Xte.columns)
    Xtr, Xva, Xte = (X.reindex(columns=cols) for X in (Xtr, Xva, Xte))

    satirlar = []
    for kume in [s.strip() for s in args.sets.split(",")]:
        A, B, Cc = (_subset(X, kume) for X in (Xtr, Xva, Xte))
        print(f"\n>>> [{kume}] running TabFM with {A.shape[1]} columns...")
        s_va, s_te = TB.fit_predict_tabfm(A, ytr, [B, Cc], tabfm_dir=TABFM_DIR)

        out = os.path.join(PREDS_DIR, f"preds_tabfm_{kume.replace('+','_')}"
                                      f"_phase_{phase}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["nctid", "score", "label", "split"])
            for nid, sc, lb in zip(id_va, s_va, yva):
                w.writerow([nid, float(sc), int(lb), "valid"])
            for nid, sc, lb in zip(id_te, s_te, yte):
                w.writerow([nid, float(sc), int(lb), "test"])

        r = H.evaluate_model(out, select_metric=C.SELECT_METRIC,
                             n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA,
                             seed=C.SEED)
        satirlar.append({
            "kume": kume, "n_sutun": A.shape[1],
            "ROC-AUC": round(r["point"]["roc_auc"], 4),
            "PR-AUC": round(r["point"]["pr_auc"], 4),
            "PR-AUC CI": f"[{r['ci']['pr_auc']['lo']:.3f}, {r['ci']['pr_auc']['hi']:.3f}]",
        })
        print(f"    ROC-AUC={satirlar[-1]['ROC-AUC']}  PR-AUC={satirlar[-1]['PR-AUC']}")

    tab = pd.DataFrame(satirlar)
    print(f"\n=== Layer ablation (phase {phase}) ===")
    print(tab.to_string(index=False))
    out = os.path.join(HERE, f"ablation_phase_{phase}.csv")
    tab.to_csv(out, index=False)
    print(f"[written] {out}")
    print("\nReading: the gap between 'meta+rdkit' and 'meta' is the net contribution "
          "of MOLECULE information. If it is ~0, it is strong negative evidence "
          "for HINT's D-MPNN.")


if __name__ == "__main__":
    main()
