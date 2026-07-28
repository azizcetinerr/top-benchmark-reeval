# -*- coding: utf-8 -*-
"""
run_optionA.py — RUN Option A (put the TabFM baseline through the harness)
=======================================================================
It DOES NOT DOWNLOAD anything new / DOES NOT WRITE to other folders. It uses the
assets under ctp/ as READ-ONLY INPUT:
  * Data   : ctp/hint/data/phase_{X}_{train,valid,test}.csv   (real TOP)
  * Model  : ctp/tabfm-main                                    (real TabFM)
  * Harness: ctp/paper/eval_harness.py + config.py             (fair evaluation)
All outputs are written ONLY to this folder.

Flow (same backbone as HINT/MEXA):
  1) extract a FEATURE table from train (context)
  2) fit TabFM on train, score valid+test   (GBM fallback if unavailable)
  3) dump the raw scores to preds/preds_tabfm_phase_{X}.csv
     {nctid, score, label, split}  — the SAME schema as HINT/MEXA
  4) evaluate with paper/eval_harness; HINT + MEXA + TRIVIAL side by side
  5) interpret the decision matrix (relative to the TRIVIAL baseline!)

Run:
  cd ctp/tabfm_baseline_optionA
  python run_optionA.py                 # with config.PHASE (= III)
  python run_optionA.py --phase I
  python run_optionA.py --full-rdkit    # ~200 RDKit descriptors
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True                # do NOT write .pyc to other folders

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # .../ctp
PAPER_DIR = os.path.join(ROOT, "paper")       # READ-ONLY
TABFM_DIR = os.path.join(ROOT, "tabfm-main")  # READ-ONLY

PREDS_DIR = os.path.join(HERE, "preds")
os.makedirs(PREDS_DIR, exist_ok=True)

sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)

import config as C                            # noqa: E402
import eval_harness as H                      # noqa: E402
import tabfm_baseline as TB                   # noqa: E402


def _hint_csv(phase, split):
    return os.path.join(C.HINT_DIR, "data", f"phase_{phase}_{split}.csv")


def dump_tabfm(phase, full_rdkit=False, n_estimators=32):
    """context=train -> valid+test skorla -> preds/preds_tabfm_phase_{X}.csv"""
    df_tr = pd.read_csv(_hint_csv(phase, "train"))
    df_va = pd.read_csv(_hint_csv(phase, "valid"))
    df_te = pd.read_csv(_hint_csv(phase, "test"))
    print(f"[data] train={len(df_tr)} valid={len(df_va)} test={len(df_te)} "
          f"| pos_rate(train)={df_tr.label.mean():.3f}  (small data = TabFM sweet spot)")

    Xtr, ytr, _ = TB.build_feature_table(df_tr, full_rdkit)
    Xva, yva, id_va = TB.build_feature_table(df_va, full_rdkit)
    Xte, yte, id_te = TB.build_feature_table(df_te, full_rdkit)

    cols = Xtr.columns.union(Xva.columns).union(Xte.columns)
    Xtr, Xva, Xte = (X.reindex(columns=cols) for X in (Xtr, Xva, Xte))
    print(f"[feature] {len(cols)} columns (metadata + RDKit + protocol)")

    s_va, s_te = TB.fit_predict_tabfm(Xtr, ytr, [Xva, Xte], tabfm_dir=TABFM_DIR,
                                      n_estimators=n_estimators)

    out_csv = os.path.join(PREDS_DIR, f"preds_tabfm_phase_{phase}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nctid", "score", "label", "split"])
        for nid, sc, lb in zip(id_va, s_va, yva):
            w.writerow([nid, float(sc), int(lb), "valid"])
        for nid, sc, lb in zip(id_te, s_te, yte):
            w.writerow([nid, float(sc), int(lb), "test"])
    print(f"[tabfm] written -> {out_csv} ({len(id_va)+len(id_te)} rows)")
    return out_csv


def _eval(path):
    return H.evaluate_model(path, select_metric=C.SELECT_METRIC,
                            n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)


def main():
    ap = argparse.ArgumentParser(description="Option A — TabFM baseline runner")
    ap.add_argument("--phase", default=C.PHASE, help="I | II | III (default: config.PHASE)")
    ap.add_argument("--full-rdkit", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=32,
                    help="number of TabFM ensemble members (default 32). "
                         "If slow on CPU, try 8: ~4x faster, context not truncated.")
    args = ap.parse_args()
    phase = args.phase
    print(f"[phase] {phase}  (config.PHASE = {C.PHASE})")

    tabfm_preds = dump_tabfm(phase, full_rdkit=args.full_rdkit,
                             n_estimators=args.n_estimators)
    results = {"TabFM(OptionA)": _eval(tabfm_preds)}


    rakip_var = False
    for name, key in [("HINT", "hint"), ("MEXA", "mexa")]:
        p = os.path.join(PAPER_DIR, "preds", f"preds_{key}_phase_{phase}.csv")
        if os.path.exists(p):
            results[name] = _eval(p)
            rakip_var = True
        else:
            print(f"[note] no {name} scores: {os.path.basename(p)}")
    if not rakip_var:
        mevcut = sorted(f for f in os.listdir(os.path.join(PAPER_DIR, "preds"))
                        if f.startswith("preds_"))
        print(f"[WARNING] no competitor scores in paper/ for phase {phase}. "
              f"Available: {mevcut}\n"
              f"        There may be a PHASE MISMATCH — config.PHASE={C.PHASE}. "
              f"Run with the same phase!")


    test_labels = pd.read_csv(tabfm_preds).query("split == 'test'")["label"].to_numpy()

    table = H.compare_table(results, test_labels=test_labels)
    print(f"\n=== Option A — fair comparison (phase {phase}) ===")
    print(table.to_string(index=False))
    out_tab = os.path.join(HERE, f"results_tabfm_phase_{phase}.csv")
    table.to_csv(out_tab, index=False)
    print(f"[written] {out_tab}")


    base = H.baseline_metrics(test_labels)
    t_pr = results["TabFM(OptionA)"]["point"]["pr_auc"]
    print(f"\n--- Reading ---")
    print(f"TRIVIAL baseline PR-AUC = {base['pr_auc']:.3f}. "
          f"TabFM PR-AUC = {t_pr:.3f} -> "
          + ("BELOW/EQUAL to baseline: no signal, the features do not help."
             if t_pr <= base['pr_auc'] + 0.005 else "above baseline: there is real signal."))
    if "HINT" in results:
        h_pr = results["HINT"]["point"]["pr_auc"]
        d = t_pr - h_pr
        print(f"PR-AUC difference (TabFM - HINT) = {d:+.4f}")
        if abs(d) < 0.02:
            print("TabFM ≈ HINT: the molecule+graph+BioBERT machinery adds little value "
                  "(a strong ablation finding).")
        elif d < 0:
            print("TabFM << HINT: the structural signal really matters — HINT's complexity is justified.")
        else:
            print("TabFM > HINT: CAUTION — either HINT was not trained enough on this data "
                  "or a feature label is leaking. Do not accept it as is.")


if __name__ == "__main__":
    main()
