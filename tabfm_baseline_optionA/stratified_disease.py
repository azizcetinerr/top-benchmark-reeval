# -*- coding: utf-8 -*-
"""
stratified_disease.py — E5: disease-area breakdown
===================================================
Finding: the signal comes from the disease area (ICD chapter). Does the TabFM ~ HINT
parity hold in EVERY disease area, or does it separate in some?

Maps the existing test preds (HINT, TabFM, + requested members) by nctid to the ICD
top-chapter and computes ROC/PR per chapter. No GPU/RDKit.

Run:  python stratified_disease.py --phase III
"""
import argparse
import os
import re
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER_DIR)
import config as C          # noqa: E402
import tabfm_baseline as TB  # noqa: E402

CHAPTER_NAME = {
    "C": "Neoplasm (cancer)", "D": "Blood/immune", "E": "Endocrine/metabolic",
    "F": "Mental", "G": "Nervous system", "I": "Circulatory", "J": "Respiratory",
    "K": "Digestive", "L": "Skin", "M": "Musculoskeletal", "N": "Urogenital",
    "O": "Pregnancy", "R": "Symptom", "Z": "Health status",
}


def nctid_to_chapter(phase):
    df = pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{phase}_test.csv"))
    out = {}
    for _, r in df.iterrows():
        icds = TB._to_list(r.get("icdcodes"))
        ch = TB._icd_chapter(icds[0]) if icds else "?"
        out[str(r["nctid"])] = ch
    return out


def load_test(name, phase):
    p = (os.path.join(HERE, "preds", f"preds_tabfm_phase_{phase}.csv")
         if name == "tabfm"
         else os.path.join(PAPER_DIR, "preds", f"preds_{name}_phase_{phase}.csv"))
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.split == "test"][["nctid", "score", "label"]].copy()
    d["nctid"] = d["nctid"].astype(str)
    return d.drop_duplicates("nctid").set_index("nctid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--models", default="hint,tabfm")
    ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args()
    phase = args.phase
    names = [m.strip() for m in args.models.split(",")]

    ch = nctid_to_chapter(phase)
    data = {m: load_test(m, phase) for m in names}
    data = {m: d for m, d in data.items() if d is not None}
    common = None
    for d in data.values():
        common = set(d.index) if common is None else (common & set(d.index))
    common = sorted(common)
    print(f"[common test] {len(common)} trial | models: {list(data)}")

    base = data[names[0]].loc[common]
    y = base["label"].to_numpy()
    chap = np.array([ch.get(n, "?") for n in common])

    rows = []
    for c in sorted(set(chap)):
        mask = chap == c
        yy = y[mask]
        if mask.sum() < args.min_n or yy.min() == yy.max():
            continue
        row = {"ICD": c, "alan": CHAPTER_NAME.get(c, c), "n": int(mask.sum()),
               "poz_oran": round(float(yy.mean()), 3)}
        for m in data:
            s = data[m].loc[common, "score"].to_numpy()[mask]
            row[f"{m}_ROC"] = round(roc_auc_score(yy, s), 3)
            row[f"{m}_PR"] = round(average_precision_score(yy, s), 3)
        rows.append(row)

    tab = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(f"\n=== Disease-area breakdown (phase {phase}, n>={args.min_n}) ===")
    print(tab.to_string(index=False))
    out = os.path.join(HERE, f"disease_breakdown_phase_{phase}.csv")
    tab.to_csv(out, index=False)
    print(f"[written] {out}")

    if "hint" in data and "tabfm" in data:
        d = tab["tabfm_ROC"] - tab["hint_ROC"]
        print(f"\nPer-area TabFM-HINT ROC difference: mean={d.mean():+.3f}, "
              f"range=[{d.min():+.3f}, {d.max():+.3f}]")
        print("Reading: if the difference is consistent and small across areas, "
              "the 'TabFM ~ HINT' parity holds independently of disease area.")


if __name__ == "__main__":
    main()
