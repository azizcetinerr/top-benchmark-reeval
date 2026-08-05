# -*- coding: utf-8 -*-
"""
target_encoding_phases.py -- the leak-free target-encoding baseline of
reviewer_batch6.py (#13c), run on all three phases instead of phase III only.

Two features per trial, both computed out-of-fold on train so that no label
leaks into its own row:
  1. the historical success rate of the trial's (ICD code, phase) cell
  2. the historical success rate of the trial's molecules
A gradient-boosting classifier is fitted on those two columns.

Run:  python3 target_encoding_phases.py
Writes: target_encoding_phases.csv
"""
import os, ast, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "hint", "data")


def parse(v):
    try:
        return [str(x).strip() for x in ast.literal_eval(v)]
    except Exception:
        return [t for t in str(v).replace(';', ' ').replace("'", "")
                .replace('[', '').replace(']', '').split() if t]


code = lambda v: (parse(v)[0] if parse(v) else "NA")


def channels(tr, te):
    """Test-side target-encoded columns, fitted on the whole training split."""
    glob = tr.label.mean()
    lab = defaultdict(list)
    for r in tr.itertuples():
        for s in parse(r.smiless):
            lab[s].append(r.label)
    mm = {k: np.mean(v) for k, v in lab.items()}
    cell = tr.assign(k=tr.icdcodes.map(code) + "|" + tr.phase.astype(str))
    rate = cell.groupby("k").label.mean()
    tk = te.icdcodes.map(code) + "|" + te.phase.astype(str)
    icd = tk.map(lambda k: rate.get(k, glob)).values
    mol = (te.smiless
           .map(lambda v: (lambda L: [mm[s] for s in L if s in mm])(parse(v)))
           .map(lambda x: np.mean(x) if x else glob).values)
    return icd, mol


def oof_train_features(tr):
    """Out-of-fold target encoding for the training rows (5-fold, seed 0)."""
    X = np.zeros((len(tr), 2))
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
        sub = tr.iloc[a]
        c2 = sub.assign(k=sub.icdcodes.map(code) + "|" + sub.phase.astype(str))
        r2 = c2.groupby("k").label.mean()
        g2 = sub.label.mean()
        l2 = defaultdict(list)
        for rr in sub.itertuples():
            for s in parse(rr.smiless):
                l2[s].append(rr.label)
        m2 = {k: np.mean(v) for k, v in l2.items()}
        bb = tr.iloc[b]
        X[b, 0] = ((bb.icdcodes.map(code) + "|" + bb.phase.astype(str))
                   .map(lambda k: r2.get(k, g2)).values)
        X[b, 1] = (bb.smiless
                   .map(lambda v: (lambda L: [m2[s] for s in L if s in m2])(parse(v)))
                   .map(lambda x: np.mean(x) if x else g2).values)
    return X


def main():
    rows = []
    for ph in ("I", "II", "III"):
        tr = pd.read_csv(f"{D}/phase_{ph}_train.csv")
        te = pd.read_csv(f"{D}/phase_{ph}_test.csv")
        y = te.label.values
        icd, mol = channels(tr, te)
        Xtr, ytr = oof_train_features(tr), tr.label.values
        Xte = np.c_[icd, mol]
        p = GBM(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        roc, pr = roc_auc_score(y, p), average_precision_score(y, p)
        lookup = (icd + mol) / 2
        rows.append(dict(phase=ph, n_test=len(te), pos_rate=round(y.mean(), 3),
                         target_encoding_roc=round(roc, 4),
                         target_encoding_pr=round(pr, 4),
                         plain_lookup_roc=round(roc_auc_score(y, lookup), 4)))
        print(f"phase {ph}: target-encoding GBM ROC={roc:.4f} PR={pr:.4f} "
              f"| plain two-column lookup ROC={rows[-1]['plain_lookup_roc']:.4f}")
    out = os.path.join(HERE, "target_encoding_phases.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
