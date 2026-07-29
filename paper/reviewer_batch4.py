# -*- coding: utf-8 -*-
"""
reviewer_batch4.py — reproduces the numbers added for the 4th reviewer batch.

  #3  validation-SELECTED combined-lookup weight (w on valid, read on test)
  #5a MEXA's reported HINT PR-AUC 0.603 is below the phase-III prevalence floor
  #7  per-phase MDE (bootstrap SE of dAUC HINT-TabFM, MDE = 2.8 * SE)

Run:  python reviewer_batch4.py
"""
import os, ast, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "hint", "data")

def parse(v):
    try: return [str(x).strip() for x in ast.literal_eval(v)]
    except Exception: return [t for t in str(v).replace(';', ' ').replace("'", "").replace('[', '').replace(']', '').split() if t]

def lookup_channels():
    tr = pd.read_csv(f"{D}/phase_III_train.csv"); va = pd.read_csv(f"{D}/phase_III_valid.csv"); te = pd.read_csv(f"{D}/phase_III_test.csv")
    glob = tr.label.mean(); lab = defaultdict(list)
    for r in tr.itertuples():
        for s in parse(r.smiless): lab[s].append(r.label)
    mm = {k: np.mean(v) for k, v in lab.items()}
    code = lambda v: (parse(v)[0] if parse(v) else "NA")
    rate = tr.assign(k=tr.icdcodes.map(code)+"|"+tr.phase.astype(str)).groupby("k").label.mean()
    def chan(df):
        mol = df.smiless.map(lambda v: (lambda L: [mm[s] for s in L if s in mm])(parse(v))).map(lambda x: np.mean(x) if x else glob).values
        icd = (df.icdcodes.map(code)+"|"+df.phase.astype(str)).map(lambda k: rate.get(k, glob)).values
        return icd.astype(float), mol.astype(float)
    return chan(va), va.label.values, chan(te), te.label.values, te.label.mean()

def main():
    (i_va, m_va), yva, (i_te, m_te), yte, prev = lookup_channels()
    print("=== #3: validation-selected combined-lookup weight (phase III) ===")
    ws = np.linspace(0, 1, 101)
    w = max(ws, key=lambda w: roc_auc_score(yva, w*i_va+(1-w)*m_va))
    print(f"  fixed 0.5 mean : test ROC={roc_auc_score(yte,0.5*i_te+0.5*m_te):.3f}")
    print(f"  valid-selected w={w:.2f} (disease): test ROC={roc_auc_score(yte,w*i_te+(1-w)*m_te):.3f}")

    print("=== #5a: MEXA HINT PR-AUC 0.603 vs prevalence floor ===")
    print(f"  phase III test prevalence = {prev:.4f} -> PR-AUC floor = {prev:.3f}; MEXA reports 0.603 < floor (impossible for ROC 0.685)")

    print("=== #7: per-phase MDE (bootstrap SE of dAUC HINT-TabFM; MDE=2.8*SE) ===")
    def load(f):
        x = pd.read_csv(f); x = x[x.split == 'test']; x['nctid'] = x.nctid.astype(str); return x.set_index('nctid')
    A = os.path.join(ROOT, "tabfm_baseline_optionA", "preds")
    for ph in ["I", "II", "III"]:
        h = load(f"{HERE}/preds/preds_hint_phase_{ph}.csv"); t = load(f"{A}/preds_tabfm_phase_{ph}.csv")
        c = h.index.intersection(t.index); y = h.loc[c, 'label'].astype(int).values
        sh = h.loc[c, 'score'].values.astype(float); st = t.loc[c, 'score'].values.astype(float)
        rng = np.random.default_rng(0); d = np.empty(1000)
        for i in range(1000):
            idx = rng.integers(0, len(c), len(c))
            try: d[i] = roc_auc_score(y[idx], sh[idx])-roc_auc_score(y[idx], st[idx])
            except Exception: d[i] = np.nan
        se = np.nanstd(d)
        print(f"  phase {ph}: n={len(c)} SE={se:.3f} MDE={2.802*se:.3f}")

if __name__ == "__main__":
    main()
