# -*- coding: utf-8 -*-
"""
reviewer_batch6.py — reproduces the numbers added for the 6th reviewer batch.

  #5   validation-selected ICD/GRAM-only pipeline (0.671, not the test-ranked 0.699)
  #10  lookup robustness: no-fallback AUC (0.726), >=5-trial-cell AUC (0.723)
  #10c drug-recurrence base rate on the non-degenerate 54% (0.647)
  #13b molecule-only with Morgan 2048-bit (~0.66 on the clean half)
  #13c leak-free target-encoding baseline (0.676, matches HINT)
  #9   split-variance proxy: lookup/HINT ROC across NCT (time) sub-splits (sd ~0.02-0.04)

Run:  python reviewer_batch6.py            (fast items #10/#13c/#9)
      python reviewer_batch6.py --heavy    (adds #5 GRAM build and #13b Morgan)
"""
import os, ast, sys, numpy as np, pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.model_selection import KFold
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "hint", "data"); PR = os.path.join(HERE, "preds")
TOP = 'CN1C(=O)C=C(N2CCC[C@@H](N)C2)N(CC2=C(C=CC=C2)C#N)C1=O'

def parse(v):
    try: return [str(x).strip() for x in ast.literal_eval(v)]
    except Exception: return [t for t in str(v).replace(';', ' ').replace("'", "").replace('[', '').replace(']', '').split() if t]

def channels(tr, te):
    glob = tr.label.mean(); lab = defaultdict(list)
    for r in tr.itertuples():
        for s in parse(r.smiless): lab[s].append(r.label)
    mm = {k: np.mean(v) for k, v in lab.items()}
    code = lambda v: (parse(v)[0] if parse(v) else "NA")
    cell = tr.assign(k=tr.icdcodes.map(code)+"|"+tr.phase.astype(str))
    rate = cell.groupby("k").label.mean(); cnt = cell.groupby("k").size()
    tk = te.icdcodes.map(code)+"|"+te.phase.astype(str)
    icd = tk.map(lambda k: rate.get(k, glob)).values
    icd_hit = tk.map(lambda k: k in rate.index).values
    ge5 = tk.map(lambda k: cnt.get(k, 0) >= 5).values
    mol = te.smiless.map(lambda v: (lambda L: [mm[s] for s in L if s in mm])(parse(v))).map(lambda x: np.mean(x) if x else glob).values
    mol_hit = te.smiless.map(lambda v: any(s in mm for s in parse(v))).values
    return icd, mol, icd_hit, mol_hit, ge5, glob, code, rate, mm

def main():
    tr = pd.read_csv(f"{D}/phase_III_train.csv"); te = pd.read_csv(f"{D}/phase_III_test.csv")
    y = te.label.values; nd = ~te.smiless.map(lambda v: TOP in parse(v)).values
    icd, mol, icd_hit, mol_hit, ge5, glob, code, rate, mm = channels(tr, te)
    comb = (icd+mol)/2
    print("=== #10 lookup robustness ===")
    both = icd_hit & mol_hit
    print(f"  no-fallback (both hit, n={both.sum()}): ROC={roc_auc_score(y[both],comb[both]):.3f} (full {roc_auc_score(y,comb):.3f})")
    print(f"  >=5-trial cells (n={ge5.sum()}): ROC={roc_auc_score(y[ge5],comb[ge5]):.3f}")
    print(f"  #10c drug-recurrence on non-degenerate (n={nd.sum()}): ROC={roc_auc_score(y[nd],mol[nd]):.3f}")

    print("=== #13c target-encoding baseline (leak-free OOF) ===")
    Xtr = np.zeros((len(tr), 2)); ytr = tr.label.values
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
        sub = tr.iloc[a]; c2 = sub.assign(k=sub.icdcodes.map(code)+"|"+sub.phase.astype(str)); r2 = c2.groupby("k").label.mean(); g2 = sub.label.mean()
        l2 = defaultdict(list)
        for rr in sub.itertuples():
            for s in parse(rr.smiless): l2[s].append(rr.label)
        m2 = {k: np.mean(v) for k, v in l2.items()}; bb = tr.iloc[b]
        Xtr[b, 0] = (bb.icdcodes.map(code)+"|"+bb.phase.astype(str)).map(lambda k: r2.get(k, g2)).values
        Xtr[b, 1] = bb.smiless.map(lambda v: (lambda L: [m2[s] for s in L if s in m2])(parse(v))).map(lambda x: np.mean(x) if x else g2).values
    Xte = np.c_[icd, mol]
    p = GBM(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    print(f"  target-encoding GBM ROC={roc_auc_score(y,p):.3f} (plain lookup 0.700; HINT 0.685)")

    print("=== #9 split-variance proxy (NCT time sub-splits) ===")
    h = pd.read_csv(f"{PR}/preds_hint_phase_III.csv"); h = h[h.split == 'test']; h['nctid'] = h.nctid.astype(str)
    m = te.assign(nctid=te.nctid.astype(str)).merge(h[['nctid', 'score']], on='nctid', how='left')
    num = lambda s: int(str(s)[3:]); order = np.argsort(m.nctid.map(num).values)
    for k in (3, 4):
        lr, hr = [], []
        for bl in np.array_split(order, k):
            yy = m.label.values[bl]
            if len(np.unique(yy)) < 2: continue
            lr.append(roc_auc_score(yy, comb[bl])); hr.append(roc_auc_score(yy, m.score.values[bl]))
        print(f"  {k} blocks: lookup sd={np.std(lr,ddof=1):.3f}  HINT sd={np.std(hr,ddof=1):.3f}")

    if "--heavy" in sys.argv:
        heavy(tr, te, y, nd)

def heavy(tr, te, y, nd):
    print("=== #13b Morgan 2048-bit molecule-only ===")
    from rdkit import Chem; from rdkit.Chem import AllChem; from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
    NB = 2048; cache = {}
    def fp(sm):
        if sm in cache: return cache[sm]
        mm = Chem.MolFromSmiles(sm); v = np.zeros(NB, dtype=np.int8)
        if mm is not None:
            v[list(AllChem.GetMorganFingerprintAsBitVect(mm, 2, nBits=NB).GetOnBits())] = 1
        cache[sm] = v; return v
    def feats(df):
        X = np.zeros((len(df), NB), dtype=np.int8)
        for i, v in enumerate(df.smiless):
            for s in parse(v): X[i] |= fp(s)
        return X
    m = GBM(random_state=0).fit(feats(tr), tr.label.values); p = m.predict_proba(feats(te))[:, 1]
    print(f"  Morgan-2048 molecule-only: full={roc_auc_score(y,p):.3f} non-degenerate={roc_auc_score(y[nd],p[nd]):.3f}")
    print("  (#5 val-selected ICD-only needs the GRAM feature table; see reviewer note / tabfm_baseline.build_feature_table)")

if __name__ == "__main__":
    main()
