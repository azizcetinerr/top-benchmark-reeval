# -*- coding: utf-8 -*-
"""
reviewer_batch5.py — reproduces the numbers added for the 5th reviewer batch.

  Q3a  single-SMILES degeneracy across phases and splits (train & test)
  Q3b  drop the degenerate trials -> HINT's Only_Molecule branch on the clean 54%
  Q5   ICD-code(+phase) signature is coarse (base rate, not per-trial memorisation)
  Q6   survivorship check: phase-III positive rate across NCT-number (time) quintiles
  Q9   stratified vs unstratified bootstrap CI at phase I

Run:  python reviewer_batch5.py
"""
import os, ast, glob, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "hint", "data"); PR = os.path.join(HERE, "preds")
TOP = 'CN1C(=O)C=C(N2CCC[C@@H](N)C2)N(CC2=C(C=CC=C2)C#N)C1=O'

def parse(v):
    try: return [str(x).strip() for x in ast.literal_eval(v)]
    except Exception: return [t for t in str(v).replace(';', ' ').replace("'", "").replace('[', '').replace(']', '').split() if t]

def main():
    print("=== Q3a: single-SMILES degeneracy across phases (train & test) ===")
    for ph in ["I", "II", "III"]:
        for sp in ["train", "test"]:
            df = pd.read_csv(f"{D}/phase_{ph}_{sp}.csv")
            cnt = Counter(s for v in df.smiless for s in parse(v)); top, _ = cnt.most_common(1)[0]
            print(f"  phase {ph:3} {sp:5}: most-common SMILES covers {100*df.smiless.map(lambda v: top in parse(v)).mean():4.0f}% of {len(df)} trials")

    print("=== Q3b: drop 528 degenerate -> HINT Only_Molecule on clean 54% (phase III test) ===")
    te = pd.read_csv(f"{D}/phase_III_test.csv"); te['nctid'] = te.nctid.astype(str)
    deg = set(te[te.smiless.map(lambda v: TOP in parse(v))].nctid)
    fom = glob.glob(f"{PR}/*Only_Molecule*phase_III*.csv")
    if fom:
        p = pd.read_csv(fom[0]); p = p[p.split == 'test']; p['nctid'] = p.nctid.astype(str)
        nd = p[~p.nctid.isin(deg)]
        print(f"  HINT Only_Molecule ROC: full={roc_auc_score(p.label,p.score):.3f} | non-degenerate({len(nd)})={roc_auc_score(nd.label,nd.score):.3f}")

    print("=== Q5: ICD-code(+phase) signature coarseness (base rate, not memorisation) ===")
    tr = pd.read_csv(f"{D}/phase_III_train.csv")
    sig = lambda df: [(df.phase.iloc[i],) + tuple(sorted(set(parse(df.icdcodes.iloc[i])))) for i in range(len(df))]
    strn = Counter(sig(tr)); ste = sig(te); m = [strn[s] for s in ste if s in strn]
    print(f"  test {len(ste)} trials | {len(set(ste))} distinct signatures ({100*len(set(ste))/len(ste):.0f}% unique); "
          f"{100*len(m)/len(ste):.0f}% match train; median {int(np.median(m))} train trials/signature")

    print("=== Q6: survivorship? phase-III positive rate across NCT (time) quintiles ===")
    num = lambda s: int(str(s)[3:]) if str(s).startswith("NCT") else 0
    te2 = te.assign(x=te.nctid.map(num)); te2['q'] = pd.qcut(te2.x, 5, labels=False, duplicates='drop')
    print("  early->late:", [f"{te2[te2.q==q].label.mean():.2f}" for q in sorted(te2.q.dropna().unique())])

    print("=== Q9: stratified vs unstratified bootstrap CI, HINT phase I ===")
    h = pd.read_csv(f"{PR}/preds_hint_phase_I.csv"); h = h[h.split == 'test']
    y = h.label.values.astype(int); s = h.score.values.astype(float); rng = np.random.default_rng(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    def boot(strat):
        o = []
        for _ in range(1000):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) if strat else rng.integers(0, len(y), len(y))
            o.append(roc_auc_score(y[idx], s[idx]))
        return np.round(np.percentile(o, [2.5, 97.5]), 3)
    print(f"  phase I n={len(y)}: unstratified={boot(False)} stratified={boot(True)}")

if __name__ == "__main__":
    main()
