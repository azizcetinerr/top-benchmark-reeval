# -*- coding: utf-8 -*-
"""
data_audit.py — benchmark data-quality audits requested by reviewers (Q2, Q3, Q4, R-a).

  Q2  split is temporal (benchmark/data_split.py: train=completion_year<2014,
      test=start_year>=2014); confirmed here by NCT-number distribution.
  Q3  group-level recurrence: fraction of test SMILES / trials already seen in train, and a
      molecule -> mean-train-label lookup (does drug recurrence leak the label?).
  Q4  degenerate molecules: how many distinct drug names collapse to the single most common
      SMILES. NB the shared SMILES is a real molecule (alogliptin), not a null placeholder;
      the collapse is a mapping artefact of drug2smiles (arbitrary list(set)[0] + >=7-char
      substring fallback), and no-SMILES trials are dropped, not defaulted.
  R-a therapeutic-area base-rate lookup: predict each test trial by the train success rate of
      its (ICD code, phase) cell -- and the coarser (ICD chapter, phase) cell.

Run:  python data_audit.py --phase III
"""
import argparse, os, ast, sys, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import config as C

def parse(v):
    try: return [str(x).strip() for x in ast.literal_eval(v)]
    except Exception: return [t for t in str(v).replace(";", " ").replace("'", "").replace("[", "").replace("]", "").split() if t]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE); ph = ap.parse_args().phase
    D = os.path.join(C.HINT_DIR, "data")
    tr = pd.read_csv(os.path.join(D, f"phase_{ph}_train.csv"))
    te = pd.read_csv(os.path.join(D, f"phase_{ph}_test.csv"))
    for df in (tr, te):
        df["sm"] = df.smiless.map(parse); df["dr"] = df.drugs.map(parse)
    glob = tr.label.mean()
    print(f"=== phase {ph}: train={len(tr)} test={len(te)} pos(test)={te.label.mean():.3f} ===")

    # Q2 -- temporal split evidence
    num = lambda s: int(str(s)[3:]) if str(s).startswith("NCT") else 0
    print(f"[Q2] NCT-number median  train={tr.nctid.map(num).median():.0f}  test={te.nctid.map(num).median():.0f} "
          f"(test later => temporal; split_year=2014 in data_split.py)")

    # Q3 -- recurrence + molecule-label leakage
    lab = defaultdict(list)
    for r in tr.itertuples():
        for s in r.sm: lab[s].append(r.label)
    molmean = {k: np.mean(v) for k, v in lab.items()}
    seen = te.sm.apply(lambda L: len(L) > 0 and all(s in molmean for s in L))
    te["mp"] = te.sm.apply(lambda L: np.mean([molmean[s] for s in L if s in molmean]) if any(s in molmean for s in L) else glob)
    print(f"[Q3] test trials whose every SMILES was in train: {seen.sum()}/{len(te)} ({100*seen.mean():.0f}%)")
    print(f"[Q3] molecule->mean-train-label lookup: ROC={roc_auc_score(te.label,te.mp):.3f} "
          f"PR={average_precision_score(te.label,te.mp):.3f}  (exploitable drug-recurrence signal)")

    # Q4 -- placeholder SMILES
    cnt = Counter(s for r in te.itertuples() for s in r.sm)
    top, ntop = cnt.most_common(1)[0]
    mask = te.sm.apply(lambda L: top in L)
    ndrugs = len(set(d for r in te[mask].itertuples() for d in r.dr))
    nplac = sum(1 for r in te[mask].itertuples() for d in r.dr if d.lower() == "placebo")
    print(f"[Q4] most common test SMILES covers {mask.sum()}/{len(te)} trials ({100*mask.mean():.0f}%), "
          f"shared by {ndrugs} distinct drug names (incl. {nplac} 'placebo') -> name->structure collision "
          f"(real molecule, not a null default)")

    # R-a -- base-rate lookups
    def cell(df, keyfn):
        df = df.copy(); df["k"] = df.icdcodes.map(keyfn); df["p"] = df.phase.astype(str); return df
    def lookup(keyfn, name):
        t = cell(tr, keyfn); e = cell(te, keyfn)
        rate = t.groupby(["k", "p"]).label.mean()
        pred = [rate.get((x.k, x.p), glob) for x in e.itertuples()]
        print(f"[R-a] {name} base-rate lookup: ROC={roc_auc_score(te.label,pred):.3f} "
              f"PR={average_precision_score(te.label,pred):.3f}")
    lookup(lambda v: (parse(v)[0][0] if parse(v) else "NA"), "ICD-chapter x phase")
    lookup(lambda v: (parse(v)[0] if parse(v) else "NA"), "ICD-code x phase")
    # combined disease-code + drug-recurrence lookup (the headline)
    icd_rate = cell(tr, lambda v: (parse(v)[0] if parse(v) else "NA")).groupby(["k", "p"]).label.mean()
    te_icd = cell(te, lambda v: (parse(v)[0] if parse(v) else "NA"))
    s_icd = np.array([icd_rate.get((x.k, x.p), glob) for x in te_icd.itertuples()])
    s_mol = te.sm.apply(lambda L: np.mean([molmean[s] for s in L if s in molmean]) if any(s in molmean for s in L) else glob).values
    s_comb = (s_icd + s_mol) / 2
    print(f"[R-a] COMBINED (ICD-code + drug-recurrence, mean): "
          f"ROC={roc_auc_score(te.label,s_comb):.3f} PR={average_precision_score(te.label,s_comb):.3f}")
    print("  (HINT phase III: ROC 0.685 / PR 0.852 -- combined lookup is above it)")

if __name__ == "__main__":
    main()
