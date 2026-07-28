# -*- coding: utf-8 -*-
"""
reviewer_batch3.py — reproduces the numbers added for the 3rd reviewer batch.

  ITEM 1  molecule-only ROC on the non-degenerate 54% of phase-III test trials
  ITEM 3  base-rate lookup coverage / fallback / (no) smoothing
  ITEM 5  drug recurrence = same-programme continuation?
  ITEM 2  forward-chaining (temporal) nested CV  [run with --fc]
  ITEM 9  Wilcoxon (A/B) + paired permutation (HINT-TabFM) + threshold bootstrap

Items 4 and 6 are PDF-sourced (HINT PR-AUC across papers; verbatim label quote) and are
not computed here. Run:  python reviewer_batch3.py        (fast: items 1,3,5,9)
                         python reviewer_batch3.py --fc   (adds item 2, slower)
"""
import os, ast, sys, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.metrics import roc_auc_score, f1_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tabfm_baseline_optionA"))
sys.path.insert(0, HERE)
import config as C, tabfm_baseline as TB
D = os.path.join(C.HINT_DIR, "data")
TOP = 'CN1C(=O)C=C(N2CCC[C@@H](N)C2)N(CC2=C(C=CC=C2)C#N)C1=O'

def parse(v):
    try: return [str(x).strip() for x in ast.literal_eval(v)]
    except Exception: return [t for t in str(v).replace(';', ' ').replace("'", "").replace('[', '').replace(']', '').split() if t]

def feats(df):
    X, y, _ = TB.build_feature_table(df, include_gram=True); X.columns = X.columns.map(str)
    return X, np.asarray(y)

def enc(X, cc): return X[cc].apply(pd.to_numeric, errors='coerce').fillna(0).values

def main():
    tr = pd.read_csv(f"{D}/phase_III_train.csv"); te = pd.read_csv(f"{D}/phase_III_test.csv")
    Xtr, ytr = feats(tr); Xte, yte = feats(te)
    cols = sorted(set(Xtr.columns) & set(Xte.columns))
    Xtr = Xtr.reindex(columns=cols); Xte = Xte.reindex(columns=cols)
    rd = [c for c in cols if c.startswith('rdkit_')]; gram = [c for c in cols if c.startswith('gram_')]
    def roc(cc, mask):
        m = GBM(random_state=0).fit(enc(Xtr, cc), ytr); p = m.predict_proba(enc(Xte, cc))[:, 1]
        return roc_auc_score(yte[mask], p[mask])

    print("=== ITEM 1: molecule-only on non-degenerate 54% ===")
    deg = te.smiless.map(lambda v: TOP in parse(v)).values; nd = ~deg
    print(f"  non-degenerate: {nd.sum()}/{len(te)} ({100*nd.mean():.0f}%)")
    print(f"  molecule-only ROC non-degenerate={roc(rd, nd):.3f} (full={roc(rd, np.ones(len(te), bool)):.3f}); disease-only={roc(gram, nd):.3f}")
    best = max((max(a, 1-a) for c in rd for a in [roc_auc_score(yte[nd], pd.to_numeric(Xte[c], errors='coerce').fillna(0).values[nd])] if pd.to_numeric(Xte[c], errors='coerce').fillna(0).values[nd].std() > 0), default=0)
    print(f"  best single molecule feature (non-degenerate): {best:.3f}")

    print("=== ITEM 3: base-rate lookup coverage/fallback ===")
    lab = defaultdict(list)
    for r in tr.itertuples():
        for s in parse(r.smiless): lab[s].append(r.label)
    molmean = {k: np.mean(v) for k, v in lab.items()}; glob = tr.label.mean()
    code = lambda v: (parse(v)[0] if parse(v) else "NA")
    icdrate = tr.assign(k=tr.icdcodes.map(code)+"|"+tr.phase.astype(str)).groupby("k").label.mean()
    molcov = te.smiless.map(lambda v: any(s in molmean for s in parse(v))).mean()
    icdcov = (te.icdcodes.map(code)+"|"+te.phase.astype(str)).map(lambda k: k in icdrate.index).mean()
    celln = tr.assign(k=tr.icdcodes.map(code)+"|"+tr.phase.astype(str)).groupby("k").size()
    print(f"  fallback=global base rate {glob:.3f}; mol coverage {100*molcov:.0f}%; ICD-cell coverage {100*icdcov:.0f}%; single-trial cells {100*(celln==1).mean():.0f}% (no smoothing)")

    print("=== ITEM 5: drug recurrence = same-programme? ===")
    trpairs = set()
    for r in tr.itertuples():
        for s in parse(r.smiless):
            for ic in parse(r.icdcodes): trpairs.add((s, ic))
    rec = te.smiless.map(lambda v: any(s in molmean for s in parse(v))).values
    same = te[rec].apply(lambda r: any((s, i) in trpairs for s in parse(r.smiless) for i in parse(r.icdcodes)), axis=1)
    print(f"  recur(>=1) {rec.sum()}/{len(te)} ({100*rec.mean():.0f}%); of those same (drug,ICD)={100*same.mean():.0f}% (programme continuation)")

    item9(tr, te)
    if "--fc" in sys.argv: forward_chaining()

def item9(tr, te):
    from scipy.stats import wilcoxon
    print("=== ITEM 9: Wilcoxon / permutation / threshold bootstrap ===")
    ab = pd.read_csv(f"{HERE}/ab_test_results_phase_III.csv").drop_duplicates(['arm', 'seed', 'rho1', 'rho2', 'thr'])
    pv = ab.pivot_table(index=['seed', 'rho1', 'rho2', 'thr'], columns='arm', values='test_roc').dropna()
    d = (pv['BUGGY']-pv['FIXED']).values; w, p = wilcoxon(d)
    print(f"  9a Wilcoxon A/B: {len(d)} pairs, mean {d.mean():+.4f}, p={p:.3f}")
    load = lambda f: (lambda x: x[x.split == 'test'].assign(nctid=lambda z: z.nctid.astype(str)).set_index('nctid'))(pd.read_csv(f))
    h = load(f"{HERE}/preds/preds_hint_phase_III.csv"); t = load(f"{os.path.dirname(HERE)}/tabfm_baseline_optionA/preds/preds_tabfm_phase_III.csv")
    c = h.index.intersection(t.index); y = h.loc[c, 'label'].astype(int).values
    sh = h.loc[c, 'score'].values.astype(float); st = t.loc[c, 'score'].values.astype(float)
    obs = roc_auc_score(y, sh)-roc_auc_score(y, st); rng = np.random.default_rng(0); null = np.empty(2000)
    for i in range(2000):
        sw = rng.random(len(c)) < 0.5
        null[i] = roc_auc_score(y, np.where(sw, st, sh))-roc_auc_score(y, np.where(sw, sh, st))
    print(f"  9b permutation HINT-TabFM: obs {obs:+.4f}, p={np.mean(np.abs(null) >= abs(obs)):.3f}")
    hv = pd.read_csv(f"{HERE}/preds/preds_hint_phase_III.csv"); val = hv[hv.split == 'valid']; tst = hv[hv.split == 'test']
    yv = val.label.values.astype(int); sv = val.score.values.astype(float); yt = tst.label.values.astype(int); ss = tst.score.values.astype(float)
    grid = np.quantile(sv, np.linspace(.02, .98, 80))
    def f1g(yy, s, T):
        P = s[:, None] >= T[None, :]; tp = (P & (yy[:, None] == 1)).sum(0); fp = (P & (yy[:, None] == 0)).sum(0); fn = ((~P) & (yy[:, None] == 1)).sum(0)
        return np.where(2*tp+fp+fn > 0, 2*tp/(2*tp+fp+fn), 0.0)
    rng = np.random.default_rng(1); ths = np.array([grid[np.argmax(f1g(yv[i], sv[i], grid))] for i in (rng.integers(0, len(yv), len(yv)) for _ in range(500))])
    fs = f1g(yt, ss, ths)
    print(f"  9c threshold bootstrap: thr {ths.mean():.3f}±{ths.std():.3f}; test F1 {fs.mean():.3f}±{fs.std():.3f} (trivial 0.857)")

def forward_chaining():
    print("=== ITEM 2: forward-chaining (temporal) CV ===")
    tr = pd.read_csv(f"{D}/phase_III_train.csv"); va = pd.read_csv(f"{D}/phase_III_valid.csv"); te = pd.read_csv(f"{D}/phase_III_test.csv")
    tv = pd.concat([tr, va], ignore_index=True); num = lambda s: int(str(s)[3:]) if str(s).startswith("NCT") else 0
    tv = tv.assign(_t=tv.nctid.map(num)).sort_values("_t").reset_index(drop=True)
    Xtv, ytv = feats(tv); Xte, yte = feats(te)
    gram = sorted(c for c in (set(Xtv.columns) & set(Xte.columns)) if c.startswith('gram_'))
    G = enc(Xtv, gram); bl = np.array_split(np.arange(len(G)), 5)
    b0 = bl[0]; corr = np.abs([np.corrcoef(G[b0, j], ytv[b0])[0, 1] if G[b0, j].std() > 0 else 0 for j in range(G.shape[1])])
    keep = [gram[j] for j in np.argsort(-corr)[:500]]
    Xa = enc(Xtv, keep); Xt = enc(Xte, keep); sc = []
    for i in range(1, 5):
        trI = np.concatenate(bl[:i]); m = GBM(random_state=0).fit(Xa[trI], ytv[trI]); sc.append(roc_auc_score(ytv[bl[i]], m.predict_proba(Xa[bl[i]])[:, 1]))
    m = GBM(random_state=0).fit(Xa, ytv)
    print(f"  FC-val {np.mean(sc):.3f}  TEST {roc_auc_score(yte, m.predict_proba(Xt)[:, 1]):.3f}  (random nested-CV 0.671; HINT 0.685)")

if __name__ == "__main__":
    main()
