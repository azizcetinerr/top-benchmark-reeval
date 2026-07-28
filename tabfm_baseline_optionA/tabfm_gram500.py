# -*- coding: utf-8 -*-
"""
tabfm_gram500.py — can TabFM use the winning GRAM feature set once it is squeezed to the
500-feature cap? (reviewer item 6)

TabFM subsamples to max_num_features=500 per ensemble member, so the full ~4500-column GRAM set
is never seen whole (Section 'Where the signal lives'). Here we pre-select the 500 most
informative disease-ontology features on the TRAIN split (by absolute point-biserial
correlation with the label) and hand TabFM meta + top-500-GRAM, so it sees the winning
representation within its cap. Compares to HINT and to TabFM on the 38-column table.

Needs the jax[cuda12] `tabfm` env + GPU (g124). Run:
    python tabfm_gram500.py --phase III
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C, tabfm_baseline as TB

HINT = {"I": (0.574, 0.638), "II": (0.621, 0.674), "III": (0.685, 0.852)}

def boot_ci(y, s, fn, n=1000, seed=2023):
    rng = np.random.default_rng(seed); m = len(y)
    return np.round(np.percentile([fn(y[i], s[i]) for i in (rng.integers(0, m, m) for _ in range(n))], [2.5, 97.5]), 3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--n_est", type=int, default=32,
                    help="TabFM ensemble members; lower (e.g. 8) is much faster on CPU")
    ap.add_argument("--proxy", action="store_true",
                    help="CPU-only: use a gradient booster instead of TabFM to test whether the "
                         "500-cap loses disease signal (no GPU / no TabFM needed)")
    a = ap.parse_args(); ph = a.phase
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))

    if a.proxy:
        from sklearn.ensemble import HistGradientBoostingClassifier as GBM
        from sklearn.metrics import roc_auc_score, average_precision_score
        bf = lambda df: TB.build_feature_table(df, include_gram=True)
        Xtr, ytr, _ = bf(d("train")); Xte, yte, _ = bf(d("test"))
        ytr, yte = np.asarray(ytr), np.asarray(yte)
        cols = sorted(set(Xtr.columns) | set(Xte.columns)); Xtr = Xtr.reindex(columns=cols); Xte = Xte.reindex(columns=cols)
        base = [c for c in cols if c in ("phase", "icd_chapter") or not str(c).startswith("gram_")]
        gram = [c for c in cols if str(c).startswith("gram_")]
        def enc(X, cc):
            X = X[cc].copy()
            for c in cc:
                X[c] = X[c].astype("category").cat.codes if c in ("phase", "icd_chapter") else pd.to_numeric(X[c], errors="coerce").fillna(0)
            return X.values
        G = pd.DataFrame(Xtr[gram]).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        corr = np.abs([np.corrcoef(G[:, j], ytr)[0, 1] if G[:, j].std() > 0 else 0 for j in range(G.shape[1])])
        print(f"=== phase {ph}: is the 500-cap the bottleneck? (classical GBM proxy) ===")
        def gbm(cc, tag):
            m = GBM(random_state=0).fit(enc(Xtr, cc), ytr); p = m.predict_proba(enc(Xte, cc))[:, 1]
            print(f"  {tag:24} ncol={len(cc):4}  ROC={roc_auc_score(yte,p):.3f}  PR={average_precision_score(yte,p):.3f}")
        for k in (500, 1000):
            gbm(base + [gram[j] for j in np.argsort(-corr)[: max(k - len(base), 0)]], f"meta+top{k}-GRAM")
        gbm(base + gram, "meta+FULL-GRAM"); gbm(base, "meta only")
        print(f"  (HINT {HINT[ph][0]:.3f}). top-500 ~ full GRAM => the disease signal saturates below "
              "500 features, so the cap is not the bottleneck.")
        return

    from tabfm import TabFMClassifier, tabfm_v1_0_0_jax as v1
    bf = lambda df: TB.build_feature_table(df, include_gram=True)
    Xtr, ytr, _ = bf(d("train")); Xte, yte, _ = bf(d("test"))
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    cols = sorted(set(Xtr.columns) | set(Xte.columns))
    Xtr = Xtr.reindex(columns=cols); Xte = Xte.reindex(columns=cols)
    base = [c for c in cols if c in ("phase", "icd_chapter") or not str(c).startswith("gram_")]
    gram = [c for c in cols if str(c).startswith("gram_")]
    # rank GRAM columns by |point-biserial corr| with the label on TRAIN
    G = Xtr[gram].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    corr = np.abs([np.corrcoef(G[:, j], ytr)[0, 1] if G[:, j].std() > 0 else 0 for j in range(G.shape[1])])
    keep_gram = [gram[j] for j in np.argsort(-corr)[: max(a.k - len(base), 0)]]
    keep = base + keep_gram
    print(f"=== phase {ph}: {len(cols)} GRAM+meta cols -> top {len(keep)} (<= {a.k} cap) for TabFM ===")

    def run(X_tr, X_te, tag):
        clf = TabFMClassifier(model=v1.load(), n_estimators=a.n_est)
        clf.fit(X_tr, ytr); p = clf.predict_proba(X_te)[:, 1]
        print(f"  {tag:26} ROC={roc_auc_score(yte,p):.3f} {boot_ci(yte,p,roc_auc_score)}  "
              f"PR={average_precision_score(yte,p):.3f}")
    run(Xtr[keep], Xte[keep], f"TabFM meta+top{a.k}-GRAM")
    run(Xtr[base], Xte[base], "TabFM meta+RDKit+protocol")
    print(f"  {'HINT (run A)':26} ROC={HINT[ph][0]:.3f}          PR={HINT[ph][1]:.3f}")
    print("\nReading: if meta+top500-GRAM lifts TabFM to the classical GRAM result (~0.69), the "
          "500-cap was the only thing holding TabFM back; if not, the cap is not the bottleneck.")

if __name__ == "__main__":
    main()
