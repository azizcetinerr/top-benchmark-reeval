# -*- coding: utf-8 -*-
"""
representation_probe.py — is the tabular signal a real (disease) relationship or a
spurious shortcut/artefact? Runs on the classical GBM proxy that shares TabFM's
inputs (CPU). The TabFM version of the same probes runs on GPU: tabfm_probe_gpu.py.

Four probes per phase:
  1. modality ablation  — full vs drop-molecule vs drop-protocol vs disease-only vs
     DROP-DISEASE. If dropping disease collapses ROC toward chance, disease is the signal.
  2. label-shuffle null — shuffle TRAIN labels, refit, score test. A model exploiting a
     real signal collapses to ~0.5; if it stays high, there is leakage/memorisation.
  3. group permutation  — permute each feature group in the test set, measure ROC drop.
  4. univariate max     — the single most predictive feature (a >0.7 here would flag a leak).

Usage:  python representation_probe.py --phase III
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C
import tabfm_baseline as TB

def groups_of(cols):
    mol = [c for c in cols if c.startswith("rdkit_")]
    prot = [c for c in cols if c.startswith("crit_")]
    disease = [c for c in cols if c == "icd_chapter" or c.startswith("icd_chapter_")
               or c in ("n_diseases", "n_icdcodes", "n_icd_chapters")]
    other = [c for c in cols if c not in mol + prot + disease]  # phase, n_drugs, n_smiles
    return {"molecule": mol, "protocol": prot, "disease": disease, "other": other}

def encode(Xtr, Xte):
    """One-hot categoricals, align train/test columns."""
    cat = [c for c in Xtr.columns if Xtr[c].dtype == object or c in ("phase", "icd_chapter")]
    Etr = pd.get_dummies(Xtr, columns=cat, dummy_na=True)
    Ete = pd.get_dummies(Xte, columns=cat, dummy_na=True)
    Ete = Ete.reindex(columns=Etr.columns, fill_value=0)
    return Etr, Ete

def fit_score(Etr, ytr, Ete, yte, cols=None, seed=0):
    tr = Etr if cols is None else Etr[cols]
    te = Ete if cols is None else Ete[cols]
    m = GBM(random_state=seed).fit(tr.values, ytr)
    p = m.predict_proba(te.values)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p), m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE)
    ph = ap.parse_args().phase
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))
    Xtr, ytr, _ = TB.build_feature_table(d("train"))
    Xte, yte, _ = TB.build_feature_table(d("test"))
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    Etr, Ete = encode(Xtr, Xte)
    g = groups_of(Etr.columns)

    print(f"\n================ phase {ph} (GBM proxy, test n={len(yte)}, pos={yte.mean():.3f}) ================")

    # 1) modality ablation ---------------------------------------------------
    print("\n[1] modality / disease ablation")
    sets = {
        "full":            list(Etr.columns),
        "drop-molecule":   [c for c in Etr.columns if c not in g["molecule"]],
        "drop-protocol":   [c for c in Etr.columns if c not in g["protocol"]],
        "DROP-DISEASE":    [c for c in Etr.columns if c not in g["disease"]],
        "disease-only":    g["disease"] + g["other"],
    }
    for name, cols in sets.items():
        r, pr, _ = fit_score(Etr, ytr, Ete, yte, cols)
        print(f"    {name:14} ncol={len(cols):3}  ROC={r:.4f}  PR={pr:.4f}")

    # 2) label-shuffle null control -----------------------------------------
    print("\n[2] label-shuffle null (full features; expect ROC ~ 0.5)")
    rng = np.random.default_rng(2023); rocs = []
    for i in range(10):
        ysh = rng.permutation(ytr)
        r, _, _ = fit_score(Etr, ysh, Ete, yte, seed=i)
        rocs.append(r)
    rocs = np.array(rocs)
    print(f"    shuffled-label test ROC = {rocs.mean():.4f} ± {rocs.std():.4f} "
          f"(range {rocs.min():.3f}-{rocs.max():.3f})  vs real full ROC = "
          f"{fit_score(Etr,ytr,Ete,yte)[0]:.4f}")

    # 3) group permutation importance ---------------------------------------
    print("\n[3] group permutation importance (ROC drop when a group is shuffled in test)")
    r_full, _, m = fit_score(Etr, ytr, Ete, yte)
    rng = np.random.default_rng(0)
    for name, cols in g.items():
        if not cols: continue
        drops = []
        for _ in range(10):
            Ep = Ete.copy()
            idx = rng.permutation(len(Ep))
            Ep[cols] = Ep[cols].values[idx]
            drops.append(r_full - roc_auc_score(yte, m.predict_proba(Ep.values)[:, 1]))
        print(f"    {name:10} ({len(cols):2} cols): ΔROC = {np.mean(drops):+.4f} ± {np.std(drops):.4f}")

    # 4) univariate max ------------------------------------------------------
    print("\n[4] strongest single feature (a >0.70 would flag a leak)")
    best = ("", 0.5)
    for c in Ete.columns:
        v = Ete[c].values.astype(float)
        if np.nanstd(v) == 0: continue
        a = roc_auc_score(yte, v); a = max(a, 1 - a)
        if a > best[1]: best = (c, a)
    print(f"    max univariate test AUC = {best[1]:.4f}  ({best[0]})")

if __name__ == "__main__":
    main()
