# -*- coding: utf-8 -*-
"""
tabfm_probe_gpu.py — representation-vs-shortcut probes run on the ACTUAL TabFM
foundation model (not the GBM proxy). Needs the jax[cuda12] `tabfm` env + GPU;
run on the cluster (g124). The CPU proxy version is representation_probe.py.

Same four probes, so the two can be compared directly:
  1. modality / disease ablation — TabFM given full vs drop-molecule vs drop-protocol
     vs DROP-DISEASE vs disease-only feature tables.
  2. label-shuffle null control  — shuffle the in-context TRAIN labels TabFM conditions
     on; a model using a real signal must fall to ~0.5. This is the sharpest test that
     TabFM is not exploiting a dataset artefact or memorised pattern.
  3. group permutation importance — permute each feature group in the test table,
     measure TabFM's ROC drop (few repeats — each is a full in-context inference).
  4. univariate max — strongest single feature (model-agnostic; a >0.70 flags a leak).

Usage (on g124, tabfm env, GPU):
    python tabfm_probe_gpu.py --phase III
    python tabfm_probe_gpu.py --phase all
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C
import tabfm_baseline as TB


def load_tabfm(n_estimators=32):
    from tabfm import TabFMClassifier, tabfm_v1_0_0_jax as v1
    return lambda: TabFMClassifier(model=v1.load(), n_estimators=n_estimators)


def groups_of(cols):
    mol = [c for c in cols if str(c).startswith("rdkit_")]
    prot = [c for c in cols if str(c).startswith("crit_")]
    disease = [c for c in cols if c in ("icd_chapter", "n_diseases", "n_icdcodes", "n_icd_chapters")]
    other = [c for c in cols if c not in mol + prot + disease]
    return {"molecule": mol, "protocol": prot, "disease": disease, "other": other}


def tabfm_score(mk, Xtr, ytr, Xte, yte, cols=None):
    tr = Xtr if cols is None else Xtr[cols]
    te = Xte if cols is None else Xte[cols]
    clf = mk(); clf.fit(tr, np.asarray(ytr))
    p = clf.predict_proba(te)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def run_phase(ph, mk):
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))
    Xtr, ytr, _ = TB.build_feature_table(d("train"))
    Xte, yte, _ = TB.build_feature_table(d("test"))
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    cols = list(Xtr.columns); g = groups_of(cols)
    print(f"\n============ TabFM phase {ph} (test n={len(yte)}, pos={yte.mean():.3f}) ============")

    print("\n[1] modality / disease ablation (TabFM)")
    sets = {"full": cols,
            "drop-molecule": [c for c in cols if c not in g["molecule"]],
            "drop-protocol": [c for c in cols if c not in g["protocol"]],
            "DROP-DISEASE": [c for c in cols if c not in g["disease"]],
            "disease-only": g["disease"] + g["other"]}
    for name, cc in sets.items():
        r, pr = tabfm_score(mk, Xtr, ytr, Xte, yte, cc)
        print(f"    {name:14} ncol={len(cc):3}  ROC={r:.4f}  PR={pr:.4f}")

    print("\n[2] label-shuffle null (shuffle in-context TRAIN labels; expect ROC ~ 0.5)")
    rng = np.random.default_rng(2023); rocs = []
    for i in range(5):
        r, _ = tabfm_score(mk, Xtr, rng.permutation(ytr), Xte, yte)
        rocs.append(r)
    rocs = np.array(rocs)
    r_real, _ = tabfm_score(mk, Xtr, ytr, Xte, yte)
    print(f"    shuffled-label ROC = {rocs.mean():.4f} ± {rocs.std():.4f} "
          f"(range {rocs.min():.3f}-{rocs.max():.3f})  vs real ROC = {r_real:.4f}")

    print("\n[3] group permutation importance (permute group in test; 3 repeats)")
    rng = np.random.default_rng(0)
    clf = mk(); clf.fit(Xtr, ytr)
    r_full = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    for name, cc in g.items():
        if not cc: continue
        drops = []
        for _ in range(3):
            Xp = Xte.copy(); Xp[cc] = Xp[cc].values[rng.permutation(len(Xp))]
            drops.append(r_full - roc_auc_score(yte, clf.predict_proba(Xp)[:, 1]))
        print(f"    {name:10} ({len(cc):2} cols): ΔROC = {np.mean(drops):+.4f}")

    print("\n[4] strongest single feature (model-agnostic; >0.70 flags a leak)")
    best = ("", 0.5)
    for c in cols:
        v = pd.to_numeric(Xte[c], errors="coerce").values
        if np.nanstd(v) == 0 or np.isnan(v).all(): continue
        a = roc_auc_score(yte, np.nan_to_num(v)); a = max(a, 1 - a)
        if a > best[1]: best = (c, a)
    print(f"    max univariate test AUC = {best[1]:.4f}  ({best[0]})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="III")
    phase = ap.parse_args().phase
    import jax
    if not any(d.platform in ("gpu", "cuda") for d in jax.devices()):
        print("WARNING: no GPU visible to JAX; TabFM will be slow on CPU.")
    mk = load_tabfm()
    for ph in (["I", "II", "III"] if phase == "all" else [phase]):
        run_phase(ph, mk)


if __name__ == "__main__":
    main()
