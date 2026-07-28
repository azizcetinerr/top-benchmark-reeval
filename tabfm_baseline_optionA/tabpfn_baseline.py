# -*- coding: utf-8 -*-
"""
tabpfn_baseline.py — TabPFN v2 as a SECOND zero-training foundation model (reviewer Q3).

Why: the paper's zero-training headline currently rests on TabFM, whose only reference is a
GitHub repo. TabPFN v2 (Hollmann et al., Nature 2025) is peer-reviewed; running it as a second,
independent zero-training model closes the "headline claim on a non-peer-reviewed model"
objection. Both are prior-fitted transformers that predict in-context with no gradient
training. Like TabFM, TabPFN caps at ~500 features, so we use the 38-column
metadata+RDKit+protocol table (the same table on which TabFM's parity with HINT is
established), not the ~4500-column GRAM set.

Install:  pip install tabpfn            # v2; CPU works for this data size, GPU faster
Run:      python tabpfn_baseline.py --phase III        # or I / II
Output is a preds CSV in the shared schema, then evaluate with the harness:
          preds/preds_tabpfn_phase_{X}.csv  ->  compare via paper/eval_harness.py
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import config as C, tabfm_baseline as TB

def boot_ci(y, s, fn, n=1000, seed=2023):
    rng = np.random.default_rng(seed); m = len(y)
    return np.round(np.percentile([fn(y[i], s[i]) for i in (rng.integers(0,m,m) for _ in range(n))], [2.5,97.5]), 3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default=C.PHASE); ph = ap.parse_args().phase
    from tabpfn import TabPFNClassifier
    d = lambda sp: pd.read_csv(os.path.join(C.HINT_DIR, "data", f"phase_{ph}_{sp}.csv"))
    Xtr, ytr, _ = TB.build_feature_table(d("train"))          # 38-col table (<=500 feat)
    Xte, yte, itest = TB.build_feature_table(d("test"))
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    # numeric encode (TabPFN takes numeric arrays; one-hot the two categoricals)
    A = pd.get_dummies(pd.concat([Xtr, Xte], keys=["tr","te"]), columns=["phase","icd_chapter"], dummy_na=True)
    Etr, Ete = A.xs("tr").fillna(0).values.astype(float), A.xs("te").fillna(0).values.astype(float)

    # device: use GPU if torch sees one, else CPU (and lift the >1000-sample CPU guard).
    dev = "cpu"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    clf = TabPFNClassifier(device=dev, ignore_pretraining_limits=True)   # zero training; in-context
    print(f"  [TabPFN device={dev}; train n={len(ytr)}]")
    clf.fit(Etr, ytr)
    p = clf.predict_proba(Ete)[:, 1]
    roc, pr = roc_auc_score(yte, p), average_precision_score(yte, p)
    print(f"=== TabPFN v2, phase {ph} (test n={len(yte)}) ===")
    print(f"  TabPFN ROC={roc:.3f} {boot_ci(yte,p,roc_auc_score)}  PR={pr:.3f} {boot_ci(yte,p,average_precision_score)}")
    hint_roc = {"I": 0.574, "II": 0.621, "III": 0.685}[ph]
    print(f"  HINT   ROC={hint_roc:.3f}   (overlapping CIs => a second, peer-reviewed zero-training model also matches HINT)")

    out = os.path.join(HERE, "preds", f"preds_tabpfn_phase_{ph}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"nctid": itest, "score": p, "label": yte, "split": "test"}).to_csv(out, index=False)
    print(f"  [written] {out}  (evaluate with paper/eval_harness.py for CI + threshold)")

if __name__ == "__main__":
    main()
