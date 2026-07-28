# -*- coding: utf-8 -*-
"""
tabfm_baseline.py — Option A: TabFM standalone baseline (feature + model module)
==============================================================================
Added as a THIRD model to the HINT+MEXA project / paper harness.

This module does two things:
  1) build_feature_table(): extracts THREE feature layers from the HINT TOP CSV
     (metadata + RDKit descriptors + protocol scalars).
  2) fit_predict_tabfm(): using the REAL TabFM in ctp/tabfm-main
     (context = train), produces valid/test probabilities. If the TabFM
     environment (jax/absl) is missing, a GBM fallback kicks in — the pipeline still runs.

Evaluation is NOT done in THIS module. Raw scores {nctid, score, label, split}
are dumped, and paper/eval_harness.py (threshold@val + PR/ROC on scores + bootstrap)
evaluates them by the SAME rules as HINT/MEXA, so the comparison is fair.

Usage: called by run_optionA.py. Paths come from there / paper/config.py.
"""
from __future__ import annotations

import ast
import re
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Columns that could indirectly leak the label — the leakage guard never turns
# these into features. 'status'/'why_stop' reveal the outcome; 'label' is the target.
LEAKY_COLUMNS = {"status", "why_stop", "label", "nctid"}


# --------------------------------------------------------------------------- #
# 1. Parsers — HINT CSVs store lists (sometimes DOUBLE-encoded)
# --------------------------------------------------------------------------- #

def _to_list(cell) -> List[str]:
    """Converts '["a","b"]', 'a;b' or nested stringified lists into a flat list."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    out: List[str] = []
    stack = [cell]
    guard = 0
    while stack and guard < 5000:
        guard += 1
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue
        s = str(item).strip()
        if not s:
            continue
        if len(s) >= 2 and s[0] in "[(" and s[-1] in "])":
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    stack.extend(parsed)
                else:
                    stack.append(parsed)
                continue
            except (ValueError, SyntaxError):
                pass
        token = s.strip("'\"")
        # a single element may still be a list-string (double encoding)
        if len(token) >= 2 and token[0] in "[(" and token[-1] in "])":
            try:
                parsed = ast.literal_eval(token)
                if isinstance(parsed, (list, tuple)):
                    stack.extend(parsed)
                    continue
            except (ValueError, SyntaxError):
                pass
        if token:
            out.append(token)
    return out


def _icd_chapter(code: str) -> str:
    m = re.search(r"[A-Za-z]", str(code))
    return m.group(0).upper() if m else "?"


# --- GRAM ancestor dictionary (lazy) — code -> list of ancestor codes ---
_GRAM = None


def _gram_dict():
    global _GRAM
    if _GRAM is None:
        import os
        import pickle
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "hint", "data", "icdcode2ancestor_dict.pkl")
        try:
            _GRAM = pickle.load(open(p, "rb"))
        except Exception:
            _GRAM = {}
    return _GRAM


# --------------------------------------------------------------------------- #
# 2. Layer 3 — protocol (criteria) scalars (NOT the BioBERT embedding)
# --------------------------------------------------------------------------- #

def protocol_scalars(criteria: str) -> dict:
    text = str(criteria) if criteria is not None else ""
    low = text.lower()
    exc_idx = low.find("exclusion")
    if exc_idx == -1:
        inc_text, exc_text = text, ""
    else:
        inc_text, exc_text = text[:exc_idx], text[exc_idx:]

    def _bullets(t: str) -> int:
        return len(re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])", t))

    n_inc, n_exc = _bullets(inc_text), _bullets(exc_text)
    n_lines = text.count("\n") + 1
    return {
        "crit_char_len": len(text),
        "crit_word_len": len(text.split()),
        "crit_n_lines": n_lines,
        "crit_n_criteria": max(n_inc + n_exc, n_lines),
        "crit_n_inclusion": n_inc,
        "crit_n_exclusion": n_exc,
        "crit_excl_incl_ratio": (n_exc / n_inc) if n_inc > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# 3. Layer 2 — RDKit physicochemical descriptors (KEY POINT)
# --------------------------------------------------------------------------- #

_KEY_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumRotatableBonds", "RingCount",
    "NumHDonors", "NumHAcceptors", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NumValenceElectrons", "qed",
]


def _rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
        # Silence the error spam that floods the screen for broken SMILES;
        # instead build_feature_table() reports a one-line summary.
        RDLogger.DisableLog("rdApp.*")
        return Chem, Descriptors, list(Descriptors.descList)
    except Exception:
        return None, None, None


def molecule_features(smiles_list: List[str], full: bool = False,
                      agg: tuple = ("mean", "max")) -> dict:
    """RDKit descriptors from SMILES; aggregated at the trial level (agg).
    We cannot supply the D-MPNN graph, but we keep the molecule physicochemistry."""
    Chem, Descriptors, desc_list = _rdkit()
    if Chem is None:
        return {}                              # no RDKit -> layer empty, code still runs
    chosen = desc_list if full else [(n, f) for (n, f) in desc_list
                                     if n in set(_KEY_DESCRIPTORS)]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        vals = {}
        for name, fn in chosen:
            try:
                vals[name] = float(fn(mol))
            except Exception:
                vals[name] = np.nan
        rows.append(vals)

    aggfns = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin,
              "sum": np.nansum}
    feats = {}
    if not rows:
        for name, _ in chosen:
            for a in agg:
                feats[f"rdkit_{name}_{a}"] = np.nan
        return feats
    df = pd.DataFrame(rows)
    for name, _ in chosen:
        col = df[name].to_numpy(dtype=float)
        for a in agg:
            feats[f"rdkit_{name}_{a}"] = float(aggfns[a](col)) if len(col) else np.nan
    return feats


def morgan_features(smiles_list: List[str], n_bits: int = 256,
                    radius: int = 2) -> dict:
    """Morgan/ECFP fingerprint — the BIT-VECTOR counterpart of the molecular graph.
    A separate 'molecule' signal, independent of the RDKit descriptors. Aggregated at
    the trial level by bit-union (a bit is 1 if set in any drug)."""
    Chem, _, _ = _rdkit()
    if Chem is None:
        return {}
    from rdkit.Chem import AllChem
    union = np.zeros(n_bits, dtype=int)
    seen = False
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=int)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        union |= arr
        seen = True
    if not seen:
        return {f"morgan_{i}": 0 for i in range(n_bits)}
    return {f"morgan_{i}": int(union[i]) for i in range(n_bits)}


def gram_features(icds: List[str]) -> dict:
    """GRAM ontology ancestors — hierarchical encoding of the disease (multi-hot).
    A finer disease encoding than the ICD chapter. The vocabulary learned on train
    is aligned via reindex in build_feature_table."""
    gram = _gram_dict()
    if not gram:
        return {}
    anc = set()
    for c in icds:
        anc.add(str(c))
        for a in gram.get(str(c), []):
            anc.add(str(a))
    feats = {f"gram_{a}": 1 for a in anc}
    feats["n_gram_ancestors"] = len(anc)
    return feats


# --------------------------------------------------------------------------- #
# 4. Layer 1 + assembly
# --------------------------------------------------------------------------- #

def row_features(row: pd.Series, full_rdkit: bool = False,
                 rdkit_agg: tuple = ("mean", "max"),
                 include_rdkit: bool = True, include_protocol: bool = True,
                 include_morgan: bool = False, include_gram: bool = False,
                 morgan_bits: int = 256, icd_mode: str = "both") -> dict:
    smiles = _to_list(row.get("smiless"))
    icds = _to_list(row.get("icdcodes"))
    diseases = _to_list(row.get("diseases"))
    drugs = _to_list(row.get("drugs"))
    feats = {
        "phase": str(row.get("phase", "")).strip(),          # kategorik
        "n_drugs": len(drugs),
        "n_smiles": len(smiles),
        "n_diseases": len(diseases),
    }
    # ICD encoding: chapter (categorical) / count (scalar) / both
    if icd_mode in ("both", "count"):
        feats["n_icdcodes"] = len(icds)
        feats["n_icd_chapters"] = len({_icd_chapter(c) for c in icds}) if icds else 0
    if icd_mode in ("both", "chapter"):
        feats["icd_chapter"] = (_icd_chapter(icds[0]) if icds else "?")
    if include_rdkit:
        feats.update(molecule_features(smiles, full=full_rdkit, agg=rdkit_agg))
    if include_morgan:
        feats.update(morgan_features(smiles, n_bits=morgan_bits))
    if include_gram:
        feats.update(gram_features(icds))
    if include_protocol:
        feats.update(protocol_scalars(row.get("criteria", "")))
    return feats


CAT_COLS = ["phase", "icd_chapter"]


def build_feature_table(df: pd.DataFrame, full_rdkit: bool = False, **opts):
    """CSV DataFrame -> (X, y, nctid). Categoricals stay STRING (TabFM likes it);
    leakage columns are excluded. **opts pass through to row_features (Morgan/GRAM/
    icd_mode/rdkit_agg/include_* vb.)."""
    leaked = [c for c in df.columns if c in LEAKY_COLUMNS and c != "label"]
    if leaked:
        print(f"[leakage-guard] excluded from features: {leaked}")
    X = pd.DataFrame([row_features(r, full_rdkit=full_rdkit, **opts)
                      for _, r in df.iterrows()])
    # Morgan/GRAM sparse multi-hot: a bit/ancestor absent in a row = 0 (not NaN)
    sparse = [c for c in X.columns if c.startswith(("morgan_", "gram_"))]
    if sparse:
        X[sparse] = X[sparse].fillna(0)

    # --- Molecule-coverage report (a number you need for a fair comparison) ---
    # The HINT data has broken/missing SMILES (like '\N'). For those trials the
    # RDKit layer stays all-NaN -> molecule information is SILENTLY lost.
    rd_cols = [c for c in X.columns if c.startswith("rdkit_")]
    if rd_cols:
        bos = X[rd_cols].isna().all(axis=1).sum()
        pay = 100.0 * bos / max(len(X), 1)
        print(f"[molecule] {len(X)-bos}/{len(X)} trials have RDKit features; "
              f"{bos} trials ({pay:.1f}%) have no molecule (broken/missing SMILES)")

    y = df["label"].astype(int).values if "label" in df.columns else None
    nctid = df["nctid"].astype(str).values if "nctid" in df.columns else \
        np.array([str(i) for i in range(len(df))])
    return X, y, nctid


# --------------------------------------------------------------------------- #
# 5. TabFM (context = train) + GBM fallback
# --------------------------------------------------------------------------- #

def _try_load_tabfm(tabfm_dir: str, n_estimators: int = 32, verbose: bool = True,
                    random_state: int = 42):
    """
    n_estimators: number of TabFM ensemble members (default 32). Each member is a
    separate in-context forward pass over the whole train context; on CPU the
    main cost comes from here. Lowering it to 8 is ~4x faster and, since it does
    NOT truncate the CONTEXT, a scientifically honest setting (should be reported).
    """
    if tabfm_dir and tabfm_dir not in sys.path:
        sys.path.insert(0, tabfm_dir)
    try:
        import tabfm
        model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
        print(f"[tabfm] REAL TabFM (JAX) loaded | n_estimators={n_estimators}"
              f" | random_state={random_state}")
        return tabfm.TabFMClassifier(model=model, n_estimators=n_estimators,
                                     verbose=verbose, random_state=random_state)
    except Exception as e:
        print(f"[tabfm] real TabFM failed to load ({type(e).__name__}) -> GBM fallback")
        if sys.version_info < (3, 11):
            print(f"        THIS IS PROBABLY THE CAUSE: tabfm-main 'requires-python >=3.11', "
                  f"you have Python {sys.version_info.major}.{sys.version_info.minor} "
                  f"(chex==0.1.92 is therefore not installed).\n"
                  f"        Fix: use a separate environment ->\n"
                  f"          conda create -n tabfm python=3.11 -y && conda activate tabfm\n"
                  f"          pip install -r <ctp>/tabfm-main/requirements.txt\n"
                  f"          pip install rdkit scikit-learn pandas")
        print("        NOTE: the fallback result is NOT a TabFM result — pipeline test only.")
        return None


def fit_predict_tabfm(X_train, y_train, X_list, tabfm_dir: str = "",
                      n_estimators: int = 32, verbose: bool = True,
                      random_state: int = 42):
    """
    context = train. X_list: DataFrames to score (e.g. [X_valid, X_test]).
    Returns: a positive-class probability array for each.

    FAIR SPLIT: the outer train/valid/test boundary is the SAME as HINT (temporal).
    TabFM's own internal OOF K-fold is INSIDE THE MODEL — it does not touch the outer boundary.
    """
    clf = _try_load_tabfm(tabfm_dir, n_estimators=n_estimators, verbose=verbose,
                          random_state=random_state)
    if clf is not None:
        clf.fit(X_train, y_train)              # TabFM handles categoricals natively
        return [np.asarray(clf.predict_proba(X)[:, 1]) for X in X_list]

    # ---- Fallback: HistGradientBoosting (categoricals via one-hot) ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    cats = [c for c in CAT_COLS if c in X_train.columns]
    Xtr = pd.get_dummies(X_train, columns=cats, dummy_na=False)
    dummied = [pd.get_dummies(X, columns=cats, dummy_na=False) for X in X_list]
    cols = Xtr.columns
    for d in dummied:
        cols = cols.union(d.columns)
    Xtr = Xtr.reindex(columns=cols, fill_value=0)
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(Xtr.values, y_train)
    outs = []
    for d in dummied:
        d = d.reindex(columns=cols, fill_value=0)
        outs.append(clf.predict_proba(d.values)[:, 1])
    return outs
