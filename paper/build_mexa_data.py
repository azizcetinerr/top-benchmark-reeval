# -*- coding: utf-8 -*-
"""
paper/build_mexa_data.py — Build MEXA embedded data from the HINT CSVs
=========================================================================
MEXA's training data (DataPath/{trainphase,validphase,test}/phase_X/{nctid}.pkl)
is NOT in this repo. This script produces it from HINT's data CSVs, because
MEXA's input already comes from the same TOP benchmark and the HINT CSVs
(nctid,label,icdcodes,smiless,criteria) are compatible with the MEXA embedding
base class.

TWO MODES (both can be done via the Family customization in the README):

  MODE='faithful'  -> MEXA's ORIGINAL embedders:
        smiless : MACAW  (15-dim, random_state=SEED  <- Item 4 fix)
        icdcodes: icd2vec/icdcodex (64-dim, ICD-10-CM hierarchy)
        criteria: BioBERT dmis-lab/biobert-v1.1 (768-dim)
     This mode makes MEXA's numbers FAIRLY comparable with the paper AND HINT.
     Heavy dependencies: macaw, icdcodex, transformers (+BioBERT download).

  MODE='custom'    -> lighter Family subclasses:
        smiless : RDKit Morgan fingerprint -> 15-dim (deterministic folding)
        icdcodes: deterministic hash -> 64-dim
        criteria: still BioBERT (768 text encoder is unavoidable)
     Valid ONLY for the 'fixed vs buggy' INTERNAL A/B; invalid for the
     ABSOLUTE MEXA-vs-HINT comparison (input differs from the original).

OUTPUT SCHEMA (what MEXA ctp/dataset.py expects):
  each {nctid}.pkl = {
     'nctid': str, 'label': int,
     'icdcodes': [ 2D tensor [n_codes, 64], ... ],     # each disease entry
     'smiless' : [ 1D tensor [15], ... ],              # each molecule
     'criteria': [ [in_emb[ns,L,768], in_mask[ns,L]],  # inclusion
                   [ex_emb[ns,L,768], ex_mask[ns,L]] ] # exclusion (zeros if absent)
  }

NOTE: This script is heavy, requires external libraries and a model download;
you will most likely run it IN YOUR OWN ENVIRONMENT. Imports are kept inside
functions so the file can be compiled and inspected without the libraries.
"""
import os
import sys
import ast
import hashlib
import numpy as np

# ----------------------------------------------------------------------
# CONFIG — adjust from here
# ----------------------------------------------------------------------
MODE   = os.environ.get('MEXA_BUILD_MODE', 'faithful')   # 'faithful' | 'custom'
SEED   = 2023                                             # Item 4: MACAW random_state
PHASES = ['I', 'II', 'III']

_HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(_HERE)
HINT_DATA = os.path.join(ROOT, 'hint', 'data')           # phase_X_{split}.csv is here
OUT_DATAPATH = os.path.join(ROOT, 'mexa', 'DataPath') + os.sep   # same as config.MEXA_DATA_FOLDER

# MEXA dataset subset -> folder mapping
SPLIT_DIR = {'train': 'trainphase', 'valid': 'validphase', 'test': 'test'}
# split suffix in the CSV file name
SPLIT_CSV = {'train': 'train', 'valid': 'valid', 'test': 'test'}

BIOBERT_MAXLEN = 32   # same as cembed.py


# ----------------------------------------------------------------------
# CSV field parsing (same logic as the original ensemble.py)
# ----------------------------------------------------------------------
def parse_smiless(cell):
    """"['smiles1', 'smiles2']" -> ['smiles1','smiles2']"""
    try:
        return [s for s in ast.literal_eval(cell)]
    except Exception:
        txt = cell[1:-1]
        return [i.strip()[1:-1] for i in txt.split(',')]

def parse_icdcodes(cell):
    """HINT icdcodes cell -> [[code,...], ...] (code list per disease)"""
    try:
        outer = ast.literal_eval(cell)          # ["['A00','B01']", ...]
        return [ast.literal_eval(x) if isinstance(x, str) else list(x) for x in outer]
    except Exception:
        itxt = cell[2:-2]
        out = []
        for i in itxt.split('", "'):
            i = i[1:-1]
            out.append([j.strip()[1:-1] for j in i.split(',')])
        return out


# ----------------------------------------------------------------------
# ICD-10-CM hierarchy version
# ----------------------------------------------------------------------
# In icdcodex the available version range depends on the library version and the
# error message can be misleading ("2019 to 2025" but it rejects 2025 — upper
# bound exclusive). So instead of hard-coding a year we find the first working
# version.
# ICD_VERSION=None -> automatic (tries newest to oldest).
ICD_VERSION = os.environ.get('ICD_VERSION')   # e.g. '2024'; otherwise automatic


def _load_icd_hierarchy(hierarchy, preferred=None):
    """Find the first working version for hierarchy.icd10cm(...).
    Returns: (graph_tuple, used_version). The version matters for reproducibility."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += [2024, 2023, 2022, 2021, 2020, 2019]

    last_err = None
    for y in candidates:
        for val in (y, str(y)):            # some versions expect int, some str
            try:
                return hierarchy.icd10cm(val), val
            except Exception as e:
                last_err = e
    try:                                   # last resort: no argument (library default)
        return hierarchy.icd10cm(), 'default'
    except Exception:
        raise RuntimeError(
            f'icdcodex hierarchy could not be loaded with any version. Last error: {last_err}')


# ----------------------------------------------------------------------
# FAITHFUL embedder dictionaries (unique value -> embedding)
# ----------------------------------------------------------------------
def build_faithful_smiles_dict(all_smiles, emb_size=15, seed=SEED):
    """SMILES -> emb_size via MACAW. Item 4: random_state=seed -> reproducible."""
    import torch
    from macaw import MACAW
    try:
        mcw = MACAW(n_components=emb_size, random_state=seed)   # <-- Item 4 FIX
    except TypeError:
        # if the library version does not accept random_state, fall back to global seed
        np.random.seed(seed)
        mcw = MACAW(n_components=emb_size)
        print('[warning] MACAW did not accept random_state; tried np.random.seed.')
    smiles_lst = sorted(set(all_smiles))
    mcw.fit(smiles_lst)
    embs = mcw.transform(smiles_lst)
    d = {}
    for emb, smi in zip(embs, smiles_lst):
        t = torch.from_numpy(np.asarray(emb, dtype=np.float32))
        if torch.isnan(t).any():
            t = torch.zeros(emb_size)
        d[smi] = t
    return d

def build_faithful_icd_dict(all_merged_codes, emb_size=64, seed=SEED):
    """ICD code -> emb_size via icd2vec (icdcodex)."""
    import torch
    from icdcodex import icd2vec, hierarchy
    try:
        embedder = icd2vec.Icd2Vec(num_embedding_dimensions=emb_size, workers=-1, seed=seed)
    except TypeError:
        # if this icdcodex version does not accept 'seed', build without it (weaker reproducibility)
        np.random.seed(seed)
        embedder = icd2vec.Icd2Vec(num_embedding_dimensions=emb_size, workers=-1)
        print('[warning] Icd2Vec did not accept seed; tried np.random.seed.')

    graph, used_version = _load_icd_hierarchy(hierarchy, ICD_VERSION)
    print(f'[icd] ICD-10-CM version used: {used_version}  '
          f'(note this for REPRODUCIBILITY — if the version changes, embeddings change)')
    embedder.fit(*graph)
    d = {}
    for merged in sorted(set(all_merged_codes)):
        codes = merged.split('~')
        try:
            # to_vec returns a vector PER CODE -> [n_codes, emb_size] (must stay 2D!)
            vec = torch.from_numpy(np.asarray(embedder.to_vec(codes), dtype=np.float32))
            d[merged] = vec.unsqueeze(0) if vec.dim() == 1 else vec
        except Exception:
            d[merged] = torch.zeros(1, emb_size)      # [1, emb_size]
    return d


# ----------------------------------------------------------------------
# CUSTOM embedders (lightweight, deterministic)
# ----------------------------------------------------------------------
def custom_smiles_emb(smiles, emb_size=15):
    """RDKit Morgan fingerprint -> deterministic folding to emb_size."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return torch.zeros(emb_size)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    folded = np.zeros(emb_size, dtype=np.float32)          # 2048 -> emb_size
    np.add.at(folded, np.arange(arr.shape[0]) % emb_size, arr)  # modular folding (feature hashing)
    folded = folded / (np.linalg.norm(folded) + 1e-6)
    return torch.from_numpy(folded.astype(np.float32))

def custom_icd_code_emb(code, emb_size=64):
    """Single ICD code -> deterministic 64-dim (hash based)."""
    import torch
    h = hashlib.sha256(code.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], 'little'))
    v = rng.standard_normal(emb_size).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-6)
    return torch.from_numpy(v)


# ----------------------------------------------------------------------
# Criteria (BioBERT in both modes — 768 text encoder is unavoidable)
# ----------------------------------------------------------------------
def make_criteria_embedder():
    """Reuse MEXA's original criteriaEmbedding (clean_step + biobert).
    Loads the model ONCE and caches it (the original loaded it on every call — slow)."""
    sys.path.insert(0, os.path.join(ROOT, 'mexa'))
    from criteria_embedding.cembed import criteriaEmbedding
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    mdl = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
    mdl.eval()

    def biobert(data):
        enc = tok(data, add_special_tokens=True, max_length=BIOBERT_MAXLEN,
                  padding="max_length", truncation=True, return_attention_mask=True)
        ids = torch.tensor(enc['input_ids']); am = torch.tensor(enc['attention_mask'])
        with torch.no_grad():
            out = mdl(ids, attention_mask=am)[0]
        return [out, am]

    return criteriaEmbedding, biobert


# ----------------------------------------------------------------------
# Main producer: one split CSV -> merged pkls
# ----------------------------------------------------------------------
def build_split(phase, split, mode=MODE, smiles_dict=None, icd_dict=None,
                crit_cls=None, crit_biobert=None):
    import csv, torch
    from utils.utils import save_pkl                      # mexa/utils
    # same key as merge_code in ensemble.py
    merge_code = lambda codes: '~'.join(codes)

    csv_path = os.path.join(HINT_DATA, f'phase_{phase}_{SPLIT_CSV[split]}.csv')
    out_dir  = os.path.join(OUT_DATAPATH, SPLIT_DIR[split], f'phase_{phase}') + os.sep
    os.makedirs(out_dir, exist_ok=True)

    # use the original cleaner for criteria via a one-line helper
    ce = crit_cls(csv_path, 'criteria', out_dir) if crit_cls else None

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        nctid = row['nctid']; label = int(row['label'])

        # --- smiless ---
        smis = parse_smiless(row['smiless'])
        if mode == 'faithful':
            s_emb = [smiles_dict.get(s, torch.zeros(15)) for s in smis]
        else:
            s_emb = [custom_smiles_emb(s, 15) for s in smis]

        # --- icdcodes --- (each disease entry -> 2D tensor [n_codes, 64])
        icds = parse_icdcodes(row['icdcodes'])
        i_emb = []
        for codes in icds:
            if mode == 'faithful':
                # icd2vec.to_vec(codes) returns one vector PER CODE -> already [n_codes, 64].
                # (There used to be an .unsqueeze(0) here; it made [1,n,64] and broke the schema.)
                mat = icd_dict.get(merge_code(codes))
                if mat is None:
                    mat = torch.zeros(1, 64)
                if mat.dim() == 1:                 # safety: [64] -> [1,64]
                    mat = mat.unsqueeze(0)
                i_emb.append(mat)                  # [n_codes, 64]
            else:
                mat = torch.stack([custom_icd_code_emb(c, 64) for c in codes])  # [n,64]
                i_emb.append(mat)

        # --- criteria --- (BioBERT, same in both modes)
        clean = ce.clean_step(idx)          # original pattern extractor
        if len(clean) == 0:
            c_emb = [[torch.zeros(1, BIOBERT_MAXLEN, 768), torch.zeros(1, BIOBERT_MAXLEN)],
                     [torch.zeros(1, BIOBERT_MAXLEN, 768), torch.zeros(1, BIOBERT_MAXLEN)]]
        elif len(clean) == 1:
            c_emb = [crit_biobert(clean[0]),
                     [torch.zeros(1, BIOBERT_MAXLEN, 768), torch.zeros(1, BIOBERT_MAXLEN)]]
        else:
            c_emb = [crit_biobert(clean[0]), crit_biobert(clean[1])]

        save_pkl({'nctid': nctid, 'label': label,
                  'icdcodes': i_emb, 'smiless': s_emb, 'criteria': c_emb},
                 out_dir + f'{nctid}.pkl')

    print(f'[build] phase {phase} / {split}: {len(rows)} trials -> {out_dir}')


def main():
    import csv
    sys.path.insert(0, os.path.join(ROOT, 'mexa'))
    print(f'MODE = {MODE} | SEED = {SEED}')
    print(f'HINT_DATA = {HINT_DATA}')
    print(f'OUT_DATAPATH = {OUT_DATAPATH}')

    smiles_dict = icd_dict = None
    if MODE == 'faithful':
        # collect unique smiles/icd across all splits, build the dictionaries once
        all_s, all_i = [], []
        for phase in PHASES:
            for split in ['train', 'valid', 'test']:
                p = os.path.join(HINT_DATA, f'phase_{phase}_{SPLIT_CSV[split]}.csv')
                for row in csv.DictReader(open(p, newline='')):
                    all_s += parse_smiless(row['smiless'])
                    for codes in parse_icdcodes(row['icdcodes']):
                        all_i.append('~'.join(codes))
        print(f'[dict] unique smiles={len(set(all_s))}, icd-keys={len(set(all_i))}')
        smiles_dict = build_faithful_smiles_dict(all_s, 15, SEED)
        icd_dict    = build_faithful_icd_dict(all_i, 64, SEED)

    crit_cls, crit_biobert = make_criteria_embedder()

    for phase in PHASES:
        for split in ['train', 'valid', 'test']:
            build_split(phase, split, MODE, smiles_dict, icd_dict, crit_cls, crit_biobert)
    print('DONE. config.py MEXA_DATA_FOLDER =', OUT_DATAPATH)


if __name__ == '__main__':
    main()
