# -*- coding: utf-8 -*-
"""
paper/fix_icd_shape.py — REPAIR the icdcodes shape in the produced pkls
===================================================================
BUG: in the faithful path of build_mexa_data.py an extra dimension was added to
the icd embedding (.unsqueeze(0)). icd2vec.to_vec(codes) already returns a vector
PER CODE, i.e. [n_codes, 64]. With the extra dimension it became [1, n_codes, 64],
so when the dataset's icollect did sum(axis=0) it stayed [n_codes,64] and itokens
came out as (5, n, 64) — when it should be (5, 64).

This script fixes ONLY the 'icdcodes' field; it does NOT touch the expensive
criteria/BioBERT embeddings. So you do not need to rerun BioBERT from scratch.

Idempotent: it skips files that are already correct, so rerunning is harmless.

Usage (run AFTER the build is FINISHED):
    cd paper
    python fix_icd_shape.py            # all phases/splits
    python fix_icd_shape.py I          # phase I only
"""
import os
import sys
from glob import glob


def fix_one(path, torch, read_pkl, save_pkl):
    """Repairs one pkl. Returns: 'fixed' | 'ok' | 'skip'"""
    d = read_pkl(path)
    icds = d.get('icdcodes')
    if not isinstance(icds, (list, tuple)) or len(icds) == 0:
        return 'skip'

    new, changed = [], False
    for e in icds:
        if not hasattr(e, 'dim'):
            new.append(e)
            continue
        if e.dim() == 3 and e.shape[0] == 1:      # [1, n, 64] -> [n, 64]
            new.append(e.squeeze(0)); changed = True
        elif e.dim() == 1:                        # [64] -> [1, 64]
            new.append(e.unsqueeze(0)); changed = True
        else:
            new.append(e)

    if not changed:
        return 'ok'
    d['icdcodes'] = new
    save_pkl(d, path)
    return 'fixed'


def main(only_phase=None):
    import config as C
    sys.path.insert(0, C.MEXA_DIR)
    import torch
    from utils.utils import read_pkl, save_pkl

    data_folder = C.MEXA_DATA_FOLDER
    phases = [only_phase] if only_phase else ['I', 'II', 'III']
    print(f'DataPath: {data_folder}')

    total = {'fixed': 0, 'ok': 0, 'skip': 0}
    for phase in phases:
        for sub in ['trainphase', 'validphase', 'test']:
            files = sorted(glob(f'{data_folder}{sub}/phase_{phase}/*.pkl'))
            if not files:
                continue
            counts = {'fixed': 0, 'ok': 0, 'skip': 0}
            for p in files:
                counts[fix_one(p, torch, read_pkl, save_pkl)] += 1
            for k in total:
                total[k] += counts[k]
            print(f'  phase {phase:3s} / {sub:11s}: {len(files):5d} files  '
                  f'repaired={counts["fixed"]}  already-correct={counts["ok"]}  skipped={counts["skip"]}')

    print(f'\nTOTAL  repaired={total["fixed"]}  already-correct={total["ok"]}  skipped={total["skip"]}')
    print('Now validate:  python check_mexa_data.py I')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
