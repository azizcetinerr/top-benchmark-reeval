# -*- coding: utf-8 -*-
"""
paper/dump_mexa_preds.py — Dump MEXA raw scores
==================================================
Loads MEXA's ready checkpoint, runs a forward pass on the val and test sets,
applies sigmoid to the logit output and writes preds/preds_mexa_phase_{X}.csv
as {nctid, score, label, split}.

CAUTION: MEXA's embedded data (output of createDataset.py, *.pkl) is required.
That data is NOT in this repo right now. Without the data the function raises a
clear error; once the data is ready (MEXA_DATA_FOLDER set) the same code runs.

MEXA preds = logits. ROC/PR ranking metrics are insensitive to monotonic
transformation, and since the threshold is selected on validation in the
harness, the logit/probability difference is immaterial. Still, we apply sigmoid
and write a 0-1 score so it is interpretable.
"""
import os
import sys
import csv
import json
import argparse
from glob import glob


def _load_args(args_json):
    with open(args_json) as f:
        d = json.load(f)
    return argparse.Namespace(**d)


def _find_args_json(ckpt_path):
    """Find args.json. There are two layouts:
       - downloaded ckpts : <dir>/rocbest.pth.tar      + <dir>/args.json
       - training runs    : <run>/checkpoints/x.pth.tar + <run>/args.json
    """
    d = os.path.dirname(ckpt_path)
    for cand in (os.path.join(d, 'args.json'),
                 os.path.join(os.path.dirname(d), 'args.json')):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f'args.json not found (looked in: {d} and its parent folder)')


def dump_mexa(phase, ckpt_path, data_folder, out_csv, mexa_dir, device='cpu',
              moduledict_fix=None, seed=2023):
    """moduledict_fix:
         None  -> AUTO-detect from the checkpoint (if the state_dict has an
                  attention key, build the fixed model; otherwise the original/buggy one).
         True  -> for the fixed checkpoints we retrained.
         False -> for their downloaded checkpoints (saved from the buggy model).
       seed: for reproducibility, since attention is randomly initialized in the buggy model."""
    cwd = os.getcwd()
    sys.path.insert(0, mexa_dir)
    os.chdir(mexa_dir)
    try:
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        from ctp.dataset import ClinicalTtrialsPredictionDatasetH as CTPDataset
        from ctp.models_nlayers import ClinicalTtrialsPredictionModelH_nlayers as CTPModel
        from utils.utils import random_seed

        args = _load_args(_find_args_json(ckpt_path))
        args.device = device

        # --- detect the checkpoint type / build the model with the SAME type ---
        # weights_only=False is REQUIRED: the default became True in torch>=2.6, but this
        # ckpt carries numpy scalars inside 'stastics' -> it won't open with weights_only=True.
        # (Source is trusted: the MEXA checkpoint we downloaded ourselves. Same on the HINT side.)
        raw_state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
        has_attn_keys = any(k.startswith(('cross_att_layers', 'self_att_layers', 'routers'))
                            for k in raw_state)
        if moduledict_fix is None:
            moduledict_fix = has_attn_keys
        args.moduledict_fix = moduledict_fix
        random_seed(seed)   # make the random attention reproducible in the buggy model
        print(f'[MEXA] checkpoint contains attention keys: {has_attn_keys} '
              f'-> model moduledict_fix={moduledict_fix}')

        max_length = {
            'icds': args.imax_length, 'smiless': args.smax_length,
            'in_criteria': args.in_cmax_length, 'ex_criteria': args.ex_cmax_length,
        }
        reduce = {'icds': args.ireduce, 'criteria': args.creduce}

        # --- is the data present? ---
        for split in ['valid', 'test']:
            sub = 'validphase' if split == 'valid' else 'test'
            probe = glob(f'{data_folder}{sub}/phase_{phase}/*.pkl')
            if len(probe) == 0:
                raise FileNotFoundError(
                    f'[MEXA] {split} data not found: {data_folder}{sub}/phase_{phase}/\n'
                    f'       embedded data must be produced with createDataset.py '
                    f'(is MEXA_DATA_FOLDER set?).')

        model = CTPModel(args, max_length=max_length)
        model.load_state_dict(raw_state, strict=False)
        model.to(device)
        model.eval()

        rows = []
        for split in ['valid', 'test']:
            ds = CTPDataset(data_folder, subset=split, phase=phase,
                            max_length=max_length, reduce=reduce)
            loader = DataLoader(ds, batch_size=args.testsize, shuffle=False, num_workers=1)
            with torch.no_grad():
                for data in loader:
                    g = lambda k: data[k].to(device)
                    out = model(g('itokens'), g('imasks'), g('in_ctokens'), g('in_cmasks'),
                                g('ex_ctokens'), g('ex_cmasks'), g('stokens'), g('smasks'),
                                data['label'].to(device))
                    logits = out['preds'].view(-1).cpu().numpy()
                    scores = 1.0 / (1.0 + np.exp(-logits))        # sigmoid
                    labels = data['label'].view(-1).cpu().numpy()
                    nctids = data['nctid']
                    for nctid, score, label in zip(nctids, scores, labels):
                        rows.append((nctid, float(score), int(label), split))
            print(f'[MEXA] {split}: done')
    finally:
        os.chdir(cwd)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['nctid', 'score', 'label', 'split'])
        w.writerows(rows)
    print(f'[MEXA] written -> {out_csv} ({len(rows)} rows)')
    return out_csv


if __name__ == '__main__':
    import config as C
    dump_mexa(C.PHASE, C.MEXA_CKPT, C.MEXA_DATA_FOLDER, C.MEXA_PREDS, C.MEXA_DIR)
