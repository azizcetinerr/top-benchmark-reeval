# -*- coding: utf-8 -*-
"""
paper/check_mexa_data.py — VALIDATE the produced MEXA data
===========================================================
When build_mexa_data.py finishes a phase, this reads that phase's .pkl files with
MEXA's OWN dataset class, checks the schema is correct and runs a forward pass.
Goal: catch a schema error before the WHOLE build finishes.

Usage (while the build continues in another terminal):
    cd paper
    python check_mexa_data.py            # default: phase I
    python check_mexa_data.py II         # another phase
"""
import os
import sys


def check(phase='I', device='cpu', n_show=2):
    import config as C
    sys.path.insert(0, C.MEXA_DIR)
    cwd = os.getcwd()
    os.chdir(C.MEXA_DIR)
    try:
        import json, argparse, torch
        from glob import glob
        from torch.utils.data import DataLoader
        from ctp.dataset import ClinicalTtrialsPredictionDatasetH as CTPDataset
        from ctp.models_nlayers import ClinicalTtrialsPredictionModelH_nlayers as CTPModel

        data_folder = C.MEXA_DATA_FOLDER
        print(f'DataPath: {data_folder}')

        # --- 1) are the files in place? ---
        counts = {}
        for split, sub in [('train', 'trainphase'), ('valid', 'validphase'), ('test', 'test')]:
            files = glob(f'{data_folder}{sub}/phase_{phase}/*.pkl')
            counts[split] = len(files)
            print(f'  {split:6s}: {len(files):5d} pkl')
        if min(counts.values()) == 0:
            print('!! Some split is empty — the build may not have finished this phase.')
            return

        # --- 2) get model settings from args.json ---
        args_path = os.path.join(os.path.dirname(C._MEXA_CKPTS[phase]), 'args.json')
        args = argparse.Namespace(**json.load(open(args_path)))
        args.device = device
        max_length = {'icds': args.imax_length, 'smiless': args.smax_length,
                      'in_criteria': args.in_cmax_length, 'ex_criteria': args.ex_cmax_length}
        reduce = {'icds': args.ireduce, 'criteria': args.creduce}

        # --- 3) does the dataset read, are the tensor shapes correct? ---
        ds = CTPDataset(data_folder, subset='valid', phase=phase,
                        max_length=max_length, reduce=reduce)
        print(f'\nvalid dataset length: {len(ds)}')
        expected = {
            'itokens':    (max_length['icds'],        args.itoken_size),
            'stokens':    (max_length['smiless'],     args.stoken_size),
            'in_ctokens': (max_length['in_criteria'], args.ctoken_size),
            'ex_ctokens': (max_length['ex_criteria'], args.ctoken_size),
        }
        ok = True
        for i in range(min(n_show, len(ds))):
            d = ds[i]
            print(f'\n  sample {i} (nctid={d["nctid"]}, label={int(d["label"])})')
            for k, exp in expected.items():
                got = tuple(d[k].shape)
                flag = 'OK' if got == exp else f'EXPECTED {exp}'
                if got != exp:
                    ok = False
                print(f'    {k:11s} {str(got):12s} {flag}')
            for mk in ['imasks', 'smasks', 'in_cmasks', 'ex_cmasks']:
                assert d[mk].dtype == torch.bool, f'{mk} is not bool!'
            print('    masks bool: OK')

        if not ok:
            print('\n!! SCHEMA MISMATCH — check the output sizes in build_mexa_data.py.')
            return

        # --- 4) real forward pass (end-to-end compatibility) ---
        loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False)
        batch = next(iter(loader))
        model = CTPModel(args, max_length=max_length).to(device)
        model.eval()
        with torch.no_grad():
            out = model(batch['itokens'].to(device), batch['imasks'].to(device),
                        batch['in_ctokens'].to(device), batch['in_cmasks'].to(device),
                        batch['ex_ctokens'].to(device), batch['ex_cmasks'].to(device),
                        batch['stokens'].to(device), batch['smasks'].to(device),
                        batch['label'].to(device))
        print(f'\nforward OK — preds {tuple(out["preds"].shape)}, '
              f'loss {float(out["loss"]):.4f}')
        print('\nRESULT: phase %s data is COMPATIBLE with the MEXA dataset + model.' % phase)
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    check(sys.argv[1] if len(sys.argv) > 1 else 'I')
