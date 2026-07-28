# -*- coding: utf-8 -*-
"""
paper/seed_sensitivity.py — Seed sensitivity of the MEXA checkpoint
==================================================================
WHY: MEXA's attention/router layers are in a plain dict, so they were NEVER
written to the checkpoint (Item 2). Therefore every time the checkpoint is loaded
these layers are RE-INITIALIZED RANDOMLY. Result: MEXA's reported score cannot be
reproduced from the checkpoint — every seed means a different model.

This script evaluates the same checkpoint with N different seeds and measures how
much the metrics move. If a wide range comes out, we have PROVEN that the reported
score comes from a random initialization, not from the architecture.

Usage:
    cd paper
    python seed_sensitivity.py            # 5 seeds, config.PHASE
    python seed_sensitivity.py 10 III     # 10 seeds, phase III
"""
import os
import sys
import numpy as np


def run(n_seeds=5, phase=None, seeds=None):
    import config as C
    import eval_harness as H
    import dump_mexa_preds as DM

    phase = phase or C.PHASE
    seeds = seeds or [2023 + i for i in range(n_seeds)]
    ckpt = C._MEXA_CKPTS[phase]

    rows = []
    for sd in seeds:
        out_csv = os.path.join(C.PREDS_DIR, f'_seedtest_mexa_phase_{phase}_seed{sd}.csv')
        DM.dump_mexa(phase, ckpt, C.MEXA_DATA_FOLDER, out_csv, C.MEXA_DIR,
                     moduledict_fix=False, seed=sd)     # their (buggy) checkpoint
        r = H.evaluate_model(out_csv, select_metric=C.SELECT_METRIC,
                             n_boot=200, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
        rows.append({
            'seed': sd,
            'threshold': r['threshold'],
            'roc_auc': r['point']['roc_auc'],
            'pr_auc': r['point']['pr_auc'],
            'f1': r['point']['f1'],
        })
        print(f"  seed {sd}: ROC={r['point']['roc_auc']:.4f}  "
              f"PR={r['point']['pr_auc']:.4f}  F1={r['point']['f1']:.4f}")

    print('\n================ SEED SENSITIVITY ================')
    print(f'phase {phase} · same checkpoint · {len(seeds)} different seeds')
    for m in ['roc_auc', 'pr_auc', 'f1']:
        v = np.array([r[m] for r in rows])
        print(f'  {m:8s} mean={v.mean():.4f}  std={v.std():.4f}  '
              f'min={v.min():.4f}  max={v.max():.4f}  range={v.max()-v.min():.4f}')

    roc = np.array([r['roc_auc'] for r in rows])
    print(f'\n  number of seeds above ROC-AUC 0.5 (chance): '
          f'{(roc > 0.5).sum()}/{len(roc)}')
    print('  -> If the range is wide: the reported score comes from a random '
          'initialization, not the architecture.')

    try:
        import pandas as pd
        p = os.path.join(C.PAPER_DIR, f'seed_sensitivity_phase_{phase}.csv')
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f'\nsaved -> {p}')
    except Exception:
        pass
    return rows


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ph = sys.argv[2] if len(sys.argv) > 2 else None
    run(n, ph)
