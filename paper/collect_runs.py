# -*- coding: utf-8 -*-
"""
paper/collect_runs.py — Collect all training runs into one table
================================================================
Reads each run's args.json + history.csv under mexa/ab-results/, selects the best
epoch by validation (Item 1 protocol) and reports THAT EPOCH's test score.
Selection is NOT done by test.

Also a 'converged?' column: if the best epoch is the last epoch, the run is
probably still climbing -> it needs longer training.

Usage:
    cd paper
    python collect_runs.py
"""
import os
import json
import sys
from glob import glob


def collect(results_root=None, metric='roc'):
    import pandas as pd
    import config as C

    root = results_root or os.path.join(C.MEXA_DIR, 'ab-results')
    hist_files = sorted(glob(os.path.join(root, '**', 'history.csv'), recursive=True))
    if not hist_files:
        print(f'No run found: {root}')
        return None

    rows = []
    for hf in hist_files:
        run_dir = os.path.dirname(hf)
        aj = os.path.join(run_dir, 'args.json')
        args = json.load(open(aj)) if os.path.exists(aj) else {}
        # only runs of the selected phase (config.PHASE / CTP_PHASE)
        if args.get('phase') != C.PHASE:
            continue
        try:
            h = pd.read_csv(hf)
        except Exception:
            continue
        if len(h) == 0 or f'valid_{metric}' not in h.columns:
            continue

        # --- Item 1: the BEST EPOCH is selected FROM VALIDATION ---
        i = h[f'valid_{metric}'].idxmax()
        best = h.loc[i]
        n_ep = len(h)
        rows.append({
            'fix': args.get('moduledict_fix', '?'),
            'lr': args.get('lr', '?'),
            'phase': args.get('phase', '?'),
            'epochs_run': n_ep,
            'best_epoch': int(best['epoch']),
            f'valid_{metric}': round(float(best[f'valid_{metric}']), 4),
            f'test_{metric}': round(float(best[f'test_{metric}']), 4),
            'test_pr': round(float(best['test_pr']), 4) if 'test_pr' in h.columns else None,
            'test_f1': round(float(best['test_f1']), 4) if 'test_f1' in h.columns else None,
            # if best epoch == last epoch it was probably still improving
            # (output file is ENGLISH)
            'converged': 'NO (still improving)' if int(best['epoch']) == n_ep else 'yes',
            'run': os.path.basename(run_dir),
        })

    df = pd.DataFrame(rows).sort_values(['fix', f'valid_{metric}'], ascending=[False, False])
    print(df.to_string(index=False))

    # --- winner per arm (by validation) ---
    print('\n=== WINNER PER ARM (selected by validation) ===')
    for fix_val, name in [(True, 'FIXED  (cross-attention trained)'),
                          (False, 'BUGGY  (original, attention frozen)')]:
        sub = df[df['fix'] == fix_val]
        if len(sub) == 0:
            continue
        w = sub.iloc[0]
        print(f'  {name}: lr={w["lr"]}  valid_{metric}={w[f"valid_{metric}"]}  '
              f'-> test_{metric}={w[f"test_{metric}"]}  [{w["converged"]}]')

    out = os.path.join(C.PAPER_DIR, 'runs_summary.csv')
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')
    return df


if __name__ == '__main__':
    collect(sys.argv[1] if len(sys.argv) > 1 else None)
