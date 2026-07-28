# -*- coding: utf-8 -*-
"""
paper/eval_runs.py — Evaluate trained runs on test with the COMMON harness
=============================================================================
For each training run it finds the best checkpoint selected on VALIDATION,
dumps the test predictions and passes them through the common harness
(threshold@val, PR/ROC on continuous scores, bootstrap CI). Test is not
involved in any selection decision.

For runs trained with --skip_test_eval the test columns in history.csv are
NaN; the real test scores are produced HERE, from the checkpoint.

Usage:
    cd paper
    python eval_runs.py                  # lr=1e-2, converged runs
    python eval_runs.py --lr 0.01 --min_epochs 15
"""
import os
import re
import json
import argparse
from glob import glob


def find_best_ckpt(run_dir, metric='roc'):
    """When save_checkpoint is best it writes '{metric}best@{epoch}.pth.tar'.
    The 'best' file with the highest epoch = the last best selected on validation."""
    cks = glob(os.path.join(run_dir, 'checkpoints', f'{metric}best@*.pth.tar'))
    if not cks:
        return None, None
    def ep(p):
        m = re.search(r'best@(\d+)\.pth\.tar$', p)
        return int(m.group(1)) if m else -1
    best = max(cks, key=ep)
    return best, ep(best)


def main(lr=None, min_epochs=0, phase=None, n_boot=1000, filters=None):
    """filters: {'rho1':0.05, 'threshold':0.3, ...} — filters runs by the given
    hyperparameters. If None, no filtering (all values appear in the table)."""
    import pandas as pd
    import config as C
    import eval_harness as H
    import dump_mexa_preds as DM

    phase = phase or C.PHASE
    filters = filters or {}
    root = os.path.join(C.MEXA_DIR, 'ab-results')
    runs = sorted(glob(os.path.join(root, '**', 'args.json'), recursive=True))

    rows = []
    for aj in runs:
        run_dir = os.path.dirname(aj)
        args = json.load(open(aj))
        if args.get('phase') != phase:
            continue
        if lr is not None and abs(float(args.get('lr', -1)) - lr) > 1e-12:
            continue
        if any(abs(float(args.get(k, -1e9)) - v) > 1e-12 for k, v in filters.items()):
            continue
        hist = os.path.join(run_dir, 'history.csv')
        if not os.path.exists(hist):
            continue
        if len(pd.read_csv(hist)) < min_epochs:
            continue

        ck, ep = find_best_ckpt(run_dir)
        if ck is None:
            print(f'[skip] no best checkpoint: {os.path.basename(run_dir)}')
            continue

        fix = args.get('moduledict_fix', None)
        tag = 'FIXED' if fix else 'BUGGY'   # output ENGLISH
        # CAUTION: run folder names can be the SAME across phases
        # (e.g. rho10.05_rho20.01_id4 in both phase I and phase III). If the
        # phase and arm name don't enter the file name, the phase I output
        # overwrites the phase III one.
        out_csv = os.path.join(
            C.PREDS_DIR,
            f'_run_ph{phase}_{"fix" if fix else "bug"}_s{args.get("seed")}'
            f'_{os.path.basename(run_dir)}_ep{ep}.csv')
        print(f'\n--- {tag}  seed={args.get("seed")}  lr={args.get("lr")}  '
              f'best_epoch={ep}  ({os.path.basename(run_dir)})')
        try:
            DM.dump_mexa(phase, ck, C.MEXA_DATA_FOLDER, out_csv, C.MEXA_DIR,
                         moduledict_fix=None, seed=args.get('seed', 2023))
        except Exception as e:
            print(f'  [error] {e}')
            continue

        r = H.evaluate_model(out_csv, select_metric=C.SELECT_METRIC,
                             n_boot=n_boot, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
        rows.append({
            'arm': tag, 'seed': args.get('seed'), 'lr': args.get('lr'),
            # hyperparameter columns — for grouping in the rho/threshold sweep
            'rho1': args.get('rho1'), 'rho2': args.get('rho2'),
            'thr': args.get('threshold'),
            'best_epoch': ep,
            'test_roc': round(r['point']['roc_auc'], 4),
            'test_pr': round(r['point']['pr_auc'], 4),
            'test_f1': round(r['point']['f1'], 4),
            'threshold': round(r['threshold'], 4),
            'roc_lo': round(r['ci']['roc_auc']['lo'], 4),
            'roc_hi': round(r['ci']['roc_auc']['hi'], 4),
        })
        print(f"  test ROC={rows[-1]['test_roc']}  PR={rows[-1]['test_pr']}  "
              f"F1={rows[-1]['test_f1']}")

    if not rows:
        print('\nNo run found to evaluate.')
        return None

    df = pd.DataFrame(rows).sort_values(['arm', 'seed'])
    print('\n================ PER-RUN TEST RESULTS ================')
    print(df.to_string(index=False))

    # --- ARM SUMMARY: GROUP BY CONFIGURATION ---
    # CAUTION: averaging runs with different (rho1,rho2,threshold) into a single
    # pool produces a misleading "arm average".
    # So the summary is given per configuration and the comparison is done only
    # over configurations where both arms are present.
    df['cfg'] = df.apply(lambda r: f"rho1={r.rho1} rho2={r.rho2} thr={r.thr}", axis=1)
    print('\n================ ARM SUMMARY (per configuration) ================')
    for cfg, g in df.groupby('cfg'):
        arms = sorted(g.arm.unique())
        print(f'  [{cfg}]')
        for arm in ['FIXED', 'BUGGY']:
            s = g[g.arm == arm]
            if len(s) == 0:
                continue
            sd = f'±{s.test_roc.std(ddof=1):.4f}' if len(s) > 1 else ''
            print(f'     {arm:6s} n={len(s)}  ROC={s.test_roc.mean():.4f}{sd}  '
                  f'PR={s.test_pr.mean():.4f}  F1={s.test_f1.mean():.4f}')
        if len(arms) < 2:
            print('     (single arm — comparison not possible)')

    # statistical comparison: ONLY configs where both arms are present
    both = [c for c, g in df.groupby('cfg') if g.arm.nunique() == 2]
    if not both:
        print('\n  !! No config with both arms present — comparison skipped.')
        out = os.path.join(C.PAPER_DIR, f'ab_test_results_phase_{phase}.csv')
        df.drop(columns=['cfg']).to_csv(out, index=False)
        print(f'\nsaved -> {out}')
        return df
    sub = df[df.cfg.isin(both)]
    print(f'\n  comparison over config(s): {both}')
    f, b = sub[sub.arm == 'FIXED'], sub[sub.arm == 'BUGGY']
    if len(f) > 1 and len(b) > 1:
        from scipy import stats as st
        t, p = st.ttest_ind(b.test_roc, f.test_roc, equal_var=False)
        print(f'\n  test ROC diff (buggy-fixed) = {b.test_roc.mean()-f.test_roc.mean():+.4f}'
              f'   Welch p={p:.3f}')
        print('  Reference: HINT test ROC=0.6851 | TRIVIAL=0.500')

    out = os.path.join(C.PAPER_DIR, f'ab_test_results_phase_{phase}.csv')
    df.drop(columns=['cfg']).to_csv(out, index=False)
    print(f'\nsaved -> {out}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, default=0.01,
                    help='only this lr; use --lr -1 for all')
    ap.add_argument('--min_epochs', type=int, default=15)
    ap.add_argument('--phase', default=None)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--rho1', type=float, default=None)
    ap.add_argument('--rho2', type=float, default=None)
    ap.add_argument('--threshold', type=float, default=None)
    a = ap.parse_args()
    filt = {k: v for k, v in
            [('rho1', a.rho1), ('rho2', a.rho2), ('threshold', a.threshold)]
            if v is not None}
    main(None if a.lr is not None and a.lr < 0 else a.lr,
         a.min_epochs, a.phase, a.n_boot, filt)
