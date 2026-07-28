# -*- coding: utf-8 -*-
"""
paper/run_hint_phase.py — Run HINT end-to-end for one phase
=================================================================
For phase I / II / III: load the HINT checkpoint, dump the val+test raw scores,
pass them through the common harness, produce the per-trial CSV and print the
fair table.

A one-command version of main.py's Blocks 2 + 8 + 10 + 11 + 12. When adding
phases I and II, you can use this instead of stepping through blocks in a
notebook.

Usage:
    cd paper
    CTP_PHASE=I  python run_hint_phase.py
    CTP_PHASE=II python run_hint_phase.py
"""
import os
import sys


def run(phase=None):
    import importlib
    import config as C
    importlib.reload(C)
    if phase:
        os.environ['CTP_PHASE'] = phase
        importlib.reload(C)

    import eval_harness as H
    import dump_hint_preds as DH
    import build_results_csv as BR

    print(f'=== HINT | phase {C.PHASE} ===')
    print(f'ckpt : {C.HINT_CKPT}')
    if not os.path.exists(C.HINT_CKPT):
        print('!! no checkpoint — HINT should have phase_{X}.ckpt under save_model/.')
        return None

    # 1) dump raw scores
    DH.dump_hint(C.PHASE, C.HINT_CKPT, C.HINT_VALID_CSV, C.HINT_TEST_CSV,
                 C.HINT_PREDS, C.HINT_DIR)

    # 2) common harness (threshold@val, PR/ROC on score, bootstrap CI)
    res = H.evaluate_model(C.HINT_PREDS, select_metric=C.SELECT_METRIC,
                           n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
    df = H.load_preds(C.HINT_PREDS)
    _, yte = H.split_scores(df, 'test')

    print(f"\nselected threshold (validation): {res['threshold']:.4f}")
    print(f"test  ROC={res['point']['roc_auc']:.4f}  "
          f"PR={res['point']['pr_auc']:.4f}  F1={res['point']['f1']:.4f}")

    # 3) Item 5 evidence: compare against the buggy (binarised) PR-AUC
    from sklearn.metrics import average_precision_score, f1_score
    s, y = H.split_scores(df, 'test')
    pr_ok = average_precision_score(y, s)
    pr_bad = average_precision_score(y, (s >= 0.5).astype(int))
    f1_fixed = f1_score(y, (s >= 0.5).astype(int))
    print(f"\nItem 5: PR-AUC correct={pr_ok:.4f}  broken(binary@0.5)={pr_bad:.4f}  "
          f"diff={pr_ok-pr_bad:+.4f}")
    print(f"Item 3: F1 fixed@0.5={f1_fixed:.4f}  threshold@val={res['point']['f1']:.4f}")

    # 4) fair table (including the TRIVIAL row)
    table = H.compare_table({'HINT': res}, test_labels=yte)
    print('\n' + table.to_string(index=False))
    out = os.path.join(C.PAPER_DIR, f'results_phase_{C.PHASE}.csv')
    table.to_csv(out, index=False)

    # 5) per-trial CSV
    BR.build_results(C.HINT_PREDS, C.HINT_TEST_CSV,
                     os.path.join(C.PAPER_DIR, f'results_hint_phase_{C.PHASE}.csv'),
                     threshold=res['threshold'], split='test')
    print(f'\nsaved -> {out}')
    return res


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
