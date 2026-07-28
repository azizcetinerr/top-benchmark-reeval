# -*- coding: utf-8 -*-
"""
paper/regenerate_tables.py — Regenerate all result tables in ENGLISH
==========================================================================
The result files must be fully ENGLISH since they go directly into the paper.
The old tables had Turkish labels ('TRIVIAL (hepsi=1...)', 'FIXLI', 'BOZUK',
'yakinsadi', ...). This script regenerates them all from the RAW PREDICTION
files we have — it DOES NOT REQUIRE RETRAINING.

What it regenerates:
  results_phase_{X}.csv          <- preds_hint_phase_{X}.csv
  baselines_phase_{X}.csv      <- preds_{logreg,randomforest,gradboost}_phase_{X}.csv
  hint_variants_phase_{X}.csv  <- preds_hintvar_{VARIANT}_phase_{X}.csv
  ab_test_results_phase_{X}.csv<- only the 'arm' column FIXLI/BOZUK -> FIXED/BUGGY
  runs_summary.csv             <- 'yakinsadi' column -> 'converged' (English values)

Usage:
    cd paper
    python regenerate_tables.py
"""
import os
import glob


VARIANTS = ['Only_Molecule', 'Only_Disease', 'Interaction', 'HINT_nograph', 'HINTModel']
BASELINES = {'LogReg': 'logreg', 'RandomForest': 'randomforest', 'GradBoost': 'gradboost'}


def _table_from_preds(preds_map, C, H):
    """{display_name: preds_csv} -> compare_table (including the TRIVIAL row)."""
    evald, yte = {}, None
    for name, path in preds_map.items():
        if not os.path.exists(path):
            continue
        evald[name] = H.evaluate_model(path, select_metric=C.SELECT_METRIC,
                                       n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA,
                                       seed=C.SEED)
        _, yte = H.split_scores(H.load_preds(path), 'test')
    if not evald:
        return None
    return H.compare_table(evald, test_labels=yte)


def main():
    import pandas as pd
    import config as C
    import eval_harness as H

    P, D = C.PAPER_DIR, C.PREDS_DIR
    done = []

    for ph in ['I', 'II', 'III']:
        # 1) HINT single-model table
        t = _table_from_preds({'HINT': os.path.join(D, f'preds_hint_phase_{ph}.csv')}, C, H)
        if t is not None:
            p = os.path.join(P, f'results_phase_{ph}.csv'); t.to_csv(p, index=False); done.append(p)

        # 2) simple ML baselines
        t = _table_from_preds({k: os.path.join(D, f'preds_{v}_phase_{ph}.csv')
                               for k, v in BASELINES.items()}, C, H)
        if t is not None:
            p = os.path.join(P, f'baselines_phase_{ph}.csv'); t.to_csv(p, index=False); done.append(p)

        # 3) HINT ablation variants
        t = _table_from_preds({v: os.path.join(D, f'preds_hintvar_{v}_phase_{ph}.csv')
                               for v in VARIANTS}, C, H)
        if t is not None:
            p = os.path.join(P, f'hint_variants_phase_{ph}.csv'); t.to_csv(p, index=False); done.append(p)

        # 4) A/B results — only translate the 'arm' labels
        p = os.path.join(P, f'ab_test_results_phase_{ph}.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'arm' in df.columns:
                df['arm'] = df['arm'].replace({'FIXLI': 'FIXED', 'BOZUK': 'BUGGY'})
                df.to_csv(p, index=False); done.append(p)

    # 5) runs_summary — column name + values
    p = os.path.join(P, 'runs_summary.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        if 'yakinsadi' in df.columns:
            df = df.rename(columns={'yakinsadi': 'converged'})
            df['converged'] = df['converged'].replace({
                'evet': 'yes', 'HAYIR (hala tirmaniyor)': 'NO (still improving)'})
            df.to_csv(p, index=False); done.append(p)

    print('Regenerated (English):')
    for p in done:
        print('  ' + os.path.basename(p))

    # verification: any residual Turkish?
    bad = []
    for f in glob.glob(os.path.join(P, '*.csv')):
        txt = open(f, encoding='utf-8', errors='ignore').read()
        if any(w in txt for w in ['hepsi=1', 'rastgele', 'FIXLI', 'BOZUK',
                                  'yakinsadi', 'HAYIR', 'evet']):
            bad.append(os.path.basename(f))
    print('\nRemaining Turkish labels: ' + (', '.join(bad) if bad else 'NONE — all clean'))


if __name__ == '__main__':
    main()
