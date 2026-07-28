# -*- coding: utf-8 -*-
"""
paper/main.py — CONTROL PANEL  (run block by block, piece by piece)
=================================================================
This file is a "# %%" cell script. VS Code / Spyder / PyCharm / Jupyter
see each "# %%" block as a separate cell. Run them ONE BY ONE, by hand,
from top to bottom — nothing flows automatically. There is NO 'if __name__'
at the bottom.

Idea (backbone):
  HINT and MEXA do NOT import each other. Each runs in its own context and
  dumps its raw scores (score+label, val+test) to disk. Then the COMMON
  harness evaluates all scores with the SAME rules -> fair comparison.

Block map:
  0  Config & import
  1  Common harness
  2  HINT: load & dump scores
  3  MEXA: load & dump scores
  6  Step 1  — validation-based epoch selection   (requires retraining)
  7  Step 2  — nn.ModuleDict fix effect            (requires retraining)
  8  Step 3  — threshold calibration              (RUNS with checkpoint)
  9  Step 4  — reproducibility (random_state)     (short check)
  10 Step 5  — PR-AUC fix                          (RUNS with checkpoint)
  11 Results table (fair, single protocol)
  --  Step 6 — Hybrid (future work): HINT-encoder + MEXA mode-expert
"""

# %% ------------------------------------------------------------------
# BLOCK 0 — Config & import
# ---------------------------------------------------------------------
import importlib
import os
import config as C
importlib.reload(C)   # after changing config, just rerun this cell
print('Phase:', C.PHASE, '| seed:', C.SEED, '| selection metric:', C.SELECT_METRIC)
print('HINT ckpt:', C.HINT_CKPT)
print('MEXA ckpt:', C.MEXA_CKPT)


# %% ------------------------------------------------------------------
# BLOCK 1 — Common evaluation harness (single source of truth)
#   Item 3 (threshold@val), Item 5 (PR/ROC on score), bootstrap CI all here.
# ---------------------------------------------------------------------
import eval_harness as H
importlib.reload(H)
print('Harness loaded: threshold@validation + PR/ROC on continuous score + bootstrap CI')


# %% ------------------------------------------------------------------
# BLOCK 2 — HINT: load the ready checkpoint & dump raw scores (val + test)
#   The HINT repo is not modified; only generate_predict is called.
#   Output: preds/preds_hint_phase_{X}.csv
# ---------------------------------------------------------------------
import dump_hint_preds as DH
importlib.reload(DH)
DH.dump_hint(C.PHASE, C.HINT_CKPT, C.HINT_VALID_CSV, C.HINT_TEST_CSV,
             C.HINT_PREDS, C.HINT_DIR)


# %% ------------------------------------------------------------------
# BLOCK 3 — MEXA: load the ready checkpoint & dump raw scores (val + test)
#   CAUTION: MEXA embedded data is required. Without it, it raises a clear error.
#   Output: preds/preds_mexa_phase_{X}.csv
# ---------------------------------------------------------------------
import dump_mexa_preds as DM
importlib.reload(DM)
try:
    DM.dump_mexa(C.PHASE, C.MEXA_CKPT, C.MEXA_DATA_FOLDER, C.MEXA_PREDS, C.MEXA_DIR)
except FileNotFoundError as e:
    print(e)
    print('>> Once the MEXA data is ready, rerun this block.')


# %% ------------------------------------------------------------------
# BLOCK 8 — STEP 3: Threshold calibration  (RUNS with checkpoint)
#   Select the threshold on validation, report test F1 at that threshold.
#   Instead of a fixed 0.5, a real threshold -> F1 measures something meaningful.
# ---------------------------------------------------------------------
hint_res = H.evaluate_model(C.HINT_PREDS, select_metric=C.SELECT_METRIC,
                            n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
print('HINT selected threshold (val):', round(hint_res['threshold'], 4))
print('HINT test F1 :', round(hint_res['point']['f1'], 4),
      '| compare with the old fixed-0.5 history/paper value')


# %% ------------------------------------------------------------------
# BLOCK 10 — STEP 5: PR-AUC fix  (RUNS with checkpoint)
#   HINT's evaluation() computes PR-AUC on binarised 0/1 scores
#   (broken). The harness computes it on the CONTINUOUS score.
# ---------------------------------------------------------------------
print('HINT ROC-AUC (on score):', round(hint_res['point']['roc_auc'], 4))
print('HINT PR-AUC  (on score, FIXED):', round(hint_res['point']['pr_auc'], 4))
print('CI:', hint_res['ci']['pr_auc'])


# %% ------------------------------------------------------------------
# BLOCK 9 — STEP 4: Reproducibility (random_state) — short check
#   MACAW is deterministic with random_state; seed=2023 in build_mexa_data.py.
#   Since the bootstrap seed is fixed here, the CIs are reproducible.
# ---------------------------------------------------------------------
r2 = H.evaluate_model(C.HINT_PREDS, select_metric=C.SELECT_METRIC,
                      n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
same = abs(r2['ci']['roc_auc']['mean'] - hint_res['ci']['roc_auc']['mean']) < 1e-12
print('Same seed -> same CI?', same)


# %% ------------------------------------------------------------------
# BLOCK 11 — RESULTS TABLE (fair, single protocol)
#   The second row fills in once the MEXA data is ready.
# ---------------------------------------------------------------------
results = {'HINT': hint_res}
if os.path.exists(C.MEXA_PREDS):
    results['MEXA'] = H.evaluate_model(C.MEXA_PREDS, select_metric=C.SELECT_METRIC,
                                       n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
# test labels for the TRIVIAL baseline row (needed to read F1 correctly)
_df = H.load_preds(C.HINT_PREDS)
_, _test_labels = H.split_scores(_df, 'test')
table = H.compare_table(results, test_labels=_test_labels)
print(table.to_string(index=False))
table.to_csv(os.path.join(C.PAPER_DIR, f'results_phase_{C.PHASE}.csv'), index=False)


# %% ------------------------------------------------------------------
# BLOCK 12 — PER-TRIAL RESULTS CSV (see each success one by one)
#   Raw metadata (diseases, icdcodes, drugs, smiless, criteria) + prediction +
#   correct? -> results_hint_phase_{X}.csv. Threshold = the one selected on val.
# ---------------------------------------------------------------------
import build_results_csv as BR
importlib.reload(BR)
BR.build_results(C.HINT_PREDS, C.HINT_TEST_CSV,
                 os.path.join(C.PAPER_DIR, f'results_hint_phase_{C.PHASE}.csv'),
                 threshold=hint_res['threshold'], split='test')
# When the MEXA data is ready:
# BR.build_results(C.MEXA_PREDS, <mexa_raw_csv>, f'results_mexa_phase_{C.PHASE}.csv',
#                  threshold=results['MEXA']['threshold'], split='test')


# %% ------------------------------------------------------------------
# BLOCK 6 — STEP 1: Validation-based epoch selection
#   NOTE: This step requires RETRAINING (we did it in mexa/main.py: val-selection +
#   early stopping). Since the only 'rocbest' checkpoint at hand is test-selected,
#   it cannot be reproduced EXACTLY from the checkpoint here.
#   After retraining: take the test score of the val-best epoch from history.csv
#   and pass it through the harness above.
# ---------------------------------------------------------------------
print('STEP 1: retrain with mexa/main.py --patience ..., then fill in this block.')


# %% ------------------------------------------------------------------
# BLOCK 7 — STEP 2: nn.ModuleDict fix effect
#   NOTE: This also requires RETRAINING. Cross-attention was never trained in
#   the ready checkpoint (dict -> optimizer never saw it). Train with vs without
#   the fix and compare the two scores through the harness.
# ---------------------------------------------------------------------
print('STEP 2: train the ModuleDict-fixed model, dump its scores, compare via harness.')


# %% ------------------------------------------------------------------
# STEP 6 — HYBRID (future work)
#   HINT-encoder (understanding) + MEXA mode-expert (interpretation). Precondition:
#   Steps 1-2 must be done and cross-attention shown to actually help.
# ---------------------------------------------------------------------
print('STEP 6 (future work): hybrid — depends on the Step 1-2 results.')
