# -*- coding: utf-8 -*-
"""
paper/config.py — Single place for settings (core of Block 0)
========================================================
All paths, phase and protocol settings for FAIRLY comparing HINT and MEXA are
here. main.py imports this file.

Backbone idea:
  Each repo runs in its OWN context and dumps its raw scores (score+label,
  val+test) to disk. Then a single common harness (eval_harness.py) evaluates
  all scores with the SAME rules. The repos do NOT import each other.
"""
import os

# --- Repo roots (relative to this file) ---
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(PAPER_DIR)          # .../ctp
# The repository contains benchmark inputs under ROOT/hint, not a vendored full
# HINT/MEXA source tree. Point these variables at patched upstream clones for
# optional checkpoint inference or retraining.
HINT_DIR  = os.path.abspath(os.environ.get('CTP_HINT_DIR', os.path.join(ROOT, 'hint')))
MEXA_DIR  = os.path.abspath(os.environ.get('CTP_MEXA_DIR', os.path.join(ROOT, 'mexa')))

# Folder where raw scores are dumped: {nctid, score, label, split}
PREDS_DIR = os.path.join(PAPER_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

# --- Experiment settings ---
# You can change PHASE via an environment variable without editing this file:
#     export CTP_PHASE=I     (or II / III)
PHASE = os.environ.get('CTP_PHASE', 'III')     # 'I' | 'II' | 'III'
SEED  = int(os.environ.get('CTP_SEED', 2023))  # Item 4: reproducibility

# --- Common evaluation protocol (SAME for both models) ---
SELECT_METRIC   = 'f1'        # Item 3: the threshold is selected on validation by this metric
BOOTSTRAP_N     = 1000        # number of bootstrap repetitions
BOOTSTRAP_ALPHA = 0.05        # 95% confidence interval

# --- HINT side ---
HINT_CKPT = os.path.join(HINT_DIR, 'save_model', f'phase_{PHASE}.ckpt')
HINT_VALID_CSV = os.path.join(HINT_DIR, 'data', f'phase_{PHASE}_valid.csv')
HINT_TEST_CSV  = os.path.join(HINT_DIR, 'data', f'phase_{PHASE}_test.csv')

# --- MEXA side ---
# MEXA embedded data (output of createDataset.py). Without it, the dump won't run.
MEXA_DATA_FOLDER = os.environ.get('MEXA_DATA_FOLDER', os.path.join(MEXA_DIR, 'DataPath') + os.sep)
_MEXA_CKPTS = {
    'I':   os.path.join(MEXA_DIR, 'checkpoints', 'phaseI',   'rocbest@26.pth.tar'),
    'II':  os.path.join(MEXA_DIR, 'checkpoints', 'phaseII',  'model@16.pth.tar'),
    'III': os.path.join(MEXA_DIR, 'checkpoints', 'phaseIII', 'rocbest.pth.tar'),
}
MEXA_CKPT = _MEXA_CKPTS[PHASE]

# --- Dumped score files ---
def preds_path(model_name, phase=None):
    phase = phase or PHASE
    return os.path.join(PREDS_DIR, f'preds_{model_name}_phase_{phase}.csv')

HINT_PREDS = preds_path('hint')
MEXA_PREDS = preds_path('mexa')
