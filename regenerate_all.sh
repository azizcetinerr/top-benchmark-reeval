#!/usr/bin/env bash
# regenerate_all.sh — reproduce every number in the paper from stored per-trial predictions.
#
# Default run is CPU-only and needs no GPU, no TabFM, no network: it regenerates every table
# and statistic in the paper from the released preds/ CSVs and the benchmark data CSVs.
# Pass --full to additionally run the long / GPU-dependent jobs (TabFM, HINT retrain, D5 fix).
#
# Usage:
#   bash regenerate_all.sh            # fast, CPU-only, ~2-5 min
#   bash regenerate_all.sh --full     # adds GPU/long jobs (needs the tabfm jax[cuda12] env)
#
# Output: everything is teed to regenerate_all_<date>.log next to this script.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="$ROOT/paper"
A="$ROOT/tabfm_baseline_optionA"
LOG="$ROOT/regenerate_all_$(date +%Y%m%d_%H%M%S).log"
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1

run() {  # run <label> <dir> -- <cmd...>
  local label="$1" dir="$2"; shift 3
  echo -e "\n\n########## $label ##########" | tee -a "$LOG"
  ( cd "$dir" && "$@" ) 2>&1 | tee -a "$LOG"
  echo "---- exit ${PIPESTATUS[0]} : $label ----" | tee -a "$LOG"
}

echo "regenerate_all.sh  root=$ROOT  full=$FULL  $(date)" | tee "$LOG"
PY=python3

# ---- 1. Table cross-check: every typeset number vs. raw predictions -------------------------
run "verify_paper_tables (all tables vs raw data)" "$PAPER" -- $PY verify_paper_tables.py

# ---- 2. Statistical rigor: DeLong dAUC, TOST, MDE, balanced-acc/MCC --------------------------
run "stats_rigor (DeLong / TOST / MDE / MCC)" "$PAPER" -- $PY stats_rigor.py

# ---- 3. Calibration: Brier + equal-mass ECE with bootstrap CIs -------------------------------
run "calibration (Brier + ECE, tab:calib)" "$PAPER" -- $PY calibration.py

# ---- 4. Decision-curve analysis (net benefit, fig:dca) --------------------------------------
run "dca (net benefit)" "$PAPER" -- $PY dca.py

# ---- 5. Data-quality audits: temporal split, recurrence, SMILES collision, base-rate lookup --
for ph in I II III; do
  run "data_audit phase $ph (Q2/Q3/Q4 + base-rate lookups)" "$PAPER" -- $PY data_audit.py --phase "$ph"
done

# ---- 6. GRAM-500 cap question, CPU classical-GBM proxy (no GPU / no TabFM) -------------------
run "tabfm_gram500 --proxy (500-cap is not the bottleneck)" "$A" -- $PY tabfm_gram500.py --phase III --proxy

# ---- 7. Tuned gradient-boosting baselines (uses stored feature tables; CPU) ------------------
run "gbdt_baselines phase III (XGB/LGBM/CatBoost)" "$A" -- $PY gbdt_baselines.py --phase III

echo -e "\n\n===== FAST (CPU) REGENERATION COMPLETE. Log: $LOG =====" | tee -a "$LOG"

if [[ "$FULL" -eq 0 ]]; then
  cat <<EOF | tee -a "$LOG"

Skipped the long / GPU-dependent jobs. Re-run with '--full' on a working GPU node to add them:
  * TabFM 5-seed baseline           (tabfm_baseline_optionA/tabfm_seeds.py)         [GPU]
  * TabFM single-member determinism (tabfm_baseline_optionA/tabfm_single_member.py) [GPU]
  * OWED: D5 fix + >=5-seed HINT     (paper/quantify_d5.py ; see D5_QUANTIFY.md)      [GPU, long]
These do not change any number already in the paper; the first two confirm TabFM's seed sd,
and the third is the one measurement the paper flags as still owed.
EOF
  exit 0
fi

# ---- FULL: long / GPU-dependent jobs (need the tabfm jax[cuda12] env on g122/g124) ----------
run "tabfm_seeds phase III (5-seed TabFM sd)"        "$A"     -- $PY tabfm_seeds.py --phase III
run "tabfm_single_member (determinism check)"        "$A"     -- $PY tabfm_single_member.py --phase III
run "quantify_d5 (D5 fix + HINT re-run, THE owed one)" "$PAPER" -- $PY quantify_d5.py --phase III

echo -e "\n\n===== FULL REGENERATION COMPLETE. Log: $LOG =====" | tee -a "$LOG"
