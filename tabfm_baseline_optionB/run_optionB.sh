#!/bin/bash
# =====================================================================
# run_optionB.sh — Option B (prediction-level blending) pipeline
# =====================================================================
# Runs on the g124 compute node (A6000 48GB). Mirrors Option A's
# run_all.sh convention.
#
# Split of work:
#   GPU steps  -> (re)generate the TabFM member scores (jax[cuda12]).
#                 Only TabFM needs a GPU. HINT/MEXA scores are read from
#                 ../paper/preds and are NOT regenerated here.
#   CPU steps  -> the blend itself (option_B_blend.py) is pure CSV +
#                 scikit-learn; no GPU required.
#
# Environment:  conda activate tabfm   (jax[cuda12] + rdkit + sklearn + pandas)
# For GPU steps, first grab the node:
#   salloc --nodelist=g124 --gres=gpu:1 -c 8 --mem=64G --time=02:00:00
#
# Usage:
#   bash run_optionB.sh cpu     # blend only (needs member preds present)
#   bash run_optionB.sh gpu     # regenerate TabFM member preds only
#   bash run_optionB.sh all     # gpu then cpu
# ---------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-all}"

HERE="$(pwd)"
SECENEK_A="$(cd .. && pwd)/tabfm_baseline_optionA"
PAPER_PREDS="$(cd .. && pwd)/paper/preds"
PHASES=(I II III)

# ---------------------------------------------------------------------
# Preflight: which member score files are present?
# ---------------------------------------------------------------------
preflight () {
  echo "########## Preflight: member prediction files ##########"
  for ph in "${PHASES[@]}"; do
    for f in "$SECENEK_A/preds/preds_tabfm_phase_${ph}.csv" \
             "$PAPER_PREDS/preds_hint_phase_${ph}.csv" \
             "$PAPER_PREDS/preds_mexa_phase_${ph}.csv"; do
      if [ -f "$f" ]; then echo "  [ok]   $(basename "$f")"
      else echo "  [MISS] $(basename "$f")  (phase $ph)"; fi
    done
  done
  echo "Note: MEXA exists for phase III only; I/II blends fall back to HINT+TabFM."
}

# ---------------------------------------------------------------------
# GPU: regenerate the TabFM member scores for every phase.
# run_optionA.py reads ../hint/data and writes optionA/preds/preds_tabfm_phase_X.csv
# ---------------------------------------------------------------------
run_gpu () {
  echo "########## GPU: (re)generate TabFM member scores ##########"
  python - <<'PY'
import jax
print("JAX:", jax.__version__, "| devices:", jax.devices())
if not any(d.platform in ("gpu", "cuda") for d in jax.devices()):
    raise SystemExit("ERROR: JAX sees no GPU. Is jax[cuda12] installed? Aborting.")
PY
  for ph in "${PHASES[@]}"; do
    echo "----- TabFM phase $ph -----"
    python "$SECENEK_A/run_optionA.py" --phase "$ph"
  done
}

# ---------------------------------------------------------------------
# CPU: the Option B blend for all phases and all fusion methods.
# ---------------------------------------------------------------------
run_cpu () {
  echo "########## CPU: Option B blend (all phases, all methods) ##########"
  # all methods = stack, zavg, rank, nnls
  python option_B_blend.py --phase all --method all --n-boot 1000
  echo "----- Outputs written to $HERE -----"
  ls -1 "$HERE"/optionB_*.csv 2>/dev/null || true
}

preflight
case "$MODE" in
  cpu) run_cpu ;;
  gpu) run_gpu ;;
  all) run_gpu; run_cpu ;;
  *)   echo "usage: bash run_optionB.sh [cpu|gpu|all]"; exit 1 ;;
esac
echo "=== done ($MODE) ==="
