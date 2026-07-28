#!/bin/bash
# =====================================================================
# run_all.sh — run the full exploration matrix (E1-E5) in order
# =====================================================================
# Environment: conda activate tabfm  (rdkit + jax[cuda12] + sklearn + pandas)
# For GPU steps, FIRST:  salloc --gres=gpu:1 -c 8 --mem=64G --time=03:00:00
#
# Usage:
#   bash run_all.sh cpu     # CPU-only steps (grid, disease breakdown, fusion)
#   bash run_all.sh gpu     # GPU-only steps (TabFM multi-seed)
#   bash run_all.sh all     # all
# ---------------------------------------------------------------------
set -e
cd "$(dirname "$0")"
MODE="${1:-all}"

run_cpu () {
  echo "########## E1+E2: tabular model x feature grid ##########"
  for ph in I II III; do
    python tabular_grid.py --phase $ph            # + --full-rdkit if desired (stable env)
  done
  echo "########## E5: disease-area breakdown ##########"
  python stratified_disease.py --phase III --models hint,tabfm,mexa
  python stratified_disease.py --phase II  --models hint,tabfm
  python stratified_disease.py --phase I   --models hint,tabfm
  echo "########## E3: fusion-method sensitivity ##########"
  for m in stack zavg rank; do
    python ensemble_compare.py --phase III --method $m
  done
  python ensemble_compare.py --phase I --method stack
  python ensemble_compare.py --phase II --method stack
}

run_gpu () {
  echo "########## E4: TabFM multi-seed noise floor (GPU) ##########"
  for ph in III II I; do
    python tabfm_seeds.py --phase $ph --seeds 0,1,2,3,4
  done
}

case "$MODE" in
  cpu) run_cpu ;;
  gpu) run_gpu ;;
  all) run_cpu; run_gpu ;;
  *) echo "usage: bash run_all.sh [cpu|gpu|all]"; exit 1 ;;
esac
echo "=== done ($MODE) ==="
