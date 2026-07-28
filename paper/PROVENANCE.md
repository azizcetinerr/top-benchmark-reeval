# Provenance — every number to its source file

Rule: no sentence in the paper states a number that is not traceable to a row in one of
these files. Every number is derived from raw per-trial predictions; none is hand-entered.

The paper keeps its tables **inline in the `.tex`** — no derived `table_*.csv` files are
stored. Reproduce and check everything with:

```bash
cd paper
python regenerate_tables.py        # rebuilds results_/baselines_/hint_variants_ from preds/
python verify_paper_tables.py      # recomputes tab:ab/reload/f1/praucbug from raw and
                                   # prints computed-vs-typeset side by side
```

`preds/` holds raw per-trial predictions (`nctid, score, label, split`) and is the ground
truth for everything. Every table is derived; none is hand-entered. Table numbers were last
verified against raw data with `verify_paper_tables.py` (all match; the only residuals are a
sample-vs-population sd convention and two validation-calibrated F1 cells that differ by
≤0.0013 from threshold tie-breaking).

---

## Tables to typeset (all inline in the .tex)

| Tag | Content | Take values from | Status |
|---|---|---|---|
| **T1** | Defect list, with before/after | §3 text + `preds/preds_hintvar_*_phase_*.csv` (PR-AUC), `preds/preds_hint_phase_*.csv` (F1) | ready |
| **T2** | Fair comparison, 3 phases | `results_phase_{I,II,III}.csv` + `ab_test_results_phase_{I,II,III}.csv` (MEXA arms) + `seed_sensitivity_phase_{I,II,III}.csv` (released) | ready |
| **T3** | HINT ablation ladder | `hint_variants_phase_{I,II,III}.csv` | ready |
| **T4** | Classical baselines | `baselines_phase_{I,II,III}.csv` | ready |
| **T5** | ICD representation ablation | `icd_representation_phase_{I,II,III}.csv` | ready |
| **T6** | A/B: trained vs untrained cross-attention | `ab_test_results_phase_{I,II,III}.csv` (dedup on arm,seed,cfg; paired on seed) | ready |
| **T7** | F1 under four thresholding rules | `preds/preds_hint_phase_{I,II,III}.csv` (F1 at each rule) | ready |
| **T8** | PR-AUC: correct vs binarised | `preds/preds_hintvar_*_phase_{I,II,III}.csv` (continuous vs binarised@0.5) | ready |
| **T9** | Ranking distortion caused by the PR-AUC bug | `preds/preds_hintvar_*_phase_{I,II,III}.csv` (Spearman ρ of the two PR-AUC vectors) | ready |
| **T10** | Dataset splits and class balance | raw HINT `data/phase_*_{train,valid,test}.csv` (train counts) + `preds/` (valid/test) | ready (inline) |
| **T11** | The five dissolved claims | §5 — narrative + `ab_test_results_*`, `preds/preds_hintvar_*`, `icd_representation_*`, `tabular_grid_*`, `optionB_phaseIII_multiplecomparison.csv` | ready |
| **T12** | Prediction-level fusion (Option B) | `../tabfm_baseline_optionB/optionB_{stack,zavg,rank,nnls}_phase_III.csv` | ready |
| **T13** | Fusion vs best single + multiple-comparison correction | `../tabfm_baseline_optionB/optionB_phaseIII_multiplecomparison.csv` | ready |

## Figures to draw

| Tag | Content | Plot from | Status |
|---|---|---|---|
| **F1** | Forest plot: all models × 3 phases, ROC + 95% CI, with trivial baseline as vertical line | `results_phase_*.csv`, `baselines_phase_*.csv`, `hint_variants_phase_*.csv` (CI columns already present as `"0.685 [0.647, 0.720]"` — parse) | needs plotting script |
| **F2** | Two-bugs-concealing-each-other chain | diagram, hand-drawn — no data | conceptual |
| **F3** | Released MEXA checkpoint across 5 seeds vs the claimed gain | `seed_sensitivity_phase_*.csv` + `preds/_seedtest_mexa_phase_*_seed*.csv` | needs plotting script |
| **F4** | Scatter: PR-AUC correct (x) vs binarised (y), 15 points, identity line | `preds/preds_hintvar_*_phase_*.csv` | needs plotting script |
| **F5** | Paired dot plot: BUGGY vs FIXED per seed | `ab_test_results_phase_III.csv` (dedup on `arm,seed,cfg` first) | needs plotting script |

---

## Claim → source, sentence by sentence

### Abstract / §1
| Claim | Source |
|---|---|
| MEXA claims 11.3% F1 / 12.2% PR-AUC / 2.5% ROC-AUC | MEXA-CTP paper, Contributions bullet + Conclusion (verbatim) |
| Per-phase ranges 5.3–19.2 / 4.1–27.9 / 1.1–3.5% | MEXA-CTP paper §4.2 |
| HINT 0.685 vs MEXA 0.674 (phase III) | `results_phase_III.csv`; `ab_test_results_phase_III.csv` (arm means) |
| Cross-attention Δ = 0.0004, p = 0.96, 7 seeds | `ab_test_results_phase_III.csv` (dedup, paired on seed) |
| PR-AUC understated by 0.05–0.06 | `preds/preds_hintvar_*_phase_*.csv` (continuous − binarised) |
| Spearman ρ as low as 0.30 | `preds/preds_hintvar_*_phase_I.csv` (rank of the two PR-AUC vectors) |
| Phase-III fusion gain +0.034 over TabFM, not over the best single model | `../tabfm_baseline_optionB/optionB_*_phase_III.csv`, `optionB_phaseIII_multiplecomparison.csv` |

### §3 Defects
| Claim | Source |
|---|---|
| Attention/routers absent from released checkpoint | `table_checkpoint_keys.csv`, `table_param_registration.csv`, `g1_console.txt` |
| PR-AUC understatement, all 15 variant×phase cells | `preds/preds_hintvar_*_phase_*.csv` |
| Calibration changes F1 by +0.119 on phase I | `preds/preds_hint_phase_I.csv` (0.5934 → 0.7125) |
| lr=5e-2 diverges once attention trains | `runs_summary.csv` col `converged` |
| NaN onset @ 9/28/184; below-threshold 13%→100% | `table_masking_collapse.csv`, `table_masking_summary.csv`, `g2_console.txt` |

### §4 Results
| Claim | Source |
|---|---|
| §4.1 HINT row (ROC/PR/F1 + CIs) | `results_phase_{I,II,III}.csv` |
| §4.1 MEXA retrained arms | `ab_test_results_phase_{I,II,III}.csv` (dedup, arm means) |
| §4.1 MEXA released = 0.518 | `seed_sensitivity_phase_III.csv` (seed 2023) |
| §4.1 trivial baseline row | `results_phase_*.csv` last row (`TRIVIAL (all-positive, pos=…)`) |
| Released ckpt spans 0.478–0.534, range 0.055, sd 0.023; F1@0.5 0.380–0.857 | `seed_sensitivity_phase_III.csv` (ROC/sd) + `preds/_seedtest_mexa_phase_III_seed*.csv` (F1@0.5) |
| §4.2 all Δ and p values | `ab_test_results_phase_*.csv` |
| §4.3 ablation ladder | `hint_variants_phase_*.csv` |
| §4.3 drug branch near chance (0.540) | `hint_variants_phase_III.csv` row `Only_Molecule` |
| §4.4 HINT vs GradBoost gaps | `results_phase_*.csv` minus `baselines_phase_*.csv` row `GradBoost` |
| §4.4 validation-selected ICD pipelines | `icd_representation_phase_*.csv` |
| §4.5 four thresholding rules | `preds/preds_hint_phase_*.csv` |
| §4.6 metric-weakness ordering | MEXA paper (claims) + `preds/preds_hintvar_*` + `preds/preds_hint_*` |
| §4.7 fusion does not exceed best single | `../tabfm_baseline_optionB/optionB_*_phase_*.csv`, `optionB_phaseIII_multiplecomparison.csv` |

### §5 Dissolved claims
| Claim | Source |
|---|---|
| Claim 2: 3-seed p ≈ 0.056 → 7-seed p = 0.964 | current value from `ab_test_results_phase_III.csv` (paired); the *earlier* 3-seed value is historical — recompute by subsetting to seeds 2023–2025 |
| Claim 3: ranking direction inconsistent across phases | `preds/preds_hintvar_*_phase_*.csv` — `best_correct` vs `best_binarised` **disagree only on phase III** |
| Claim 4: validation- vs test-selected ICD pipeline | `icd_representation_phase_*.csv` (val-selected) vs `icd_representation_raw_phase_*.csv` (per-seed raw) |
| Claim 5: blending gives a medium gain → no gain over best single, survives no correction | `../tabfm_baseline_optionB/optionB_phaseIII_multiplecomparison.csv` |
| MEXA arm means drifted 0.681→0.674 | `ab_test_results_phase_III.csv` vs earlier drafts |
| "4.9%" misquote | MEXA-CTP paper — figure absent |

---

## ⚠ Gaps — all resolved

| # | Claim in paper | Status |
|---|---|---|
| ~~G1~~ | "0/264 mechanism params in optimizer"; checkpoints have 48–49 keys, none attention | RESOLVED — `dump_checkpoint_keys.py` → `table_checkpoint_keys.csv`, `table_param_registration.csv`, `g1_console.txt` |
| ~~G2~~ | lr-scaled NaN onset (9/28/184) and guard-doesn't-save-collapse | RESOLVED — `masking_collapse.py` → `table_masking_collapse.csv`, `table_masking_summary.csv`, `g2_console.txt` |
| ~~G3~~ | phase III: 3094/344/1146 train/valid/test | RESOLVED — train counts read from raw HINT `data/phase_*_train.csv`; typeset inline in tab:splits |
| ~~G4/G5~~ | reload variance for all three phases (5 seeds each) | RESOLVED — `seed_sensitivity.py` → `seed_sensitivity_phase_{I,II,III}.csv` + `preds/_seedtest_mexa_*`; tab:reload |
| ~~G6~~ | orphan `preds_tabfm_phase_I.csv` | RESOLVED — now used by the TabFM baseline (tab:tabfm) and Option B fusion |

---

## Known irregularities to disclose, not hide

| # | Issue | File | How to handle |
|---|---|---|---|
| N1 | `ab_test_results_phase_III.csv` contains **duplicate rows** — seed 2023 at `rho1=0.05, rho2=0.01, thr=0.3` appears 4× | `ab_test_results_phase_III.csv` | always `drop_duplicates(['arm','seed','cfg'])` before averaging; `verify_paper_tables.py` does |
| N2 | Phase I A/B has only **2 matched pairs** after dedup | `ab_test_results_phase_I.csv` | report as uninformative, not as weak evidence |
| N3 | `HINTModel` = 0.708 in `hint_variants_phase_III.csv` but HINT = 0.685 in `results_phase_III.csv` — different runs of the same architecture | both | disclose in the caption; it is evidence for the noise-floor thesis |
| N4 | Phase I: calibrated F1 (0.7125) is *equal to* the trivial baseline, and the test-oracle sweep returns 0.7119 — slightly **below** it, because the sweep grid excludes thresholds under the minimum score | `preds/preds_hint_phase_I.csv` | say plainly: on phase I the F1-optimal decision rule *is* "predict all positive" |
| N5 | Released MEXA F1@0.5 swings 0.380 → 0.857 across seeds | `preds/_seedtest_mexa_phase_III_seed*.csv` | stronger evidence than the ROC spread; worth its own sentence |
| N6 | Phase-III fusion beats TabFM (+0.034 ROC) but **not** the best single model (HINT); no contrast survives BH-FDR/Bonferroni over the 24-contrast family | `../tabfm_baseline_optionB/optionB_phaseIII_multiplecomparison.csv` | disclose both framings; the gain is over the weaker member, not the best model |
