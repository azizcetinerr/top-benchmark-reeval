# Project backbone — Is Progress on the TOP Benchmark Measurable?

A single-page map of the whole study: the argument, the evidence, every experiment with its
number and script, the review-response log, and what is done vs. owed. All numbers are phase-III
test unless noted and are reproducible from this repository.

---

## 1. One-line thesis
On the TOP benchmark, reported progress in clinical-trial-outcome prediction is **not measurable**:
HINT, MEXA-CTP, tuned gradient boosting, a zero-training tabular foundation model, and a two-column
historical base-rate lookup are statistically indistinguishable, and all effect sizes lie below the
benchmark's sampling-noise floor.

## 2. Argument spine (claim → evidence)
1. **The published comparison is ill-posed** → different epoch-selection (code selects on *test*
   though the paper says validation), fixed-0.5 F1, PR-AUC on binarised scores, no trivial
   baseline, no difference test. *(§2.1)*
2. **Fix it with one shared harness** → threshold@validation, ranking on continuous scores,
   bootstrap CIs, trivial baseline, one evaluator for every model. *(§2.2)*
3. **Under the fair protocol the MEXA-CTP advantage disappears** and neither deep model beats
   gradient boosting. *(§4.1)*
4. **The mechanisms don't work**: MEXA-CTP's mode-experts were never trained (stored in a `dict`);
   HINT's ablation ladder is flat (disease branch ≈ full model). *(§3, §4.1)*
5. **A zero-training tabular model and a validation-selected grid match HINT**; all signal is in the
   disease codes. *(§4.2)*
6. **A two-column base-rate lookup tops everything** (ROC 0.700) — the actuarial table industry
   already uses. *(§4.2)*
7. **It's a measurability limit**: per-phase MDE 0.035–0.067; every literature gain is below it. *(§4.5)*
8. **We overturned seven of our own findings** as direct evidence of the noise floor. *(§5)*

## 3. The data (and a prior discrepancy)
| Phase | train/valid/test (shipped = ours) | test pos. rate | MEXA-CTP Table 1 (train/test/success) |
|---|---|---|---|
| I   | 1044 / 117 / 627  | 0.553 | 1,088 / 312 / 70% |
| II  | 4005 / 446 / 1654 | 0.555 | 2,611 / 789 / 33% |
| III | 3094 / 344 / 1146 | 0.750 | 4,313 / 1,147 / 30% |

MEXA-CTP's shipped `DataPath` equals ours exactly, but its **paper Table 1 matches neither its own
data nor its results** (its reported HINT ROC .573/.621/.685 = ours, i.e. computed on the shipped
split, not the Table-1 split). Split is temporal (cut 2014-01-01).

## 4. Defect catalogue
| ID | Where | Defect | Consequence | Status |
|---|---|---|---|---|
| D1 | MEXA | attention/routers in a plain `dict` | 0/264 tensors trained; absent from checkpoints | mechanism never trained |
| D2 | MEXA | full-masking → NaN | trains only because D1 froze routers | two bugs conceal each other |
| D3 | HINT | PR-AUC on binarised preds | understates 0.02–0.08; reorders variants | metric artefact |
| D4 | HINT | F1 threshold hard-coded 0.5 | F1 = class prior | metric artefact |
| D5 | HINT | empty `sentence2embedding.pkl` | criteria branch = zero vector | **repaired + measured inert (ΔROC −0.0001, p=0.93)** |
| D6 | MEXA | `.to(get_device())` hard-codes GPU | no CPU run | runtime |
| D7 | HINT | ablation variants miss `device` | 2 variants un-instantiable | runtime |
| D8 | MEXA | lr 5e-2 diverges once attn trains | survived only via D1 | runtime |
| D9 | MEXA | `squeeze()` drops batch dim (B=1) | silent shape bug | runtime |
| D10 | HINT | checkpoint load fails on PyTorch ≥2.6 | reproducibility break (hit us in Route B) | runtime |

## 5. The phase-III ladder (headline result)
| Model / baseline | ROC-AUC |
|---|---|
| Trivial (all-positive) | 0.500 |
| MEXA-CTP (released checkpoint) | 0.518 |
| Logistic regression | 0.554 |
| Random forest | 0.620 |
| Gradient boosting | 0.651 |
| TabFM (zero-training) | 0.668 |
| ICD-only (validation-selected) | 0.671 |
| MEXA-CTP (retrained) | 0.674 |
| Target-encoding (2 features) | 0.676 |
| CatBoost (tuned) | 0.680 |
| **HINT** | **0.685** |
| XGBoost (tuned) | 0.691 |
| ICD-code base-rate lookup | 0.694 |
| **Two-column base-rate lookup** | **0.700** (0.726 on well-supported cells) |

Complexity ↑ → score unchanged; simplicity ↑ → score rises.

## 6. Experiment inventory (what → number → script)
- **Fair head-to-head** — MEXA advantage gone; released ckpt ≈ chance (0.518). → `paper/eval_harness.py`, `paper/verify_paper_tables.py`
- **A/B: train the cross-attention?** — Δ=+0.0004, p=0.96 (buggy ≥ fixed). → `ab_test_results_*`
- **HINT ablation ladder** — disease branch ≈ full; drug branch ≈ chance (0.540). → `preds_hintvar_*`
- **Classical + tuned GBDT** — XGB 0.691, Cat 0.680, LGBM 0.647; DeLong HINT−GradBoost p=0.052. → `tabfm_baseline_optionA/gbdt_baselines.py`
- **Zero-training TabFM / grid** — TabFM 0.668; val-selected grid 0.690; TabPFN-v2 0.647. → `tabfm_baseline_optionA/`
- **Where signal lives (GRAM)** — GRAM carries it; top-500 ≈ full (0.642 vs 0.649). → `tabfm_gram500.py --proxy`
- **Molecule is untestable** — one SMILES (alogliptin) = 46% of test (528/1146, 759 drugs, 410 placebo); on the clean 54%, HINT `Only_Molecule` 0.559, Morgan-2048 0.66. → `data_audit.py`, `reviewer_batch5/6.py`
- **Drug recurrence = programme continuation** — 91% reuse ≥1 train drug; 79% same (drug, indication). → `reviewer_batch5.py`
- **Base-rate lookup** — combined 0.700; no-fallback 0.726 (> HINT 0.712 there); ≥5-trial cells 0.723; target-encoding 0.676. → `data_audit.py`, `reviewer_batch6.py`
- **Calibration / DCA** — TabFM better on I/II, HINT on III; lookup ≥ HINT/TabFM in net benefit. → `calibration.py`, `dca.py`
- **PR-AUC incomparable** — same HINT, 4 papers: 0.603 / 0.797 / 0.811 / 0.852. MEXA's 0.603 < 0.75 prevalence floor (impossible); MEXA F1 0.857 = trivial 0.8569; MEXA's own Table 3: token selection ≈0 on ROC. → PDFs + `reviewer_batch4.py`
- **Power** — DeLong HINT−TabFM p=0.33, −base-rate p=0.52; permutation p=0.34; Wilcoxon A/B p=0.81; MDE I/II/III = 0.067/0.035/0.050; TOST δ=0.02 not equivalent. → `stats_rigor.py`, `reviewer_batch3.py`
- **Variance sources (labelled)** — test-sampling SE 0.018 (dominant); HINT 5-seed sd **0.0024**; TabFM 0.0011; MEXA retrain 0.011–0.015 (~5× HINT); MEXA reload 0.023 (defect, not seed); split-var gauge ~0.02–0.04. → `reviewer_batch5.py`
- **Six→seven overturned findings** — routing-harmful, attn-hurts, PR-bug (narrowed), 22-feature, tabular-beats-HINT, fusion-gain, and **seed-noise-large** (→ 0.0024). → §5, `tab:dissolved`

## 7. Review-response log (thematic)
- Relative→absolute MEXA gain (0.008–0.020, not %); code-vs-paper epoch selection; MEXA split
  discrepancy; encoder table (HINT/MEXA-pub/MEXA-ours differ); Cauchy-loss gradient flow;
  D5 conservative→measured; multiplicity policy (primary/secondary); per-phase MDE; industry-LOA
  framing; forward-chaining CV + asymmetry; GRAM≠leakage; survivorship (no NCT gradient);
  ICD-2024 anachronism + 0.75-vs-~55–60% literature; ladder table; abstract de-referenced.

## 8. Repository & reproduction
```
paper/tex/paper-last.tex   # manuscript (single file, no \input)
paper/*.py + preds/ + *.csv # analysis, per-trial predictions, result tables
paper/DISCLOSURE.md         # responsible disclosure: HINT #15, MEXA #2
paper/reviewer_batch{3,4,5,6}.py, regen_biobert_cache.py
tabfm_baseline_optionA/     # tabular/TabFM/GBDT/probes/GRAM-500
tabfm_baseline_optionB/     # prediction-level fusion
hint/data/ + hint/benchmark/
regenerate_all.sh           # one command → every table from stored preds
```
`bash regenerate_all.sh` (CPU, ~2–5 min).  Audited: HINT `8dc0497`, MEXA `b898dbb` (2026-07).

## 9. Done vs. owed
- **Done**: all tables verified from raw data; D5 measured inert; HINT 5-seed sd measured;
  disclosure filed (#15, #2); commits stamped; repo public (English, reproducible).
- **Owed (optional, cluster)**: Route B — full HINT retrain-from-scratch with live cache (D10
  patch ready); real TabFM on top-500 GRAM (proxy already answers it); forward-chaining applied to
  HINT itself (asymmetry currently stated, conservative).

## 10. Venue
Targets (all single-anonymized, so the public repo/name are fine): **AI in Medicine** (best fit) →
**Machine Learning with Applications** → **Array**.
