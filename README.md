# Is Progress on the TOP Benchmark Measurable? A Forensic Re-evaluation of HINT and MEXA-CTP

Code, per-trial predictions, result tables and paper source for a re-evaluation of clinical
trial outcome prediction on the **TOP benchmark** (HINT vs MEXA-CTP, plus tabular and
zero-training baselines).

**Summary of the finding.** Under a shared, defect-corrected evaluation protocol, the reported
advantage of MEXA-CTP over HINT does not survive, and neither deep model beats gradient boosting
or a zero-training tabular foundation model on the same inputs. All predictive signal is carried
by disease-ontology codes; a two-column base-rate lookup (disease code + drug history) reaches
ROC-AUC 0.700 on phase III — above every deep and foundation model. On phase III, the minimum
detectable ROC-AUC difference at 80% power is ≈0.05, so every between-model gain reported in this
literature is below the benchmark's noise floor at its current size.

## Repository layout

```
paper/
  tex/paper-last.tex        # the manuscript (single self-contained file)
  tex/cas-refs.bib          # bibliography
  *.py                      # analysis + evaluation scripts
  preds/                    # per-trial predictions (nctid, score, label, split)
  *.csv                     # result tables the paper is built from
  DISCLOSURE.md             # responsible-disclosure record (defect reports)
  PROVENANCE.md             # maps every number in the paper to its source
tabfm_baseline_optionA/    # tabular / TabFM baselines, GBDT, probes, GRAM-500
tabfm_baseline_optionB/    # prediction-level fusion (stacking / z / rank / NNLS)
hint/
  data/                     # TOP phase split CSVs (train/valid/test) used by the scripts
  benchmark/                # HINT's benchmark builder (referenced for the SMILES-collision trace)
regenerate_all.sh           # one command to reproduce every table/statistic from stored preds
```

## Reproducing the results

Every table in the paper is regenerated from the stored predictions **without retraining**.

```bash
# Python 3.10+; see requirements below
bash regenerate_all.sh          # CPU-only, no GPU/network; ~2–5 min
bash regenerate_all.sh --full   # adds the long / GPU-dependent jobs (TabFM, D5 HINT re-run)
```

Individual analyses:

```bash
python paper/verify_paper_tables.py     # cross-checks every typeset number vs raw preds
python paper/stats_rigor.py             # DeLong ΔAUC, TOST, MDE, MCC / balanced acc
python paper/calibration.py             # Brier + equal-mass ECE with bootstrap CIs
python paper/dca.py                     # decision-curve analysis (net benefit)
python paper/data_audit.py --phase III  # temporal split, drug recurrence, SMILES collision, base-rate lookup
python paper/reviewer_batch3.py         # molecule-only 54%, coverage, same-programme, Wilcoxon/permutation, threshold stability
python tabfm_baseline_optionA/tabfm_gram500.py --phase III --proxy   # 500-feature cap is not the bottleneck (CPU proxy)
```

### Requirements

Python 3.10+ with: `numpy pandas scikit-learn scipy rdkit xgboost lightgbm catboost`.
The zero-training foundation-model runs additionally need the `tabfm` (JAX/CUDA) or `tabpfn`
environments and a GPU; these are optional and gated behind `--full`.

### Data

The `hint/data/` phase split CSVs are included for convenience, together with the GRAM
disease-ancestor map (`icdcode2ancestor_dict.pkl`) and the shipped empty criteria-embedding
cache (`sentence2embedding.pkl`, 6 bytes — the defect D5 evidence). The full raw TOP benchmark
and the HINT/MEXA-CTP model code come from the upstream repositories:

- HINT / TOP benchmark: https://github.com/futianfan/clinical-trial-outcome-prediction
- MEXA-CTP: the authors' released repository

Raw model checkpoints, the 55 MB `raw_data.csv`, ADMET/ICD auxiliary files and large intermediate
result files are **not** included here; they are re-derivable from the upstream data.

## Responsible disclosure

The code-level defects documented in the paper (D1–D10) are framed as ecosystem reproducibility
hazards, not author misconduct. See `paper/DISCLOSURE.md` for the ready-to-file issue text and the
record of reports to the upstream maintainers. **Before publication, file the two issues and record
the URLs + audited commit hashes in that file.**

## Citation

Çetiner, A. M. and Adebayo, K. J. *Is Progress on the TOP Benchmark Measurable? A Forensic
Re-evaluation of HINT and MEXA-CTP.* (working paper; see `paper/tex/paper-last.tex`).

## License

Code released under the MIT License (see `LICENSE`). The TOP benchmark data and the HINT/MEXA-CTP
implementations remain under their respective upstream licenses.
