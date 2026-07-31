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
patches/                    # study changes to the audited HINT/MEXA upstream commits
requirements.txt            # CPU stored-prediction/classical-baseline environment
requirements-optional.txt   # upstream inference/cache-generation extras
regenerate_all.sh           # one command to reproduce every table/statistic from stored preds
```

## Reproducing the results

Install the CPU dependencies:

```bash
python -m pip install -r requirements.txt
```

The primary reproduction path uses the released per-trial predictions and does
**not** clone either upstream model, load a checkpoint, or retrain:

```bash
bash regenerate_all.sh          # CPU-only, no GPU/network; ~2–5 min
python paper/quantify_d5.py --phase III  # D5 populated-cache comparison from stored predictions
```

Full model inference or retraining is a separate, optional path. Apply the
study patches described in [`patches/README.md`](patches/README.md) to clean
clones of the audited HINT and MEXA-CTP commits, install
`requirements-optional.txt` plus each upstream project's own environment, and
provide the external artefacts described below. TabFM's recorded JAX
environment is in `tabfm_baseline_optionA/requirements_tabfm_jax.txt`.
`bash regenerate_all.sh --full` runs the available long/GPU checks once that
environment and the external artefacts have been supplied.

For upstream-model scripts, set clone/data locations without editing source:

```bash
export CTP_HINT_DIR=/path/to/patched/hint
export CTP_MEXA_DIR=/path/to/patched/mexa
export MEXA_DATA_FOLDER=/path/to/mexa/DataPath/
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

Python 3.10+ is recommended. `requirements.txt` covers stored-prediction
analysis and the classical baselines. `requirements-optional.txt` documents
the additional HINT/cache-generation packages. TabFM/TabPFN and the original
HINT/MEXA training stacks remain separate GPU environments because their
platform-specific dependencies cannot be represented by one portable root
requirements file.

### Data

The `hint/data/` phase split CSVs are included for convenience, together with the GRAM
disease-ancestor map (`icdcode2ancestor_dict.pkl`) and the shipped empty criteria-embedding
cache (`sentence2embedding.pkl`, 6 bytes — the defect D5 evidence). The populated-cache
phase-III per-trial scores used to quantify D5 are released as
`paper/preds/preds_hint_criteria_phase_III.csv` (see `paper/PROVENANCE.md`).
The full raw TOP benchmark and the HINT/MEXA-CTP model code come from the
upstream repositories:

- HINT: https://github.com/futianfan/clinical-trial-outcome-prediction at `8dc0497f23fdb84e2905da7655924a91e6e79798`
- MEXA-CTP: https://github.com/murai-lab/MEXA-CTP at `b898dbb41930197187fe3867e3616ffd406a48a6`

The study-authored modifications to those exact revisions are fully released
as unified patches in `patches/`; third-party source trees are not vendored.
Raw model checkpoints, the populated 1.7 GB BioBERT cache, MEXA's generated
`DataPath/`, the 55 MB `raw_data.csv`, ADMET/ICD auxiliary files and large
intermediate results are **not** included because of size, licensing, or
third-party provenance. The stored-prediction path does not require them.

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
