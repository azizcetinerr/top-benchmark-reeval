# Cover letter — Artificial Intelligence in Medicine

**Manuscript title:** Is Progress on the TOP Benchmark Measurable? A Forensic Re-evaluation of HINT and MEXA-CTP

**Authors:** Aziz Muammer Çetiner; Dr. Kolawole John Adebayo (corresponding author)  
**Affiliation:** ADAPT Centre, Maynooth University, Maynooth, Kildare, Ireland  
**Emails:** cetiner.aziz1@gmail.com; kolawole.adebayo@mu.ie

Dear Editors,

We submit the enclosed manuscript for consideration as an original research article in *Artificial Intelligence in Medicine*.

Predicting clinical-trial outcomes is a high-stakes ML task whose published progress is measured almost exclusively against HINT on the TOP benchmark. We ask whether that progress is measurable. Under a shared, defect-corrected evaluation harness we find that it is not.

Building the harness required correcting ten code-level defects in the released HINT and MEXA-CTP implementations. Two are severe: MEXA-CTP’s mode-experts mechanism (the paper’s central claim) was stored in a plain Python `dict`, so it never entered the optimizer and was never trained; HINT’s shipped eligibility-criteria embedding cache is empty, silently replacing that modality with a zero vector. After repairing both defects and re-scoring, the published MEXA-CTP advantage over HINT disappears, the repaired mechanism contributes nothing, and the restored criteria input is inert (ΔROC-AUC = −0.0001, *p* = 0.93). Neither deep model beats gradient boosting on the same inputs. A zero-training tabular foundation model and a validation-selected classical grid match HINT in every phase. All recoverable signal lies in disease-ontology codes; molecular structure is untestable on this benchmark (one SMILES covers 46% of phase-III test trials). A two-column historical base-rate lookup reaches ROC-AUC 0.700 — above every deep and foundation model tested.

A power analysis makes the methodological claim quantitative: at 80% power the minimum detectable ROC-AUC difference is 0.035–0.067 depending on phase. Every between-model gain reported in this literature falls below that floor. Of seven directional findings we ourselves formed, six were overturned and one narrowed under replication.

We believe these results are directly relevant to AIM’s readership: they audit a widely reused clinical ML benchmark, document reproducibility hazards that propagate through successor work, and give concrete protocol requirements (multiple seeds, validation-based selection, confidence intervals, a trivial baseline, and an explicit minimum detectable effect) before improvement claims can be treated as verifiable.

**Code and data.** All code, raw per-trial predictions, and regenerated tables are available at https://github.com/azizcetinerr/top-benchmark-reeval. Defects were disclosed to the maintainers before submission (HINT issue #15; MEXA-CTP issue #2).

We confirm that the work is original, not under consideration elsewhere, and that all authors approve this submission. We have no competing interests.

Thank you for your consideration.

Sincerely,  
Dr. Kolawole John Adebayo (corresponding author)  
On behalf of all authors
