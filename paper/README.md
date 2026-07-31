# HINT vs MEXA-CTP — A Fair Re-evaluation

This project re-evaluates two clinical trial outcome prediction models — **HINT** and
**MEXA-CTP** — under a *single, correct, shared* protocol.

**Motivation.** The published claim that "MEXA outperforms HINT by 4.9%" is not
interpretable, because the two models were measured under different rules, and several of
those rules were themselves incorrect. Before asking *which model is better*, we first had
to make the measurement valid.

What began as a protocol-alignment exercise turned into something larger: we found that
MEXA's central architectural mechanism was never trained, never saved, and — once repaired —
**cannot be trained at all as published**, because a second bug was hidden behind the first.

---

## 1. Backbone: how the comparison is made fair

HINT and MEXA are separate repositories with separate dependencies and conflicting module
names. Importing both into one process is fragile. The architecture we adopted instead:

> **Each model runs in its own context and dumps raw predictions (score + label, for both
> validation and test) to disk. A single shared evaluation harness then scores everything
> under identical rules.**

```
   HINT repo                            MEXA repo
 (own context)                        (own context)
       │ raw score+label                    │ raw score+label
       ▼ (valid + test)                     ▼ (valid + test)
  preds_hint_*.csv                     preds_mexa_*.csv
       └────────────────┬───────────────────┘
                        ▼
             SHARED EVALUATION HARNESS          ← single source of truth
     threshold@validation · PR/ROC on scores · bootstrap CI · trivial baseline
                        ▼
                 FAIR COMPARISON TABLE
```

This is what makes the comparison valid: threshold selection, metric computation, and
confidence intervals all flow through **one** piece of code. The failure mode we were
trying to escape — *two repositories making the same mistake in different ways* — becomes
structurally impossible, because the repositories no longer perform evaluation at all.

`paper/main.py` is the control panel: a `# %%` cell script executed block by block, never
end-to-end. Heavy computation lives in the repos and the harness.

---

## 2. Bugs found → fixes applied

### 2.1 Original five items

| # | Problem | Why it invalidates results | Fix | Status |
|---|---|---|---|---|
| 1 | **Test-set epoch selection** | MEXA selected its best epoch by *test* ROC across 100 epochs; HINT selected on validation. MEXA was measured under a protocol that gave it an advantage. | Selection moved to validation; early stopping added (`--patience`). | Code ✅ · requires retraining |
| 2 | **`nn.ModuleDict` bug** | Attention layers and routers were stored in plain Python `dict`s. PyTorch does not register submodules inside plain dicts → they never entered the optimizer and were never written to `state_dict`. The paper's central mechanism was **never trained and never saved**. | Converted to `nn.ModuleDict`, with a toggle to reproduce the original behaviour for controlled A/B. | ✅ **verified empirically** |
| 3 | **Fixed decision threshold** | `decisions = (preds > 0)` / `threshold = 0.5` hard-coded. F1 therefore measured the class prior, not the model. Ironically HINT *has* `select_threshold_for_binary`, but it is commented out in `bootstrap_test`. | Harness selects the threshold on **validation** (maximising F1) and reports test metrics at that threshold. | ✅ |
| 4 | **Non-reproducible embeddings** | MEXA's data pipeline did not pass `random_state`, so SMILES embeddings were not reproducible. | `build_mexa_data.py` sets MACAW `random_state=2023`; ICD hierarchy version is logged. | ✅ |
| 5 | **Broken PR-AUC** | HINT's `evaluation()` binarises predictions **before** computing `average_precision_score`, so PR-AUC is computed on 0/1 values rather than scores. HINT is the reference model of the TOP benchmark, so this number may have propagated into other papers. | Harness computes ROC and PR-AUC on continuous scores. | ✅ **quantified** |

### 2.2 Additional bugs discovered during this work

| Problem | Detail | Fix | Status |
|---|---|---|---|
| **Full-masking → NaN** *(most significant)* | Once routers become trainable, `CauchyLoss` drives routing probabilities toward 0. When *all* tokens of a sequence fall below `threshold`, the entire `key_padding_mask` row becomes `True`; softmax over all `-inf` yields **NaN**, destroying the weights. First NaN scales inversely with lr (step 9/28/184 for 5e-2/1e-2/1e-3; none at 1e-4 in 300 steps, collapse still progressing). | `remasks` now guarantees at least one token remains unmasked (the highest-probability one); this prevents NaN but not collapse — every sequence still reduces to one token. | ✅ chain proven, fix verified |
| **Invalid hyperparameters** | The published `lr=5e-2` diverges once the attention layers actually receive gradients. It only survived because ~7.5k parameters were frozen. | Learning rate re-tuned on validation, identically for both arms. | ✅ |
| **CPU incompatibility** | `PositionalEncoding` used `.to(embeddings.get_device())`, which returns `-1` on CPU → `RuntimeError`. MEXA **could not run on CPU at all**; it had only ever been executed on CUDA. | Use `.device`. GPU behaviour unchanged. | ✅ |
| **`squeeze()` batch collapse** | `prob.squeeze()` in `remasks` collapses the batch dimension when `B == 1`. | `squeeze(-1)`. | ✅ |
| **`torch.load` incompatibility** | PyTorch ≥ 2.6 changed `weights_only` to `True` by default; both checkpoints contain numpy scalars and fail to load. | `weights_only=False` (trusted, locally obtained checkpoints). | ✅ |
| **ICD version mismatch** | `icdcodex` rejects version 2025 despite an error message that implies it is available (exclusive upper bound). | Auto-detect the newest working version and log it. | ✅ |

### 2.3 The central architectural finding: two bugs concealing each other

MEXA's architecture, **as published, cannot be trained as designed.** The causal chain:

1. `CauchyLoss = log(1 + (p/ε)²)` exists precisely to push routing probabilities toward 0
   (sparse routing).
2. `remasks` masks every token whose probability falls below `threshold = 0.3`.
3. If *all* tokens in a sequence fall below it, the whole row is masked.
4. `nn.MultiheadAttention` with a fully-masked `key_padding_mask` row produces **NaN**.

**Why was this never observed?** Because the routers lived in a plain `dict` and were
**never trained** — their outputs stayed frozen near 0.5, so the Cauchy loss never actually
did anything, so full masking never occurred. **The `ModuleDict` bug was the only reason
training did not crash.**

Empirical evidence (all reproduced):

| Claim | Result |
|---|---|
| Fully-masked row → attention NaN | confirmed |
| `remasks` can mask an entire row | confirmed |
| Collapse scales inversely with lr: first NaN at step 9/28/184 for lr 5e-2/1e-2/1e-3; none at 1e-4 in 300 steps | confirmed |
| Tokens below threshold at step 300 rise with lr: 13% → 100% (1e-4 → 5e-2) | confirmed — structural, not step-size |
| Guard prevents NaN but not collapse: at 5e-2 mean router prob → 0.003, every sequence reduced to a single token | confirmed |

The guard is applied to **both** arms. In the buggy arm the routers stay frozen near 0.5,
so it essentially never triggers — the comparison remains fair.

### 2.4 Evidence for the `ModuleDict` bug

| Configuration | self-att | cross-att | routers | trainable params |
|---|---|---|---|---|
| Original (plain `dict`) | 0 tensors | 0 tensors | 0 tensors | 312,034 |
| Fixed (`nn.ModuleDict`) | 72 | 144 | 48 | 319,562 |

- In the fixed model, **144/144** cross-attention tensors are in the optimizer and **all 144
  changed** after one backward + step → the mechanism genuinely trains.
- In the original model, **0/264** mechanism tensors (72 self-att + 144 cross-att + 48
  router) are in the optimizer → training is impossible.
- The released checkpoints contain **49 keys (phases I, III) / 48 (phase II)**, none of them
  attention or router — all **264** mechanism tensors are absent, as is any trace of them in
  the stored `optimizer` state, confirming at the serialization level that the mechanism was
  never saved.

### 2.5 The guard prevents NaN but not collapse

Instrumenting `remasks` and training phase II for 300 steps (seed 2023, real data)
shows the collapse is a deterministic consequence of the Cauchy pressure, not a random
numerical fluke. Without the guard, a fully-masked row triggers NaN, and the onset
scales inversely with the learning rate:

| lr | first NaN step (no guard) | tokens below threshold @ step 300 |
|---|---|---|
| 5e-2 | 9 | 100% |
| 1e-2 | 28 | 88% |
| 1e-3 | 184 | 81% |
| 1e-4 | none in 300 | 13% |

Adding the minimal guard (keep the highest-probability token) removes the NaN but **not
the collapse**: at lr 5e-2 the mean router probability falls to **0.003** and every
sequence is reduced to a single surviving token, so the cross-attention it is meant to
learn operates over one token. The architecture can be kept numerically alive, but not
architecturally — exactly the mechanism no prior work ever exercised.

---

## 3. Results

All results: phase III, TOP benchmark splits, `valid n=344 (66.6% positive)`,
`test n=1146 (75.0% positive)`. Epochs selected on validation; the test set never
influenced any selection decision.

### 3.1 Main comparison table

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| MEXA-**buggy**, properly retrained | **0.6975** | **0.8705** | 0.804 |
| HINT (released checkpoint) | 0.6851 [0.647, 0.720] | 0.8517 [0.824, 0.878] | 0.826 [0.806, 0.845] |
| MEXA-**fixed**, properly retrained | 0.6572 | 0.8465 | 0.808 |
| MEXA (released checkpoint) | 0.519 [0.484, 0.553] | 0.789 [0.761, 0.817] | 0.855 [0.839, 0.871] |
| **TRIVIAL** (predict all positive) | 0.500 (chance) | 0.750 (chance) | **0.857** |

### 3.2 F1 inverts the ranking — a metric artefact

- By **F1**, MEXA (0.855) *beats* HINT (0.826).
- But the **trivial all-positive classifier scores 0.857**, beating both.
- MEXA's F1 is high precisely because it is **more degenerate**: the closer a model gets to
  predicting everything positive, the higher its F1 climbs on a 75%-positive test set.
- By **ROC-AUC** the ranking reverses, and MEXA's confidence interval **includes 0.5**.

**Conclusion:** on this benchmark F1 measures the class prior, not model quality. Every
result must be reported alongside the trivial baseline. The harness now does this
automatically.

Even threshold tuning *on the test set* (an upper bound we would never use in practice)
only reaches F1 = 0.8596 — barely above trivial.

| Threshold | F1 |
|---|---|
| Fixed 0.5 (original) | 0.8123 |
| Validation-calibrated 0.461 | 0.8263 |
| Test-oracle 0.327 (upper bound) | 0.8596 |
| **Trivial (all positive)** | **0.8569** |

### 3.3 Item 5 quantified — the PR-AUC bug

| Computation | PR-AUC |
|---|---|
| Correct (continuous scores) | **0.8517** |
| Broken (binarised at 0.5) | 0.7995 |

The bug **understates** PR-AUC by 0.052. The direction matters: the broken number makes
HINT look *worse* than it is, so any work citing HINT's published PR-AUC as a baseline was
comparing against an unfairly weakened reference.

### 3.4 Seed sensitivity of the released MEXA checkpoint

Because the attention and router weights were never saved, they are re-initialised randomly
on every load. Same checkpoint, five seeds:

| Metric | mean | std | min | max | range |
|---|---|---|---|---|---|
| ROC-AUC | 0.5175 | 0.0204 | 0.4784 | 0.5335 | **0.0552** |
| PR-AUC | 0.7748 | 0.0116 | 0.7545 | 0.7879 | 0.0334 |
| F1 | 0.8548 | 0.0027 | 0.8495 | 0.8569 | 0.0074 |

- The finding is **robust, not a fluke**: every seed lands near chance; one falls *below* 0.5.
- F1 is almost constant and pinned to the trivial baseline — it carries no information here.
- **Critical comparison:** the claimed improvement is 4.9%; the ROC variation caused purely
  by an unsaved random initialisation is **5.5%**. The reported effect is smaller than the
  noise of an artefact that should not exist.

### 3.5 ⭐⭐ Decisive result — all three phases (9 run pairs)

Phases I, II and III; lr = 1e-2, 3 seeds each, all converged via early stopping. Epochs
selected on validation; test never influenced selection.

#### Paired analysis — runs are matched by (phase, seed)

Phase III extended to **7 seeds** (phases I and II have 3 each).

| Analysis | Difference (buggy − fixed) | p |
|---|---|---|
| **Phase III, 7 seeds** | **+0.0004** | **0.96** |
| Phase I, 3 seeds | +0.0323 | 0.099 |
| Phase II, 3 seeds | +0.0101 | 0.172 |
| **All, 13 matched pairs** | **+0.0100** | **0.129** (Wilcoxon 0.080) |

In the best-powered phase (III, 7 seeds) the difference is **exactly zero**.

> **Conclusion:** training the cross-attention has **no effect whatsoever** on performance.
> MEXA's mode-expert mechanism performs at least as well frozen at random initialisation as
> when trained. Phases I/II show a weak "it hurts" tendency (6/6 pairs positive) but it is
> not significant and rests on few seeds.

#### ⚠️ Methodological note #2 — more seeds dissolved a second effect

| Stage | Phase III difference | Pooled (all phases) |
|---|---|---|
| 3 seeds | +0.0031 (p = 0.76) | +0.0152, **p = 0.056** |
| **7 seeds** | **+0.0004 (p = 0.96)** | +0.0100, **p = 0.129** |

The "marginally harmful (p ≈ 0.055)" result reported in the previous round came from
phase III's 3-seed sample. At 7 seeds the effect disappeared.

**This is the same lesson a second time** (the first: the threshold hypothesis, 1 seed →
3 seeds). Small-sample effects dissolved repeatedly in this project. This belongs in the
paper's limitations: *on this benchmark, seed noise exceeds the effect sizes being sought.*

#### Hyperparameter sweep — the mechanism's fairest remaining chance (phase III, seed 2023)

Sweeping only the learning rate was not enough, so `threshold` (routing aggressiveness),
`rho1` (Cauchy sparsity) and `rho2` (NT-Xent contrastive) were swept as well.

| threshold | valid ROC | | rho2 | valid ROC |
|---|---|---|---|---|
| 0.0 (routing OFF) | 0.7633 | | 0.0 (contrastive OFF) | 0.7452 |
| **0.1 (weak)** | **0.7645** | | 0.001 | 0.7541 |
| 0.3 (published) | 0.7446 | | 0.01 (published) | 0.7446 |
| 0.5 (aggressive) | 0.7471 | | 0.1 | 0.7452 |

**Two further components turn out to be inert or harmful:**
1. **The routing/masking mechanism hurts.** At the published threshold (0.3) it scores
   0.7446; weakening it (0.1) or switching it off entirely (0.0) **improves** results by ~0.02.
2. **The NT-Xent contrastive loss contributes nothing.** Disabling it completely
   (rho2 = 0 → 0.7452) matches the published setting (0.01 → 0.7446).
3. `rho1` is irrelevant as long as it is non-zero (0.01 / 0.05 / 0.2 all ≈ 0.744).

**The critical comparison:**

| | valid ROC |
|---|---|
| **Best fixed** configuration in the whole sweep (thr = 0.1) | 0.7645 |
| **Buggy** arm at its **untuned** default | 0.7683 |

> We tuned the fixed arm across three hyperparameters and left the buggy arm untuned.
> **Even under this asymmetric test — deliberately biased in the fixed arm's favour — it
> failed to win (−0.0038).** The mechanism was given every fair chance and could not use it.
>
> *Caveat: one seed (2023) per configuration and validation figures; the 0.7645 vs 0.7683
> gap is within noise. The honest statement is "at best a tie, never a win."*

#### ⭐ Every architectural component is inert — hyperparameter sweep

Phase III, lr = 1e-2, **3 seeds**, fixed arm. The routing `threshold` controls `remasks`;
`0.0` disables the mechanism entirely.

| threshold | valid (3 seeds) | test (3 seeds) |
|---|---|---|
| 0.0 (routing off) | 0.7484 ± 0.0221 | 0.6836 ± 0.0113 |
| 0.1 | 0.7516 ± 0.0152 | 0.6829 ± 0.0130 |
| 0.3 (**published**) | 0.7462 ± 0.0024 | 0.6776 ± 0.0023 |
| — (buggy arm) | 0.7580 ± 0.0114 | 0.6807 ± 0.0156 |

The `thr=0.0` vs `0.3` difference on test is **+0.006, p = 0.46** → not significant.

> **Conclusion:** disabling the routing/sparsification mechanism entirely changes nothing.
> Cross-attention (trained or not), routing (on or off), and NT-Xent (present or absent) —
> **none of MEXA's distinguishing components produces a measurable effect.** Performance
> comes entirely from the input projections + compensation self-attention + prediction head.

#### ⚠️ Methodological note — a single seed misled us (recorded deliberately)

The first sweep used a **single seed** (2023) and showed **+0.0187** for `thr=0.0`. This was
interpreted as a mechanistic explanation ("routing is actively harmful"). The reasoning was:
"the fixed arm's seed std is 0.0024, so +0.019 ≈ 8 std must be real."

**That reasoning was wrong.** The std came from the `thr=0.3` configuration, and seed
variance is not constant across configurations (`thr=0.0` has std 0.0221 — roughly ten
times larger). Estimating noise on one configuration and applying it to another is invalid.
With three seeds the effect shrank to +0.0022 and disappeared.

Lesson: `thr=0.3` being unusually stable (std 0.0024) was misleading; noise must be
estimated **per configuration**.

#### The NT-Xent contrastive loss is also inert

| rho2 (NT-Xent weight) | valid ROC |
|---|---|
| 0.0 (fully off) | 0.7452 |
| 0.001 | 0.7541 |
| 0.01 (default) | 0.7446 |
| 0.1 | 0.7452 |

Switching the loss off entirely changes nothing. MEXA's second contribution produces no
measurable effect either. (For `rho1`/Cauchy, 0.0 → 0.7307 is slightly *worse*, suggesting
it acts as a regulariser on the router outputs.)

> ⚠️ This sweep is **single-seed**. Given the fixed arm's seed std of 0.0024, the threshold
> effect (+0.019 ≈ 8 std) is probably real, but multi-seed confirmation is required.

#### Test ROC-AUC by phase

| Phase | HINT | MEXA-buggy | MEXA-fixed | LogReg | TRIVIAL |
|---|---|---|---|---|---|
| I | 0.574 | 0.569 | **0.537** | **0.562** | 0.500 |
| II | 0.621 | 0.617 | 0.606 | — | 0.500 |
| III | 0.685 | 0.681 | 0.678 | — | 0.500 |

- **In all three phases HINT ≈ MEXA-buggy ≈ MEXA-fixed** — separated by less than 0.005
  (except phase I). The "4.9% improvement" claim survives in no phase.
- **On phase I, logistic regression (0.562) beats the properly fixed MEXA (0.537)** and
  comes close to HINT (0.574). The deep architectures contribute essentially nothing here.
- Predictability rises with phase number (I < II < III) — late-phase trials are easier.

### 3.4b ⭐⭐⭐ HEADLINE — the task reduces to 22 binary features

`icd_representation_ablation.py` compares four representations of the *same* ICD codes
under the same classifiers and the same harness. All three phases, test ROC-AUC.

**(a) The ICD representation does not matter.**

| Phase | icd2vec (64-d node2vec) | chapter one-hot (22–23-d) | diff | CI overlap |
|---|---|---|---|---|
| I | 0.5596 [0.515, 0.603] | 0.6091 [0.565, 0.651] | −0.050 | yes |
| II | 0.6204 [0.594, 0.646] | 0.6207 [0.595, 0.649] | −0.0003 | yes |
| III | 0.6984 [0.661, 0.735] | 0.6763 [0.640, 0.711] | +0.022 | yes |

Direction inconsistent, mean −0.009, every CI overlaps. MEXA's node2vec embedding is
indistinguishable from a 23-dimensional indicator vector.

**(b) Finer granularity does not help — it hurts.**

| Phase | chapter (22–23 dims) | full ICD code (1301–2358 dims) | diff |
|---|---|---|---|
| I | 0.6091 | 0.5517 | **+0.057** |
| II | 0.6207 | 0.6162 | +0.005 |
| III | 0.6763 | 0.6721 | +0.004 |

Chapter-level beats full-code in all three phases. Sub-chapter detail adds nothing.

**(c) Simple ICD-only pipelines land within ~0.03 ROC of the full architecture.**

Pipeline (representation × classifier) selected on **validation**, then its test score
reported — 5 seeds, 12 candidate pipelines, 3 classifier families (LogReg, RandomForest, MLP).

| Phase | validation-selected pipeline | its test ROC | HINT (full) | diff (simple − HINT) |
|---|---|---|---|---|
| I | icd2vec + MLP | 0.5427 ± 0.0254 | 0.5740 | **−0.031** |
| II | chapter + MLP | 0.6121 ± 0.0073 | 0.6210 | **−0.009** |
| III | icd2vec + RandomForest | 0.6988 ± 0.0009 | 0.6851 | **+0.014** |

Mean: HINT is **0.009 ahead**. A simple pipeline on ICD codes alone comes within ~0.03 ROC
of the full deep architecture in every phase, and beats it on phase III — but the deep
model's small advantage can no longer be called zero.

**Neural-vs-neural control** (same classifier family as HINT):

| Phase | chapter one-hot + MLP | HINT | diff |
|---|---|---|---|
| I | 0.5664 ± 0.0335 | 0.5740 | +0.008 |
| II | 0.6121 ± 0.0073 | 0.6210 | +0.009 |
| III | 0.6324 ± 0.0179 | 0.6851 | **+0.053** |

On phase III the gap is ~3× the MLP's seed sd — real, not noise. The earlier "22 binary
features match the deep architecture" reading rested specifically on RandomForest.

#### ⚠️ Methodological note #4 — a weakened claim, and why

An earlier draft claimed **"22 binary features match or beat the full deep architecture,"**
based on chapter-one-hot + RandomForest (0.6763 vs HINT 0.6851 on phase III, and one-hot
*winning* on phase I).

That comparison selected the best of 12 simple pipelines **by looking at the test score** —
precisely the error this project criticises in MEXA (test-set epoch selection). Selecting
the pipeline on validation instead reverses the average: HINT +0.009 rather than −0.009.

Phase I shows why the discipline matters: validation picks `icd2vec+MLP` (val ROC 0.6475)
whose test ROC is 0.5427, while the test-best pipeline (`chapter+LogReg`, 0.6091) is not
selected. With only 117 validation trials, pipeline selection is itself unreliable.

**This is the fourth strong directional claim in this project weakened or retracted by
tightening the methodology** (after: the threshold hypothesis, the fixed-vs-buggy gap, and
the "bug flatters complex models" claim). The consistency of that pattern is itself the most
robust finding in this work.

---

#### ⭐ Simple baselines — how much do the deep architectures actually gain?

Classical models trained on the same MEXA embeddings (ICD-64 + SMILES-15 + BioBERT-768,
mask-aware mean pooling) and pushed through the **same harness**.

| Phase | HINT | MEXA (best) | GradBoost | RandomForest | LogReg | TRIVIAL |
|---|---|---|---|---|---|---|
| I | 0.574 | 0.569 | 0.546 | 0.550 | 0.562 | 0.500 |
| II | 0.621 | 0.617 | 0.616 | 0.610 | 0.585 | 0.500 |
| III | 0.685 | 0.681 | 0.651 | 0.619 | 0.554 | 0.500 |

**HINT − GradBoost gap:** +0.028 (I) · +0.005 (II) · +0.034 (III)

| Phase | HINT CI | GradBoost CI | verdict |
|---|---|---|---|
| II | [0.593, 0.646] | [0.588, 0.642] | **overlap → indistinguishable** |
| III | [0.647, 0.720] | [0.615, 0.688] | **overlap → indistinguishable** |

> **Conclusion:** in *no phase* does HINT significantly outperform gradient boosting. The
> largest gap is +0.034 on phase III, and even there the confidence intervals overlap. The
> contribution of the deep architectures lies within measurement uncertainty relative to a
> tree ensemble on the same embeddings.

**LogReg pathology on phase III:** ROC 0.554 but F1 0.857 (= TRIVIAL) at threshold ≈ 0.
With 1615 dimensions and 3094 samples the linear model degenerates into predicting
everything positive. The fair strongest baseline on phase III is therefore GradBoost,
not LogReg.

> ⚠️ **Caveat:** the baselines use **MEXA's embeddings**, not HINT's own encoders (D-MPNN,
> GRAM, BioBERT). A fully matched comparison would also run GradBoost on HINT's own
> features. The baselines are also single-seed (sklearn `random_state` fixed).

#### ⭐ HINT's own ablation ladder (phase III, epochs = 5, single seed)

Fresh encoders per variant, HINT's own `learn()` (validation-loss selection), then the
shared harness.

| Variant | ROC-AUC | PR-AUC (correct) | PR-AUC (broken) | diff |
|---|---|---|---|---|
| Only_Molecule (drug only) | 0.540 [0.501, 0.579] | 0.7800 | 0.7590 | +0.021 |
| HINT_nograph | 0.695 [0.658, 0.727] | 0.8585 | 0.7996 | +0.059 |
| **Only_Disease (ICD only)** | **0.700** [0.663, 0.735] | **0.8647** | 0.8031 | +0.062 |
| HINTModel (full) | 0.708 [0.672, 0.741] | 0.8627 | 0.8154 | +0.047 |
| Interaction | 0.711 [0.674, 0.743] | 0.8588 | 0.8111 | +0.048 |
| TRIVIAL | 0.500 | 0.750 | — | — |

**Finding A — the ladder is flat. REPLICATED ACROSS ALL THREE PHASES.**

| Phase | Only_Disease (ICD only) | HINTModel (full) | diff |
|---|---|---|---|
| I | 0.5734 | 0.5622 | **−0.011** (disease ahead) |
| II | 0.6257 | 0.6177 | **−0.008** (disease ahead) |
| III | 0.6996 | 0.7083 | +0.009 (full ahead) |

In **two of three phases the ICD-codes-only variant beats the full model**; in the third
the gap is +0.009, within noise. HINT's graph, risk module and multimodality contribute
nothing measurable in any phase. The molecule branch alone is weakest (0.548 on phase II,
0.540 on phase III).

> **All predictive signal comes from the ICD codes** — not drug structure, not protocol
> text, not the interaction graph. Independently confirmed in all three phases.

**Finding B — the broken PR-AUC distorts rankings, but UNPREDICTABLY.**

Spearman correlation between the correct and broken rankings:

| Phase | Spearman ρ | HINTModel rank shift |
|---|---|---|
| I | **0.30** (ranking nearly scrambled) | 2 → 5 (**penalised**) |
| II | 0.90 | 2 → 2 (neutral) |
| III | 0.70 | 2 → 1 (**flattered**) |

The bug's magnitude varies by model (+0.020 … +0.081), so it does perturb rankings — but
the **direction is inconsistent**: it penalises the complex model on phase I and flatters
it on phase III.

> **Conclusion:** the broken PR-AUC injects **unpredictable noise** into ablation rankings.
> No ablation conclusion drawn with this metric is trustworthy — but there is no systematic
> "complexity bias".

#### ⚠️ Methodological note #3 — a retracted claim

The previous round, looking only at phase III, asserted that **"the bug systematically
flatters the complex model."** Adding phases I and II showed the direction is inconsistent:
on phase I HINTModel drops *three* ranks. **The claim is retracted.**

This is the **third** directional claim in this project that failed to replicate (the
others: the threshold hypothesis, the fixed-vs-buggy gap). The common cause is the same:
the differences under comparison are smaller than the noise, so any pattern seen in a
single run or a single phase is most likely noise.

**New bug (fourth):** `Only_Molecule` and `Only_Disease` never passed `device` to
`super().__init__()`, which `Interaction` requires. Both ablation variants therefore
**could not be instantiated at all** (`TypeError`). Fixed.

#### Items 3 and 5 replicate across all phases

| Phase | PR-AUC correct | PR-AUC broken | diff | F1 fixed@0.5 | F1 val-calibrated |
|---|---|---|---|---|---|
| I | 0.6359 | 0.5739 | +0.0620 | 0.5934 | 0.7112 |
| II | 0.6724 | 0.6097 | +0.0627 | 0.6367 | 0.6869 |
| III | 0.8517 | 0.7995 | +0.0522 | 0.8123 | 0.8263 |

The PR-AUC bug consistently understates by ~0.05–0.06 — not a single-phase artefact.
Threshold calibration matters far more on phase I (+0.118).

---

### 3.5b Phase III detail (converged, 3 seeds)

Learning rate was swept identically for both arms; lr = 1e-2 won in both. All six runs
converged via early stopping. Epochs selected on validation; test never influenced selection.

| Arm | test ROC-AUC | test PR-AUC | test F1 |
|---|---|---|---|
| **FIXED** (cross-attention trained) | 0.6776 ± **0.0023** | 0.8531 ± 0.0036 | 0.8508 ± 0.0039 |
| **BUGGY** (attention frozen, original) | 0.6807 ± 0.0156 | 0.8542 ± 0.0109 | 0.8393 ± 0.0144 |

**Difference = +0.0031 (favouring buggy), Welch p = 0.764.**

#### Principal finding
**Training the cross-attention has no measurable effect on performance.** MEXA's
mode-expert mechanism — the paper's central contribution — is **inert**: the model performs
identically when those layers are frozen at random initialisation and never trained at all.
Whatever predictive ability the model has comes not from the claimed interaction mechanism
but from its remaining, much simpler components (input projections + compensation
self-attention + prediction head).

This is the first time the paper's core claim has been tested — and it was **not supported**.

#### Secondary finding — the mechanism's real benefit is reproducibility
The fixed arm's standard deviation is **one seventh** of the buggy arm's (0.0023 vs 0.0156
on test; 0.0024 vs 0.0114 on validation). Training the cross-attention does not improve
accuracy, but it markedly improves run-to-run stability — frozen random layers are far more
seed-sensitive. The mechanism does have a benefit; it is simply not the claimed one.

#### Final comparison

| Model | test ROC-AUC |
|---|---|
| HINT | 0.6851 [0.647, 0.720] |
| MEXA-buggy, properly retrained | 0.6807 ± 0.0156 |
| MEXA-fixed, properly retrained | 0.6776 ± 0.0023 |
| MEXA released checkpoint | 0.519 [0.484, 0.553] |
| TRIVIAL (all positive) | 0.500 |

- **Under a fair protocol MEXA and HINT are statistically indistinguishable** — HINT's
  confidence interval overlaps both MEXA arms comfortably. The "4.9% improvement" claim does
  not survive.
- **The released checkpoint is far below its own architecture** (0.519 vs ≈0.68).
- Even a properly retrained MEXA remains **below the trivial baseline on F1** (0.85 vs 0.857).

---

### 3.6 Retraining — initial (unconverged) sweep, kept for the record

Learning rate was swept identically for both arms (5e-2, 1e-2, 1e-3, 1e-4); lr = 1e-2 won
in both cases. Three observations:

1. **A properly retrained MEXA is genuinely competitive** — even with frozen attention it
   reaches ROC 0.6975, above HINT's 0.6851. The architecture is defensible.
2. **The released checkpoint is far worse than its own architecture properly trained**
   (0.52 vs 0.6975, ≈0.18 ROC). The published weights were poorly selected.
3. **Training the cross-attention does not help — it appears to hurt** (fixed 0.6572 vs
   buggy 0.6975), consistently on both validation and test. The paper's central claim has
   now been tested for the first time and was **not supported**.

> ⚠️ **These are not yet conclusions.** All runs stopped at 10 epochs, and for every top run
> the best epoch *was* the last epoch — i.e. they were still improving. We are comparing
> unconverged curves from a single seed.

---

## 4. Known gaps and limitations

**Not yet established:**

- **Phases I and II.** All results above are phase III only. The decisive fixed-vs-buggy
  comparison should be repeated there.
- **Statistical power.** n = 3 seeds per arm. The result is a clear null (p = 0.76), but a
  larger n would tighten the equivalence claim.
- **Other hyperparameters.** Only the learning rate was swept. `rho1`/`rho2` (Cauchy and
  NT-Xent weights) and `threshold` were left at their published values; the routing
  threshold in particular interacts directly with the full-masking bug.
- **Other baselines.** Item 5 implies every TOP-benchmark baseline should be re-measured
  with correct PR-AUC; only HINT has been redone.

**Methodological caveats:**

- **Distribution shift between splits.** Validation is 66.6% positive, test is 75.0%. The
  validation-selected threshold therefore underperforms the test-oracle threshold. The
  protocol is correct (test must not be inspected), but this shift should be reported.
- **Criteria parsing degradation.** MEXA's `find_pattern` falls back to `pattern_5` (collapse
  everything into one blob) for trials matching no known template. A small fraction of trials
  therefore have degraded criteria structure.
- **The guard changes the architecture slightly.** Keeping one token unmasked is a minimal,
  intent-preserving intervention, but it *is* a modification. It is applied to both arms.
- **`icd2vec` is stochastic** and its author now discourages node2vec for ICD representation.
  We keep it deliberately, for faithfulness to MEXA's inputs.

---

## 5. Future plans

**Completed:**

| Item | Status |
|---|---|
| Decisive fixed-vs-buggy experiment | ✅ all three phases; null confirmed (Δ = 0.0004, p = 0.96 on phase III with 7 seeds) |
| `rho1` / `rho2` / `threshold` sweep | ✅ phase III; threshold effect refuted at 3 seeds, NT-Xent inert |
| Re-measure TOP baselines with corrected PR-AUC | ✅ 5 HINT variants + 3 classical models, all three phases |
| Seed count | ⚠️ phase III at 7 seeds; phases I and II still at 3 |

**Next:**

1. **ICD representation ablation** (highest value — follows directly from Finding A).
   Since all predictive signal comes from the ICD codes, the decisive question is whether
   the *representation* of those codes matters at all. `icd_representation_ablation.py`
   compares MEXA's published icd2vec (64-d node2vec) against multi-hot encodings at three
   granularities (chapter ≈ 23 dims, 3-char category ≈ 662, full code ≈ 2276), same
   classifier and same harness. Three outcomes are distinguishable:
   - icd2vec ≫ one-hot → the embedding method matters; modern embeddings worth trying
   - icd2vec ≈ one-hot → the embedding method is irrelevant; **the problem, not the
     architecture or the inputs, is the limiting factor**
   - chapter ≈ full code → the signal is coarse: only "which disease family" matters
2. Raise phases I and II to 7 seeds (mechanical; tightens the equivalence claim).
4. Re-measure all TOP-benchmark baselines with the corrected PR-AUC (Item 5's broadest
   consequence — HINT is the benchmark's reference model, so the broken figure may have
   propagated).
5. **Modern ICD embeddings ablation.** MEXA's disease representation relies on a method its
   own library author no longer recommends. Same model, different disease embedding — a
   clean ablation that asks whether MEXA is limited by its architecture or its inputs.

**Longer term:**

3. **Modern LM-based ICD embeddings** — only worth doing if item 1 shows the representation
   matters at all. If icd2vec ≈ one-hot, a fancier embedding will not help either.

4. **Hybrid model (Step 6) — DROPPED.** The original plan was HINT encoders + MEXA's
   mode-expert fusion. Every architectural component of MEXA was shown to be inert
   (cross-attention, routing, NT-Xent), and HINT's own ablation ladder is flat
   (ICD-codes-only matches the full model in all three phases). Grafting an inert mechanism
   onto a flat ladder has no motivation. **This direction is dropped, and that is itself a
   result** — reported explicitly rather than quietly abandoned.

5. **Paper.** Fair comparison table; before/after for each of the six bugs; the
   two-bugs-concealing-each-other finding; the F1-is-uninformative result; and the
   methodological spine — *on this benchmark seed noise exceeds the effect sizes being
   claimed*, evidenced by three directional findings that dissolved under replication.

---

## 6. File guide

### `paper/` — orchestration layer (new)
| File | Purpose |
|---|---|
| `config.py` | Single point of configuration: phase, seed, paths, protocol. |
| `eval_harness.py` | **Shared evaluator** — threshold@validation, ROC/PR on scores, bootstrap CI, trivial baseline. |
| `dump_hint_preds.py` | HINT checkpoint → raw valid+test scores (non-invasive; uses `generate_predict`). |
| `dump_mexa_preds.py` | MEXA checkpoint → raw scores. Auto-detects checkpoint type (fixed vs original). |
| `build_results_csv.py` | Per-trial CSV: metadata + prediction + correctness. |
| `build_mexa_data.py` | Builds MEXA's embedded data from HINT's CSVs (faithful / custom modes). |
| `check_mexa_data.py` | Validates generated data against MEXA's own dataset class + forward pass. |
| `fix_icd_shape.py` | Idempotent repair of the ICD tensor shape (avoids recomputing BioBERT). |
| `seed_sensitivity.py` | Measures metric variation across random initialisations. |
| `collect_runs.py` | Aggregates all training runs; selects best epoch by validation. |
| `main.py` | `# %%` control panel — blocks 0–12 plus Step 6. |

### Modified repository files (surgical, all marked with comments)
| File | Change |
|---|---|
| `mexa/main.py` | Item 1: validation loader, validation-based selection, early stopping. Item 2: `--no_moduledict_fix` toggle, sanity check. Plus divergence guard and `--skip_test_eval`. |
| `mexa/ctp/models_nlayers.py` | Item 2: three `dict` → `nn.ModuleDict` (toggleable) + `sanity_check_parameters()`. Plus full-mask guard, CPU portability fix, `squeeze(-1)` fix. |
| `hint/HINT/model.py` | D7: pass `device` through the two ablation constructors. |
| `hint/HINT/learn_phaseIII.py` | D10: explicitly allow loading the released full-object checkpoint under PyTorch ≥2.6. |

These study changes are published without vendoring either upstream tree:
`../patches/README.md` maps them to the audited commits and provides
`git apply --check` instructions. HINT scores are still extracted and evaluated
by the external shared harness; the HINT patch only removes the D7/D10 runtime
blockers.

---

## 7. Reproducibility record

| Setting | Value |
|---|---|
| Seed | 2023 |
| ICD-10-CM version | 2024 |
| SMILES embedding | MACAW, 15 dimensions, `random_state=2023` |
| ICD embedding | icd2vec / icdcodex, 64 dimensions |
| Criteria embedding | BioBERT `dmis-lab/biobert-v1.1`, 768 dimensions |
| Splits | phase III: train 3094 / valid 344 / test 1146 (identical to HINT's splits) |
| Bootstrap | 1000 resamples, 95% CI |

**Verified in this environment:** ModuleDict fix (parameter visibility + gradient updates),
full-mask NaN chain and its repair, CPU forward pass, checkpoint auto-detection, harness
behaviour on real data, data schema compatibility, custom embedders.

**Requires the local environment:** faithful embedding generation (MACAW / icd2vec /
BioBERT) and all model training.
