# Study patches for upstream HINT and MEXA-CTP

This directory publishes only the changes made for this study. It does not
vendor either third-party repository.

## Audited revisions

| Patch | Upstream repository | Base commit | Defects / protocol changes |
|---|---|---|---|
| `hint-8dc0497-study-fixes.patch` | [HINT](https://github.com/futianfan/clinical-trial-outcome-prediction) | `8dc0497f23fdb84e2905da7655924a91e6e79798` | D7: pass `device` to the two ablation constructors; D10: opt in to full-object checkpoint loading on PyTorch >=2.6 |
| `mexa-b898dbb-study-fixes.patch` | [MEXA-CTP](https://github.com/murai-lab/MEXA-CTP) | `b898dbb41930197187fe3867e3616ffd406a48a6` | D1: register attention/router modules; D2: prevent all-token masking; D6: portable tensor placement; D8: explicit divergence handling; D9: preserve the batch dimension; validation-only checkpoint selection, patience, and an A/B compatibility flag |

The patches intentionally exclude checkpoints, generated caches, `DataPath/`,
training outputs, bytecode, and unrelated local changes. The explanatory
comments in the patches are part of the study record.

## Apply

Clone each upstream repository, check out the audited commit, and apply the
corresponding patch from this repository:

```bash
git clone https://github.com/futianfan/clinical-trial-outcome-prediction.git hint
git -C hint checkout 8dc0497f23fdb84e2905da7655924a91e6e79798
git -C hint apply --check /path/to/top-benchmark-reeval/patches/hint-8dc0497-study-fixes.patch
git -C hint apply /path/to/top-benchmark-reeval/patches/hint-8dc0497-study-fixes.patch

git clone https://github.com/murai-lab/MEXA-CTP.git mexa
git -C mexa checkout b898dbb41930197187fe3867e3616ffd406a48a6
git -C mexa apply --check /path/to/top-benchmark-reeval/patches/mexa-b898dbb-study-fixes.patch
git -C mexa apply /path/to/top-benchmark-reeval/patches/mexa-b898dbb-study-fixes.patch
```

For MEXA-CTP, the corrected behavior is the default. Pass
`--no_moduledict_fix` only to reproduce the original unregistered-module arm.
`--patience` controls validation-based early stopping, while
`--skip_test_eval` avoids test evaluation during training without changing
model selection.

HINT's D3/D4 metric and threshold corrections are implemented in this
repository's shared evaluation harness rather than by changing upstream HINT.
D5 is evidenced by the upstream six-byte empty cache and quantified using the
released `paper/preds/preds_hint_criteria_phase_III.csv`.
