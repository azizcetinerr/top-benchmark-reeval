# Generating MEXA's training data — two routes

MEXA's embedded data (`DataPath/{trainphase,validphase,test}/phase_X/{nctid}.pkl`) is not
distributed with the repository. `build_mexa_data.py` generates it from **HINT's `data/`
CSVs**. The inputs are compatible because both models derive from the same TOP benchmark,
and MEXA's `Embedding` base class reads exactly the columns HINT's CSVs provide
(`nctid`, `label`, `icdcodes`, `smiless`, `criteria`).

Choose whichever route's dependencies install in your environment.

## Common to both routes
```bash
pip install torch transformers      # BioBERT criteria embeddings (768-d) are required either way
```
> BioBERT (`dmis-lab/biobert-v1.1`) downloads from HuggingFace on first run (~400 MB).
> Behind a restrictive proxy you may need to pre-cache the model.

## Route 1 — faithful (scientifically correct comparison)
Uses MEXA's original embedders, so MEXA's numbers remain comparable both to its own paper
and to HINT.
```bash
pip install macaw icdcodex          # SMILES (15-d) + ICD (64-d)
export MEXA_BUILD_MODE=faithful
python paper/build_mexa_data.py
```
- `macaw`: SMILES → 15 dimensions, with **`random_state=2023` (Item 4 fix)** for
  reproducibility. MACAW uses RDKit internally, so `MorganGenerator DEPRECATION` warnings
  are expected and harmless.
- `icdcodex`: ICD-10-CM hierarchy → 64 dimensions.

### A note on the ICD version
`icdcodex`'s error message is misleading — it says "2019 to 2025" but rejects 2025, because
the upper bound is exclusive. The script no longer hard-codes a year: it finds the newest
working version automatically and **prints which one it used**. Record that version, since
changing it changes the embeddings. To force a specific version:
```bash
export ICD_VERSION=2024
```

## Route 2 — custom (lighter; valid only for the internal A/B)
Use this only for the internal *fixed vs buggy cross-attention* comparison, where both arms
share identical inputs. It is **not valid** for absolute MEXA-vs-HINT comparison, because
the inputs differ from the published model.
```bash
pip install rdkit                   # SMILES Morgan fingerprints → 15-d (deterministic)
export MEXA_BUILD_MODE=custom
python paper/build_mexa_data.py
```
- SMILES: RDKit Morgan fingerprint, modular-folded to 15 dimensions.
- ICD: deterministic per-code hash → 64 dimensions.
- Criteria: still BioBERT (a 768-d text encoder is unavoidable).

## Output schema
Each `{nctid}.pkl` contains:

| Key | Shape / type |
|---|---|
| `nctid` | `str` |
| `label` | `int` (0/1) |
| `icdcodes` | list of 2-D tensors `[n_codes, 64]`, one per disease entry |
| `smiless` | list of 1-D tensors `[15]`, one per molecule |
| `criteria` | `[[inclusion_emb, mask], [exclusion_emb, mask]]`, BioBERT 768-d |

## After generation

**1. Repair the ICD tensor shape** (only needed for data built before the fix; idempotent
and does not touch the expensive BioBERT embeddings):
```bash
cd paper
python fix_icd_shape.py
```

**2. Validate** against MEXA's own dataset class and a real forward pass:
```bash
python check_mexa_data.py III        # expect: itokens (5, 64) OK ... forward OK
```

**3. Point `config.py`'s `MEXA_DATA_FOLDER` at the generated `DataPath/`, then run the
Item 2 A/B experiment:**
```bash
cd mexa
export DATA_FOLDER=/path/to/mexa/DataPath/

python main.py --job train --phase III --device cpu --epochs 100 --patience 10 --lr 1e-2 --skip_test_eval
python main.py --job train --phase III --device cpu --epochs 100 --patience 10 --lr 1e-2 --skip_test_eval --no_moduledict_fix
```
Each run prints `sanity_check_parameters` at startup, showing whether the cross-attention
is registered and trainable. Note that the published `lr=5e-2` **diverges** once attention
actually trains — the learning rate must be re-tuned on validation, identically for both arms.

**4. Aggregate and compare:**
```bash
cd paper
python collect_runs.py               # selects best epoch by validation for every run
```
Then use `main.py` block 3 (dump MEXA scores) and block 11 (fair table).

## Verification status
- CSV parsing (`parse_smiless`, `parse_icdcodes`): tested on real HINT data ✓
- Custom embedders: correct dimensions and determinism verified ✓
- Output schema: validated through MEXA's dataset class + forward pass ✓
- Faithful embedders (MACAW / icd2vec / BioBERT): require your environment; verify there.
