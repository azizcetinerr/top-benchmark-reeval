# Quantifying defect D5 (HINT's empty criteria-embedding cache)

D5: `hint/data/sentence2embedding.pkl` ships as an empty dict, so `protocol2feature` feeds a
zero 768-vector to HINT's eligibility-criteria branch for every trial. The paper currently
**documents** this and does not quantify it. These steps quantify it on the cluster (g124),
then `quantify_d5.py` reports the effect. Nothing here needs a rewrite of the paper's other
numbers — we work on copies and restore the shipped state at the end.

Two routes:
- **Route A (quick, inference-only):** regenerate the cache, re-score the *released* HINT
  checkpoint with populated criteria. Answers: is the shipped checkpoint degraded at eval by
  the empty cache?
- **Route B (definitive):** retrain HINT with the populated cache. Answers: how much does a
  properly-fed criteria modality change HINT, and how much of the published MEXA>HINT gap it
  accounts for. Heavier (a full HINT training run).

---

## 0. Environment (g124)

```bash
ssh lir-grove && salloc --nodelist=g124 --gres=gpu:1 -c 8 --mem=64G --time=03:00:00
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate hint   # HINT's env (torch)
export DATA_FOLDER=$HOME/ctp/mexa/DataPath/
cd ~/ctp/hint
pip install biobert-embedding tqdm 2>/dev/null    # if missing (BioBERT weights auto-download)
```

## 1. Back up the shipped state (so the paper stays reproducible)

```bash
cd ~/ctp/hint
cp data/sentence2embedding.pkl data/sentence2embedding.EMPTY.pkl        # 6-byte original
cp ~/ctp/paper/preds/preds_hint_phase_III.csv ~/ctp/paper/preds/preds_hint_phase_III.EMPTYCACHE.csv
python -c "import pickle; print('cache len before:', len(pickle.load(open('data/sentence2embedding.pkl','rb'))))"  # 0
```

## 2. Regenerate the BioBERT criteria cache

```bash
cd ~/ctp/hint
python -c "import sys; sys.path.insert(0,'HINT'); from protocol_encode import save_sentence_bert_dict_pkl as g; g()"
python -c "import pickle; print('cache len after:', len(pickle.load(open('data/sentence2embedding.pkl','rb'))))"  # >0 now
```

If `save_sentence_bert_dict_pkl` errors on the `biobert_embedding` import, use the repo's
`dmis-lab/biobert-v1.1` via `transformers` instead (the paper's stated encoder); the cache
just needs a `{sentence: 768-vector}` dict at `data/sentence2embedding.pkl`.

## 3. Route A — re-score the released checkpoint with the populated cache

```bash
cd ~/ctp/paper
CTP_PHASE=III python run_hint_phase.py        # dumps preds/preds_hint_phase_III.csv (now with criteria)
mv preds/preds_hint_phase_III.csv preds/preds_hint_criteria_phase_III.csv
# restore the shipped (empty-cache) predictions so all other tables stay unchanged:
cp preds/preds_hint_phase_III.EMPTYCACHE.csv preds/preds_hint_phase_III.csv
```

## 4. Route B — retrain HINT with the populated cache (definitive)

```bash
cd ~/ctp/hint
CTP_PHASE=III python HINT/learn_phaseIII.py           # retrain with criteria alive; writes a new ckpt
# then dump its scores the same way and rename to preds_hint_criteria_phase_III.csv (as in step 3)
```

## 5. Quantify

```bash
cd ~/ctp/paper
python quantify_d5.py --phase III
```

Outputs, on the common test set:
- `HINT(cache) − HINT(empty)` on ROC-AUC and PR-AUC (paired bootstrap, 2000): the direct cost
  of D5.
- `HINT(empty) − MEXA` vs `HINT(cache) − MEXA`: whether restoring the criteria branch narrows
  the published MEXA>HINT gap.

## 6. Restore shipped state

```bash
cd ~/ctp/hint && cp data/sentence2embedding.EMPTY.pkl data/sentence2embedding.pkl   # back to 6-byte empty
```

## What to write in the paper afterwards

- If `HINT(cache) − HINT(empty)` is ~0: the shipped checkpoint never learned to use criteria
  (trained on zeros too), so D5 is a modality that was dead in training and eval alike — a
  clean statement, and the ablation-ladder interpretation stands with the caveat already added.
- If `HINT(cache) − HINT(empty)` is large and positive: D5 materially depressed HINT in every
  public re-run, and a share of the published MEXA>HINT gap is attributable to it — promote the
  D5 limitation to a quantified result and update Section~4.1.
