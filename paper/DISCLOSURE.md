# Responsible-disclosure record

This file backs the "Responsible disclosure" paragraph in the paper. It contains
the filed issue text, issue URLs, and exact audited revisions.

Audited repositories:
- HINT: `https://github.com/futianfan/clinical-trial-outcome-prediction` — commit `8dc0497f23fdb84e2905da7655924a91e6e79798` (audited 2026-07)
- MEXA-CTP: `https://github.com/murai-lab/MEXA-CTP` — commit `b898dbb41930197187fe3867e3616ffd406a48a6` (audited 2026-07)

---

## Issue 1 — HINT repository

**Title:** Empty `sentence2embedding.pkl` silently zeroes the eligibility-criteria modality; two evaluation-metric bugs

**Body:**

> Thanks for releasing HINT and the TOP benchmark — we've been building on it and found a few
> reproducibility issues we wanted to report so they can be fixed for everyone downstream. None
> of these implies anything about the model design; they are code/data-shipping issues in the
> released repo.
>
> **1. `data/sentence2embedding.pkl` ships empty (6-byte empty dict).**
> The eligibility-criteria branch looks up each trial's BioBERT sentence embedding in this
> cache; on a miss it returns a zero 768-vector. Because the shipped file is an empty dict,
> *every* trial gets a zero criteria embedding, so the criteria modality is silently inactive
> for anyone who runs the repo as shipped. Repro: `pickle.load(open('data/sentence2embedding.pkl','rb'))`
> returns `{}`. Suggested fix: regenerate and ship the cache, or fail loudly on an empty cache.
>
> **2. PR-AUC is computed on binarised predictions.** `average_precision_score` is called on
> thresholded 0/1 predictions rather than probabilities, which understates PR-AUC and can
> reorder variants. Suggested fix: pass probabilities.
>
> **3. F1 uses a hard-coded 0.5 threshold.** On a test set with ~0.75 positives this makes F1
> track the class prior. Suggested fix: select the threshold on validation, or report F1 at a
> stated operating point alongside a trivial all-positive baseline.
>
> **4. (minor) Two ablation variants are un-instantiable** because `device` is missing from the
> `super().__init__` call, and checkpoint loading fails under PyTorch ≥2.6 (`weights_only`
> default change).
>
> Happy to send a PR for any of these. Commit audited: `8dc0497f23fdb84e2905da7655924a91e6e79798`.

---

## Issue 2 — MEXA-CTP repository

**Title:** Mode-experts attention/router params stored in a plain dict never enter the optimizer (never trained)

**Body:**

> Thanks for releasing MEXA-CTP. While reproducing it we found the mode-experts cross-attention
> mechanism is not actually trained as shipped, plus a couple of related runtime issues. Flagging
> them for a fix.
>
> **1. Attention/router parameters live in a plain Python `dict`, not a `nn.Module` container.**
> As a result they are not registered parameters: they are invisible to the optimizer (0 of 264
> tensors receive gradients), are unchanged after a backward+step, and are absent from every
> saved checkpoint. The published result is produced with these parameters frozen at
> initialisation. Repro: count `len(list(model.parameters()))` vs. the tensors in the dict, and
> diff a parameter before/after one `optimizer.step()`. Suggested fix: register them via
> `nn.ModuleDict`/`nn.ParameterDict` or module attributes.
>
> **2. Full-masking produces NaN.** At every learning rate that actually trains, full sequence
> masking yields NaN; with a minimal one-token guard, training survives but every sequence
> collapses to a single token. Suggested fix: keep at least one unmasked token / guard the
> softmax denominator.
>
> **3. Published lr 5e-2 diverges once the attention params are actually trained** — it only
> "worked" because those params were frozen (issue 1). **4. `.to(get_device())` hard-codes GPU,**
> so the model cannot run on CPU. **5. `squeeze()` collapses the batch dimension when B=1.**
>
> Note on evaluation: because the trained-mechanism ablation shows the sign opposite to a useful
> mechanism (buggy ≥ fixed) and the only powered comparison is not significant, we are *not*
> claiming your reported numbers are wrong — only that the mechanism as shipped is inert and the
> comparison to HINT is confounded by the metric bugs above. Details in our paper; happy to
> share the preprint and send PRs. Commit audited: `b898dbb41930197187fe3867e3616ffd406a48a6`.

---

## Filing record (complete before submission)

| Repo | Defects | Issue URL | Date filed | Commit audited |
|------|---------|-----------|------------|----------------|
| HINT | D3, D4, D5, D7, D10 | https://github.com/futianfan/clinical-trial-outcome-prediction/issues/15 | 2026-07-29 | `8dc0497f23fdb84e2905da7655924a91e6e79798` |
| MEXA-CTP | D1, D2, D6, D8, D9 | https://github.com/murai-lab/MEXA-CTP/issues/2 | 2026-07-29 | `b898dbb41930197187fe3867e3616ffd406a48a6` |
