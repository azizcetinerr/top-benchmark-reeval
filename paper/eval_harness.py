# -*- coding: utf-8 -*-
"""
paper/eval_harness.py — Common evaluation harness (SINGLE source of truth)
============================================================================
Both models (HINT, MEXA) pass their raw scores through this code. This way
the "two repos make the same mistake in different ways" problem is fixed at
the root: evaluation is now done by this single module, not by the repos.

Fixes included:
  * Item 3 — Threshold is NOT fixed at 0.5; it is selected to maximize F1 on
             the VALIDATION set. The test score is reported at this threshold.
  * Item 5 — PR-AUC and ROC-AUC are computed on CONTINUOUS scores, NOT binarised 0/1
             scores (ranking metrics).
  * Bootstrap confidence interval is produced with the SAME protocol for both models.

Score scale does not matter: ROC/PR ranking metrics are insensitive to
monotonic transformation (HINT gives sigmoid probabilities, MEXA gives logits —
both are valid). Since the threshold is selected per model on its OWN
validation scores, the scale difference is not a problem.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)


# ----------------------------------------------------------------------
# 1) Read raw scores
# ----------------------------------------------------------------------
def load_preds(csv_path):
    """CSV with columns {nctid, score, label, split} -> DataFrame."""
    df = pd.read_csv(csv_path)
    need = {'nctid', 'score', 'label', 'split'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'{csv_path}: missing column(s): {missing}')
    df['label'] = df['label'].astype(int)
    df['score'] = df['score'].astype(float)
    return df


def split_scores(df, split):
    sub = df[df['split'] == split]
    return sub['score'].to_numpy(), sub['label'].to_numpy()


# ----------------------------------------------------------------------
# 2) Item 3 — Threshold selection (maximize F1 on VALIDATION)
# ----------------------------------------------------------------------
def select_threshold(scores, labels, metric='f1'):
    """Scans candidate thresholds and returns the one that maximizes the
    metric on the given (validation) set. Candidate thresholds = midpoints
    between unique scores."""
    order = np.argsort(scores)
    s = scores[order]
    uniq = np.unique(s)
    if len(uniq) == 1:
        return float(uniq[0])
    cands = (uniq[:-1] + uniq[1:]) / 2.0          # midpoints of adjacent scores
    cands = np.concatenate(([uniq[0] - 1e-6], cands, [uniq[-1] + 1e-6]))

    def score_at(t):
        pred = (scores >= t).astype(int)
        if metric == 'f1':
            return f1_score(labels, pred, zero_division=0)
        elif metric == 'precision':
            return precision_score(labels, pred, zero_division=0)
        elif metric == 'recall':
            return recall_score(labels, pred, zero_division=0)
        elif metric == 'acc':
            return accuracy_score(labels, pred)
        raise ValueError(metric)

    best_t, best_v = 0.5, -1.0
    for t in cands:
        v = score_at(t)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


# ----------------------------------------------------------------------
# 3) Metrikler
# ----------------------------------------------------------------------
def ranking_metrics(scores, labels):
    """Threshold-INDEPENDENT metrics — on continuous scores (Item 5 fix)."""
    return {
        'roc_auc': roc_auc_score(labels, scores),
        'pr_auc':  average_precision_score(labels, scores),   # <-- continuous score!
    }


def threshold_metrics(scores, labels, threshold):
    """Threshold-dependent metrics — at the selected threshold."""
    pred = (scores >= threshold).astype(int)
    return {
        'f1':        f1_score(labels, pred, zero_division=0),
        'precision': precision_score(labels, pred, zero_division=0),
        'recall':    recall_score(labels, pred, zero_division=0),
        'acc':       accuracy_score(labels, pred),
        'pred_pos_ratio': float(pred.mean()),
        'label_pos_ratio': float(labels.mean()),
    }


# ----------------------------------------------------------------------
# 4) Bootstrap confidence interval (SAME protocol for both models)
# ----------------------------------------------------------------------
def bootstrap_ci(scores, labels, threshold, n=1000, alpha=0.05, seed=2023):
    rng = np.random.default_rng(seed)
    N = len(scores)
    keys = ['roc_auc', 'pr_auc', 'f1']
    samples = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, N, N)          # sampling with replacement
        s, l = scores[idx], labels[idx]
        if l.min() == l.max():               # skip if only a single class remained
            continue
        rm = ranking_metrics(s, l)
        tm = threshold_metrics(s, l, threshold)
        samples['roc_auc'].append(rm['roc_auc'])
        samples['pr_auc'].append(rm['pr_auc'])
        samples['f1'].append(tm['f1'])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k in keys:
        arr = np.array(samples[k])
        out[k] = {
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'lo':   float(np.percentile(arr, lo)),
            'hi':   float(np.percentile(arr, hi)),
        }
    return out


# ----------------------------------------------------------------------
# 5) Top level: evaluate a single model with the fair protocol
# ----------------------------------------------------------------------
def evaluate_model(preds_csv, select_metric='f1', n_boot=1000, alpha=0.05, seed=2023):
    """Evaluates a model's dumped score file with the fair protocol:
       the threshold is selected on VALIDATION, metrics are reported on TEST
       at this threshold."""
    df = load_preds(preds_csv)
    v_s, v_l = split_scores(df, 'valid')
    t_s, t_l = split_scores(df, 'test')

    threshold = select_threshold(v_s, v_l, metric=select_metric)   # Madde 3

    point = {}
    point.update(ranking_metrics(t_s, t_l))                        # Madde 5
    point.update(threshold_metrics(t_s, t_l, threshold))
    ci = bootstrap_ci(t_s, t_l, threshold, n=n_boot, alpha=alpha, seed=seed)

    return {
        'threshold': threshold,
        'n_valid': len(v_l),
        'n_test': len(t_l),
        'point': point,     # point estimate on the test set
        'ci': ci,           # bootstrap confidence interval
    }


def baseline_metrics(labels):
    """TRIVIAL baselines — must be reported alongside every result.
    In this benchmark the test set is ~75% positive; even an empty classifier
    that says 'succeed for all' gets a high F1. Without the baseline shown, F1
    is misleading.
      - f1_all_pos : predict 1 for all -> F1 (majority line)
      - pr_auc     : PR-AUC of a random ranker = positive rate
      - roc_auc    : ROC-AUC of a random ranker = 0.5
    """
    labels = np.asarray(labels)
    pos_rate = float(labels.mean())
    allpos = np.ones_like(labels)
    return {
        'f1_all_pos': f1_score(labels, allpos, zero_division=0),
        'acc_all_pos': float((allpos == labels).mean()),
        'pr_auc': pos_rate,      # random-ranking PR-AUC floor
        'roc_auc': 0.5,
        'pos_rate': pos_rate,
    }


def compare_table(results_by_model, test_labels=None):
    """{'HINT': evaluate_model(...), 'MEXA': ...} -> side-by-side DataFrame."""
    rows = []
    for name, r in results_by_model.items():
        rows.append({
            'model': name,
            'threshold': round(r['threshold'], 4),
            'ROC-AUC': f"{r['ci']['roc_auc']['mean']:.3f} "
                       f"[{r['ci']['roc_auc']['lo']:.3f}, {r['ci']['roc_auc']['hi']:.3f}]",
            'PR-AUC':  f"{r['ci']['pr_auc']['mean']:.3f} "
                       f"[{r['ci']['pr_auc']['lo']:.3f}, {r['ci']['pr_auc']['hi']:.3f}]",
            'F1':      f"{r['ci']['f1']['mean']:.3f} "
                       f"[{r['ci']['f1']['lo']:.3f}, {r['ci']['f1']['hi']:.3f}]",
        })

    # --- TRIVIAL baseline row (without it F1 is misleading) ---
    # NOTE: output files/tables are ENGLISH (so they go directly into the paper).
    if test_labels is not None:
        b = baseline_metrics(test_labels)
        rows.append({
            'model': f"TRIVIAL (all-positive, pos={b['pos_rate']:.3f})",
            'threshold': '-',
            'ROC-AUC': f"{b['roc_auc']:.3f} (chance)",
            'PR-AUC':  f"{b['pr_auc']:.3f} (chance)",
            'F1':      f"{b['f1_all_pos']:.3f}",
        })
    return pd.DataFrame(rows)
