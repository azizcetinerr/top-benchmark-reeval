# -*- coding: utf-8 -*-
"""
paper/build_results_csv.py — Per-trial results CSV
======================================================
Produces a HUMAN-READABLE output showing, for each clinical trial, the model's
prediction and whether it was correct. This lets us inspect success one by one.

Merges two sources on nctid:
  1) Raw data CSV (metadata: diseases, icdcodes, drugs, smiless, criteria, phase)
  2) Dumped scores (preds_*.csv: nctid, score, label, split)

Output columns:
  nctid, phase, diseases, icdcodes, drugs, smiless, criteria,
  label(true), score(0-1), prediction(0/1 at threshold), correct(1/0)

The threshold is the one selected on validation in the common harness (Item 3).
"""
import csv
import os


def _read_raw_metadata(raw_csv):
    """Converts the raw HINT data file into a nctid -> {column: value} dict."""
    meta = {}
    with open(raw_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row['nctid']] = row
    return meta


def build_results(preds_csv, raw_csv, out_csv, threshold=0.5, split='test'):
    meta = _read_raw_metadata(raw_csv)

    out_cols = ['nctid', 'phase', 'diseases', 'icdcodes', 'drugs', 'smiless',
                'criteria', 'label', 'score', 'prediction', 'correct']
    n, n_correct, missing = 0, 0, 0

    with open(preds_csv, newline='') as f:
        preds = [r for r in csv.DictReader(f) if r['split'] == split]

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in preds:
            nctid = r['nctid']
            score = float(r['score'])
            label = int(r['label'])
            pred = int(score >= threshold)
            correct = int(pred == label)
            m = meta.get(nctid, {})
            if not m:
                missing += 1
            w.writerow({
                'nctid': nctid,
                'phase': m.get('phase', ''),
                'diseases': m.get('diseases', ''),
                'icdcodes': m.get('icdcodes', ''),
                'drugs': m.get('drugs', ''),
                'smiless': m.get('smiless', ''),
                'criteria': m.get('criteria', ''),
                'label': label,
                'score': round(score, 6),
                'prediction': pred,
                'correct': correct,
            })
            n += 1
            n_correct += correct

    acc = n_correct / n if n else 0.0
    print(f'[results] {out_csv}')
    print(f'[results] {split}: {n} trials | correct {n_correct} ({acc:.1%}) '
          f'| threshold {threshold:.4f}' + (f' | metadata missing: {missing}' if missing else ''))
    return out_csv


if __name__ == '__main__':
    import config as C
    from eval_harness import evaluate_model
    res = evaluate_model(C.HINT_PREDS, select_metric=C.SELECT_METRIC, seed=C.SEED)
    build_results(C.HINT_PREDS, C.HINT_TEST_CSV,
                  os.path.join(C.PAPER_DIR, f'results_hint_phase_{C.PHASE}.csv'),
                  threshold=res['threshold'], split='test')
