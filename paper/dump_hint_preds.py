# -*- coding: utf-8 -*-
"""
paper/dump_hint_preds.py — Dump HINT raw scores
==================================================
Runs WITHOUT MODIFYING the HINT repo: loads the ready checkpoint, calls
model.generate_predict() on the val and test sets (returns sigmoid
probabilities) and writes preds/preds_hint_phase_{X}.csv as
{nctid, score, label, split}.

Why generate_predict? Because it is a clean public method: it returns (loss,
predict_all, label_all, nctid_all) and predict_all is already a torch.sigmoid
output. This way, without touching HINT's broken evaluation()/bootstrap_test()
at all, we just get the raw scores — evaluation is done in the common harness.
"""
import os
import sys
import csv


def dump_hint(phase, ckpt_path, valid_csv, test_csv, out_csv, hint_dir):
    # import the HINT modules in their own working directory
    cwd = os.getcwd()
    sys.path.insert(0, hint_dir)
    os.chdir(hint_dir)                      # for relative paths inside the ckpt
    try:
        import torch
        from HINT.dataloader import csv_three_feature_2_dataloader

        valid_loader = csv_three_feature_2_dataloader(valid_csv, shuffle=False, batch_size=32)
        test_loader  = csv_three_feature_2_dataloader(test_csv,  shuffle=False, batch_size=32)

        # weights_only=False is required for torch>=2.6 (ckpt is a full model object)
        model = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        model.eval()

        rows = []
        for split, loader in [('valid', valid_loader), ('test', test_loader)]:
            _, predict_all, label_all, nctid_all = model.generate_predict(loader)
            for nctid, score, label in zip(nctid_all, predict_all, label_all):
                rows.append((nctid, float(score), int(label), split))
            print(f'[HINT] {split}: {len(label_all)} samples')
    finally:
        os.chdir(cwd)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['nctid', 'score', 'label', 'split'])
        w.writerows(rows)
    print(f'[HINT] written -> {out_csv} ({len(rows)} rows)')
    return out_csv


if __name__ == '__main__':
    import config as C
    dump_hint(C.PHASE, C.HINT_CKPT, C.HINT_VALID_CSV, C.HINT_TEST_CSV,
              C.HINT_PREDS, C.HINT_DIR)
