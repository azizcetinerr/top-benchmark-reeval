# -*- coding: utf-8 -*-
"""
paper/train_hint_variants.py — Train HINT's own ablation variants and
re-measure them with the COMMON HARNESS  (TASK 5 / broad leg of Item 5)
==========================================================================
WHY: HINT is the reference model of the TOP benchmark. The published scores of
its own ablation variants (FFNN, Interaction, HINT_nograph, Only_Molecule,
Only_Disease) may have been computed with a BROKEN PR-AUC. We re-measure them
all with the same protocol (threshold@validation, PR/ROC on continuous score,
bootstrap CI).

Variant ladder — each removes one component:
  Only_Molecule : drug only (molecule)
  Only_Disease  : disease only (ICD)
  Interaction   : three modalities, NO graph, NO risk
  HINT_nograph  : + risk module, NO graph
  HINTModel     : + dynamic attentive graph (full model)

HINT's own learn() selects the best model by VALIDATION LOSS — so the Item 1
protocol is already satisfied.

NOTE: HINT training is slow (MPNN + GRAM + BioBERT). It can take a long time per
variant on CPU. If you run it at the same time as MEXA training, they compete
for the CPU.

Usage:
    cd paper
    python train_hint_variants.py                       # config.PHASE, all variants
    CTP_PHASE=III python train_hint_variants.py --epochs 5
    python train_hint_variants.py --only HINTModel,Interaction
"""
import os
import sys
import csv
import argparse


VARIANTS = ['Only_Molecule', 'Only_Disease', 'Interaction', 'HINT_nograph', 'HINTModel']


def build_variant(name, encoders, device, prefix, epochs, lr):
    """Builds the variant according to its constructor signature."""
    from HINT.model import (Interaction, HINT_nograph, HINTModel,
                            Only_Molecule, Only_Disease)
    mol, dis, pro = encoders
    common = dict(molecule_encoder=mol, disease_encoder=dis, protocol_encoder=pro,
                  device=device, global_embed_size=50, highway_num_layer=2,
                  prefix_name=prefix, epoch=epochs, lr=lr, weight_decay=0)
    if name == 'HINTModel':
        return HINTModel(gnn_hidden_size=50, **common)
    return {'Interaction': Interaction, 'HINT_nograph': HINT_nograph,
            'Only_Molecule': Only_Molecule, 'Only_Disease': Only_Disease}[name](**common)


def dump_preds(model, loaders, out_csv):
    """Write val+test raw scores as {nctid,score,label,split} via generate_predict."""
    model.eval()
    rows = []
    for split, loader in loaders:
        _, predict_all, label_all, nctid_all = model.generate_predict(loader)
        rows += [(n, float(s), int(l), split)
                 for n, s, l in zip(nctid_all, predict_all, label_all)]
        print(f'    {split}: {len(label_all)} samples')
    model.train()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['nctid', 'score', 'label', 'split']); w.writerows(rows)
    return out_csv


def main(phase=None, epochs=5, lr=1e-3, only=None):
    import config as C
    import eval_harness as H

    phase = phase or C.PHASE
    wanted = [v.strip() for v in only.split(',')] if only else VARIANTS

    cwd = os.getcwd()
    sys.path.insert(0, C.HINT_DIR)
    os.chdir(C.HINT_DIR)                       # HINT works with relative paths
    try:
        import torch
        torch.manual_seed(C.SEED)
        os.makedirs('figure', exist_ok=True)

        from HINT.dataloader import csv_three_feature_2_dataloader
        from HINT.molecule_encode import MPNN
        from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
        from HINT.protocol_encode import Protocol_Embedding

        device = torch.device('cpu')
        base = f'phase_{phase}'
        tr = csv_three_feature_2_dataloader(f'data/{base}_train.csv', shuffle=True,  batch_size=32)
        va = csv_three_feature_2_dataloader(f'data/{base}_valid.csv', shuffle=False, batch_size=32)
        te = csv_three_feature_2_dataloader(f'data/{base}_test.csv',  shuffle=False, batch_size=32)

        print(f'phase {phase} | epochs {epochs} | lr {lr} | seed {C.SEED}')
        print('building encoders (MPNN / GRAM / BioBERT)...')
        icd2anc = build_icdcode2ancestor_dict()

        results = {}
        for name in wanted:
            print(f'\n=== {name} ===')
            # FRESH encoder for each variant (so weights don't leak between them)
            mol = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
            dis = GRAM(embedding_dim=50, icdcode2ancestor=icd2anc, device=device)
            pro = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
            model = build_variant(name, (mol, dis, pro), device,
                                  f'{base}_{name}', epochs, lr)

            # HINT's own learn() selects the best model by VALIDATION LOSS
            print('  training...')
            model.learn(tr, va, te)

            out = os.path.join(C.PREDS_DIR, f'preds_hintvar_{name}_phase_{phase}.csv')
            dump_preds(model, [('valid', va), ('test', te)], out)
            results[name] = out
            torch.save(model, os.path.join(C.HINT_DIR, 'save_model',
                                           f'{base}_{name}_ours.ckpt'))
    finally:
        os.chdir(cwd)

    # --- evaluate with the common harness ---
    print('\n================ VARIANTS — COMMON HARNESS ================')
    evald, yte = {}, None
    for name, path in results.items():
        r = H.evaluate_model(path, select_metric=C.SELECT_METRIC,
                             n_boot=C.BOOTSTRAP_N, alpha=C.BOOTSTRAP_ALPHA, seed=C.SEED)
        evald[name] = r
        df = H.load_preds(path); _, yte = H.split_scores(df, 'test')
        # Item 5 evidence: compare against the buggy (binarised) PR-AUC
        from sklearn.metrics import average_precision_score
        s, y = H.split_scores(df, 'test')
        pr_ok, pr_bad = average_precision_score(y, s), average_precision_score(y, (s >= 0.5).astype(int))
        print(f"  {name:14s} ROC={r['point']['roc_auc']:.4f}  "
              f"PR(correct)={pr_ok:.4f}  PR(broken)={pr_bad:.4f}  diff={pr_ok-pr_bad:+.4f}")

    table = H.compare_table(evald, test_labels=yte)
    print('\n' + table.to_string(index=False))
    out = os.path.join(C.PAPER_DIR, f'hint_variants_phase_{phase}.csv')
    table.to_csv(out, index=False)
    print(f'\nsaved -> {out}')
    return evald


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default=None)
    ap.add_argument('--epochs', type=int, default=5, help='learn_phaseIII.py uses 5')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--only', default=None, help='e.g. HINTModel,Interaction')
    a = ap.parse_args()
    main(a.phase, a.epochs, a.lr, a.only)
