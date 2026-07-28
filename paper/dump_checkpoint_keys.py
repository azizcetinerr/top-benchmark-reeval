# -*- coding: utf-8 -*-
"""
paper/dump_checkpoint_keys.py — GAP G1 evidence
==============================================
Proves the paper's Defect 1 claim ("attention/routers are in a plain dict -> never
trained, never saved") at the FILE level.

Produces three independent pieces of evidence:

  A) PARAMETER REGISTRATION — builds the model with moduledict_fix=False (original)
     and True (fixed), and counts how many attention/router tensors appear in
     named_parameters(). Original: 0 / 0 / 0. Fixed: 72 / 144 / 48.

  B) OPTIMIZER MEMBERSHIP + REAL UPDATE — attaches an optimizer to both models,
     runs a single synthetic backward+step and measures for the cross-attention
     tensors
     (i) whether they are in the optimizer's param groups,
     (ii) whether they actually CHANGE after the step.
     Original: 0/144 in optimizer, 0/144 changed. Fixed: 144/144, 144/144.

  C) PUBLISHED CHECKPOINTS — dumps the state_dict key list of the .pth.tar files
     in mexa/checkpoints/phase{I,II,III}; assigns each key to a group and flags
     whether it is an attention/router key.

Outputs (under paper/):
    table_checkpoint_keys.csv     — one row per checkpoint key
    table_param_registration.csv  — A + B summary table (paper Table 2.4)
    g1_console.txt                — exact copy of what is printed to the screen

Usage:
    cd paper
    python dump_checkpoint_keys.py
    # optional: a single phase only
    python dump_checkpoint_keys.py --phases II III
"""
import os
import sys
import json
import glob
import argparse
import contextlib

import torch
import torch.optim as optim
import pandas as pd

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER_DIR)
MEXA_DIR = os.path.join(ROOT, 'mexa')
sys.path.insert(0, MEXA_DIR)

from ctp.models_nlayers import ClinicalTtrialsPredictionModelH_nlayers as CTPModel  # noqa: E402

MAX_LENGTH = {'icds': 5, 'smiless': 5, 'in_criteria': 5, 'ex_criteria': 3}

# These three prefixes are the entirety of what the paper calls the "mechanism".
MECH_PREFIXES = ('self_att_layers', 'cross_att_layers', 'routers')


class Args:
    """An argparse.Namespace mimic from args.json + safe defaults."""

    def __init__(self, d):
        self.__dict__.update(d)


def build_args(phase, moduledict_fix, ckpt_args_path=None):
    """Take model hyperparameters from the checkpoint's own args.json.
    If absent, fall back to main.py defaults. Device is always CPU (reproducibility)."""
    d = dict(
        phase=phase, device='cpu',
        itoken_size=64, stoken_size=15, ctoken_size=768,
        dropout=0.005, nhead=2, nlayer=2, emb_size=8,
        epsilon=0.25, temperature=0.2, rho1=5e-2, rho2=1e-2,
        threshold=0.3, weighted=False,
    )
    if ckpt_args_path and os.path.exists(ckpt_args_path):
        with open(ckpt_args_path) as f:
            saved = json.load(f)
        for k in list(d):
            if k in saved and k not in ('device',):
                d[k] = saved[k]
    d['moduledict_fix'] = moduledict_fix
    return Args(d)


def group_of(key):
    for p in MECH_PREFIXES:
        if key.startswith(p):
            return p
    return key.split('.')[0]


def synthetic_batch(batch_size=4, seed=0):
    """A synthetic batch that needs no real data. We only test the question
    'does the gradient flow'; the values do not need to be meaningful.
    The masks are deliberately FULLY open (no row is fully masked) so that the
    NaN issue from G2 does not kick in and contaminate the measurement."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    B = batch_size
    return dict(
        itokens=r(B, MAX_LENGTH['icds'], 64),
        imasks=torch.zeros(B, MAX_LENGTH['icds'], dtype=torch.bool),
        in_ctokens=r(B, MAX_LENGTH['in_criteria'], 768),
        in_cmasks=torch.zeros(B, MAX_LENGTH['in_criteria'], dtype=torch.bool),
        ex_ctokens=r(B, MAX_LENGTH['ex_criteria'], 768),
        ex_cmasks=torch.zeros(B, MAX_LENGTH['ex_criteria'], dtype=torch.bool),
        stokens=r(B, MAX_LENGTH['smiless'], 15),
        smasks=torch.zeros(B, MAX_LENGTH['smiless'], dtype=torch.bool),
        labels=torch.randint(0, 2, (B,), generator=g).float(),
    )


# ----------------------------------------------------------------------
# A + B: parameter registration, optimizer membership, real update
# ----------------------------------------------------------------------
def probe_model(phase, moduledict_fix, ckpt_args_path, lr=1e-3):
    torch.manual_seed(2023)
    args = build_args(phase, moduledict_fix, ckpt_args_path)
    model = CTPModel(args, max_length=MAX_LENGTH)
    model.to('cpu')

    registered = dict(model.named_parameters())
    counts = {p: sum(1 for k in registered if k.startswith(p)) for p in MECH_PREFIXES}
    total_params = sum(p.numel() for p in model.parameters())
    total_tensors = len(registered)

    # --- collect the mechanism's REAL tensors from the model itself ---
    # Important: as a plain dict these modules do NOT appear in named_parameters(),
    # but they live as objects. We reach them via the containers so that the
    # "how many are in the optimizer" question can be asked over the SAME
    # denominator (144) in both configurations.
    def mech_tensors(container):
        out = []
        for name, mod in container.items():
            for pname, p in mod.named_parameters():
                out.append((f'{name}.{pname}', p))
        return out

    cross = mech_tensors(model.cross_att_layers)
    selfa = mech_tensors(model.self_att_layers)
    routers = mech_tensors(model.routers)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}

    in_opt = {
        'cross_att_layers': sum(1 for _, p in cross if id(p) in opt_ids),
        'self_att_layers': sum(1 for _, p in selfa if id(p) in opt_ids),
        'routers': sum(1 for _, p in routers if id(p) in opt_ids),
    }
    n_tensors = {
        'cross_att_layers': len(cross),
        'self_att_layers': len(selfa),
        'routers': len(routers),
    }

    # --- Single backward + step: does it actually change? ---
    before = {k: p.detach().clone() for grp, lst in
              (('cross_att_layers', cross), ('self_att_layers', selfa), ('routers', routers))
              for k, p in [(f'{grp}|{n}', t) for n, t in lst]}

    batch = synthetic_batch()
    model.train()
    optimizer.zero_grad()
    out = model(batch['itokens'], batch['imasks'],
                batch['in_ctokens'], batch['in_cmasks'],
                batch['ex_ctokens'], batch['ex_cmasks'],
                batch['stokens'], batch['smasks'], batch['labels'])
    loss = out['loss']
    loss_val = float(loss.detach())
    loss.backward()

    has_grad = {g: 0 for g in n_tensors}
    for grp, lst in (('cross_att_layers', cross), ('self_att_layers', selfa), ('routers', routers)):
        for _, p in lst:
            if p.grad is not None and torch.any(p.grad != 0):
                has_grad[grp] += 1

    optimizer.step()

    changed = {g: 0 for g in n_tensors}
    for grp, lst in (('cross_att_layers', cross), ('self_att_layers', selfa), ('routers', routers)):
        for n, p in lst:
            if not torch.equal(before[f'{grp}|{n}'], p.detach()):
                changed[grp] += 1

    return dict(
        phase=phase,
        configuration='Fixed (nn.ModuleDict)' if moduledict_fix else 'Original (plain dict)',
        moduledict_fix=moduledict_fix,
        registered_self_att=counts['self_att_layers'],
        registered_cross_att=counts['cross_att_layers'],
        registered_routers=counts['routers'],
        registered_tensors_total=total_tensors,
        trainable_params=total_params,
        cross_att_tensors=n_tensors['cross_att_layers'],
        cross_att_in_optimizer=in_opt['cross_att_layers'],
        cross_att_with_grad=has_grad['cross_att_layers'],
        cross_att_changed=changed['cross_att_layers'],
        self_att_tensors=n_tensors['self_att_layers'],
        self_att_in_optimizer=in_opt['self_att_layers'],
        self_att_changed=changed['self_att_layers'],
        router_tensors=n_tensors['routers'],
        router_in_optimizer=in_opt['routers'],
        router_changed=changed['routers'],
        train_loss_one_step=loss_val,
        state_dict_keys=len(model.state_dict()),
    ), model


# ----------------------------------------------------------------------
# C: key dump of the published checkpoints
# ----------------------------------------------------------------------
def find_ckpt(phase):
    d = os.path.join(MEXA_DIR, 'checkpoints', f'phase{phase}')
    hits = sorted(glob.glob(os.path.join(d, '*.pth.tar')))
    return (hits[0] if hits else None), os.path.join(d, 'args.json')


def load_state_dict(path):
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:      # torch < 2.0
        ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict):
        for k in ('state_dict', 'model', 'model_state_dict'):
            if k in ck and isinstance(ck[k], dict):
                return ck[k], sorted([x for x in ck if x != k])
        return ck, []
    return ck.state_dict(), []


def dump_keys(phase, ckpt_path, fixed_model):
    sd, siblings = load_state_dict(ckpt_path)
    fixed_keys = set(fixed_model.state_dict().keys())
    rows = []
    for k, v in sd.items():
        shape = tuple(v.shape) if hasattr(v, 'shape') else ()
        numel = int(v.numel()) if hasattr(v, 'numel') else 0
        g = group_of(k)
        rows.append(dict(
            phase=phase,
            checkpoint=os.path.basename(ckpt_path),
            key=k,
            shape=str(shape),
            numel=numel,
            group=g,
            is_mechanism=g in MECH_PREFIXES,
            exists_in_fixed_model=k in fixed_keys,
        ))
    return rows, siblings, set(sd.keys()), fixed_keys


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phases', nargs='+', default=['I', 'II', 'III'])
    ap.add_argument('--out-dir', default=PAPER_DIR)
    a = ap.parse_args()

    buf = []

    def say(*x):
        line = ' '.join(str(i) for i in x)
        print(line)
        buf.append(line)

    say('=' * 78)
    say('GAP G1 — parameter registration + checkpoint key dump')
    say(f'torch {torch.__version__}')
    say('=' * 78)

    key_rows, param_rows = [], []

    for phase in a.phases:
        ckpt_path, args_path = find_ckpt(phase)
        say(f'\n--- Phase {phase} ---')
        say(f'checkpoint : {ckpt_path}')
        say(f'args.json  : {args_path if os.path.exists(args_path) else "(none, defaults)"}')

        # A + B
        for fix in (False, True):
            row, model = probe_model(phase, fix, args_path)
            param_rows.append(row)
            say(f"\n[{row['configuration']}]")
            say(f"  named_parameters : self-att {row['registered_self_att']:3d} | "
                f"cross-att {row['registered_cross_att']:3d} | routers {row['registered_routers']:3d}")
            say(f"  trainable params : {row['trainable_params']:,}")
            say(f"  optimizer members: cross-att {row['cross_att_in_optimizer']}/{row['cross_att_tensors']} | "
                f"self-att {row['self_att_in_optimizer']}/{row['self_att_tensors']} | "
                f"router {row['router_in_optimizer']}/{row['router_tensors']}")
            say(f"  after 1 step     : cross-att {row['cross_att_changed']}/{row['cross_att_tensors']} changed | "
                f"self-att {row['self_att_changed']}/{row['self_att_tensors']} | "
                f"router {row['router_changed']}/{row['router_tensors']}")
            say(f"  state_dict keys  : {row['state_dict_keys']}")
            if fix:
                fixed_model = model

        # C
        if ckpt_path and os.path.exists(ckpt_path):
            rows, siblings, ck_keys, fixed_keys = dump_keys(phase, ckpt_path, fixed_model)
            key_rows += rows
            mech = [r for r in rows if r['is_mechanism']]
            missing = sorted(k for k in fixed_keys if k not in ck_keys)
            mech_missing = [k for k in missing if group_of(k) in MECH_PREFIXES]
            say(f"\n[Published checkpoint]")
            say(f"  total keys            : {len(rows)}")
            say(f"  mechanism keys        : {len(mech)}   <-- claim: 0")
            say(f"  top-level prefixes    : {sorted({r['group'] for r in rows})}")
            if siblings:
                say(f"  checkpoint side fields: {siblings}")
            say(f"  keys in fixed model but NOT in checkpoint: {len(missing)} "
                f"({len(mech_missing)} of them are mechanism)")
        else:
            say('  [SKIPPED] checkpoint not found')

    # --- write ---
    os.makedirs(a.out_dir, exist_ok=True)
    p1 = os.path.join(a.out_dir, 'table_checkpoint_keys.csv')
    p2 = os.path.join(a.out_dir, 'table_param_registration.csv')
    p3 = os.path.join(a.out_dir, 'g1_console.txt')
    pd.DataFrame(key_rows).to_csv(p1, index=False)
    pd.DataFrame(param_rows).to_csv(p2, index=False)

    say('\n' + '=' * 78)
    say('WRITTEN:')
    say(f'  {p1}  ({len(key_rows)} rows)')
    say(f'  {p2}  ({len(param_rows)} rows)')
    say(f'  {p3}')

    with open(p3, 'w') as f:
        f.write('\n'.join(buf) + '\n')


if __name__ == '__main__':
    main()
