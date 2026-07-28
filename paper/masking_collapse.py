# -*- coding: utf-8 -*-
"""
paper/masking_collapse.py — GAP G2 evidence
==========================================
Proves the paper's Defect 2 claim ("once the routers are actually trained the
Cauchy loss pushes the probabilities below the threshold; when ALL tokens of a
sequence are masked, the softmax is taken over all -inf and produces NaN -> the
architecture cannot be trained as published") with a STEP-BY-STEP LOG.

How it works
-------------
The ctp.models_nlayers.remasks function is monkey-patched. The patched version:
  * on each call counts how many sequences are FULLY masked (measured BEFORE the
    safeguard is applied — i.e. the answer to "what would the published code do here"),
  * records the mean of the router probabilities and the fraction of tokens below
    the threshold,
  * if --safeguard off, applies the original (unprotected) behavior verbatim -> NaN;
    if --safeguard on, applies the repo's "at least one token stays open" fix.

So the ONLY variable is the safeguard: two runs with the same seed, same data, same lr.

Call sites (the fixed order inside forward, 8 of them):
    msi, msc, mis, mic, mincs, minci, mexcs, mexci

Outputs (under paper/):
    table_masking_collapse.csv   — one row per (step, lr, safeguard, call-site)
    table_masking_summary.csv    — per-run summary: first NaN step, final masking fraction
    g2_console.txt               — exact copy of what is printed to the screen

Usage:
    cd paper
    python masking_collapse.py --phase II --steps 300
    # lr sweep (for the paper's "diverges at every lr" claim):
    python masking_collapse.py --phase II --steps 300 --lrs 5e-2 1e-2 1e-3 1e-4
    # without data, with a synthetic batch:
    python masking_collapse.py --phase II --synthetic
"""
import os
import sys
import json
import glob
import argparse

import torch
import torch.optim as optim
import pandas as pd

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER_DIR)
MEXA_DIR = os.path.join(ROOT, 'mexa')
sys.path.insert(0, MEXA_DIR)

import ctp.models_nlayers as M                                        # noqa: E402
from ctp.models_nlayers import ClinicalTtrialsPredictionModelH_nlayers as CTPModel  # noqa: E402

MAX_LENGTH = {'icds': 5, 'smiless': 5, 'in_criteria': 5, 'ex_criteria': 3}
SITES = ['msi', 'msc', 'mis', 'mic', 'mincs', 'minci', 'mexcs', 'mexci']


class Args:
    def __init__(self, d):
        self.__dict__.update(d)


def build_args(phase, ckpt_args_path=None, threshold=None, rho1=None):
    d = dict(
        phase=phase, device='cpu',
        itoken_size=64, stoken_size=15, ctoken_size=768,
        dropout=0.005, nhead=2, nlayer=2, emb_size=8,
        epsilon=0.25, temperature=0.2, rho1=5e-2, rho2=1e-2,
        threshold=0.3, weighted=False, moduledict_fix=True,
    )
    if ckpt_args_path and os.path.exists(ckpt_args_path):
        with open(ckpt_args_path) as f:
            saved = json.load(f)
        for k in list(d):
            if k in saved and k not in ('device', 'moduledict_fix'):
                d[k] = saved[k]
    if threshold is not None:
        d['threshold'] = threshold
    if rho1 is not None:
        d['rho1'] = rho1
    return Args(d)


# ----------------------------------------------------------------------
# Patched remasks
# ----------------------------------------------------------------------
class MaskProbe:
    """Wraps remasks, collects statistics per call."""

    def __init__(self, safeguard):
        self.safeguard = safeguard      # True: repo fix, False: published code
        self.call_idx = 0
        self.records = []               # records of this step

    def reset(self):
        self.call_idx = 0
        self.records = []

    def __call__(self, masks, prob, threshold):
        site = SITES[self.call_idx] if self.call_idx < len(SITES) else f'site{self.call_idx}'
        self.call_idx += 1

        if threshold <= 0 or threshold >= 1:
            return masks

        p = prob.squeeze(-1)                       # [B, L]
        new = masks | (p < threshold)

        # --- MEASUREMENT: BEFORE the safeguard is applied ---
        all_masked = new.all(dim=1)                # rows where the published code would produce NaN
        B = int(p.shape[0])
        real = (~masks)                            # non-padding tokens
        n_real = int(real.sum())
        below = int(((p < threshold) & real).sum())

        self.records.append(dict(
            site=site,
            batch_rows=B,
            fully_masked_rows=int(all_masked.sum()),
            fully_masked_frac=float(all_masked.float().mean()),
            real_tokens=n_real,
            tokens_below_threshold=below,
            token_below_frac=(below / n_real) if n_real else float('nan'),
            mean_router_prob=float(p[real].mean().detach()) if n_real else float('nan'),
            min_router_prob=float(p[real].min().detach()) if n_real else float('nan'),
        ))

        if not self.safeguard:
            return new                             # <-- published behavior: open to NaN

        if all_masked.any():
            cand = p.masked_fill(masks, float('-inf'))
            keep = cand.argmax(dim=1)
            idx = all_masked.nonzero(as_tuple=True)[0]
            new[idx, keep[idx]] = False
        return new


# ----------------------------------------------------------------------
# Veri
# ----------------------------------------------------------------------
def synthetic_batch(batch_size, seed=0):
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


def real_loader(phase, batch_size, data_folder):
    from ctp.dataset import ClinicalTtrialsPredictionDatasetH as CTPDataset
    from torch.utils.data import DataLoader
    ds = CTPDataset(data_folder, subset='train', phase=phase,
                    max_length=MAX_LENGTH, reduce={'icds': 'sum', 'criteria': 'first'})
    if len(ds) == 0:
        raise FileNotFoundError(f'no phase_{phase} data in {data_folder}')
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0), len(ds)


def to_batch(data):
    return dict(
        itokens=data['itokens'], imasks=data['imasks'],
        in_ctokens=data['in_ctokens'], in_cmasks=data['in_cmasks'],
        ex_ctokens=data['ex_ctokens'], ex_cmasks=data['ex_cmasks'],
        stokens=data['stokens'], smasks=data['smasks'],
        labels=data['label'].float(),
    )


# ----------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------
def run(phase, lr, safeguard, steps, seed, args_path, batches, threshold, rho1, say):
    torch.manual_seed(seed)
    args = build_args(phase, args_path, threshold, rho1)
    model = CTPModel(args, max_length=MAX_LENGTH).to('cpu')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    probe = MaskProbe(safeguard)
    orig_remasks = M.remasks
    M.remasks = probe

    tag = f'lr={lr:g} safeguard={"on" if safeguard else "off"}'
    say(f'\n>>> {tag}  (threshold={args.threshold}, rho1={args.rho1}, seed={seed})')

    rows = []
    first_nan_step = None
    model.train()
    try:
        for step in range(1, steps + 1):
            batch = batches(step)
            probe.reset()
            optimizer.zero_grad()
            out = model(batch['itokens'], batch['imasks'],
                        batch['in_ctokens'], batch['in_cmasks'],
                        batch['ex_ctokens'], batch['ex_cmasks'],
                        batch['stokens'], batch['smasks'], batch['labels'])
            loss = out['loss']
            loss_val = float(loss.detach())
            cauchy = float(out['cauchyloss'].detach())
            bce = float(out['bceloss'].detach())
            is_nan = loss_val != loss_val

            for rec in probe.records:
                rows.append(dict(
                    phase=phase, lr=lr, safeguard='on' if safeguard else 'off',
                    seed=seed, step=step, loss=loss_val, bceloss=bce,
                    cauchyloss=cauchy, loss_is_nan=is_nan, **rec))

            if is_nan and first_nan_step is None:
                first_nan_step = step
                say(f'    [NaN] step {step}: loss became NaN. '
                    f'fully-masked row ratio (this step) = '
                    f'{sum(r["fully_masked_rows"] for r in probe.records)}/'
                    f'{sum(r["batch_rows"] for r in probe.records)}')
                break

            loss.backward()
            optimizer.step()

            if step % max(1, steps // 10) == 0 or step == 1:
                fm = sum(r['fully_masked_rows'] for r in probe.records)
                tot = sum(r['batch_rows'] for r in probe.records)
                mp = sum(r['mean_router_prob'] for r in probe.records) / len(probe.records)
                say(f'    step {step:4d} | loss {loss_val:8.4f} | cauchy {cauchy:7.4f} | '
                    f'mean router p {mp:.4f} | fully-masked {fm}/{tot} ({fm/tot:.1%})')
    finally:
        M.remasks = orig_remasks

    last = [r for r in rows if r['step'] == max(r2['step'] for r2 in rows)] if rows else []
    fm = sum(r['fully_masked_rows'] for r in last)
    tot = sum(r['batch_rows'] for r in last)
    summary = dict(
        phase=phase, lr=lr, safeguard='on' if safeguard else 'off', seed=seed,
        steps_requested=steps,
        steps_completed=max((r['step'] for r in rows), default=0),
        first_nan_step=first_nan_step,
        diverged=first_nan_step is not None,
        final_loss=last[0]['loss'] if last else float('nan'),
        final_cauchyloss=last[0]['cauchyloss'] if last else float('nan'),
        final_fully_masked_frac=(fm / tot) if tot else float('nan'),
        final_mean_router_prob=(sum(r['mean_router_prob'] for r in last) / len(last)) if last else float('nan'),
        final_token_below_frac=(sum(r['token_below_frac'] for r in last) / len(last)) if last else float('nan'),
    )
    return rows, summary


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='II', choices=['I', 'II', 'III'])
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lrs', nargs='+', type=float, default=[5e-2])
    ap.add_argument('--seed', type=int, default=2023)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--threshold', type=float, default=None, help='override the args.json value')
    ap.add_argument('--rho1', type=float, default=None, help='Cauchy weight; override the args.json value')
    ap.add_argument('--synthetic', action='store_true', help='synthetic batch instead of real data')
    ap.add_argument('--data-folder', default=os.environ.get(
        'MEXA_DATA_FOLDER', os.path.join(MEXA_DIR, 'DataPath') + os.sep))
    ap.add_argument('--out-dir', default=PAPER_DIR)
    a = ap.parse_args()

    buf = []

    def say(*x):
        line = ' '.join(str(i) for i in x)
        print(line, flush=True)
        buf.append(line)

    say('=' * 78)
    say('GAP G2 — masking collapse log')
    say(f'torch {torch.__version__} | phase {a.phase} | steps {a.steps} | lrs {a.lrs}')
    say('=' * 78)

    args_path = os.path.join(MEXA_DIR, 'checkpoints', f'phase{a.phase}', 'args.json')

    # --- data source ---
    if a.synthetic:
        say('[data] synthetic batch')
        fixed = synthetic_batch(a.batch_size, seed=a.seed)
        batches = lambda step: fixed        # same batch over and over: cleanest controlled setting
    else:
        try:
            loader, n = real_loader(a.phase, a.batch_size, a.data_folder)
            say(f'[data] real: {a.data_folder} phase_{a.phase}, {n} samples, batch {a.batch_size}')
            cache = {}

            def batches(step, _loader=loader):
                if 'it' not in cache:
                    cache['it'] = iter(_loader)
                try:
                    d = next(cache['it'])
                except StopIteration:
                    cache['it'] = iter(_loader)
                    d = next(cache['it'])
                return to_batch(d)
        except Exception as e:
            say(f'[data] could not load real data ({e}); falling back to synthetic')
            fixed = synthetic_batch(a.batch_size, seed=a.seed)
            batches = lambda step: fixed

    all_rows, all_sum = [], []
    for lr in a.lrs:
        for safeguard in (False, True):
            rows, s = run(a.phase, lr, safeguard, a.steps, a.seed,
                          args_path, batches, a.threshold, a.rho1, say)
            all_rows += rows
            all_sum.append(s)

    os.makedirs(a.out_dir, exist_ok=True)
    p1 = os.path.join(a.out_dir, 'table_masking_collapse.csv')
    p2 = os.path.join(a.out_dir, 'table_masking_summary.csv')
    p3 = os.path.join(a.out_dir, 'g2_console.txt')
    pd.DataFrame(all_rows).to_csv(p1, index=False)
    df = pd.DataFrame(all_sum)
    df.to_csv(p2, index=False)

    say('\n' + '=' * 78)
    say('SUMMARY')
    say(df.to_string(index=False))
    say('\nWRITTEN:')
    say(f'  {p1}  ({len(all_rows)} rows)')
    say(f'  {p2}  ({len(all_sum)} rows)')
    say(f'  {p3}')

    with open(p3, 'w') as f:
        f.write('\n'.join(buf) + '\n')


if __name__ == '__main__':
    main()
