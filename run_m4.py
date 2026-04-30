"""
run_m4.py — M4 Short-Term Forecasting for PatchLinear

Usage:
    python run_m4.py [--seed 2021] [--train_epochs 30] [--batch_size 16]

Setup:
    1. Copy exp_m4.py to exp/exp_m4.py
    2. Copy m4_summary.py to utils/m4_summary.py
    3. Copy losses.py to utils/losses.py
    4. Put M4 data in ./dataset/m4/:
         training.npz, test.npz, M4-info.csv, submission-Naive2.csv
       (download from TimeMixer repo or M4 competition GitHub)

Results are printed per frequency and saved to logs/m4/results.txt
"""

import argparse
import os
import random
import numpy as np
import torch

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed',          type=int,   default=2021)
parser.add_argument('--root_path',     type=str,   default='./dataset/m4/')
parser.add_argument('--checkpoints',   type=str,   default='./checkpoints/m4/')
parser.add_argument('--forecast_dir',  type=str,   default='./results/m4_forecasts/')
parser.add_argument('--train_epochs',  type=int,   default=30)
parser.add_argument('--batch_size',    type=int,   default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_workers',   type=int,   default=4)
parser.add_argument('--gpu',           type=int,   default=0)
parser.add_argument('--use_gpu',       type=bool,  default=True)

# Model arch — M4 uses small seasonal stream, no cross-channel
parser.add_argument('--d_model',     type=int,   default=64)
parser.add_argument('--t_ff',        type=int,   default=128)
parser.add_argument('--c_ff',        type=int,   default=16)
parser.add_argument('--patch_len',   type=int,   default=8)
parser.add_argument('--stride',      type=int,   default=4)
parser.add_argument('--dw_kernel',   type=int,   default=3)
parser.add_argument('--alpha',       type=float, default=0.3)
parser.add_argument('--t_dropout',     type=float, default=0.1)
parser.add_argument('--c_dropout',     type=float, default=0.1)
parser.add_argument('--embed_dropout', type=float, default=0.1)
parser.add_argument('--head_dropout',  type=float, default=0.0)
# Simplified: no decomp, seas stream only, no cross-channel
parser.add_argument('--use_decomp',        type=int, default=0)
parser.add_argument('--use_trend_stream',  type=int, default=0)
parser.add_argument('--use_seas_stream',   type=int, default=1)
parser.add_argument('--use_fusion_gate',   type=int, default=0)
parser.add_argument('--use_cross_channel', type=int, default=0)
parser.add_argument('--use_alpha_gate',    type=int, default=0)
parser.add_argument('--use_reparam',       type=int, default=0)

args = parser.parse_args()

# Seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Convert int flags to bool
for flag in ['use_decomp','use_trend_stream','use_seas_stream',
             'use_fusion_gate','use_cross_channel','use_alpha_gate','use_reparam']:
    setattr(args, flag, bool(getattr(args, flag)))
args.use_gpu = args.use_gpu and torch.cuda.is_available()

from exp.exp_m4 import Exp_M4
from data_provider.m4 import M4Meta

# ── M4 frequencies and their horizons ────────────────────────────────────────
# seq_len = 2 × pred_len (standard M4 practice)
FREQUENCIES = [
    # (seasonal_pattern, pred_len)
    ('Yearly',    6),
    ('Quarterly', 8),
    ('Monthly',  18),
    ('Weekly',   13),
    ('Daily',    14),
    ('Hourly',   48),
]

os.makedirs('logs/m4', exist_ok=True)
all_results = {}

print("=" * 65)
print("PatchLinear — M4 Short-Term Forecasting")
print(f"Config: d={args.d_model}, p={args.patch_len}, k={args.dw_kernel}, "
      f"lr={args.learning_rate}, epochs={args.train_epochs}")
print("=" * 65)

for freq_name, pred_len in FREQUENCIES:
    seq_len = max(pred_len * 2, 24)  # at least 24 lookback steps
    print(f"\n>>> {freq_name}  pred_len={pred_len}  seq_len={seq_len}")

    args.seasonal_patterns = freq_name
    args.pred_len = pred_len
    args.seq_len  = seq_len
    args.label_len = 0

    setting = (f"M4_{freq_name}_pl{pred_len}_d{args.d_model}"
               f"_p{args.patch_len}_k{args.dw_kernel}_s{args.seed}")

    exp = Exp_M4(args)

    # Train
    exp.train(setting)

    # Predict — save forecast CSVs
    exp.predict(setting, args.forecast_dir)

# ── Evaluate all frequencies together ────────────────────────────────────────
# M4Summary reads all frequency CSVs at once
print("\n" + "=" * 65)
print("EVALUATION (SMAPE / OWA per frequency)")
print("=" * 65)

# Use a dummy args for evaluation (freq doesn't matter for summary)
args.seasonal_patterns = 'Monthly'
args.pred_len = 18; args.seq_len = 36
exp_eval = Exp_M4(args)
smapes, owas = exp_eval.evaluate(args.forecast_dir)

print(f"\n{'Frequency':<14} {'SMAPE':>8} {'OWA':>8}")
print("-" * 34)
for key in ['Yearly','Quarterly','Monthly','Others','Average']:
    s = smapes.get(key, float('nan'))
    o = owas.get(key,   float('nan'))
    print(f"{key:<14} {s:>8.3f} {o:>8.3f}")

# Save results
results_path = 'logs/m4/results.txt'
with open(results_path, 'w') as f:
    f.write(f"PatchLinear M4 Results (seed={args.seed})\n")
    f.write(f"{'Frequency':<14} {'SMAPE':>8} {'OWA':>8}\n")
    for key in ['Yearly','Quarterly','Monthly','Others','Average']:
        s = smapes.get(key, float('nan'))
        o = owas.get(key, float('nan'))
        f.write(f"{key:<14} {s:>8.3f} {o:>8.3f}\n")
print(f"\nResults saved to {results_path}")
