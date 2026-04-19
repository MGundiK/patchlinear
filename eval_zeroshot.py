"""
Zero-shot evaluation: load a checkpoint trained on src, evaluate on tgt test set.
Usage:
  python eval_zeroshot.py --src ETTh1 --tgt ETTh2 --pred_len 96 --seed 2021
"""
import argparse, os, sys, re
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--src',      type=str, required=True)
parser.add_argument('--tgt',      type=str, required=True)
parser.add_argument('--pred_len', type=int, required=True)
parser.add_argument('--seed',     type=int, default=2021)
parser.add_argument('--root_path',type=str, default='./dataset/')
parser.add_argument('--checkpoints', type=str, default='./checkpoints')
# Model args (must match training)
parser.add_argument('--d_model',    type=int, default=64)
parser.add_argument('--t_ff',       type=int, default=128)
parser.add_argument('--c_ff',       type=int, default=16)
parser.add_argument('--patch_len',  type=int, default=16)
parser.add_argument('--stride',     type=int, default=8)
parser.add_argument('--dw_kernel',  type=int, default=7)
parser.add_argument('--alpha',      type=float, default=0.3)
parser.add_argument('--use_decomp',        type=int, default=1)
parser.add_argument('--use_trend_stream',  type=int, default=1)
parser.add_argument('--use_seas_stream',   type=int, default=1)
parser.add_argument('--use_fusion_gate',   type=int, default=1)
parser.add_argument('--use_cross_channel', type=int, default=1)
parser.add_argument('--use_alpha_gate',    type=int, default=1)
parser.add_argument('--t_dropout',     type=float, default=0.1)
parser.add_argument('--c_dropout',     type=float, default=0.1)
parser.add_argument('--embed_dropout', type=float, default=0.1)
parser.add_argument('--head_dropout',  type=float, default=0.1)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Dataset info
DS_INFO = {
    'ETTh1': {'path':'ETTh1.csv', 'type':'ETTh1', 'enc_in':7},
    'ETTh2': {'path':'ETTh2.csv', 'type':'ETTh2', 'enc_in':7},
    'ETTm1': {'path':'ETTm1.csv', 'type':'ETTm1', 'enc_in':7},
    'ETTm2': {'path':'ETTm2.csv', 'type':'ETTm2', 'enc_in':7},
}

src_info = DS_INFO[args.src]
tgt_info = DS_INFO[args.tgt]

# Find checkpoint — look for src-based setting
ckpt_dir = args.checkpoints
pattern = f"ZS_{args.src}_pl{args.pred_len}_PatchLinear_{src_info['type']}_ftM_sl96_pl{args.pred_len}"
candidates = [d for d in os.listdir(ckpt_dir) if d.startswith(pattern.split('_PatchLinear')[0])]
if not candidates:
    print(f"ERROR: No checkpoint found matching ZS_{args.src}_pl{args.pred_len}*")
    print(f"Available: {os.listdir(ckpt_dir)[:5]}")
    sys.exit(1)

ckpt_path = os.path.join(ckpt_dir, candidates[0], 'checkpoint.pth')
if not os.path.exists(ckpt_path):
    print(f"ERROR: checkpoint.pth not found in {os.path.join(ckpt_dir, candidates[0])}")
    sys.exit(1)

print(f"Loading: {ckpt_path}")

# Build model args namespace for PatchLinear
class ModelArgs:
    pass
margs = ModelArgs()
margs.seq_len    = 96
margs.pred_len   = args.pred_len
margs.enc_in     = src_info['enc_in']  # same for all ETT
margs.d_model    = args.d_model
margs.t_ff       = args.t_ff
margs.c_ff       = args.c_ff
margs.patch_len  = args.patch_len
margs.stride     = args.stride
margs.dw_kernel  = args.dw_kernel
margs.alpha      = args.alpha
margs.use_decomp        = bool(args.use_decomp)
margs.use_trend_stream  = bool(args.use_trend_stream)
margs.use_seas_stream   = bool(args.use_seas_stream)
margs.use_fusion_gate   = bool(args.use_fusion_gate)
margs.use_cross_channel = bool(args.use_cross_channel)
margs.use_alpha_gate    = bool(args.use_alpha_gate)
margs.t_dropout     = args.t_dropout
margs.c_dropout     = args.c_dropout
margs.embed_dropout = args.embed_dropout
margs.head_dropout  = args.head_dropout
margs.small_kernel  = 3
margs.use_reparam   = False
margs.task_name     = 'long_term_forecast'

from models.PatchLinear import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(margs).float().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

# Load TARGET test data using data_provider
# We temporarily set up args namespace for data_provider
import types
dp_args = types.SimpleNamespace(
    root_path   = args.root_path,
    data_path   = tgt_info['path'],
    data        = tgt_info['type'],
    features    = 'M',
    seq_len     = 96,
    label_len   = 48,
    pred_len    = args.pred_len,
    enc_in      = tgt_info['enc_in'],
    target      = 'OT',
    freq        = 'h',
    embed       = 'timeF',
    num_workers = 4,
    batch_size  = 32,
    inverse     = False,
    scale       = True,
    train_only  = False,
    seasonal_patterns = 'Monthly',
)

from data_provider.data_factory import data_provider
_, test_loader = data_provider(dp_args, 'test')

# Evaluate
preds, trues = [], []
with torch.no_grad():
    for batch_x, batch_y, _, _ in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        out = model(batch_x)
        out = out[:, -args.pred_len:, :]
        y   = batch_y[:, -args.pred_len:, :]
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())

preds = np.concatenate(preds, 0)
trues = np.concatenate(trues, 0)
mse = np.mean((preds - trues)**2)
mae = np.mean(np.abs(preds - trues))
print(f"Zero-shot {args.src}→{args.tgt} pl={args.pred_len}: mse:{mse:.6f}  mae:{mae:.6f}")

# Append to results file
with open('result_zeroshot.txt', 'a') as f:
    f.write(f"ZS_{args.src}_to_{args.tgt}_pl{args.pred_len}\n")
    f.write(f"mse:{mse:.6f}, mae:{mae:.6f}\n\n")
