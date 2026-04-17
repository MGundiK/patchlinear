"""
Entry point for imputation experiments.
Mirrors run.py but routes to Exp_Imputation and adds mask_rate arg.
"""
import argparse
import os
import torch
import random
import numpy as np

parser = argparse.ArgumentParser(description='PatchLinear — Imputation')

# ── reproducibility ───────────────────────────────────────────────────────────
parser.add_argument('--seed', type=int, default=2021)

# ── basic ─────────────────────────────────────────────────────────────────────
parser.add_argument('--is_training',  type=int, default=1)
parser.add_argument('--model_id',     type=str, default='test')
parser.add_argument('--model',        type=str, default='PatchLinear')
parser.add_argument('--checkpoints',  type=str, default='./checkpoints/')
parser.add_argument('--task_name',    type=str, default='imputation')

# ── data ──────────────────────────────────────────────────────────────────────
parser.add_argument('--data',       type=str, default='ETTh1')
parser.add_argument('--root_path',  type=str, default='./dataset/')
parser.add_argument('--data_path',  type=str, default='ETTh1.csv')
parser.add_argument('--features',   type=str, default='M')
parser.add_argument('--target',     type=str, default='OT')
parser.add_argument('--freq',       type=str, default='h')
parser.add_argument('--embed',      type=str, default='timeF')

# ── imputation specific ───────────────────────────────────────────────────────
parser.add_argument('--seq_len',   type=int,   default=1024,
                    help='reconstruction window — 1024 per TimeMixer++ benchmark')
parser.add_argument('--mask_rate', type=float, default=0.25,
                    help='fraction of timesteps to mask (0.125/0.25/0.375/0.5)')
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len',  type=int, default=0)
parser.add_argument('--enc_in',    type=int, default=7)

# ── model architecture ────────────────────────────────────────────────────────
parser.add_argument('--d_model',    type=int,   default=64)
parser.add_argument('--t_ff',       type=int,   default=128)
parser.add_argument('--c_ff',       type=int,   default=16)
parser.add_argument('--patch_len',  type=int,   default=16)
parser.add_argument('--stride',     type=int,   default=8)
parser.add_argument('--dw_kernel',  type=int,   default=7)
parser.add_argument('--small_kernel', type=int, default=3)
parser.add_argument('--alpha',      type=float, default=0.3)

# ── dropout ───────────────────────────────────────────────────────────────────
parser.add_argument('--t_dropout',     type=float, default=0.1)
parser.add_argument('--c_dropout',     type=float, default=0.1)
parser.add_argument('--embed_dropout', type=float, default=0.1)
parser.add_argument('--head_dropout',  type=float, default=0.1)

# ── ablation switches ─────────────────────────────────────────────────────────
parser.add_argument('--use_reparam',       type=int, default=0)
parser.add_argument('--use_decomp',        type=int, default=1)
parser.add_argument('--use_trend_stream',  type=int, default=1)
parser.add_argument('--use_seas_stream',   type=int, default=1)
parser.add_argument('--use_fusion_gate',   type=int, default=1)
parser.add_argument('--use_cross_channel', type=int, default=1)
parser.add_argument('--use_alpha_gate',    type=int, default=1)

# ── optimisation ──────────────────────────────────────────────────────────────
parser.add_argument('--num_workers',    type=int,   default=10)
parser.add_argument('--itr',           type=int,   default=1)
parser.add_argument('--train_epochs',  type=int,   default=100)
parser.add_argument('--batch_size',    type=int,   default=16)
parser.add_argument('--patience',      type=int,   default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--des',           type=str,   default='Exp')
parser.add_argument('--lradj',         type=str,   default='type1')
parser.add_argument('--use_amp',       action='store_true', default=False)
parser.add_argument('--train_only',    type=bool,  default=False)

# ── GPU ───────────────────────────────────────────────────────────────────────
parser.add_argument('--use_gpu',       type=bool, default=True)
parser.add_argument('--gpu',           type=int,  default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices',       type=str,  default='0,1,2,3')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# convert int flags to bool
for flag in ['use_reparam','use_decomp','use_trend_stream','use_seas_stream',
             'use_fusion_gate','use_cross_channel','use_alpha_gate']:
    setattr(args, flag, bool(getattr(args, flag)))

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices    = args.devices.replace(' ', '')
    device_ids      = args.devices.split(',')
    args.device_ids = [int(x) for x in device_ids]
    args.gpu        = args.device_ids[0]

print('Args:', args)

from exp.exp_imputation import Exp_Imputation
Exp = Exp_Imputation

if args.is_training:
    for ii in range(args.itr):
        setting = (
            f"{args.model_id}_{args.model}_{args.data}_"
            f"ft{args.features}_sl{args.seq_len}_mr{args.mask_rate:.3f}_"
            f"d{args.d_model}_{args.des}_{ii}"
        )
        exp = Exp(args)
        print(f">>>>>>>start training: {setting}")
        exp.train(setting)
        print(f">>>>>>>testing: {setting}")
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = (
        f"{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_mr{args.mask_rate:.3f}_"
        f"d{args.d_model}_{args.des}_{ii}"
    )
    exp = Exp(args)
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
