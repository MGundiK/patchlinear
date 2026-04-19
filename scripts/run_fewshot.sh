#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Few-Shot Forecasting (10% training data)
# Protocol: TimeMixer++ Section 4.1.5
# Prerequisites:
#   1. Add to run.py:     parser.add_argument('--few_shot_ratio', type=float, default=1.0)
#   2. Add to data_factory.py (inside else branch, after data_set = Data(...)):
#        if flag == 'train' and hasattr(args, 'few_shot_ratio') and args.few_shot_ratio < 1.0:
#            from torch.utils.data import Subset
#            n = max(1, int(len(data_set) * args.few_shot_ratio))
#            data_set = Subset(data_set, range(n))
# =============================================================================

# NO set -e — don't abort on first failure, show all errors
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG_DIR="logs/fewshot"
mkdir -p "$LOG_DIR"
mkdir -p logs   # ensure parent exists for tee
> result_fewshot.txt

SEED=2021

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1 || true
  if grep -q "mse:" "$LOG"; then
    grep "mse:" "$LOG" | tail -1
  else
    echo "  [FAILED] $(head -3 $LOG | tail -1)"
  fi
}

# ── ETTh1 ──────────────────────────────────────────────────────────────────
echo "=== ETTh1 ==="
for PL in 96 192 336 720; do
  run_one "ETTh1_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTh1.csv --data ETTh1 --features M \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --model_id "ETTh1_fs_pl${PL}"
done

# ── ETTh2 ──────────────────────────────────────────────────────────────────
echo "=== ETTh2 ==="
for PL in 96 192 336 720; do
  run_one "ETTh2_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTh2.csv --data ETTh2 --features M \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --model_id "ETTh2_fs_pl${PL}"
done

# ── ETTm1 ──────────────────────────────────────────────────────────────────
echo "=== ETTm1 ==="
for PL in 96 192 336 720; do
  run_one "ETTm1_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTm1.csv --data ETTm1 --features M \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --model_id "ETTm1_fs_pl${PL}"
done

# ── ETTm2 ──────────────────────────────────────────────────────────────────
echo "=== ETTm2 ==="
for PL in 96 192 336 720; do
  run_one "ETTm2_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path ETTm2.csv --data ETTm2 --features M \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0001 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --model_id "ETTm2_fs_pl${PL}"
done

# ── Weather ────────────────────────────────────────────────────────────────
echo "=== Weather ==="
for PL in 96 192 336 720; do
  run_one "Weather_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path weather.csv --data custom --features M \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 21 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 2048 --learning_rate 0.0005 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --model_id "Weather_fs_pl${PL}"
done

# ── Electricity (ECL) ──────────────────────────────────────────────────────
echo "=== Electricity ==="
for PL in 96 192 336 720; do
  run_one "ECL_fs_pl${PL}" \
    --is_training 1 --root_path ./dataset/ \
    --data_path electricity.csv --data custom --features M \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 321 \
    --d_model 256 --t_ff 512 --c_ff 128 \
    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
    --use_cross_channel 1 --use_alpha_gate 1 \
    --batch_size 256 --learning_rate 0.01 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --seed ${SEED} --des Exp \
    --few_shot_ratio 0.1 \
    --model_id "ECL_fs_pl${PL}"
done

echo ""
echo "=== FEW-SHOT SUMMARY ==="
python3 - << 'PYEOF'
import re, os, numpy as np

log_dir = "logs/fewshot"
groups = {
    'ETT(Avg)': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
    'Weather':  ['Weather'],
    'ECL':      ['ECL'],
}

# TimeMixer++ baselines for comparison
baselines = {
    'ETT(Avg)': (0.396, 0.421),
    'Weather':  (0.241, 0.271),
    'ECL':      (0.168, 0.271),
}

print(f"{'Dataset':<12} {'PL MSE':>8} {'PL MAE':>8}  {'TM++ MSE':>9} {'TM++ MAE':>9}")
print("─"*55)
for label, dsets in groups.items():
    mse_all = []; mae_all = []
    for ds in dsets:
        for pl in [96, 192, 336, 720]:
            f = f"{log_dir}/{ds}_fs_pl{pl}.log"
            if not os.path.exists(f): continue
            for line in open(f):
                m = re.search(r'mse:([\d.]+),?\s*mae:([\d.]+)', line)
                if m:
                    mse_all.append(float(m.group(1)))
                    mae_all.append(float(m.group(2)))
    if mse_all:
        tm_mse, tm_mae = baselines[label]
        print(f"{label:<12} {np.mean(mse_all):>8.3f} {np.mean(mae_all):>8.3f}  {tm_mse:>9.3f} {tm_mae:>9.3f}")
    else:
        print(f"{label:<12} no results found")
PYEOF
