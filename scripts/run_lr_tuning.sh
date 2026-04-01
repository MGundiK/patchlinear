#!/usr/bin/env bash
# =============================================================================
# PatchLinear — learning rate tuning
#
# Runs Exchange, Electricity, Solar, and ILI at alternative LRs.
# Results append to result.txt; per-run logs go to logs/lr_tuning/.
#
# Usage:
#   bash scripts/run_lr_tuning.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/lr_tuning"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
}

# =============================================================================
# EXCHANGE
# Current: 0.00001  |  Try: 0.00005, 0.0001
# =============================================================================
for LR in 0.00005 0.0001; do
  for PRED_LEN in 96 192 336 720; do
    run_one "Exchange_pl${PRED_LEN}_lr${LR}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path exchange_rate.csv --data custom \
      --features M --seq_len 96 --label_len 48 \
      --pred_len "${PRED_LEN}" --enc_in 8 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 32 --learning_rate "${LR}" \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed 2021 --des Exp \
      --model_id "Exchange_${PRED_LEN}_lr${LR}"
  done
  echo "Exchange lr=${LR} done"
done

# =============================================================================
# ELECTRICITY
# Current: 0.005  |  Try: 0.001, 0.0005
# =============================================================================
for LR in 0.001 0.0005; do
  for PRED_LEN in 96 192 336 720; do
    run_one "Electricity_pl${PRED_LEN}_lr${LR}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path electricity.csv --data custom \
      --features M --seq_len 96 --label_len 48 \
      --pred_len "${PRED_LEN}" --enc_in 321 \
      --d_model 64 --t_ff 128 --c_ff 80 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 256 --learning_rate "${LR}" \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed 2021 --des Exp \
      --model_id "Electricity_${PRED_LEN}_lr${LR}"
  done
  echo "Electricity lr=${LR} done"
done

# =============================================================================
# SOLAR
# Uses Dataset_Solar (data=Solar, .txt file, no date column, no timeenc).
# Current: 0.005  |  Try: 0.001, 0.0005
# =============================================================================
for LR in 0.001 0.0005; do
  for PRED_LEN in 96 192 336 720; do
    run_one "Solar_pl${PRED_LEN}_lr${LR}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path solar.txt --data Solar \
      --features M --seq_len 96 --label_len 48 \
      --pred_len "${PRED_LEN}" --enc_in 137 \
      --d_model 64 --t_ff 128 --c_ff 34 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 512 --learning_rate "${LR}" \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed 2021 --des Exp \
      --model_id "Solar_${PRED_LEN}_lr${LR}"
  done
  echo "Solar lr=${LR} done"
done

# =============================================================================
# ILI
# Special params: seq_len=36, label_len=18, patch_len=6, stride=3,
#                 dw_kernel=3, lradj=type3, pred_lens={24,36,48,60}
#                 use_decomp=0, use_seas_stream=0, use_fusion_gate=0
# Current: 0.01  |  Try: 0.005, 0.001
# =============================================================================
for LR in 0.005 0.001; do
  for PRED_LEN in 24 36 48 60; do
    run_one "ILI_pl${PRED_LEN}_lr${LR}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path national_illness.csv --data custom \
      --features M --seq_len 36 --label_len 18 \
      --pred_len "${PRED_LEN}" --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
      --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
      --batch_size 32 --learning_rate "${LR}" \
      --lradj type3 --train_epochs 100 --patience 10 \
      --model PatchLinear --seed 2021 --des Exp \
      --model_id "ILI_${PRED_LEN}_lr${LR}"
  done
  echo "ILI lr=${LR} done"
done

# =============================================================================
echo ""
echo "LR tuning complete."
echo "Results : ${SCRIPT_DIR}/result.txt"
echo "Logs    : ${LOG_DIR}/"

if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_lr_tuning_${STAMP}.txt"
  echo "Backed up to Drive"
fi
