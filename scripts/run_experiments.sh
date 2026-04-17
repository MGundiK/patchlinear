#!/usr/bin/env bash
# =============================================================================
# PatchLinear — main experiments (final tuned configs)
#
# Key per-dataset configs:
#   ETTh1/h2/m1   lr=0.0005, patch=16, stride=8, dw_k=7, c_ff=16
#   ETTm2         lr=0.0001, patch=16, stride=8, dw_k=7, c_ff=16
#   Weather       lr=0.0005, patch=16, stride=8, dw_k=7, c_ff=16
#   Traffic       lr=0.005,  patch=16, stride=8, dw_k=7, c_ff=128
#   Electricity   lr=0.01,   patch=16, stride=8, dw_k=7, c_ff=128  ← updated
#   Exchange      lr=2e-6,   patch=8,  stride=4, dw_k=3, c_ff=16   ← updated
#   Solar         lr=0.0008, patch=16, stride=8, dw_k=7, c_ff=34   ← updated
#   ILI           lr=0.01,   patch=6,  stride=3, dw_k=3, c_ff=16
#                 use_decomp=0, use_seas_stream=0, use_fusion_gate=0
#
# Usage:
#   bash scripts/run_experiments.sh           # single seed (2021)
#   bash scripts/run_experiments.sh --seeds   # 3 seeds (2021,2022,2023)
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MULTI_SEED=0
for arg in "$@"; do case $arg in --seeds) MULTI_SEED=1;; esac; done
SEEDS=(2021 2022 2023)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
}

run_dataset() {
  local NAME=$1; shift
  local PRED_LENS=$1; shift
  IFS=',' read -ra HORIZONS <<< "$PRED_LENS"
  if [ "$MULTI_SEED" -eq 1 ]; then
    for SEED in "${SEEDS[@]}"; do
      for PL in "${HORIZONS[@]}"; do
        run_one "${NAME}_pl${PL}_s${SEED}" --seed "$SEED" --model_id "${NAME}_${PL}_s${SEED}" "$@" --pred_len "$PL"
      done
    done
  else
    for PL in "${HORIZONS[@]}"; do
      run_one "${NAME}_pl${PL}" --seed 2021 --model_id "${NAME}_${PL}" "$@" --pred_len "$PL"
    done
  fi
  echo "${NAME} done"
}

COMMON="--is_training 1 --root_path ./dataset/ --features M --seq_len 96 --label_len 48
        --d_model 64 --t_ff 128 --lradj sigmoid --train_epochs 100 --patience 10
        --model PatchLinear --des Exp"

# ── ETTh1 ─────────────────────────────────────────────────────────────────────
run_dataset ETTh1 96,192,336,720 $COMMON \
  --data_path ETTh1.csv --data ETTh1 --enc_in 7 \
  --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --batch_size 2048 --learning_rate 0.0005

# ── ETTh2 ─────────────────────────────────────────────────────────────────────
run_dataset ETTh2 96,192,336,720 $COMMON \
  --data_path ETTh2.csv --data ETTh2 --enc_in 7 \
  --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --batch_size 2048 --learning_rate 0.0005

# ── ETTm1 ─────────────────────────────────────────────────────────────────────
run_dataset ETTm1 96,192,336,720 $COMMON \
  --data_path ETTm1.csv --data ETTm1 --enc_in 7 \
  --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --batch_size 2048 --learning_rate 0.0005

# ── ETTm2 ─────────────────────────────────────────────────────────────────────
run_dataset ETTm2 96,192,336,720 $COMMON \
  --data_path ETTm2.csv --data ETTm2 --enc_in 7 \
  --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --batch_size 2048 --learning_rate 0.0001

# ── Weather ───────────────────────────────────────────────────────────────────
run_dataset Weather 96,192,336,720 $COMMON \
  --data_path weather.csv --data custom --enc_in 21 \
  --c_ff 16 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --batch_size 2048 --learning_rate 0.0005

# ── Traffic (updated: d_model=256, t_ff=512, batch=64, patience=5) ────────────
run_dataset Traffic 96,192,336,720 \
  --is_training 1 --root_path ./dataset/ --features M --seq_len 96 --label_len 48 \
  --data_path traffic.csv --data custom --enc_in 862 \
  --d_model 256 --t_ff 512 --c_ff 128 \
  --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
  --use_cross_channel 1 --use_alpha_gate 1 \
  --batch_size 64 --learning_rate 0.005 \
  --lradj sigmoid --train_epochs 100 --patience 5 \
  --model PatchLinear --des Exp

# ── Electricity (updated: d_model=256, t_ff=512, c_ff=128, lr=0.01) ────────────
run_dataset Electricity 96,192,336,720 \
  --is_training 1 --root_path ./dataset/ --features M --seq_len 96 --label_len 48 \
  --data_path electricity.csv --data custom --enc_in 321 \
  --d_model 256 --t_ff 512 --c_ff 128 \
  --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
  --use_cross_channel 1 --use_alpha_gate 1 \
  --batch_size 256 --learning_rate 0.01 \
  --lradj sigmoid --train_epochs 100 --patience 10 \
  --model PatchLinear --des Exp

# ── Exchange (updated: patch=8, stride=4, dw_k=3, lr=2e-6) ───────────────────
run_dataset Exchange 96,192,336,720 $COMMON \
  --data_path exchange_rate.csv --data custom --enc_in 8 \
  --c_ff 16 --patch_len 8 --stride 4 --dw_kernel 3 --alpha 0.3 \
  --batch_size 32 --learning_rate 0.000002

# ── Solar (updated: d_model=192, t_ff=384, lr=0.0008) ────────────────────────
run_dataset Solar 96,192,336,720 \
  --is_training 1 --root_path ./dataset/ --features M --seq_len 96 --label_len 48 \
  --data_path solar.txt --data Solar --enc_in 137 \
  --d_model 192 --t_ff 384 \
  --c_ff 34 --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
  --use_cross_channel 1 --use_alpha_gate 1 \
  --batch_size 512 --learning_rate 0.0008 \
  --lradj sigmoid --train_epochs 100 --patience 10 \
  --model PatchLinear --des Exp

# ── ILI (updated: simplified config, lr=0.01, type3 lradj) ───────────────────
ILI_COMMON="--is_training 1 --root_path ./dataset/ --features M
            --data_path national_illness.csv --data custom --enc_in 7
            --seq_len 36 --label_len 18
            --d_model 64 --t_ff 128 --c_ff 8
            --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3
            --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0
            --batch_size 32 --learning_rate 0.01
            --lradj type3 --train_epochs 100 --patience 10
            --model PatchLinear --des Exp"

if [ "$MULTI_SEED" -eq 1 ]; then
  for SEED in "${SEEDS[@]}"; do
    for PL in 24 36 48 60; do
      run_one "ILI_pl${PL}_s${SEED}" --seed "$SEED" --model_id "ILI_${PL}_s${SEED}" $ILI_COMMON --pred_len "$PL"
    done
  done
else
  for PL in 24 36 48 60; do
    run_one "ILI_pl${PL}" --seed 2021 --model_id "ILI_${PL}" $ILI_COMMON --pred_len "$PL"
  done
fi
echo "ILI done"

echo ""
echo "All experiments complete."
if [ -d "/content/drive/MyDrive" ]; then
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_results_$(date +%Y%m%d_%H%M%S).txt"
  echo "Backed up to Drive"
fi
