#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Univariate forecasting (features=S)
# Updated with final per-dataset lr/patch configs
#
# Notes:
#   - enc_in=1: d_model scaling for multivariate doesn't apply here
#   - Solar excluded (Dataset_Solar incompatible with features=S)
#   - Exchange uses tuned lr=2e-6, patch=8, stride=4, dk=3 (temporal tuning)
#   - ILI uses horizons {24,36,48,60}, seq_len=36
#
# Usage:
#   bash scripts/run_univariate.sh           # single seed 2021
#   bash scripts/run_univariate.sh --seeds   # 3 seeds 2021/2022/2023
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MULTI_SEED=0
for arg in "$@"; do case $arg in --seeds) MULTI_SEED=1 ;; esac; done
SEEDS=(2021 2022 2023)

LOG_DIR="$SCRIPT_DIR/logs/univariate"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
}

run_seeds() {
  local NAME=$1; local PL=$2; shift 2
  if [ "$MULTI_SEED" -eq 1 ]; then
    for SEED in "${SEEDS[@]}"; do
      run_one "${NAME}_pl${PL}_s${SEED}" --seed $SEED --model_id "${NAME}_S_${PL}_s${SEED}" "$@" --pred_len $PL
    done
  else
    run_one "${NAME}_pl${PL}" --seed 2021 --model_id "${NAME}_S_${PL}" "$@" --pred_len $PL
  fi
}

BASE="--is_training 1 --root_path ./dataset/ --features S
      --seq_len 96 --label_len 48 --enc_in 1
      --d_model 64 --t_ff 128 --c_ff 16
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3
      --lradj sigmoid --train_epochs 100 --patience 10
      --model PatchLinear --des Exp"

# ── ETTh1 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_seeds ETTh1 $PL $BASE --data_path ETTh1.csv --data ETTh1 \
    --batch_size 2048 --learning_rate 0.0005
done

# ── ETTh2 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_seeds ETTh2 $PL $BASE --data_path ETTh2.csv --data ETTh2 \
    --batch_size 2048 --learning_rate 0.0005
done

# ── ETTm1 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_seeds ETTm1 $PL $BASE --data_path ETTm1.csv --data ETTm1 \
    --batch_size 2048 --learning_rate 0.0005
done

# ── ETTm2 ─────────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_seeds ETTm2 $PL $BASE --data_path ETTm2.csv --data ETTm2 \
    --batch_size 2048 --learning_rate 0.0001
done

# ── Weather ───────────────────────────────────────────────────────────────────
for PL in 96 192 336 720; do
  run_seeds Weather $PL $BASE --data_path weather.csv --data custom \
    --batch_size 2048 --learning_rate 0.0005
done

# ── Traffic ───────────────────────────────────────────────────────────────────
# Note: enc_in=1 for univariate — d=256 doesn't apply
for PL in 96 192 336 720; do
  run_seeds Traffic $PL $BASE --data_path traffic.csv --data custom \
    --batch_size 256 --learning_rate 0.005
done

# ── Electricity ───────────────────────────────────────────────────────────────
# Note: enc_in=1 for univariate — d=256/c_ff=128 don't apply
for PL in 96 192 336 720; do
  run_seeds Electricity $PL $BASE --data_path electricity.csv --data custom \
    --batch_size 256 --learning_rate 0.005
done

# ── Exchange (tuned: lr=2e-6, patch=8, stride=4, dk=3) ───────────────────────
for PL in 96 192 336 720; do
  run_seeds Exchange $PL \
    --is_training 1 --root_path ./dataset/ --features S \
    --seq_len 96 --label_len 48 --enc_in 1 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 8 --stride 4 --dw_kernel 3 --alpha 0.3 \
    --batch_size 32 --learning_rate 0.000002 \
    --lradj sigmoid --train_epochs 100 --patience 10 \
    --model PatchLinear --des Exp \
    --data_path exchange_rate.csv --data custom
done

# ── ILI (horizons 24/36/48/60, seq_len=36) ────────────────────────────────────
# Note: simplified config (no seas/decomp/fusion) — enc_in=1, c_ff=8 irrelevant
for PL in 24 36 48 60; do
  run_seeds ILI $PL \
    --is_training 1 --root_path ./dataset/ --features S \
    --seq_len 36 --label_len 18 --enc_in 1 \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
    --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
    --batch_size 32 --learning_rate 0.01 \
    --lradj type3 --train_epochs 100 --patience 10 \
    --model PatchLinear --des Exp \
    --data_path national_illness.csv --data custom
done

echo "Univariate experiments complete."
