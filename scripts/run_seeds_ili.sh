#!/usr/bin/env bash
# Multi-seed runs — ILI (lr=0.01, simplified config)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/multiseed"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || true
}

for SEED in 2021 2022 2023; do
  for PRED_LEN in 24 36 48 60; do
    run_one "ILI_pl${PRED_LEN}_s${SEED}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path national_illness.csv --data custom \
      --features M --seq_len 36 --label_len 18 \
      --pred_len "$PRED_LEN" --enc_in 7 \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3 \
      --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
      --batch_size 32 --learning_rate 0.01 \
      --lradj type3 --train_epochs 100 --patience 10 \
      --model PatchLinear --seed "$SEED" --des Exp \
      --model_id "ILI_${PRED_LEN}_s${SEED}"
  done
  echo "ILI seed=${SEED} done"
done

echo "ILI multi-seed complete"
if [ -d "/content/drive/MyDrive" ]; then
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_seeds_ili_$(date +%Y%m%d_%H%M%S).txt"
  echo "Backed up to Drive"
fi
