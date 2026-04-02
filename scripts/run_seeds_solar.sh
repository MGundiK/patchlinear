#!/usr/bin/env bash
# Multi-seed runs — Solar (updated lr=0.003)
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
  for PRED_LEN in 96 192 336 720; do
    run_one "Solar_pl${PRED_LEN}_s${SEED}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path solar.txt --data Solar \
      --features M --seq_len 96 --label_len 48 \
      --pred_len "$PRED_LEN" --enc_in 137 \
      --d_model 64 --t_ff 128 --c_ff 34 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 512 --learning_rate 0.003 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed "$SEED" --des Exp \
      --model_id "Solar_${PRED_LEN}_s${SEED}"
  done
  echo "Solar seed=${SEED} done"
done

echo "Solar multi-seed complete"
if [ -d "/content/drive/MyDrive" ]; then
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_seeds_solar_$(date +%Y%m%d_%H%M%S).txt"
  echo "Backed up to Drive"
fi
