#!/usr/bin/env bash
# Multi-seed runs — Traffic
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
    run_one "Traffic_pl${PRED_LEN}_s${SEED}" \
      --is_training 1 --root_path ./dataset/ \
      --data_path traffic.csv --data custom \
      --features M --seq_len 96 --label_len 48 \
      --pred_len "$PRED_LEN" --enc_in 862 \
      --d_model 64 --t_ff 128 --c_ff 128 \
      --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
      --batch_size 96 --learning_rate 0.005 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed "$SEED" --des Exp \
      --model_id "Traffic_${PRED_LEN}_s${SEED}"
  done
  echo "Traffic seed=${SEED} done"
done

echo "Traffic multi-seed complete"
if [ -d "/content/drive/MyDrive" ]; then
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_seeds_traffic_$(date +%Y%m%d_%H%M%S).txt"
  echo "Backed up to Drive"
fi
