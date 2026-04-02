#!/usr/bin/env bash
# Multi-seed runs — ETT family (ETTh1, ETTh2, ETTm1, ETTm2)
# Seeds: 2021, 2022, 2023
# Usage: bash scripts/run_seeds_ett.sh

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
  for DS_ENTRY in \
    "ETTh1 ETTh1.csv ETTh1 7 2048 0.0005" \
    "ETTh2 ETTh2.csv ETTh2 7 2048 0.0005" \
    "ETTm1 ETTm1.csv ETTm1 7 2048 0.0005" \
    "ETTm2 ETTm2.csv ETTm2 7 2048 0.0001"
  do
    read -r NAME DATA_PATH DATA_TYPE ENC_IN BATCH LR <<< "$DS_ENTRY"
    for PRED_LEN in 96 192 336 720; do
      run_one "${NAME}_pl${PRED_LEN}_s${SEED}" \
        --is_training 1 --root_path ./dataset/ \
        --data_path "$DATA_PATH" --data "$DATA_TYPE" \
        --features M --seq_len 96 --label_len 48 \
        --pred_len "$PRED_LEN" --enc_in "$ENC_IN" \
        --d_model 64 --t_ff 128 --c_ff 16 \
        --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
        --batch_size "$BATCH" --learning_rate "$LR" \
        --lradj sigmoid --train_epochs 100 --patience 10 \
        --model PatchLinear --seed "$SEED" --des Exp \
        --model_id "${NAME}_${PRED_LEN}_s${SEED}"
    done
    echo "${NAME} seed=${SEED} done"
  done
done

echo "ETT multi-seed complete"
if [ -d "/content/drive/MyDrive" ]; then
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_seeds_ett_$(date +%Y%m%d_%H%M%S).txt"
  echo "Backed up to Drive"
fi
