#!/usr/bin/env bash
# Electricity d_model phase 2 — each run gets its own log file
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/elec_dmodel2"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local PL=$1; shift
  local LOG="${LOG_DIR}/${TAG}_pl${PL}.log"
  echo ">>> Elec pl${PL} ${TAG} — logging to ${LOG}"
  python -u run.py "$@" --pred_len ${PL} \
    --model_id "Elec_${TAG}_pl${PL}" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || echo "  [FAILED — check ${LOG}]"
}

BASE="--is_training 1 --root_path ./dataset/
  --data_path electricity.csv --data custom
  --features M --seq_len 96 --label_len 48 --enc_in 321
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
  --use_cross_channel 1 --use_alpha_gate 1
  --c_ff 128 --alpha 0.3 --batch_size 256 --learning_rate 0.01
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --seed 2021 --des Exp"

for PL in 192 720; do

  # d=320, default patch/dk
  run_one "d320_t640_p16s8_dk7" ${PL} $BASE \
    --d_model 320 --t_ff 640 --patch_len 16 --stride 8 --dw_kernel 7

  # d=384, default patch/dk
  run_one "d384_t768_p16s8_dk7" ${PL} $BASE \
    --d_model 384 --t_ff 768 --patch_len 16 --stride 8 --dw_kernel 7

  # d=256 + p32s16 (larger patch helped H=720 at d64)
  run_one "d256_t512_p32s16_dk7" ${PL} $BASE \
    --d_model 256 --t_ff 512 --patch_len 32 --stride 16 --dw_kernel 7

  # d=256 + dk13 (wider kernel helped H=720 at d64)
  run_one "d256_t512_p16s8_dk13" ${PL} $BASE \
    --d_model 256 --t_ff 512 --patch_len 16 --stride 8 --dw_kernel 13

  # d=256 + p32s16 + dk13 (combined)
  run_one "d256_t512_p32s16_dk13" ${PL} $BASE \
    --d_model 256 --t_ff 512 --patch_len 32 --stride 16 --dw_kernel 13

  # d=320 + p32s16 (larger d + larger patch)
  run_one "d320_t640_p32s16_dk7" ${PL} $BASE \
    --d_model 320 --t_ff 640 --patch_len 32 --stride 16 --dw_kernel 7

done

echo ""
echo "=== Results summary ==="
echo "H=192 (target: beat 0.150, CARD=0.160):"
grep "mse:" "$LOG_DIR"/*_pl192.log | grep -oP 'pl192.*mse:\K[\d.]+' | \
  paste - <(ls "$LOG_DIR"/*_pl192.log | xargs -I{} basename {}) | \
  sort -n | head -10 || true

echo ""
echo "H=720 (target: reach CARD=0.197):"
for f in "$LOG_DIR"/*_pl720.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "?")
  echo "  $(basename $f .log): $MSE"
done | sort -t: -k2 -n
