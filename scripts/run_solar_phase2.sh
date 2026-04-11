#!/usr/bin/env bash
# Solar Phase 2 — 4 best configs × all 4 horizons × 3 seeds
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/solar_phase2"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; local PL=$2; local SEED=$3; shift 3
  local LOG="${LOG_DIR}/${TAG}_pl${PL}_s${SEED}.log"
  echo ">>> Solar pl${PL} ${TAG} s${SEED}"
  python -u run.py "$@" --pred_len ${PL} --seed ${SEED} \
    --model_id "Solar_${TAG}_pl${PL}_s${SEED}" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || echo "  [FAILED — check ${LOG}]"
}

BASE="--is_training 1 --root_path ./dataset/
  --data_path solar.txt --data Solar
  --features M --seq_len 96 --label_len 48 --enc_in 137
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
  --use_cross_channel 1 --use_alpha_gate 1
  --alpha 0.3 --batch_size 512 --learning_rate 0.0008
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --des Exp"

for SEED in 2021 2022 2023; do
  for PL in 96 192 336 720; do

    # 1. d192 — best H96 (0.178), H720 in ul range (0.247)
    run_one "d192" ${PL} ${SEED} $BASE \
      --d_model 192 --t_ff 384 --c_ff 34 --patch_len 16 --stride 8 --dw_kernel 7

    # 2. d160 — safest, best H720 (0.237), solid H96 (0.186)
    run_one "d160" ${PL} ${SEED} $BASE \
      --d_model 160 --t_ff 320 --c_ff 34 --patch_len 16 --stride 8 --dw_kernel 7

    # 3. d128_p8s4 — ties d192 on H96 (0.178), different balance
    run_one "d128_p8s4" ${PL} ${SEED} $BASE \
      --d_model 128 --t_ff 256 --c_ff 34 --patch_len 8 --stride 4 --dw_kernel 7

    # 4. d256_p8s4 — best H720 combo (0.241), H96 borderline
    run_one "d256_p8s4" ${PL} ${SEED} $BASE \
      --d_model 256 --t_ff 512 --c_ff 34 --patch_len 8 --stride 4 --dw_kernel 7

  done
  echo "Seed ${SEED} done"
done

echo ""
echo "=== PHASE 2 RESULTS ==="
echo "Baseline: H96=0.194 H192=0.226 H336=0.241 H720=0.257"
echo "TimeMixer: H96=0.189 H192=0.222 H336=0.231 H720=0.223"
echo ""
for CFG in d192 d160 d128_p8s4 d256_p8s4; do
  echo "--- ${CFG} ---"
  for PL in 96 192 336 720; do
    MEAN=$(for SEED in 2021 2022 2023; do
      grep "mse:" "${LOG_DIR}/${CFG}_pl${PL}_s${SEED}.log" 2>/dev/null | \
        tail -1 | grep -oP 'mse:\K[\d.]+' || echo "0"
    done | awk '{s+=$1;n++} END{printf "%.4f",s/n}')
    echo "  H=${PL}: mean MSE=${MEAN}"
  done
done
