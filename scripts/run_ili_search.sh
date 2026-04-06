#!/usr/bin/env bash
# =============================================================================
# ILI comprehensive hyperparameter search
#
# Phase 1: Single-seed sweep on H=36 (worst gap vs xPatch 1.315)
#   Active params: d_model, t_ff, c_ff, use_cross_channel, use_alpha_gate, lr
#   patch/stride/dw_kernel/alpha are irrelevant (seas stream disabled)
#
# Phase 2 (manual): run top 5 configs from Phase 1 on all 4 horizons × 3 seeds
#
# Usage: bash scripts/run_ili_search.sh
# Results: logs/ili_search/ and result_ili_search.txt
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/ili_search"
mkdir -p "$LOG_DIR"
RESULT_FILE="$SCRIPT_DIR/result_ili_search.txt"
> "$RESULT_FILE"

SEED=2021
PL=36  # focus on H=36 — worst gap (ours 1.454 vs xPatch 1.315)

ILI_BASE="--is_training 1 --root_path ./dataset/
  --data_path national_illness.csv --data custom
  --features M --seq_len 36 --label_len 18 --pred_len ${PL} --enc_in 7
  --patch_len 6 --stride 3 --dw_kernel 3 --alpha 0.3
  --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0
  --batch_size 32 --lradj type3
  --train_epochs 100 --patience 10
  --model PatchLinear --seed ${SEED} --des Exp"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  python -u run.py "$@" > "$LOG" 2>&1
  MSE=$(grep "mse:" "$LOG" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "FAILED")
  MAE=$(grep "mse:" "$LOG" | tail -1 | grep -oP 'mae:\K[\d.]+' || echo "FAILED")
  echo "${TAG}: mse=${MSE} mae=${MAE}"
  echo "${TAG}: mse=${MSE} mae=${MAE}" >> "$RESULT_FILE"
}

echo "=== Phase 1: ILI H=36 single-seed sweep ===" | tee -a "$RESULT_FILE"
echo "Baseline (d64,t128,c16,cc1,ag1,lr0.01): ~1.454"
echo ""

# ── d_model sweep (t_ff=2*d_model, c_ff=16, cc=1, ag=1, lr=0.01) ────────────
echo "--- d_model sweep ---" | tee -a "$RESULT_FILE"
for D in 16 32 48 64 96 128 192; do
  T=$((D*2))
  run_one "d${D}_t${T}_c16_cc1_ag1_lr001" $ILI_BASE \
    --d_model ${D} --t_ff ${T} --c_ff 16 \
    --use_cross_channel 1 --use_alpha_gate 1 --learning_rate 0.01 \
    --model_id "ILI_36_d${D}_t${T}_c16"
done

# ── t_ff ratio sweep (d=64) ───────────────────────────────────────────────────
echo "--- t_ff ratio sweep (d=64) ---" | tee -a "$RESULT_FILE"
for T in 32 64 96 128 192 256 320; do
  run_one "d64_t${T}_c16_cc1_ag1_lr001" $ILI_BASE \
    --d_model 64 --t_ff ${T} --c_ff 16 \
    --use_cross_channel 1 --use_alpha_gate 1 --learning_rate 0.01 \
    --model_id "ILI_36_d64_t${T}_c16"
done

# ── c_ff sweep (d=64, t=128) ──────────────────────────────────────────────────
echo "--- c_ff sweep ---" | tee -a "$RESULT_FILE"
for C in 1 2 4 7 8 14 16 28 32; do
  run_one "d64_t128_c${C}_cc1_ag1_lr001" $ILI_BASE \
    --d_model 64 --t_ff 128 --c_ff ${C} \
    --use_cross_channel 1 --use_alpha_gate 1 --learning_rate 0.01 \
    --model_id "ILI_36_d64_t128_c${C}"
done

# ── use_cross_channel / use_alpha_gate combinations ───────────────────────────
echo "--- cc/ag flag combinations (d=64) ---" | tee -a "$RESULT_FILE"
for CC in 0 1; do
  for AG in 0 1; do
    run_one "d64_t128_c16_cc${CC}_ag${AG}_lr001" $ILI_BASE \
      --d_model 64 --t_ff 128 --c_ff 16 \
      --use_cross_channel ${CC} --use_alpha_gate ${AG} --learning_rate 0.01 \
      --model_id "ILI_36_d64_cc${CC}_ag${AG}"
  done
done

# ── lr sweep (d=64, t=128, c=16, cc=1, ag=1) ─────────────────────────────────
echo "--- lr sweep ---" | tee -a "$RESULT_FILE"
for LR in 0.001 0.003 0.005 0.008 0.01 0.015 0.02 0.03 0.05; do
  run_one "d64_t128_c16_cc1_ag1_lr${LR}" $ILI_BASE \
    --d_model 64 --t_ff 128 --c_ff 16 \
    --use_cross_channel 1 --use_alpha_gate 1 --learning_rate ${LR} \
    --model_id "ILI_36_d64_lr${LR}"
done

# ── best d_model combinations with varying t_ff and lr ───────────────────────
echo "--- promising combos ---" | tee -a "$RESULT_FILE"
for D in 32 96; do
  for T_RATIO in 1 2 4; do
    T=$((D*T_RATIO))
    for LR in 0.005 0.01 0.02; do
      run_one "d${D}_t${T}_c16_cc1_ag1_lr${LR}" $ILI_BASE \
        --d_model ${D} --t_ff ${T} --c_ff 16 \
        --use_cross_channel 1 --use_alpha_gate 1 --learning_rate ${LR} \
        --model_id "ILI_36_d${D}_t${T}_lr${LR}"
    done
  done
done

echo ""
echo "=== Phase 1 complete. Check $RESULT_FILE for results ===" | tee -a "$RESULT_FILE"
echo "Sort by MSE:"
grep "mse=" "$RESULT_FILE" | grep -v "^==\|^--\|^Base" | \
  sed 's/.*mse=//;s/ mae=/ /' | \
  awk '{print $1, $0}' | sort -n | head -10
