#!/usr/bin/env bash
# =============================================================================
# PatchLinear — unified experiment runner
#
# Usage (from any directory):
#   bash scripts/run_experiments.sh                  # full model only
#   bash scripts/run_experiments.sh --ablations      # full model + ablations
#   bash scripts/run_experiments.sh --ablations-only # ablations only
#   bash scripts/run_experiments.sh --seeds          # full model, 3 seeds
#   bash scripts/run_experiments.sh --ablations --seeds
#
# Output always written to the PROJECT ROOT (one level above scripts/).
# Log files : PROJECT_ROOT/logs/<tag>.log
# Results   : PROJECT_ROOT/result.txt
# =============================================================================

set -euo pipefail

# -- Resolve project root (one level above this script) ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# -- Parse flags -------------------------------------------------------------
RUN_FULL=1
RUN_ABLATIONS=0
MULTI_SEED=0
for arg in "$@"; do
  case $arg in
    --ablations)       RUN_ABLATIONS=1 ;;
    --ablations-only)  RUN_ABLATIONS=1; RUN_FULL=0 ;;
    --seeds)           MULTI_SEED=1 ;;
  esac
done

# -- Shared defaults ---------------------------------------------------------
MODEL_NAME=PatchLinear
ALPHA=0.3
TRAIN_EPOCHS=100
PATIENCE=10
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# -- Helper: c_ff = max(16, min(enc_in // 4, 128)) --------------------------
c_ff_for() {
  local v=$(( $1 / 4 ))
  [ $v -lt 16 ]  && v=16
  [ $v -gt 128 ] && v=128
  echo $v
}

# -- run_single(tag, args...) ------------------------------------------------
# Writes full output to LOG_DIR/tag.log.
# result.txt lands in SCRIPT_DIR (project root) because we cd'd there above.
run_single() {
  local tag="$1"; shift
  local log_file="${LOG_DIR}/${tag}.log"
  echo ">>> ${tag}"
  python -u run.py "$@" > "$log_file" 2>&1
  grep "mse:" "$log_file" | tail -1 || true
}

# =============================================================================
# STANDARD DATASETS
# Fields: name  data_path  data_type  enc_in  batch  lr  pred_lens
#
# Shared across all standard datasets:
#   seq_len=96, label_len=48, patch_len=16, stride=8, dw_kernel=7, lradj=sigmoid
# =============================================================================
declare -a DATASETS=(
  "ETTh1       ETTh1.csv            ETTh1      7    2048  0.0005   96,192,336,720"
  "ETTh2       ETTh2.csv            ETTh2      7    2048  0.0005   96,192,336,720"
  "ETTm1       ETTm1.csv            ETTm1      7    2048  0.0005   96,192,336,720"
  "ETTm2       ETTm2.csv            ETTm2      7    2048  0.0001   96,192,336,720"
  "Weather     weather.csv          custom     21   2048  0.0005   96,192,336,720"
  "Traffic     traffic.csv          custom     862  96    0.005    96,192,336,720"
  "Electricity electricity.csv      custom     321  256   0.005    96,192,336,720"
  "Exchange    exchange_rate.csv    custom     8    32    0.00001  96,192,336,720"
  "Solar       solar.txt            Solar      137  512   0.005    96,192,336,720"
)

# =============================================================================
# ABLATION CONFIGURATIONS
# Each entry changes exactly one flag — unambiguous attribution.
# =============================================================================
declare -a ABLATION_CONFIGS=(
  "A1_no_decomp        --use_decomp 0"
  "A2a_trend_only      --use_seas_stream 0 --use_fusion_gate 0"
  "A2b_seasonal_only   --use_trend_stream 0 --use_fusion_gate 0"
  "A3_no_fusion_gate   --use_fusion_gate 0"
  "A4a_no_cross_ch     --use_cross_channel 0"
  "A4b_no_alpha        --use_alpha_gate 0"
  "A6_dw_k3            --dw_kernel 3"
  "A6_dw_k13           --dw_kernel 13"
)

SEEDS=(2021 2022 2023)

# =============================================================================
# ILI helper function
#
# ILI uses different parameters from all standard datasets:
#   seq_len=36    only 36 weekly observations available as lookback
#   label_len=18  decoder start token
#   pred_lens     {24,36,48,60} — weekly horizons, not quarterly
#   patch_len=6   shorter patches to fit the short sequence
#   stride=3      half of patch_len for overlap
#   dw_kernel=3   ERF=(3-1)*3+6=12 steps, one third of input — appropriate scale
#   batch_size=32 small dataset; large batches cause instability
#   lr=0.01       higher lr for fast convergence on small data
#   lradj=type3   cosine decay works better than sigmoid on small data
#   use_decomp=0  EMA decomposition hurts on spiky epidemic data (confirmed
#                 by ablation: full model 1.737 vs simplified 1.526 avg MSE)
#   use_seas_stream=0   overfits on <700 training windows
#   use_fusion_gate=0   no stream to fuse when seasonal stream is off
# =============================================================================
run_ili() {
  local seed=$1
  local tag_suffix=$2
  local extra_flags=()
  if [ $# -ge 3 ]; then extra_flags=("${@:3}"); fi

  for pred_len in 24 36 48 60; do
    local tag="ILI_pl${pred_len}${tag_suffix}"
    run_single "$tag" \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path national_illness.csv \
      --model_id "ILI_${pred_len}${tag_suffix}" \
      --model "${MODEL_NAME}" \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len "${pred_len}" \
      --enc_in 7 \
      --d_model 64 \
      --t_ff 128 \
      --c_ff 16 \
      --patch_len 6 \
      --stride 3 \
      --dw_kernel 3 \
      --alpha "${ALPHA}" \
      --use_decomp 0 \
      --use_seas_stream 0 \
      --use_fusion_gate 0 \
      --batch_size 32 \
      --learning_rate 0.01 \
      --lradj type3 \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --seed "${seed}" \
      --des Exp \
      "${extra_flags[@]+"${extra_flags[@]}"}"
  done
}

# =============================================================================
# Main loop — standard datasets
# =============================================================================
for dataset_entry in "${DATASETS[@]}"; do
  read -r NAME DATA_PATH DATA_TYPE ENC_IN BATCH_SIZE LR PRED_LENS_CSV \
    <<< "$dataset_entry"

  C_FF=$(c_ff_for "$ENC_IN")
  IFS=',' read -ra PRED_LENS <<< "$PRED_LENS_CSV"

  for PRED_LEN in "${PRED_LENS[@]}"; do

    COMMON=(
      --is_training 1
      --root_path ./dataset/
      --data_path "${DATA_PATH}"
      --data "${DATA_TYPE}"
      --features M
      --seq_len 96
      --label_len 48
      --pred_len "${PRED_LEN}"
      --enc_in "${ENC_IN}"
      --d_model 64
      --t_ff 128
      --c_ff "${C_FF}"
      --patch_len 16
      --stride 8
      --dw_kernel 7
      --alpha "${ALPHA}"
      --batch_size "${BATCH_SIZE}"
      --learning_rate "${LR}"
      --lradj sigmoid
      --train_epochs "${TRAIN_EPOCHS}"
      --patience "${PATIENCE}"
      --model "${MODEL_NAME}"
      --des Exp
    )

    # -- Full model -----------------------------------------------------------
    if [ "$RUN_FULL" -eq 1 ]; then
      if [ "$MULTI_SEED" -eq 1 ]; then
        for SEED in "${SEEDS[@]}"; do
          run_single "${NAME}_pl${PRED_LEN}_full_s${SEED}" \
            --model_id "${NAME}_${PRED_LEN}_full_s${SEED}" \
            --seed "${SEED}" "${COMMON[@]}"
        done
      else
        run_single "${NAME}_pl${PRED_LEN}_full" \
          --model_id "${NAME}_${PRED_LEN}_full" \
          --seed 2021 "${COMMON[@]}"
      fi
    fi

    # -- Ablations ------------------------------------------------------------
    if [ "$RUN_ABLATIONS" -eq 1 ]; then
      for ablation_entry in "${ABLATION_CONFIGS[@]}"; do
        ABLATION_NAME="${ablation_entry%%  *}"
        ABLATION_ARGS="${ablation_entry#*  }"
        # shellcheck disable=SC2086
        run_single "${NAME}_pl${PRED_LEN}_${ABLATION_NAME}" \
          --model_id "${NAME}_${PRED_LEN}_${ABLATION_NAME}" \
          --seed 2021 "${COMMON[@]}" \
          $ABLATION_ARGS
      done
    fi

  done
done

# =============================================================================
# ILI
# =============================================================================
if [ "$RUN_FULL" -eq 1 ]; then
  if [ "$MULTI_SEED" -eq 1 ]; then
    for SEED in "${SEEDS[@]}"; do
      run_ili "${SEED}" "_full_s${SEED}"
    done
  else
    run_ili 2021 "_full"
  fi
fi

if [ "$RUN_ABLATIONS" -eq 1 ]; then
  for ablation_entry in "${ABLATION_CONFIGS[@]}"; do
    ABLATION_NAME="${ablation_entry%%  *}"
    ABLATION_ARGS="${ablation_entry#*  }"
    # shellcheck disable=SC2086
    run_ili 2021 "_${ABLATION_NAME}" $ABLATION_ARGS
  done
fi

# =============================================================================
echo ""
echo "All experiments complete."
echo "Results : ${SCRIPT_DIR}/result.txt"
echo "Logs    : ${LOG_DIR}/"

# Optional: back up result.txt to Drive if mounted
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_result_${STAMP}.txt"
  echo "Backed up to Drive: PatchLinear_result_${STAMP}.txt"
fi
