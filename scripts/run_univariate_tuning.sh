#!/usr/bin/env bash
# =============================================================================
# Univariate tuning — targeting datasets where PL loses to XLinear
#
# Priority order:
#   1. Traffic H=192/336 (gap +0.006-0.011, achievable)
#   2. ETTm1/ETTm2 (gap +0.001-0.003, rounding — different lr/patch might flip)
#   3. Electricity (gap +0.106-0.184, structural but worth confirming)
#
# All: features=S, enc_in=1, d_model=64, single seed s2021 for speed
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/univariate_tuning"
mkdir -p "$LOG_DIR"

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1
  grep "mse:" "$LOG" | tail -1 || echo "  [FAILED]"
}

BASE="--is_training 1 --root_path ./dataset/ --features S
      --seq_len 96 --label_len 48 --enc_in 1
      --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3
      --dw_kernel 7 --lradj sigmoid
      --train_epochs 100 --patience 10
      --model PatchLinear --seed 2021 --des Exp"

# =============================================================================
# TRAFFIC — test lr and patch variants (features=S, batch=256)
# Current: lr=0.005, p16s8  XLinear: H96=0.114, H192=0.128, H336=0.143
# =============================================================================
echo "=== TRAFFIC ==="
for PL in 96 192 336 720; do
  for LR in 0.001 0.003 0.01; do
    run_one "Traffic_pl${PL}_lr${LR}" $BASE \
      --data_path traffic.csv --data custom \
      --batch_size 256 --learning_rate ${LR} \
      --patch_len 16 --stride 8 --pred_len ${PL} \
      --model_id "TrafficS_pl${PL}_lr${LR}"
  done
  # patch variants
  for PS in "8 4" "32 16"; do
    P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
    run_one "Traffic_pl${PL}_p${P}s${S}" $BASE \
      --data_path traffic.csv --data custom \
      --batch_size 256 --learning_rate 0.005 \
      --patch_len ${P} --stride ${S} --pred_len ${PL} \
      --model_id "TrafficS_pl${PL}_p${P}s${S}"
  done
done

# =============================================================================
# ETTm1 — tiny gaps (+0.001), just try different lr
# Current: lr=0.0005, p16s8
# =============================================================================
echo "=== ETTm1 ==="
for PL in 96 192 336 720; do
  for LR in 0.0001 0.001 0.005; do
    run_one "ETTm1_pl${PL}_lr${LR}" $BASE \
      --data_path ETTm1.csv --data ETTm1 \
      --batch_size 2048 --learning_rate ${LR} \
      --patch_len 16 --stride 8 --pred_len ${PL} \
      --model_id "ETTm1S_pl${PL}_lr${LR}"
  done
done

# =============================================================================
# ETTm2 — gaps +0.001-0.003
# Current: lr=0.0001, p16s8
# =============================================================================
echo "=== ETTm2 ==="
for PL in 96 192 336 720; do
  for LR in 0.0005 0.001 0.005; do
    run_one "ETTm2_pl${PL}_lr${LR}" $BASE \
      --data_path ETTm2.csv --data ETTm2 \
      --batch_size 2048 --learning_rate ${LR} \
      --patch_len 16 --stride 8 --pred_len ${PL} \
      --model_id "ETTm2S_pl${PL}_lr${LR}"
  done
done

# =============================================================================
# ELECTRICITY — confirm structural gap, try different configs
# Current: lr=0.005  XLinear: H96=0.203, we get 0.309
# =============================================================================
echo "=== ELECTRICITY ==="
for PL in 96 720; do  # just H=96 and H=720 to confirm
  for LR in 0.001 0.01; do
    run_one "Elec_pl${PL}_lr${LR}" $BASE \
      --data_path electricity.csv --data custom \
      --batch_size 256 --learning_rate ${LR} \
      --patch_len 16 --stride 8 --pred_len ${PL} \
      --model_id "ElecS_pl${PL}_lr${LR}"
  done
  run_one "Elec_pl${PL}_p8s4" $BASE \
    --data_path electricity.csv --data custom \
    --batch_size 256 --learning_rate 0.005 \
    --patch_len 8 --stride 4 --pred_len ${PL} \
    --model_id "ElecS_pl${PL}_p8s4"
done

echo ""
echo "=== SUMMARY ==="
echo "--- Traffic (XLinear targets: H96=0.114 H192=0.128 H336=0.143 H720=0.186) ---"
for PL in 96 192 336 720; do
  echo "  H=${PL}:"
  for f in "$LOG_DIR"/Traffic_pl${PL}_*.log; do
    MSE=$(grep "mse:" "$f"|tail -1|grep -oP 'mse:\K[\d.]+' || echo "?")
    printf "    %-40s %s\n" "$(basename $f .log)" "$MSE"
  done | sort -k2 -n
done

echo "--- ETTm1 (XL: H96=0.028 H192=0.044 H336=0.059 H720=0.083) ---"
for PL in 96 192 336 720; do
  BEST=$(for f in "$LOG_DIR"/ETTm1_pl${PL}_*.log; do
    MSE=$(grep "mse:" "$f"|tail -1|grep -oP 'mse:\K[\d.]+' || echo "9")
    echo "$MSE $(basename $f .log)"
  done | sort -n | head -1)
  echo "  H=${PL}: $BEST"
done

echo "--- ETTm2 (XL: H96=0.060 H192=0.093 H336=0.126 H720=0.178) ---"
for PL in 96 192 336 720; do
  BEST=$(for f in "$LOG_DIR"/ETTm2_pl${PL}_*.log; do
    MSE=$(grep "mse:" "$f"|tail -1|grep -oP 'mse:\K[\d.]+' || echo "9")
    echo "$MSE $(basename $f .log)"
  done | sort -n | head -1)
  echo "  H=${PL}: $BEST"
done

echo "--- Electricity (XL: H96=0.203, we get 0.309 — structural?) ---"
for f in "$LOG_DIR"/Elec_*.log; do
  MSE=$(grep "mse:" "$f"|tail -1|grep -oP 'mse:\K[\d.]+' || echo "?")
  echo "  $(basename $f .log): $MSE"
done | sort -t: -k2 -n
