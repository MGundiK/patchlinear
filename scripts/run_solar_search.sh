#!/usr/bin/env bash
# =============================================================================
# Solar hyperparameter search — single seed (2021), H=96 and H=720
#
# Current: MSE 3ul+0bold  MAE 2bold+2ul
# Target:  push H=96/192 MSE toward TimeMixer (0.189/0.222)
#          push H=720 MSE from 0.257 into underline range (0.249=xPatch)
#
# Key lesson from Electricity: d_model scaling is the biggest lever
# Solar: enc_in=137, batch=512, lr=0.0008
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/solar_search"
mkdir -p "$LOG_DIR"

SEED=2021

run2() {
  local TAG=$1; shift
  for PL in 96 720; do
    local LOG="${LOG_DIR}/${TAG}_pl${PL}.log"
    echo ">>> Solar pl${PL} ${TAG}"
    python -u run.py "$@" --pred_len ${PL} \
      --model_id "Solar_${TAG}_pl${PL}" > "$LOG" 2>&1
    grep "mse:" "$LOG" | tail -1 || echo "  [FAILED]"
  done
}

BASE="--is_training 1 --root_path ./dataset/
  --data_path solar.txt --data Solar
  --features M --seq_len 96 --label_len 48 --enc_in 137
  --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1
  --use_cross_channel 1 --use_alpha_gate 1
  --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3
  --batch_size 512 --learning_rate 0.0008
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --seed ${SEED} --des Exp"

# Default (c_ff=34 since max(16,min(137//4,128))=34)
DFLT="--d_model 64 --t_ff 128 --c_ff 34"

echo "=== SECTION 1: d_model sweep (t_ff=2d, c_ff=34) ==="
for D in 32 48 96 128 160 192 256; do
  T=$((D*2))
  run2 "d${D}_t${T}_c34" $BASE --d_model ${D} --t_ff ${T} --c_ff 34
done

echo "=== SECTION 2: c_ff sweep (d=64) ==="
for C in 16 24 48 64 96 128; do
  run2 "d64_c${C}" $BASE --d_model 64 --t_ff 128 --c_ff ${C}
done

echo "=== SECTION 3: patch/stride ==="
for PS in "8 4" "24 12" "32 16"; do
  P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
  run2 "p${P}s${S}" $BASE $DFLT --patch_len ${P} --stride ${S}
done

echo "=== SECTION 4: dw_kernel ==="
for DK in 3 13; do
  run2 "dk${DK}" $BASE $DFLT --dw_kernel ${DK}
done

echo "=== SECTION 5: Architecture flag combos ==="
for DEC in 0 1; do for SEAS in 0 1; do for FUS in 0 1; do
  [[ $DEC -eq 1 && $SEAS -eq 1 && $FUS -eq 1 ]] && continue
  run2 "dec${DEC}_s${SEAS}_f${FUS}" \
    $BASE $DFLT \
    --use_decomp ${DEC} --use_seas_stream ${SEAS} --use_fusion_gate ${FUS}
done; done; done

echo "=== SECTION 6: Promising d_model × patch combos ==="
for CFG in "128 256 34 8 4 7" "256 512 34 16 8 7" "256 512 34 8 4 7" "128 256 34 16 8 13"; do
  D=$(echo $CFG|cut -d' ' -f1); T=$(echo $CFG|cut -d' ' -f2)
  C=$(echo $CFG|cut -d' ' -f3); P=$(echo $CFG|cut -d' ' -f4)
  S=$(echo $CFG|cut -d' ' -f5); DK=$(echo $CFG|cut -d' ' -f6)
  run2 "d${D}_c${C}_p${P}s${S}_dk${DK}" \
    $BASE --d_model ${D} --t_ff ${T} --c_ff ${C} \
    --patch_len ${P} --stride ${S} --dw_kernel ${DK}
done

echo ""
echo "=== TOP 10 by H=96 MSE ==="
for f in "$LOG_DIR"/*_pl96.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "9.999")
  echo "$MSE $(basename $f _pl96.log)"
done | sort -n | head -10

echo ""
echo "=== TOP 10 by H=720 MSE ==="
for f in "$LOG_DIR"/*_pl720.log; do
  MSE=$(grep "mse:" "$f" | tail -1 | grep -oP 'mse:\K[\d.]+' || echo "9.999")
  echo "$MSE $(basename $f _pl720.log)"
done | sort -n | head -10

echo ""
echo "Targets: H=96 MSE<0.189 (TimeMixer), H=720 MSE<0.249 (xPatch ul threshold)"
