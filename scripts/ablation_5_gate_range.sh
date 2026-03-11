#!/bin/bash

# ============================================================
# ABLATION 5: FUSION GATE CONSTRAINT RANGE
# ============================================================
# How tightly should the fusion gate be constrained?
#   [0.0, 1.0] → unconstrained (can fully suppress a stream)
#   [0.1, 0.9] → GLPatch v8 default (at least 10% from each stream)
#   [0.2, 0.8] → tighter (at least 20% from each)
#   [0.3, 0.7] → very tight (at least 30% from each)
#   [0.5, 0.5] → fixed equal (equivalent to static s+t, fusion disabled)
#
# This ablation validates the constraint design.
# Unconstrained gates caused collapse in early experiments (v5).
#
# Datasets: ETTh2, Weather
# Gating ENABLED at pre_pointwise, bottleneck=32 (default)
# Estimated: ~1.5 hours (5 ranges × 2 datasets × 4 horizons = 40 runs)

ma_type=ema; alpha=0.3; beta=0.3
model=GLPatch_ablation
seq_len=96
LOGDIR="./logs/ablation/gate_range"

declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh2]=0.0007 [Weather]=0.0005 )
DS_PATH=( [ETTh2]=ETTh2.csv [Weather]=weather.csv )
DS_FLAG=( [ETTh2]=ETTh2 [Weather]=custom )
DS_ENC=( [ETTh2]=7 [Weather]=21 )
DS_BATCH=( [ETTh2]=2048 [Weather]=2048 )

# Format: "name:min:max"
RANGES=(
  "unconstrained:0.0:1.0"
  "range_01_09:0.1:0.9"
  "range_02_08:0.2:0.8"
  "range_03_07:0.3:0.7"
  "fixed_equal:0.5:0.5"
)

echo ""
echo "========== [$(date '+%H:%M:%S')] ABLATION 5: GATE CONSTRAINT RANGE =========="
echo ""

for ds in ETTh2 Weather; do
  lr=${DS_LR[$ds]}; data_path=${DS_PATH[$ds]}; data_flag=${DS_FLAG[$ds]}
  enc_in=${DS_ENC[$ds]}; batch=${DS_BATCH[$ds]}

  for range_str in "${RANGES[@]}"; do
    IFS=':' read -r rname gmin gmax <<< "$range_str"
    vdir="${LOGDIR}/${ds}/${rname}"
    mkdir -p ${vdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} gate=[${gmin}, ${gmax}] (${rname})"

    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} ${rname} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id abl_range_${ds}_${rname}_${pred_len} --model $model --data ${data_flag} \
        --features M --seq_len $seq_len --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_gating 1 --use_fusion 1 \
        --gate_min $gmin --gate_max $gmax \
        --use_amp --num_workers 2 \
        2>&1 | tee ${vdir}/${pred_len}.log
    done

    echo "  === ${ds} ${rname} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${vdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

echo ""
echo "========== GATE CONSTRAINT SUMMARY =========="
for ds in ETTh2 Weather; do
  echo ""
  echo "${ds}:"
  echo "  Range           |   96 MSE    192 MSE    336 MSE    720 MSE"
  echo "  ----------------+--------------------------------------------"
  for range_str in "${RANGES[@]}"; do
    IFS=':' read -r rname gmin gmax <<< "$range_str"
    vdir="${LOGDIR}/${ds}/${rname}"
    printf "  %-16s|" "[${gmin},${gmax}]"
    for pred_len in 96 192 336 720; do
      mse=$(grep "mse:" ${vdir}/${pred_len}.log 2>/dev/null | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      if [ -n "$mse" ]; then printf "  %8s" $(printf "%.4f" $mse); else printf "     N/A"; fi
    done
    echo ""
  done
done
