#!/bin/bash

# ============================================================
# ABLATION 2: GATING POSITION
# ============================================================
# Where to place inter-patch gating in the non-linear stream:
#   - pre_depthwise:  before depthwise conv (on raw patch embeddings)
#   - pre_pointwise:  after depthwise+residual, before pointwise (GLPatch v8 default)
#   - post_pointwise: after pointwise conv (on mixed patch features)
#
# Datasets: ETTh2, Weather (2 datasets sufficient for position ablation)
# Single seed, all 4 horizons
# Fusion is ENABLED for all variants (isolates gating position effect)
# Estimated: ~1 hour (3 positions × 2 datasets × 4 horizons = 24 runs)

ma_type=ema; alpha=0.3; beta=0.3
model=GLPatch_ablation
seq_len=96
LOGDIR="./logs/ablation/gate_position"

declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh2]=0.0007 [Weather]=0.0005 )
DS_PATH=( [ETTh2]=ETTh2.csv [Weather]=weather.csv )
DS_FLAG=( [ETTh2]=ETTh2 [Weather]=custom )
DS_ENC=( [ETTh2]=7 [Weather]=21 )
DS_BATCH=( [ETTh2]=2048 [Weather]=2048 )

POSITIONS=("pre_depthwise" "pre_pointwise" "post_pointwise")

echo ""
echo "========== [$(date '+%H:%M:%S')] ABLATION 2: GATING POSITION =========="
echo ""

for ds in ETTh2 Weather; do
  lr=${DS_LR[$ds]}; data_path=${DS_PATH[$ds]}; data_flag=${DS_FLAG[$ds]}
  enc_in=${DS_ENC[$ds]}; batch=${DS_BATCH[$ds]}

  for pos in "${POSITIONS[@]}"; do
    vdir="${LOGDIR}/${ds}/${pos}"
    mkdir -p ${vdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} position=${pos}"

    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} ${pos} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id abl_pos_${ds}_${pos}_${pred_len} --model $model --data ${data_flag} \
        --features M --seq_len $seq_len --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_gating 1 --use_fusion 1 --gate_position $pos \
        --use_amp --num_workers 2 \
        2>&1 | tee ${vdir}/${pred_len}.log
    done

    echo "  === ${ds} ${pos} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${vdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

echo ""
echo "========== GATING POSITION SUMMARY =========="
for ds in ETTh2 Weather; do
  echo ""
  echo "${ds}:"
  echo "  Position          |   96 MSE    192 MSE    336 MSE    720 MSE"
  echo "  ------------------+--------------------------------------------"
  for pos in "${POSITIONS[@]}"; do
    vdir="${LOGDIR}/${ds}/${pos}"
    printf "  %-18s |" "${pos}"
    for pred_len in 96 192 336 720; do
      mse=$(grep "mse:" ${vdir}/${pred_len}.log 2>/dev/null | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      if [ -n "$mse" ]; then
        printf "  %8s" $(printf "%.4f" $mse)
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
done
