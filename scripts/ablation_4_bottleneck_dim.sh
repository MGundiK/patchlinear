#!/bin/bash

# ============================================================
# ABLATION 4: FUSION GATE BOTTLENECK DIMENSION
# ============================================================
# How many hidden dims for the fusion bottleneck?
#   dim=8     → very compressed
#   dim=16    → compressed
#   dim=32    → GLPatch v8 default
#   dim=64    → wider
#   dim=-1    → full-rank (no bottleneck, O(H²) params)
#
# This ablation justifies the bottleneck design.
# At pred_len=720: dim=32 has ~46K params, full-rank has ~1M.
#
# Datasets: ETTh2, Weather, Electricity (include one large dataset)
# Gating ENABLED at pre_pointwise (default)
# Estimated: ~3 hours (5 dims × 3 datasets × 4 horizons = 60 runs)

ma_type=ema; alpha=0.3; beta=0.3
model=GLPatch_ablation
seq_len=96
LOGDIR="./logs/ablation/gate_dim"

declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh2]=0.0007 [Weather]=0.0005 [Electricity]=0.005 )
DS_PATH=( [ETTh2]=ETTh2.csv [Weather]=weather.csv [Electricity]=electricity.csv )
DS_FLAG=( [ETTh2]=ETTh2 [Weather]=custom [Electricity]=custom )
DS_ENC=( [ETTh2]=7 [Weather]=21 [Electricity]=321 )
DS_BATCH=( [ETTh2]=2048 [Weather]=2048 [Electricity]=256 )

DIMS=("8" "16" "32" "64" "-1")
DIM_NAMES=("dim8" "dim16" "dim32" "dim64" "full_rank")

echo ""
echo "========== [$(date '+%H:%M:%S')] ABLATION 4: BOTTLENECK DIMENSION =========="
echo ""

for ds in ETTh2 Weather Electricity; do
  lr=${DS_LR[$ds]}; data_path=${DS_PATH[$ds]}; data_flag=${DS_FLAG[$ds]}
  enc_in=${DS_ENC[$ds]}; batch=${DS_BATCH[$ds]}

  for idx in "${!DIMS[@]}"; do
    dim=${DIMS[$idx]}
    dname=${DIM_NAMES[$idx]}
    vdir="${LOGDIR}/${ds}/${dname}"
    mkdir -p ${vdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} gate_dim=${dim} (${dname})"

    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} ${dname} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id abl_dim_${ds}_${dname}_${pred_len} --model $model --data ${data_flag} \
        --features M --seq_len $seq_len --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_gating 1 --use_fusion 1 --gate_hidden_dim $dim \
        --use_amp --num_workers 2 \
        2>&1 | tee ${vdir}/${pred_len}.log
    done

    echo "  === ${ds} ${dname} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${vdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

echo ""
echo "========== BOTTLENECK DIMENSION SUMMARY =========="
for ds in ETTh2 Weather Electricity; do
  echo ""
  echo "${ds}:"
  echo "  Dimension    |   96 MSE    192 MSE    336 MSE    720 MSE"
  echo "  -------------+--------------------------------------------"
  for idx in "${!DIMS[@]}"; do
    dname=${DIM_NAMES[$idx]}
    vdir="${LOGDIR}/${ds}/${dname}"
    printf "  %-13s|" "${dname}"
    for pred_len in 96 192 336 720; do
      mse=$(grep "mse:" ${vdir}/${pred_len}.log 2>/dev/null | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      if [ -n "$mse" ]; then printf "  %8s" $(printf "%.4f" $mse); else printf "     N/A"; fi
    done
    echo ""
  done
done
