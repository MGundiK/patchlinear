#!/bin/bash

# ============================================================
# ABLATION 3: RESIDUAL BLEND α_res INITIALIZATION
# ============================================================
# How aggressively should gating be integrated?
#   α_res=0.01  → very conservative (1% gating at init)
#   α_res=0.05  → conservative (5%, GLPatch v8 default)
#   α_res=0.1   → moderate
#   α_res=0.5   → aggressive (equal blend at init)
#   α_res=1.0   → full replacement (gating replaces original)
#
# Datasets: ETTh2, Weather
# Fusion ENABLED, gating at pre_pointwise (default)
# Estimated: ~1.5 hours (5 values × 2 datasets × 4 horizons = 40 runs)

ma_type=ema; alpha=0.3; beta=0.3
model=GLPatch_ablation
seq_len=96
LOGDIR="./logs/ablation/res_alpha"

declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh2]=0.0007 [Weather]=0.0005 )
DS_PATH=( [ETTh2]=ETTh2.csv [Weather]=weather.csv )
DS_FLAG=( [ETTh2]=ETTh2 [Weather]=custom )
DS_ENC=( [ETTh2]=7 [Weather]=21 )
DS_BATCH=( [ETTh2]=2048 [Weather]=2048 )

ALPHAS=("0.01" "0.05" "0.1" "0.5" "1.0")

echo ""
echo "========== [$(date '+%H:%M:%S')] ABLATION 3: α_res INITIALIZATION =========="
echo ""

for ds in ETTh2 Weather; do
  lr=${DS_LR[$ds]}; data_path=${DS_PATH[$ds]}; data_flag=${DS_FLAG[$ds]}
  enc_in=${DS_ENC[$ds]}; batch=${DS_BATCH[$ds]}

  for ra in "${ALPHAS[@]}"; do
    vdir="${LOGDIR}/${ds}/alpha_${ra}"
    mkdir -p ${vdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} α_res=${ra}"

    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} α=${ra} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id abl_alpha_${ds}_${ra}_${pred_len} --model $model --data ${data_flag} \
        --features M --seq_len $seq_len --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_gating 1 --use_fusion 1 --res_alpha_init $ra \
        --use_amp --num_workers 2 \
        2>&1 | tee ${vdir}/${pred_len}.log
    done

    echo "  === ${ds} α_res=${ra} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${vdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

echo ""
echo "========== α_res ABLATION SUMMARY =========="
for ds in ETTh2 Weather; do
  echo ""
  echo "${ds}:"
  echo "  α_res   |   96 MSE    192 MSE    336 MSE    720 MSE"
  echo "  --------+--------------------------------------------"
  for ra in "${ALPHAS[@]}"; do
    vdir="${LOGDIR}/${ds}/alpha_${ra}"
    printf "  %-8s|" "${ra}"
    for pred_len in 96 192 336 720; do
      mse=$(grep "mse:" ${vdir}/${pred_len}.log 2>/dev/null | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      if [ -n "$mse" ]; then printf "  %8s" $(printf "%.4f" $mse); else printf "     N/A"; fi
    done
    echo ""
  done
done
