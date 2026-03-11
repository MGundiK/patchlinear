#!/bin/bash

# ============================================================
# ABLATION 1: COMPONENT ABLATION
# ============================================================
# Tests each component independently + combined.
# This is the MOST IMPORTANT ablation for reviewers.
#
# Variants:
#   A. xPatch baseline:  no gating, no fusion (static s+t)
#   B. + Gating only:    gating enabled, static fusion
#   C. + Fusion only:    no gating, adaptive fusion enabled
#   D. GLPatch (full):   both gating + fusion
#
# Datasets: ETTh2 (lr=0.0007), Weather (lr=0.0005),
#           Electricity (lr=0.005), Solar (lr=0.01)
#
# Single seed (2021), all 4 horizons per dataset
# Estimated: ~3 hours (4 variants × 4 datasets × 4 horizons = 64 runs)

# PREREQUISITE: Run setup_ablation.sh first!

ma_type=ema; alpha=0.3; beta=0.3
model=GLPatch_ablation
seq_len=96
LOGDIR="./logs/ablation/component"

declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh2]=0.0007 [Weather]=0.0005 [Electricity]=0.005 [Solar]=0.01 )
DS_PATH=( [ETTh2]=ETTh2.csv [Weather]=weather.csv [Electricity]=electricity.csv [Solar]=solar.txt )
DS_FLAG=( [ETTh2]=ETTh2 [Weather]=custom [Electricity]=custom [Solar]=Solar )
DS_ENC=( [ETTh2]=7 [Weather]=21 [Electricity]=321 [Solar]=137 )
DS_BATCH=( [ETTh2]=2048 [Weather]=2048 [Electricity]=256 [Solar]=512 )

# Variant configs: name, use_gating, use_fusion
VARIANTS=(
  "A_baseline:0:0"
  "B_gating_only:1:0"
  "C_fusion_only:0:1"
  "D_full_glpatch:1:1"
)

echo ""
echo "========== [$(date '+%H:%M:%S')] ABLATION 1: COMPONENT ABLATION =========="
echo ""

for ds in ETTh2 Weather Electricity Solar; do
  lr=${DS_LR[$ds]}
  data_path=${DS_PATH[$ds]}
  data_flag=${DS_FLAG[$ds]}
  enc_in=${DS_ENC[$ds]}
  batch=${DS_BATCH[$ds]}

  for variant_str in "${VARIANTS[@]}"; do
    IFS=':' read -r vname use_g use_f <<< "$variant_str"
    vdir="${LOGDIR}/${ds}/${vname}"
    mkdir -p ${vdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} ${vname} (gating=${use_g}, fusion=${use_f})"

    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} ${vname} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id abl_comp_${ds}_${vname}_${pred_len} --model $model --data ${data_flag} \
        --features M --seq_len $seq_len --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_gating $use_g --use_fusion $use_f \
        --use_amp --num_workers 2 \
        2>&1 | tee ${vdir}/${pred_len}.log
    done

    echo "  === ${ds} ${vname} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${vdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== COMPONENT ABLATION SUMMARY =========="
for ds in ETTh2 Weather Electricity Solar; do
  echo ""
  echo "${ds}:"
  echo "  Variant              |   96 MSE    192 MSE    336 MSE    720 MSE  |  Avg MSE"
  echo "  ---------------------+--------------------------------------------+---------"
  for variant_str in "${VARIANTS[@]}"; do
    IFS=':' read -r vname use_g use_f <<< "$variant_str"
    vdir="${LOGDIR}/${ds}/${vname}"
    total=0; count=0
    printf "  %-21s |" "${vname}"
    for pred_len in 96 192 336 720; do
      mse=$(grep "mse:" ${vdir}/${pred_len}.log 2>/dev/null | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      if [ -n "$mse" ]; then
        printf "  %8s" $(printf "%.4f" $mse)
        total=$(echo "$total + $mse" | bc -l)
        count=$((count + 1))
      else
        printf "     N/A"
      fi
    done
    if [ $count -gt 0 ]; then
      avg=$(echo "$total / $count" | bc -l)
      printf "  | %7s\n" $(printf "%.4f" $avg)
    else
      printf "  |     N/A\n"
    fi
  done
done
echo ""
echo "Log files: ${LOGDIR}/<dataset>/<variant>/<pred_len>.log"
