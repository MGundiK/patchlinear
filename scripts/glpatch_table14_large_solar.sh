#!/bin/bash

# ============================================================
# GLPatch TABLE 14 — LARGE DATASETS
# ============================================================
# Matches xPatch Table 14 configs:
#   Traffic:     sl=720, lr=0.005, batch=46
#   Electricity: sl=720, lr=0.001, batch=128
#   Solar:       sl=720, lr=0.005, batch=256
#
# Also searches sl={336,512,720} to find GLPatch's best.
# Estimated: ~15-20 hours total

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
LOGDIR="./logs/table14"

echo ""
echo "========== [$(date '+%H:%M:%S')] GLPatch TABLE 14 — LARGE DATASETS =========="
echo ""


# ============================================================
# Solar: xPatch T14 uses sl=720, lr=0.005, batch=256
# ============================================================
for sl in 336 512 720; do
  sdir="${LOGDIR}/sl${sl}/Solar"
  mkdir -p ${sdir}

  if [ $sl -le 336 ]; then batch=512;
  elif [ $sl -le 512 ]; then batch=384;
  else batch=256; fi

  echo ">>> [$(date '+%H:%M:%S')] Solar sl=${sl} lr=0.005 batch=${batch}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Solar sl=${sl} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path solar.txt \
      --model_id t14_Solar_sl${sl}_${pred_len}_${ma_type} --model $model_name --data Solar \
      --features M --seq_len $sl --pred_len $pred_len --enc_in 137 \
      --des 'Exp' --itr 1 --batch_size $batch --learning_rate 0.005 \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === Solar sl=${sl} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done




# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] TABLE 14 LARGE DATASETS COMPLETE =========="
echo ""

for ds in  Solar ; do
  echo "${ds}:"
  echo "  seq_len |     96      192      336      720"
  echo "  --------+------------------------------------"
  for sl in 336 512 720; do
    dir="${LOGDIR}/sl${sl}/${ds}"
    printf "  %6d  |" $sl
    for pred_len in 96 192 336 720; do
      logfile="${dir}/${pred_len}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
  echo ""
done
