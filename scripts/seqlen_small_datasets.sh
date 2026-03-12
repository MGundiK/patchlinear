#!/bin/bash

# ============================================================
# INCREASED SEQ_LEN EXPERIMENTS (Small/Fast Datasets)
# ============================================================
# Tests whether GLPatch benefits more from longer lookback windows.
# At seq_len=96 there are ~7 patches — gating has little to differentiate.
# At seq_len=336 there are ~21 patches — gating should matter more.
#
# seq_lens: 192, 336, 512
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, ILI
# pred_lens: 96, 192, 336, 720 (24,36,48,60 for ILI)
# Single seed (2021), uses GLPatch model (not ablation)
#
# NOTE: patch_len=16, stride=8 unchanged — more patches come from longer input
#   seq_len=96  → 7 patches (baseline, already run)
#   seq_len=192 → 13 patches
#   seq_len=336 → 21 patches
#   seq_len=512 → 32 patches
#
# Estimated runtime:
#   ETTh1/h2: ~5 min each per seq_len
#   ETTm1/m2: ~15 min each per seq_len
#   Weather:  ~20 min per seq_len
#   Exchange: ~5 min per seq_len
#   ILI:      ~2 min per seq_len
#   Total per seq_len: ~70 min
#   Total: ~3.5 hours for 3 seq_lens
#
# IMPORTANT: ILI uses seq_len=36 as default (very short series).
#   We test seq_len=36 (baseline), 48, 60, 72 for ILI instead.

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
LOGDIR="./logs/seqlen"

# Dataset configs: name, lr, data_path, data_flag, enc_in, batch_size
declare -A DS_LR DS_PATH DS_FLAG DS_ENC DS_BATCH
DS_LR=( [ETTh1]=0.0005 [ETTh2]=0.0007 [ETTm1]=0.0007 [ETTm2]=0.00005 [Weather]=0.0005 [Exchange]=0.000005 )
DS_PATH=( [ETTh1]=ETTh1.csv [ETTh2]=ETTh2.csv [ETTm1]=ETTm1.csv [ETTm2]=ETTm2.csv [Weather]=weather.csv [Exchange]=exchange_rate.csv )
DS_FLAG=( [ETTh1]=ETTh1 [ETTh2]=ETTh2 [ETTm1]=ETTm1 [ETTm2]=ETTm2 [Weather]=custom [Exchange]=custom )
DS_ENC=( [ETTh1]=7 [ETTh2]=7 [ETTm1]=7 [ETTm2]=7 [Weather]=21 [Exchange]=8 )
DS_BATCH=( [ETTh1]=2048 [ETTh2]=2048 [ETTm1]=2048 [ETTm2]=2048 [Weather]=2048 [Exchange]=32 )

echo ""
echo "========== [$(date '+%H:%M:%S')] INCREASED SEQ_LEN EXPERIMENTS =========="
echo "seq_lens: 192, 336, 512 (baseline seq_len=96 already run)"
echo ""

# ============================================================
# Standard datasets: seq_len = 192, 336, 512
# ============================================================
for sl in 192 336 512; do
  echo ""
  echo "################################################################"
  echo "  seq_len = ${sl} — [$(date '+%H:%M:%S')]"
  echo "################################################################"
  echo ""

  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
    lr=${DS_LR[$ds]}; data_path=${DS_PATH[$ds]}; data_flag=${DS_FLAG[$ds]}
    enc_in=${DS_ENC[$ds]}; batch=${DS_BATCH[$ds]}

    sdir="${LOGDIR}/sl${sl}/${ds}"
    mkdir -p ${sdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} seq_len=${sl} lr=${lr}"

    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} sl=${sl} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id sl${sl}_${ds}_${pred_len}_${ma_type} --model $model_name --data ${data_flag} \
        --features M --seq_len $sl --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size $batch --learning_rate $lr \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_amp --num_workers 2 \
        2>&1 | tee ${sdir}/${pred_len}.log
    done

    echo "  === ${ds} sl=${sl} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

# ============================================================
# ILI: special case — shorter series, different pred_lens
# Baseline is seq_len=36. Test 48, 60, 72.
# ============================================================
echo ""
echo "################################################################"
echo "  ILI — seq_len = 48, 60, 72"
echo "################################################################"
echo ""

for sl in 48 60 72; do
  sdir="${LOGDIR}/sl${sl}/ILI"
  mkdir -p ${sdir}

  # label_len = sl // 2
  ll=$((sl / 2))

  echo ">>> [$(date '+%H:%M:%S')] ILI seq_len=${sl} label_len=${ll}"

  for pred_len in 24 36 48 60; do
    echo "  [$(date '+%H:%M:%S')] ILI sl=${sl} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path national_illness.csv \
      --model_id sl${sl}_ILI_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $sl --label_len $ll --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.02 \
      --lradj 'type3' --patch_len 6 --stride 3 \
      --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ILI sl=${sl} complete ==="
  for pred_len in 24 36 48 60; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] SEQ_LEN EXPERIMENTS COMPLETE =========="
echo ""
echo "RESULTS BY SEQ_LEN (MSE extracted from logs):"
echo "=============================================="

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  echo ""
  echo "${ds}:"
  echo "  seq_len |     96      192      336      720"
  echo "  --------+------------------------------------"

  # Baseline (seq_len=96) from final_r2 logs
  printf "       96 |"
  for pred_len in 96 192 336 720; do
    logfile="./logs/final_r2/${ds}/${pred_len}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""

  for sl in 192 336 512; do
    printf "      %3d |" $sl
    for pred_len in 96 192 336 720; do
      logfile="${LOGDIR}/sl${sl}/${ds}/${pred_len}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
done

echo ""
echo "ILI:"
echo "  seq_len |     24       36       48       60"
echo "  --------+------------------------------------"

printf "       36 |"
for pred_len in 24 36 48 60; do
  logfile="./logs/final_r2/ILI/${pred_len}.log"
  if [ -f "$logfile" ]; then
    mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
    [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
  else
    printf "     N/A"
  fi
done
echo ""

for sl in 48 60 72; do
  printf "      %3d |" $sl
  for pred_len in 24 36 48 60; do
    logfile="${LOGDIR}/sl${sl}/ILI/${pred_len}.log"
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
echo "Log files: ${LOGDIR}/sl<SEQ_LEN>/<dataset>/<pred_len>.log"
echo ""
echo "Total runs: 6 datasets × 3 seq_lens × 4 horizons + ILI × 3 × 4 = 84 runs"
