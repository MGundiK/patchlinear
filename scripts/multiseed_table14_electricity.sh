#!/bin/bash

# ============================================================
# MULTI-SEED TABLE 14 — ELECTRICITY
# ============================================================
# Config: sl=512, lr=0.001, batch=192
# Seeds: 2022, 2023, 2024 (seed 2021 already done)
# Estimated: ~2 hrs per seed × 3 = ~6 hrs

alpha=0.3; beta=0.3; model_name=GLPatch
LOGDIR="./logs/multiseed_t14/Electricity"
mkdir -p ${LOGDIR}

for seed in 2022 2023 2024; do
  echo ""
  echo ">>> [$(date '+%H:%M:%S')] Electricity seed=${seed} sl=512 lr=0.001"

  sed -i "s/fix_seed = [0-9]*/fix_seed = ${seed}/" run.py

  sdir="${LOGDIR}/seed${seed}"; mkdir -p ${sdir}
  for pl in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Electricity seed=${seed} pl=${pl}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path electricity.csv \
      --model_id ms_Elec_s${seed}_${pl} --model $model_name --data custom \
      --features M --seq_len 512 --pred_len $pl --enc_in 321 \
      --des 'Exp' --itr 1 --batch_size 192 --learning_rate 0.001 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done

  echo "  === Electricity seed=${seed} complete ==="
  for pl in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pl}.log | tail -1)
    echo "    pl=${pl}: ${result}"
  done
  echo ""
done

sed -i "s/fix_seed = [0-9]*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021"
echo ""
echo "========== ELECTRICITY MULTI-SEED COMPLETE [$(date '+%H:%M:%S')] =========="
