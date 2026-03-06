#!/bin/bash

# GLPatch v8 — FINAL BENCHMARK (R2 updates)
# Only re-runs datasets where the best single LR changed.
#
# Changes from R1 final:
#   ETTh2:       0.0005 → 0.0007
#   ETTm1:       0.0005 → 0.0007
#   ETTm2:       0.0001 → 0.00005  (marginal, optional)
#   Traffic:     0.005  → 0.002
#   Solar:       0.005  → 0.01
#
# Unchanged (use existing R1/v8 results):
#   ETTh1:       0.0005
#   Weather:     0.0005
#   Exchange:    0.000005
#   ILI:         0.02
#   Electricity: 0.005
#
# Estimated runtime: ~4-5 hours on Colab A100 with AMP
#   ETTh2:  ~5 min
#   ETTm1:  ~10 min
#   ETTm2:  ~15 min
#   Traffic: ~5 hours (the bottleneck)
#   Solar:   ~45 min

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

mkdir -p ./logs/final_r2

# ================================================================
# ETTh2 — lr=0.0007 (was 0.0005)
# Expected: 3/4 wins (96✅, 192~, 336❌, 720✅)
# ================================================================
echo "========== [$(date '+%H:%M')] ETTh2 lr=0.0007 =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M')] ETTh2 pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv \
    --model_id ETTh2_final_${pred_len}_${ma_type} --model $model_name --data ETTh2 \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0007 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2
done
echo "=== ETTh2 complete ==="
grep "mse:" result.txt | tail -4
echo ""

# ================================================================
# ETTm1 — lr=0.0007 (was 0.0005)
# Expected: 4/4 sweep
# ================================================================
echo "========== [$(date '+%H:%M')] ETTm1 lr=0.0007 =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M')] ETTm1 pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv \
    --model_id ETTm1_final_${pred_len}_${ma_type} --model $model_name --data ETTm1 \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0007 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2
done
echo "=== ETTm1 complete ==="
grep "mse:" result.txt | tail -4
echo ""

# ================================================================
# ETTm2 — lr=0.00005 (was 0.0001)
# Marginal improvement — can skip if time-constrained
# Expected: 4/4 sweep
# ================================================================
echo "========== [$(date '+%H:%M')] ETTm2 lr=0.00005 =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M')] ETTm2 pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv \
    --model_id ETTm2_final_${pred_len}_${ma_type} --model $model_name --data ETTm2 \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.00005 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2
done
echo "=== ETTm2 complete ==="
grep "mse:" result.txt | tail -4
echo ""

# ================================================================
# Solar — lr=0.01 (was 0.005)
# Expected: 3/4 wins (96✅, 192✅, 336✅, 720❌)
# ================================================================
echo "========== [$(date '+%H:%M')] Solar lr=0.01 =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M')] Solar pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path solar.txt \
    --model_id solar_final_${pred_len}_${ma_type} --model $model_name --data Solar \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 137 \
    --des 'Exp' --itr 1 --batch_size 512 --learning_rate 0.01 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2
done
echo "=== Solar complete ==="
grep "mse:" result.txt | tail -4
echo ""

# ================================================================
# Traffic — lr=0.002 (was 0.005)
# Expected: 3/4 wins (96❌, 192✅, 336✅, 720✅)
# NOTE: This is the slowest — run last
# ================================================================
echo "========== [$(date '+%H:%M')] Traffic lr=0.002 =========="
for pred_len in 96 192 336 720; do
  echo ">>> [$(date '+%H:%M')] Traffic pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
    --model_id traffic_final_${pred_len}_${ma_type} --model $model_name --data custom \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 \
    --des 'Exp' --itr 1 --batch_size 96 --learning_rate 0.002 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2
done
echo "=== Traffic complete ==="
grep "mse:" result.txt | tail -4
echo ""

echo ""
echo "========== [$(date '+%H:%M')] FINAL BENCHMARK R2 COMPLETE =========="
echo ""
echo "5 datasets × 4 horizons = 20 runs"
echo ""
echo "FINAL LR MAP (for paper):"
echo "  ETTh1:       0.0005   (unchanged, use v8 results)"
echo "  ETTh2:       0.0007   (re-run above)"
echo "  ETTm1:       0.0007   (re-run above)"
echo "  ETTm2:       0.00005  (re-run above)"
echo "  Weather:     0.0005   (unchanged, use v8 results)"
echo "  Exchange:    0.000005 (unchanged, use R1 final results)"
echo "  ILI:         0.02     (unchanged, use R1 final results)"
echo "  Electricity: 0.005    (unchanged, use v8 results)"
echo "  Traffic:     0.002    (re-run above)"
echo "  Solar:       0.01     (re-run above)"
