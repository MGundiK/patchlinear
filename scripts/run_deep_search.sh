#!/usr/bin/env bash
# =============================================================================
# Deep targeted search: ETTh1, ETTh2, ILI
# Sweeps d_model, label_len (ETT) and d_model, c_ff, seas_stream,
# label_len, lr (ILI) at best (L, patch, kernel) found so far.
# Single-seed 2021. Winners get 3-seed confirmation separately.
# =============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_DIR="logs/deep_search"
mkdir -p "$LOG_DIR"
SEED=2021

run() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1 || true
  grep -m1 "mse:" "$LOG" | tail -1 || echo "  [FAILED]"
}

BASE_ETT="--is_training 1 --root_path ./dataset/ --features M
  --lradj sigmoid --train_epochs 100 --patience 10
  --model PatchLinear --seed ${SEED} --des Exp"

# ═══════════════════════════════════════════════════════════════════
# ETTh1
# Best (L,p,k) per horizon: H192=L336p24k3, H336=L512p24k3, H720=L192p16k13
# ═══════════════════════════════════════════════════════════════════
echo "=== ETTh1 ==="
for D in 64 96 128; do
  TFF=$((D*2))
  for LL in 48 96 192; do
    TAG="ETTh1_pl192_L336_p24k3_d${D}_ll${LL}"
    run "$TAG" $BASE_ETT \
      --data_path ETTh1.csv --data ETTh1 --enc_in 7 \
      --seq_len 336 --label_len ${LL} --pred_len 192 \
      --patch_len 24 --stride 12 --dw_kernel 3 \
      --d_model ${D} --t_ff ${TFF} --c_ff 16 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --model_id "$TAG"
  done
done

for D in 64 96 128; do
  TFF=$((D*2))
  for LL in 48 96 192; do
    TAG="ETTh1_pl336_L512_p24k3_d${D}_ll${LL}"
    run "$TAG" $BASE_ETT \
      --data_path ETTh1.csv --data ETTh1 --enc_in 7 \
      --seq_len 512 --label_len ${LL} --pred_len 336 \
      --patch_len 24 --stride 12 --dw_kernel 3 \
      --d_model ${D} --t_ff ${TFF} --c_ff 16 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --model_id "$TAG"
  done
done

for D in 64 96 128; do
  TFF=$((D*2))
  for LL in 48 96 192 336; do
    TAG="ETTh1_pl720_L192_p16k13_d${D}_ll${LL}"
    run "$TAG" $BASE_ETT \
      --data_path ETTh1.csv --data ETTh1 --enc_in 7 \
      --seq_len 192 --label_len ${LL} --pred_len 720 \
      --patch_len 16 --stride 8 --dw_kernel 13 \
      --d_model ${D} --t_ff ${TFF} --c_ff 16 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --model_id "$TAG"
  done
done

# k=21 for H720 (extending the kernel that already helped)
for K in 21 25; do
  TAG="ETTh1_pl720_L192_p16k${K}_d64_ll48"
  run "$TAG" $BASE_ETT \
    --data_path ETTh1.csv --data ETTh1 --enc_in 7 \
    --seq_len 192 --label_len 48 --pred_len 720 \
    --patch_len 16 --stride 8 --dw_kernel ${K} \
    --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
    --batch_size 2048 --learning_rate 0.0005 \
    --model_id "$TAG"
done

# ═══════════════════════════════════════════════════════════════════
# ETTh2
# Best (L,p,k): H336=L512p32k13, H720=L336p24k3
# ═══════════════════════════════════════════════════════════════════
echo "=== ETTh2 ==="
for D in 64 96 128; do
  TFF=$((D*2))
  for LL in 48 96 192; do
    TAG="ETTh2_pl336_L512_p32k13_d${D}_ll${LL}"
    run "$TAG" $BASE_ETT \
      --data_path ETTh2.csv --data ETTh2 --enc_in 7 \
      --seq_len 512 --label_len ${LL} --pred_len 336 \
      --patch_len 32 --stride 16 --dw_kernel 13 \
      --d_model ${D} --t_ff ${TFF} --c_ff 16 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --model_id "$TAG"
  done
done

for D in 64 96 128; do
  TFF=$((D*2))
  for LL in 48 96 192; do
    TAG="ETTh2_pl720_L336_p24k3_d${D}_ll${LL}"
    run "$TAG" $BASE_ETT \
      --data_path ETTh2.csv --data ETTh2 --enc_in 7 \
      --seq_len 336 --label_len ${LL} --pred_len 720 \
      --patch_len 24 --stride 12 --dw_kernel 3 \
      --d_model ${D} --t_ff ${TFF} --c_ff 16 --alpha 0.3 \
      --batch_size 2048 --learning_rate 0.0005 \
      --model_id "$TAG"
  done
done

# ═══════════════════════════════════════════════════════════════════
# ILI — comprehensive search
# ═══════════════════════════════════════════════════════════════════
echo "=== ILI Phase A: d_model x c_ff (seas=0, L=104) ==="
for PL in 24 36 48 60; do
  for D in 16 32 48 64 96 128; do
    TFF=$((D*2))
    for C in 2 4 8 16 32; do
      TAG="ILI_pl${PL}_L104_d${D}_c${C}_seas0"
      run "$TAG" $BASE_ETT \
        --data_path national_illness.csv --data custom --enc_in 7 \
        --seq_len 104 --label_len 18 --pred_len ${PL} \
        --patch_len 6 --stride 3 --dw_kernel 3 \
        --d_model ${D} --t_ff ${TFF} --c_ff ${C} --alpha 0.3 \
        --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
        --batch_size 32 --learning_rate 0.01 \
        --lradj type3 --train_epochs 100 --patience 10 \
        --model PatchLinear --seed ${SEED} --des Exp \
        --model_id "$TAG"
    done
  done
done

echo "=== ILI Phase B: seasonal stream re-enable ==="
for PL in 24 36 48 60; do
  for L in 36 104 148; do
    for D in 32 64 96; do
      TFF=$((D*2))
      for C in 4 8 16; do
        TAG="ILI_pl${PL}_L${L}_d${D}_c${C}_seas1"
        run "$TAG" $BASE_ETT \
          --data_path national_illness.csv --data custom --enc_in 7 \
          --seq_len ${L} --label_len 18 --pred_len ${PL} \
          --patch_len 6 --stride 3 --dw_kernel 3 \
          --d_model ${D} --t_ff ${TFF} --c_ff ${C} --alpha 0.3 \
          --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
          --batch_size 32 --learning_rate 0.01 \
          --lradj type3 --train_epochs 100 --patience 10 \
          --model PatchLinear --seed ${SEED} --des Exp \
          --model_id "$TAG"
      done
    done
  done
done

echo "=== ILI Phase C: label_len x lr at best config ==="
# Will run best config from A after seeing results. Script uses L=104, best d/c from A.
# Placeholder: run with current best (d=64,c=8) across label_len and lr
for PL in 24 36 48 60; do
  for LL in 18 24 36 48; do
    for LR in 0.005 0.01 0.02; do
      LR_STR=$(echo $LR | sed 's/\./_/g')
      TAG="ILI_pl${PL}_L104_d64_c8_ll${LL}_lr${LR_STR}_seas0"
      run "$TAG" $BASE_ETT \
        --data_path national_illness.csv --data custom --enc_in 7 \
        --seq_len 104 --label_len ${LL} --pred_len ${PL} \
        --patch_len 6 --stride 3 --dw_kernel 3 \
        --d_model 64 --t_ff 128 --c_ff 8 --alpha 0.3 \
        --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
        --batch_size 32 --learning_rate ${LR} \
        --lradj type3 --train_epochs 100 --patience 10 \
        --model PatchLinear --seed ${SEED} --des Exp \
        --model_id "$TAG"
    done
  done
done

echo ""
echo "=== SUMMARY ==="
python3 - << 'PYEOF'
import re, os, numpy as np
from collections import defaultdict

log_dir = "logs/deep_search"
xpatch = {
    'ETTh1': {192:0.376, 336:0.391, 720:0.442},
    'ETTh2': {336:0.312, 720:0.384},
    'ILI':   {24:1.188,  36:1.226,  48:1.254,  60:1.455},
}
prev = {
    'ETTh1': {192:0.378, 336:0.402, 720:0.463},
    'ETTh2': {336:0.315, 720:0.390},
    'ILI':   {24:1.367,  36:1.384,  48:1.270,  60:1.447},
}

best = defaultdict(lambda: defaultdict(lambda: (float('inf'), None)))

for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.log'): continue
    path = os.path.join(log_dir, fname)
    for line in open(path):
        m = re.search(r'mse:\s*([\d.]+)', line)
        if m:
            mse = float(m.group(1))
            tag = fname[:-4]
            for ds in ['ETTh1','ETTh2','ILI']:
                if not tag.startswith(ds): continue
                for pl in ([192,336,720] if 'ETT' in ds else [24,36,48,60]):
                    if f'_pl{pl}_' in tag or f'_pl{pl}' in tag.split('_L')[0]:
                        if mse < best[ds][pl][0]:
                            best[ds][pl] = (mse, tag)

print(f"\n{'DS':<8} {'pl':>4}  {'Best MSE':>9}  {'vs prev':>8}  {'vs xP14':>8}  {'win?':>6}  Config")
print("─"*80)
for ds in ['ETTh1','ETTh2','ILI']:
    for pl in ([192,336,720] if 'ETT' in ds else [24,36,48,60]):
        if best[ds][pl][0] < float('inf'):
            mse, cfg = best[ds][pl]
            vp = mse - prev[ds][pl]
            vx = mse - xpatch[ds][pl]
            win = '★ WIN' if round(mse,3) <= round(xpatch[ds][pl],3) else ('≈tie' if vx < 0.002 else '—')
            print(f"{ds:<8} {pl:>4}  {mse:>9.3f}  {vp:>+8.3f}  {vx:>+8.3f}  {win:>6}  {cfg}")
        else:
            print(f"{ds:<8} {pl:>4}  {'pending':>9}")
PYEOF
