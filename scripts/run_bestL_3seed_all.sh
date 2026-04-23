#!/usr/bin/env bash
# =============================================================================
# 3-seed confirmation for all remaining single-seed best-L configs
# 72 runs total — ETT/ETTm ~3min, Traffic/Elec ~8min, Solar ~20min each
# =============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_DIR="logs/bestL_3seed"
mkdir -p "$LOG_DIR"

run3() {
  local TAG=$1; shift
  echo ">>> ${TAG}"
  for SEED in 2021 2022 2023; do
    local LOG="${LOG_DIR}/${TAG}_s${SEED}.log"
    python -u run.py --is_training 1 "$@" \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "${TAG}_s${SEED}" \
      > "$LOG" 2>&1 || true
    grep -m1 "mse:" "$LOG" | tail -1 || echo "  [s${SEED} FAILED]"
  done
}

# ── ETTh1 (d=64, lr=5e-4, batch=2048, enc_in=7) ──────────────────
echo "=== ETTh1 ==="
BASE="--root_path ./dataset/ --features M --data_path ETTh1.csv --data ETTh1
  --enc_in 7 --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3
  --batch_size 2048 --learning_rate 0.0005
  --lradj sigmoid --train_epochs 100 --patience 10"

run3 ETTh1_pl96_L512_p16k7   $BASE --seq_len 512 --label_len 48 --pred_len 96  --patch_len 16 --stride 8  --dw_kernel 7
run3 ETTh1_pl192_L336_p24k3  $BASE --seq_len 336 --label_len 48 --pred_len 192 --patch_len 24 --stride 12 --dw_kernel 3
run3 ETTh1_pl336_L512_p24k3  $BASE --seq_len 512 --label_len 48 --pred_len 336 --patch_len 24 --stride 12 --dw_kernel 3
run3 ETTh1_pl720_L192_p16k13 $BASE --seq_len 192 --label_len 48 --pred_len 720 --patch_len 16 --stride 8  --dw_kernel 13

# ── ETTh2 (same base config) ──────────────────────────────────────
echo "=== ETTh2 ==="
BASE="--root_path ./dataset/ --features M --data_path ETTh2.csv --data ETTh2
  --enc_in 7 --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3
  --batch_size 2048 --learning_rate 0.0005
  --lradj sigmoid --train_epochs 100 --patience 10"

run3 ETTh2_pl192_L720_p16k7   $BASE --seq_len 720 --label_len 48 --pred_len 192 --patch_len 16 --stride 8  --dw_kernel 7
run3 ETTh2_pl336_L512_p32k13  $BASE --seq_len 512 --label_len 48 --pred_len 336 --patch_len 32 --stride 16 --dw_kernel 13
run3 ETTh2_pl720_L336_p24k3   $BASE --seq_len 336 --label_len 48 --pred_len 720 --patch_len 24 --stride 12 --dw_kernel 3

# ── ETTm1 (same base config) ──────────────────────────────────────
echo "=== ETTm1 ==="
BASE="--root_path ./dataset/ --features M --data_path ETTm1.csv --data ETTm1
  --enc_in 7 --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3
  --batch_size 2048 --learning_rate 0.0005
  --lradj sigmoid --train_epochs 100 --patience 10"

run3 ETTm1_pl192_L192_p16k7  $BASE --seq_len 192 --label_len 48 --pred_len 192 --patch_len 16 --stride 8  --dw_kernel 7
run3 ETTm1_pl336_L336_p16k7  $BASE --seq_len 336 --label_len 48 --pred_len 336 --patch_len 16 --stride 8  --dw_kernel 7
run3 ETTm1_pl720_L336_p16k7  $BASE --seq_len 336 --label_len 48 --pred_len 720 --patch_len 16 --stride 8  --dw_kernel 7

# ── ETTm2 (lr=1e-4) ───────────────────────────────────────────────
echo "=== ETTm2 ==="
BASE="--root_path ./dataset/ --features M --data_path ETTm2.csv --data ETTm2
  --enc_in 7 --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3
  --batch_size 2048 --learning_rate 0.0001
  --lradj sigmoid --train_epochs 100 --patience 10"

run3 ETTm2_pl96_L512_p16k7   $BASE --seq_len 512 --label_len 48 --pred_len 96  --patch_len 16 --stride 8 --dw_kernel 7
run3 ETTm2_pl192_L512_p16k7  $BASE --seq_len 512 --label_len 48 --pred_len 192 --patch_len 16 --stride 8 --dw_kernel 7

# ── Traffic (d=256, t_ff=512, c_ff=128, lr=5e-3, batch=64, patience=5) ──
echo "=== Traffic ==="
BASE="--root_path ./dataset/ --features M --data_path traffic.csv --data custom
  --enc_in 862 --d_model 256 --t_ff 512 --c_ff 128 --alpha 0.3
  --batch_size 64 --learning_rate 0.005 --patience 5
  --lradj sigmoid --train_epochs 100"

run3 Traffic_pl96_L720_p16k7  $BASE --seq_len 720 --label_len 48 --pred_len 96  --patch_len 16 --stride 8 --dw_kernel 7
run3 Traffic_pl192_L720_p16k7 $BASE --seq_len 720 --label_len 48 --pred_len 192 --patch_len 16 --stride 8 --dw_kernel 7
run3 Traffic_pl336_L720_p16k7 $BASE --seq_len 720 --label_len 48 --pred_len 336 --patch_len 16 --stride 8 --dw_kernel 7
run3 Traffic_pl720_L512_p16k7 $BASE --seq_len 512 --label_len 48 --pred_len 720 --patch_len 16 --stride 8 --dw_kernel 7

# ── Electricity (d=256, t_ff=512, c_ff=128, lr=1e-2, batch=256) ──
echo "=== Electricity ==="
BASE="--root_path ./dataset/ --features M --data_path electricity.csv --data custom
  --enc_in 321 --d_model 256 --t_ff 512 --c_ff 128 --alpha 0.3
  --batch_size 256 --learning_rate 0.01
  --lradj sigmoid --train_epochs 100 --patience 10"

run3 Elec_pl96_L192_p16k7   $BASE --seq_len 192 --label_len 48 --pred_len 96  --patch_len 16 --stride 8 --dw_kernel 7
run3 Elec_pl192_L192_p16k7  $BASE --seq_len 192 --label_len 48 --pred_len 192 --patch_len 16 --stride 8 --dw_kernel 7
run3 Elec_pl336_L192_p16k7  $BASE --seq_len 192 --label_len 48 --pred_len 336 --patch_len 16 --stride 8 --dw_kernel 7
run3 Elec_pl720_L336_p16k7  $BASE --seq_len 336 --label_len 48 --pred_len 720 --patch_len 16 --stride 8 --dw_kernel 7

# ── Solar (d=192, t_ff=384, c_ff=34, lr=8e-4, batch=512, enc_in=137) ──
echo "=== Solar ==="
BASE="--root_path ./dataset/ --features M --data_path solar_AL.txt --data Solar
  --enc_in 137 --d_model 192 --t_ff 384 --c_ff 34 --alpha 0.3
  --batch_size 512 --learning_rate 0.0008
  --lradj sigmoid --train_epochs 100 --patience 10"

run3 Solar_pl96_L720_p16k7   $BASE --seq_len 720 --label_len 48 --pred_len 96  --patch_len 16 --stride 8 --dw_kernel 7
run3 Solar_pl192_L512_p16k7  $BASE --seq_len 512 --label_len 48 --pred_len 192 --patch_len 16 --stride 8 --dw_kernel 7
run3 Solar_pl336_L512_p16k7  $BASE --seq_len 512 --label_len 48 --pred_len 336 --patch_len 16 --stride 8 --dw_kernel 7
run3 Solar_pl720_L720_p16k7  $BASE --seq_len 720 --label_len 48 --pred_len 720 --patch_len 16 --stride 8 --dw_kernel 7

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "=== SUMMARY ==="
python3 - << 'PYEOF'
import re, os, numpy as np
from collections import defaultdict

log_dir = "logs/bestL_3seed"
xpatch = {
    'ETTh1_pl96':(0.354,0.379),'ETTh1_pl192':(0.376,0.395),'ETTh1_pl336':(0.391,0.415),'ETTh1_pl720':(0.442,0.459),
    'ETTh2_pl192':(0.275,0.330),'ETTh2_pl336':(0.312,0.360),'ETTh2_pl720':(0.384,0.418),
    'ETTm1_pl192':(0.315,0.355),'ETTm1_pl336':(0.355,0.376),'ETTm1_pl720':(0.419,0.411),
    'ETTm2_pl96':(0.153,0.240),'ETTm2_pl192':(0.213,0.280),
    'Traffic_pl96':(0.364,0.233),'Traffic_pl192':(0.377,0.241),'Traffic_pl336':(0.388,0.243),'Traffic_pl720':(0.437,0.273),
    'Elec_pl96':(0.126,0.217),'Elec_pl192':(0.140,0.232),'Elec_pl336':(0.156,0.249),'Elec_pl720':(0.190,0.281),
    'Solar_pl96':(0.173,0.197),'Solar_pl192':(0.193,0.216),'Solar_pl336':(0.196,0.224),'Solar_pl720':(0.212,0.219),
}

results = defaultdict(list)
for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.log'): continue
    tag = re.sub(r'_s\d{4}\.log$','',fname)
    for line in open(os.path.join(log_dir,fname)):
        m = re.search(r'mse:\s*([\d.]+).*?mae:\s*([\d.]+)',line)
        if m: results[tag].append((float(m.group(1)),float(m.group(2))))

print(f"{'Config':<30} {'MSE':>7} {'±':>6}  {'MAE':>7}  {'xP MSE':>7}  {'Δ':>7}  result")
print("─"*85)
wins=0; total=0
for tag in sorted(results.keys()):
    vals = results[tag]
    if len(vals)<2: continue
    mm=np.mean([v[0] for v in vals]); ms=np.std([v[0] for v in vals])
    am=np.mean([v[1] for v in vals])
    xk = None
    for k in xpatch:
        ds,pl = k.rsplit('_pl',1)
        if tag.startswith(ds) and f'pl{pl}_' in tag:
            xk=k; break
    if xk:
        xm,xa=xpatch[xk]; d=mm-xm
        win='★ WIN' if round(mm,3)<=round(xm,3) else ('≈tie' if d<0.003 else '—')
        if round(mm,3)<=round(xm,3): wins+=1
        total+=1
        print(f"{tag:<30} {mm:>7.3f} {ms:>6.4f}  {am:>7.3f}  {xm:>7.3f}  {d:>+7.3f}  {win}")
print(f"\nMSE wins: {wins}/{total}")
PYEOF
