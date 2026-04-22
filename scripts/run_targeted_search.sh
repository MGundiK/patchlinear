#!/usr/bin/env bash
# =============================================================================
# Targeted per-horizon hyperparameter search
# Varies: L (seq_len) × patch_len/stride × dw_kernel
# All other params fixed at dataset-tuned values
# Single-seed (2021) — confirm winners with 3 seeds separately
# =============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_DIR="logs/targeted"
mkdir -p "$LOG_DIR" logs
SEED=2021

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  python -u run.py "$@" > "$LOG" 2>&1 || true
  if grep -q "mse:" "$LOG"; then
    grep "mse:" "$LOG" | tail -1
  else
    echo "  [FAILED] $(tail -3 $LOG | head -1)"
  fi
}

# ── ETTh1 — pl=192,336,720 ────────────────────────────────────────────────
# Base: d=64, t_ff=128, c_ff=16, lr=5e-4, batch=2048, enc_in=7
echo "=== ETTh1 ==="
for PL in 192 336 720; do
  for L in 96 192 336 512 720; do
    for PS in "8 4" "16 8" "24 12" "32 16"; do
      P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
      for K in 3 7 13; do
        TAG="ETTh1_pl${PL}_L${L}_p${P}s${S}_k${K}"
        run_one "$TAG" \
          --is_training 1 --root_path ./dataset/ \
          --data_path ETTh1.csv --data ETTh1 --features M \
          --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
          --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
          --patch_len ${P} --stride ${S} --dw_kernel ${K} \
          --batch_size 2048 --learning_rate 0.0005 \
          --lradj sigmoid --train_epochs 100 --patience 10 \
          --model PatchLinear --seed ${SEED} --des Exp \
          --model_id "$TAG"
      done
    done
  done
done

# ── ETTh2 — pl=96,336,720 ─────────────────────────────────────────────────
echo "=== ETTh2 ==="
for PL in 96 336 720; do
  for L in 96 192 336 512 720; do
    for PS in "8 4" "16 8" "24 12" "32 16"; do
      P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
      for K in 3 7 13; do
        TAG="ETTh2_pl${PL}_L${L}_p${P}s${S}_k${K}"
        run_one "$TAG" \
          --is_training 1 --root_path ./dataset/ \
          --data_path ETTh2.csv --data ETTh2 --features M \
          --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
          --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
          --patch_len ${P} --stride ${S} --dw_kernel ${K} \
          --batch_size 2048 --learning_rate 0.0005 \
          --lradj sigmoid --train_epochs 100 --patience 10 \
          --model PatchLinear --seed ${SEED} --des Exp \
          --model_id "$TAG"
      done
    done
  done
done

# ── ETTm1 — pl=96 ─────────────────────────────────────────────────────────
echo "=== ETTm1 ==="
for L in 96 192 336 512 720; do
  for PS in "8 4" "16 8" "24 12" "32 16"; do
    P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
    for K in 3 7 13; do
      TAG="ETTm1_pl96_L${L}_p${P}s${S}_k${K}"
      run_one "$TAG" \
        --is_training 1 --root_path ./dataset/ \
        --data_path ETTm1.csv --data ETTm1 --features M \
        --seq_len ${L} --label_len 48 --pred_len 96 --enc_in 7 \
        --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
        --patch_len ${P} --stride ${S} --dw_kernel ${K} \
        --batch_size 2048 --learning_rate 0.0005 \
        --lradj sigmoid --train_epochs 100 --patience 10 \
        --model PatchLinear --seed ${SEED} --des Exp \
        --model_id "$TAG"
    done
  done
done

# ── ETTm2 — pl=336,720 ────────────────────────────────────────────────────
# Note: ETTm2 uses lr=1e-4
echo "=== ETTm2 ==="
for PL in 336 720; do
  for L in 96 192 336 512 720; do
    for PS in "8 4" "16 8" "24 12" "32 16"; do
      P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
      for K in 3 7 13; do
        TAG="ETTm2_pl${PL}_L${L}_p${P}s${S}_k${K}"
        run_one "$TAG" \
          --is_training 1 --root_path ./dataset/ \
          --data_path ETTm2.csv --data ETTm2 --features M \
          --seq_len ${L} --label_len 48 --pred_len ${PL} --enc_in 7 \
          --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
          --patch_len ${P} --stride ${S} --dw_kernel ${K} \
          --batch_size 2048 --learning_rate 0.0001 \
          --lradj sigmoid --train_epochs 100 --patience 10 \
          --model PatchLinear --seed ${SEED} --des Exp \
          --model_id "$TAG"
      done
    done
  done
done

# ── ILI — pl=24,36,48 ─────────────────────────────────────────────────────
# Base: d=64, c=8, lr=0.01, lradj=type3, simplified (no decomp/seas/fusion)
# patch variants must satisfy: L > patch_len
echo "=== ILI ==="
for PL in 24 36 48; do
  for L in 36 104 148; do
    for PS in "4 2" "6 3" "8 4" "12 6"; do
      P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
      if [ ${L} -le ${P} ]; then continue; fi  # skip invalid
      TAG="ILI_pl${PL}_L${L}_p${P}s${S}"
      run_one "$TAG" \
        --is_training 1 --root_path ./dataset/ \
        --data_path national_illness.csv --data custom --features M \
        --seq_len ${L} --label_len 18 --pred_len ${PL} --enc_in 7 \
        --d_model 64 --t_ff 128 --c_ff 8 --alpha 0.3 \
        --patch_len ${P} --stride ${S} --dw_kernel 3 \
        --use_decomp 0 --use_seas_stream 0 --use_fusion_gate 0 \
        --batch_size 32 --learning_rate 0.01 \
        --lradj type3 --train_epochs 100 --patience 10 \
        --model PatchLinear --seed ${SEED} --des Exp \
        --model_id "$TAG"
    done
  done
done

# ── Solar — pl=96 ─────────────────────────────────────────────────────────
# Base: d=192, t_ff=384, c_ff=34, lr=8e-4, batch=512, enc_in=137
echo "=== Solar ==="
for L in 96 192 336 512 720; do
  for PS in "8 4" "16 8" "24 12" "32 16"; do
    P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
    for K in 3 7 13; do
      TAG="Solar_pl96_L${L}_p${P}s${S}_k${K}"
      run_one "$TAG" \
        --is_training 1 --root_path ./dataset/ \
        --data_path solar.txt --data Solar --features M \
        --seq_len ${L} --label_len 48 --pred_len 96 --enc_in 137 \
        --d_model 192 --t_ff 384 --c_ff 34 --alpha 0.3 \
        --patch_len ${P} --stride ${S} --dw_kernel ${K} \
        --batch_size 512 --learning_rate 0.0008 \
        --lradj sigmoid --train_epochs 100 --patience 10 \
        --model PatchLinear --seed ${SEED} --des Exp \
        --model_id "$TAG"
    done
  done
done

# ── Exchange — pl=720 ─────────────────────────────────────────────────────
# Base: d=64, c=16, lr=2e-6, batch=32, enc_in=8, k=3 (fixed)
# Focus: try different L and patch sizes — k=3 already optimal
echo "=== Exchange ==="
for L in 96 192 336 512 720; do
  for PS in "4 2" "8 4" "16 8"; do
    P=$(echo $PS|cut -d' ' -f1); S=$(echo $PS|cut -d' ' -f2)
    TAG="Exchange_pl720_L${L}_p${P}s${S}"
    run_one "$TAG" \
      --is_training 1 --root_path ./dataset/ \
      --data_path exchange_rate.csv --data custom --features M \
      --seq_len ${L} --label_len 48 --pred_len 720 --enc_in 8 \
      --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
      --patch_len ${P} --stride ${S} --dw_kernel 3 \
      --batch_size 32 --learning_rate 0.000002 \
      --lradj sigmoid --train_epochs 100 --patience 10 \
      --model PatchLinear --seed ${SEED} --des Exp \
      --model_id "$TAG"
  done
done

echo ""
echo "=== SUMMARY — best per (dataset, pred_len) ==="
python3 - << 'PYEOF'
import re, os, numpy as np

log_dir = "logs/targeted"
targets = {
    'ETTh1':  [192,336,720], 'ETTh2':  [96,336,720],
    'ETTm1':  [96],          'ETTm2':  [336,720],
    'ILI':    [24,36,48],    'Solar':  [96],
    'Exchange':[720],
}
xpatch = {
    'ETTh1':{192:0.376,336:0.391,720:0.442},
    'ETTh2':{96:0.226,336:0.312,720:0.384},
    'ETTm1':{96:0.275}, 'ETTm2':{336:0.264,720:0.338},
    'ILI':{24:1.188,36:1.226,48:1.254},
    'Solar':{96:0.173}, 'Exchange':{720:0.867},
}
prev_best = {
    'ETTh1':{192:0.386,336:0.412,720:0.493},
    'ETTh2':{96:0.228,336:0.315,720:0.392},
    'ETTm1':{96:0.277}, 'ETTm2':{336:0.265,720:0.339},
    'ILI':{24:1.373,36:1.314,48:1.275},
    'Solar':{96:0.174}, 'Exchange':{720:0.938},
}

print(f"{'DS':<12} {'pl':>4}  {'New best':>9}  {'vs prev':>8}  {'vs xP14':>8}  {'Config'}")
print("─"*75)
for ds, pls in targets.items():
    for pl in pls:
        best_mse=float('inf'); best_cfg=None
        for f in os.listdir(log_dir):
            if not f.startswith(f"{ds}_pl{pl}_") or not f.endswith('.log'): continue
            path=os.path.join(log_dir,f)
            for line in open(path):
                m=re.search(r'mse:\s*([\d.]+)',line)
                if m:
                    v=float(m.group(1))
                    if v<best_mse: best_mse=v; best_cfg=f[:-4]
        if best_cfg:
            xl=xpatch[ds].get(pl,'—')
            pb=prev_best[ds].get(pl,'—')
            vs_prev=f"{best_mse-pb:+.3f}" if isinstance(pb,float) else '—'
            vs_xl=f"{best_mse-xl:+.3f}" if isinstance(xl,float) else '—'
            win='★' if isinstance(xl,float) and round(best_mse,3)<=round(xl,3) else ''
            print(f"{ds:<12} {pl:>4}  {best_mse:>9.3f}  {vs_prev:>8}  {vs_xl:>8}  {best_cfg} {win}")
        else:
            print(f"{ds:<12} {pl:>4}  {'pending':>9}")
PYEOF
