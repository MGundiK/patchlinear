#!/usr/bin/env bash
# =============================================================================
# PatchLinear — ETT FAMILY STRIDE SWEEP
#
# Goal: ETT sweeps in HANDOVER fixed stride = patch_len/2 (s=8 with p=16).
# This sweep tests whether other stride ratios at the dataset's best
# (L, p) point can find improvements — particularly for ETTh1 H=336/720 and
# ETTh2 H=720 where there's room to grow.
#
# Strategy: at the per-horizon best (L, p, k) from the existing best-L
# table (Table 8), vary stride only.
#
#   patch_len = 16 → stride ∈ {4, 8, 12}
#   patch_len = 24 → stride ∈ {6, 12, 18}
#   patch_len = 32 → stride ∈ {8, 16, 24}
#   patch_len =  8 → stride ∈ {2, 4, 6}
#
# Each (dataset, horizon) cell gets 2 new strides (the existing one is
# already tested). Default is patch_len/2; this sweep tests ratios of
# 1/4 and 3/4.
#
# Coverage: 4 ETT × 4 horizons × 2 new strides × 1 seed = 32 single-seed runs
# Phase 2: 3-seed on top 2 wins per dataset = ~24 runs if 2 wins per dataset
# Total: ~32 + 24 = ~56 runs (~14h on T4)
#
# This is the lowest-priority sweep — small expected effect. Run last,
# only if the other three sweeps left compute budget.
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/sweep_ett_stride.sh 2>&1 | tee logs/sweep_ett_stride_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/sweep_ett_stride"
mkdir -p "$LOG_DIR"

TAG="sweep_ett_stride_$(date +%Y%m%d_%H%M%S)"
echo "Tag:  ${TAG}"
echo "Logs: ${LOG_DIR}"
echo ""

run_one() {
  local TAGNAME=$1; shift
  local LOG="${LOG_DIR}/${TAGNAME}.log"
  echo ">>> ${TAGNAME}"
  if python -u run.py "$@" > "$LOG" 2>&1; then
    grep "mse:" "$LOG" | tail -1 || echo "  (no mse line)"
  else
    echo "  !! FAILED  (see ${LOG})"
  fi
}

# Per-cell best (L, p, k) from Table 8 / table_bestL_final.tex
# Format: "dataset H L p k"
# Stride sweep: at each cell, try s = p/4 and s = 3p/4 (skip default p/2)
CELLS=(
  # ETTh1
  "ETTh1 96  512 16 7"
  "ETTh1 192 336 24 3"
  "ETTh1 336 512 24 3"
  "ETTh1 720 192 16 13"
  # ETTh2
  "ETTh2 96  720 32 3"
  "ETTh2 192 720 16 7"
  "ETTh2 336 512 32 13"
  "ETTh2 720 336 24 3"
  # ETTm1
  "ETTm1 96  336 24 3"
  "ETTm1 192 192 16 7"
  "ETTm1 336 336 16 7"
  "ETTm1 720 336 16 7"
  # ETTm2
  "ETTm2 96  512 16 7"
  "ETTm2 192 512 16 7"
  "ETTm2 336 336 16 3"
  "ETTm2 720 720  8 3"
)

# Map dataset → (data_path, data_name)
declare -A DATAPATH=(
  [ETTh1]="ETTh1.csv"
  [ETTh2]="ETTh2.csv"
  [ETTm1]="ETTm1.csv"
  [ETTm2]="ETTm2.csv"
)
declare -A DATANAME=(
  [ETTh1]="ETTh1"
  [ETTh2]="ETTh2"
  [ETTm1]="ETTm1"
  [ETTm2]="ETTm2"
)

ETT_COMMON="--is_training 1 --root_path ./dataset/ --features M \
            --label_len 48 --enc_in 7 \
            --d_model 64 --t_ff 128 --c_ff 16 --alpha 0.3 \
            --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
            --use_cross_channel 1 --use_alpha_gate 1 \
            --batch_size 2048 --lradj sigmoid \
            --train_epochs 100 --patience 10 --model PatchLinear"

# ETTm2 uses lr=0.0001, all others use lr=0.0005
get_lr() {
  case "$1" in
    ETTm2) echo "0.0001" ;;
    *)     echo "0.0005" ;;
  esac
}

SEED=2021

echo "================================================================"
echo "  ETT stride sweep — single-seed s=2021"
echo "  ${#CELLS[@]} cells × 2 new strides each = $((${#CELLS[@]} * 2)) runs"
echo "================================================================"

for cell in "${CELLS[@]}"; do
  read DS H L P K <<< "$cell"
  LR=$(get_lr "$DS")

  # New strides: P/4 and 3P/4 (rounded down). Skip if equal to default P/2.
  S_QUARTER=$((P / 4))
  S_THREEQ=$((3 * P / 4))
  S_DEFAULT=$((P / 2))

  for S in $S_QUARTER $S_THREEQ; do
    if [ "$S" -lt 1 ]; then continue; fi
    if [ "$S" -eq "$S_DEFAULT" ]; then continue; fi

    DES="ETTStride_p${P}_s${S}_k${K}_L${L}"
    TAGNAME="${DS}_pl${H}_${DES}_seed${SEED}"
    MID="${DS}_pl${H}_${DES}"

    run_one "$TAGNAME" \
      --seed "$SEED" --model_id "$MID" --des "$DES" \
      $ETT_COMMON \
      --data_path "${DATAPATH[$DS]}" --data "${DATANAME[$DS]}" \
      --seq_len "$L" --pred_len "$H" \
      --patch_len "$P" --stride "$S" --dw_kernel "$K" \
      --learning_rate "$LR"
  done
done

# ── summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STRIDE SWEEP SUMMARY"
echo "================================================================"

python3 - <<'PY'
import re, os

# Reference: best-L table 3-seed numbers (from table_bestL_final.tex)
REF = {
    ("ETTh1", 96):  0.352,  ("ETTh1", 192): 0.380,
    ("ETTh1", 336): 0.407,  ("ETTh1", 720): 0.487,
    ("ETTh2", 96):  0.226,  ("ETTh2", 192): 0.276,
    ("ETTh2", 336): 0.314,  ("ETTh2", 720): 0.390,
    ("ETTm1", 96):  0.275,  ("ETTm1", 192): 0.314,
    ("ETTm1", 336): 0.351,  ("ETTm1", 720): 0.417,
    ("ETTm2", 96):  0.149,  ("ETTm2", 192): 0.209,
    ("ETTm2", 336): 0.263,  ("ETTm2", 720): 0.339,
}

pat_set = re.compile(
    r'^(?P<ds>ETT[hm][12])_pl(?P<pl>\d+)'
    r'_(?P<des>ETTStride_p\d+_s\d+_k\d+_L\d+)'
    r'_PatchLinear_.+_(?P=des)_\d+$'
)
pat_des = re.compile(r'ETTStride_p(?P<p>\d+)_s(?P<s>\d+)_k(?P<k>\d+)_L(?P<L>\d+)')
pat_met = re.compile(r'mse:([\d.]+),\s*mae:([\d.]+)')

if not os.path.exists('result.txt'):
    print("No result.txt yet.")
    raise SystemExit(0)

rows = []
with open('result.txt') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
i = 0
while i < len(lines) - 1:
    m = pat_set.match(lines[i])
    if m:
        mm = pat_met.search(lines[i+1])
        if mm:
            dm = pat_des.match(m.group('des'))
            if dm:
                rows.append((m.group('ds'), int(m.group('pl')),
                             int(dm.group('p')), int(dm.group('s')),
                             int(dm.group('k')), int(dm.group('L')),
                             float(mm.group(1)), float(mm.group(2))))
    i += 1

if not rows:
    print("No stride sweep rows yet.")
    raise SystemExit(0)

# Group by (dataset, horizon)
from collections import defaultdict
groups = defaultdict(list)
for ds, pl, p, s, k, L, mse, mae in rows:
    groups[(ds, pl)].append((p, s, k, L, mse, mae))

print()
total_improvements = 0
for (ds, pl), entries in sorted(groups.items()):
    ref = REF.get((ds, pl), None)
    print(f"\n{ds} H={pl}  (paper best-L 3-seed: {ref:.3f})" if ref else f"\n{ds} H={pl}")
    print(f"  {'p':>3}  {'s':>3}  {'k':>3}  {'L':>4}  {'MSE':>9}  {'MAE':>9}  Δ")
    print("  " + "-" * 50)
    entries.sort(key=lambda e: e[4])
    for p, s, k, L, mse, mae in entries:
        delta_str = ""
        if ref:
            d = (mse - ref) / ref * 100
            delta_str = f"{d:+5.1f}%"
            if mse < ref:
                delta_str += " ✓"
                total_improvements += 1
        print(f"  {p:>3}  {s:>3}  {k:>3}  {L:>4}  {mse:>9.6f}  {mae:>9.6f}  {delta_str}")

print(f"\n  Total cells improving over paper best-L: {total_improvements}")
print(f"  → run 3-seed confirmation on the most promising")
PY

# ── Drive backup ────────────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_sweep_ett_stride_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_sweep_ett_stride_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."
