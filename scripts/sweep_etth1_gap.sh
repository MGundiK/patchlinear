#!/usr/bin/env bash
# =============================================================================
# PatchLinear — ETTh1 H336/H720 GAP-CLOSING SWEEP
#
# Goal: close the visible gap to xPatch on ETTh1 H=336 (+0.016) and
# H=720 (+0.045). Per HANDOVER_RESULTS §11, the L=96 + d=64 configuration
# leaves these two horizons as the worst PL cells on ETT.
#
# Hypothesis: longer L + higher d_model + larger DW kernel may close the gap.
#
# Strategy: 2-phase
#   Phase 1: 24 single-seed (s=2021) configs on H=336 + H=720
#            d ∈ {64, 128}, L ∈ {512, 720}, k ∈ {7, 13, 19}
#   Phase 2: 3-seed confirmation on top 2 configs per horizon
#
# Phase 1: 2 horizons × 2 d × 2 L × 3 k × 1 seed = 24 runs (~6h on T4)
# Phase 2: 4 candidates × 2 horizons × 3 seeds = 24 runs (~6h)
# Total worst case: ~12h. Practical: Phase 1 first, decide, then Phase 2.
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/sweep_etth1_gap.sh 2>&1 | tee logs/sweep_etth1_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/sweep_etth1"
mkdir -p "$LOG_DIR"

TAG="sweep_etth1_$(date +%Y%m%d_%H%M%S)"
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

# ETTh1 base config (deviates only on d_model, t_ff, seq_len, dw_kernel)
ETTH1_COMMON="--is_training 1 --root_path ./dataset/ --features M \
              --label_len 48 \
              --data_path ETTh1.csv --data ETTh1 --enc_in 7 \
              --c_ff 16 --patch_len 16 --stride 8 --alpha 0.3 \
              --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
              --use_cross_channel 1 --use_alpha_gate 1 \
              --batch_size 2048 --learning_rate 0.0005 \
              --lradj sigmoid --train_epochs 100 --patience 10 \
              --model PatchLinear"

# ── Phase 1: single-seed scouting grid ─────────────────────────────────────
echo "================================================================"
echo "  PHASE 1: ETTh1 H336/H720 scouting (single-seed s=2021)"
echo "  d ∈ {64,128}, L ∈ {512,720}, k ∈ {7,13,19}"
echo "================================================================"

PHASE1_HORIZONS=(336 720)
PHASE1_D=(64 128)
PHASE1_L=(512 720)
PHASE1_K=(7 13 19)
SEED=2021

for PL in "${PHASE1_HORIZONS[@]}"; do
  for D in "${PHASE1_D[@]}"; do
    # Scale t_ff with d_model (2x convention)
    TFF=$((D * 2))
    for L in "${PHASE1_L[@]}"; do
      for K in "${PHASE1_K[@]}"; do
        DES="ETThSweep_d${D}_L${L}_k${K}"
        TAG="ETTh1_pl${PL}_${DES}_s${SEED}"
        MID="ETTh1_pl${PL}_${DES}"
        run_one "$TAG" \
          --seed "$SEED" \
          --model_id "$MID" \
          --des "$DES" \
          $ETTH1_COMMON \
          --pred_len "$PL" \
          --seq_len "$L" \
          --d_model "$D" \
          --t_ff "$TFF" \
          --dw_kernel "$K"
      done
    done
  done
done

# ── Phase 1 summary ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  PHASE 1 SUMMARY"
echo "================================================================"

python3 - <<'PY'
import re, os

# Reference numbers from HANDOVER_RESULTS §1.1 (3-seed) and xPatch Table 14
REF = {
    336: {"PL_3seed": 0.442, "xPatch": 0.391, "gap": +0.051},
    720: {"PL_3seed": 0.490, "xPatch": 0.442, "gap": +0.048},
}

# Match scouting rows
pat_set = re.compile(
    r'^ETTh1_pl(?P<pl>\d+)_(?P<des>ETThSweep_d\d+_L\d+_k\d+)'
    r'_PatchLinear_.+_(?P=des)_\d+$'
)
pat_des = re.compile(r'ETThSweep_d(?P<d>\d+)_L(?P<L>\d+)_k(?P<k>\d+)')
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
                rows.append((
                    int(m.group('pl')),
                    int(dm.group('d')),
                    int(dm.group('L')),
                    int(dm.group('k')),
                    float(mm.group(1)),
                    float(mm.group(2)),
                ))
    i += 1

# Keep last occurrence per (pl, d, L, k)
last = {}
for pl, d, L, k, mse, mae in rows:
    last[(pl, d, L, k)] = (mse, mae)

if not last:
    print("No ETTh1 sweep rows found yet.")
    raise SystemExit(0)

for pl in [336, 720]:
    print(f"\nETTh1 H={pl}  (PL paper 3-seed: {REF[pl]['PL_3seed']:.3f}  |  xPatch: {REF[pl]['xPatch']:.3f}  |  gap: +{REF[pl]['gap']:.3f})")
    print(f"  {'config':<22}  {'MSE':>9}  {'MAE':>9}  {'Δ vs paper':>12}")
    print("  " + "-" * 60)
    rows_at_pl = [(d, L, k, mse, mae) for (p, d, L, k), (mse, mae) in last.items() if p == pl]
    rows_at_pl.sort(key=lambda r: r[3])  # sort by MSE
    for d, L, k, mse, mae in rows_at_pl:
        delta = (mse - REF[pl]['PL_3seed']) / REF[pl]['PL_3seed'] * 100
        marker = "  ✓ improves" if mse < REF[pl]['PL_3seed'] else ""
        marker += "  ✓ beats xPatch" if mse < REF[pl]['xPatch'] else ""
        print(f"  d={d:<3} L={L:<3} k={k:<3}     {mse:>9.6f}  {mae:>9.6f}  {delta:>+11.1f}%{marker}")

print()
print("="*70)
print("  PHASE 2 RECOMMENDATION")
print("="*70)
print("  Pick 2 best configs per horizon (lowest MSE).")
print("  Run Phase 2 with --seeds 2022, 2023 to confirm 3-seed.")
print("  Manual: copy the winning (d, L, k) into the Phase 2 block below")
print("          (currently commented out) and uncomment.")
PY

# ── Phase 2: 3-seed confirmation (MANUAL — uncomment after Phase 1) ────────
# After Phase 1, uncomment and edit this block with the top 2 configs per
# horizon from the summary above. Example:
#
# PHASE2_CONFIGS=(
#   "336 128 720 13"   # H d L k
#   "336 128 720 19"
#   "720 128 720 13"
#   "720 64  720 19"
# )
# PHASE2_SEEDS=(2022 2023)
# for cfg in "${PHASE2_CONFIGS[@]}"; do
#   read PL D L K <<< "$cfg"
#   TFF=$((D * 2))
#   for SEED in "${PHASE2_SEEDS[@]}"; do
#     DES="ETThSweep_d${D}_L${L}_k${K}"
#     TAG="ETTh1_pl${PL}_${DES}_s${SEED}"
#     MID="ETTh1_pl${PL}_${DES}"
#     run_one "$TAG" --seed "$SEED" --model_id "$MID" --des "$DES" \
#       $ETTH1_COMMON --pred_len "$PL" --seq_len "$L" \
#       --d_model "$D" --t_ff "$TFF" --dw_kernel "$K"
#   done
# done

# ── Drive backup ────────────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_sweep_etth1_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_sweep_etth1_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."
