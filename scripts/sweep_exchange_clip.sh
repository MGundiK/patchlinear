#!/usr/bin/env bash
# =============================================================================
# PatchLinear — EXCHANGE H=720 GRADIENT CLIPPING EXPERIMENT
#
# Goal: investigate whether the H=720 collapse on Exchange (PL 0.938 vs
# xPatch 0.891) is caused by exploding gradients or is genuinely structural.
#
# Hypothesis: Exchange near-random-walk dynamics + arctan-weighted MAE loss
# at H=720 may produce occasional large gradients that destabilise training.
# Adding gradient clipping (max_norm=0.5 or 1.0) might prevent collapse.
#
# Strategy:
#   Phase 1: longer L (192, 336, 512) with current lr=2e-6, single-seed,
#            see if any L > 96 stops collapsing under gradient clipping.
#   Phase 2: if Phase 1 finds a working config, 3-seed confirm.
#
# NOTE: gradient clipping is NOT currently a flag in run.py. This script
# requires adding `--max_grad_norm` flag. See bottom of script for the
# minimal patch needed to exp_main.py if you want to actually run this.
#
# Phase 1: 3 L × 2 clip_norm × 1 seed = 6 runs (~3h on T4)
#
# Usage:
#   1. Apply the patch at the bottom to run.py and exp/exp_main.py first.
#   2. Then run as usual:
#      %%bash
#      mkdir -p /content/PatchLinear/logs
#      cd /content/PatchLinear
#      bash scripts/sweep_exchange_clip.sh 2>&1 | tee logs/sweep_exchange_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/sweep_exchange"
mkdir -p "$LOG_DIR"

TAG="sweep_exchange_$(date +%Y%m%d_%H%M%S)"
echo "Tag:  ${TAG}"
echo "Logs: ${LOG_DIR}"
echo ""

# ── Sanity check: --max_grad_norm flag must exist ──────────────────────────
if ! python -c "import argparse; import sys; sys.argv=['run.py','--help']; \
                exec(open('run.py').read())" 2>&1 | grep -q "max_grad_norm"; then
  cat <<'EOF'
ERROR: run.py does not support --max_grad_norm.
You need to apply the patch described at the bottom of this script first.

Quick patch summary:
  1. In run.py, add:
       parser.add_argument("--max_grad_norm", type=float, default=0.0,
           help="If > 0, clip grad norm to this value (0 = no clipping)")
  2. In exp/exp_main.py, train() loop, after `loss.backward()`:
       if getattr(self.args, 'max_grad_norm', 0.0) > 0:
           torch.nn.utils.clip_grad_norm_(
               self.model.parameters(), self.args.max_grad_norm)
EOF
  exit 1
fi

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

# Exchange config (deviates only on seq_len, max_grad_norm)
EXCHANGE_COMMON="--is_training 1 --root_path ./dataset/ --features M \
                 --label_len 48 \
                 --data_path exchange_rate.csv --data custom --enc_in 8 \
                 --d_model 64 --t_ff 128 --c_ff 16 \
                 --patch_len 8 --stride 4 --dw_kernel 3 --alpha 0.3 \
                 --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
                 --use_cross_channel 1 --use_alpha_gate 1 \
                 --batch_size 32 --learning_rate 0.000002 \
                 --lradj sigmoid --train_epochs 100 --patience 10 \
                 --model PatchLinear"

# ── Phase 1: clipping × longer L ───────────────────────────────────────────
echo "================================================================"
echo "  Exchange H=720 clip sweep — single-seed s=2021"
echo "  Reference: PL paper 0.938 (collapsed), xPatch 0.891"
echo "================================================================"

PHASE1_L=(192 336 512)
PHASE1_CLIP=(0.5 1.0)
SEED=2021
PL=720

for L in "${PHASE1_L[@]}"; do
  for CLIP in "${PHASE1_CLIP[@]}"; do
    DES="ExchClip_L${L}_c${CLIP}"
    TAGNAME="Exchange_pl720_${DES}_s${SEED}"
    MID="Exchange_pl720_${DES}"
    run_one "$TAGNAME" \
      --seed "$SEED" --model_id "$MID" --des "$DES" \
      $EXCHANGE_COMMON \
      --pred_len "$PL" \
      --seq_len "$L" \
      --max_grad_norm "$CLIP"
  done
done

# ── summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"

python3 - <<'PY'
import re, os

REF_PL = 0.938
REF_XPATCH = 0.891

pat_set = re.compile(
    r'^Exchange_pl720_(?P<des>ExchClip_L\d+_c[\d.]+)'
    r'_PatchLinear_.+_(?P=des)_\d+$'
)
pat_des = re.compile(r'ExchClip_L(?P<L>\d+)_c(?P<c>[\d.]+)')
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
                rows.append((int(dm.group('L')), float(dm.group('c')),
                             float(mm.group(1)), float(mm.group(2))))
    i += 1

if not rows:
    print("No clip sweep rows yet.")
    raise SystemExit(0)

rows.sort(key=lambda r: r[2])
print(f"\nExchange H=720  (PL paper: {REF_PL}, xPatch: {REF_XPATCH})")
print(f"  {'L':>4}  {'clip':>5}  {'MSE':>9}  {'MAE':>9}  status")
print("  " + "-" * 50)
for L, c, mse, mae in rows:
    status = ""
    if mse < REF_XPATCH:
        status = "BEATS xPatch"
    elif mse < REF_PL:
        status = "improves over paper"
    elif mse > 1.0:
        status = "still collapsing"
    print(f"  {L:>4}  {c:>5.1f}  {mse:>9.6f}  {mae:>9.6f}  {status}")
PY

# ── Drive backup ────────────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_sweep_exchange_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_sweep_exchange_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."

# =============================================================================
# REQUIRED CODE PATCH (apply BEFORE running this script):
#
# 1. In run.py — add this argument near the other optimisation flags:
#
#    parser.add_argument("--max_grad_norm", type=float, default=0.0,
#        help="If > 0, clip gradient norm to this value (0 = no clipping)")
#
# 2. In exp/exp_main.py — find the train() loop, locate `loss.backward()`,
#    and add ONE LINE immediately after it (before model_optim.step()):
#
#        loss.backward()
#        if getattr(self.args, 'max_grad_norm', 0.0) > 0:
#            torch.nn.utils.clip_grad_norm_(
#                self.model.parameters(), self.args.max_grad_norm)
#        model_optim.step()
#
# Both changes are backward-compatible — default 0.0 means no clipping,
# matching previous behaviour exactly.
# =============================================================================
