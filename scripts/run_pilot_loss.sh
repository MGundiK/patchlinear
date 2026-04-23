#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Traffic PILOT (slim): train_loss & early_stop_metric
#
# Hypothesis: on Traffic, early stopping on vali(wMAE) selects the wrong
# checkpoint (vali wMAE keeps dropping while test MSE turns back up).
#
# Variants (3):
#   Base       : train=wmae,  early_stop=wmae    — previous default behavior
#   MseSelect  : train=wmae,  early_stop=mse     — pure selection fix
#   HuberTrain : train=huber, early_stop=mse     — also makes objective MSE-like
#   (MseTrain dropped — risky at lr=0.005, MseSelect+HuberTrain already tell
#    us whether selection vs. objective is the bigger lever.)
#
# Dataset: Traffic only.
# Horizons: 96, 720.
# Seed:    2021 only.
# Total:   6 runs.
#
# Resume-aware: if a matching row exists in result.txt, the run is SKIPPED.
# This preserves whatever in-flight Base run you may already have.
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/run_pilot_loss.sh 2>&1 | tee logs/pilot_loss_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/pilot_loss"
mkdir -p "$LOG_DIR"

PILOT_TAG="pilot_loss_$(date +%Y%m%d_%H%M%S)"
echo "Pilot tag: ${PILOT_TAG}"
echo "Logs:      ${LOG_DIR}"
echo ""

# ── resume helper ──────────────────────────────────────────────────────────
# Check if a given model_id already has results in result.txt. We look for
# a setting line ending in the model_id's trailing "_<Variant>_0" pattern
# — `run.py` always appends `_<des>_<ii>` and we use des=Variant.
already_done() {
  local MODEL_ID=$1       # e.g. Traffic_96_s2021_Base
  local VARIANT=$2        # e.g. Base
  if [ ! -f result.txt ]; then return 1; fi
  # Match any line that starts with MODEL_ID and ends with _VARIANT_<digits>
  # (the setting line), and confirm the next line has mse:/mae:.
  grep -q "^${MODEL_ID}_.*_${VARIANT}_[0-9]\+\$" result.txt
}

run_one() {
  local TAG=$1; shift
  local MODEL_ID=$1; shift
  local VARIANT=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"

  if already_done "$MODEL_ID" "$VARIANT"; then
    local EXISTING
    EXISTING=$(grep -A1 "^${MODEL_ID}_.*_${VARIANT}_[0-9]\+\$" result.txt \
               | grep "mse:" | tail -1)
    echo ">>> ${TAG}  [SKIP — already in result.txt]"
    echo "    ${EXISTING}"
    return 0
  fi

  echo ">>> ${TAG}"
  if python -u run.py --model_id "$MODEL_ID" "$@" > "$LOG" 2>&1; then
    grep "mse:" "$LOG" | tail -1 || echo "  (no mse line found in log)"
  else
    echo "  !! FAILED  (see ${LOG})"
  fi
}

HORIZONS=(96 720)
SEED=2021

# Traffic config mirrors the "updated" block in run_experiments.sh
TRAFFIC_COMMON="--is_training 1 --root_path ./dataset/ --features M \
                --seq_len 96 --label_len 48 \
                --data_path traffic.csv --data custom --enc_in 862 \
                --d_model 256 --t_ff 512 --c_ff 128 \
                --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
                --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
                --use_cross_channel 1 --use_alpha_gate 1 \
                --batch_size 64 --learning_rate 0.005 \
                --lradj sigmoid --train_epochs 100 --patience 5 \
                --model PatchLinear"

BASE_OVR="--des Base       --train_loss wmae  --early_stop_metric wmae"
MSEL_OVR="--des MseSelect  --train_loss wmae  --early_stop_metric mse"
HUBT_OVR="--des HuberTrain --train_loss huber --early_stop_metric mse --huber_delta 1.0"

VARIANTS=("Base" "MseSelect" "HuberTrain")

echo "============================================================"
echo "  Dataset: Traffic (1 seed, 2 horizons, 3 variants = 6 runs)"
echo "============================================================"

for PL in "${HORIZONS[@]}"; do
  for VAR in "${VARIANTS[@]}"; do
    case "$VAR" in
      Base)       OVR="$BASE_OVR" ;;
      MseSelect)  OVR="$MSEL_OVR" ;;
      HuberTrain) OVR="$HUBT_OVR" ;;
    esac
    TAG="Traffic_pl${PL}_s${SEED}_${VAR}"
    MODEL_ID="Traffic_${PL}_s${SEED}_${VAR}"
    run_one "$TAG" "$MODEL_ID" "$VAR" --seed "$SEED" \
            $TRAFFIC_COMMON $OVR --pred_len "$PL"
  done
done

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PILOT SUMMARY  (${PILOT_TAG})"
echo "============================================================"

python3 - <<'PY'
import re, os

# Match pilot rows:  Traffic_<PL>_s<SEED>_<Variant>_PatchLinear_..._<Variant>_0
pat_set = re.compile(
    r'^Traffic_(?P<pl>\d+)_s(?P<seed>\d+)_(?P<var>Base|MseSelect|HuberTrain)'
    r'_PatchLinear_.+_(?P=var)_\d+$'
)
pat_met = re.compile(r'mse:([\d.]+),\s*mae:([\d.]+)')

if not os.path.exists('result.txt'):
    print("No result.txt — nothing to summarise.")
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
            rows.append((
                int(m.group('pl')),
                int(m.group('seed')),
                m.group('var'),
                float(mm.group(1)),
                float(mm.group(2)),
            ))
    i += 1

# Keep only the LAST occurrence per (pl, seed, var) in case of reruns
last = {}
for pl, seed, var, mse, mae in rows:
    last[(pl, seed, var)] = (mse, mae)

variants = ['Base', 'MseSelect', 'HuberTrain']
pls = sorted({pl for (pl, _, _) in last})

if not pls:
    print("No Traffic pilot rows found yet.")
    raise SystemExit(0)

header = f"  {'horizon':<10}"
for v in variants:
    header += f"  {v+' MSE':>14}  {v+' MAE':>14}"
header += "  winner (Δ vs Base)"
print(header)
print("  " + "-" * (len(header) - 2))

for pl in pls:
    line = f"  pl={pl:<7}"
    mses = {}
    for v in variants:
        if (pl, 2021, v) in last:
            mse, mae = last[(pl, 2021, v)]
            line += f"  {mse:>14.6f}  {mae:>14.6f}"
            mses[v] = mse
        else:
            line += f"  {'—':>14}  {'—':>14}"
    if mses:
        winner = min(mses, key=mses.get)
        base = mses.get('Base')
        if base and winner != 'Base':
            delta = (mses[winner] - base) / base * 100
            line += f"  {winner} ({delta:+.1f}%)"
        else:
            line += f"  {winner}"
    print(line)
PY

# ── Drive backup (Colab) ────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_pilot_loss_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_pilot_loss_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."
