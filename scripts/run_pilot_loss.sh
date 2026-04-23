#!/usr/bin/env bash
# =============================================================================
# PatchLinear — Traffic PILOT: train_loss & early_stop_metric
#
# Hypothesis: on Traffic, early stopping on vali(wMAE) selects the wrong
# checkpoint (vali wMAE keeps dropping while test MSE has turned back up).
#
# Four variants per (horizon, seed):
#   Base       : train=wmae,  early_stop=wmae   (previous default behavior)
#   MseSelect  : train=wmae,  early_stop=mse    (pure selection fix)
#   HuberTrain : train=huber, early_stop=mse    (training objective also MSE-like)
#   MseTrain   : train=mse,   early_stop=mse    (train directly on MSE)
#
# Dataset: Traffic only.
# Horizons: 96, 720.
# Seeds: 2021, 2022, 2023.
# Total runs: 2 * 4 * 3 = 24.
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/run_pilot_loss.sh 2>&1 | tee logs/pilot_loss_master.log
# =============================================================================

set -uo pipefail   # no -e: one failed run should not kill the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/pilot_loss"
mkdir -p "$LOG_DIR"

PILOT_TAG="pilot_loss_$(date +%Y%m%d_%H%M%S)"
echo "Pilot tag: ${PILOT_TAG}"
echo "Logs:      ${LOG_DIR}"
echo ""

run_one() {
  local TAG=$1; shift
  local LOG="${LOG_DIR}/${TAG}.log"
  echo ">>> ${TAG}"
  if python -u run.py "$@" > "$LOG" 2>&1; then
    grep "mse:" "$LOG" | tail -1 || echo "  (no mse line found in log)"
  else
    echo "  !! FAILED  (see ${LOG})"
  fi
}

HORIZONS=(96 720)
SEEDS=(2021 2022 2023)

# Traffic config (matches the "updated" block in run_experiments.sh)
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

# Variant overrides (each sets its own --des so setting names differ)
BASE_OVR="--des Base      --train_loss wmae  --early_stop_metric wmae"
MSEL_OVR="--des MseSelect --train_loss wmae  --early_stop_metric mse"
HUBT_OVR="--des HuberTrain --train_loss huber --early_stop_metric mse --huber_delta 1.0"
MSET_OVR="--des MseTrain  --train_loss mse   --early_stop_metric mse"

VARIANTS=("Base" "MseSelect" "HuberTrain" "MseTrain")

echo "============================================================"
echo "  Dataset: Traffic"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
  for PL in "${HORIZONS[@]}"; do
    for VAR in "${VARIANTS[@]}"; do
      case "$VAR" in
        Base)       OVR="$BASE_OVR" ;;
        MseSelect)  OVR="$MSEL_OVR" ;;
        HuberTrain) OVR="$HUBT_OVR" ;;
        MseTrain)   OVR="$MSET_OVR" ;;
      esac
      TAG="Traffic_pl${PL}_s${SEED}_${VAR}"
      MODEL_ID="Traffic_${PL}_s${SEED}_${VAR}"
      run_one "$TAG" --seed "$SEED" --model_id "$MODEL_ID" \
                     $TRAFFIC_COMMON $OVR --pred_len "$PL"
    done
  done
done

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PILOT SUMMARY  (${PILOT_TAG})"
echo "============================================================"

python3 - <<'PY'
import re, os, collections, statistics

# Match pilot rows:  Traffic_<PL>_s<SEED>_<Variant>_PatchLinear_..._<Variant>_0
pat_set = re.compile(
    r'^Traffic_(?P<pl>\d+)_s(?P<seed>\d+)_(?P<var>Base|MseSelect|HuberTrain|MseTrain)'
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

variants = ['Base', 'MseSelect', 'HuberTrain', 'MseTrain']
pls      = sorted({pl for (pl, _, _) in last})
seeds    = sorted({s  for (_, s, _) in last})

if not pls:
    print("No Traffic pilot rows found yet.")
    raise SystemExit(0)

# Per-seed table
for seed in seeds:
    print(f"\nSeed {seed}")
    header = f"  {'horizon':<10}"
    for v in variants:
        header += f"  {v+' MSE':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pl in pls:
        line = f"  pl={pl:<7}"
        for v in variants:
            if (pl, seed, v) in last:
                line += f"  {last[(pl, seed, v)][0]:>13.6f}"
            else:
                line += f"  {'—':>13}"
        print(line)

# Mean-across-seeds table (the one that actually matters)
print("\nMean across seeds")
header = f"  {'horizon':<10}"
for v in variants:
    header += f"  {v+' MSE':>13}"
header += "  winner (Δ vs Base)"
print(header)
print("  " + "-" * (len(header) - 2))

overall = collections.Counter()
for pl in pls:
    means = {}
    for v in variants:
        vals = [last[(pl, s, v)][0] for s in seeds if (pl, s, v) in last]
        if vals:
            means[v] = statistics.mean(vals)
    line = f"  pl={pl:<7}"
    for v in variants:
        if v in means:
            line += f"  {means[v]:>13.6f}"
        else:
            line += f"  {'—':>13}"
    if means:
        winner = min(means, key=means.get)
        overall[winner] += 1
        base = means.get('Base')
        if base and winner != 'Base':
            delta = (means[winner] - base) / base * 100
            line += f"  {winner} ({delta:+.1f}%)"
        else:
            line += f"  {winner}"
    print(line)

print("\nWin counts (best mean-MSE per horizon):")
for v in variants:
    print(f"  {v}: {overall[v]}")
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
