#!/usr/bin/env bash
# =============================================================================
# PatchLinear — SOLAR d=256 FINAL 3-SEED RUN
#
# Goal: replace the current d=192 paper Solar config with d=256_p8s4, which
# per HANDOVER_RESULTS §11.2 won 4 bold + 1 ul vs d=192's 3 bold + 1 ul on
# Phase 2 testing, but was rejected for "efficiency narrative" reasons.
# Now that paper framing is being adjusted, the win-count gain matters more
# than the parameter saving.
#
# Config: d=256, t_ff=512, p=8, s=4, dw_kernel=7, c_ff=34, lr=0.0008
# Other settings inherited from run_experiments.sh Solar block.
#
# Total: 4 horizons × 3 seeds = 12 runs (~3-4h on T4)
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/sweep_solar_d256.sh 2>&1 | tee logs/sweep_solar_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/sweep_solar"
mkdir -p "$LOG_DIR"

TAG="sweep_solar_$(date +%Y%m%d_%H%M%S)"
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

HORIZONS=(96 192 336 720)
SEEDS=(2021 2022 2023)
DES="SolarD256"

# Solar config: d=256, t_ff=512, p=8, s=4, dw_kernel=7, c_ff=34
SOLAR_COMMON="--is_training 1 --root_path ./dataset/ --features M \
              --seq_len 96 --label_len 48 \
              --data_path solar.txt --data Solar --enc_in 137 \
              --d_model 256 --t_ff 512 --c_ff 34 \
              --patch_len 8 --stride 4 --dw_kernel 7 --alpha 0.3 \
              --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
              --use_cross_channel 1 --use_alpha_gate 1 \
              --batch_size 512 --learning_rate 0.0008 \
              --lradj sigmoid --train_epochs 100 --patience 10 \
              --model PatchLinear"

echo "================================================================"
echo "  Solar d=256_p8s4 — 4 horizons × 3 seeds = 12 runs"
echo "================================================================"

for SEED in "${SEEDS[@]}"; do
  for PL in "${HORIZONS[@]}"; do
    TAGNAME="Solar_pl${PL}_s${SEED}_${DES}"
    MID="Solar_pl${PL}_s${SEED}_${DES}"
    run_one "$TAGNAME" \
      --seed "$SEED" --model_id "$MID" --des "$DES" \
      $SOLAR_COMMON --pred_len "$PL"
  done
done

# ── summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUMMARY: d=256 vs d=192 (paper) reference"
echo "================================================================"

python3 - <<'PY'
import re, os, statistics

# Reference: d=192 paper numbers from HANDOVER_RESULTS §11.2
D192 = {
    96:  {"mse": [0.182516, 0.189273, 0.188137], "mae_mean": 0.201},
    192: {"mse": [0.230668, 0.219682, 0.214775], "mae_mean": 0.224},
    336: {"mse": [0.236370, 0.251896, 0.238434], "mae_mean": 0.241},
    720: {"mse": [0.247653, 0.262697, 0.257897], "mae_mean": 0.246},
}

pat_set = re.compile(
    r'^Solar_pl(?P<pl>\d+)_s(?P<seed>\d+)_SolarD256'
    r'_PatchLinear_.+_SolarD256_\d+$'
)
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
            rows.append((int(m.group('pl')), int(m.group('seed')),
                         float(mm.group(1)), float(mm.group(2))))
    i += 1

new = {}
for pl, seed, mse, mae in rows:
    new[(pl, seed)] = (mse, mae)

if not new:
    print("No d=256 rows found yet.")
    raise SystemExit(0)

print(f"\n  {'horizon':<8}  {'d=192 MSE':>10}  {'d=256 MSE':>10}  {'Δ':>7}    "
      f"{'d=192 MAE':>10}  {'d=256 MAE':>10}  {'Δ':>7}")
print("  " + "-" * 76)
for pl in [96, 192, 336, 720]:
    new_mses = [new[(pl, s)][0] for s in [2021, 2022, 2023] if (pl, s) in new]
    new_maes = [new[(pl, s)][1] for s in [2021, 2022, 2023] if (pl, s) in new]
    n = len(new_mses)
    if n == 0:
        print(f"  pl={pl:<5}     —           —          —          —           —          —")
        continue
    d192_mse = statistics.mean(D192[pl]["mse"])
    d192_mae = D192[pl]["mae_mean"]
    new_mse = statistics.mean(new_mses)
    new_mae = statistics.mean(new_maes)
    dmse = (new_mse - d192_mse) / d192_mse * 100
    dmae = (new_mae - d192_mae) / d192_mae * 100
    n_mark = f" (n={n})" if n < 3 else ""
    print(f"  pl={pl:<5}  {d192_mse:>10.6f}  {new_mse:>10.6f}  {dmse:>+6.1f}%    "
          f"{d192_mae:>10.6f}  {new_mae:>10.6f}  {dmae:>+6.1f}%{n_mark}")
PY

# ── Drive backup ────────────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_sweep_solar_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_sweep_solar_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."
