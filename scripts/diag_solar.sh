#!/usr/bin/env bash
# =============================================================================
# PatchLinear — SOLAR DIAGNOSTIC: cross-session drift test
#
# Question: are the d=256 numbers we just got "real measurements of d=256",
# or did the environment drift since the paper-table runs?
#
# Test: rerun the EXACT paper-table d=192 Solar config in the current
# session and compare to the paper-table values.
#
# Paper-table d=192 Solar (3-seed):
#   H=96   seed 2021: 0.182516, mean 0.187
#   H=192  seed 2021: 0.230668, mean 0.222
#   H=336  seed 2021: 0.236370, mean 0.242
#   H=720  seed 2021: 0.247653, mean 0.256
#
# This script reruns just seed 2021 across all 4 horizons.
# 4 runs total, ~75-90 min on T4.
#
# Diagnosis logic:
#   |delta| < 1.5%  → environment is reproducible, code is fine
#                     → d=256 numbers we got are genuine (d=256 just is worse)
#                     → no rollback needed
#   |delta| 1.5-3%  → mild drift, normal for cross-session
#                     → can't cleanly compare d=256 vs d=192 without
#                       same-session matched runs
#   |delta| > 3%    → significant environmental drift
#                     → need to investigate; rollback might help
#                     → check pip freeze / cuda versions
#
# Usage (Colab):
#   %%bash
#   mkdir -p /content/PatchLinear/logs
#   cd /content/PatchLinear
#   bash scripts/diag_solar.sh 2>&1 | tee logs/diag_solar_master.log
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/diag_solar"
mkdir -p "$LOG_DIR"

TAG="diag_solar_$(date +%Y%m%d_%H%M%S)"
echo "Tag:  ${TAG}"
echo "Logs: ${LOG_DIR}"
echo ""

# ── environment fingerprint ────────────────────────────────────────────────
# Capture this so if results differ from the paper, you can compare against
# whatever environment fingerprint the paper-table runs had (if you saved one).
echo "================================================================"
echo "  ENVIRONMENT FINGERPRINT"
echo "================================================================"
{
  date
  echo ""
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1 | head -20 || echo "nvidia-smi unavailable"
  echo ""
  echo "--- python/torch/cuda versions ---"
  python -c "
import sys, torch, numpy
print(f'python:    {sys.version.split()[0]}')
print(f'torch:     {torch.__version__}')
print(f'cuda:      {torch.version.cuda}')
print(f'cudnn:     {torch.backends.cudnn.version()}')
print(f'numpy:     {numpy.__version__}')
print(f'gpu:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
"
  echo ""
  echo "--- git status ---"
  git rev-parse HEAD 2>/dev/null || echo "(not a git repo)"
  git status --porcelain 2>/dev/null | head -10
} | tee "${LOG_DIR}/env_fingerprint.txt"
echo ""

# ── run the paper-table d=192 config, seed 2021, all 4 horizons ────────────
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

# IMPORTANT: this is the EXACT paper-table Solar config (d=192).
# Any deviation from this defeats the diagnostic purpose.
SOLAR_PAPER_CONFIG="--is_training 1 --root_path ./dataset/ --features M \
                    --seq_len 96 --label_len 48 \
                    --data_path solar.txt --data Solar --enc_in 137 \
                    --d_model 192 --t_ff 384 --c_ff 34 \
                    --patch_len 16 --stride 8 --dw_kernel 7 --alpha 0.3 \
                    --use_decomp 1 --use_seas_stream 1 --use_fusion_gate 1 \
                    --use_cross_channel 1 --use_alpha_gate 1 \
                    --batch_size 512 --learning_rate 0.0008 \
                    --lradj sigmoid --train_epochs 100 --patience 10 \
                    --model PatchLinear"

echo "================================================================"
echo "  Solar d=192 (paper config) — seed 2021, all 4 horizons"
echo "================================================================"

SEED=2021
DES="DiagD192"

for PL in 96 192 336 720; do
  TAGNAME="Solar_pl${PL}_s${SEED}_${DES}"
  MID="Solar_pl${PL}_s${SEED}_${DES}"
  run_one "$TAGNAME" \
    --seed "$SEED" --model_id "$MID" --des "$DES" \
    $SOLAR_PAPER_CONFIG --pred_len "$PL"
done

# ── diagnosis ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  DIAGNOSIS"
echo "================================================================"

python3 - <<'PY'
import re, os

# Paper-table seed 2021 values from HANDOVER_RESULTS §11.2 (and earlier)
# These are the per-seed values, NOT 3-seed means.
PAPER_S2021 = {
    96:  0.182516,
    192: 0.230668,
    336: 0.236370,
    720: 0.247653,
}

pat_set = re.compile(
    r'^Solar_pl(?P<pl>\d+)_s2021_DiagD192'
    r'_PatchLinear_.+_DiagD192_\d+$'
)
pat_met = re.compile(r'mse:([\d.]+),\s*mae:([\d.]+)')

if not os.path.exists('result.txt'):
    print("No result.txt yet.")
    raise SystemExit(0)

rows = {}
with open('result.txt') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
i = 0
while i < len(lines) - 1:
    m = pat_set.match(lines[i])
    if m:
        mm = pat_met.search(lines[i+1])
        if mm:
            rows[int(m.group('pl'))] = (float(mm.group(1)), float(mm.group(2)))
    i += 1

if not rows:
    print("No diagnostic rows found yet.")
    raise SystemExit(0)

print()
print(f"  {'horizon':<8}  {'paper s2021':>11}  {'today s2021':>11}  {'Δ MSE':>8}  verdict")
print("  " + "-" * 70)

deltas = []
for pl in [96, 192, 336, 720]:
    if pl not in rows:
        print(f"  pl={pl:<5}  {PAPER_S2021[pl]:>11.6f}  {'—':>11}  {'—':>8}  not run")
        continue
    paper = PAPER_S2021[pl]
    today, _ = rows[pl]
    pct = (today - paper) / paper * 100
    deltas.append(pct)
    if abs(pct) < 0.5:
        verdict = "matches  (cuDNN-deterministic)"
    elif abs(pct) < 1.5:
        verdict = "OK  (within typical seed noise)"
    elif abs(pct) < 3.0:
        verdict = "mild drift"
    elif abs(pct) < 6.0:
        verdict = "noticeable drift"
    else:
        verdict = "SIGNIFICANT DRIFT"
    print(f"  pl={pl:<5}  {paper:>11.6f}  {today:>11.6f}  {pct:>+7.1f}%  {verdict}")

if len(deltas) == 4:
    avg_abs = sum(abs(d) for d in deltas) / 4
    avg_signed = sum(deltas) / 4
    print()
    print(f"  Mean absolute delta: {avg_abs:.1f}%")
    print(f"  Mean signed delta:   {avg_signed:+.1f}%")
    print()
    print("  ─── INTERPRETATION ───────────────────────────────────────────")
    if avg_abs < 1.5:
        print("    Environment is reproducible. Your repo is fine.")
        print("    The Solar d=256 numbers are GENUINE measurements of d=256.")
        print("    d=256 is just genuinely worse than d=192 cross-session,")
        print("    and §11.2's claim was advantaged by session-matching.")
        print("    → Keep paper Solar at d=192. No action needed.")
    elif avg_abs < 3.0:
        print("    Mild drift, normal for cross-session Colab runs.")
        print("    Cannot cleanly compare d=256 vs d=192 without same-session")
        print("    matched runs. The d=256 sweep was inconclusive.")
        print("    → Keep paper Solar at d=192 (it's internally consistent")
        print("      with the rest of the paper table).")
    elif avg_abs < 6.0 and avg_signed > 0:
        print("    Noticeable drift, all in the 'worse than paper' direction.")
        print("    Something has changed between paper-runs and now — could be")
        print("    Colab GPU assignment, cuDNN version, torch update, or repo.")
        print("    → Check git diff against paper commit.")
        print("    → Check pip freeze if you have a saved one from paper runs.")
        print("    → Consider rollback IF git diff shows training-path changes.")
    else:
        print("    SIGNIFICANT drift. Something material has changed.")
        print("    → DO investigate before any further sweep work.")
        print("    → Steps:")
        print("       1. git diff <paper-tag> HEAD -- run.py exp/exp_main.py models/")
        print("       2. Compare current pip freeze with paper-time saved freeze")
        print("       3. Check if Colab GPU class differs (T4 vs L4 vs A100)")
PY

# ── Drive backup ────────────────────────────────────────────────────────────
if [ -d "/content/drive/MyDrive" ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp "${SCRIPT_DIR}/result.txt" \
     "/content/drive/MyDrive/PatchLinear_diag_solar_${STAMP}.txt"
  cp -r "${LOG_DIR}" \
     "/content/drive/MyDrive/PatchLinear_diag_solar_logs_${STAMP}/"
  echo "Backed up to Drive (stamp: ${STAMP})"
fi

echo ""
echo "Done."
