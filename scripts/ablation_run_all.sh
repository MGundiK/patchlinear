#!/bin/bash

# ============================================================
# MASTER ABLATION RUNNER
# ============================================================
# Runs all 5 ablation experiments in priority order.
# Total estimated: ~10 hours on Colab A100
#
# Priority order:
#   1. Component ablation    (~3 hrs)  — MUST HAVE for any paper
#   2. Gating position       (~1 hr)   — justifies pre_pointwise choice
#   3. Gate constraint range  (~1.5 hrs) — justifies [0.1, 0.9] constraint
#   4. Bottleneck dimension  (~3 hrs)  — justifies dim=32 choice
#   5. α_res initialization  (~1.5 hrs) — justifies 0.05 init
#
# If time-constrained: run 1 + 2, skip the rest
# (component + position are the minimum for a paper)
#
# PREREQUISITE: Run setup_ablation.sh first!

set -e

echo "============================================================"
echo "  GLPatch ABLATION EXPERIMENTS — MASTER RUNNER"
echo "  Start: $(date)"
echo "============================================================"

# Verify setup
if ! grep -q "GLPatch_ablation" exp/exp_main.py; then
    echo "ERROR: Run setup_ablation.sh first!"
    exit 1
fi
echo "Setup verified OK"
echo ""

# 1. Component ablation (MOST IMPORTANT)
echo "############################################################"
echo "  1/5 — COMPONENT ABLATION"
echo "############################################################"
bash scripts/ablation_1_component.sh

# 2. Gating position
echo "############################################################"
echo "  2/5 — GATING POSITION"
echo "############################################################"
bash scripts/ablation_2_gate_position.sh

# 3. Gate constraint range
echo "############################################################"
echo "  3/5 — GATE CONSTRAINT RANGE"
echo "############################################################"
bash scripts/ablation_5_gate_range.sh

# 4. Bottleneck dimension
echo "############################################################"
echo "  4/5 — BOTTLENECK DIMENSION"
echo "############################################################"
bash scripts/ablation_4_bottleneck_dim.sh

# 5. α_res initialization
echo "############################################################"
echo "  5/5 — RESIDUAL ALPHA INIT"
echo "############################################################"
bash scripts/ablation_3_res_alpha.sh

echo ""
echo "============================================================"
echo "  ALL ABLATION EXPERIMENTS COMPLETE"
echo "  End: $(date)"
echo "============================================================"
echo ""
echo "Log structure:"
echo "  ./logs/ablation/"
echo "  ├── component/       # Ablation 1"
echo "  ├── gate_position/   # Ablation 2"
echo "  ├── res_alpha/       # Ablation 3"
echo "  ├── gate_dim/        # Ablation 4"
echo "  └── gate_range/      # Ablation 5"
