#!/bin/bash

# ============================================================
# SETUP: Apply this patch before running ablation experiments
# Run from the repo root (e.g. /content/patch_gl_v2)
# ============================================================

echo "Patching codebase for ablation experiments..."

# 1. Copy ablation files
cp layers/network_glpatch_ablation.py layers/network_glpatch_ablation.py 2>/dev/null
cp models/GLPatch_ablation.py models/GLPatch_ablation.py 2>/dev/null
echo "  ✓ Ablation model files in place"

# 2. Register GLPatch_ablation in exp_main.py (add import + dict entry)
if ! grep -q "GLPatch_ablation" exp/exp_main.py; then
    sed -i '/from models import GLPatch/a from models import GLPatch_ablation' exp/exp_main.py
    sed -i "s/'GLPatch': GLPatch,/'GLPatch': GLPatch,\n            'GLPatch_ablation': GLPatch_ablation,/" exp/exp_main.py
    echo "  ✓ Registered GLPatch_ablation in exp/exp_main.py"
else
    echo "  ✓ GLPatch_ablation already registered in exp/exp_main.py"
fi

# 3. Add ablation args to run.py (before args = parser.parse_args())
if ! grep -q "use_gating" run.py; then
    sed -i "/parser.add_argument('--test_flop'/a\\
\\
# Ablation flags for GLPatch_ablation model\\
parser.add_argument('--use_gating', type=int, default=1, help='Enable inter-patch gating (1=yes, 0=no)')\\
parser.add_argument('--use_fusion', type=int, default=1, help='Enable adaptive stream fusion (1=yes, 0=no)')\\
parser.add_argument('--gate_position', type=str, default='pre_pointwise', help='Gating position: pre_depthwise, pre_pointwise, post_pointwise')\\
parser.add_argument('--res_alpha_init', type=float, default=0.05, help='Initial value for gating residual blend')\\
parser.add_argument('--gate_hidden_dim', type=int, default=32, help='Bottleneck dim for fusion gate (-1 for full-rank)')\\
parser.add_argument('--gate_min', type=float, default=0.1, help='Lower bound of gate constraint')\\
parser.add_argument('--gate_max', type=float, default=0.9, help='Upper bound of gate constraint')\\
parser.add_argument('--gate_reduction', type=int, default=4, help='Reduction ratio for gating MLP')" run.py
    echo "  ✓ Added ablation args to run.py"
else
    echo "  ✓ Ablation args already in run.py"
fi

echo ""
echo "Patch complete. You can now run:"
echo "  python run.py --model GLPatch_ablation --use_gating 0 --use_fusion 0  # = xPatch baseline"
echo "  python run.py --model GLPatch_ablation --use_gating 1 --use_fusion 0  # + gating only"
echo "  python run.py --model GLPatch_ablation --use_gating 0 --use_fusion 1  # + fusion only"
echo "  python run.py --model GLPatch_ablation --use_gating 1 --use_fusion 1  # full GLPatch (default)"
