#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/baduc/MoVE-KD"

MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/finetune/move-kd-biomedclip-750}"
OUT_DIR="${OUT_DIR:-${ROOT}/playground/data/radimagenet_test/eval_full_move_kd_biomedclip_750}"

export MODEL_PATH
export OUT_DIR

echo "Evaluating MoVE-KD BiomedCLIP 750-step checkpoint"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUT_DIR}"

"${ROOT}/scripts/eval/eval_radimagenet_move_kd_full.sh"
