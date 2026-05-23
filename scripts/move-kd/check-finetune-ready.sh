#!/bin/bash
set -euo pipefail

missing=0

check_file() {
    if [ -f "$1" ]; then
        echo "OK   $1"
    else
        echo "MISS $1"
        missing=1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "OK   $1"
    else
        echo "MISS $1"
        missing=1
    fi
}

echo "CUDA"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda_device_{i}: {torch.cuda.get_device_name(i)}")
PY

echo
echo "Required full fine-tune assets"
check_file playground/data/llava_v1_5_mix665k.json
check_file checkpoints/pretrain/move-kd-7b-v1.1/mm_projector.bin
check_file checkpoints/pretrain/move-kd-7b-v1.1/mole_encoder.bin
check_file checkpoints/pretrain/move-kd-7b-v1.1/encoder_adapter.bin

echo
echo "Debug fine-tune assets"
check_file playground/data/llava_v1_5_debug_8.json
check_file checkpoints/pretrain/debug-move-kd-7b-v1.1/mm_projector.bin
check_file checkpoints/pretrain/debug-move-kd-7b-v1.1/mole_encoder.bin
check_file checkpoints/pretrain/debug-move-kd-7b-v1.1/encoder_adapter.bin
check_dir "$HOME/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5"
check_dir "$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336"

exit "$missing"
