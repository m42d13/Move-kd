#!/bin/bash
set -euo pipefail

mkdir -p ./.cache/triton
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
if [ -z "${HF_HOME:-}" ] && [ -d "$HOME/.cache/huggingface" ]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_finetune.json \
    --model_name_or_path "${MODEL_NAME_OR_PATH:-lmsys/vicuna-7b-v1.5}" \
    --freeze_backbone True \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain/move-kd-7b-v1.1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR:-./checkpoints/finetune/full-adapter-move-kd-7b-v1.1}" \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS:-100}" \
    --per_device_train_batch_size "${BATCH_SIZE:-1}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-100}" \
    --save_total_limit 1 \
    --learning_rate "${LR:-2e-5}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --lazy_preprocess True \
    --report_to tensorboard \
    --finetune False \
    --tune_encoder True \
    --tune_all False \
    --moe_encoder True \
    --moe_encoder_dense False \
    --moe_encoder_lora_rank 32 \
    --moe_encoder_lora_alpha 1 \
    --moe_encoder_num_experts 3 \
    --moe_encoder_balance_w 0.01 \
    --kd True \
    --kd_w 0.5 \
    --kd_memory_w 0.8 \
    --pretrain_mole_encoder ./checkpoints/pretrain/move-kd-7b-v1.1/mole_encoder.bin \
    --pretrain_encoder_adapter ./checkpoints/pretrain/move-kd-7b-v1.1/encoder_adapter.bin \
    --tune_encoder_adapter True \
    --encoder_teachers clip sam eva convnext biomed_clip \
    --token_weight True \
    --teacher_weight True
