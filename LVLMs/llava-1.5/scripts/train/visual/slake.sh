#!/bin/bash
ROOT_PATH=/data/aofei
dataset=Slake
# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers
# export CUDA_VISIBLE_DEVICES='2,3'
epoch_num=6
dir=visual/epoch${epoch_num}

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${ROOT_PATH}/LLM/llava-v1.5 \
    --version v1 \
    --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_bboxes.json \
    --image_folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${ROOT_PATH}/LLM/llava-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/llava_1.5/${dir}/checkpoints \
    --num_train_epochs $epoch_num \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_bbox True \
    --visual_focus True
