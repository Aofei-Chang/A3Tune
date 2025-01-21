ROOT_PATH=/data/aofei
dataset=Slake

export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

dir=llava_med_1.5/lora/abd/epoch6
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks_abd.json \
    --image_folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb