ROOT_PATH=/data/aofei
dataset=VQA_RAD

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

dir=llava_med_1.5/lora/epoch3_bbox_01

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    llava/train/train_mem2.py \
    --model_name_or_path ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training.json \
    --image_folder ${ROOT_PATH}/hallucination/${dataset}/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --report_to wandb \
    --use_bbox True
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \