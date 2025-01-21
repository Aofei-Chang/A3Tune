ROOT_PATH=/data/aofei
dataset=VQA_RAD

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

dir=llava_1.5/full

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
    --lora_enable False \
    --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training.json \
    --image_folder ${ROOT_PATH}/hallucination/${dataset}/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
    # --tune_mm_mlp_adapter True \