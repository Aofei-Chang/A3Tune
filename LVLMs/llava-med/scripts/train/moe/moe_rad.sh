ROOT_PATH=/data/aofei
dataset=VQA_RAD

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

visual_enhance_ratio=0.15
bbox_ratio=0.1
epoch_num=9

dir=llava_med/moe/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
    --lora_enable True \
    --mm_projector_lr 2e-5 \
    --tune_mm_mlp_adapter False \
    --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_bboxes.json \
    --image_folder ${ROOT_PATH}/hallucination/${dataset}/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
    --num_train_epochs $epoch_num \
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
    --use_bbox True \
    --visual_focus True \
    --visual_enhance_ratio $visual_enhance_ratio \
    --bbox_ratio $bbox_ratio \
    --use_moe True \
    --expert_num 4
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \