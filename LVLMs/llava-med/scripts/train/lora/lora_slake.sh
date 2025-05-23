#!/bin/bash

seeds=(4)
ROOT_PATH=/data/aofei
dataset=Slake

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    dir="llava_med/lora/epoch6_v2/0.1_data/seed${seed}"

    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
        llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
        --lora_enable True \
        --mm_projector_lr 2e-5 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/${dataset}/data/few_shot/training_masks_10%.json \
        --image_folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end True \
        --bf16 True \
        --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
        --num_train_epochs 6 \
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
        --seed $seed
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    wait

    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt

    wait
done