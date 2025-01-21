#!/bin/bash

seeds=(4)
ROOT_PATH=/data/aofei
dataset=IU_Xray

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    dir="llava_med/lora_report/epoch8/seed${seed}"

    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
        llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
        --lora_enable True \
        --mm_projector_lr 2e-5 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/${dataset}/data_report/training.json \
        --image_folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end True \
        --bf16 True \
        --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
        --num_train_epochs 8 \
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

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints
    #     --baseline DoLa \

    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.csv \
    #     --report True

    # wait
done