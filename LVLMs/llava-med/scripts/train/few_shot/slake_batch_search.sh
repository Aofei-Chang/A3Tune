#!/bin/bash

ROOT_PATH=/data/aofei
dataset=Slake

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

epoch_num=6


for bbox_ratio in 1 1.5 2; do
    for visual_enhance_ratio in 0.1 0.2 0.3; do
        echo "Running with bbox_ratio: $bbox_ratio and visual_enhance_ratio: $visual_enhance_ratio"
        dir=llava_med/ours_few_shot/10%_top1024/all_expert_4_8_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}

        # torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
        #     llava/train/train_mem.py \
        #     --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
        #     --lora_enable True \
        #     --lora_r 64 \
        #     --lora_alpha 16 \
        #     --moe_lora_r 16 \
        #     --moe_lora_alpha 8 \
        #     --mm_projector_lr 2e-5 \
        #     --tune_mm_mlp_adapter False \
        #     --data_path ${ROOT_PATH}/hallucination/${dataset}/data/few_shot/training_masks_10%.json \
        #     --segment_path ${ROOT_PATH}/hallucination/${dataset}/data/training_segments.npz \
        #     --image_folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        #     --vision_tower openai/clip-vit-large-patch14 \
        #     --mm_vision_select_layer -2 \
        #     --mm_use_im_start_end True \
        #     --bf16 True \
        #     --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
        #     --num_train_epochs $epoch_num \
        #     --per_device_train_batch_size 1 \
        #     --per_device_eval_batch_size 4 \
        #     --gradient_accumulation_steps 8 \
        #     --evaluation_strategy "no" \
        #     --save_strategy "steps" \
        #     --save_steps 1000 \
        #     --save_total_limit 3 \
        #     --learning_rate 2e-4 \
        #     --weight_decay 0. \
        #     --warmup_ratio 0.03 \
        #     --lr_scheduler_type "cosine" \
        #     --logging_steps 1 \
        #     --tf32 True \
        #     --model_max_length 1024 \
        #     --gradient_checkpointing False \
        #     --lazy_preprocess True \
        #     --report_to wandb \
        #     --use_bbox True \
        #     --visual_focus False \
        #     --visual_enhance_ratio $visual_enhance_ratio \
        #     --bbox_ratio $bbox_ratio \
        #     --use_moe True \
        #     --dense_moe True \
        #     --use_mask True \
        #     --query_expert_num 4 \
        #     --visual_expert_num 8 \
        #     --use_kl False \
        #     --top_heads 1024
        #     # --fsdp "full_shard auto_wrap" \
        #     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

        # wait

        # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        #     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        #     --lora_r 16 \
        #     --lora_alpha 8 \
        #     --q_expert_num 4 \
        #     --k_expert_num 8 \
        #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
        #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

        python llava/eval/run_eval.py \
            --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
            --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
            --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt

        wait
    done
done
