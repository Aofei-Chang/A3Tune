#!/bin/bash

seeds=(4)
seed=4
heads=(128)
# heads=(0)
ROOT_PATH=/data/aofei
dataset=Slake

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers
head=128
visual_enhance_ratio=0
# visual_enhance_ratio=0
# bbox_ratios=(0.01 0.03 0.05 0.08 0.1)
bbox_ratios=(0.01 0.03 0.06)
epoch_num=6
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0
# for seed in "${seeds[@]}"; do
for bbox_ratio in "${bbox_ratios[@]}"; do
    echo "Running with head: $head, bbox ratio: $bbox_ratio"
    dir=llava_med_1.5/ours_v2_true_search/KL_expert_8_16_rank16_8_lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/head${head}/epoch${epoch_num}_seed${seed}
    # dir=llava_med_1.5/lora_1207/epoch${epoch_num}_seed${seed}

    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --lora_enable True --lora_r 64 --lora_alpha 16 --mm_projector_lr 2e-5 \
        --version v1 \
        --moe_lora_r 16 \
        --moe_lora_alpha 8 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks.json \
        --segment_path ${ROOT_PATH}/hallucination/${dataset}/data/training_segments.npz \
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
        --num_train_epochs $epoch_num \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
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
        --visual_focus False \
        --visual_enhance_ratio $visual_enhance_ratio \
        --bbox_ratio $bbox_ratio \
        --use_moe True \
        --dense_moe True \
        --use_mask True \
        --query_expert_num 8 \
        --visual_expert_num 16 \
        --seed $seed \
        --top_heads $head \
        --use_kl True \
        --moe_balance_ratio 0.01 \
        --top_visual_moe_experts 3
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
        

    wait

    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --lora_r 16 \
        --lora_alpha 8 \
        --q_expert_num 8 \
        --k_expert_num 16 \
        --top_moe_num 3 \
        --baseline beam \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt

    wait
done


