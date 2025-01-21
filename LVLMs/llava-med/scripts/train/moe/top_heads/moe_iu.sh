#!/bin/bash

seeds=(4)
seed=4
ROOT_PATH=/data/aofei
dataset=IU_Xray

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

# visual_enhance_ratio=0.03
visual_enhance_ratio=0
# bbox_ratio=0.06
epoch_num=6
head_num=128

# heads=(32 64 128 256 512 1024)
# heads=(128)
# bbox_ratios=(0.01 0.02 0.03 0.04 0.05 0.1)
# for head_num in "${heads[@]}"; do
# for bbox_ratio in 0.03 0.06 0.09 0.12; do
#     echo "Running with seed: $seed, head_num: $head_num, bbox_ratio: $bbox_ratio"
#     # dir=llava_med/ours_v0/KL_loss/all_expert_8_16_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}
#     dir=llava_med/ours_v1_woKL/top${head_num}_balance_top3/all_expert_8_16_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}
    # dir=llava_med/epoch9/seed${seed}

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
    #     --data_path ${ROOT_PATH}/hallucination/CARES/${dataset}/training_masks_top4.json \
    #     --segment_path ${ROOT_PATH}/hallucination/CARES/${dataset}/training_segments_top4.npz \
    #     --image_folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
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
    #     --query_expert_num 8 \
    #     --visual_expert_num 16 \
    #     --use_kl False \
    #     --seed $seed \
    #     --top_heads $head_num \
    #     --use_visual_prompt False \
    #     --moe_balance_ratio 0.01 \
    #     --top_visual_moe_experts 3
    #     # --fsdp "full_shard auto_wrap" \
    #     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

    # wait

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
    #     --lora_r 16 \
    #     --lora_alpha 8 \
    #     --q_expert_num 8 \
    #     --k_expert_num 16 \
    #     --top_moe_num 3 \
    #     --baseline greedy \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    # python llava/eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt

    # wait
    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
    #     --lora_r 16 \
    #     --lora_alpha 8 \
    #     --q_expert_num 8 \
    #     --k_expert_num 16 \
    #     --top_moe_num 3 \
    #     --baseline beam \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

#     python llava/eval/run_eval.py \
#         --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#         --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt
#     wait
# done

for bbox_ratio in 0.9 0.12; do
    echo "Running with seed: $seed, head_num: $head_num, bbox_ratio: $bbox_ratio"
    # dir=llava_med/ours_v0/KL_loss/all_expert_8_16_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}
    dir=llava_med/ours_v1_woKL/top${head_num}_balance_top3/all_expert_4_8_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}
    # dir=llava_med/epoch9/seed${seed}

    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
        llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --moe_lora_r 16 \
        --moe_lora_alpha 8 \
        --mm_projector_lr 2e-5 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/CARES/${dataset}/training_masks_top4.json \
        --segment_path ${ROOT_PATH}/hallucination/CARES/${dataset}/training_segments_top4.npz \
        --image_folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
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
        --query_expert_num 4 \
        --visual_expert_num 8 \
        --use_kl False \
        --seed $seed \
        --top_heads $head_num \
        --use_visual_prompt False \
        --moe_balance_ratio 0.01 \
        --top_visual_moe_experts 2
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

    wait

    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
        --lora_r 16 \
        --lora_alpha 8 \
        --q_expert_num 4 \
        --k_expert_num 8 \
        --top_moe_num 2 \
        --baseline greedy \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt

    wait
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
        --lora_r 16 \
        --lora_alpha 8 \
        --q_expert_num 4 \
        --k_expert_num 8 \
        --top_moe_num 2 \
        --baseline beam \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt
done