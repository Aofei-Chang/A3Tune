#!/bin/bash

seeds=(4)
seed=4
ROOT_PATH=/data/aofei
dataset=Slake

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

visual_enhance_ratio=0
# visual_enhance_ratio=0
bbox_ratio=0.03
epoch_num=6


# 1.without attention tuning
# dir=llava_med/ablation/no_attn_tuning

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
#     llava/train/train_mem.py \
#     --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
#     --lora_enable True \
#     --moe_lora_r 16 \
#     --moe_lora_alpha 8 \
#     --mm_projector_lr 2e-5 \
#     --tune_mm_mlp_adapter False \
#     --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks.json \
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
#     --save_steps 500 \
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
#     --use_bbox False \
#     --visual_focus False \
#     --visual_enhance_ratio 0 \
#     --bbox_ratio 0 \
#     --use_moe True \
#     --dense_moe True \
#     --use_mask False \
#     --query_expert_num 8 \
#     --visual_expert_num 16 \
#     --seed $seed \
#     --top_heads 128 \
#     --moe_balance_ratio 0.01 \
#     --top_visual_moe_experts 3
#     # --fsdp "full_shard auto_wrap" \
#     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

# python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
#     --lora_r 16 \
#     --lora_alpha 8 \
#     --q_expert_num 8 \
#     --k_expert_num 16 \
#     --top_moe_num 3 \
#     --baseline beam \
#     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

# python llava/eval/run_eval.py \
#     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt

# python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
#     --lora_r 16 \
#     --lora_alpha 8 \
#     --q_expert_num 8 \
#     --k_expert_num 16 \
#     --top_moe_num 3 \
#     --baseline greedy \
#     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
#     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

# python llava/eval/run_eval.py \
#     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
#     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt



# 2.no query MoE
# bbox_ratio=0.1

# dir=llava_med/ablation/no_query_moe

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
#     llava/train/train_mem.py \
#     --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
#     --lora_enable True \
#     --moe_lora_r 16 \
#     --moe_lora_alpha 8 \
#     --mm_projector_lr 2e-5 \
#     --tune_mm_mlp_adapter False \
#     --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks.json \
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
#     --save_steps 500 \
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
#     --visual_enhance_ratio 0 \
#     --bbox_ratio $bbox_ratio \
#     --use_moe True \
#     --dense_moe True \
#     --use_mask True \
#     --query_expert_num 1 \
#     --visual_expert_num 16 \
#     --seed $seed \
#     --top_heads 128 \
#     --moe_balance_ratio 0.01 \
#     --top_visual_moe_experts 3
#     # --fsdp "full_shard auto_wrap" \
#     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

# python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
#     --lora_r 16 \
#     --lora_alpha 8 \
#     --q_expert_num 1 \
#     --k_expert_num 16 \
#     --top_moe_num 3 \
#     --baseline beam \
#     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

# python llava/eval/run_eval.py \
#     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt

# python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
#     --lora_r 16 \
#     --lora_alpha 8 \
#     --q_expert_num 1 \
#     --k_expert_num 16 \
#     --top_moe_num 3 \
#     --baseline greedy \
#     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
#     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

# python llava/eval/run_eval.py \
#     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
#     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt


# 3.no key MoE
# bbox_ratio=0.1

# dir=llava_med/ablation/no_key_moe

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
#     llava/train/train_mem.py \
#     --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
#     --lora_enable True \
#     --moe_lora_r 16 \
#     --moe_lora_alpha 8 \
#     --mm_projector_lr 2e-5 \
#     --tune_mm_mlp_adapter False \
#     --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks.json \
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
#     --save_steps 500 \
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
#     --visual_enhance_ratio 0 \
#     --bbox_ratio $bbox_ratio \
#     --use_moe True \
#     --dense_moe True \
#     --use_mask True \
#     --query_expert_num 8 \
#     --visual_expert_num 1 \
#     --seed $seed \
#     --top_heads 128 \
#     --moe_balance_ratio 0 \
#     --top_visual_moe_experts 1
#     # --fsdp "full_shard auto_wrap" \
#     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

# python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
#     --lora_r 16 \
#     --lora_alpha 8 \
#     --q_expert_num 8 \
#     --k_expert_num 1 \
#     --top_moe_num 1 \
#     --baseline beam \
#     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

# python llava/eval/run_eval.py \
#     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
#     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt

# python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
#     --lora_r 16 \
#     --lora_alpha 8 \
#     --q_expert_num 8 \
#     --k_expert_num 1 \
#     --top_moe_num 1 \
#     --baseline greedy \
#     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
#     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

# python llava/eval/run_eval.py \
#     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
#     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt


# 4.no MoE
bbox_ratio=0.03

dir=llava_med/ablation/no_moe_v1

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
    --lora_enable True \
    --moe_lora_r 16 \
    --moe_lora_alpha 8 \
    --mm_projector_lr 2e-5 \
    --tune_mm_mlp_adapter False \
    --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks.json \
    --segment_path ${ROOT_PATH}/hallucination/${dataset}/data/training_segments.npz \
    --image_folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
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
    --visual_focus False \
    --visual_enhance_ratio 0 \
    --bbox_ratio $bbox_ratio \
    --use_moe False \
    --dense_moe False \
    --use_mask True \
    --query_expert_num 1 \
    --visual_expert_num 1 \
    --seed $seed \
    --top_heads 128 \
    --moe_balance_ratio 0 \
    --top_visual_moe_experts 1
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --baseline beam \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --baseline greedy \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt