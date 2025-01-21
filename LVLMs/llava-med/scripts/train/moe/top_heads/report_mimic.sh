#!/bin/bash

seeds=(4)
seed=4
ROOT_PATH=/data/aofei
dataset=MIMIC_CXR

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

# visual_enhance_ratio=0.03
visual_enhance_ratio=0
# bbox_ratio=0.06
epoch_num=12
head_num=128

# heads=(32 64 128 256 512 1024)
heads=(128)
# bbox_ratios=(0.05 0.08 0.1 0.12)
bbox_ratios=(0.04 0.05 0.08)

# for head_num in "${heads[@]}"; do
for bbox_ratio in "${bbox_ratios[@]}"; do
    echo "Running with seed: $seed, head_num: $head_num, bbox_ratio: $bbox_ratio"
    # dir=llava_med/ours_v0/KL_loss/all_expert_8_16_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}
    dir=llava_med/ours_v1_wo_KL/top${head_num}_balance_top2/all_expert_1_8_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}

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
    #     --data_path ${ROOT_PATH}/hallucination/${dataset}/data_report/training_masks_top4.json \
    #     --segment_path ${ROOT_PATH}/hallucination/${dataset}/data_report/training_segments_top4.npz \
    #     --image_folder ${ROOT_PATH}/hallucination/MIMIC_CXR/sampled_files_train \
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
    #     --query_expert_num 1 \
    #     --visual_expert_num 8 \
    #     --use_kl False \
    #     --seed $seed \
    #     --top_heads $head_num \
    #     --use_visual_prompt False \
    #     --moe_balance_ratio 0.01 \
    #     --top_visual_moe_experts 2
    #     # --fsdp "full_shard auto_wrap" \
    #     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

    # wait

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
    #     --lora_r 16 \
    #     --lora_alpha 8 \
    #     --q_expert_num 1 \
    #     --k_expert_num 8 \
    #     --top_moe_num 2 \
    #     --baseline beam \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints
    
    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
    #     --lora_r 16 \
    #     --lora_alpha 8 \
    #     --q_expert_num 1 \
    #     --k_expert_num 8 \
    #     --top_moe_num 2 \
    #     --baseline greedy \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.csv \
    #     --report True

    # wait

    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.csv \
    #     --report True

    # wait

    python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
        --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics_beam.txt \
        --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.csv

    python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
        --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
        --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics_greedy.txt \
        --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.csv
    
    wait
done


seeds=(4)
ROOT_PATH=/data/aofei
dataset=MIMIC_CXR

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    dir="llava_med/lora_report/epoch12/seed${seed}"

    # torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    #     llava/train/train_mem.py \
    #     --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
    #     --lora_enable True \
    #     --mm_projector_lr 2e-5 \
    #     --tune_mm_mlp_adapter False \
    #     --data_path ${ROOT_PATH}/hallucination/${dataset}/data_report/training.json \
    #     --image_folder ${ROOT_PATH}/hallucination/MIMIC_CXR/sampled_files_train \
    #     --vision_tower openai/clip-vit-large-patch14 \
    #     --mm_vision_select_layer -2 \
    #     --mm_use_im_start_end True \
    #     --bf16 True \
    #     --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
    #     --num_train_epochs 12 \
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
    #     --seed $seed
    #     # --fsdp "full_shard auto_wrap" \
    #     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    # wait

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_original.jsonl

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints
    # conda activate report_eval2

    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_original.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_original.csv \
    #     --report True

    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.csv \
    #     --report True

    python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
        --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
        --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics.txt \
        --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.csv
    python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
        --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_original.jsonl \
        --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics_original.txt \
        --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_original.csv
    

    baselines=(DoLa PAI opera beam greedy nucleus VCD m3id avisc) 
    # conda activate report_eval2
    # baselines=(m3id avisc)
#     baselines=(m3id)
    for baseline in "${baselines[@]}"; do
        # python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
        #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
        #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
        #     --baseline ${baseline}

        # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
        #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
        #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.csv \
        #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.csv \
        #     --report True
        # wait
        python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
                --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
                --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics_${baseline}.txt \
                --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.csv

        wait
    done
done